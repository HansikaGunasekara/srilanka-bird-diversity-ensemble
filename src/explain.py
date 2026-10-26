"""
Explainability pass for the province classifier (assignment section 4, XAI).

Three independent methods, deliberately chosen so they cross-check each other:

  1. SHAP (TreeSHAP)        - exact Shapley values for the LightGBM member.
                              Global ranking, per-province attribution, and one
                              local waterfall for a single prediction.
  2. Permutation importance - model-agnostic, measured on held-out data.
                              This is the honest counterweight to the impurity
                              importances reported in train.py, which are known
                              to inflate high-cardinality continuous features
                              (latitude/longitude are exactly that case).
  3. Partial dependence     - the marginal shape of the response, so we can say
                              *which direction* a feature pushes a province, not
                              merely that it matters.

Why LightGBM is the explained member: TreeSHAP is exact and tractable on it, it
scores 0.9990 on its own (within 0.0005 of the full ensemble), and the stacked
meta-learner is a linear reweighting of base probabilities - so the base model's
attributions are the substantive explanation of the ensemble's behaviour.

Outputs
  artifacts/explainability.json   machine-readable, consumed by app.py
  docs/06_shap_global.png         mean |SHAP| ranking, all classes
  docs/07_shap_beeswarm.png       per-observation attribution for one province
  docs/08_shap_waterfall.png      a single prediction, explained
  docs/09_pdp.png                 partial dependence of the top features
  docs/10_importance_methods.png  impurity vs permutation vs SHAP, side by side

Run:
  .venv/bin/python src/explain.py --sample 6000
"""
import os, sys, json, time, argparse, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
DOCS = os.path.join(ROOT, "docs"); os.makedirs(DOCS, exist_ok=True)
TARGET = "stateProvince"

# Matches the deployed app's palette so the report and the app read as one system.
INK = "#16202B"
BLUE = "#2A78D6"
ORANGE = "#C4521F"
SURFACE = "#FCFCFB"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#C9CDD2", "font.size": 10,
})


def pretty(name):
    """Human-readable feature labels for the figures."""
    return {
        "decimalLatitude": "latitude", "decimalLongitude": "longitude",
        "elevation_meters": "elevation", "temp_mean": "mean temp",
        "humid_mean": "mean humidity", "wind_mean": "mean wind",
        "shortwave_radiation": "solar radiation", "avg_rad": "night lights",
        "lka_general_2020": "population density", "LandCover_Class": "land cover",
        "Total_Aerosol_Extinction": "aerosol extinction",
        "Carbon_Mass": "carbon aerosol", "Dust_Mass": "dust aerosol",
        "SO2_Mass": "SO2", "Sulfate_Mass": "sulfate", "Sea_Salt_Mass": "sea salt",
        "individualCount": "individual count", "doy": "day of year",
        "sin_doy": "sin(day of year)", "cos_doy": "cos(day of year)",
    }.get(name, name)


def load_bundle():
    for stem in ("deploy", "full"):
        p = os.path.join(ART, f"ensemble_{stem}.joblib")
        if os.path.exists(p):
            print(f"loading {os.path.basename(p)}")
            return joblib.load(p), stem
    raise SystemExit("no ensemble bundle in artifacts/ - run src/train.py --save first")


def holdout(df, feats, sample, seed=0):
    """Rebuild the *same* stratified split train.py used, then take its test side."""
    from sklearn.model_selection import train_test_split
    sub = df.sample(400_000, random_state=seed) if len(df) > 400_000 else df
    X = sub[feats].to_numpy(np.float32)
    ycat = sub[TARGET].astype("category")
    y = ycat.cat.codes.to_numpy()
    _, ite = train_test_split(np.arange(len(y)), test_size=0.25,
                              random_state=seed, stratify=y)
    rng = np.random.default_rng(seed)
    pick = rng.choice(ite, size=min(sample, len(ite)), replace=False)
    return X[pick], y[pick], list(ycat.cat.categories)


def to_3d(sv, n, f, c):
    """shap returns (n,f,c) or a length-c list of (n,f) depending on version."""
    if isinstance(sv, list):
        return np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape == (n, f, c):
        return sv
    if sv.ndim == 3 and sv.shape == (c, n, f):
        return np.transpose(sv, (1, 2, 0))
    raise ValueError(f"unexpected shap shape {sv.shape}")


def main(a):
    import shap
    from sklearn.inspection import permutation_importance, partial_dependence

    bundle, stem = load_bundle()
    feats, classes = bundle["features"], bundle["classes"]
    lgbm = bundle["base"]["LightGBM"]
    labels = [pretty(f) for f in feats]

    df = load()
    X, y, cls_from_data = holdout(df, feats, a.sample)
    assert cls_from_data == classes, "class order drifted from the saved bundle"
    print(f"explaining on {len(X):,} held-out rows, {len(feats)} features, {len(classes)} classes")

    # ---------------------------------------------------------------- 1. SHAP
    t = time.time()
    expl = shap.TreeExplainer(lgbm)
    sv = to_3d(expl.shap_values(X), len(X), len(feats), len(classes))
    print(f"  TreeSHAP done in {time.time()-t:.1f}s, values {sv.shape}")

    # Global: mean |SHAP| over observations AND classes.
    glob = np.abs(sv).mean(axis=(0, 2))
    order = np.argsort(glob)[::-1]

    fig, ax = plt.subplots(figsize=(8, 7))
    yy = np.arange(len(feats))[::-1]
    ax.barh(yy, glob[order], color=BLUE, height=0.68)
    ax.set_yticks(yy); ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel("mean |SHAP value|  (average impact on class probability)")
    ax.set_title("Global feature importance — TreeSHAP, averaged over 25 provinces",
                 loc="left", fontsize=12, color=INK, pad=12)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#E6E8EA", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f"{DOCS}/06_shap_global.png", dpi=160); plt.close(fig)

    # Per-province: which feature carries each class, and in which direction.
    per_prov = {}
    for ci, cname in enumerate(classes):
        m = np.abs(sv[:, :, ci]).mean(axis=0)
        top = np.argsort(m)[::-1][:5]
        per_prov[cname] = [
            {"feature": feats[i], "label": labels[i],
             "mean_abs_shap": float(m[i]),
             # sign of the mean signed SHAP among rows truly of this class
             "direction": ("raises" if sv[y == ci][:, i, ci].mean() > 0 else "lowers")
             if (y == ci).any() else "n/a"}
            for i in top]

    # Beeswarm for the most distinctive highland province we actually hold rows for.
    pick_name = next((p for p in ("Nuwara", "Nuwara Eliya", "Kandy", "Colombo")
                      if p in classes), classes[0])
    pick = classes.index(pick_name)
    ex = shap.Explanation(values=sv[:, :, pick], data=X,
                          feature_names=labels,
                          base_values=np.repeat(np.ravel(expl.expected_value)[pick], len(X)))
    plt.figure(figsize=(8, 7))
    shap.plots.beeswarm(ex, max_display=14, show=False, color_bar=True)
    plt.title(f"How each observation votes for “{pick_name}”", loc="left",
              fontsize=12, color=INK, pad=12)
    plt.tight_layout(); plt.savefig(f"{DOCS}/07_shap_beeswarm.png", dpi=160,
                                    facecolor=SURFACE); plt.close()

    # Local: one correctly-classified row, explained end to end.
    pred = lgbm.predict(X)
    hit = np.flatnonzero(pred == y)
    li = int(hit[0]) if len(hit) else 0
    lc = int(y[li])
    lex = shap.Explanation(values=sv[li, :, lc], data=X[li],
                           feature_names=labels,
                           base_values=float(np.ravel(expl.expected_value)[lc]))
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(lex, max_display=12, show=False)
    plt.title(f"One prediction explained — true and predicted class “{classes[lc]}”",
              loc="left", fontsize=12, color=INK, pad=12)
    plt.tight_layout(); plt.savefig(f"{DOCS}/08_shap_waterfall.png", dpi=160,
                                    facecolor=SURFACE); plt.close()

    # ------------------------------------------- 2. Permutation importance
    t = time.time()
    nperm = min(a.perm_sample, len(X))
    pi = permutation_importance(lgbm, X[:nperm], y[:nperm], n_repeats=a.repeats,
                                random_state=0, n_jobs=-1, scoring="accuracy")
    print(f"  permutation importance done in {time.time()-t:.1f}s on {nperm:,} rows")

    # ------------------------------------------------ 3. Partial dependence
    top4 = [int(i) for i in order[:4]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, fi in zip(axes.ravel(), top4):
        pd_res = partial_dependence(lgbm, X[:a.pdp_sample], [fi],
                                    kind="average", grid_resolution=40)
        gx = pd_res["grid_values"][0]
        avg = pd_res["average"]  # (n_classes, grid)
        for ci, col in ((pick, ORANGE), (classes.index("Colombo") if "Colombo" in classes else 0, BLUE)):
            ax.plot(gx, avg[ci], color=col, lw=2, label=classes[ci])
        ax.set_xlabel(labels[fi]); ax.set_ylabel("partial dependence\n(class probability)")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.grid(color="#E6E8EA", lw=0.8); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Partial dependence — the direction each top feature pushes a province",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{DOCS}/09_pdp.png", dpi=160); plt.close(fig)

    # ------------------------- 4. the three methods, side by side (the argument)
    imp_json = os.path.join(ART, f"metrics_{'deploy' if stem=='deploy' else 'full'}.json")
    impurity = None
    if os.path.exists(imp_json):
        blob = json.load(open(imp_json))
        fi_all = (blob[0] if isinstance(blob, list) else blob).get("feature_importances", {})
        if "LightGBM" in fi_all:
            impurity = np.array([fi_all["LightGBM"].get(f, 0.0) for f in feats])

    def norm(v):
        v = np.asarray(v, float); s = v.sum()
        return v / s if s else v

    series = [("SHAP (|mean|)", norm(glob), BLUE),
              ("Permutation", norm(np.clip(pi.importances_mean, 0, None)), ORANGE)]
    if impurity is not None:
        series.append(("Impurity (biased)", norm(impurity), "#8A94A0"))

    fig, ax = plt.subplots(figsize=(10, 7.5))
    yy = np.arange(len(feats))[::-1]
    h = 0.8 / len(series)
    for k, (nm, vals, col) in enumerate(series):
        ax.barh(yy + (k - (len(series) - 1) / 2) * h, vals[order], height=h,
                color=col, label=nm)
    ax.set_yticks(yy); ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel("share of total importance (normalised)")
    ax.set_title("Three importance methods disagree — and that disagreement is the finding",
                 loc="left", fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False); ax.grid(axis="x", color="#E6E8EA", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{DOCS}/10_importance_methods.png", dpi=160); plt.close(fig)

    # ----------------------------------------------------------------- export
    out = {
        "explained_model": "LightGBM",
        "bundle": stem,
        "n_rows_explained": int(len(X)),
        "n_rows_permutation": int(nperm),
        "permutation_repeats": int(a.repeats),
        "features": feats,
        "labels": labels,
        "global_shap": {feats[i]: float(glob[i]) for i in order},
        "permutation_importance": {
            feats[i]: {"mean": float(pi.importances_mean[i]),
                       "std": float(pi.importances_std[i])} for i in order},
        "impurity_importance": ({feats[i]: float(impurity[i]) for i in order}
                                if impurity is not None else None),
        "ranking": {
            "shap": [feats[i] for i in order],
            "permutation": [feats[i] for i in np.argsort(pi.importances_mean)[::-1]],
            "impurity": ([feats[i] for i in np.argsort(impurity)[::-1]]
                         if impurity is not None else None),
        },
        "per_province_top_features": per_prov,
        "local_example": {
            "row_index": li, "true_class": classes[lc], "predicted_class": classes[int(pred[li])],
            "base_value": float(np.ravel(expl.expected_value)[lc]),
            "contributions": sorted(
                [{"feature": feats[i], "label": labels[i],
                  "value": float(X[li, i]), "shap": float(sv[li, i, lc])}
                 for i in range(len(feats))],
                key=lambda d: abs(d["shap"]), reverse=True)[:12],
        },
    }
    p = os.path.join(ART, "explainability.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")
    print("figures -> docs/06..10_*.png")

    print("\ntop 8 by each method")
    print(f"  {'SHAP':<22}{'permutation':<22}{'impurity':<22}")
    for k in range(8):
        r = out["ranking"]
        imp_r = r["impurity"][k] if r["impurity"] else "-"
        print(f"  {pretty(r['shap'][k]):<22}{pretty(r['permutation'][k]):<22}{pretty(imp_r):<22}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=6000,
                    help="held-out rows to compute SHAP on")
    ap.add_argument("--perm-sample", type=int, default=4000,
                    help="rows for permutation importance (it refits nothing but predicts a lot)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--pdp-sample", type=int, default=2000)
    main(ap.parse_args())
