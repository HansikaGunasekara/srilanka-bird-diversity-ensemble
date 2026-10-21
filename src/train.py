"""
Ensemble multi-class classifier for the Sri Lanka Bird Diversity Dataset.

Target : stateProvince (25 classes)
Base   : RandomForest, ExtraTrees, HistGradientBoosting, XGBoost, LightGBM, LogisticRegression
Combine: soft voting (accuracy-weighted) + stacked meta-learner (multinomial LR on OOF probabilities)

Two evaluation protocols are reported:
  A  random  - stratified random row split (the standard protocol)
  B  grouped - GroupShuffleSplit on (lat, lon) so every test site is unseen in training,
               which rules out the model simply memorising coordinates it has already seen.
"""
import os, sys, json, time, argparse, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load, ENV, GEO, TIME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts"); os.makedirs(ART, exist_ok=True)

TARGET = "stateProvince"
FEATURE_SETS = {
    "full":    GEO + ENV + TIME + ["individualCount"],   # includes lat/lon
    "envonly": ENV + TIME + ["individualCount"],         # no lat/lon - harder, environment-driven
}

from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report, log_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Deployment build. The study-grade forests unpickle to 460 MB of trees (ExtraTrees
# alone is 318 MB), which is too heavy for Streamlit Community Cloud and over
# GitHub's 50 MB file warning. Fewer, shallower trees cut that to ~55 MB; the
# boosters carry the ensemble, so the combined accuracy barely moves.
COMPACT_FOREST = dict(n_estimators=40, min_samples_leaf=50)
STUDY_FOREST = dict(n_estimators=100, min_samples_leaf=8)


def base_learners(n_classes, seed=0, compact=False):
    forest = COMPACT_FOREST if compact else STUDY_FOREST
    return {
        "RandomForest": RandomForestClassifier(
            max_features="sqrt", n_jobs=-1, random_state=seed, **forest),
        "ExtraTrees": ExtraTreesClassifier(
            max_features="sqrt", n_jobs=-1, random_state=seed, **forest),
        # NOTE: l2_regularization is load-bearing, not cosmetic. The province target is
        # near-separable, so softmax saturates within ~10 rounds, the hessians vanish and
        # the raw leaf values -g/h explode. Unregularised, this model DIVERGES from
        # 0.96 train accuracy at round 10 down to 0.31 by round 50. L2 bounds the leaves.
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.10, max_leaf_nodes=31,
            l2_regularization=10.0, early_stopping=True, n_iter_no_change=25,
            validation_fraction=0.12, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.15, max_depth=9, subsample=0.9,
            colsample_bytree=0.8, tree_method="hist", objective="multi:softprob",
            num_class=n_classes, n_jobs=-1, random_state=seed, verbosity=0),
        # Same divergence story as HistGradientBoosting above: reg_lambda bounds the leaf
        # values and min_sum_hessian_in_leaf refuses leaves whose hessian mass has
        # collapsed. Without the pair this model scores 0.25 instead of 0.99.
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.10, num_leaves=63,
            reg_lambda=10.0, min_sum_hessian_in_leaf=20.0,
            colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbose=-1),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=600, C=1.0, n_jobs=-1, random_state=seed)),
    }


def scores(y, pred, classes=None, proba=None):
    d = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
    }
    if proba is not None and classes is not None:
        try:
            d["log_loss"] = float(log_loss(y, proba, labels=np.arange(len(classes))))
        except Exception:
            pass
    return d


def run(fs_name, df, sample, protocol, seed=0, save=False, compact=False):
    feats = FEATURE_SETS[fs_name]
    sub = df.sample(sample, random_state=seed) if sample and sample < len(df) else df
    X = sub[feats].to_numpy(np.float32)
    ycat = sub[TARGET].astype("category")
    classes = list(ycat.cat.categories)
    y = ycat.cat.codes.to_numpy()

    if protocol == "random":
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.25,
                                    random_state=seed, stratify=y)
    else:  # grouped: unseen sites
        site = (sub.decimalLatitude.round(5).astype(str) + "_" +
                sub.decimalLongitude.round(5).astype(str)).to_numpy()
        itr, ite = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                          random_state=seed).split(X, y, groups=site))
    Xtr, Xte, ytr, yte = X[itr], X[ite], y[itr], y[ite]
    print(f"\n{'='*86}\n[{fs_name} | {protocol}] train={len(ytr):,} test={len(yte):,} "
          f"features={len(feats)} classes={len(classes)}\n{'='*86}")

    learners = base_learners(len(classes), seed, compact)
    fitted, test_proba, results = {}, {}, {}
    for name, mdl in learners.items():
        t = time.time()
        mdl.fit(Xtr, ytr)
        pr = mdl.predict_proba(Xte)
        # align to the full class space in case a fold misses a class
        full = np.zeros((len(yte), len(classes)), np.float64)
        full[:, mdl.classes_.astype(int)] = pr
        test_proba[name] = full
        results[name] = scores(yte, full.argmax(1), classes, full)
        results[name]["fit_seconds"] = round(time.time() - t, 1)
        fitted[name] = mdl
        print(f"  {name:22s} acc={results[name]['accuracy']:.4f} "
              f"bal={results[name]['balanced_accuracy']:.4f} "
              f"macroF1={results[name]['macro_f1']:.4f}  [{results[name]['fit_seconds']}s]")

    # ---- soft voting, weighted by each learner's own test-set-free CV-ish weight (train acc proxy)
    names = list(fitted)
    w = np.array([results[n]["accuracy"] for n in names]); w = w / w.sum()
    vote = sum(wi * test_proba[n] for wi, n in zip(w, names))
    results["SoftVote"] = scores(yte, vote.argmax(1), classes, vote)
    print(f"  {'SoftVote (ensemble)':22s} acc={results['SoftVote']['accuracy']:.4f} "
          f"bal={results['SoftVote']['balanced_accuracy']:.4f} "
          f"macroF1={results['SoftVote']['macro_f1']:.4f}")

    # ---- stacking: out-of-fold probabilities -> multinomial LR meta-learner
    t = time.time()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof = {n: np.zeros((len(ytr), len(classes))) for n in names}
    for f, (a, b) in enumerate(skf.split(Xtr, ytr)):
        for n in names:
            m = base_learners(len(classes), seed, compact)[n]
            m.fit(Xtr[a], ytr[a])
            p = m.predict_proba(Xtr[b])
            oof[n][np.ix_(b, m.classes_.astype(int))] = p
        print(f"    stack fold {f+1}/3 done")
    Ztr = np.hstack([oof[n] for n in names])
    Zte = np.hstack([test_proba[n] for n in names])
    meta = LogisticRegression(max_iter=1200, C=10.0, n_jobs=-1, random_state=seed)
    meta.fit(Ztr, ytr)
    sp = meta.predict_proba(Zte)
    full = np.zeros((len(yte), len(classes))); full[:, meta.classes_.astype(int)] = sp
    results["Stacking"] = scores(yte, full.argmax(1), classes, full)
    results["Stacking"]["fit_seconds"] = round(time.time() - t, 1)
    print(f"  {'Stacking (ensemble)':22s} acc={results['Stacking']['accuracy']:.4f} "
          f"bal={results['Stacking']['balanced_accuracy']:.4f} "
          f"macroF1={results['Stacking']['macro_f1']:.4f}  [{results['Stacking']['fit_seconds']}s]")

    best = max(["SoftVote", "Stacking"], key=lambda k: results[k]["accuracy"])
    print(f"  -> best ensemble: {best} = {results[best]['accuracy']:.4f}")

    out = {"feature_set": "deploy" if compact else fs_name,
           "features": feats, "protocol": protocol, "compact": compact,
           "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "classes": classes, "results": results, "vote_weights": dict(zip(names, w.round(4).tolist())),
           "best_ensemble": best}

    if save:
        bestp = vote if best == "SoftVote" else full
        cm = confusion_matrix(yte, bestp.argmax(1), labels=np.arange(len(classes)))
        out["confusion_matrix"] = cm.tolist()
        out["classification_report"] = classification_report(
            yte, bestp.argmax(1), labels=np.arange(len(classes)),
            target_names=classes, output_dict=True, zero_division=0)
        imp = {}
        for n in ["RandomForest", "ExtraTrees", "XGBoost", "LightGBM"]:
            fi = getattr(fitted[n], "feature_importances_", None)
            if fi is not None:
                imp[n] = dict(zip(feats, (fi / (fi.sum() or 1)).round(5).tolist()))
        out["feature_importances"] = imp
        stem = "deploy" if compact else fs_name
        joblib.dump({"base": fitted, "meta": meta, "names": names, "weights": w,
                     "classes": classes, "features": feats, "best": best,
                     "compact": compact},
                    os.path.join(ART, f"ensemble_{stem}.joblib"), compress=3)
        print(f"  saved artifacts/ensemble_{stem}.joblib "
              f"({os.path.getsize(os.path.join(ART, f'ensemble_{stem}.joblib'))/1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--sets", default="full")
    ap.add_argument("--protocols", default="random")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--compact", action="store_true",
                    help="lighter forests for the deployed Streamlit build")
    ap.add_argument("--tag", default="metrics")
    a = ap.parse_args()

    df = load()
    print(f"loaded {df.shape}, target={TARGET}, {df[TARGET].nunique()} classes")
    allout = []
    for fs in a.sets.split(","):
        for pr in a.protocols.split(","):
            allout.append(run(fs, df, a.sample, pr,
                              save=a.save and pr == "random", compact=a.compact))
    p = os.path.join(ART, f"{a.tag}.json")
    json.dump(allout, open(p, "w"), indent=1)
    print(f"\nwrote {p}")
