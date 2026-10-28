"""
Streamlit front end for the Sri Lanka Bird Diversity ensemble classifier.

Target: stateProvince (25 classes) from environmental, climatic and geographic features.

Three tabs, in the order a reader needs them: make a prediction, see how good the
model is, see what it learned. Everything secondary sits behind an expander.

Run:  streamlit run app.py
"""
import os

# Must run BEFORE numpy/sklearn import, so their OpenMP runtimes start single-threaded.
# These models predict ONE row per rerun, and for a single row the thread fan-out costs
# far more than it saves: HistGradientBoosting.predict_proba takes 225 ms multi-threaded
# versus 24 ms on one thread. Batch training is unaffected — it runs from src/train.py.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import numpy as np, pandas as pd, altair as alt, streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")

# ---- validated palette (dataviz skill, light surface #fcfcfb) -------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"          # categorical slots 1-2, pass all-pairs both modes
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GOOD, CRIT = "#0ca30c", "#d03b3b"

st.set_page_config(page_title="Sri Lanka Bird Diversity", page_icon="🪶", layout="wide")


def theme():
    return {"config": {
        "background": SURFACE,
        "view": {"stroke": "transparent"},
        "font": "-apple-system, Segoe UI, Roboto, sans-serif",
        "title": {"color": INK, "fontSize": 14, "fontWeight": 600, "anchor": "start"},
        "axis": {"labelColor": MUTED, "titleColor": INK2, "gridColor": GRID,
                 "domainColor": BASELINE, "tickColor": BASELINE, "labelFontSize": 11,
                 "titleFontSize": 11, "titleFontWeight": 500},
        "legend": {"labelColor": INK2, "titleColor": INK, "labelFontSize": 11,
                   "titleFontSize": 11, "symbolType": "square"},
    }}


alt.themes.register("viz", theme)
alt.themes.enable("viz")


# ---- loading -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ensemble…")
def load_model():
    """Prefer the compact deployment bundle; fall back to the study-grade one."""
    import joblib
    for stem in ("ensemble_deploy", "ensemble_full"):
        p = os.path.join(ART, f"{stem}.joblib")
        if os.path.exists(p):
            try:
                b = joblib.load(p)
            except Exception as exc:          # usually a scikit-learn version mismatch
                import sklearn
                st.error(
                    f"Could not load `{stem}.joblib`: {type(exc).__name__}: {exc}\n\n"
                    f"This bundle was written by scikit-learn 1.6.1; this environment has "
                    f"{sklearn.__version__}. Pin `scikit-learn==1.6.1` in requirements.txt "
                    f"or rebuild the bundle with `src/train.py --save --compact`.")
                st.stop()
            b["_stem"] = stem
            return b
    return None


@st.cache_data
def load_json(name):
    p = os.path.join(ART, name)
    return json.load(open(p)) if os.path.exists(p) else None


@st.cache_data
def load_samples():
    p = os.path.join(ART, "samples.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


bundle = load_model()
fstats = load_json("feature_stats.json")
metrics = ((load_json("metrics_full.json") or []) +
           (load_json("metrics_envonly.json") or []) +
           (load_json("metrics_deploy.json") or []))
facts = load_json("dataset_facts.json")
xai = load_json("explainability.json")
overfit = load_json("overfitting_check.json")
samples = load_samples()

if bundle is None or fstats is None:
    st.error(
        "Artifacts missing. Run `python src/make_app_assets.py`, then "
        "`python src/train.py --sample 300000 --sets full --protocols random,grouped "
        "--save --compact --tag metrics_deploy`.")
    st.stop()

FEATS, CLASSES = bundle["features"], bundle["classes"]
STATS, GROUPS = fstats["stats"], fstats["groups"]


@st.cache_data(max_entries=256, show_spinner=False)
def _predict_cached(key):
    """Cached on the exact feature vector, so revisiting a slider position is free."""
    x = np.array([key], dtype=np.float32)
    out, stack_in = {}, []
    for n in bundle["names"]:
        m = bundle["base"][n]
        pr = np.zeros(len(CLASSES))
        pr[m.classes_.astype(int)] = m.predict_proba(x)[0]
        out[n] = pr
        stack_in.append(pr)
    out["SoftVote"] = sum(w * out[n] for w, n in zip(bundle["weights"], bundle["names"]))
    meta = bundle["meta"]
    sp = np.zeros(len(CLASSES))
    sp[meta.classes_.astype(int)] = meta.predict_proba(np.hstack(stack_in).reshape(1, -1))[0]
    out["Stacking"] = sp
    return out


def predict_all(row):
    """Return {model_name: proba_vector} plus the two ensemble combinations."""
    return _predict_cached(tuple(float(row[f]) for f in FEATS))


st.title("🪶 Which district did this bird record come from?")
st.caption("Stacked ensemble over 1,552,048 Sri Lankan bird observations · "
           "25 districts · environmental, climatic and geographic features")

tabs = st.tabs(["Predict", "How good is it?", "What it learned"])

# =================================================================== PREDICT ====
with tabs[0]:
    left, right = st.columns([5, 4], gap="large")

    with left:
        sel = None
        if samples is not None:
            opts = [f"{r.stateProvince} — {r.verbatimScientificName}"
                    for r in samples.itertuples()]

            # "pick" is a widget key, so it cannot be assigned after the selectbox has
            # been instantiated in this run. An on_click callback runs before the next
            # run builds its widgets, which is the only safe place to set it.
            def shuffle_pick():
                st.session_state.pick = opts[np.random.randint(len(opts))]

            c1, c2 = st.columns([3, 1])
            with c1:
                pick = st.selectbox("Pick a real observation from the dataset", opts,
                                    index=0, key="pick")
                sel = samples.iloc[opts.index(pick)]
            with c2:
                st.write("")
                st.button("🎲 Random", use_container_width=True,
                          on_click=shuffle_pick)

        # All 24 inputs stay available, but folded away: the common path is to load a
        # real record and read the prediction, not to set two dozen sliders by hand.
        vals = {}
        with st.expander("Or set the inputs by hand", expanded=False):
            for gname, cols in GROUPS.items():
                st.markdown(f"**{gname}**")
                for c in cols:
                    s = STATS[c]
                    default = float(sel[c]) if sel is not None else s["median"]
                    lo, hi = s["q01"], s["q99"]
                    lo, hi = min(lo, default), max(hi, default)
                    if s["n_unique"] < 30:      # discrete -> integer step
                        vals[c] = st.slider(s["label"], int(s["min"]), int(s["max"]),
                                            int(default), 1, key=f"s_{c}")
                    else:
                        vals[c] = st.slider(s["label"], float(lo), float(hi),
                                            float(default), (hi - lo) / 200 or 0.01,
                                            format="%.4g", key=f"s_{c}")
        # derived seasonal features follow day-of-year, never set by hand
        vals["sin_doy"] = float(np.sin(2 * np.pi * vals["doy"] / 365.25))
        vals["cos_doy"] = float(np.cos(2 * np.pi * vals["doy"] / 365.25))

        if sel is not None:
            st.caption(
                f"Loaded: **{sel['verbatimScientificName']}** · "
                f"{sel['decimalLatitude']:.3f}°N, {sel['decimalLongitude']:.3f}°E · "
                f"{STATS['elevation_meters']['label']} {sel['elevation_meters']:.0f} m")

    with right:
        probs = predict_all(vals)
        best = bundle["best"]
        p = probs[best]
        top = int(p.argmax())
        st.metric("Predicted district", CLASSES[top], f"{p[top]*100:.1f}% confidence")
        if sel is not None:
            ok = CLASSES[top] == sel["stateProvince"]
            st.markdown(
                f"<span style='color:{GOOD if ok else CRIT};font-weight:600'>"
                f"{'✓ matches' if ok else '✗ differs from'} the recorded district "
                f"({sel['stateProvince']})</span>", unsafe_allow_html=True)

        d = (pd.DataFrame({"District": CLASSES, "Probability": p})
             .sort_values("Probability", ascending=False).head(5))
        bars = alt.Chart(d).mark_bar(
            color=S1, height=18, cornerRadiusTopRight=4, cornerRadiusBottomRight=4
        ).encode(
            x=alt.X("Probability:Q", title=None,
                    scale=alt.Scale(domain=[0, 1], nice=False),
                    axis=alt.Axis(format="%", tickCount=5)),
            y=alt.Y("District:N", sort="-x", title=None,
                    axis=alt.Axis(labelOverlap=False, labelLimit=140)),
            tooltip=[alt.Tooltip("District:N"),
                     alt.Tooltip("Probability:Q", format=".2%")],
        )
        # selective direct labels: a number on every bar would stack several "0.0%"
        labels = bars.mark_text(align="left", dx=5, color=INK2, fontSize=11).encode(
            text=alt.condition(alt.datum.Probability >= 0.005,
                               alt.Text("Probability:Q", format=".1%"), alt.value("")))
        st.altair_chart((bars + labels).properties(
            height=170, title="Most likely districts"), use_container_width=True)

        votes = {n: CLASSES[int(probs[n].argmax())]
                 for n in list(bundle["names"]) + ["SoftVote", "Stacking"]}
        agree = sum(v == CLASSES[top] for v in votes.values())
        st.caption(f"**{agree} of {len(votes)}** models agree on {CLASSES[top]}. "
                   "The displayed confidence is a ranking, not a calibrated frequency "
                   "— the boosted members saturate near 1.0.")
        with st.expander("Each model's own answer"):
            st.dataframe(
                pd.DataFrame([{"Model": n, "Prediction": v,
                               "Confidence": f"{probs[n].max()*100:.1f}%"}
                              for n, v in votes.items()]),
                use_container_width=True, hide_index=True)

    with st.expander("About the data"):
        if facts:
            c = st.columns(4)
            c[0].metric("Records", f"{facts['n_records']:,}")
            c[1].metric("Bird species", facts["n_species"])
            c[2].metric("Districts", facts["n_provinces"])
            c[3].metric("Distinct sites", f"{facts['n_sites']:,}")
        st.markdown("""
**Source** — [`DilushaChandrasiri/SriLanka-Bird-Diversity-Dataset`](https://huggingface.co/datasets/DilushaChandrasiri/SriLanka-Bird-Diversity-Dataset)
on Hugging Face, a tabular CSV of citizen-science occurrence records joined to
sixteen satellite and reanalysis covariates. No observer names or other personal
data are present.

**Why the target is district, not species** — every environmental column is derived
from location and date, so the 1.55 M rows collapse to 63,392 distinct feature
vectors: about 24 rows share a byte-identical input while carrying different species
labels. Top-1 species accuracy is capped at **5.54%** for any model on these
features. `stateProvince` — 25 districts, despite the column name — has a measured
ceiling of 100% and is the dataset's natural target.
""")

# ========================================================== HOW GOOD IS IT? ====
with tabs[1]:
    if not metrics:
        st.info("Run training to populate this tab.")
    else:
        st.markdown("#### The model is strong on surveyed sites and weaker on new ones")
        st.markdown(
            "One accuracy figure would misrepresent this model, so four are reported. "
            "Each row removes a different crutch; the bottom row is the honest number "
            "for a location the model has never seen.")
        st.dataframe(pd.DataFrame([
            {"Feature set": "full", "Protocol": "random",
             "What the guard removes": "nothing — the conventional split",
             "Accuracy": 0.9995, "Balanced acc.": 0.9975},
            {"Feature set": "full", "Protocol": "grouped",
             "What the guard removes": "every test site is unseen in training",
             "Accuracy": 0.9925, "Balanced acc.": 0.9910},
            {"Feature set": "envonly", "Protocol": "random",
             "What the guard removes": "latitude and longitude dropped",
             "Accuracy": 0.9971, "Balanced acc.": 0.9932},
            {"Feature set": "envonly", "Protocol": "grouped",
             "What the guard removes": "both at once — the hardest case",
             "Accuracy": 0.9032, "Balanced acc.": 0.8576},
        ]).style.format({"Accuracy": "{:.4f}", "Balanced acc.": "{:.4f}"}),
            use_container_width=True, hide_index=True)
        st.caption("Each guard alone costs under a point. Both together cost 9.6 "
                   "points and fall below the 95% target — reported, not buried.")

        if overfit:
            st.markdown("#### Is 0.9995 overfitting? No — the train–test gap is measured")
            ro = overfit["results"]
            od = pd.DataFrame([{"Model": k, "Train": v["train"], "Test": v["test"],
                                "Gap": v["gap"]} for k, v in ro.items()])
            worst = od.Gap.max()
            c1, c2 = st.columns([2, 3], gap="large")
            with c1:
                st.dataframe(od.style.format(
                    {"Train": "{:.4f}", "Test": "{:.4f}", "Gap": "{:+.4f}"}),
                    use_container_width=True, hide_index=True)
            with c2:
                st.markdown(
                    f"""Overfitting means a model that has memorised its training set:
high accuracy on data it has seen, materially lower accuracy on data it has not.
Measured on the same split, the largest train–test gap across all eight models is
**{worst*100:.2f} percentage points**, and the ensemble's is
**{ro['Stacking']['gap']*100:.2f}**.

A gap that small is not memorisation — it says the task is genuinely easy, because
district is nearly a deterministic function of latitude and longitude. The real
limitation is not overfitting but **transfer to unseen sites**, which the table
above quantifies at 0.9032.""")
                st.caption(overfit["note"])

        st.markdown("#### Every model, every protocol")
        FS_LABEL = {"full": "full (with lat/lon)", "envonly": "envonly (no lat/lon)",
                    "deploy": "deployed build (compact trees)"}
        rec = []
        for blk in metrics:
            for name, r in blk["results"].items():
                rec.append({"Model": name, "Protocol": blk["protocol"],
                            "Features": FS_LABEL.get(blk["feature_set"], blk["feature_set"]),
                            "Kind": "Ensemble" if name in ("SoftVote", "Stacking") else "Base learner",
                            "Accuracy": r["accuracy"],
                            "Balanced accuracy": r["balanced_accuracy"],
                            "Macro F1": r["macro_f1"]})
        dm = pd.DataFrame(rec)

        c1, c2 = st.columns(2)
        with c1:
            prots = sorted(dm.Protocol.unique())
            prot = st.radio("Protocol", prots,
                            index=prots.index("random") if "random" in prots else 0,
                            horizontal=True,
                            help="random = stratified row split (the conventional "
                                 "protocol). grouped = GroupShuffleSplit on (lat, lon), "
                                 "so every test site is unseen in training.")
        with c2:
            fsets = sorted(dm.Features.unique())
            _full = next((f for f in fsets if f.startswith("full")), fsets[0])
            fsel = st.radio("Feature set", fsets, index=fsets.index(_full),
                            horizontal=True,
                            help="envonly drops latitude and longitude entirely.")
        dd = dm[(dm.Protocol == prot) & (dm.Features == fsel)].sort_values("Accuracy")
        if dd.empty:
            st.info("That combination was not trained.")
        else:
            base = alt.Chart(dd).encode(
                y=alt.Y("Model:N", sort="-x", title=None,
                        axis=alt.Axis(labelOverlap=False, labelLimit=170)),
                x=alt.X("Accuracy:Q", title="Accuracy",
                        scale=alt.Scale(domain=[0, 1.08], nice=False),
                        axis=alt.Axis(format="%", tickCount=6)),
                color=alt.Color("Kind:N",
                                scale=alt.Scale(domain=["Base learner", "Ensemble"],
                                                range=[S1, S2]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Kind:N"),
                         alt.Tooltip("Accuracy:Q", format=".4f"),
                         alt.Tooltip("Balanced accuracy:Q", format=".4f"),
                         alt.Tooltip("Macro F1:Q", format=".4f")])
            bars = base.mark_bar(height=18, cornerRadiusTopRight=4,
                                 cornerRadiusBottomRight=4)
            txt = base.mark_text(align="left", dx=5, color=INK2, fontSize=11).encode(
                text=alt.Text("Accuracy:Q", format=".2%"), color=alt.value(INK2))
            rule = alt.Chart(pd.DataFrame({"t": [0.95]})).mark_rule(
                color=MUTED, strokeDash=[4, 3]).encode(x="t:Q")
            st.altair_chart((bars + txt + rule).properties(height=330),
                            use_container_width=True)
            st.caption("Dashed line marks the 95% assignment threshold.")

        blk = next((b for b in metrics if b.get("confusion_matrix")), None)
        if blk:
            with st.expander("Confusion matrix and weakest classes"):
                cm = np.array(blk["confusion_matrix"], float)
                cls = blk["classes"]
                norm = st.toggle("Normalise by true class (recall)", value=True)
                m = cm / np.maximum(cm.sum(1, keepdims=True), 1) if norm else cm
                d = pd.DataFrame([{"True": cls[i], "Predicted": cls[j],
                                   "Value": m[i, j], "Count": int(cm[i, j])}
                                  for i in range(len(cls)) for j in range(len(cls))])
                hm = alt.Chart(d).mark_rect(stroke=SURFACE, strokeWidth=1).encode(
                    x=alt.X("Predicted:N", sort=cls,
                            axis=alt.Axis(labelAngle=-45, labelOverlap=False,
                                          labelFontSize=10)),
                    y=alt.Y("True:N", sort=cls,
                            axis=alt.Axis(labelOverlap=False, labelFontSize=10,
                                          labelLimit=140)),
                    color=alt.Color("Value:Q",
                                    scale=alt.Scale(range=[SURFACE] + BLUE_RAMP,
                                                    type="sqrt"),
                                    legend=alt.Legend(title="Recall" if norm else "Count",
                                                      format=".0%" if norm else "d")),
                    tooltip=[alt.Tooltip("True:N"), alt.Tooltip("Predicted:N"),
                             alt.Tooltip("Count:Q", format="d"),
                             alt.Tooltip("Value:Q", format=".2%" if norm else "d")])
                st.altair_chart(hm.properties(height=520), use_container_width=True)
                rep = blk.get("classification_report", {})
                pc = pd.DataFrame([{"District": k, "Precision": v["precision"],
                                    "Recall": v["recall"], "F1": v["f1-score"],
                                    "Support": int(v["support"])}
                                   for k, v in rep.items() if k in cls]).sort_values("F1")
                st.markdown("**Weakest districts by F1**")
                st.dataframe(pc.head(6).style.format(
                    {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                    use_container_width=True, hide_index=True)

# ========================================================= WHAT IT LEARNED =====
with tabs[2]:
    if not xai:
        st.info("Run `python src/explain.py --sample 6000` to populate this tab.")
    else:
        st.markdown("#### Which features the model relies on")
        st.markdown(
            f"TreeSHAP attributions for the {xai['explained_model']} member, computed "
            f"on {xai['n_rows_explained']:,} held-out observations it never saw during "
            "training. Latitude and longitude lead, but together take only **49.2%** "
            "of attributed importance — terrain, population density, air chemistry and "
            "night-time light carry the rest.")

        lab = dict(zip(xai["features"], xai["labels"]))
        gs = (pd.DataFrame({"Feature": [lab.get(k, k) for k in xai["global_shap"]],
                            "SHAP": list(xai["global_shap"].values())})
              .sort_values("SHAP", ascending=False).head(12))
        b = alt.Chart(gs).mark_bar(color=S1, height=15, cornerRadiusTopRight=4,
                                   cornerRadiusBottomRight=4).encode(
            x=alt.X("SHAP:Q", title="mean |SHAP value|", axis=alt.Axis(tickCount=6)),
            y=alt.Y("Feature:N", sort="-x", title=None,
                    axis=alt.Axis(labelOverlap=False, labelLimit=190)),
            tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("SHAP:Q", format=".3f")])
        t = b.mark_text(align="left", dx=5, color=INK2, fontSize=11).encode(
            text=alt.Text("SHAP:Q", format=".2f"))
        st.altair_chart((b + t).properties(height=330), use_container_width=True)

        st.markdown("#### What drives one district")
        pp = xai.get("per_province_top_features", {})
        if pp:
            who = st.selectbox("District", sorted(pp), key="xai_prov")
            dr = pd.DataFrame([{"Feature": r["label"],
                                "mean |SHAP|": r["mean_abs_shap"],
                                "High values": r["direction"]} for r in pp[who]])
            st.dataframe(dr.style.format({"mean |SHAP|": "{:.3f}"}),
                         use_container_width=True, hide_index=True)
            st.caption(
                "The learned rules line up with physical geography: Nuwara Eliya is "
                "identified by high elevation, low temperature and low marine aerosol "
                "— highland, cold and landlocked; Colombo by SO₂ and night-time lights; "
                "Jaffna, which has no relief, leans on solar radiation instead.")

        with st.expander("Three importance methods, and why they disagree"):
            st.markdown("""
Asked how much of this one fitted model rests on latitude and longitude, the three
methods give **39.4%** (impurity), **49.2%** (SHAP) and **97.9%** (permutation).

Permutation implies the 22 environmental columns are worth 2.1% between them, and
that reading is demonstrably wrong: retrained with no coordinates at all, the
ensemble still reaches 0.9971. The cause is redundancy — every environmental column
was joined on location, so shuffling elevation changes little while latitude is still
in the matrix. Permutation importance measures *marginal* contribution given
everything else present, which under strong redundancy understates every redundant
feature.

Read on its own, any one of the three would have given a wrong answer about this
dataset. That is the argument for running more than one.
""")
