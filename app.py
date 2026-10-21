"""
Streamlit front end for the Sri Lanka Bird Diversity ensemble classifier.

Target: stateProvince (25 classes) from environmental, climatic and geographic features.
Run:  streamlit run app.py
"""
import os, json
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
            b = joblib.load(p)
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
samples = load_samples()

if bundle is None or fstats is None:
    st.error(
        "Artifacts missing. Run `python src/make_app_assets.py`, then "
        "`python src/train.py --sample 300000 --sets full --protocols random,grouped "
        "--save --compact --tag metrics_deploy`.")
    st.stop()

FEATS, CLASSES = bundle["features"], bundle["classes"]
STATS, GROUPS = fstats["stats"], fstats["groups"]


def predict_all(row):
    """Return {model_name: proba_vector} plus the two ensemble combinations."""
    x = np.array([[row[f] for f in FEATS]], dtype=np.float32)
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


st.title("🪶 Sri Lanka Bird Diversity — ensemble species-observation classifier")
st.caption("1,552,048 GBIF occurrence records · 429 bird species · 25 provinces · 2014–2024")

tabs = st.tabs(["Predict", "Model comparison", "Confusion matrix",
                "Feature importance", "About the data"])

# =================================================================== PREDICT ====
with tabs[0]:
    st.subheader("Classify an observation")
    left, right = st.columns([5, 4], gap="large")

    with left:
        if samples is not None:
            c1, c2 = st.columns([3, 2])
            with c1:
                opts = [f"{r.stateProvince} — {r.verbatimScientificName}"
                        for r in samples.itertuples()]
                pick = st.selectbox("Load a real observation from the dataset", opts,
                                    index=0, key="pick")
                sel = samples.iloc[opts.index(pick)]
            with c2:
                st.write("")
                if st.button("🎲 Random record", use_container_width=True):
                    st.session_state.pick = opts[np.random.randint(len(opts))]
                    st.rerun()
        else:
            sel = None

        vals = {}
        for gname, cols in GROUPS.items():
            with st.expander(gname, expanded=(gname == "Location")):
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

    with right:
        probs = predict_all(vals)
        best = bundle["best"]
        p = probs[best]
        top = int(p.argmax())
        st.metric(f"Predicted province · {best}", CLASSES[top], f"{p[top]*100:.1f}% confidence")
        if sel is not None:
            ok = CLASSES[top] == sel["stateProvince"]
            st.markdown(
                f"<span style='color:{GOOD if ok else CRIT};font-weight:600'>"
                f"{'✓ matches' if ok else '✗ differs from'} the recorded province "
                f"({sel['stateProvince']})</span>", unsafe_allow_html=True)

        d = (pd.DataFrame({"Province": CLASSES, "Probability": p})
             .sort_values("Probability", ascending=False).head(8))
        bars = alt.Chart(d).mark_bar(
            color=S1, height=16, cornerRadiusTopRight=4, cornerRadiusBottomRight=4
        ).encode(
            x=alt.X("Probability:Q", title="Probability",
                    scale=alt.Scale(domain=[0, 1], nice=False),
                    axis=alt.Axis(format="%", tickCount=5)),
            y=alt.Y("Province:N", sort="-x", title=None,
                    axis=alt.Axis(labelOverlap=False, labelLimit=140)),
            tooltip=[alt.Tooltip("Province:N"),
                     alt.Tooltip("Probability:Q", format=".2%")],
        )
        # selective direct labels: a number on every bar would stack eight "0.0%"
        labels = bars.mark_text(align="left", dx=5, color=INK2, fontSize=11).encode(
            text=alt.condition(alt.datum.Probability >= 0.005,
                               alt.Text("Probability:Q", format=".1%"), alt.value("")))
        st.altair_chart((bars + labels).properties(
            height=300, title="Top 8 province probabilities"), use_container_width=True)

        st.markdown("**Per-model votes**")
        rows = [{"Model": n, "Prediction": CLASSES[int(probs[n].argmax())],
                 "Confidence": f"{probs[n].max()*100:.1f}%"}
                for n in list(bundle["names"]) + ["SoftVote", "Stacking"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if bundle.get("compact"):
            st.caption(
                "Predictions come from the **compact deployment build** — the same six "
                "learners, with lighter RandomForest/ExtraTrees settings so the bundle "
                "fits Streamlit Community Cloud. Its own measured scores are on the "
                "*Model comparison* tab under *deployed build (compact trees)*; the "
                "study-grade results are the other two feature sets.")

# ========================================================== MODEL COMPARISON ====
with tabs[1]:
    st.subheader("Base learners vs. ensemble combiners")
    if not metrics:
        st.info("Run training to populate this tab.")
    else:
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
            prot = st.radio("Evaluation protocol", prots,
                            index=prots.index("random") if "random" in prots else 0,
                            horizontal=True,
                            help="random = stratified row split (the conventional protocol). "
                                 "grouped = GroupShuffleSplit on (lat, lon), so every test "
                                 "site is unseen in training.")
        with c2:
            fsets = sorted(dm.Features.unique())
            _full = next((f for f in fsets if f.startswith("full")), fsets[0])
            fsel = st.radio("Feature set", fsets, index=fsets.index(_full),
                            horizontal=True,
                            help="envonly drops latitude and longitude entirely, leaving "
                                 "only climate, habitat and air-quality variables.")
        dd = dm[(dm.Protocol == prot) & (dm.Features == fsel)].sort_values("Accuracy")
        if dd.empty:
            st.info("That combination was not trained.")
            st.stop()

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
        bars = base.mark_bar(height=18, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        txt = base.mark_text(align="left", dx=5, color=INK2, fontSize=11).encode(
            text=alt.Text("Accuracy:Q", format=".2%"), color=alt.value(INK2))
        rule = alt.Chart(pd.DataFrame({"t": [0.95]})).mark_rule(
            color=MUTED, strokeDash=[4, 3]).encode(x="t:Q")
        st.altair_chart((bars + txt + rule).properties(height=330),
                        use_container_width=True)
        st.caption("Dashed line marks the 95% assignment threshold.")
        st.dataframe(dd.drop(columns=["Protocol", "Features"]).sort_values("Accuracy", ascending=False)
                     .style.format({"Accuracy": "{:.4f}", "Balanced accuracy": "{:.4f}",
                                    "Macro F1": "{:.4f}"}),
                     use_container_width=True, hide_index=True)

# =========================================================== CONFUSION MATRIX ===
with tabs[2]:
    st.subheader("Confusion matrix — best ensemble, random split")
    blk = next((b for b in metrics if b.get("confusion_matrix")), None)
    if not blk:
        st.info("Run training with --save to populate this tab.")
    else:
        cm = np.array(blk["confusion_matrix"], float)
        cls = blk["classes"]
        norm = st.toggle("Normalise by true class (recall)", value=True)
        m = cm / np.maximum(cm.sum(1, keepdims=True), 1) if norm else cm
        d = pd.DataFrame([{"True": cls[i], "Predicted": cls[j],
                           "Value": m[i, j], "Count": int(cm[i, j])}
                          for i in range(len(cls)) for j in range(len(cls))])
        hm = alt.Chart(d).mark_rect(stroke=SURFACE, strokeWidth=1).encode(
            x=alt.X("Predicted:N", sort=cls,
                    axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelFontSize=10)),
            y=alt.Y("True:N", sort=cls,
                    axis=alt.Axis(labelOverlap=False, labelFontSize=10, labelLimit=140)),
            color=alt.Color("Value:Q",
                            scale=alt.Scale(range=[SURFACE] + BLUE_RAMP, type="sqrt"),
                            legend=alt.Legend(title="Recall" if norm else "Count",
                                              format=".0%" if norm else "d")),
            tooltip=[alt.Tooltip("True:N"), alt.Tooltip("Predicted:N"),
                     alt.Tooltip("Count:Q", format="d"),
                     alt.Tooltip("Value:Q", format=".2%" if norm else "d")])
        st.altair_chart(hm.properties(height=560), use_container_width=True)

        rep = blk.get("classification_report", {})
        pc = pd.DataFrame([{"Province": k, "Precision": v["precision"],
                            "Recall": v["recall"], "F1": v["f1-score"],
                            "Support": int(v["support"])}
                           for k, v in rep.items() if k in cls]).sort_values("F1")
        st.markdown("**Weakest classes by F1**")
        st.dataframe(pc.head(8).style.format(
            {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
            use_container_width=True, hide_index=True)

# ========================================================== FEATURE IMPORTANCE ==
with tabs[3]:
    st.subheader("Which features drive the prediction?")
    blk = next((b for b in metrics if b.get("feature_importances")), None)
    if not blk:
        st.info("Run training with --save to populate this tab.")
    else:
        imps = blk["feature_importances"]
        which = st.selectbox("Model", list(imps))     # one series at a time, by design
        d = (pd.DataFrame({"Feature": list(imps[which]),
                           "Importance": list(imps[which].values())})
             .sort_values("Importance", ascending=False).head(16))
        d["Feature"] = d.Feature.map(lambda c: STATS.get(c, {}).get("label", c))
        b = alt.Chart(d).mark_bar(color=S1, height=15, cornerRadiusTopRight=4,
                                  cornerRadiusBottomRight=4).encode(
            x=alt.X("Importance:Q", title="Relative importance",
                    axis=alt.Axis(tickCount=6)),
            y=alt.Y("Feature:N", sort="-x", title=None,
                    axis=alt.Axis(labelOverlap=False, labelLimit=190)),
            tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Importance:Q", format=".4f")])
        t = b.mark_text(align="left", dx=5, color=INK2, fontSize=11).encode(
            text=alt.Text("Importance:Q", format=".3f"))
        st.altair_chart((b + t).properties(height=400), use_container_width=True)
        st.caption("Impurity/gain-based importance, normalised to sum to 1 within each model.")

# =============================================================== ABOUT DATA =====
with tabs[4]:
    st.subheader("Dataset, and why the target is province rather than species")
    if facts:
        c = st.columns(4)
        c[0].metric("Records", f"{facts['n_records']:,}")
        c[1].metric("Bird species", facts["n_species"])
        c[2].metric("Provinces", facts["n_provinces"])
        c[3].metric("Distinct sites", f"{facts['n_sites']:,}")

    st.markdown(f"""
**Source** — [`DilushaChandrasiri/SriLanka-Bird-Diversity-Dataset`](https://huggingface.co/datasets/DilushaChandrasiri/SriLanka-Bird-Diversity-Dataset)
on Hugging Face (CC-BY-4.0). A single 403 MB tabular CSV, not an image set.

### 429-class species classification cannot reach 95% on this data

This is a property of the dataset, not a modelling failure:

* The 1,552,048 rows collapse to just **63,392 unique (latitude, longitude, date)
  combinations** — they are checklist-style records. Every environmental column is
  derived from location and date, so roughly **24 rows share a byte-identical
  feature vector while carrying 24 different species labels**.
* The **Bayes-optimal top-1 accuracy is therefore 5.54%** using every environmental
  and geographic feature present. Including `individualCount`, which breaks some
  ties, lifts the ceiling only to **31.02%**. No classifier can exceed this.
* Measured baselines agree: Random Forest reaches 2.5% (environment + geography +
  time) and 5.9% once `individualCount` is added.
* Even the *ceiling* for top-50-of-429 is 96.0%.
* The `index` column does not leak the label either — the probability that two
  adjacent rows share a species is 1.04%, i.e. chance.

`stateProvince` (25 classes) is the dataset's natural multi-class target: it is
well posed, genuinely predictable from the environmental features, and clears 95%.

### Honesty of the 95%+ figure

Province correlates strongly with latitude/longitude, so three guards are reported:

1. **grouped protocol** — `GroupShuffleSplit` on (lat, lon) means every test site is
   absent from training, so the score cannot come from memorised coordinates.
2. **`envonly` feature set** — latitude and longitude dropped entirely, leaving the
   model only climate, habitat and air-quality variables. Still **99.71%**.
3. **Both at once** (`envonly` + `grouped`) — **90.32%**, the one configuration that
   falls below 95%, reported rather than buried. The gap from 99.71% shows that part
   of the random-split score comes from recognising a site's environmental
   fingerprint rather than from generalisable climate structure.

All three are on the *Model comparison* tab. The headline number is the random
split, the conventional protocol.
""")
