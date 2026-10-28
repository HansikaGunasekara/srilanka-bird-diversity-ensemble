"""
Train-vs-test accuracy for the deployed bundle, on the exact split train.py used.

This is the measurement that actually tests for overfitting: a model that has
memorised its training set shows a large train-test gap. Accuracy on held-out
data being high is not itself evidence of overfitting.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, joblib
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prep import load

TARGET = "stateProvince"
SEED, SAMPLE = 0, 300_000

bundle = joblib.load(os.path.join(ROOT, "artifacts", "ensemble_deploy.joblib"))
feats, classes = bundle["features"], bundle["classes"]

df = load()
sub = df.sample(SAMPLE, random_state=SEED)
X = sub[feats].to_numpy(np.float32)
ycat = sub[TARGET].astype("category")
y = ycat.cat.codes.to_numpy()
assert list(ycat.cat.categories) == classes, "class order drifted"

itr, ite = train_test_split(np.arange(len(y)), test_size=0.25,
                            random_state=SEED, stratify=y)
print(f"train {len(itr):,}  test {len(ite):,}", flush=True)

def proba(m, Xa):
    pr = np.zeros((len(Xa), len(classes)), dtype=np.float32)
    pr[:, m.classes_.astype(int)] = m.predict_proba(Xa)
    return pr

out = {}
store = {"train": {}, "test": {}}
for name in bundle["names"]:
    m = bundle["base"][name]
    t0 = time.time()
    row = {}
    for split, idx in (("train", itr), ("test", ite)):
        pr = proba(m, X[idx])
        store[split][name] = pr
        row[split] = float((pr.argmax(1) == y[idx]).mean())
    row["gap"] = row["train"] - row["test"]
    out[name] = row
    print(f"{name:<22} train={row['train']:.4f} test={row['test']:.4f} "
          f"gap={row['gap']:+.4f}  ({time.time()-t0:.0f}s)", flush=True)

# --- the two ensemble combiners
w, names = bundle["weights"], list(bundle["names"])
meta = bundle["meta"]
for key in ("SoftVote", "Stacking"):
    row = {}
    for split, idx in (("train", itr), ("test", ite)):
        if key == "SoftVote":
            pr = sum(wi * store[split][n] for wi, n in zip(w, names))
        else:
            Z = np.hstack([store[split][n] for n in names])
            pr = np.zeros((len(idx), len(classes)), dtype=np.float32)
            pr[:, meta.classes_.astype(int)] = meta.predict_proba(Z)
        row[split] = float((pr.argmax(1) == y[idx]).mean())
    row["gap"] = row["train"] - row["test"]
    out[key] = row
    print(f"{key:<22} train={row['train']:.4f} test={row['test']:.4f} "
          f"gap={row['gap']:+.4f}", flush=True)

res = {"split": {"n_train": int(len(itr)), "n_test": int(len(ite)),
                 "sample": SAMPLE, "seed": SEED, "protocol": "random",
                 "feature_set": "deploy"},
       "note": ("Stacking's train figure feeds the meta-learner the base models' "
                "in-sample probabilities, which are more confident than the "
                "out-of-fold probabilities it was fitted on; read it as an upper "
                "bound on its train-side optimism."),
       "results": out}
p = os.path.join(ROOT, "artifacts", "overfitting_check.json")
json.dump(res, open(p, "w"), indent=2)
print("wrote", p)
