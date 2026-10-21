"""
Reproduces the impossibility proof: 429-class species classification cannot reach
95% on this dataset, because the label is not a function of the features.

The dataset is checklist-structured. Every environmental column is derived from
(latitude, longitude, date), so many rows share a byte-identical feature vector
while carrying different species labels. For any such group of identical inputs
the best any classifier can do is predict that group's most frequent label, so

    Bayes-optimal accuracy = sum_over_groups( max_label_count ) / n_rows

is a hard ceiling for every model, ensemble or otherwise.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load, ENV, GEO, TIME

TARGET = "verbatimScientificName"


def ceiling(df, keys, name):
    n = len(df)
    groups = df.groupby(keys, observed=True, sort=False).ngroups
    counts = df.groupby(keys + [TARGET], observed=True, sort=False).size()
    best = counts.groupby(level=list(range(len(keys))), observed=True).max().sum()
    print(f"  {name:46s} groups={groups:>8,}  mean_size={n/groups:6.2f}  "
          f"ceiling={best/n:7.4f}")
    return best / n


if __name__ == "__main__":
    df = load()
    n = len(df)
    print(f"rows={n:,}  species={df[TARGET].nunique()}  provinces={df.stateProvince.nunique()}\n")

    CHECKLIST = GEO + ["year", "month", "doy"]
    print("=== Bayes-optimal top-1 accuracy for 429-class species classification ===")
    ceiling(df, CHECKLIST, "lat + lon + date (the checklist key)")
    ceiling(df, GEO + ENV + TIME, "every env + geo + time feature")
    ceiling(df, GEO + ENV + TIME + ["individualCount"], "...plus individualCount")
    ceiling(df, GEO, "lat + lon only (the site)")

    print("\n=== top-k ceilings (same checklist grouping) ===")
    counts = df.groupby(CHECKLIST + [TARGET], observed=True, sort=False).size()
    tops = (counts.groupby(level=list(range(len(CHECKLIST))), observed=True)
                  .apply(lambda s: np.sort(s.values)[::-1]))
    for k in (1, 3, 5, 10, 20, 30, 50, 75, 100, 150):
        print(f"  top-{k:<4d} ceiling = {sum(t[:k].sum() for t in tops)/n:.4f}")

    print("\n=== does the `index` column leak the label? ===")
    print("  (`index` is monotonic in the raw CSV, so adjacency is original file order)")
    y = df[TARGET].values
    print(f"  P(two adjacent rows share a species) = {(y[1:] == y[:-1]).mean():.4f}"
          f"   [chance = {1/df[TARGET].nunique():.4f}]")

    print("\n=== for contrast: the province target ===")
    ceiling(df.assign(**{TARGET: df.stateProvince}), GEO + ENV + TIME,
            "stateProvince <- every env + geo + time feature")
