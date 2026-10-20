# 05 Clustering: customer types

The second approach, next to the hazard model (`02`). It does three jobs, taken from `../research/03-clustering.md`:

1. **Find customer types** (k-means on April to September behaviour, built on the training customers only).
2. **Per-cluster metrics**: the hazard model's AUC inside each cluster, with no refitting.
3. **Leave-one-cluster-out**: refit the hazard model without one cluster and score it on that cluster, which is the
   brief's "leave out some types of data and observe your performance".

It also tests whether clusters *help prediction*, with the Bijak & Thomas (2012) design: a cluster label as a covariate,
and a separate model per cluster. They found segmentation gave no considerable gain on real credit data. Whether we see the same is an
open question here, and either answer is worth reporting.

## Where the code comes from

* [`sklearn.cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and
  [`silhouette_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), following
  scikit-learn's example
  [*Selecting the number of clusters with silhouette analysis on KMeans clustering*](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
  (silhouette: Rousseeuw, 1987).
* The leave-one-cluster-out loop and the per-cluster scoring are my own.
* Needs `01-person-month-data.md` run first (for the `hazard-*.csv` files) and the shared split in `data/processed/`.

## Leakage rules (from the research notes)

* Clusters are fitted on **train** customers only; validation and test customers are assigned to the nearest centre.
* Only April to September data and static covariates go into the clustering. **The October label is never an input**
  (it is used only afterwards, to describe the clusters).
* The test set is not used here at all.

## 1. Customer-level features, April to September

One row per customer, built from the cleaned split. `PAY_k` is the status in month k (1 = September), and `-2/-1/0` are
status codes, not months of delay, so "months behind" counts only `PAY_k >= 1`.

```python
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "processed").exists())
PROCESSED = ROOT / "data" / "processed"
Y = "default payment next month"
SEED = 2026

raw = {part: pd.read_csv(PROCESSED / f"taiwan-{part}.csv", index_col="ID") for part in ("train", "validation")}


def customer_features(df):
    """Six months of behaviour -> a handful of numbers per customer. No October information."""
    status = df[[f"PAY_{k}" for k in range(1, 7)]]
    util = pd.DataFrame({k: (df[f"BILL_AMT{k}"] / df["LIMIT_BAL"]).clip(-0.5, 1.5) for k in range(1, 7)})
    paid = pd.DataFrame({k: (df[f"PAY_AMT{k}"] / df[f"BILL_AMT{k + 1}"]).where(df[f"BILL_AMT{k + 1}"] > 0, 1).clip(0, 1)
                         for k in range(1, 6)})
    months = np.arange(6, 0, -1)                         # April..September as 1..6 (column 6 is April)
    util_by_time = util[[6, 5, 4, 3, 2, 1]]              # oldest first
    f = pd.DataFrame(index=df.index)
    f["mean_util"] = util.mean(axis=1)
    f["util_trend"] = ((util_by_time.to_numpy() - util_by_time.mean(axis=1).to_numpy()[:, None])
                       * (np.arange(6) - 2.5)).sum(axis=1) / ((np.arange(6) - 2.5) ** 2).sum()   # slope per month
    f["months_behind"] = (status >= 1).sum(axis=1)       # how many of the 6 months were late
    f["worst_status"] = status.clip(lower=0).max(axis=1)
    f["months_inactive"] = (status == -2).sum(axis=1)    # -2 is thought to mean "no consumption"
    f["mean_payratio"] = paid.mean(axis=1)
    f["log_limit"] = np.log(df["LIMIT_BAL"])
    f["age"] = df["AGE"]
    return f


feats = {part: customer_features(df) for part, df in raw.items()}
print(feats["train"].describe().T[["mean", "std", "min", "max"]].round(2))
```

## 2. Choose k on the training customers

Standardise with the training mean and sd, then fit k-means for several k. The silhouette is computed on a 5,000-customer
sample (the full 18,000 is slow and adds nothing). Also look at inertia, and, most importantly, whether the clusters
make sense to a person.

```python
scaler = StandardScaler().fit(feats["train"])
Z = {part: pd.DataFrame(scaler.transform(f), index=f.index, columns=f.columns) for part, f in feats.items()}

rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z["train"])
    rows.append({"k": k, "inertia": km.inertia_,
                 "silhouette": silhouette_score(Z["train"], km.labels_, sample_size=5000, random_state=SEED),
                 "smallest cluster": int(np.bincount(km.labels_).min())})
print(pd.DataFrame(rows).round(3).to_string(index=False))
```

## 3. Fit the chosen k and describe the clusters

Set `K` after looking at section 2: the silhouette peak, but not at the cost of tiny or unreadable clusters. `K = 5` is
the starting value (the research notes expect 4 to 6). The October default rate is shown here *only to describe* the
clusters; it was not an input.

```python
K = 5
km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(Z["train"])

cluster = {"train": pd.Series(km.labels_, index=Z["train"].index, name="cluster"),
           "validation": pd.Series(km.predict(Z["validation"]), index=Z["validation"].index, name="cluster")}

profile = feats["train"].groupby(cluster["train"]).mean()
profile["customers"] = cluster["train"].value_counts().sort_index()
profile["default rate (Oct)"] = raw["train"][Y].groupby(cluster["train"]).mean()
print(profile.round(2).to_string())
print("\nvalidation customers per cluster:", cluster["validation"].value_counts().sort_index().to_dict())
```

Give each cluster a plain-English name from the profile table (for example "pays in full", "revolves", "drifting
behind", "chronically late", "inactive"), and say how stable they are: rerun with another `SEED` and compare.

## 4. The hazard model, for scoring within clusters

The same logit hazard as `02` (the code is repeated so this file runs on its own).

```python
import statsmodels.api as sm

OCTOBER = 7
hz = {part: pd.read_csv(PROCESSED / f"hazard-{part}.csv", index_col="ID") for part in ("train", "validation")}
FEATURES = [c for c in hz["train"].columns if c not in ("t", "event")]


def fit_logit(data, features=FEATURES):
    # Inside one cluster some 0/1 columns can be all zero (e.g. no customer with status -2); drop them,
    # or the design matrix is rank-deficient. predict_october uses whichever columns the fit kept.
    keep = [f for f in features if data[f].nunique() > 1]
    X = sm.add_constant(data[keep].astype(float))
    return sm.GLM(data["event"], X, family=sm.families.Binomial()).fit()


def predict_october(res, data, features=None):
    rows = data[data["t"] == OCTOBER]
    keep = [c for c in res.model.exog_names if c != "const"]
    p = res.predict(sm.add_constant(rows[keep].astype(float), has_constant="add"))
    return pd.Series(np.asarray(p), index=rows.index)


def auc(y, p):
    return roc_auc_score(y, p) if y.nunique() == 2 else np.nan


def with_cluster(part):
    """Hazard rows with each customer's cluster attached (by ID, so no index join on repeated IDs)."""
    df = hz[part].copy()
    df["cluster"] = cluster[part].reindex(df.index).to_numpy()
    return df


train, val = with_cluster("train"), with_cluster("validation")
full = fit_logit(train)
p_val = predict_october(full, val)
val_oct = val[val["t"] == OCTOBER]
y_val = val_oct["event"]
print("overall validation AUC:", round(roc_auc_score(y_val, p_val), 4))
```

## 5. Metrics within each cluster (no refitting)

An overall AUC can hide a model that is good on easy customers and weak on the risky ones. Within a cluster, the
"easy separation between clusters" part of AUC is removed. A bootstrap interval is given because some clusters have few
defaults.

```python
rng = np.random.default_rng(SEED)


def bootstrap_auc(y, p, n=300):
    y, p = y.to_numpy(), np.asarray(p)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, [2.5, 97.5])


rows = []
for c in range(K):
    mask = (val_oct["cluster"] == c).to_numpy()
    y_c, p_c = y_val[mask], p_val[mask]
    lo, hi = bootstrap_auc(y_c, p_c)
    rows.append({"cluster": c, "customers": int(mask.sum()), "defaults": int(y_c.sum()),
                 "default rate": y_c.mean(), "AUC": auc(y_c, p_c), "95% low": lo, "95% high": hi})
within = pd.DataFrame(rows)
print(within.round(3).to_string(index=False))
```

## 6. Leave one cluster out

For each cluster: refit the hazard model on the training customers **not** in that cluster, then score the validation
customers who **are** in it (types of customer the model has never seen). Compare with the full model on the same
customers. A large drop means the model does not generalise to that type.

```python
rows = []
for c in range(K):
    left_out = fit_logit(train[train["cluster"] != c])
    mask = (val_oct["cluster"] == c).to_numpy()
    y_c = y_val[mask]
    p_without = predict_october(left_out, val)[val_oct.index][mask]
    rows.append({"left out": c, "AUC with cluster in training": auc(y_c, p_val[mask]),
                 "AUC with cluster left out": auc(y_c, p_without)})
loco = pd.DataFrame(rows)
loco["change"] = loco["AUC with cluster left out"] - loco["AUC with cluster in training"]
print(loco.round(4).to_string(index=False))
```

## 7. Do clusters help prediction? (Bijak & Thomas test)

Two uses of the clusters as a *model*: the label as a covariate, and a separate hazard model per cluster. Compared with the
single model's validation AUC from section 4.

```python
cluster_dummies = [f"cluster_{c}" for c in range(1, K)]


def add_dummies(df):
    df = df.copy()
    for c in range(1, K):
        df[f"cluster_{c}"] = (df["cluster"] == c).astype(int)
    return df


train_d, val_d = add_dummies(train), add_dummies(val)

# (a) cluster label as a covariate
as_covariate = fit_logit(train_d, FEATURES + cluster_dummies)
auc_covariate = roc_auc_score(y_val, predict_october(as_covariate, val_d))

# (b) a separate model per cluster, predictions stitched back together
p_segment = pd.Series(index=val_oct.index, dtype=float)
for c in range(K):
    model_c = fit_logit(train[train["cluster"] == c])
    in_c = val[val["cluster"] == c]
    p_segment[val_oct.index[(val_oct["cluster"] == c).to_numpy()]] = predict_october(model_c, in_c)
auc_segment = roc_auc_score(y_val, p_segment)

print(pd.Series({"single model": roc_auc_score(y_val, p_val), "cluster label as covariate": auc_covariate,
                 "separate model per cluster": auc_segment}).round(4))
```

## What to say in the report

* **Per-cluster AUC** (section 5) is the honest picture of where the hazard model is weak. Quote the intervals, not
  just the point estimates.
* **Leave-one-cluster-out** (section 6) is the generalisation evidence the brief asks for. The same procedure applies to
  Oscar's and Louis's models, which needs each of them to make their fitting code callable as a function.
* **Section 7:** Bijak & Thomas (2012) found no considerable gain from segmentation. In a first run here, a cluster
  label as a covariate added little, but a separate model per cluster raised validation AUC by about 0.015. Do not
  over-read it: the clusters are built from status and payment behaviour, so separate models mostly add status-by-
  behaviour interactions that the single logit lacks. Check it against a single model with those interactions, against
  another seed and k, and on a bootstrap interval, before claiming that segmentation helps.
* **Stability:** k-means depends on the features chosen and the seed. State both, and say how the clusters change with
  another seed or with k - 1 and k + 1.
