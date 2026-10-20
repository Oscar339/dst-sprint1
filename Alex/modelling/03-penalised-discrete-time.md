# 03 Penalised discrete-time hazard (L1 / L2)

`../research/01-hazard-models.md`, model 3: the logit hazard of `02` with a lasso (L1) or ridge (L2) penalty,
$C$ chosen on the validation set. This mirrors Oscar's "C, L1 vs L2" tuning in `Oscar/02-logistic-regression.ipynb`,
so the comparison with his model is like for like: the only difference is the data structure (person-months, not one
row per customer).

## Where the code comes from

* [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
  with `C` the *inverse* penalty strength (small C = strong penalty). The scikit-learn user guide section
  [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) gives the
  penalised objective.
* A penalised discrete-time hazard is a penalised logistic regression on the person-period table
  (Tutz & Schmid, 2016, ch. 7), so no special software is needed.
* The penalised **cloglog** version is not in scikit-learn. statsmodels has
  [`GLM.fit_regularized`](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit_regularized.html)
  (elastic net), used at the end.

## 1. Load

```python
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "processed").exists())
PROCESSED = ROOT / "data" / "processed"
OCTOBER = 7

hz = {part: pd.read_csv(PROCESSED / f"hazard-{part}.csv", index_col="ID") for part in ("train", "validation")}
FEATURES = [c for c in hz["train"].columns if c not in ("t", "event")]
train, val = hz["train"], hz["validation"]
val_oct = val[val["t"] == OCTOBER]
```

## 2. Tune C and the penalty on the validation set

The model is trained on all the training person-months. It is scored on the validation customers' October row,
because that is the number the group metric uses. The month dummies are penalised too (scikit-learn cannot skip
them); with 90,000 rows and 5 of them the shrinkage on them is tiny, but state it.

```python
GRID = np.logspace(-4, 1, 11)       # C from 0.0001 to 10
rows = []
for penalty, l1_ratio in (("l1", 1.0), ("l2", 0.0)):
    for C in GRID:
        # l1_ratio is how scikit-learn (1.8+) picks the penalty: 1 = lasso, 0 = ridge. `penalty=` is deprecated.
        model = LogisticRegression(l1_ratio=l1_ratio, C=C, solver="liblinear" if penalty == "l1" else "lbfgs",
                                   max_iter=2000)
        model.fit(train[FEATURES], train["event"])
        p = model.predict_proba(val_oct[FEATURES])[:, 1]
        rows.append({"penalty": penalty, "C": C, "AUC": roc_auc_score(val_oct["event"], p),
                     "log loss": log_loss(val_oct["event"], p),
                     "non-zero coefs": int((model.coef_ != 0).sum())})

tuning = pd.DataFrame(rows)
print(tuning.round(4).to_string(index=False))
best = tuning.loc[tuning["AUC"].idxmax()]
print("\nbest on validation AUC:", best.to_dict())
```

## 3. Plot AUC against C

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 3.6))
for penalty, colour in (("l1", "#eb6834"), ("l2", "#8a8984")):
    sub = tuning[tuning["penalty"] == penalty]
    ax.plot(sub["C"], sub["AUC"], marker="o", color=colour, label=penalty.upper())
ax.set_xscale("log")
ax.set_xlabel("C (small = strong penalty)")
ax.set_ylabel("Validation AUC, October")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
```

## 4. What the lasso kept

```python
best_l1_C = float(tuning[tuning["penalty"] == "l1"].sort_values("AUC").iloc[-1]["C"])
final = LogisticRegression(l1_ratio=1.0, C=best_l1_C, solver="liblinear", max_iter=2000).fit(train[FEATURES], train["event"])
coefs = pd.Series(final.coef_[0], index=FEATURES)
print("dropped by the lasso:", list(coefs[coefs == 0].index))
print(coefs[coefs != 0].sort_values(key=abs, ascending=False).round(3))
```

## 5. Penalised cloglog (statsmodels, slow)

`fit_regularized` is an elastic net: `alpha` is the penalty weight and `L1_wt=1` is the lasso. It is **slow here
(about two minutes for 50 iterations on 90,000 rows)**, so this is one fit at a single `alpha`, not a grid; skip it
unless the cloglog link turns out to matter. `alpha` is on the per-row scale, roughly $1/(C \cdot n)$ in
scikit-learn terms.

```python
X_train = sm.add_constant(train[FEATURES].astype(float))
X_val = sm.add_constant(val_oct[FEATURES].astype(float), has_constant="add")
glm = sm.GLM(train["event"], X_train, family=sm.families.Binomial(link=sm.families.links.CLogLog()))

alpha = 1e-4
fit = glm.fit_regularized(alpha=alpha, L1_wt=1.0, maxiter=50)
p = 1 - np.exp(-np.exp(np.asarray(X_val @ fit.params)))        # inverse cloglog link
print(f"cloglog lasso, alpha={alpha:g}  AUC={roc_auc_score(val_oct['event'], p):.4f}  "
      f"non-zero={int((fit.params != 0).sum())}")
```

## Notes for the report

* Tuning on the validation set and then quoting the validation score is optimistic. Quote the **test** score once,
  at the end, for the one chosen model (see `02`, section 7, for the writer).
* A penalty shrinks probabilities toward the middle, which can hurt calibration even when AUC is unchanged; compare
  log loss (and Brier) as well as AUC.
* **What it showed (validation, October row):** AUC rises with C and is flat at about 0.744 from C of roughly 1 upward,
  for both L1 and L2, and log loss is flat too. The unpenalised model is as good as any penalised one: with 90,000 rows
  and 19 features there is nothing to shrink. That is a finding, not a failure. The lasso only drops coefficients at
  small C, where AUC is clearly worse.
* The penalised cloglog fit stops with a `ConvergenceWarning` after 50 iterations. Its AUC (about 0.744) agrees with
  the others, but do not quote its coefficients without raising `maxiter`.
