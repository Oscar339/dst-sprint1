# 02 Discrete-time hazard: logit and cloglog

The core hazard models (`../research/01-hazard-models.md`, models 1 and 2). Both are a binomial GLM on the
person-month table from `01-person-month-data.md`:

$$\operatorname{link}\, h(t \mid x) = \alpha_t + \beta^\top x_{t-1}$$

* **logit**: $\log\frac{h}{1-h}$, coefficients are log odds ratios of the monthly hazard.
* **cloglog**: $\log(-\log(1-h))$, the exact grouped-time version of the Cox model (Prentice & Gloeckler, 1978);
  $\exp(\beta)$ is a hazard ratio.

$\alpha_t$ is one intercept per month (the month dummies from `01`), so the baseline hazard is fully flexible.

## Where the code comes from

* The recipe (expand to person-period rows, binomial GLM on the 0/1 event, a per-period intercept as the baseline
  hazard) is the standard one in Allison (1982) and Singer & Willett (1993), and is explained for Python by
  [Andrew Wheeler, *Discrete time survival models in python*](https://andrewpwheeler.com/2020/09/26/discrete-time-survival-models-in-python/)
  (2020; his code is in [apwheele/Blog_Code](https://github.com/apwheele/Blog_Code/tree/master/Python/Discrete_Time)).
  He uses a logistic regression with splines for time; here the months are dummies, since we have only five.
* The credit-risk version of the same idea, with a worked tutorial, is Botha, Verster & Scheepers (2025),
  [arXiv:2507.15441](https://arxiv.org/abs/2507.15441). I could not find their code online, so none is copied from it.
* The statsmodels pieces are from its documentation: [`sm.GLM`](https://www.statsmodels.org/stable/glm.html)
  with `families.Binomial(link=...)`, and `fit(cov_type="cluster", ...)` for customer-clustered standard errors.
  The link objects are `links.Logit()` and `links.CLogLog()`.

## 1. Load the person-month tables

Run `01-person-month-data.md` first.

```python
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "processed").exists())
PROCESSED = ROOT / "data" / "processed"
OCTOBER = 7

# Train and validation only. The test set is not touched until the model is final (see README.md).
hz = {part: pd.read_csv(PROCESSED / f"hazard-{part}.csv", index_col="ID") for part in ("train", "validation")}

FEATURES = [c for c in hz["train"].columns if c not in ("t", "event")]
train, val = hz["train"], hz["validation"]
print(len(FEATURES), "features;", len(train), "training rows;", len(val), "validation rows")
```

## 2. Fit

`sm.add_constant` gives the intercept, which together with the month dummies is $\alpha_t$ (June is the reference month).
The clustered standard errors treat each customer's five rows as one group; the coefficients are unchanged.

```python
def fit_hazard(link, data=train, features=FEATURES):
    X = sm.add_constant(data[features].astype(float))
    model = sm.GLM(data["event"], X, family=sm.families.Binomial(link=link))
    return model.fit(cov_type="cluster", cov_kwds={"groups": data.index.to_numpy()})


fits = {"logit": fit_hazard(sm.families.links.Logit()),
        "cloglog": fit_hazard(sm.families.links.CLogLog())}

for name, res in fits.items():
    print(f"\n=== {name} ===")
    print(res.summary().tables[1])
```

## 3. Score on the validation customers' October row

The predicted probability of default in October is the hazard $\hat h(7 \mid x_6)$ for the customer's `t = 7` row
(September covariates). That is the one number per customer that every group model has to produce.

```python
def october(data):
    return data[data["t"] == OCTOBER]


def predict_october(res, data, features=FEATURES):
    rows = october(data)
    return pd.Series(res.predict(sm.add_constant(rows[features].astype(float), has_constant="add")),
                     index=rows.index, name="probability")


def score(y, p):
    return {"AUC": roc_auc_score(y, p), "avg precision": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "log loss": log_loss(y, p)}


y_val = october(val)["event"]
results = {f"hazard {name}": score(y_val, predict_october(res, val)) for name, res in fits.items()}
print(pd.DataFrame(results).T.round(4))
```

## 4. The comparison that matters: pooled months versus October only

The same logistic model fitted on the October rows alone is the ordinary one-row-per-customer model (Oscar's
setup, with my features). If the hazard model has any advantage it is from using the extra months, so this is the
fair test of that. It goes in the report whichever way it falls.

```python
oct_features = [f for f in FEATURES if not f.startswith("t_")]      # no month dummies: it is all October
oct_fit = fit_hazard(sm.families.links.Logit(), data=october(train), features=oct_features)
results["October rows only (logit)"] = score(y_val, predict_october(oct_fit, val, features=oct_features))
print(pd.DataFrame(results).T.round(4))
```

## 5. Month effects and hazard ratios

```python
res = fits["cloglog"]
table = pd.DataFrame({"coef": res.params, "hazard ratio": np.exp(res.params), "z": res.tvalues})
print(table.round(3).sort_values("z", key=abs, ascending=False).head(12))
print("\nbaseline month effects (cloglog), relative to June:")
print(table.filter(like="t_", axis=0).round(3))
```

## 6. Expected time to default (what a hazard model adds)

From the monthly hazards we can build a survival curve, $S(t) = \prod_{s \le t}(1 - h_s)$. Here, for each validation
customer with covariates held at their September values, the chance of an event in at least one of the five modelled
months. This is **not** the October-only number above; it is the cumulative version the one-row models cannot give.

```python
res = fits["logit"]
rows = val[val["t"] == OCTOBER].copy()                      # September covariates, one row per customer
hazards = []
for t in range(3, OCTOBER + 1):
    X = rows[FEATURES].astype(float).copy()
    for c in [f for f in FEATURES if f.startswith("t_")]:
        X[c] = int(c == f"t_{t}")                           # switch the month dummy
    hazards.append(res.predict(sm.add_constant(X, has_constant="add")))
survival = np.prod([1 - h for h in hazards], axis=0)
print("mean chance of at least one event over t = 3..7:", round(float(1 - survival.mean()), 3))
```

## 7. Saving test predictions (not run yet)

The brief's rule is that nothing is chosen on the test set. When the hazard model is final, run this once. The
file name follows the repo convention (`data/output/predictions-<name>.csv`); the columns are `ID` and the
predicted probability of default.

```python
def write_test_predictions(name, res, features=FEATURES):
    test = pd.read_csv(PROCESSED / "hazard-test.csv", index_col="ID")
    p = predict_october(res, test, features)
    out = ROOT / "data" / "output" / f"predictions-{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(out)
    return out

# write_test_predictions("hazard-logit", fits["logit"])      # <- uncomment only when the model is final
```

## Notes for the report

* logit and cloglog should give almost the same ranking here: monthly hazards are small (10 to 22%), where the two
  links nearly agree. Say so, and say why; the comparison is a "mathematical details of choices" result, not a win.
* The month dummies absorb the proxy-versus-label difference: `t_7` is the October level. Because the months before
  October use a proxy event, the shared coefficients $\beta$ are a blend of "what predicts being 2+ months behind"
  and "what predicts October default". Section 4 shows whether that blend helps or hurts.
* Standard errors are clustered by customer; point estimates are unchanged by that.
