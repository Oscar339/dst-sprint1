# 04 Cox with time-varying covariates

`../research/01-hazard-models.md`, model 7: the continuous-time counterpart of `02`. Same person-month rows, but
each row is an interval $(t-1, t]$ and the Cox model

$$h(t \mid x) = h_0(t)\,\exp(\beta^\top x_{t-1})$$

is fitted by partial likelihood. It is a **comparison model**, not a likely winner: the data is monthly, so there are
thousands of events in each month (massive ties), and lifelines gives no direct prediction from this fitter. Both
points are the "what goes wrong" evidence for the report.

## Where the code comes from

* [lifelines, *Time varying survival regression*](https://lifelines.readthedocs.io/en/stable/Time%20varying%20survival%20regression.html):
  the long data layout (`id`, `start`, left-exclusive, `stop`, right-inclusive, covariates, `event`) and the call
  `CoxTimeVaryingFitter(penalizer=...).fit(df, id_col=, event_col=, start_col=, stop_col=)`.
* [`CoxTimeVaryingFitter` API](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxTimeVaryingFitter.html):
  ties are handled by Efron's method.
* The prediction step below (October probability from the baseline hazard increment) is my own. It follows from
  lifelines' source: `baseline_cumulative_hazard_` is the Breslow-type baseline for the *centred* covariates, and
  `predict_partial_hazard` returns $\exp((x-\bar x)^\top\beta)$, so the two multiply correctly.

## 1. Load and set up the long format

No month dummies: in a Cox model the baseline hazard $h_0(t)$ is the month effect. The risk set is "everyone, every
month" (an Andersen-Gill style recurrent-event set-up), so a customer can have several events.

```python
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "processed").exists())
PROCESSED = ROOT / "data" / "processed"
OCTOBER = 7

hz = {part: pd.read_csv(PROCESSED / f"hazard-{part}.csv", index_col="ID") for part in ("train", "validation")}
FEATURES = [c for c in hz["train"].columns if c not in ("t", "event") and not c.startswith("t_")]


def long_format(df):
    out = df[FEATURES].astype(float)
    out["event"] = df["event"]
    out["id"] = df.index
    out["start"] = df["t"].to_numpy() - 1           # the interval (t-1, t]
    out["stop"] = df["t"].to_numpy()
    out["event"] = out["event"].astype(bool)
    return out.reset_index(drop=True)


# Float columns: lifelines standardises them internally, and 0/1 integer columns cause trouble
train, val = long_format(hz["train"]), long_format(hz["validation"])
print(train.head(6)[["id", "start", "stop", "event", "util_lag", "status_lag_2"]])
```

## 2. Fit

A small ridge penalty (`penalizer`) is there for numerical stability, not for regularisation. **The default Newton step
(0.95) diverges on this data** (coefficients of about 45 and a `ConvergenceWarning`), because the status dummies are
nearly separable in some months. A smaller step, passed through `fit_options`, converges cleanly. Always check for the
warning.

```python
ctv = CoxTimeVaryingFitter(penalizer=0.01)
ctv.fit(train, id_col="id", event_col="event", start_col="start", stop_col="stop",
        fit_options={"step_size": 0.3}, show_progress=False)
ctv.print_summary(decimals=3, columns=["coef", "exp(coef)", "se(coef)", "z", "p"])
```

## 3. Predict P(default in October) and score

lifelines has no `predict` on this fitter, so build the October hazard by hand. The baseline hazard *increment* for
month 7 is the cumulative baseline at 7 minus at 6, scaled by the customer's partial hazard:

$$\hat h(7 \mid x) = 1 - \exp\{-\,[\hat H_0(7) - \hat H_0(6)]\; e^{(x-\bar x)^\top\hat\beta}\}$$

```python
H0 = ctv.baseline_cumulative_hazard_["baseline hazard"]
print(H0.round(4).to_string())
h0_october = H0.loc[OCTOBER] - H0.loc[OCTOBER - 1]

val_oct = val[val["stop"] == OCTOBER]
partial = ctv.predict_partial_hazard(val_oct[FEATURES])
p = 1 - np.exp(-h0_october * partial.to_numpy())

y = val_oct["event"].astype(int)
print(f"\nCox time-varying, validation October:  AUC={roc_auc_score(y, p):.4f}  "
      f"Brier={brier_score_loss(y, p):.4f}  log loss={log_loss(y, p):.4f}")
```

## 4. What this shows (and what to say about it)

* **Ranking** (AUC) should be close to the discrete-time models, because they use the same covariates and rows.
* **Probabilities** (Brier, log loss) depend on the baseline hazard estimate. With only five distinct event times, the
  Breslow-type baseline is crude, and the October event is a *real default* where the earlier months are a *proxy*.
  Compare against `02` before concluding anything.
* **Ties:** every event happens at one of five times, so Efron's approximation is doing a lot of work. The discrete-time
  model is exact for grouped data. This is the reason to prefer `02` here.
* **Prediction:** to predict beyond one month ahead, future covariates are needed; lifelines does not do that for this
  fitter. Options are to freeze covariates at September, or to simulate them (Louis's Markov chain could supply that).

## Notes for the report

The model is included to show *why* the discrete-time form is the better fit to monthly data, with numbers, not to
compete for the best score.
