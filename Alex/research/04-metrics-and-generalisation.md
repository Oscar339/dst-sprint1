# Metrics and generalisation

The brief: half the effort goes on the performance metric. When does it say
something about real-world generalisation, and is our best model truly best?
Show it with data or simulation.

## 1. The group metric (agreed together)

Sprint 0 suggested Brier score + AUC as headlines, plus a confusion matrix at
a threshold chosen on validation.

| Metric | Measures | Pros | Cons |
|---|---|---|---|
| AUC / Gini | Ranking | Industry standard, threshold-free | Ignores calibration. Hand (2009) shows it implicitly uses a different cost weighting for each model, so comparisons between models can be incoherent |
| Brier score | Probability accuracy | Proper scoring rule (Gneiting & Raftery, 2007); combines calibration and ranking | Hard to interpret alone; dominated by the 78% non-defaulters |
| Log loss | Probability accuracy | Proper; punishes confident errors | Unbounded; one bad prediction dominates |
| H-measure | Ranking under a stated cost distribution | Hand's (2009) fix for AUC's incoherence | Less familiar; needs `hmeasure` package |
| Expected cost / profit | Decision value | Closest to real-world use | Needs a cost ratio we have to assume |

## 2. Survival-specific metrics (my extras)

Not suitable as the group headline (other models don't produce a term
structure), but worth reporting for the hazard models.

| Metric | Source | Note |
|---|---|---|
| Harrell's C-index | Harrell et al. (1982) | Optimistic with heavy censoring (scikit-survival docs). Ours: ~78% censored |
| Uno's IPCW C-index | `sksurv.metrics.concordance_index_ipcw` | Corrects censoring bias; needs training outcomes for the censoring distribution |
| Time-dependent AUC | `sksurv.metrics.cumulative_dynamic_auc` | AUC at each horizon t |
| Time-dependent / integrated Brier score | Graf et al. (1999) | Sprint 0 caveat: everyone is censored in October, so weights explode at the last time. Stop at month 6 |
| Term-structure MAE | Botha & Muller (2025) code | Actual vs expected default rate by month; calibration over time |

## 3. Generalisation experiments (my share)

### a. Leave-one-cluster-out
See `03-clustering.md` §1. Covariate shift on purpose.

### b. Temporal validation
- Train on transitions up to August, test on the September → October step.
- Is month-ahead performance the same as random-split performance?
- Only time-indexed models (mine, Louis's Markov chain) can do this cleanly. The others can approximate it by training on an earlier month's proxy label.
- Connects to concept drift: Bellotti & Crook (2009, 2013) bring in macroeconomic variables because default behaviour changes over time. The Taiwan data is from 2005, the time of the Taiwan card debt crisis (Yeh & Lien, 2009). **Check this claim and find a source.**

### c. Is the best model truly best?
- **Bootstrap confidence intervals** on each model's test metric.
- **Paired comparisons** on the same test customers: DeLong et al. (1988) for AUC, paired bootstrap for Brier.
- Sprint 0's AUC gap was 0.632 vs 0.613. Is that bigger than noise on 6,000 test customers?
- **Rank stability:** how often does the winner change across bootstrap resamples, metrics or clusters?

### d. Simulation: when do metrics disagree?
Using a simulated dataset where the true P(default) is known (see `05-other-ideas.md`):
- A model with good ranking but bad calibration vs one with good calibration but weaker ranking. AUC and Brier should pick different winners.
- Score each model against the **true** probabilities, not just the 0/1 labels. Which metric best tracks the true error?

## 4. Context from the literature

- Lessmann et al. (2015) benchmarked many classifiers across 8 credit datasets and found the ranking of methods depends on the performance measure. Exactly our point; cite it.
- Tong et al. (2012): Cox, cure and logistic models had similar discrimination, but survival models were better calibrated. So AUC alone would miss the survival models' advantage.
- Yeh & Lien (2009), on our dataset, compared methods on predicted probability of default, not just classification accuracy.
