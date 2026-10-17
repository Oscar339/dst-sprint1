# Metrics and generalisation

The brief: half the effort goes on the performance metric. When does it say
something about real-world generalisation, and is our best model truly best?
Show it with data or simulation.

## 1. The group metric (agreed together)

Sprint 0 suggested Brier score + AUC as headlines, plus a confusion matrix at
a threshold chosen on validation.

| Metric | Measures | Pros | Cons |
|---|---|---|---|
| AUC / Gini | Ranking | Industry standard, explainable | Ignores calibration. |
| Brier score | Probability accuracy | Proper scoring rule combines calibration and ranking | Hard to interpret |
| Log loss | Probability accuracy | Punishes confident errors | Unbounded so one bad prediction can dominate |
| H-measure | Ranking under a stated cost distribution | Hand's (2009) fix for AUC's incoherence | Less well known and needs `hmeasure` package |
| Expected cost / profit | Decision value | Closest to real-world use | Needs a cost ratio we have to assume |

## 2. Generalisation experiments

### a. Leave-one-cluster-out
See `03-clustering.md`

### b. Temporal validation
- Train on transitions up to August, test on the September → October step.
- Is month-ahead performance the same as random-split performance?
- Only time-indexed models (mine, Louis's Markov chain) can do this cleanly. The others can approximate it by training on an earlier month's proxy label.
 
### c. Is the "best" model truly best?
- Bootstrap confidence intervals on each model's test metric.
- Paired comparison on the same test customers: DeLong et al. (1988) for AUC, paired bootstrap for Brier.
- Sprint 0's AUC gap was 0.632 vs 0.613. Is that bigger than noise on 6,000 test customers?
- Rank stability: how often does the winner change across bootstrap resamples, metrics or clusters?

### d. Simulation: when do metrics disagree?
Using a simulated dataset where the true P(default) is known (see `05-other-ideas.md`):
- A model with good ranking but bad calibration vs one with good calibration but weaker ranking. AUC and Brier should pick different winners.
- Score each model against the **true** probabilities, not just the 0/1 labels. Which metric best tracks the true error?

## 4. Context from the literature

- Lessmann et al. (2015) benchmarked many classifiers across 8 credit datasets and found the ranking of methods depends on the performance measure. Cite it.
- Tong et al. (2012): Cox, cure and logistic models had similar discrimination, but survival models were better calibrated. So AUC alone would miss the survival models' advantage.
- Yeh & Lien (2009), on our dataset, compared methods on predicted probability of default, not just classification accuracy.
