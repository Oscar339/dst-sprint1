# Initial research — Alex

15 Oct 2026. Background reading for my part of Sprint 1: hazard (survival)
models, the choices inside them, clustering, and my share of the metric work.
**Draft notes to be rewritten**, not report text.

Builds on Sprint 0 (`dst-sprint0/Alex/02-survival-analysis.md` and
`02-hazard-model.md`), so the basics (survival function, censoring,
Kaplan–Meier, Cox) are not repeated here.

| File | What it covers |
|---|---|
| `01-hazard-models.md` | The hazard model families, pros and cons of each, and how well each fits our data |
| `02-model-choices.md` | What we can vary inside a hazard model (event, risk set, covariates, link, baseline...) |
| `03-clustering.md` | Whether clustering is useful, and three ways to use it |
| `04-metrics-and-generalisation.md` | Survival metrics, and testing whether the agreed metric reflects generalisation |
| `05-other-ideas.md` | Landmarking, simulation, cost-based scoring, and other smaller ideas |
| `references.md` | Every source, with links. ✓ = details checked online on 15 Oct 2026 |

## Who does what (agreed so far)

| Person | Models |
|---|---|
| Oscar | GLM (logistic regression), EDA |
| Louis | Markov chain, decision trees |
| Alex | Hazard / survival models; clustering idea |

**Overlaps to agree with Louis:**
- A Markov chain with an absorbing "default" state is a multi-state hazard model (Leow & Crook, 2014).
- Random survival forests are trees.

Suggest: I keep time-to-event models, Louis keeps state transitions and trees,
and we each write a paragraph in the report on how the two relate.

## Constraints from our data (from Sprint 0)

- 7 monthly time points: Apr–Sep 2005 repayment status, then the October default label.
- No account age, so the data are left-truncated (customers had accounts before April).
- No default date. Monthly events have to be built from `PAY_*`, e.g. "2+ months behind". The October label is the only real default.
- 22% default rate.
- Sprint 0 hazard model:
  - on the October label, AUC 0.613 / Brier 0.145;
  - logistic regression with the same covariates scored 0.632 / 0.137;
  - it dropped the 17% of customers already behind in Apr–Jun.
- For Sprint 1, every model must give P(default in October) for **every** test ID on the shared split.
