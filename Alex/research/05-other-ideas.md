# Other ideas

## 1. Simulation testbed (high value, shared)

Simulate customers from a fitted discrete-time hazard model, so the true
monthly hazards and true P(October default) are known.

Uses:
1. **Check our code:** fit the model to simulated data and recover the true β.
2. **Shumway's (2001) argument:** fit a one-period logistic regression to the same simulated data. Show where its probabilities are biased (e.g. customers observed for different lengths of time).
3. **Metric experiments:** score every model against the true probabilities (`04-metrics-and-generalisation.md` §3d).
4. **Controlled shift:** change one feature of the simulation (e.g. a stronger effect of utilisation) and see which metric notices.

Method references:
- Bender, Augustin & Blettner (2005) for continuous-time (Cox) simulation.
- For discrete time it's simpler: each month, draw Bernoulli(h(t | x)) for every customer still at risk.

**Pros:** strong "insight" and "simulation" credit; reusable by Oscar and Louis.
**Cons:** the conclusions only hold if the simulation is realistic. Calibrate it to the real data and say so.

## 2. Landmarking

Fit a model only on customers still at risk at a landmark month, predicting
ahead from there (van Houwelingen, 2007). Stack several landmarks for one
"dynamic" model.

**Pros:** the cleanest way to get a fair P(October default) from time-varying
covariates without inventing future values. Bridges hazard and static models.
**Cons:** with 7 months, only a few landmarks are possible.

## 3. Expected-cost scoring

Turn probabilities into accept / decline decisions at a threshold chosen on
validation, and score: cost = c_FN × missed defaulters + c_FP × rejected good
customers.
- Plot each model's cost against the cost ratio c_FN / c_FP (e.g. 1 to 10). Does the winner change?
- This is the decision-level view of Hand's (2009) argument (`04-metrics...`).

## 4. Combining with Louis's Markov chain

To predict October from June, a hazard model needs July–September covariates.
Louis's Markov chain can simulate those status paths. Joint piece: "hazard
model + simulated covariates" vs "hazard model with frozen covariates".

## 5. "If we wanted to win" (for the reflection)

- Gradient boosting (e.g. `HistGradientBoostingClassifier`) on the person-month table: a discrete-time hazard with a non-linear predictor.
- Stack the three members' predictions.
- Calibrate the final model (isotonic, on validation).
- Lessmann et al. (2015): heterogeneous ensembles did well in their benchmark.

## 6. Fairness check (optional)

Metric and calibration by `SEX` and age band. Relevant because credit scoring
is regulated. Low priority.

## 7. Report structure idea

One `report/0X-HazardModels.ipynb`:
1. Why a hazard model (literature, 3–4 sentences).
2. Person-month table: event, risk set, lags (with a diagram).
3. Kaplan–Meier by cluster or limit band (descriptive).
4. Models: logit / cloglog / penalised / one insight model / Cox comparison.
5. Sensitivity table.
6. Predictions for the shared test set → `data/output/predictions-alex.csv`.
7. Simulation.
8. Limitations.

Metric experiments go in the shared metric notebook.
