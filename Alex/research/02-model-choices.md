# What we can vary inside a hazard model

In this section, we will go through what we can vary in terms of the data we've got. We will look at the pros and cons of handling the data in different ways.

## 1. Event definition

The data has no default date. Months 1–6 need a proxy built from `PAY_*`.

| Option | Pros | Cons |
|---|---|---|
| 1+ month behind | Many events, stable estimates | Mostly people who are just late, not defaulters |
| 2+ months behind (Sprint 0 choice) | Middle ground; matches common "early delinquency" usage | Still a proxy |
| 3+ months behind (90 days) | Matches the most common definition of default (90 dpd) | Few events before October |

## 2. Risk set

| Option | Pros | Cons |
|---|---|---|
| First event only, drop people already behind at the start | Clean single-event survival model | Drops the riskiest customers (we did this in Sprint 0) |
| First event only, keep them with a "prior delinquency" covariate | Scores everyone | Their "time" axis means something different |
| Recurrent: everyone at risk every month, previous status as a covariate | Scores everyone anduses all transitions | Similar to a Markov chain |

## 3. Time origin and landmark

From the data we have, there is no account age. Therefore, time means calendar month, not account age (left
truncation).

| Option | Pros | Cons |
|---|---|---|
| t = 1 is April | Uses all data | Time 1 is arbitrary, not account opening |
| Landmark at month k (e.g. June), predict ahead | Matches an "early warning" use which is a fair comparison with Oscar's model | Less data |
| Several landmarks stacked (van Houwelingen, 2007) | Uses more data (historical), one model for any prediction date | More complex |

## 4. Covariates

| Choice | Options | Notes |
|---|---|---|
| Timing | Lag 1, lag 2, frozen at landmark | Make sure to lag behind by at least 1 month, as a lag of 0 would mean that we are using the months own data to predict itself |
| Status coding | −2 / −1 / 0 as categories; 1, 2, 3+ months behind | `data/README.md`: −2/−1/0 are status codes, not amounts of delay. SO, this approach seems good |
| Utilisation | bill / limit | From Sprint 0, this is one of the strongest indicators after month |
| Payment ratio | paid / previous bill | Captures pays in full vs minimum |
| Trend | Slope of utilisation or status over the last 3 months | Malik & Thomas (2012) needed a 2nd-order Markov chain, i.e. history beyond the last month matters |
| Static | log limit, age, sex, education, marriage | From Sprint 0, we learnt that age added nothing. Look at Oscar's trend analysis and information values |
| Cluster label | From `03-clustering.md` | We could create additional covariates by additional modelling, mentioned in lectures. Further research needed to see if this is a valid approach |

## 5. Baseline hazard $\alpha_t$

| Option | Pros | Cons |
|---|---|---|
| Month dummies | Fully flexible | One parameter per month; can't extrapolate |
| Linear / quadratic in t | Fewer parameters | Assumes a shape |
| Constant | Simplest | Ignores the clear month-to-month differences seen in Sprint 0 |
| Spline | Smooth | Pointless with ~4 time points |

## 6. Time-varying effects (proportional hazards check)

Covariate × month interactions. If they matter, proportional hazards fails
and a single $\beta$ is misleading (Djeundje & Crook, 2019).

## 7. Link function

Logit vs cloglog vs probit. See `01-hazard-models.md` §1–2. Expect small
differences because monthly hazards are small. Saying *why* is the insight.

## 8. Estimation choices

| Choice | Options | Notes |
|---|---|---|
| Penalty | none / L1 / L2, C tuned on validation | Same as Oscar, so comparable |
| Class weight | ×1 vs ×10 (Botha & Muller default) | Sprint 0: ×10 made the term-structure MAE go from 0.16% to 15%. Weights distort probabilities |
| Standard errors | Naive vs clustered by customer | Rows from the same person are correlated |
| Calibration after fitting | none / Platt / isotonic | Sprint 0 notebook 03 method; isotonic fixed probabilities without changing ranking |

## 9. Converting to P(October default)

$P(T \le 7 \mid T > k, x) = 1 - \prod_{t=k+1}^{7} (1 - \hat h(t \mid x))$
needs covariates for future months. The options:
- freeze covariates at month k (fair, but loses information);
- only predict one month ahead (Sep → Oct; uses the most data, but makes the hazard model nearly a one-period logit);
- simulate future covariates (e.g. with Louis's Markov chain). That's a nice joint piece.

**State which one we use. It drives the result** (Sprint 0's hazard model lost
to logistic regression partly because of freezing at June).
