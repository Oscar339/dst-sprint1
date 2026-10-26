# 01 Plan

## 1. The question

As a group, we are predicting whether a credit card customer defaults in October 2005, using the Taiwan data
(Yeh, 2009). Each of us builds one model and all of them are scored on the same test customers. Oscar is doing
logistic regression, Louis is doing a Markov chain and decision trees, and my part is hazard modelling and clustering.

A hazard model asks, each month, "given this customer has made it this far, what is the chance of the event now?"
(Allison, 1982). This suits our data well, as the data is monthly - six months of repayment status, bills and
payments from April to September, followed by the October default label.

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| Main model | Discrete-time hazard with a logit link, which is a logistic regression on a person-month table | Logistic regression is the industry standard in credit scoring, see research note 01 |
| Other models | Cloglog link, L1 / L2 penalty, Cox with time-varying covariates | Each one changes one thing from the main model, so we can see what effect the change has |
| Not built | Frailty, mixture cure, AFT, recurrent-event Cox | From research note 01, the mixture cure model has no Python package that takes covariates, and with only 5 time points we cannot check the shape an AFT model assumes |
| Event | 2+ months behind for June to September, and the real default label for October | The data has no default date, see research note 02 |
| Risk set | Everyone, every month, with last month's status as a covariate | Keeps the riskiest customers, so every test customer gets a prediction (in Sprint 0 we dropped them) |
| Covariates | Lagged by one month: status (as categories), utilisation and payment ratio, plus log limit, age, sex, education and marriage | Research note 02. The status codes -2, -1 and 0 are categories, not months of delay (`data/README.md`) |
| Comparing models | AIC, BIC and 10-fold cross-validation with whole customers left out (10-fold is an estimated starting point, which we can change later if needed) | The comparison has to account for the number of parameters or use held-out data. A customer's rows are not independent, so the whole customer is left out for validation |
| Choosing the penalty | Cross-validation on the training customers | One held-out set gives a noisy choice, averaging over 10 folds is a good starting point |
| Metrics | AUC with ROC and precision-recall curves, Brier score, log loss and calibration | With 22% default, accuracy could be misleading. AUC and the curves measure ranking, while the Brier score and log loss reward honest probabilities |
| Uncertainty | Bootstrap over customers, and the test set only used once | Resampling needs independent units, which are customers rather than customer-months. Testing on data that was used to choose the model gives results that are too optimistic |
| Clustering | k-means on April-September behaviour, trying a range of K, compared with a Gaussian mixture chosen by BIC | The clusters are a tool rather than true groups, so the number of clusters is something to vary, see research note 03 |

## 3. Testing generalisation

The brief asks when the performance metric tells us something about how the model would do in the real world.
"Generalise" can mean different things, and each one needs a different split of the data, so we test each one
separately:

| Generalise to | How we test it | Where |
|---|---|---|
| New customers similar to the training ones | Grouped cross-validation and the shared validation set | `03`, sections 2 and 5 |
| The next month | Fit on the months up to August and predict September | `03`, section 8 |
| A type of customer the model has never seen | Leave one cluster out | `04`, section 7 |
| A new time period | Not tested, as we only have one six-month window | Limitation |

An overall AUC can also hide poor performance within groups of customers, so we also report the metric within each
cluster (`04`, section 6).

## 4. Ground rules

* We use the same shared split as Oscar and Louis (`report/01-Data.ipynb`, seed 2026).
* Everything is chosen on the training customers using cross-validation. The validation set is then used as a
  single check, and the test set is only used once, for the final model, to write `data/output/predictions-alex.csv`.

Next, `02-data.ipynb` builds the person-month table.
