# Hazard modelling: research and plan

In this section, we look at what hazard models could be useful for our project, what choices we have when modelling with our data, and how clustering can help. It ends with what we decided to build, which is carried out in `07-HazardAndClusteringModels.ipynb`. The full working, including the longer research notes and
references, is in `Alex/`.

## Why a hazard model?

A standard credit model asks whether a customer defaults. A hazard model asks, each month, "given this customer has
made it this far, what is the chance of the event now?". This suits our data well, as the data is
monthly. Hazard models are widely used for credit scoring, and I have worked with and built them before in the credit industry, though with different data and different goals in mind.

## Types of hazard model

We first considered a broad range of hazard models:

| Model | Time | Fit to our data | Effort |
|---|---|---|---|
| Discrete-time logit | Discrete | Very good | Low |
| Discrete-time cloglog | Discrete | Very good | Low |
| Penalised discrete-time | Discrete | Very good | Low |
| Discrete-time with frailty | Discrete | Good | Medium |
| Mixture cure | Either | Good | Medium–high |
| Cox PH (static) | Continuous | Poor | Low |
| Cox with time-varying covariates | Continuous | Fair | Medium |
| Parametric AFT | Continuous | Poor | Low |

The discrete-time logit model is fitted as an ordinary logistic regression on a person-month table, with one row per
customer for each month they are at risk:

$$\mathrm{logit}\, h(t \mid x) = \alpha_t + \beta^\top x_{t-1}$$

Logistic regression is the industry standard for credit scoring, and the process is very understandable, which
makes the model auditable and usable for decision making in the financial sector. It is also directly comparable
with Oscar's logistic regression - the difference between the two is the data structure (person-months against one
row per customer), not the method. Shumway (2001) argues that single-period models give biased probabilities and a
hazard model does not, so comparing the two is worthwhile.

The other models we chose each change one thing from the logit model, so we can see what effect each change has:

* Cloglog link - $\log(-\log(1 - h)) = \alpha_t + \beta^\top x$ is the discrete version of the Cox model (Prentice &
  Gloeckler, 1978), and is the correct model if defaults really happen in continuous time but we only see them
  monthly. With small monthly hazards it will likely give similar results to the logit.
* Penalised (L1 / L2) - useful for variable selection, as the monthly bill amounts are highly correlated (seen in
  Sprint 0). A penalty can have a negative effect on calibration, and in finance it has to be used fairly.
* Cox with time-varying covariates - the most cited model in credit scoring (Stepanova & Thomas, 2002; Bellotti &
  Crook, 2009), with no assumption on the shape of the baseline hazard. It uses the monthly status, utilisation and
  payments as they change, on the same rows as the logit model, so it is a fair comparison.

We did not build the others because of personal experience and research. The frailty model adds a hidden "riskiness" 
for each customer, which could be poorly estimated with only a few rows per customer. The mixture cure model (Tong, Mues & Thomas, 2012) 
assumes some customers will never default, which fits our data, as 78% do not default in October - but with only seven months,
"cured" and "not yet defaulted" are hard to tell apart, and there is no Python package that fits it with covariates.
An AFT model assumes a distribution for the time to default, which we cannot check with so few time points.

## Choices in the data

From the data we have, there is no default date and no account age, so some choices have to be made:

| Choice | Options | What we chose and why |
|---|---|---|
| Event | 1+, 2+ or 3+ months behind | 2 or more months behind for June to September, and the default label for October. The codes go up to 8 months late, but 3+ is rare (under 2% of months), while 1+ would count a single late payment. 2+ is a stand-in event, not a default |
| Risk set | Drop customers already behind, or keep everyone | Keep everyone every month, with last month's status as a covariate. This keeps the riskiest customers, so every test customer gets a prediction (in Sprint 0 we dropped them) |
| Time | Calendar month, landmarks | Calendar month, as there is no account age |
| Covariate timing | Lag 1, lag 2, frozen | Lag by one month, as a lag of 0 would mean using a month's own data to predict itself |
| Status coding | Numbers or categories | Categories, as -2, -1 and 0 are status codes, not months of delay |
| Baseline | Month dummies, linear, constant | A separate intercept for each month, which is fully flexible |

## Clustering

In 2012, Bijak & Thomas tested segmentation, the idea of splitting customers into groups and fitting a scorecard per
group, on two UK banks and a credit bureau. Their finding was that segmented models were not considerably better than
a single scorecard on any of the real datasets. I have also looked at various subpopulations of data in my own work
and had similar findings.

However, clustering is important as clusters give us natural "types of data" to leave out, which is what the brief
asks for. We use them in two ways:

1. Leave one cluster out - refit the model without one cluster and score it on that cluster, to see whether the model
   can handle a type of customer it has never seen.
2. Metrics within each cluster - an overall AUC can hide a model that is good on easy customers and poor on the risky
   ones..

We cluster with k-means on a summary of each customer's April-September behaviour. The
clusters are built on the training data only, and the October label is never used to build them.

## What we decided

| Decision | Choice |
|---|---|
| Main model | Discrete-time logit hazard |
| Other models | Cloglog link, L1 / L2 penalty, Cox with time-varying covariates, plus a logistic regression on the October rows only, to test whether the extra months help |
| Comparing models | 10-fold cross-validation, leaving out whole customers, as a customer's rows are not independent (Lawson, 2026, 02.2). AIC and BIC as well |
| Metrics | AUC with ROC and precision-recall curves, Brier score, log loss and calibration. With 22% default, accuracy could be misleading - AUC measures ranking, while the Brier score and log loss reward honest probabilities (Lawson, 2026, 01.2) |
| Uncertainty | Bootstrap over customers, not customer-months |
| Ground rules | The same shared split as Oscar and Louis. Everything is chosen on the training customers, the validation set is used as a single check, and the test set is only used once |

The brief asks when the performance metric tells us something about how the model would do in the real world.
Generalise can mean different things, and each needs a different split of the data, so we test each one separately:

| Generalise to | How we test it |
|---|---|
| New customers similar to the training ones | Cross-validation and the shared validation set |
| The next month | Fit on the months up to August and predict September |
| A type of customer the model has never seen | Leave one cluster out |
| A new time period | Not tested, as we only have one six-month window |

## References

- Bellotti, T. & Crook, J. (2009). Credit scoring with macroeconomic variables using survival analysis. *Journal of
  the Operational Research Society* 60(12), 1699–1707. https://doi.org/10.1057/jors.2008.130
- Bijak, K. & Thomas, L. C. (2012). Does segmentation always improve model performance in credit scoring? *Expert
  Systems with Applications* 39(3), 2433–2442. https://doi.org/10.1016/j.eswa.2011.08.093
- Lawson, D. (2026). *Data Science Toolbox* (MATHM0029), lectures 01.2 Introduction to Classification and
  02.2 Cross Validation. University of Bristol. https://dsbristol.github.io/dst/
- Prentice, R. L. & Gloeckler, L. A. (1978). Regression analysis of grouped survival data with application to breast
  cancer data. *Biometrics* 34(1), 57–67. https://doi.org/10.2307/2529588
- Shumway, T. (2001). Forecasting bankruptcy more accurately: a simple hazard model. *Journal of Business* 74(1),
  101–124. https://doi.org/10.1086/209665
- Stepanova, M. & Thomas, L. (2002). Survival analysis methods for personal loan data. *Operations Research* 50(2),
  277–289. https://doi.org/10.1287/opre.50.2.277.426
- Tong, E. N. C., Mues, C. & Thomas, L. C. (2012). Mixture cure models in credit scoring: if and when borrowers
  default. *European Journal of Operational Research* 218(1), 132–139. https://doi.org/10.1016/j.ejor.2011.10.007
