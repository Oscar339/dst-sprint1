# Hazard models

Firstly, we want to consider all types of hazard model that could be useful for our project.
This is our first part of research and covers a broad spectrum, I have worked with and built hazard models before but with different data and different goals in mind so further research is needed.

A table of different types of hazard models that will be considered for our project:

| Model | Time | Fit to our data | Effort |
|---|---|---|---|
| Discrete-time logit (pooled) | Discrete | Very good | Low |
| Discrete-time cloglog | Discrete | Very good | Low | 
| Penalised discrete-time | Discrete | Very good | Low | 
| Discrete-time with frailty (random effects) | Discrete | Good | Medium |
| Mixture cure | Either | Good | Medium–high |
| Cox PH (static) | Continuous | Poor | Low |
| Cox with time-varying covariates | Continuous | Fair | Medium |
| Parametric AFT (Weibull etc.) | Continuous | Poor | Low |
| Piecewise exponential (Poisson GLM) | Either | Good | Low |
| Recurrent events (AG, PWP) | Continuous | Fair | Medium |
| Varying-coefficient hazard | Discrete | Fair | Medium |
| Random survival forest | Continuous | Fair | Medium |

---

## 1. Discrete-time hazard, logit link (pooled logit)

$\mathrm{logit}\, h(t \mid x) = \alpha_t + \beta^\top x_{t-1}$, fitted by
ordinary logistic regression on a person-month table (Allison, 1982; Singer &
Willett, 1993; Tutz & Schmid, 2016). Shumway (2001) used this for bankruptcy;
Bellotti & Crook (2013) for credit cards.

**Pros**
- Our data is monthly, so time is discrete.
- Standard general linear model (GLM) with a logit link is an industry standard for credit scoring - a discrete time hazard model with logit link build off of this.
- Time-varying covariates are easy, one row per month.
- Directly comparable with Oscar's GLM. The difference between the two is the data structure (person-months vs one row per customer), not the method. Good for the report.
- Shumway (2001) argues single-period models give biased, inconsistent probabilities and this does not. Would be useful to compare this to Oscar's GLM and see which model performs better.
- We can work out expectation time of default as well as probability of default.

**Cons**
- Coefficients are log odds ratios of the monthly hazard, not hazard ratios.
- Rows from the same customer are not independent. Standard errors are too small unless clustered by customer - this needs to be looked into further for the data we're looking at.
- With 4–7 months, a lot of the "time" signal is just the month dummies, as seen in sprint 0 (Sprint 0: person-month AUC 0.71 mostly reflects which month).

## 2. Discrete-time hazard, complementary log-log link

$\log(-\log(1 - h(t \mid x))) = \alpha_t + \beta^\top x$. This is the exact
discrete (grouped-time) version of the Cox model (Prentice & Gloeckler, 1978).

**Pros**
- Often more consistent than logit link as complementary log-log link reverses the relationship between a continuous-time hazard rate and the probability of an event occurring within a discrete interval
- Correct if defaults really happen in continuous time but we only see them monthly (true for us).
- Costs one line of code on top of model 1. Comparing the links should be easy.

**Cons**
- Asymmetric link. When monthly hazards are small (ours are 3–11%), it gives almost the same answers as logit, so the comparison may be dull.

## 3. Penalised discrete-time hazard (L1 / L2)

The same discrete-time hazard but with a penalty, tuned on validation (Tutz &
Schmid, 2016, ch. 7).

**Pros**
- Variable selection among many correlated monthly features. Bill amounts are highly collinear (Sprint 0 VIF finding).
- Matches Oscar's C / L1-vs-L2 tuning, so the comparison is like-for-like.

**Cons**
- Including a penalty shrinks probabilities and can have a negative effect on calibration of the model.

## 4. Frailty / random-effects discrete-time hazard

Adds a customer-level random intercept:
$\mathrm{logit}\, h_{it} = \alpha_t + \beta^\top x_{it} + u_i$,
$u_i \sim N(0, \sigma^2)$. Unobserved differences between customers ("some
people are just riskier") - hidden variables. Discrete-time frailty model: Tutz & Schmid (2016, Chapter 9) - models the hazard over discrete time intervals, accounting for unobserved differences in individuals’ underlying risk through a frailty term. Latent-class frailty model: Muthén & Masyn (2005) - accounts for unobserved heterogeneity by assigning individuals to latent subgroups, which may have different hazard patterns and, depending on the specification, additional within-class frailty.

**Pros**
- Fixes the non-independence of rows from the same customer (con of model 1).
- Identifiable with recurrent events (customers who fall behind, catch up, fall behind again), which our data has.
- $\sigma^2$ is interpretable: how much risk the covariates miss.
- Connects to clustering. A discrete latent class instead of a normal $u_i$ is a mixture model (`03-clustering.md`).

**Cons**
- Predicting for a *new* customer has to integrate over $u_i$ (or use the customer's own past months). Needs care so the October prediction is fair.
- Slower to fit. `statsmodels` `BinomialBayesMixedGLM` is approximate (variational Bayes). R `lme4::glmer` is the standard.
- With only up to 7 rows per customer, $\sigma^2$ may be poorly estimated.

## 5. Mixture cure model

Assumes a fraction of customers will never default ("cured"). The model
has two parts:
- an *incidence* part (logistic): is this customer susceptible?
- a *latency* part (hazard): if susceptible, when do they default?

Credit application: Tong, Mues & Thomas (2012), on UK personal loans. They
found discrimination similar to Cox and logistic regression, but survival
models were better calibrated at intermediate horizons. The cure model also
separates "if" from "when". Dirick et al. (2017) benchmarked it on 10 credit
datasets, and it was among the better methods.

**Pros**
- Matches our data: 78% do not default in October, and most never fall behind at all.
- Separates *who* is at risk from *when*. That is a real insight and directly maps onto the "if and when" literature (Banasik et al., 1999).
- Innovation marks: not covered in the course.
- Writing the EM algorithm ourselves counts as "additional implementation".

**Cons**
- Identifiability needs long follow-up. With 7 months, "cured" and "not yet defaulted" are hard to tell apart. This is the main risk and must be stated.
- Python support is thin. `lifelines` has `MixtureCureFitter` but, as far as I can tell, for univariate (no-covariate) models only (to check). Covariate version: R `smcure` (Cai et al., 2012), or our own EM.
- More parameters, so it can overfit.

## 6. Cox proportional hazards (static covariates)

$h(t \mid x) = h_0(t) \exp(\beta^\top x)$, by partial likelihood (Cox, 1972).
The credit-scoring standard (Stepanova & Thomas, 2002; Bellotti & Crook, 2009).

**Pros**
- Most cited, so a useful benchmark against the literature.
- No assumption on the shape of the baseline hazard.
- Proportional hazards can be checked with Schoenfeld residuals (Grambsch & Therneau, 1994).

**Cons**
- Continuous-time model on monthly data, so there are massive ties (thousands of events in the same month). Efron's approximation helps, but the discrete model is more honest.
- Static covariates throw away the monthly data.
- Gives a survival curve, not P(October default) directly. Needs conversion.

## 7. Cox with time-varying covariates (counting-process form)

Each customer is split into (start, stop] intervals with that month's
covariates. `lifelines` `CoxTimeVaryingFitter`.

**Pros**
- Uses monthly status, utilisation and payments as they change, like Bellotti & Crook (2013).
- Continuous-time counterpart to model 1. Same data structure, so a fair comparison.

**Cons**
- Same ties problem as 6.
- **Prediction is hard**: future covariates are unknown. `lifelines` does not predict from the time-varying fitter. We have to have fixed covariates or use landmarking (`05-other-ideas.md`).

## 8. Parametric / accelerated failure time (Weibull, log-logistic, log-normal)

Assumes a distribution for time to default. AFT: covariates stretch or shrink
time. Benchmarked in Dirick et al. (2017).

**Pros**
- Smooth hazard, extrapolates beyond the data.
- Simple to fit in `lifelines` (`WeibullAFTFitter` etc.).

**Cons**
- With only 4–7 distinct times, there is almost nothing to identify a distribution's shape. The assumption is untestable here.
- Dirick et al. (2017) found AFT models did not beat Cox-type models.
- Probably one sentence in the report: "considered, rejected because...".

## 9. Piecewise exponential model (Poisson GLM)

Constant hazard within each month. Fitted as a Poisson GLM on the person-month
table with log(exposure) as an offset (Holford, 1980; Laird & Olivier, 1981).
Equivalent to Cox when the pieces are fine enough.

**Pros**
- Shows the three models (logit, cloglog, Poisson) are the same idea on the same table. A strong "mathematics of the method" section for the reflection.
- Easy in `statsmodels`.

**Cons**
- Mostly duplicates model 2 numerically for our data.

## 10. Recurrent-event models (Andersen–Gill, PWP)

Models every time a customer falls behind, not just the first.
- **Andersen–Gill** (Andersen & Gill, 1982): one common hazard, counting process.
- **PWP** (Prentice, Williams & Peterson, 1981): stratified by event number, so the 2nd delinquency can behave differently from the 1st.
- Credit tutorial with code: Botha, Verster & Scheepers (2025).

**Pros**
- Keeps the 17% of customers dropped in Sprint 0.
- Uses "fall behind, catch up, fall behind" behaviour, which is real in our data.

**Cons**
- Few events per customer in 7 months.
- Overlaps strongly with Louis's Markov chain (both model moves in and out of arrears). Agree with Louis first.

## 11. Varying-coefficient hazard

Lets $\beta$ change over time: $\beta(t)$. Djeundje & Crook (2019) found this
improved prediction on a large credit-card sample.

**Pros**
- A direct test of proportional hazards.
- Recent credit-specific paper to cite.

**Cons**
- With 4–7 months, $\beta(t)$ is just a covariate × month interaction. Cheap to test, but don't over-sell it.

## 12. Random survival forest

A random forest for censored data (Ishwaran et al., 2008).

**Pros**
- Non-linear, no proportional hazards assumption, handles interactions.
- `scikit-survival` `RandomSurvivalForest`.

**Cons**
- Static covariates only in standard implementations.
- Overlaps Louis's trees. Either he does it, or leave it out.
- Slow on 18,000 customers (manageable).

---
