# Is clustering relevant?


In 2012, Bijak & Thomas tested segmentation, the idea of splitting customers into groups and fitting a scorecard per group, on two UK banks and a credit bureau. Their finding: segmented models were not considerably better than a single scorecard on any of the real datasets. So expect little predictive gain, and
cite this when we report it. I have also spent time looking at various subpopulations of data and have had similar findings (from work I currently do).

The value is elsewhere: clusters give us natural "types of data" to leave out,
which is what the brief asks for.

## Two uses

### 1. Leave-one-cluster-out generalisation test

> "if you believe that you can predict new types of data, you could
> demonstrate this by leaving out some types of data and observing your
> performance" — Assessment 1 brief

Procedure:
1. Cluster training customers on their April–September data.
2. For each cluster, refit every member's model without it, then score on the test customers in that cluster.
3. Compare with the in-distribution score.

**Pros**
- Directly what the brief asks for.
- Applies to all three of our models.
- Shows whether the "best" model is best everywhere or only on the common customer types.

**Cons**
- Needs every model refitted per cluster. Each member must make their fitting code callable as a function.
- Small clusters give noisy metrics. Use bootstrap confidence intervals.

### 2. Metrics within each cluster

Report the agreed metric separately per cluster, without refitting.

**Pros**
- A lot cheaper to do then 1.
- An overall AUC can hide a model that is good on easy customers and poor on the risky ones. AUC within a cluster removes the "easy separation between clusters" part.

**Cons**
- AUC within a cluster with very few defaults could be unstable, look into this.

## What to cluster on

| Input | Method | Pros | Cons |
|---|---|---|---|
| Summary features (mean utilisation, months behind, payment ratio, trend) | k-means (scaled) | Simple, fast, `sklearn` | Choice of features drives the result |
| Mixed numeric + categorical (status codes as categories) | k-prototypes (Huang, 1998) | Handles −2/−1/0 status codes properly | Needs `kmodes` package; extra weight parameter λ |
| The 6-month status sequence itself | Sequence distance (optimal matching) + hierarchical clustering (Gabadinho et al., 2011, TraMineR) | Clusters whole trajectories, e.g. "drifting behind" | R package; distance matrix is 18,000² (subsample) |
| Probabilistic | Gaussian mixture / latent class | Soft membership, likelihood for choosing k | More assumptions |

**Choosing k:** silhouette score (Rousseeuw, 1987) plus whether the clusters
make sense to a person. Expect around 4–6 recognisable groups (e.g. pays in
full / revolves / drifts behind / chronically late / inactive). Note that
`PAY_*` = −2 is thought to mean "no consumption", i.e. inactive.

## Leakage rules

- Fit the scaler and clusterer on **train only**; assign validation and test customers to the nearest cluster.
- **Do not use the October label** in clustering. Use only Apr–Sep behaviour and static covariates.
