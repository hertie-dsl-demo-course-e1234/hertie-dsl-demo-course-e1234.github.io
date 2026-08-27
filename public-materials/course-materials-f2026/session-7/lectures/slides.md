% Session 7 - Unsupervised learning: clustering and PCA
% Foundations of Machine Learning (Demo), E1234
% 20 October 2026

# No labels, no test error

Every method so far had a `y` to be wrong about. Today there is none. That changes the
epistemics as much as the algorithms: without a target there is no honest single number for
"correct", so results must be argued for, not merely reported.

Two workhorses: **k-means** (group the rows) and **PCA** (compress the columns).

# k-means: the objective

Choose `k` centroids and an assignment of points to them to minimise within-cluster sum of
squares:

```
min sum_j sum_{i in C_j} || x_i - mu_j ||^2
```

Lloyd's algorithm alternates the two halves of that problem:

1. Assign each point to its nearest centroid.
2. Recompute each centroid as the mean of its members.
3. Repeat until assignments stop changing.

Each step is non-increasing in the objective, so it converges - to a **local** optimum that
depends on the initialisation. Run it multiple times (`n_init`), use `k-means++` seeding, and
fix your seed.

# k-means: what it assumes

- Clusters are spherical and of similar spread - it is minimising Euclidean distance.
- Every point belongs to exactly one cluster, with no notion of "unclustered".
- Scaling matters enormously: a variable in euros dominates one in rooms. **Standardise
  first**, always.

Elongated, nested or density-defined clusters defeat it. Gaussian mixtures (soft, elliptical)
and DBSCAN (density-based, has an explicit noise label) are the usual next stops.

# Choosing k

- **Elbow:** plot within-cluster SS against `k` and look for the kink. Often there isn't one.
- **Silhouette:** per point, `(b - a) / max(a, b)` where `a` is mean distance within its own
  cluster and `b` to the nearest other. Near 1 is good, near 0 is borderline, negative means
  misassigned.
- **The real criterion:** does the partition mean anything to a domain expert, and is it
  stable under resampling? If a different seed or a 90% subsample reshuffles the groups, you
  have found noise.

# PCA: the objective

Find orthogonal directions of maximum variance. Equivalently: the best `q`-dimensional
linear approximation of the data in squared error.

- Centre (and usually scale) the columns.
- Take the eigenvectors of the covariance matrix, or - numerically better - the SVD of the
  centred `X`.
- Component `1` is the highest-variance direction; component `2` the highest-variance
  direction orthogonal to it; and so on.
- Eigenvalues give **variance explained**; a scree plot shows where the returns stop.

# PCA: reading it honestly

- Components are linear combinations of *all* features, so they are not features you can
  collect. "PC1 is roughly size, PC2 roughly affluence" is an interpretation you owe the
  reader evidence for - show the loadings.
- Signs are arbitrary; do not over-read a flipped axis.
- High variance is not high relevance: PCA is blind to `y`, so a discarded low-variance
  direction can be the predictive one. If prediction is the goal, prefer regularisation
  (session 5) over pre-emptive PCA.
- t-SNE and UMAP make prettier two-dimensional pictures but distort global distances -
  excellent for looking, dangerous for measuring. Read the "Misread t-SNE" piece in this
  week's readings before you put one in a report.

# A workable pipeline

1. Standardise the columns.
2. PCA; keep enough components for ~80-90% of variance; inspect the loadings.
3. Cluster in that reduced space (faster, less noise-dominated).
4. Profile the clusters on the *original* variables - that is what makes them communicable.
5. Re-run on a subsample and check the profiles survive.

# Today's demo and lab

- `demo.R`: `kmeans()` on a standardised three-variable municipal extract with `nstart = 25`,
  then `prcomp()` with a scree plot and the first two loadings printed - the same 12 rows we
  have used since session 1.
- `labs/07_unsupervised-learning` (`.Rmd`): silhouette scores across `k = 2..6`, a PCA
  biplot, and a written one-paragraph interpretation of the clusters you find. The paragraph
  is the deliverable, not the plot.

# The group project is released today

End-to-end modelling report, teams of 2-3, due 27 November 23:59. Your team repository comes
from `assignment-3-project-f2026`; the brief and rubric are in its README. Pitch your
question at the 17 November clinic.
