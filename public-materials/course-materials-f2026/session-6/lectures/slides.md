% Session 6 - Trees and ensembles
% Foundations of Machine Learning (Demo), E1234
% 13 October 2026

# A different kind of family

Everything so far has been a weighted sum of features. A tree instead **partitions** the
feature space into boxes and predicts a constant in each. That buys interactions and
non-linearity for free, and costs the smoothness we had been relying on.

# Growing a tree (CART)

Greedy, recursive, and simple:

1. For every feature and every candidate split point, compute the impurity of the two
   children.
2. Take the split with the largest impurity reduction.
3. Recurse on each child until a stopping rule fires.
4. Predict the child's mean (regression) or majority class (classification).

Impurity measures:

```
Regression:      variance / MSE within a node
Classification:  Gini  = 1 - sum_k p_k^2
                 Entropy = -sum_k p_k log p_k
```

Gini and entropy almost always choose the same splits. Do not spend time on the choice.

# What trees give you

- No scaling needed - splits are order-based, so monotone transforms are irrelevant.
- Interactions come for free: a split under a split *is* an interaction.
- Mixed feature types and missing values handled with modest effort.
- A shallow tree is genuinely readable, which matters in a policy setting.

# What trees cost you

- **High variance.** Shift a few rows and the top split can change, and everything below it
  with it.
- **Axis-aligned boxes** approximate a diagonal boundary as a staircase.
- **Unstoppable overfitting** if grown to purity: one leaf per observation, zero training
  error, no generalisation.

Control it with `max_depth`, `min_samples_leaf`, or cost-complexity pruning (`ccp_alpha`) -
which is exactly session 5's penalty idea applied to tree size.

# Bagging

Variance is the problem, so average it away:

1. Draw `B` bootstrap samples from the training set.
2. Fit a deep tree on each.
3. Average the predictions (or vote).

Averaging `B` roughly-independent estimators cuts variance by about `1/B` and leaves bias
where it was. Rows not drawn into a given bootstrap sample - about 37% - form the
**out-of-bag** set, giving a free validation estimate with no separate split.

# Random forests

Bagging plus one extra decorrelation step: at each split, consider only a random subset of
`m` features (`m ≈ sqrt(p)` for classification, `p/3` for regression).

Why it helps: bagged trees all latch onto the same dominant feature and so are correlated,
which limits the variance reduction. Withholding features at random forces diversity, and
diversity is what averaging is paid for.

Tuning, in order of impact: `n_estimators` (more is never worse, only slower),
`max_features`, `min_samples_leaf`. Forests are famously hard to make bad.

# Boosting

Not parallel averaging but sequential correction:

1. Fit a weak learner - a stump or a depth-3 tree.
2. Fit the next learner on the current residuals (gradient boosting) or on re-weighted
   errors (AdaBoost).
3. Add it in, scaled by a learning rate.

Gradient boosting is gradient descent from session 3 performed in *function* space. It is
typically the strongest tabular model available - and, unlike a forest, it will overfit if
you let it. The learning rate and the number of trees trade off directly: halve the rate,
roughly double the trees, use early stopping on a validation set.

`XGBoost`, `LightGBM` and `HistGradientBoosting` are engineering refinements of this idea,
not different ideas.

# Interpreting an ensemble

- **Impurity-based importance** is biased towards high-cardinality features. Treat with
  suspicion.
- **Permutation importance** - shuffle one column, measure the score drop - is slower and
  much more trustworthy.
- **Partial dependence / SHAP** show the shape of a feature's effect. They describe the
  *model*, not the world; a feature can be important because it proxies something you are
  not allowed to use.

# Today's demo, handout and lab

- `demo.py`: a depth-2 regression tree grown by hand over an exhaustive split search on 12
  rows - every candidate split and its impurity reduction printed, so the greedy choice is
  visible rather than asserted.
- `handouts/tree-splitting-worksheet.md`: eight rows, two features, compute the Gini
  reduction for three candidate splits and pick the root by hand.
- `labs/06_trees-and-ensembles`: a single tree, then bagging, then a random forest on the
  same split - test score and OOB score side by side, plus permutation importances.

# Assignment 1 is due today, 23:59

Push to `main` in your `assignment-1-f2026-<handle>` repository. There is a silent 48-hour
grace window at grading; there is no extension beyond it without documentation.
