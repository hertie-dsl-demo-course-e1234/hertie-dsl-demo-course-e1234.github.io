% Session 5 - Regularisation and model selection
% Foundations of Machine Learning (Demo), E1234
% 6 October 2026

# The question

You can fit a straight line, a cubic, or a degree-15 polynomial. All three minimise
training error to different degrees, and the most flexible always wins on the training set.
So how do you choose? Not by fitting better - by estimating out-of-sample error honestly.

# Bias-variance, made concrete

Expected squared error at a point decomposes as:

```
E[(y - f_hat(x))^2] = bias(f_hat)^2 + var(f_hat) + sigma^2
```

- **Bias:** error from a family too rigid to contain the truth. Falls as flexibility rises.
- **Variance:** how much `f_hat` would change with a different sample. Rises as flexibility
  rises.
- **`sigma^2`:** irreducible noise. No model beats this, and claiming to means leakage.

The sum is U-shaped in flexibility. Model selection is finding the bottom of that U without
peeking at the test set.

# Regularisation: pay for complexity

Add a penalty on the size of the coefficients:

```
Ridge (L2):  min ||y - Xb||^2 + lambda * sum_j b_j^2
Lasso (L1):  min ||y - Xb||^2 + lambda * sum_j |b_j|
```

- `lambda = 0`: ordinary least squares. `lambda -> inf`: all coefficients to zero.
- **Ridge** shrinks coefficients smoothly, keeps all features, and rescues collinear designs
  - `X'X + lambda I` is invertible even when `X'X` is not.
- **Lasso** sets some coefficients exactly to zero: it selects. The corner of the L1 ball is
  what makes exact zeros possible.
- **Elastic net** interpolates, and is the safe default with correlated features.

Never penalise the intercept, and always standardise first - a penalty on raw coefficients
otherwise means "penalise whichever feature happens to be measured in small units".

# Validation

- **Hold-out:** one split. Fast, and noisy on small `n`.
- **k-fold cross-validation:** split into `k` folds, train on `k-1`, validate on the
  remaining one, rotate, average. `k = 5` or `10`. The standard tool.
- **Stratified k-fold:** preserve class proportions - mandatory for imbalanced
  classification.
- **Grouped / time-series splits:** when rows are not exchangeable. Repeated measurements
  on the same municipality must not straddle a fold; future must not predict past.

# The selection procedure

1. Set the test set aside. Do not look at it.
2. For each candidate `lambda`, run k-fold CV on the training set; average the fold scores.
3. Pick the `lambda` with the best mean CV score - or, better, the **one-standard-error
   rule**: the simplest model within one standard error of the best. It costs almost nothing
   in accuracy and buys real stability.
4. Refit on the whole training set at that `lambda`.
5. Report the test score once.

# Nested CV, and why people get this wrong

If you use CV both to tune and to report, the reported number is optimistic - the folds have
seen the tuning decisions. When you need an unbiased performance estimate *and* tuning, use
nested CV: an inner loop to select, an outer loop to evaluate. Expensive, and correct.

# Learning curves read the diagnosis

Plot train and validation error against training-set size:

- both high, converged together -> **underfitting**. More data will not help; a richer model
  or better features will.
- large persistent gap -> **overfitting**. More data or more regularisation will help.

Two lines on one plot answers the question "should I collect more data?" - which is usually
a budget question.

# Information criteria, in passing

AIC and BIC penalise fitted likelihood by parameter count and require no resampling. Useful
for nested parametric models, awkward for the rest. Cross-validation is more general and
assumes less; prefer it unless `n` is tiny.

# Today's demo and lab

- `demo.py`: polynomial degrees 1, 3 and 15 fitted to 20 noisy points, printing train and
  validation RMSE side by side - the U-shape in three lines of output. Then a ridge path
  over the degree-15 family alone, which recovers a better validation score than any
  unpenalised degree.
- `labs/05_regularisation-and-model-selection`: `RidgeCV` and `LassoCV` over a `lambda`
  grid, the CV curve plotted, and the lasso's zeroed coefficients read off as a feature
  selection you have to justify.

# Assignment 2 is released today

Classification and evaluation, as a notebook: implement the sigmoid, log-loss and the
confusion-matrix metrics, then choose and defend a threshold. Due 3 November, 23:59.
