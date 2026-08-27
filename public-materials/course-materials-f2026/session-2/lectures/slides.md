% Session 2 - Linear regression: least squares from geometry to code
% Foundations of Machine Learning (Demo), E1234
% 15 September 2026

# Why start here

Linear regression is the smallest interesting model: a family, a loss, and a closed-form
minimiser. Every later method changes exactly one of those three things, so if the
mechanics here are solid, the rest of the course is variation rather than novelty.

# The model

For observation `i` with features `x_i` in `R^p`:

```
y_i = b_0 + b_1 x_i1 + ... + b_p x_ip + e_i
```

In matrix form, with a column of ones absorbed into `X`:

```
y = X b + e
```

"Linear" means linear **in the parameters**. `y = b_0 + b_1 x + b_2 x^2` is still a linear
model - the design matrix just has a squared column. This is more freedom than students
expect.

# The loss

Choose `b` to minimise the residual sum of squares:

```
RSS(b) = sum_i (y_i - x_i' b)^2 = ||y - X b||^2
```

Why squares? Three honest reasons: it is differentiable everywhere, it has a closed-form
minimiser, and it is the maximum-likelihood choice under Gaussian noise. One dishonest
reason: tradition. Squares also punish outliers quadratically - a real cost, and the reason
absolute-error regression exists.

# The normal equations

Set the gradient to zero:

```
d/db ||y - X b||^2 = -2 X'(y - X b) = 0
  =>  X'X b = X'y
  =>  b_hat = (X'X)^-1 X'y      (when X'X is invertible)
```

Two lines of algebra, and the whole model is solved. Note the failure condition: `X'X` is
singular exactly when a column of `X` is a linear combination of the others - perfect
collinearity. A dummy for every category *plus* an intercept is the usual culprit.

# The geometry

`X b` ranges over the column space of `X` - a `p`-dimensional plane inside `R^n`. Least
squares projects `y` orthogonally onto that plane.

- Fitted values `y_hat = H y`, where `H = X(X'X)^-1 X'` is the hat matrix, a projection.
- Residuals `y - y_hat` are orthogonal to every column of `X`. So residuals are, by
  construction, uncorrelated with your features - which is why a residual plot showing
  structure means the *functional form* is wrong, not the fit.

# Reading the fit

- **R-squared** `= 1 - RSS/TSS`: share of variance explained. Never decreases when you add
  a column, so it cannot be used to choose between models of different size.
- **RMSE**: `sqrt(RSS/n)`, in the units of `y`. Report this to a non-technical audience.
- **Coefficient**: predicted change in `y` per unit of `x_j`, *holding the other columns
  fixed*. That clause is where causal over-claims live.

# In practice, do not invert

`(X'X)^-1` is numerically poor. Solve the linear system instead - `numpy.linalg.solve`, or
better, a QR or SVD least-squares routine (`numpy.linalg.lstsq`, R's `lm`, which uses QR).
Same answer in exact arithmetic, far better conditioned in floating point.

We nonetheless implement the naive version in Assignment 1, because you should feel the
algebra once before delegating it.

# Diagnostics worth two minutes

- Residuals vs fitted: curvature means a missing transform; a fan means heteroscedasticity.
- Residuals vs each feature: structure means that feature enters non-linearly.
- Leverage: a single far-out `x` can move the whole plane. Look before you trust.

# Today's demo and lab

- `demo.R`: `lm()` on the 12-row housing extract, then the same coefficients recovered by
  hand with `solve(t(X) %*% X, t(X) %*% y)` - the closed form and the library agreeing to
  ten decimal places.
- `labs/02_linear-regression` (`.Rmd`): fit, read the summary, plot the residuals, and
  break the fit deliberately with a collinear column so you can recognise the error
  message later.

# Coming next

Session 3 removes the closed form. Once the loss has no analytic minimiser - and from
logistic regression onward it does not - we have to descend the loss surface instead.
That single change is what makes everything else in the course possible.
