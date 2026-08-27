% Session 3 - Gradient descent: loss surfaces and learning rates
% Foundations of Machine Learning (Demo), E1234
% 22 September 2026

# The problem with closed forms

`b_hat = (X'X)^-1 X'y` exists only because squared error is quadratic in `b`. Change the
loss - to log-loss, hinge loss, anything a neural network uses - and no such formula
exists. What survives is much more general: **follow the slope downhill.**

# Gradient descent in one line

```
b <- b - alpha * grad L(b)
```

- `grad L(b)`: the direction of steepest *increase* of the loss at `b`.
- The minus sign: go the other way.
- `alpha`: the learning rate - how far to step.

Repeat until the loss stops improving. That is the entire algorithm; the rest of this
session is about `alpha` and about what `grad L` costs to compute.

# The gradient for squared error

With `L(b) = (1/n) ||y - X b||^2`:

```
grad L(b) = -(2/n) X' (y - X b)
```

Read it as a mechanism: the residual vector `(y - X b)` is projected back onto each
feature. A feature that correlates with the current error gets its coefficient moved; one
that does not, does not. When no feature correlates with the residual, the gradient is zero
- which is exactly the normal equations from session 2, reached iteratively.

# Choosing the learning rate

- Too small: correct but slow; a thousand iterations to move nowhere.
- Too large: overshoot, oscillate, diverge to `nan`.
- Just right: loss falls steeply, then flattens.

Practical rule: try `alpha` in `{0.3, 0.1, 0.03, 0.01, ...}`, **plot the loss per
iteration**, and keep the largest value that still decreases monotonically. Never tune a
learning rate without looking at that curve.

# Feature scaling is not optional here

The loss surface for unscaled features is a long thin valley - one coefficient measured in
euros, another in rooms. Descent then zig-zags across the valley instead of down it, and no
single `alpha` suits both directions.

Standardise (`(x - mean) / sd`) before descending. Least squares by normal equations is
invariant to this; gradient descent is not. Remember from session 1: compute the mean and
sd on the **training** split only.

# Batch, stochastic, mini-batch

| Variant | Gradient from | Per-step cost | Path |
|---|---|---|---|
| Batch | all `n` rows | `O(np)` | smooth, deterministic |
| Stochastic (SGD) | 1 row | `O(p)` | noisy, escapes flat regions |
| Mini-batch | 32-256 rows | in between | the practical default |

The noise in SGD is not purely a defect: it is why stochastic methods work on the
non-convex surfaces of session 8.

# Convexity, briefly

Squared-error linear regression has a convex loss: one basin, so any descent that converges
converges to *the* minimum. Neural networks do not. In the non-convex case, gradient
descent finds *a* good-enough basin, and the initialisation and the noise decide which -
which is why you fix a seed and report it.

# Stopping

Stop on whichever comes first:

- the relative improvement in loss falls below a tolerance (e.g. `1e-8`);
- a maximum iteration count is hit (always have one);
- **validation** loss starts to rise - early stopping, and already a form of
  regularisation, as session 5 will make precise.

# Beyond plain descent (named, not derived)

- **Momentum:** average recent gradients; damps the zig-zag.
- **AdaGrad / RMSProp:** per-coordinate step sizes.
- **Adam:** momentum plus per-coordinate scaling; the default you will meet everywhere.

We use plain descent in the lab so the mechanism stays visible. Read the Ruder overview in
this week's readings for the family tree.

# Today's demo, handout and lab

- `demo.py`: 200 lines of nothing but the update rule - descend a one-feature regression,
  print the loss every 50 iterations, then re-run with `alpha` ten times too large and watch
  it diverge.
- `handouts/gradient-descent-worksheet.md` (and its PDF): derive the gradient for a
  two-parameter model by hand and take three steps with a pen. Fifteen minutes; do it.
- `labs/03_gradient-descent`: implement `fit_gd(X, y, alpha, n_iter)`, plot the loss curve,
  and confirm it converges to the closed-form solution from session 2.

# Assignment 1 is released today

Linear regression from scratch: the normal equations, `predict`, and R-squared, in plain
Python with no scientific stack. Due 13 October, 23:59. Start in week 1 of the three - the
debugging, not the algebra, is what takes the time.
