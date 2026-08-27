"""Session 5 demo - the bias-variance U-shape, and what a penalty buys.

Twenty noisy points from a smooth cubic. Fit polynomials of degree 1, 3 and 15, and print
train and validation RMSE side by side. Then hold the degree-15 family fixed and vary the
ridge penalty: the same flexibility becomes usable again.

Requires numpy. Run: python3 demo.py
"""

import numpy as np

rng = np.random.default_rng(2026)
NOISE_SD = 2.0

# --- data: a smooth truth plus noise ------------------------------------------------
x = rng.uniform(-3, 3, 20)
y = 0.5 * x**3 - 2 * x + rng.normal(0, NOISE_SD, x.size)

# A fixed RANDOM split, decided before fitting. Splitting on sorted x instead would make
# validation an extrapolation problem, and every polynomial would look catastrophic for a
# reason that has nothing to do with overfitting.
idx = rng.permutation(x.size)
train, val = idx[:14], idx[14:]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def design(xv, degree):
    """Vandermonde matrix [1, x, x^2, ..., x^degree] - linear IN THE PARAMETERS."""
    return np.vander(xv, degree + 1, increasing=True)


def standardise(X, stats=None):
    """Put every column but the intercept on a common scale, using TRAINING statistics.

    Not cosmetic: a penalty on raw coefficients would mean "penalise whichever column
    happens to be measured in small units", and x^15 spans seven orders of magnitude.
    """
    if stats is None:
        mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0)
        sd[sd == 0] = 1.0
        stats = (mu, sd)
    mu, sd = stats
    out = X.copy()
    out[:, 1:] = (X[:, 1:] - mu) / sd
    return out, stats


def fit_ridge(degree, lam=0.0):
    """Fit on the training rows, return (train RMSE, val RMSE, max |coef|)."""
    Xtr, stats = standardise(design(x[train], degree))
    Xva, _ = standardise(design(x[val], degree), stats)
    penalty = lam * np.eye(Xtr.shape[1])
    penalty[0, 0] = 0.0  # never penalise the intercept
    coef = np.linalg.solve(Xtr.T @ Xtr + penalty, Xtr.T @ y[train])
    return (rmse(Xtr @ coef, y[train]), rmse(Xva @ coef, y[val]),
            float(np.abs(coef[1:]).max()))


results = {d: fit_ridge(d) for d in (1, 3, 15)}
best_degree = min(results, key=lambda d: results[d][1])

print(f"noise floor: sd = {NOISE_SD} - no model can beat this\n")
print(f"{'degree':>7} {'train RMSE':>12} {'val RMSE':>10}   diagnosis")
for degree, (tr, va, _) in results.items():
    if degree == best_degree:
        note = "best out of sample - about right"
    elif tr > 1.5 * NOISE_SD or (tr > 0.9 * NOISE_SD and va > results[best_degree][1]):
        note = "underfit - too rigid to hold a cubic; both errors high"
    else:
        note = "overfit - near-zero train error, and the gap IS the diagnosis"
    print(f"{degree:>7} {tr:>12.3f} {va:>10.3f}   {note}")

print("\nSame degree-15 family throughout, now penalised - flexibility is not the problem,")
print("UNCONTROLLED flexibility is:")
print(f"{'lambda':>8} {'train RMSE':>12} {'val RMSE':>10} {'max |coef|':>12}")
for lam in (0.0, 0.01, 1.0, 10.0, 100.0):
    tr, va, big = fit_ridge(15, lam)
    print(f"{lam:>8} {tr:>12.3f} {va:>10.3f} {big:>12.2f}")

print("\nTrain error only ever improves with flexibility, so it cannot choose a model.")
print("Validation error is U-shaped in flexibility - and lambda moves you along that same")
print("curve without changing the family, which is why one grid search over lambda often")
print("replaces a hunt for the right model class. Choose lambda by cross-validation, not")
print("by reading this table: five validation rows is not an estimate (see session 10).")
