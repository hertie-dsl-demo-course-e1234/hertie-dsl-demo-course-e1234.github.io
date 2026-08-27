"""Session 4 demo - logistic regression by hand, then the threshold decision.

Two hundred synthetic applicants, two features. Fit by gradient descent on the log-loss (the
session 3 loop, unchanged, with a new gradient), then sweep the decision threshold and
watch precision and recall trade off against each other.

Run: python3 demo.py
"""

import math
import random

random.seed(2026)  # fix the seed; report it


def make_data(n=200):
    """Two features, a linear decision boundary, and honest label noise."""
    rows = []
    for _ in range(n):
        x1 = random.uniform(-2, 2)   # standardised income
        x2 = random.uniform(-2, 2)   # standardised debt ratio
        score = 1.2 * x1 - 1.0 * x2 + 0.3
        p = 1 / (1 + math.exp(-score))
        rows.append(([1.0, x1, x2], 1 if random.random() < p else 0))
    return rows


def sigmoid(z):
    """Numerically stable logistic function - exp(710) overflows, so branch."""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def predict_proba(x, b):
    return sigmoid(sum(xi * bi for xi, bi in zip(x, b)))


def log_loss(rows, b):
    eps = 1e-12
    total = 0.0
    for x, y in rows:
        p = min(max(predict_proba(x, b), eps), 1 - eps)
        total -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return total / len(rows)


def fit(rows, alpha=0.3, n_iter=500):
    """grad = (1/n) X'(p - y) - the same shape as squared error, with p for y_hat."""
    b = [0.0] * len(rows[0][0])
    n = len(rows)
    for step in range(1, n_iter + 1):
        grad = [0.0] * len(b)
        for x, y in rows:
            err = predict_proba(x, b) - y
            for j, xj in enumerate(x):
                grad[j] += err * xj / n
        b = [bj - alpha * gj for bj, gj in zip(b, grad)]
        if step % 100 == 0:
            print(f"  iter {step:4d}  log-loss = {log_loss(rows, b):.4f}")
    return b


def counts(rows, b, threshold):
    tp = fp = tn = fn = 0
    for x, y in rows:
        pred = 1 if predict_proba(x, b) >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def main():
    rows = make_data()
    print(f"n = {len(rows)}, positives = {sum(y for _, y in rows)} "
          f"(the base rate every metric below must be read against)")

    print("\nfitting:")
    b = fit(rows)
    print(f"\ncoefficients [intercept, income, debt] = {[round(v, 3) for v in b]}")
    print("odds ratios                            = "
          f"{[round(math.exp(v), 3) for v in b[1:]]}")
    print("(true generating coefficients were 0.3, 1.2, -1.0)")

    print("\nthreshold sweep - the model is fixed, only the DECISION changes:")
    print("  thresh   TP  FP  TN  FN   precision  recall     F1")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        tp, fp, tn, fn = counts(rows, b, t)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
        print(f"   {t:.1f}    {tp:3d} {fp:3d} {tn:3d} {fn:3d}     {prec:6.3f}  {rec:6.3f}  {f1:6.3f}")

    print("\n0.5 is a default, not a result. Pick the row that matches the cost of the")
    print("two errors in the decision you are actually supporting.")


if __name__ == "__main__":
    main()
