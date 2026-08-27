"""Session 3 demo - the update rule, and nothing else.

Descend the squared-error loss for y = a + b*x on standardised features, print the loss
every 50 iterations, then re-run with a learning rate ten times too large so you can see
what divergence looks like before it happens to you at 23:00.

Run: python3 demo.py
"""

ROWS = [  # area_sqm, rent_eur - the twelve flats from session 1
    (32, 540), (45, 510), (52, 640), (60, 545), (68, 720), (75, 620),
    (80, 770), (95, 860), (38, 420), (55, 640), (110, 930), (48, 400),
]


def standardise(values):
    """Return (z-scores, mean, sd). Descent needs this; the closed form does not."""
    n = len(values)
    mu = sum(values) / n
    sd = (sum((v - mu) ** 2 for v in values) / n) ** 0.5
    return [(v - mu) / sd for v in values], mu, sd


def loss(xs, ys, a, b):
    """Mean squared error at the current parameters."""
    return sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / len(xs)


def gradient(xs, ys, a, b):
    """d/da and d/db of the mean squared error.

    grad = -(2/n) * X' r, with r the residual vector. Read it as: project the current
    error back onto each feature.
    """
    n = len(xs)
    residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
    d_a = -2 / n * sum(residuals)
    d_b = -2 / n * sum(r * x for r, x in zip(residuals, xs))
    return d_a, d_b


def descend(xs, ys, alpha, n_iter=400, verbose=True):
    a, b = 0.0, 0.0
    for step in range(1, n_iter + 1):
        d_a, d_b = gradient(xs, ys, a, b)
        a -= alpha * d_a
        b -= alpha * d_b
        if verbose and (step % 50 == 0 or step == 1):
            print(f"  iter {step:4d}   loss = {loss(xs, ys, a, b):14.2f}   a = {a:8.2f}   b = {b:8.2f}")
    return a, b


def closed_form(xs, ys):
    """Session 2's answer, for comparison."""
    n = len(xs)
    x_bar, y_bar = sum(xs) / n, sum(ys) / n
    cov = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    var = sum((x - x_bar) ** 2 for x in xs)
    slope = cov / var
    return y_bar - slope * x_bar, slope


def main():
    xs_raw = [x for x, _ in ROWS]
    ys = [y for _, y in ROWS]
    xs, _, _ = standardise(xs_raw)

    print("alpha = 0.05 - converges:")
    a, b = descend(xs, ys, alpha=0.05)

    a_star, b_star = closed_form(xs, ys)
    print(f"\nclosed form on the same standardised x: a = {a_star:.4f}, b = {b_star:.4f}")
    print(f"descent got:                            a = {a:.4f}, b = {b:.4f}")
    print("Same minimum, reached iteratively - the gradient is zero exactly where the")
    print("normal equations say it is.")

    print("\nalpha = 1.5 - diverges (the loss grows geometrically; in NumPy you would")
    print("reach `nan` instead of a very large float):")
    try:
        descend(xs, ys, alpha=1.5, n_iter=60)
    except OverflowError:
        print("  ... OverflowError - the parameters left the range of a float.")

    print("\nRule: plot the loss per iteration and keep the largest alpha that still")
    print("decreases monotonically. Never tune a learning rate blind.")


if __name__ == "__main__":
    main()
