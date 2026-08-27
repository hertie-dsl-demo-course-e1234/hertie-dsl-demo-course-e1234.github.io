"""Session 1 demo - baseline first.

Two models on twelve flats: predict the mean rent, then predict from floor area. The
point is not the second model; it is that you cannot tell whether it is any good until
there is a number to beat, measured on rows the model never saw.

The same twelve rows ship as `datasets/housing-mini.csv` (released to `data/` in the
cohort materials repo); they are inlined here so the demo runs anywhere.

Run: python3 demo.py
"""

# area_sqm, rent_eur
ROWS = [
    (32, 540), (45, 510), (52, 640), (60, 545), (68, 720), (75, 620),
    (80, 770), (95, 860), (38, 420), (55, 640), (110, 930), (48, 400),
]

TRAIN, TEST = ROWS[:8], ROWS[8:]  # a fixed split, decided before looking at the data


def mean(xs):
    return sum(xs) / len(xs)


def mae(actual, predicted):
    """Mean absolute error - in euros, which is what a reader can interpret."""
    return mean([abs(a - p) for a, p in zip(actual, predicted)])


def fit_one_feature(rows):
    """Least squares for y = a + b*x, in the two-parameter closed form."""
    xs = [x for x, _ in rows]
    ys = [y for _, y in rows]
    x_bar, y_bar = mean(xs), mean(ys)
    cov = sum((x - x_bar) * (y - y_bar) for x, y in rows)
    var = sum((x - x_bar) ** 2 for x in xs)
    slope = cov / var
    return y_bar - slope * x_bar, slope


def main():
    train_y = [y for _, y in TRAIN]
    test_y = [y for _, y in TEST]

    # Model 0: the baseline. No features at all.
    baseline = mean(train_y)
    print(f"baseline (train mean)      = {baseline:7.1f} EUR")
    print(f"  train MAE                = {mae(train_y, [baseline] * len(train_y)):7.1f} EUR")
    print(f"  test  MAE                = {mae(test_y, [baseline] * len(test_y)):7.1f} EUR")

    # Model 1: one feature.
    a, b = fit_one_feature(TRAIN)
    print(f"\nfitted rent = {a:.1f} + {b:.2f} * area_sqm")
    train_hat = [a + b * x for x, _ in TRAIN]
    test_hat = [a + b * x for x, _ in TEST]
    print(f"  train MAE                = {mae(train_y, train_hat):7.1f} EUR")
    print(f"  test  MAE                = {mae(test_y, test_hat):7.1f} EUR")

    print(
        "\nRead the TEST column, not the train column: the train error can always be\n"
        "driven down, so only the held-out number tells you the feature earned its place."
    )


if __name__ == "__main__":
    main()
