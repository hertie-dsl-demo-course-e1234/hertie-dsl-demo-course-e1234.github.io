"""Session 6 demo - grow a regression tree by hand, printing every candidate split.

The greedy step is usually asserted; here it is visible. Twelve flats, two features, an
exhaustive search over split points, depth 2. Then average three bootstrapped trees to
show why bagging helps.

Run: python3 demo.py
"""

import random

random.seed(2026)

# (area_sqm, dist_metro_km, rent_eur) - the twelve flats from session 1
ROWS = [
    (32, 0.3, 540), (45, 0.9, 510), (52, 0.4, 640), (60, 1.6, 545),
    (68, 0.7, 720), (75, 2.1, 620), (80, 1.1, 770), (95, 0.5, 860),
    (38, 1.8, 420), (55, 0.6, 640), (110, 1.4, 930), (48, 2.6, 400),
]
FEATURES = ("area_sqm", "dist_metro_km")


def mean(xs):
    return sum(xs) / len(xs)


def sse(rows):
    """Impurity for a regression node: sum of squared deviations from the node mean."""
    if not rows:
        return 0.0
    mu = mean([r[2] for r in rows])
    return sum((r[2] - mu) ** 2 for r in rows)


def candidate_splits(rows, feature_index):
    """Midpoints between consecutive distinct values - every split worth trying."""
    values = sorted({r[feature_index] for r in rows})
    return [(a + b) / 2 for a, b in zip(values, values[1:])]


def best_split(rows, verbose=False):
    """Exhaustive search: the (feature, threshold) with the largest impurity reduction."""
    parent = sse(rows)
    best = None
    for j, name in enumerate(FEATURES):
        for thr in candidate_splits(rows, j):
            left = [r for r in rows if r[j] <= thr]
            right = [r for r in rows if r[j] > thr]
            if not left or not right:
                continue
            reduction = parent - (sse(left) + sse(right))
            if verbose:
                print(f"    {name:<14} <= {thr:6.2f}   n=({len(left):2d},{len(right):2d})"
                      f"   impurity reduction = {reduction:10.1f}")
            if best is None or reduction > best[2]:
                best = (j, thr, reduction, left, right)
    return best


def grow(rows, depth, max_depth, verbose=False, indent=""):
    """A node is either a leaf (predict the mean) or a split plus two children."""
    if depth == max_depth or len(rows) < 3:
        return {"leaf": mean([r[2] for r in rows]), "n": len(rows)}
    if verbose:
        print(f"{indent}searching depth {depth} node (n = {len(rows)}, impurity = {sse(rows):.0f}):")
    found = best_split(rows, verbose=verbose)
    if found is None:
        return {"leaf": mean([r[2] for r in rows]), "n": len(rows)}
    j, thr, reduction, left, right = found
    if verbose:
        print(f"{indent}  -> chose {FEATURES[j]} <= {thr:.2f} (reduction {reduction:.1f})\n")
    return {
        "feature": j, "threshold": thr,
        "left": grow(left, depth + 1, max_depth, verbose, indent + "    "),
        "right": grow(right, depth + 1, max_depth, verbose, indent + "    "),
    }


def predict(tree, row):
    while "leaf" not in tree:
        tree = tree["left"] if row[tree["feature"]] <= tree["threshold"] else tree["right"]
    return tree["leaf"]


def show(tree, indent="  "):
    if "leaf" in tree:
        print(f"{indent}predict {tree['leaf']:.0f} EUR   (n = {tree['n']})")
        return
    print(f"{indent}if {FEATURES[tree['feature']]} <= {tree['threshold']:.2f}:")
    show(tree["left"], indent + "    ")
    print(f"{indent}else:")
    show(tree["right"], indent + "    ")


def main():
    print("=== growing one depth-2 tree, showing the greedy search ===\n")
    tree = grow(ROWS, 0, 2, verbose=True)
    print("resulting tree:")
    show(tree)

    single = mean([abs(r[2] - predict(tree, r)) for r in ROWS])
    print(f"\nin-sample MAE of the single tree: {single:.1f} EUR")
    print("(in-sample, so optimistic - see session 5. It is here only for comparison.)")

    print("\n=== the variance problem: five bootstrap trees disagree ===")
    forest = []
    for b in range(5):
        sample = [random.choice(ROWS) for _ in ROWS]
        t = grow(sample, 0, 2)
        forest.append(t)
        root = f"{FEATURES[t['feature']]} <= {t['threshold']:.2f}" if "feature" in t else "leaf"
        print(f"  bootstrap {b + 1}: root split on {root}")

    print("\npredictions for two flats, tree by tree:")
    for row in (ROWS[3], ROWS[6]):  # a 60 sqm and an 80 sqm flat
        preds = [predict(t, row) for t in forest]
        spread = max(preds) - min(preds)
        print(f"  area {row[0]:3d} sqm, actual {row[2]} EUR: "
              f"{[round(p) for p in preds]}  ->  spread {spread:.0f} EUR, "
              f"mean {mean(preds):.0f}")

    print("\nSame data-generating process, resampled: different root splits and predictions")
    print("that differ by more than a hundred euros. THAT is the variance a single tree")
    print("carries, and averaging over trees is what removes it - bias unchanged. A random")
    print("forest additionally withholds features at each split, so the trees are less")
    print("correlated and the averaging is worth more.")


if __name__ == "__main__":
    main()
