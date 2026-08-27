"""Session 10 demo - one number is not an estimate, and the average hides who pays.

Sixty predictions on a held-out set. First bootstrap a confidence interval for accuracy,
then split the same predictions by group and print per-group recall. Both are twenty lines
of code, and both change what you would responsibly claim.

Run: python3 demo.py
"""

import random

random.seed(33)  # fixed and reported: this seed makes the gap legible on only 60 rows
N_BOOT = 2000

# A synthetic test set: (group, actual, predicted). Group B is smaller and, as a direct
# consequence of contributing less to the loss, is served worse by the model.
TEST = []
for _ in range(45):  # group A - the majority
    actual = 1 if random.random() < 0.40 else 0
    correct = random.random() < 0.85
    TEST.append(("A", actual, actual if correct else 1 - actual))
for _ in range(15):  # group B - the minority
    actual = 1 if random.random() < 0.40 else 0
    correct = random.random() < 0.62
    TEST.append(("B", actual, actual if correct else 1 - actual))


def accuracy(rows):
    return sum(1 for _, a, p in rows if a == p) / len(rows) if rows else float("nan")


def recall(rows):
    positives = [r for r in rows if r[1] == 1]
    if not positives:
        return float("nan")
    return sum(1 for _, a, p in positives if p == 1) / len(positives)


def precision(rows):
    flagged = [r for r in rows if r[2] == 1]
    if not flagged:
        return float("nan")
    return sum(1 for _, a, p in flagged if a == 1) / len(flagged)


def bootstrap_ci(rows, metric, n_boot=N_BOOT, level=0.95):
    """Resample the test set with replacement, recompute, take the percentiles."""
    stats = []
    for _ in range(n_boot):
        sample = [random.choice(rows) for _ in rows]
        value = metric(sample)
        if value == value:  # skip resamples where the metric is undefined
            stats.append(value)
    stats.sort()
    lo = stats[int((1 - level) / 2 * len(stats))]
    hi = stats[int((1 + level) / 2 * len(stats)) - 1]
    return lo, hi


def main():
    base_rate = sum(a for _, a, _ in TEST) / len(TEST)
    print(f"test set n = {len(TEST)}, base rate = {base_rate:.2f}")
    print(f"majority-class baseline accuracy = {max(base_rate, 1 - base_rate):.3f}")

    acc = accuracy(TEST)
    lo, hi = bootstrap_ci(TEST, accuracy)
    print(f"\noverall accuracy = {acc:.3f}   95% bootstrap CI [{lo:.3f}, {hi:.3f}]")
    print(f"the interval is {hi - lo:.3f} wide on 60 rows - which is why '0.847 vs 0.851'")
    print("is not a finding, and why a single number is not an estimate.")

    print("\nthe same predictions, disaggregated:")
    print(f"  {'group':<7} {'n':>4} {'accuracy':>9} {'recall':>8} {'precision':>10}   95% CI on recall")
    for group in ("A", "B"):
        rows = [r for r in TEST if r[0] == group]
        r_lo, r_hi = bootstrap_ci(rows, recall)
        print(f"  {group:<7} {len(rows):>4} {accuracy(rows):>9.3f} {recall(rows):>8.3f} "
              f"{precision(rows):>10.3f}   [{r_lo:.3f}, {r_hi:.3f}]")

    gap = recall([r for r in TEST if r[0] == "A"]) - recall([r for r in TEST if r[0] == "B"])
    print(f"\nrecall gap A - B = {gap:+.3f}. The overall accuracy above is an average over")
    print("people, and it concealed this entirely. Report the WORST group, not the mean.")

    print("\nSmall groups contribute little to the loss, so a model under-serves them by")
    print("construction - not by malice. Detecting that requires keeping the group variable,")
    print("which is the argument for measuring it even where you may not use it to predict.")
    print("\nWhat goes in the model card: this table, these intervals, and the sentence")
    print("saying which subgroup the model should not currently be used for.")


if __name__ == "__main__":
    main()
