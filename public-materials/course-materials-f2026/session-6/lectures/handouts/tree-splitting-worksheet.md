# Session 6 handout - choose the root split by hand

Ten minutes, pen only. The greedy step is easy to accept and easy to misremember; compute it
once.

## The data

Eight loan applications. Target: `default` (1 = defaulted).

| i | income_k | late_payments | default |
|---|----------|---------------|---------|
| 1 | 22 | 3 | 1 |
| 2 | 28 | 0 | 0 |
| 3 | 35 | 2 | 1 |
| 4 | 41 | 0 | 0 |
| 5 | 46 | 4 | 1 |
| 6 | 52 | 1 | 0 |
| 7 | 60 | 0 | 0 |
| 8 | 75 | 2 | 0 |

Parent node: 3 defaults, 5 non-defaults.

## 1. Parent impurity

Gini impurity for a node is `1 - sum_k p_k^2`. With `p_1 = 3/8`:

```
Gini(parent) = 1 - (3/8)^2 - (5/8)^2 = ______
```

## 2. Score three candidate splits

For each split, list the members of each child, compute each child's Gini, then the
**weighted** child impurity (weight by the number of rows), then the reduction
`Gini(parent) - weighted children`.

| candidate | left child (rows) | right child (rows) | Gini(L) | Gini(R) | weighted | reduction |
|---|---|---|---|---|---|---|
| `income_k <= 38` | | | | | | |
| `late_payments <= 0` | | | | | | |
| `late_payments <= 1` | | | | | | |

## 3. Pick the root

Which split does CART choose, and by how much does it beat the runner-up? Is the margin
large enough that you would expect the same root on a bootstrap resample of these eight
rows? (This is the question `demo.py` answers empirically, and the reason bagging exists.)

## 4. Read the tree you have started

Write out the two-line rule the chosen split gives you, and state the predicted class in
each child. Then say what a *second* level of splitting could still fix, and which child you
would grow first if you were allowed only one more split.

## 5. Two short questions

1. Recompute the reduction for your chosen split using **entropy**
   (`-sum_k p_k log2 p_k`) instead of Gini. Does the ranking of the three candidates
   change? (It very rarely does - which is why the choice of impurity measure is not worth
   an argument.)
2. Rescale `income_k` into euros rather than thousands. Which numbers in your table change,
   and which do not? What does that tell you about whether trees need feature scaling?
