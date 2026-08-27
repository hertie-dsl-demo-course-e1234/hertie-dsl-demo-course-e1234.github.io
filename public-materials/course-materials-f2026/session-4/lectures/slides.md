% Session 4 - Classification: logistic regression and decision thresholds
% Foundations of Machine Learning (Demo), E1234
% 29 September 2026

# From a number to a decision

The outcome is now a class: approved / rejected, at-risk / not. Two changes to session 2's
recipe are enough:

1. squash the linear score into `(0, 1)` so it can be read as a probability;
2. replace squared error with a loss that is appropriate for probabilities.

The features, the design matrix, and the descent loop from session 3 are unchanged.

# Why not just run least squares on 0/1?

You can, and it half-works. But fitted values leave `[0, 1]`, so they cannot be
probabilities; squared error punishes confident-and-correct predictions oddly; and the
implied noise model is plainly wrong. Logistic regression fixes all three for the price of
one function.

# The logistic function

```
sigma(z) = 1 / (1 + exp(-z))
```

- `sigma(0) = 0.5`, `sigma(-inf) = 0`, `sigma(+inf) = 1`, monotone throughout.
- Useful derivative: `sigma'(z) = sigma(z)(1 - sigma(z))`.

The model is `P(y = 1 | x) = sigma(x'b)`. The linear part is called the **log-odds**:

```
log( p / (1 - p) ) = x'b
```

so `b_j` is the change in log-odds per unit of `x_j`; `exp(b_j)` is an odds ratio. Report the
odds ratio to an audience, never the raw coefficient.

# The loss: log-loss (cross-entropy)

```
L(b) = -(1/n) sum_i [ y_i log p_i + (1 - y_i) log (1 - p_i) ],   p_i = sigma(x_i'b)
```

Each term is the negative log probability the model assigned to what actually happened. A
confident wrong answer (`p = 0.01` when `y = 1`) costs `log(100)` - a genuinely large
penalty, which is the point.

The gradient is remarkably clean:

```
grad L(b) = (1/n) X' (p - y)
```

Identical in form to session 3's gradient, with the probability vector in place of the
fitted values. No closed form exists, so we descend - the loss is convex, so descent
suffices.

# The threshold is a separate decision

The model outputs `p`. Turning `p` into a label needs a cut-off, and **0.5 is a default,
not a result.** The right threshold depends on the relative cost of the two errors:

- screening for a cheap follow-up test: catch cases, tolerate false positives - low
  threshold;
- triggering an audit that costs a person a week: high threshold.

Fit once, then choose the threshold from the cost structure. Sweeping it is not tuning the
model; it is specifying the decision.

# The confusion matrix

|  | predicted 1 | predicted 0 |
|---|---|---|
| actual 1 | TP | FN |
| actual 0 | FP | TN |

- **Precision** `= TP/(TP+FP)`: of those flagged, how many were right.
- **Recall** `= TP/(TP+FN)`: of those that mattered, how many were caught.
- **F1**: harmonic mean of the two - one number when both matter equally.
- **Accuracy** `= (TP+TN)/n`: dominated by the majority class, and therefore usually the
  least informative number on the page.

With 2% positives, "always predict 0" scores 98% accuracy and is worthless. Always state
the base rate next to the accuracy.

# ROC and PR curves

Sweep the threshold from 1 to 0 and trace:

- **ROC:** recall against false-positive rate. `AUC` = probability a random positive is
  scored above a random negative. Insensitive to class imbalance - a virtue and a trap.
- **Precision-recall:** the honest picture when positives are rare. Prefer it for imbalanced
  problems.

Both summarise the model across *all* thresholds; a deployed system has exactly one.

# Multi-class, in a sentence

Softmax generalises the sigmoid to `K` classes, with categorical cross-entropy as the loss.
Everything above carries over, per class; the confusion matrix simply becomes `K x K`.

# Today's demo and lab

- `demo.py`: pure-Python logistic regression on 200 synthetic applicants - sigmoid,
  log-loss, gradient, 500 descent steps - then a threshold sweep printing precision and recall at
  each cut-off.
- `labs/04_classification`: fit `LogisticRegression` on a small credit-screening extract,
  plot the ROC and PR curves, then pick a threshold from a stated cost ratio and defend the
  choice in two sentences.

# Coming next

Session 5 asks the question we have been dodging: given several models that all fit the
training data, which one do you actually pick? That is regularisation and model selection,
and it is where Assignment 2 comes from.
