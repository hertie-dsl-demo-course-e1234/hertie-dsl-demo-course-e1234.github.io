% Session 1 - Introduction: what machine learning is (and isn't)
% Foundations of Machine Learning (Demo), E1234
% 8 September 2026

# Where we are going

Ten sessions, one idea: **a model is a function chosen from a family by minimising a loss
on data.** Everything else - regression, trees, networks - is a choice of family, loss, or
minimiser.

Today: the vocabulary, the workflow, and the one mistake that invalidates more analyses
than all the others combined.

# Three kinds of learning problem

| Setting | You have | You want | Example |
|---|---|---|---|
| Supervised | features `X`, labels `y` | predict `y` for new `X` | rent from flat characteristics |
| Unsupervised | features `X` only | structure in `X` | grouping municipalities by profile |
| Reinforcement | states, actions, rewards | a policy | congestion pricing (not this course) |

Sessions 2-6 and 8-10 are supervised. Session 7 is unsupervised.

# Prediction is not explanation

Two different questions about the same data:

- **Prediction:** what will `y` be for a case I have not seen? Judged out of sample.
- **Explanation:** if I change `x`, how does `y` change? Requires an identification
  argument, not a low test error.

A model can predict rent well because it has learnt the postcode, and still tell you
nothing about what a metro station is worth. Keep the question in view; the metric you
report follows from it.

# The supervised setup

- Data: `n` observations, each a pair `(x_i, y_i)` with `x_i` in `R^p`.
- Hypothesis family: a set of candidate functions, e.g. all linear `f(x) = x'b`.
- Loss: how bad one prediction is, e.g. squared error `(y - f(x))^2`.
- Risk: the average loss on **new** data. This is the thing we actually care about,
  and the thing we can never observe.
- Empirical risk: average loss on the data we have. This is what we minimise, faute de
  mieux.

Learning is the gap management between those last two lines.

# The workflow

1. State the decision the model serves, and the metric that measures it.
2. Split the data - **before** looking at it - into train / validation / test.
3. Build the simplest baseline that could work. Write down its score.
4. Iterate on the training and validation sets only.
5. Touch the test set once, at the end, to report.
6. Document data, model, metric and known failure modes.

Step 3 is not a formality. A course-average model, or last year's value, beats a
surprising number of deployed systems.

# The one mistake

**Evaluating on data used to fit.** Any sufficiently flexible model can memorise its
training set, so training error can always be driven to zero and therefore measures
nothing.

Subtler versions of the same error, all of which we will meet again:

- scaling or imputing using statistics computed on the full dataset (session 9);
- selecting features by their correlation with `y` across all rows, then cross-validating
  the survivors (session 5);
- tuning until the test score looks good - the test set is then a training set (session 5).

# Overfitting and underfitting, informally

- **Underfit:** the family is too small to express the pattern. Train and test error are
  both high and close together.
- **Overfit:** the family is large enough to fit the noise. Train error is low, test error
  is high, and the gap is the diagnosis.

The cure is not "a more powerful model" or "a simpler model" in the abstract; it is reading
the two numbers and responding to which one is wrong.

# Notation we will use throughout

- `X`: `n x p` design matrix, rows are observations.
- `y`: length-`n` outcome vector.
- `b` (beta): parameter vector, `p` entries (plus an intercept).
- `y_hat = X b`: fitted values.
- `L(b)`: loss as a function of parameters - the object we minimise from session 3 on.

# Today's demo and lab

- `demo.py`: the baseline-first workflow on a 12-row housing extract - predict with the
  mean, then with one feature, and watch the honest error move.
- `labs/01_introduction`: a NumPy warm-up. Vectorised arithmetic, broadcasting, a train /
  test split by index, and the mean-absolute-error you will reuse in every later lab.

# Before session 2

- Read the ISLR chapters in `readings/01_introduction`.
- Get Python 3.10+ with `numpy`, `pandas`, `scikit-learn`, `matplotlib` working, or the
  equivalent R stack. Bring a laptop that runs the lab.
- Come with one question you would like to answer with data by December. We will use it in
  the project pitch.
