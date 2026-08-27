"""Session 9 demo - three cross-validation scores, one honest.

Same model, same folds, same data. The only thing that changes is WHERE the preprocessing
is fitted. Note that the two leaks differ enormously in severity - and that nothing in the
output would tell you which one you had committed.

Requires numpy + scikit-learn. Run: python3 demo.py
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(2026)

# A deliberately small, wide problem - 60 rows, 40 columns, only 3 of them real. Leakage
# bites hardest exactly here, which is where policy datasets often live.
n, p, k_real = 60, 40, 3
X = rng.normal(0, 1, (n, p)) * rng.uniform(0.5, 50, p)   # wildly different scales
signal = X[:, :k_real] / np.abs(X[:, :k_real]).std(0)
y = (0.5 * signal @ [0.9, -0.7, 0.5] + rng.normal(0, 1, n) > 0).astype(int)  # weak, real
X[rng.random(X.shape) < 0.05] = np.nan                    # 5% missing, at random

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
clf = LogisticRegression(max_iter=2000)


def steps(*extra):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), *extra, ("clf", clf)])


# --- 1. clean: every fitted transformation lives inside the pipeline -------------------
clean = cross_val_score(steps(("sel", SelectKBest(f_classif, k=5))), X, y,
                        cv=cv, scoring="roc_auc")

# --- 2. mild leak: impute + scale once, over all 60 rows ------------------------------
X_scaled = StandardScaler().fit_transform(
    SimpleImputer(strategy="median").fit_transform(X))
mild = cross_val_score(Pipeline([("sel", SelectKBest(f_classif, k=5)), ("clf", clf)]),
                       X_scaled, y, cv=cv, scoring="roc_auc")

# --- 3. severe leak: choose the features using every row, INCLUDING the labels ---------
X_selected = SelectKBest(f_classif, k=5).fit_transform(X_scaled, y)
severe = cross_val_score(clf, X_selected, y, cv=cv, scoring="roc_auc")

print("5-fold CV ROC-AUC - identical model, identical folds, 60 rows x 40 columns:\n")
for label, scores in (("preprocessing INSIDE the folds  (correct)", clean),
                      ("impute + scale before splitting (mild leak)", mild),
                      ("feature selection before splitting (severe leak)", severe)):
    print(f"  {label:<48} {scores.mean():.3f}  (sd {scores.std():.3f})")

print(f"\n  inflation, mild leak   : {mild.mean() - clean.mean():+.3f} AUC")
print(f"  inflation, severe leak : {severe.mean() - clean.mean():+.3f} AUC")

print("\nWhy the second leaks: the imputer's medians and the scaler's standard deviations")
print("were computed over all 60 rows, so each validation fold's own statistics were")
print("already baked into training. Here it happens to cost almost nothing - and the size")
print("of that effect is not something you can predict in advance, which is the point.")
print("\nWhy the third is fatal: SelectKBest ranked 40 columns by their association with y")
print("across every row, so the five survivors were chosen partly BECAUSE they happen to")
print("fit the validation rows. With 37 pure-noise columns, that is most of the score.")

print("\nNeither leak let the model see a validation label at fit time - leakage does not")
print("require that. And the inflated numbers look completely ordinary.")

print("\nThe fix is structural rather than a matter of care: cross_val_score on a Pipeline")
print("refits every step inside each fold and is arithmetically unable to leak. The same")
print("steps run by hand across notebook cells usually do.")

print("\nThe five questions before you trust any score:")
for i, q in enumerate([
    "Was every fitted transformation fitted inside the split?",
    "Would each feature genuinely be known at prediction time?",
    "Are rows independent, or do they cluster by group or time?",
    "Are there duplicate rows across the split?",
    "Is the score too good, against the base rate and a trivial baseline?",
], 1):
    print(f"  {i}. {q}")
print(f"\n(base rate here: {y.mean():.2f} - always print it next to the score)")
