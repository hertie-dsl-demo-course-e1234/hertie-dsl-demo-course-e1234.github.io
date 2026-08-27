% Session 9 - Feature engineering, pipelines and leakage
% Foundations of Machine Learning (Demo), E1234
% 3 November 2026

# Where the gains actually are

Between a well-tuned ridge and a well-tuned boosting model there is often a couple of
percentage points. Between a careless feature matrix and a careful one there is often ten.
This session is the least glamorous and the highest return in the course.

# Features that earn their place

- **Numeric:** transform for shape - `log` for right-skewed money and counts, ratios for
  scale-free comparison (`rooms per person`, not `rooms` and `people` separately), and
  differences for change over time.
- **Categorical:** one-hot for low cardinality; ordinal *only* where the order is real;
  target/impact encoding for high cardinality - computed inside the fold, or it leaks.
- **Dates:** decompose. Month, day-of-week, and "days since event" carry the signal; a raw
  timestamp does not.
- **Text:** start with TF-IDF over a modest vocabulary. It is a strong baseline and it is
  legible.
- **Interactions:** add them deliberately where theory suggests one. Trees find them
  unaided; linear models cannot.

Every added column costs variance (session 5). Adding forty features because they were
available is how a good dataset becomes an overfit model.

# Missing values are information

- Drop rows only if missingness is rare and plausibly random. It rarely is.
- Impute with the median (numeric) or a `"Missing"` level (categorical).
- **Always add an indicator column** `was_missing`. "This field was blank" is frequently
  predictive - non-response is a behaviour.
- Fit the imputer on the training fold only. `SimpleImputer` inside a `Pipeline` does this
  for you; `df.fillna(df.mean())` before splitting does not.

# Leakage: the failure that looks like success

Leakage is any information in your features that would not be available at prediction time.
The symptom is a suspiciously excellent validation score that collapses in deployment.

The recurring forms:

1. **Preprocessing before splitting.** Scaling, imputing, PCA or target encoding computed on
   all rows lets test statistics into training.
2. **A proxy for the target.** `discharge_date` when predicting length of stay;
   `payment_received` when predicting default. If a feature is filled in *after* the outcome
   is known, it cannot be used.
3. **Temporal leakage.** A random split on time-ordered data trains on the future. Use a
   forward-chaining split.
4. **Group leakage.** The same municipality, patient or student in both train and test.
   `GroupKFold`.
5. **Duplicate rows** straddling the split - a memorised row scores as a prediction.

# The pipeline discipline

Wrap **every** fitted transformation and the estimator in one object:

```python
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
cross_val_score(model, X_train, y_train, cv=5)   # every fold refits `pre` correctly
```

The point is not tidiness. It is that `cross_val_score` on a pipeline is arithmetically
unable to leak, whereas the same steps run by hand in a notebook usually do. R's `recipes` /
`tidymodels` enforces the same separation.

# The leakage checklist

Before you trust a score, answer all five:

1. Was every fitted transformation fitted **inside** the split?
2. Would each feature genuinely be known at prediction time?
3. Are rows independent, or do they cluster by group or time?
4. Are there duplicates across the split?
5. Is the score too good? Compare against the base rate and against a trivial baseline.

A validation score you cannot explain is a bug you have not found yet.

# Reproducibility, briefly

Pin your versions, fix your seeds, keep raw data read-only, and script every transformation
- no manual spreadsheet edits. If the pipeline cannot be re-run from the raw file with one
command, the result is an anecdote.

# Today's demo, handout and lab

- `demo.py`: one dataset, one model, one set of folds - and three cross-validation scores.
  Preprocessing inside the folds, then scaling fitted before the split (which here costs
  nothing), then feature selection fitted before the split (which invents 13 AUC points out
  of 37 noise columns). You cannot tell from the output which one you did.
- `handouts/pipeline-checklist.md`: the five-question checklist as a one-page sheet. Bring
  it to the project clinic; we will apply it to your team's design.
- `labs/09_feature-engineering-and-pipelines`: build a `ColumnTransformer` over mixed
  numeric and categorical columns, cross-validate it, then deliberately introduce a leak and
  measure how much it inflates the score.

# Coming next

Session 10 closes the loop: given a model that works, which number do you report, to whom,
and what do you owe the people it is applied to?
