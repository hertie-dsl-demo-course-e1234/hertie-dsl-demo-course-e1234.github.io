# Session 9 handout - the leakage checklist

One page. Print it, and run it over your project before you report any score. Bring it to
the clinic on 17 November.

## The five questions

**1. Was every fitted transformation fitted inside the split?**
Scalers, imputers, PCA, feature selection, target/impact encoding, resamplers (SMOTE),
outlier trimming, discretisation. All of them learn from data, so all of them belong inside
a `Pipeline` (or a `recipe`). If you can point to a line that touches all rows before the
split, you have found a leak.

**2. Would each feature genuinely be known at prediction time?**
For every column, ask: when the model has to make this decision in real life, does this
value exist yet? Anything recorded after the outcome - a closing date, a settlement amount,
a case status - is a proxy for the target, not a predictor.

**3. Are the rows independent?**
Repeated observations of the same municipality, household, patient or student must not
straddle a fold: use `GroupKFold`. Time-ordered rows must not be split randomly: train on
the past, validate on the future.

**4. Are there duplicates across the split?**
Deduplicate before splitting. A near-duplicate row scored on the other side of the split is
memorisation reported as prediction.

**5. Is the score too good?**
Compare against (a) the base rate or the majority-class baseline, (b) a trivial model - the
mean, last year's value, one feature. An unexplained jump is a bug you have not found yet,
not a result.

## The structural fix

```python
model = Pipeline([("pre", ColumnTransformer(...)), ("clf", Estimator())])
cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=0))
```

Every step refits inside every fold. This is not tidiness - it makes question 1 impossible
to fail by accident, which hand-run notebook cells do not.

## Before you report a number

- [ ] Raw data untouched; every transformation in code, none by hand.
- [ ] Seeds fixed and recorded (`random_state`, `set.seed`).
- [ ] Split made once, before any exploration of the target.
- [ ] Test set touched exactly once, at the end.
- [ ] Baseline score reported next to the model score.
- [ ] Interval, not a point estimate (bootstrap or repeated CV).
- [ ] Environment pinned (`requirements.txt` / `renv.lock`) and the run reproducible from a
      clean checkout with one command.

## Project sign-off

Write two sentences, for your report's methods section, naming which split scheme you used
and why it matches the dependence structure of your data. If you cannot write those two
sentences, that is the finding to bring to the clinic.
