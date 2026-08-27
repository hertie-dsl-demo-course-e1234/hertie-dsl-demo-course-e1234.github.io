% Session 10 - Evaluation and ethics: metrics, uncertainty, and who bears the error
% Foundations of Machine Learning (Demo), E1234
% 10 November 2026

# The last question

You have a model that validates well. Three things remain, and they are the ones that decide
whether the work is any good: which number you report, how certain that number is, and who
pays when it is wrong.

# Pick the metric from the decision

| Situation | Report | Not |
|---|---|---|
| Errors cost proportionally | MAE | RMSE |
| Large errors cost disproportionately | RMSE | MAE |
| Rare positive class | precision, recall, PR-AUC | accuracy |
| Ranking a shortlist | precision@k | AUC |
| Probabilities feed a downstream calculation | log-loss, Brier, a calibration plot | any threshold metric |

Always print the **baseline** next to your score: the majority class, the mean, or last
year's value. A number without a baseline is not a result.

# Calibration is separate from discrimination

A model can rank perfectly (AUC 0.9) and still be systematically overconfident. If a
probability enters a budget or a risk threshold, it must be *calibrated*: among cases scored
0.3, about 30% should be positive.

Check it with a reliability diagram - predicted probability against observed frequency, ten
bins. Fix it with Platt scaling or isotonic regression, fitted on a held-out set. Trees and
boosted ensembles are usually poorly calibrated out of the box; that is a fixable defect, but
only if you look.

# One number is not an estimate

A test score computed on 200 rows has a standard error you can compute. Report an interval.

- **Bootstrap the test set:** resample with replacement, recompute the metric, take the 2.5th
  and 97.5th percentiles. Twenty lines of code.
- **Repeated CV** across seeds shows how much of your improvement is variance in the split.
- Before claiming model A beats model B, check the intervals overlap - and compare them on
  identical folds.

"0.847 versus 0.851" is almost never a finding.

# Aggregate performance hides distribution

Overall accuracy is an average over people, and averages conceal who the errors land on.
Disaggregate: compute your metric within each meaningful subgroup, and report the worst
group, not just the mean.

Common fairness criteria - and they are mutually incompatible, which is a theorem, not an
oversight:

- **Demographic parity:** equal positive rates across groups.
- **Equal opportunity:** equal true-positive rates.
- **Equalised odds:** equal TPR *and* FPR.
- **Calibration within groups:** a score means the same thing in each group.

You cannot satisfy all of them at once when base rates differ. So the choice must be argued
from the decision and its harms - it cannot be delegated to a library.

# Where the bias comes from

- **Historical:** the labels record past decisions, including past discrimination. A model
  trained on who *was* audited learns who *gets* audited.
- **Representation:** small subgroups contribute little to the loss, so the model
  under-serves them by construction.
- **Measurement:** the proxy is not the construct. Arrests are not crime; spend is not need.
- **Deployment:** the model changes the world it predicts. Feedback loops make a model look
  right by making it self-fulfilling.

Dropping the protected attribute fixes none of this - correlated features reconstruct it -
and it removes your ability to audit. Keep it for measurement; be deliberate about use.

# What you owe the reader

Document, with the model:

- **Intended use** and, explicitly, out-of-scope uses.
- **Data:** provenance, collection period, known gaps, consent basis.
- **Performance:** overall *and* per subgroup, with intervals.
- **Limitations:** the failure modes you know about.
- **Monitoring:** what would tell you the model has drifted, and who is watching.

That is a **model card** (Mitchell et al.) plus a **datasheet** (Gebru et al.) - both in this
week's readings, both about two pages. Every project submission includes a model card.

# Drift, after deployment

- **Data drift:** the input distribution moves. Monitor feature summaries.
- **Concept drift:** the relationship moves. Monitor the metric against delayed ground truth.
- Have a retraining cadence and a rollback plan decided *before* launch, not after.

# Today's demo and lab

- `demo.py`: bootstrap a confidence interval for accuracy on a 60-row test set, then split
  the same predictions by a synthetic group column and print per-group recall - the gap is
  large enough to see, on data small enough to check by hand.
- `labs/10_evaluation-and-ethics`: subgroup evaluation with bootstrap intervals, a
  reliability diagram, and a completed one-page model card for the model you built in
  session 9's lab.

# Course wrap-up

- Project due 27 November, 23:59. Clinics 17 November - bring the checklist from session 9.
- Revision sessions 1 and 8 December; final exam 15 December, 14:00.
- If you keep one thing: **state the decision, then choose the metric, then report the
  interval.** Everything else in this course is technique in service of that.
