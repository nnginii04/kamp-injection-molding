# Normal-shot Mahalanobis ablation

Independent experiment on `exp/mahalanobis`, based on `feat/final-baseline`.
Existing baseline code, config, tests, and outputs are not modified.

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
./.venv/Scripts/python.exe run_mahalanobis.py
./.venv/Scripts/python.exe run_evaluation.py --outputs outputs/experiments/mahalanobis
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Requires original `data/raw` CSVs, baseline `outputs/manifest.json`,
`outputs/fold_id.csv`, `outputs/tables/cv_summary.csv`, and saved prediction CSVs
for `logreg`, `iforest_normal`, and `spc_maxz`. Missing or inconsistent saved
assignments are errors; the experiment never generates CV splits.

The 23 baseline process features are unchanged. Each shot is one example.
Normal training shots have `max(original row y) == 0`, excluding both-sided and
one-sided defects, including the normal mate of a one-sided defective shot.
For each saved fold, StandardScaler and LedoitWolf fit only the train normal
shots. Shrinkage is analytically estimated from that training population.
Risk score is the square root of the fitted quadratic Mahalanobis distance.
No validation-fitted normalization, ECDF, threshold search, or hyperparameter
search is performed. Raw distances are pooled across folds, so fold-specific
scale differences remain a limitation.

OOF shot scores are expanded back to original product rows with original
`y_true` and common prediction schema. `y_pred` uses a fixed training-normal
95th percentile threshold solely for interoperability. Primary results use AP
and Recall@5/10/20%, independent of this threshold. Evaluation uses pooled OOF
within each repeat, then the five-repeat mean and sample standard deviation.
Budget ties use uniform expected inclusion. Pairwise detection overlaps use
independent tie resolution if both methods have boundary ties.

The three comparator models are not retrained. They previously fitted on
product rows (IF/SPC on normal product rows), while this experiment fits only
fully normal unique shots. Thus differences cannot be attributed solely to
the covariance model. Repeated CV observations reuse the same defects and
are not independent external validation.

All results go to `outputs/experiments/mahalanobis/` (already gitignored).
`manifest.json` records baseline data/config/code/output hashes before and
after execution. `fit_parameters.json` records each fold's training-normal IDs,
scaler statistics, covariance and precision matrices for auditing. Existing
baseline metric reproduction is checked. Keep test runs separate from the
experiment run because existing tests briefly create temporary output files.
