# Logistic Regression target ablation

Run on `exp/shot-target`, based on `feat/final-baseline` (`12cc65b`).
Existing baseline code, configuration, tests and artifacts are read-only.

```powershell
$env:PYTHONUTF8='1'
./.venv/Scripts/python.exe run_shot_experiment.py
./.venv/Scripts/python.exe run_evaluation.py --outputs outputs/experiments/shot_target
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Requires the existing `data/raw` files and `outputs/fold_id.csv`. Use
`--baseline-root <directory>` if these artifacts are in a separate baseline copy;
results still go under this repository's `outputs/experiments/shot_target/`.
Missing or inconsistent folds cause an error. No fallback split generation exists.
The original `config.json` must match the source baseline configuration.

Both arms use baseline's 23 selected process features, train-only StandardScaler,
and LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).
The row arm trains on original product rows; the shot arm trains once per shot
with any-defect target `max(y)`. Shot scores are mapped back to all original rows;
CSV `y_true` remains the original row label. Label-derived quantities are never
features. Every saved block repeat/fold is reused, with no inner CV. Both CSVs
use `y_pred = (y_score >= 0.5)`; F1 is not the objective of this experiment and
does not reproduce baseline's internally tuned decision threshold.

Primary metrics are shot AP (average precision) and Recall@5/10/20%, computed on
pooled OOF for each repeat, then averaged across five repeats (sample SD).
Row metrics are reference only. Recall uses ceil(N * budget); ties use expected
hits, consistent with the common evaluator. SD across repeated folds is not a
confidence interval or a significance test.

One-example-per-shot aggregation necessarily changes sample count, class
frequencies, class weights and the scaler's training population. These are
consequences of the requested target/unit change, not new features or tuned
hyperparameters. In particular, singleton shots are retained with equal shot
weight; samples are not reweighted to compensate for deduplication.

Outputs include the two common-schema prediction CSVs, local shot mapping,
per-fold training counts, repeat metrics, comparison CSV/Markdown, input hashes
and runtime versions. `baseline_score_check.csv` checks the row arm's scores
against saved baseline block OOF predictions, when present. The runner hashes
baseline data/config/output files before and after execution to check preservation.
The experiment directory is already gitignored by the baseline's outputs rule.
