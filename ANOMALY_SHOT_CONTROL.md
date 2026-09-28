# Matched normal-shot anomaly control

This independent experiment adds new files only. It reuses the existing
Mahalanobis implementation and saved baseline block folds.

```powershell
$env:PYTHONUTF8='1'
./.venv/Scripts/python.exe run_anomaly_shot_control.py
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Outputs: `outputs/experiments/anomaly_shot_control/`.
Do not run tests concurrently with the experiment's preservation hash check,
because existing tests temporarily create output files.

All three methods fit the identical unique training shots with max(y)=0.
IF/SPC retain baseline settings. Mahalanobis retains StandardScaler+LedoitWolf.
Algorithm-specific preprocessing is retained; all fitted statistics use only
the same normal training set. Primary comparison applies the same train-normal
ECDF to each model; raw scores are saved as a predefined sensitivity analysis.
Both use the same fitted models, without additional fitting or tuning.

Existing row-trained IF/SPC OOF predictions are preserved as references.
Previous raw Mahalanobis predictions are checked for numerical reproducibility.
`y_true` remains the original row label in exported predictions. `y_pred` is a
fixed training-normal raw score 95th percentile decision, independent of the
primary AP and budget recall metrics.

`fit_audit.csv` and `fit_shot_ids.json` verify identical normal training IDs.
`comparison_primary.csv`, `comparison_raw.csv`, and
`comparison_row_reference.csv` keep comparison roles explicit. Defect ranks,
type summaries, exclusive detections, pairwise overlaps, and preservation
hashes are included. CSV statistics use expected boundary-tie inclusion;
definite exclusive detections are separately marked.
