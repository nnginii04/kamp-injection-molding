"""Run the isolated logreg target ablation against existing baseline artifacts."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import pandas as pd

from src.data import assign_shots, load_all, sha256
from src.features import check_feature_list, model_features, select_base_features
from src.shot_experiment import comparison_markdown, run_ablation, summarize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "experiments" / "shot_target"


def baseline_hashes(source):
    """Read-only inventory; experimental results are deliberately excluded."""
    paths = [ROOT / "config.json", source / "config.json"]
    for folder in (source / "data" / "raw", source / "outputs"):
        paths.extend(p for p in folder.rglob("*") if p.is_file() and "experiments" not in p.relative_to(folder).parts)
    return {str(p): sha256(p) for p in paths}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-root", type=Path, default=ROOT,
                    help="Existing baseline directory containing config.json, data/raw and outputs/fold_id.csv")
    args = ap.parse_args()
    source = args.baseline_root.resolve()
    cfg_path, fold_path = ROOT / "config.json", source / "outputs" / "fold_id.csv"
    if not fold_path.is_file():
        raise FileNotFoundError(f"Saved folds required; splits will not be generated: {fold_path}")
    before = baseline_hashes(source)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if json.loads((source / "config.json").read_text(encoding="utf-8")) != cfg:
        raise ValueError("Source baseline config differs from this baseline")
    fcfg = cfg["features"]
    if cfg["scenarios"] != ["separate"] or any(fcfg.get(k, False) for k in
            ("sequence_features", "unlabeled_iforest_score", "pair_position")) or cfg["holdout"]["enabled"]:
        raise ValueError("This ablation requires unchanged final-baseline feature/scenario settings")
    data = load_all(cfg, source / cfg["data_dir"])
    manifest_path = source / "outputs" / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("input_sha256") != data["hashes"]:
            raise ValueError("Data hashes differ from saved baseline manifest")
    base, dropped = select_base_features(data, data["feature_cols"], fcfg["drop_constant"])
    feats = check_feature_list(model_features(base, "separate", False))
    lab = pd.concat([assign_shots(d, data["feature_cols"], p) for p, d in data["labeled"].items()], ignore_index=True)
    folds = pd.read_csv(fold_path)
    pred, counts = run_ablation(lab, folds, feats, cfg)
    detail, summary = summarize(pred)
    OUT.mkdir(parents=True, exist_ok=True)
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for name, frame in pred.groupby("model"):
        frame.to_csv(pred_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    # Local mapping allows run_evaluation.py --outputs outputs/experiments/shot_target.
    lab[["product", "row_id", "shot_id"]].to_csv(OUT / "predictions_labeled_cv.csv", index=False)
    counts.to_csv(OUT / "fold_counts.csv", index=False)
    detail.to_csv(OUT / "metrics_by_repeat.csv", index=False)
    summary.to_csv(OUT / "comparison.csv", index=False)
    report = comparison_markdown(summary)
    (OUT / "comparison.md").write_text(report, encoding="utf-8")
    reference_path = source / "outputs" / "predictions" / "pipeline_logreg.csv"
    reference_check = []
    if reference_path.exists():
        ref = pd.read_csv(reference_path)
        ref = ref[ref.cv_mode.eq("block") & ref.model.eq("logreg")]
        fresh = pred[pred.model.eq("logreg_row_target")]
        keys = ["product", "repeat", "fold", "row_id", "shot_id", "y_true"]
        match = fresh.merge(ref, on=keys, how="outer", validate="one_to_one", indicator=True,
                            suffixes=("_rerun", "_baseline"))
        if not match._merge.eq("both").all():
            raise ValueError("Saved baseline predictions differ in labels, rows, or folds")
        for product, part in match.groupby("product"):
            reference_check.append(dict(product=product, rows=len(part),
                max_abs_score_diff=float((part.y_score_rerun - part.y_score_baseline).abs().max())))
        pd.DataFrame(reference_check).to_csv(OUT / "baseline_score_check.csv", index=False)
    if before != baseline_hashes(source):
        raise RuntimeError("Baseline inputs changed during the run")
    manifest = dict(baseline_root=str(source), input_sha256=data["hashes"],
                    fold_sha256=sha256(fold_path), config_sha256=sha256(cfg_path),
                    features=feats, dropped_features=dropped, cv_mode="block",
                    n_repeats=cfg["cv"]["n_repeats"], n_splits=cfg["cv"]["n_splits"],
                    split_policy="saved fold_id.csv only; no split generation", decision_threshold=0.5,
                    primary_metrics=["shot AP", "shot Recall@5%", "shot Recall@10%", "shot Recall@20%"],
                    target_definition="max(original row y) per shot; one training example per shot",
                    python=platform.python_version(),
                    command=subprocess.list2cmdline([sys.executable, *sys.argv]),
                    baseline_files_unchanged=True, baseline_files_sha256=before,
                    baseline_score_check=reference_check,
                    versions={k: importlib.metadata.version(k) for k in ("numpy", "pandas", "scikit-learn", "scipy")},
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
