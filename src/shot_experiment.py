"""Row versus any-defect shot targets, using saved block folds only.

No split generation, feature engineering, calibration, or model selection.
The shot arm has one training example per shot (including singleton shots).
"""
from __future__ import annotations

import pandas as pd

from .data import shot_table
from .evaluate import metric_block, to_shot_level
from .features import check_feature_list
from .models import make_model


PRED_COLS = ["mold", "product", "model", "cv_mode", "repeat", "fold", "row_id",
             "shot_id", "y_true", "y_score", "y_pred"]
BUDGETS = (0.05, 0.10, 0.20)
LABEL_DERIVED = {"y_shot", "conflict", "n_pos", "n_rows", "defect_type"}


def prepare(lab, folds, feats, n_repeats, n_splits):
    """Validate the entire saved assignment before any model is fitted."""
    check_feature_list(feats)
    if set(feats) & LABEL_DERIVED:
        raise ValueError("Label-derived feature requested")
    cols = [f"fold_block_r{r}" for r in range(n_repeats)]
    required = {"mold", "row_id", "shot_id", "block_id", *cols}
    if not required <= set(folds.columns):
        raise ValueError(f"Missing saved fold columns: {sorted(required - set(folds.columns))}")
    if lab.duplicated(["product", "row_id"]).any() or folds.duplicated(["mold", "row_id"]).any():
        raise ValueError("Duplicate product/row_id")
    if folds[list(required)].isna().any().any():
        raise ValueError("Missing saved fold identity or assignment")
    for col in cols:
        # Check shot consistency first, including rows not present in the input.
        if (folds.groupby("shot_id")[col].nunique() != 1).any():
            raise ValueError(f"A shot_id occurs in multiple folds: {col}")
        if (folds.groupby("block_id")[col].nunique() != 1).any():
            raise ValueError(f"A block_id occurs in multiple folds: {col}")
        for _, f in folds.groupby("mold"):
            if set(f[col]) != set(range(n_splits)):
                raise ValueError(f"Invalid fold numbers: {col}")
    row = lab.merge(folds.rename(columns={"mold": "product", "shot_id": "saved_shot_id"}),
                    on=["product", "row_id"], how="outer", validate="one_to_one", indicator=True)
    if not row._merge.eq("both").all():
        raise ValueError("Saved folds and labeled rows do not match exactly")
    if not row.shot_id.eq(row.saved_shot_id).all():
        raise ValueError("Saved shot_id differs from baseline assign_shots")
    if (row.groupby("shot_id").product.nunique() != 1).any():
        raise ValueError("shot_id shared by multiple products")
    if (row.groupby("shot_id")[feats].nunique(dropna=False) > 1).any().any():
        raise ValueError("Feature values differ within a shot")
    row = row.drop(columns=["_merge", "saved_shot_id"]).reset_index(drop=True)
    shots = row.drop_duplicates("shot_id")[["product", "shot_id", *feats, "block_id", *cols]].copy()
    targets = shot_table(row).set_index("shot_id").y_shot
    shots["y"] = shots.shot_id.map(targets)
    # Preflight both classes for every arm/fold; never silently substitute a model.
    for frame in (row, shots):
        for product, part in frame.groupby("product"):
            for col in cols:
                for fold in range(n_splits):
                    if part.loc[part[col].ne(fold), "y"].nunique() != 2:
                        raise ValueError(f"Training fold lacks both classes: {product}/{col}/{fold}")
    return row, shots.reset_index(drop=True), cols


def run_ablation(lab, folds, feats, cfg, log=print):
    row, shots, cols = prepare(lab, folds, feats, cfg["cv"]["n_repeats"], cfg["cv"]["n_splits"])
    predictions, counts = [], []
    for product, original in row.groupby("product", sort=True):
        for target, frame in (("row", row), ("shot", shots)):
            part = frame[frame["product"].eq(product)]
            for rep, col in enumerate(cols):
                for fold in range(cfg["cv"]["n_splits"]):
                    train, valid = part[part[col].ne(fold)], part[part[col].eq(fold)]
                    model = make_model("logreg", cfg["seed"])
                    model.fit(train[feats], train.y.to_numpy())
                    score = model.predict_proba(valid[feats])[:, 1]
                    dest = original[original[col].eq(fold)].copy()
                    if target == "shot":
                        dest["y_score"] = dest.shot_id.map(pd.Series(score, index=valid.shot_id))
                    else:
                        dest["y_score"] = dest.row_id.map(pd.Series(score, index=valid.row_id))
                    if dest.y_score.isna().any():
                        raise ValueError("Incomplete OOF score mapping")
                    dest = dest.rename(columns={"y": "y_true"})
                    dest["mold"], dest["model"] = product, f"logreg_{target}_target"
                    dest["cv_mode"], dest["repeat"], dest["fold"] = "block", rep, fold
                    dest["y_pred"] = dest.y_score.ge(0.5).astype(int)
                    predictions.append(dest[PRED_COLS])
                    counts.append(dict(product=product, target=target, repeat=rep, fold=fold,
                                       train_examples=len(train), train_positives=int(train.y.sum()),
                                       valid_examples=len(valid), valid_shots=valid.shot_id.nunique()))
                log(f"{product} {target}-target: repeat {rep} complete")
    pred = pd.concat(predictions, ignore_index=True).sort_values(["model", "product", "repeat", "row_id"])
    return pred.reset_index(drop=True), pd.DataFrame(counts)


def summarize(pred):
    """Pooled OOF per repeat, then mean/sample SD. Fold AP is not averaged."""
    rows = []
    for (product, model, repeat), part in pred.groupby(["product", "model", "repeat"]):
        frame = part.rename(columns={"y_true": "y", "y_score": "p", "y_pred": "pred"})
        for level, f in (("shot", to_shot_level(frame)), ("row", frame)):
            metrics = metric_block(f.y, f.p, f.pred, BUDGETS)
            rows.append(dict(product=product, model=model, repeat=repeat, level=level,
                             ap=metrics["ap"], **{f"recall@{k}%": metrics[f"recall@{k}%"] for k in (5, 10, 20)}))
    detail = pd.DataFrame(rows)
    metrics = ["ap", "recall@5%", "recall@10%", "recall@20%"]
    grouped = detail.groupby(["product", "model", "level"])[metrics]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_sd")).reset_index()
    return detail, summary


def comparison_markdown(summary):
    lines = ["# Row-target logreg vs shot-target logreg", "",
             "Same saved block folds; pooled OOF per repeat, mean ± sample SD across repeats.",
             "AP = average precision. Recall budgets use ceil(N × budget), with expected hits for ties.",
             "Both arms reuse baseline features and StandardScaler + LogisticRegression(C=0.3, class_weight=balanced).",
             "Shot training uses one example per shot and max(y); row training uses every original row.",
             "This also changes sample counts, class frequencies and effective training weights; these follow from the requested shot unit.",
             "No CV splits generated. y_pred uses 0.5 in both arms; threshold tuning is excluded.", ""]
    for level, title in (("shot", "Primary: shot-level"), ("row", "Reference only: row-level")):
        lines += [f"## {title}", "", "| Product | Model | AP | Recall@5% | Recall@10% | Recall@20% |",
                  "|---|---|---:|---:|---:|---:|"]
        for _, r in summary[summary.level.eq(level)].iterrows():
            vals = [f"{r[m + '_mean']:.4f} ± {r[m + '_sd']:.4f}" for m in ("ap", "recall@5%", "recall@10%", "recall@20%")]
            lines.append("| " + " | ".join([r["product"].upper(), r["model"], *vals]) + " |")
        lines.append("")
    return "\n".join(lines)
