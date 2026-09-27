"""베이스라인 예측 파일의 간단 점검용 지표.

정식 평가(임계값 결정·캘리브레이션·Recall@K 곡선·Inspection Score)는 C 담당의 evaluation 모듈이 맡는다.
여기서는 예측 CSV가 스키마대로 나왔는지, 값이 말이 되는지만 본다.

- fold별 지표 → mean ± std
- PR-AUC(불량=1), ROC-AUC, 임계값 0.5 기준 Precision/Recall/F1/FN, Recall@10% (상위 10% 검사 시 잡히는 불량 비율)
- 결과: outputs/evaluation/baseline_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.loading import load_config


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, frac: float) -> float:
    """확률 상위 frac 비율만 검사했을 때 잡히는 불량 비율."""
    n = max(1, int(np.ceil(len(y_true) * frac)))
    top = np.argsort(-y_prob)[:n]
    pos = y_true.sum()
    return float(y_true[top].sum() / pos) if pos > 0 else np.nan


def fold_metrics(g: pd.DataFrame, thr: float = 0.5) -> dict:
    y, p = g["y_true"].to_numpy(), g["y_prob"].to_numpy()
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": len(g), "n_defect": int(y.sum()),
        "pr_auc": average_precision_score(y, p) if y.sum() else np.nan,
        "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        "precision@0.5": prec, "recall@0.5": rec, "f1@0.5": f1, "fn@0.5": fn, "fp@0.5": fp,
        "recall@10%": recall_at_k(y, p, 0.10),
        "recall@20%": recall_at_k(y, p, 0.20),
    }


def summarize_predictions(pred_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(pred_dir.glob("*__oof.csv")):
        df = pd.read_csv(f)
        model, feature_set = df["model"].iloc[0], df["feature_set"].iloc[0]
        per_fold = pd.DataFrame([fold_metrics(g) for _, g in df.groupby(["fold_rep", "fold"])])
        for col in per_fold.columns:
            if col in ("n", "n_defect"):
                continue
            rows.append({"model": model, "feature_set": feature_set, "split": "oof(25 folds)", "metric": col,
                         "mean": per_fold[col].mean(), "std": per_fold[col].std(), "min": per_fold[col].min(),
                         "max": per_fold[col].max()})
        hold = pred_dir / f.name.replace("__oof.csv", "__holdout.csv")
        if hold.exists():
            h = fold_metrics(pd.read_csv(hold))
            for col, v in h.items():
                if col in ("n", "n_defect"):
                    continue
                rows.append({"model": model, "feature_set": feature_set, "split": "holdout", "metric": col,
                             "mean": v, "std": np.nan, "min": v, "max": v})
    return pd.DataFrame(rows)


def check_schema(pred_dir: Path, folds: pd.DataFrame, n_repeats: int) -> None:
    """예측 파일이 fold_id.csv와 어긋나지 않는지 확인."""
    from src.models.train import SCHEMA
    dev_ids = set(folds.loc[folds["holdout"] == 0, "row_id"])
    hold_ids = set(folds.loc[folds["holdout"] == 1, "row_id"])
    for f in sorted(pred_dir.glob("*__oof.csv")):
        df = pd.read_csv(f)
        assert list(df.columns) == SCHEMA, f"{f.name}: 열 순서가 스키마와 다름"
        assert len(df) == len(dev_ids) * n_repeats, f"{f.name}: 행 수 {len(df)} != dev {len(dev_ids)} × {n_repeats}"
        assert set(df["row_id"]) == dev_ids, f"{f.name}: row_id가 개발셋과 다름"
        assert df["y_prob"].between(0, 1).all(), f"{f.name}: y_prob 범위 이상"
        per_rep = df.groupby("fold_rep")["row_id"].nunique()
        assert (per_rep == len(dev_ids)).all(), f"{f.name}: rep 안에서 row_id 중복/누락"
    for f in sorted(pred_dir.glob("*__holdout.csv")):
        df = pd.read_csv(f)
        assert list(df.columns) == SCHEMA, f"{f.name}: 열 순서가 스키마와 다름"
        assert set(df["row_id"]) == hold_ids, f"{f.name}: row_id가 홀드아웃과 다름"
        assert (df["fold_rep"] == -1).all() and (df["fold"] == -1).all()
    print(f"[metrics] 예측 파일 스키마·행 수 점검 통과 ({pred_dir})")


def evaluate_baselines(cfg: dict) -> pd.DataFrame:
    from src.data.split import load_folds
    pred_dir = Path(cfg["paths"]["outputs"]) / "predictions"
    out_dir = Path(cfg["paths"]["outputs"]) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    check_schema(pred_dir, load_folds(cfg), cfg["split"]["n_repeats"])
    summary = summarize_predictions(pred_dir)
    summary.to_csv(out_dir / "baseline_summary.csv", index=False)

    show = summary[summary["metric"].isin(["pr_auc", "roc_auc", "recall@0.5", "precision@0.5", "fn@0.5", "recall@10%"])]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200,
                           "display.float_format", "{:.3f}".format):
        print(show.pivot_table(index=["model", "feature_set", "split"], columns="metric", values="mean")
              [["pr_auc", "roc_auc", "recall@0.5", "precision@0.5", "fn@0.5", "recall@10%"]])
    print(f"[metrics] saved {out_dir / 'baseline_summary.csv'}")
    return summary


if __name__ == "__main__":
    evaluate_baselines(load_config())
