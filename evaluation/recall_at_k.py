"""calculate_recall_at_k(): '점수 상위 K%만 검사하면 불량을 몇 % 잡는가'.

- Random 기준선: 무작위로 K%를 검사하면 기대 재현율 = 검사 비율(≈K%). lift = 모델 재현율 / 무작위 재현율.
- 점수 동점은 무작위로 끊었을 때의 기댓값으로 계산 (파일 순서로 자르면 결과가 부풀려질 수 있음).
- level='shot'이고 defect_type 열이 있으면 불량 유형별(공정 전체 이상 / 캐비티 국소) 재현율도 함께 낸다.

쓰는 법
    from evaluation import calculate_recall_at_k
    calculate_recall_at_k("outputs/predictions/baseline_logreg.csv")                 # 5/10/20/30%
    calculate_recall_at_k(df, ks=(0.01, 0.02, 0.05, 0.10), level="shot")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import as_frame, expected_hits, iter_runs, pct, run_column, to_level

DEFAULT_KS = (0.05, 0.10, 0.20, 0.30)
TYPES = ("process_wide", "cavity_local")


def calculate_recall_at_k(pred, y_score=None, *, ks=DEFAULT_KS, level="row", run_col="auto",
                          return_runs=False):
    """K별 한 줄 표. 열: k, n_inspect(검사 개수), recall_mean/std, precision_mean, random_recall, lift,
    caught_mean(잡은 불량), missed_mean(놓친 불량), n_pos, n_runs (+ 유형별 재현율)."""
    df = as_frame(pred, y_score)
    rc = run_column(df, run_col)
    df = to_level(df, level, rc)
    by_type = level == "shot" and "defect_type" in df.columns
    rows = []
    for r, g in iter_runs(df, rc):
        y, s = g["y_true"].to_numpy(float), g["y_score"].to_numpy(float)
        n, P = len(y), float(y.sum())
        for k in ks:
            m = int(np.ceil(n * k))
            tp = expected_hits(y, s, m)
            row = {"run": r, "k": k, "n": n, "n_pos": P, "n_inspect": m, "caught": tp, "missed": P - tp,
                   "recall": tp / P if P else np.nan, "precision": tp / m if m else np.nan,
                   "random_recall": m / n if n else np.nan, "random_precision": P / n if n else np.nan}
            if by_type:
                for t in TYPES:
                    mask = (g["defect_type"] == t).to_numpy()
                    row[f"recall_{t}"] = expected_hits(mask, s, m) / mask.sum() if mask.sum() else np.nan
            rows.append(row)
    runs = pd.DataFrame(rows)
    spec = {"n_inspect": ("n_inspect", "mean"), "recall_mean": ("recall", "mean"), "recall_std": ("recall", "std"),
            "precision_mean": ("precision", "mean"), "random_recall": ("random_recall", "mean"),
            "random_precision": ("random_precision", "mean"), "caught_mean": ("caught", "mean"),
            "missed_mean": ("missed", "mean"), "n_pos": ("n_pos", "mean"), "n_runs": ("run", "nunique")}
    if by_type:
        spec.update({f"recall_{t}_mean": (f"recall_{t}", "mean") for t in TYPES})
    out = runs.groupby("k", sort=True).agg(**spec).reset_index()
    out["lift"] = out["recall_mean"] / out["random_recall"]
    out.insert(0, "k_label", out["k"].map(pct))
    out.insert(0, "level", level)
    return (out, runs) if return_runs else out
