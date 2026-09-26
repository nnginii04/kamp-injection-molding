"""evaluate_thresholds(): 판정 기준(임계값)을 바꿔 가며 F1/Recall/Precision/FN/FP를 보는 표.

두 종류의 기준을 함께 계산한다.
  rule='threshold' : 점수 ≥ 0.05, 0.10, …, 0.95 (고정 임계값)
  rule='budget'    : 점수 상위 1%, 2%, 5%, 10% (검사 예산 기준. 점수가 확률이 아니어도 의미가 있음)

읽는 법 (중요)
- 불량이 1~2%뿐이라 확률로 보정된 점수는 대부분 0.05보다 작다 → 고정 임계값 대부분에서 아무것도 안 찍혀
  Recall 0이 나오는 게 정상이다. 그래서 flag_rate(찍은 비율) 열을 꼭 같이 본다.
- best_f1은 '이 표를 만든 데이터로 고른' 임계값이라 낙관적이다 → 참고용.
  최종 판정 기준은 교차검증 안에서 정한 규칙(run_all.py의 '교차검증 점수 상위 2%')을 쓴다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._common import as_frame, binary_counts, iter_runs, pct, quantile_higher, run_column, to_level

DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.05, 0.951, 0.05), 2))
DEFAULT_BUDGETS = (0.01, 0.02, 0.05, 0.10)


def evaluate_thresholds(pred, y_score=None, *, thresholds=DEFAULT_THRESHOLDS, budgets=DEFAULT_BUDGETS,
                        level="row", run_col="auto"):
    """기준별 한 줄 표. 열: rule, label, threshold(실제 점수 기준), flag_rate, precision, recall_mean/std,
    f1_mean/std, fn_mean, fp_mean, tp_mean, n_runs, best_f1."""
    thresholds = [] if thresholds is None else [float(t) for t in thresholds]
    budgets = [] if budgets is None else [float(b) for b in budgets]
    df = as_frame(pred, y_score)
    rc = run_column(df, run_col)
    df = to_level(df, level, rc)
    if not df["y_score"].between(0, 1).all():
        warnings.warn("y_score가 0~1 범위를 벗어납니다(확률이 아님) → 고정 임계값보다 '상위 k%' 기준(rule=budget)을 보세요.")
    rows = []
    for r, g in iter_runs(df, rc):
        y, s = g["y_true"].to_numpy(), g["y_score"].to_numpy()
        for t in thresholds:
            rows.append({"run": r, "rule": "threshold", "value": t, "threshold": t,
                         **binary_counts(y, s >= t)})
        for b in budgets:
            thr = quantile_higher(s, 1 - b)
            rows.append({"run": r, "rule": "budget", "value": b, "threshold": thr,
                         **binary_counts(y, s >= thr)})
    runs = pd.DataFrame(rows)
    out = runs.groupby(["rule", "value"], sort=False).agg(
        threshold=("threshold", "mean"), flag_rate=("flag_rate", "mean"), precision=("precision", "mean"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"), f1_mean=("f1", "mean"),
        f1_std=("f1", "std"), fn_mean=("fn", "mean"), fp_mean=("fp", "mean"), tp_mean=("tp", "mean"),
        n_runs=("run", "nunique")).reset_index()
    out.insert(2, "label", [f"점수 ≥ {v:g}" if r == "threshold" else f"상위 {pct(v)} 검사"
                            for r, v in zip(out["rule"], out["value"])])
    out.insert(0, "level", level)
    out["best_f1"] = False
    thr_rows = out[out["rule"].eq("threshold") & out["f1_mean"].gt(0)]
    if len(thr_rows):
        out.loc[thr_rows["f1_mean"].idxmax(), "best_f1"] = True
    return out
