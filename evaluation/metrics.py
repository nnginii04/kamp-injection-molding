"""evaluate_model(): 모델 하나의 성능을 한 줄로.

지표: PR-AUC, ROC-AUC, Precision, Recall, F1, FN(놓친 불량), FP(헛검사) — 교차검증 반복별로 계산한 뒤 평균±편차.

쓰는 법
    from evaluation import evaluate_model
    evaluate_model("outputs/predictions/baseline_logreg.csv")            # 예측 파일
    evaluate_model(df, level="shot")                                      # LH/RH 짝을 한 샷으로 보고
    evaluate_model(y_true, y_score, budget=0.02)                          # 배열 + '상위 2% 검사' 기준 판정

판정(Precision/Recall/F1/FN/FP를 계산할 기준) 우선순위
    threshold 지정 → 점수 ≥ threshold
    budget 지정    → 점수 상위 budget 비율
    둘 다 없고 y_pred 열이 있으면 → 모델이 낸 판정
    그것도 없으면  → 점수 ≥ 0.5
PR-AUC는 average precision(AP)으로 계산한다 (run_all.py의 ap와 같은 값). 무작위 모델의 PR-AUC ≈ 불량률.
평균±편차 단위: repeat 열이 있으면 반복별(각 반복 = 데이터 전체를 한 번 예측), 없고 fold만 있으면 fold별.
fold별일 때 FN·FP·TP는 fold를 합친 개수(전체 1회분)로 준다 (count_basis 열에 표시).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ._common import as_frame, binary_counts, iter_runs, quantile_higher, run_column, to_level

METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1", "fn", "fp", "tp", "flag_rate")


def _rank_metric(fn, y, s):
    return float(fn(y, s)) if 0 < y.sum() < len(y) else np.nan


def _decide(g: pd.DataFrame, threshold, budget):
    s = g["y_score"].to_numpy()
    if threshold is not None:
        return s >= threshold, f"점수 ≥ {threshold:g}"
    if budget is not None:
        return s >= quantile_higher(s, 1 - budget), f"상위 {budget * 100:g}% 검사"
    if "y_pred" in g.columns:
        return g["y_pred"].to_numpy().astype(bool), "모델 판정(y_pred)"
    return s >= 0.5, "점수 ≥ 0.5 (y_pred 없음)"


def evaluate_model(pred, y_score=None, y_pred=None, *, threshold=None, budget=None, level="row",
                   run_col="auto", return_runs=False):
    """모델 하나의 성능 요약 (dict). return_runs=True면 (요약, 반복별 표)를 돌려준다.

    pred      : 예측 DataFrame / CSV 경로 / y_true 배열(이때 y_score 필수)
    level     : 'row'(제품 단위) 또는 'shot'(LH/RH 짝을 한 샷으로; shot_id 필요)
    run_col   : 평균±편차를 낼 단위. 'auto'면 repeat → fold → 1회 순서로 찾음
    """
    df = as_frame(pred, y_score, y_pred)
    rc = run_column(df, run_col)
    df = to_level(df, level, rc)
    rows, rule = [], ""
    for r, g in iter_runs(df, rc):
        y, s = g["y_true"].to_numpy(), g["y_score"].to_numpy()
        flag, rule = _decide(g, threshold, budget)
        rows.append({"run": r, "n": len(g), "n_pos": int(y.sum()),
                     "pr_auc": _rank_metric(average_precision_score, y, s),
                     "roc_auc": _rank_metric(roc_auc_score, y, s), **binary_counts(y, flag)})
    runs = pd.DataFrame(rows)
    # 반복(repeat)은 매번 전체 데이터를 한 번씩 예측 → 한 회분 크기. fold는 데이터를 나눠 가진 것 → 합계.
    per_pass = rc != "fold" and runs["n"].nunique() == 1
    n = int(runs["n"].iloc[0] if per_pass else runs["n"].sum())
    n_pos = int(runs["n_pos"].iloc[0] if per_pass else runs["n_pos"].sum())
    out = {"level": level, "decision": rule, "std_over": rc or "없음(1회)", "n_runs": len(runs),
           "n": n, "n_pos": n_pos, "prevalence": n_pos / n if n else np.nan,
           "count_basis": "fold 합계" if rc == "fold" else "전체 1회분(반복 평균)"}
    for k in METRICS:
        v = runs[k].astype(float)
        if rc == "fold" and k in ("fn", "fp", "tp"):  # 개수는 fold를 합쳐야 전체 1회분이 됨
            out[f"{k}_mean"], out[f"{k}_std"] = float(v.sum()), np.nan
            continue
        out[f"{k}_mean"] = float(v.mean()) if v.notna().any() else np.nan
        out[f"{k}_std"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else np.nan
    out["pr_auc_lift"] = out["pr_auc_mean"] / out["prevalence"] if out["prevalence"] else np.nan
    return (out, runs) if return_runs else out
