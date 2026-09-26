"""평가 지표. 행(제품) 단위와 샷(LH/RH 짝) 단위를 모두 계산하고,
불량 유형(공정 전체 이상 vs 캐비티 국소)별 탐지율도 따로 본다."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

TYPE_KO = {"process_wide": "공정 전체 이상(양쪽 캐비티 불량)", "cavity_local": "캐비티 국소(한쪽만 불량)",
           "single_row": "짝 없는 단독 행 불량", "normal": "정상"}


def best_f1_threshold(y, p) -> float:
    """F1 최대 임계값 (동률이면 높은 쪽 = 검사량 적은 쪽)."""
    y, p = np.asarray(y), np.asarray(p)
    if y.sum() == 0:
        return 0.5
    best_t, best = 0.5, -1.0
    for t in np.unique(p)[::-1]:
        f = f1_score(y, p >= t, zero_division=0)
        if f > best + 1e-12:
            best, best_t = f, t
    return float(best_t)


def expected_topk(hit, p, k: int) -> float:
    """점수 상위 k개를 뽑을 때 hit(0/1)의 기대 개수. 경계에서 동점은 무작위로 끊는다고 보고 기댓값으로 계산.
    (행 순서로 동점을 끊으면, 불량이 파일 앞쪽에 몰린 CN7에서 동점 많은 모델이 부당하게 유리해짐)"""
    hit, p = np.asarray(hit, dtype=float), np.asarray(p, dtype=float)
    if k <= 0:
        return 0.0
    k = min(k, len(p))
    thr = np.sort(p)[::-1][k - 1]
    above, tie = p > thr, p == thr
    return float(hit[above].sum() + (k - above.sum()) * hit[tie].mean())


def recall_at_budget(y, p, budget: float) -> float:
    """상위 budget 비율만 검사할 때 잡는 불량 비율 (검사 우선순위 성능)."""
    y = np.asarray(y, dtype=float)
    if y.sum() == 0:
        return np.nan
    return expected_topk(y, p, int(np.ceil(len(y) * budget))) / y.sum()


def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except ValueError:
        return np.nan


def metric_block(y, p, pred, budgets, prefix="") -> dict:
    y, p, pred = np.asarray(y), np.asarray(p), np.asarray(pred).astype(int)
    out = {
        "ap": _safe(average_precision_score, y, p),
        "roc_auc": _safe(roc_auc_score, y, p),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "mcc": _safe(matthews_corrcoef, y, pred),
        "brier": _safe(brier_score_loss, y, np.clip(p, 0, 1)),
        "TP": int(((pred == 1) & (y == 1)).sum()), "FP": int(((pred == 1) & (y == 0)).sum()),
        "FN": int(((pred == 0) & (y == 1)).sum()), "TN": int(((pred == 0) & (y == 0)).sum()),
    }
    P = float(y.sum())
    for b in budgets:
        out[f"recall@{int(round(b*100))}%"] = recall_at_budget(y, p, b)
    for b in budgets:  # 임계값 대신 '상위 k% 검사'로 판정했을 때의 F1 (임계값 불안정에 영향받지 않음)
        k = int(np.ceil(len(y) * b))
        tp = expected_topk(y, p, k) if P else 0.0
        out[f"f1@{int(round(b*100))}%"] = 2 * tp / (k + P) if (k + P) else np.nan
    return {prefix + k: v for k, v in out.items()}


def to_shot_level(df: pd.DataFrame) -> pd.DataFrame:
    """짝 두 행은 입력이 같으므로 점수도 같다. 샷 라벨 = 짝 중 하나라도 불량."""
    return df.groupby("shot_id").agg(y=("y", "max"), p=("p", "max"), pred=("pred", "max")).reset_index()


def evaluate_oof(oof: pd.DataFrame, budgets) -> dict:
    """oof: columns y, p, pred, shot_id (한 repeat분)."""
    m = metric_block(oof.y, oof.p, oof.pred, budgets, prefix="row_")
    s = to_shot_level(oof)
    m.update(metric_block(s.y, s.p, s.pred, budgets, prefix="shot_"))
    return m


def shot_types(lab: pd.DataFrame) -> pd.Series:
    """샷별 불량 유형. process_wide = 짝 두 행 모두 불량(공정 전체 이상),
    cavity_local = 한쪽만 불량(캐비티 국소), single_row = 짝 없는 행이 불량."""
    g = lab.groupby("shot_id")["y"].agg(["size", "sum"])
    t = np.select([g["sum"] == 0, (g["sum"] == g["size"]) & (g["size"] >= 2),
                   (g["sum"] > 0) & (g["sum"] < g["size"])],
                  ["normal", "process_wide", "cavity_local"], "single_row")
    return pd.Series(t, index=g.index, name="defect_type")


def type_recall(oof: pd.DataFrame, types: pd.Series, budgets) -> dict:
    """불량 유형별 샷 단위 탐지율: 판정 임계값 기준, 그리고 상위 k% 검사 기준."""
    s = to_shot_level(oof)
    s["type"] = s.shot_id.map(types).fillna("normal")
    out = {}
    for t in ("process_wide", "cavity_local"):
        mask = s.type.eq(t).to_numpy()
        n = int(mask.sum())
        out[f"{t}_n"] = n
        out[f"{t}_recall_thr"] = float(s.loc[mask, "pred"].mean()) if n else np.nan
        for b in budgets:
            k = int(np.ceil(len(s) * b))
            out[f"{t}_recall@{int(round(b*100))}%"] = expected_topk(mask, s.p, k) / n if n else np.nan
    return out


def bootstrap_ci(oof: pd.DataFrame, n: int, seed: int, metrics=("row_ap", "row_f1", "shot_ap", "shot_f1")) -> dict:
    """샷 단위로 재표본추출한 95% 신뢰구간 (불량 수가 적어 불확실성이 크다는 것을 정직하게 표시)."""
    rng = np.random.default_rng(seed)
    shots = oof.shot_id.unique()
    idx = oof.groupby("shot_id").indices
    vals = {k: [] for k in metrics}
    for _ in range(n):
        pick = rng.choice(shots, size=len(shots), replace=True)
        rows = np.concatenate([idx[s] for s in pick])
        b = oof.iloc[rows].copy()
        b["shot_id"] = np.repeat(np.arange(len(pick)), [len(idx[s]) for s in pick])
        if b.y.sum() == 0:
            continue
        m = evaluate_oof(b, [])
        for k in metrics:
            vals[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}
