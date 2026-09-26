"""반복 그룹 교차검증 (누수 방지 + 임계값은 학습 fold 내부에서만 결정).

분할 단위(cv mode) 두 가지를 함께 돌려 비교한다.
- shot : LH/RH 짝(같은 샷)만 묶는다. 짝이 학습·검증에 갈라지는 누수는 막지만,
         연속된 불량 구간의 '이웃 샷'은 학습에 들어갈 수 있다 → 같은 구간 안의 성능 (낙관적)
- block: 샷을 row_id 순서대로 block_shots개씩 연속 블록으로 묶는다. 이웃 샷까지 함께 빠지므로
         '처음 보는 구간'에 대한 성능에 가깝다 (보수적)
두 결과의 차이 = 이웃 샷 유사성에 기대는 정도. 보고서 3장의 '실패 조건' 근거가 된다.

- 층화: 각 fold에 불량이 고르게 들어가게 StratifiedGroupKFold
- 임계값: 외부 검증 fold를 보지 않고, 학습 fold 내부 그룹 OOF에서 F1 최대값으로 결정
- 반복: 불량이 수십 건뿐이라 분할 운에 따라 점수가 흔들림 → 여러 seed 반복 후 평균±표준편차
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from .evaluate import best_f1_threshold, evaluate_oof
from .models import make_model


def make_groups(lab: pd.DataFrame, mode: str, block_shots: int) -> np.ndarray:
    if mode == "shot":
        return lab["shot_id"].to_numpy()
    if mode == "block":
        g = lab.groupby("shot_id").agg(first=("row_id", "min"), product=("product", "first"))
        g = g.sort_values(["product", "first"])
        rank = g.groupby("product").cumcount()
        block = g["product"].astype(str) + "_b" + (rank // block_shots).astype(str)
        return lab["shot_id"].map(block).to_numpy()
    raise ValueError(f"알 수 없는 cv mode: {mode}")


def group_splits(y, groups, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(sgkf.split(np.zeros(len(y)), y, groups))
    for tr, va in splits:
        assert set(groups[tr]).isdisjoint(groups[va]), "같은 그룹이 학습/검증에 동시에 있음 (누수)"
    return splits


def inner_threshold(model, X, y, groups, n_splits, seed) -> float:
    if y.sum() == 0 or len(np.unique(groups)) < n_splits:
        return 0.5
    oof = np.full(len(y), np.nan)
    for tr, va in group_splits(y, groups, n_splits, seed):
        if y[tr].sum() == 0:
            continue
        m = clone(model).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    ok = ~np.isnan(oof)
    return best_f1_threshold(y[ok], oof[ok]) if ok.any() else 0.5


def run_cv(lab: pd.DataFrame, feats: list[str], model_names: list[str], cfg: dict, tag: str, mode: str,
           log=print):
    cvc, seed = cfg["cv"], cfg["seed"]
    X = lab[feats].reset_index(drop=True)
    y = lab["y"].to_numpy()
    groups = make_groups(lab.reset_index(drop=True), mode, cvc.get("block_shots", 10))
    records, metrics_rows, fitted = [], [], {}
    for rep in range(cvc["n_repeats"]):
        splits = group_splits(y, groups, cvc["n_splits"], seed + rep)
        for name in model_names:
            t0 = time.time()
            oof_p = np.zeros(len(y))
            oof_pred = np.zeros(len(y), dtype=int)
            thr_list = []
            for k, (tr, va) in enumerate(splits):
                base = make_model(name, seed)
                if y[tr].sum() == 0:  # 학습 fold에 불량이 하나도 없으면 지도학습 불가 → 정보 없음 처리
                    log(f"  ! [{tag}/{mode}] repeat {rep} fold {k}: 학습 fold에 불량 없음 → {name} 예측 생략")
                    oof_p[va] = 0.0
                    continue
                thr = inner_threshold(base, X.iloc[tr], y[tr], groups[tr], cvc["inner_splits"],
                                      seed + 100 + rep)
                m = clone(base).fit(X.iloc[tr], y[tr])
                p = m.predict_proba(X.iloc[va])[:, 1]
                oof_p[va], oof_pred[va] = p, (p >= thr).astype(int)
                thr_list.append(thr)
                if rep == 0:
                    fitted[(mode, name, k)] = (m, va)
            oof = pd.DataFrame({"row": np.arange(len(y)), "y": y, "p": oof_p, "pred": oof_pred,
                                "shot_id": lab["shot_id"].to_numpy(), "group": groups,
                                "product": lab["product"].to_numpy(), "row_id": lab["row_id"].to_numpy()})
            oof["repeat"], oof["model"], oof["dataset"], oof["cv_mode"] = rep, name, tag, mode
            records.append(oof)
            thr_med = float(np.median(thr_list)) if thr_list else 0.5
            m_all = evaluate_oof(oof, cfg["inspection_budgets"])
            metrics_rows.append({"cv_mode": mode, "dataset": tag, "product": "all", "model": name,
                                 "repeat": rep, "thr_median": thr_med, **m_all})
            if tag == "pooled":  # 통합 모델도 제품별로 따로 평가
                for prod, sub in oof.groupby("product"):
                    metrics_rows.append({"cv_mode": mode, "dataset": tag, "product": prod, "model": name,
                                         "repeat": rep, "thr_median": thr_med,
                                         **evaluate_oof(sub, cfg["inspection_budgets"])})
            log(f"  [{tag}/{mode}] repeat {rep} {name:14s} shot_AP={m_all['shot_ap']:.3f} "
                f"row_F1={m_all['row_f1']:.3f} ({time.time()-t0:.1f}s)")
    return pd.concat(records, ignore_index=True), pd.DataFrame(metrics_rows), fitted
