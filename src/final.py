"""최종 모델 학습 → 확률 보정 → 예측 파일 3종 → 검사 우선순위.

예측 파일
1) predictions_labeled_cv.csv : labeled 전 행에 대한 '교차검증 예측'(각 행을 그 행을 보지 않은 모델이 예측)
   → 정답이 있어 성능을 확인할 수 있는 테스트 예측 파일
2) predictions_holdout.csv    : (config.holdout.enabled일 때) 처음부터 떼어 둔 테스트 블록 예측
3) predictions_unlabeled.csv  : unlabeled 예측 + 분포 밖(OOD) 표시
   ※ labeled/unlabeled가 파일별로 따로 표준화된 것으로 보이면 참고용

- 최종 모델: 선택된 (시나리오, 모델)을 labeled(개발용) 전체로 재학습
- 확률 보정(Platt): CV의 OOF 점수(반복 평균) → 실제 불량 여부로 로지스틱 보정
- 판정 임계값: 선택된 CV 방식에서 fold 내부로 정한 임계값의 중앙값 (검증 데이터를 보지 않고 결정)
- 위험 등급: 상위 2% 즉시검사 / 상위 10% 우선검사 / 나머지 일반 (config로 조정)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .evaluate import evaluate_oof
from .models import make_model
from .plots import plt, savefig


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(score, y):
    s, y = _logit(score), np.asarray(y)
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(s, y)
    n_splits = int(max(2, min(5, y.sum())))
    cv_p = cross_val_predict(LogisticRegression(C=1e6, max_iter=1000), s, y,
                             cv=StratifiedKFold(n_splits, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]
    return cal, cv_p


def apply_platt(cal, score):
    return cal.predict_proba(_logit(score))[:, 1]


def reliability_plot(y, p, path, title, n_bins=10):
    """예측확률 순서로 같은 행 수의 구간을 만들어 '예측 확률 평균 vs 실제 불량률' 비교.
    (동점이 많아도 구간이 무너지지 않게 순위 기준으로 나눔)"""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, n_bins)
    mp = np.array([p[c].mean() for c in chunks])
    ob = np.array([y[c].mean() for c in chunks])
    nd = [int(y[c].sum()) for c in chunks]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    top = max(mp.max(), ob.max()) * 1.1 + 1e-6
    ax.plot([0, top], [0, top], "--", color="gray", lw=1)
    ax.plot(mp, ob, "o-", color="#3b6fb6")
    for x_, y_, n_ in zip(mp, ob, nd):
        if n_:
            ax.annotate(str(n_), (x_, y_), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set(title=title, xlabel="mean predicted defect prob (per decile)", ylabel="observed defect rate")
    savefig(fig, path)


def out_of_range(train: pd.DataFrame, target: pd.DataFrame, feats: list[str]) -> np.ndarray:
    lo, hi = train[feats].min(), train[feats].max()
    return ((target[feats] < lo) | (target[feats] > hi)).sum(axis=1).to_numpy()


def train_and_predict(product, choice, frames, feats_by, oof_all, metrics, cfg, fig_dir, types,
                      test_frame: pd.DataFrame | None = None):
    scenario, model_name, mode = choice["scenario"], choice["model"], choice["cv_mode"]
    dataset = "pooled" if scenario == "pooled" else product
    lab, unl = frames[dataset]
    feats = feats_by[scenario]
    model = clone(make_model(model_name, cfg["seed"])).fit(lab[feats], lab["y"].to_numpy())

    # 1) labeled 교차검증 예측 + 보정
    oof = oof_all[(oof_all.dataset == dataset) & (oof_all.model == model_name)
                  & (oof_all["product"] == product) & (oof_all.cv_mode == mode)]
    om = oof.groupby("row").agg(row_id=("row_id", "first"), shot_id=("shot_id", "first"), y=("y", "first"),
                                p=("p", "mean"), vote=("pred", "mean")).reset_index(drop=True)
    cal, cv_cal_p = fit_platt(om.p.to_numpy(), om.y.to_numpy())
    # 판정 기준. v2 실제 결과: 블록 CV에서는 학습 fold 안의 불량 구간이 2~3개뿐이라 F1 최적 임계값이 흔들려
    # (예: CN7 logreg가 labeled의 39%, unlabeled의 45%를 불량으로 판정) → 기본값은 '검사 비율' 기준.
    # 교차검증 점수 분포의 상위 budget 분위수를 점수 임계값으로 삼는다 (실시간으로 샷마다 바로 판정 가능).
    dcfg = cfg.get("decision", {"rule": "f1"})
    rt = cfg["risk_tiers"]
    if dcfg.get("rule") == "budget":
        thr = float(np.quantile(om.p, 1 - dcfg.get("budget", 0.02)))
        rule_txt = f"교차검증 점수 상위 {dcfg.get('budget', 0.02):.0%} 분위수"
        cv_pred = (om.p >= thr).astype(int)
    else:
        thr = float(metrics[(metrics.cv_mode == mode) & (metrics.dataset == dataset) & (metrics.model == model_name)
                            & (metrics["product"].isin([product, "all"]))].thr_median.median())
        rule_txt = "학습 fold 내부 F1 최적 임계값의 중앙값"
        cv_pred = (om.vote >= 0.5).astype(int)
    t_now = float(np.quantile(om.p, 1 - rt["inspect_now_top"]))
    t_pri = float(np.quantile(om.p, 1 - rt["priority_top"]))
    lab_pred = pd.DataFrame({"product": product, "row_id": om.row_id, "shot_id": om.shot_id,
                             "defect_type": om.shot_id.map(types).fillna("normal"), "y": om.y,
                             "cv_risk_score": om.p, "cv_defect_prob": cv_cal_p,
                             "cv_pred_label": cv_pred}).sort_values("row_id")
    reliability_plot(om.y, cv_cal_p, fig_dir / f"calibration_{product}.png",
                     f"{product.upper()} calibration (cross-validated)")

    def _score(frame):
        s = model.predict_proba(frame[feats])[:, 1]
        return s, apply_platt(cal, s)

    # 2) 홀드아웃 테스트
    test_pred, test_metrics = None, {}
    if test_frame is not None and len(test_frame):
        s, prob = _score(test_frame)
        test_pred = pd.DataFrame({"product": product, "row_id": test_frame.row_id.to_numpy(),
                                  "shot_id": test_frame.shot_id.to_numpy(), "y": test_frame.y.to_numpy(),
                                  "risk_score": s, "defect_prob": prob, "pred_label": (s >= thr).astype(int)})
        ev = pd.DataFrame({"y": test_pred.y, "p": s, "pred": test_pred.pred_label, "shot_id": test_pred.shot_id})
        test_metrics = {f"test_{k}": v for k, v in evaluate_oof(ev, cfg["inspection_budgets"]).items()
                        if k in ("row_ap", "row_f1", "row_precision", "row_recall", "shot_ap", "shot_f1",
                                 "row_TP", "row_FP", "row_FN", "row_recall@5%", "row_recall@10%")}
        test_metrics["test_rows"], test_metrics["test_defects"] = len(test_pred), int(test_pred.y.sum())

    # 3) unlabeled 예측 + 분포 밖 표시
    target = unl[unl["product"] == product].copy()
    s, prob = _score(target)
    target["risk_score"], target["defect_prob"] = s, prob
    target["pred_label"] = (s >= thr).astype(int)
    target["priority_rank"] = target["risk_score"].rank(ascending=False, method="first").astype(int)
    # 등급도 학습(교차검증) 점수 분포 기준 임계값으로 → 새 데이터의 분포가 달라지면 등급 비율도 달라짐(드리프트 신호)
    target["risk_tier"] = np.where(s >= t_now, "즉시검사", np.where(s >= t_pri, "우선검사", "일반"))
    base_feats = [c for c in feats if c != "is_rg3"]
    target["out_of_range_features"] = out_of_range(lab[lab["product"] == product], target, base_feats)
    target["in_labeled_range"] = target["out_of_range_features"].eq(0)

    base_rate = float(om.y.mean())
    summary = {"product": product, "scenario": scenario, "model": model_name, "cv_mode": mode,
               "decision_rule": rule_txt, "threshold_on_score": thr,
               "labeled_cv_flag_rate": float(cv_pred.mean()),
               "unlabeled_share_즉시검사": float((target.risk_tier == "즉시검사").mean()),
               "unlabeled_share_우선검사": float((target.risk_tier == "우선검사").mean()),
               "calibrated_brier_cv": float(np.mean((cv_cal_p - om.y) ** 2)),
               "brier_if_constant_rate": base_rate * (1 - base_rate),
               "unlabeled_rows": len(target), "unlabeled_pred_positive_rate": float(target.pred_label.mean()),
               "unlabeled_mean_defect_prob": float(target.defect_prob.mean()),
               "labeled_defect_rate": base_rate,
               "unlabeled_in_labeled_range_share": float(target.in_labeled_range.mean()),
               **test_metrics}
    cols = ["product", "row_id", "defect_prob", "pred_label", "priority_rank", "risk_tier", "risk_score",
            "out_of_range_features", "in_labeled_range"]
    return target[cols].sort_values("row_id"), lab_pred, test_pred, summary, model
