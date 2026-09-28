"""추가 실험 (Phase 3) — 본 파이프라인과 같은 분할(cv_splits = outputs/fold_id.csv), pooled OOF 기준.

지표 읽는 법
- pooled OOF: 반복(repeat)마다 fold 들의 검증 예측을 이어 붙여 한 번에 계산 → 반복 평균(±표준편차). 본 지표.
- fold 평균: fold 마다 따로 계산한 AP 의 평균. 참고용 (불량이 fold 에 몇 개뿐이라 크게 흔들림).
- 행 단위 상한(row_P_ceiling / row_F1_ceiling): 짝 두 행은 입력이 같아 예측도 같으므로 넘을 수 없는 값.

실험
1) 제품별(A) vs 통합(B, 제품 구분 변수 is_rg3): 통합 모델도 제품별 fold k 를 합친 같은 분할. 제품별 성능 분리.
   + 도메인 분류기 AUC (CN7 과 RG3 를 구분할 수 있는가; 트리/선형)
2) class_weight None vs balanced
3) 기준선 3종: dummy(prior) / most_frequent / stratified
4) 최종 모델 임계값 trade-off (0.10~0.90) + 검사비율(budget) 표 — shot·block CV OOF 둘 다
5) 이전 베이스라인(archive/baseline_v0) 수치와의 비교 사다리
+ 성능 범위(shot CV 낙관 ~ block CV 비관), holdout 설계 시뮬레이션(쓰지 않은 이유), 가상 모델 선택

본 모델 선택(config.selection)은 바꾸지 않는다. 실험 모델까지 후보에 넣으면 선택이 달라지는지는 hypothetical_selection 으로 보고만 한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cv import cv_splits, group_splits, inner_threshold, make_groups, run_cv
from .evaluate import evaluate_oof, expected_topk
from .models import BASE_MODEL, COMPLEXITY, DeterministicRF, make_model

KEYS = ["cv_mode", "product", "model"]
VALS = ["row_ap", "shot_ap", "row_roc_auc", "row_recall@10%", "row_f1@2%"]


def _quiet(*_):
    return None


def ceilings(audit: dict) -> dict:
    return {p: {"row_P_ceiling": a["row_precision_ceiling"], "row_F1_ceiling": a["row_f1_ceiling"]}
            for p, a in audit.items()}


def add_ceilings(t: pd.DataFrame, ceil: dict) -> pd.DataFrame:
    t = t.copy()
    for k in ("row_P_ceiling", "row_F1_ceiling"):
        t[k] = t["product"].map(lambda p: ceil.get(p, {}).get(k, np.nan))
    return t


def fold_mean(oof: pd.DataFrame) -> pd.DataFrame:
    """fold 별 AP 의 평균 (참고). 검증 fold 에 불량이 없으면 AP 를 정의할 수 없어 제외."""
    rows = []
    for (mode, prod, model, _, fold), g in oof.groupby(["cv_mode", "product", "model", "repeat", "fold"]):
        if fold < 0 or g.y.sum() == 0:
            continue
        s = g.groupby("shot_id").agg(y=("y", "max"), p=("p", "max"))
        rows.append({"cv_mode": mode, "product": prod, "model": model,
                     "row_ap_foldmean": average_precision_score(g.y, g.p),
                     "shot_ap_foldmean": average_precision_score(s.y, s.p)})
    if not rows:
        return pd.DataFrame(columns=KEYS + ["row_ap_foldmean", "shot_ap_foldmean"])
    return pd.DataFrame(rows).groupby(KEYS).mean().reset_index()


def summarize(metrics: pd.DataFrame, oof: pd.DataFrame, ceil: dict) -> pd.DataFrame:
    """반복별 pooled OOF 지표 → 평균·표준편차, fold 평균(참고), 행 단위 상한."""
    g = metrics.groupby(KEYS)
    t = g[VALS].mean()
    t["row_ap_sd"] = g["row_ap"].std()
    t["shot_ap_sd"] = g["shot_ap"].std()
    t["n_repeats"] = g.size()
    t = t.reset_index().merge(fold_mean(oof), on=KEYS, how="left")
    order = KEYS + ["row_ap", "row_ap_sd", "shot_ap", "shot_ap_sd", "row_roc_auc", "row_recall@10%", "row_f1@2%",
                    "row_ap_foldmean", "shot_ap_foldmean", "n_repeats"]
    return add_ceilings(t[order], ceil)


def _run_separate(frames, products, feats, models, cfg, cfg_override=None):
    c = cfg_override or cfg
    oofs, mets = [], []
    for p in products:
        for mode in c["cv"]["modes"]:
            o, m, _ = run_cv(frames[p][0], feats, models, c, p, mode, log=_quiet)
            m["product"] = p
            oofs.append(o), mets.append(m)
    return pd.concat(oofs, ignore_index=True), pd.concat(mets, ignore_index=True)


def _main_subset(metrics, oof_all, models):
    m = metrics[metrics.scenario.eq("separate") & metrics.model.isin(models)]
    o = oof_all[oof_all.dataset.ne("pooled") & oof_all.model.isin(models)]
    return m, o


# 1) 제품별 vs 통합 ---------------------------------------------------------------------------
def pooled_frame(frames, products):
    labs = [frames[p][0].reset_index(drop=True) for p in products]
    lab = pd.concat(labs, ignore_index=True)
    lab["is_rg3"] = (lab["product"] == "rg3").astype(int)
    offsets = np.cumsum([0] + [len(x) for x in labs[:-1]])
    return lab, labs, offsets


def pooled_splits_fn(labs, offsets, mode, cfg):
    """통합 모델의 fold k = 제품별 fold k 의 합집합 → fold_id.csv 와 같은 분할."""
    def fn(rep):
        per = [cv_splits(x, mode, cfg, rep) for x in labs]
        return [(np.concatenate([per[i][k][0] + offsets[i] for i in range(len(labs))]),
                 np.concatenate([per[i][k][1] + offsets[i] for i in range(len(labs))]))
                for k in range(len(per[0]))]
    return fn


def exp_separate_vs_pooled(frames, products, feats, metrics, oof_all, cfg, ceil):
    models = [m for m in cfg["experiments"]["ab_models"] if m in cfg["models"]]
    lab, labs, offsets = pooled_frame(frames, products)
    oofs, mets = [], []
    for mode in cfg["cv"]["modes"]:
        o, m, _ = run_cv(lab, feats + ["is_rg3"], models, cfg, "pooled", mode, log=_quiet,
                         splits_fn=pooled_splits_fn(labs, offsets, mode, cfg))
        oofs.append(o), mets.append(m[m["product"] != "all"])
    b = summarize(pd.concat(mets, ignore_index=True), pd.concat(oofs, ignore_index=True), ceil)
    a = summarize(*_main_subset(metrics, oof_all, models), ceil)
    t = pd.concat([a.assign(experiment="A 제품별"), b.assign(experiment="B 통합+is_rg3")], ignore_index=True)
    return t[["experiment"] + [c for c in t.columns if c != "experiment"]], pd.concat(mets, ignore_index=True)


def domain_classifier_auc(frames, products, feats, seed) -> pd.DataFrame:
    """CN7 vs RG3 를 공정변수로 구분할 수 있는가 (5-fold, shot 단위 그룹). 1.0 이면 완전히 다른 분포."""
    lab = pd.concat([frames[p][0] for p in products], ignore_index=True)
    X, y = lab[feats].to_numpy(float), (lab["product"] == products[-1]).astype(int).to_numpy()
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, lab["shot_id"]))
    models = {"트리 (HistGradientBoosting)": HistGradientBoostingClassifier(max_depth=3, max_iter=150,
                                                                            random_state=seed),
              "트리 (RandomForest)": DeterministicRF(n_estimators=300, random_state=seed, n_jobs=-1),
              "선형 (LogisticRegression)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))}
    rows = []
    for name, m in models.items():
        p = cross_val_predict(m, X, y, cv=splits, method="predict_proba")[:, 1]
        rows.append({"domain_classifier": name, "roc_auc": roc_auc_score(y, p)})
    return pd.DataFrame(rows)


# 1b) 교차 금형 ------------------------------------------------------------------------------
def exp_cross_mold(frames, products, feats, metrics, cfg, ceil) -> pd.DataFrame:
    """한 금형의 labeled 전체로 학습 → 다른 금형의 labeled 전체를 채점 (CN7→RG3, RG3→CN7).

    대회 파일은 금형별로 따로 표준화돼 있으므로, 이 실험은 '설정값 차이를 지운 뒤 변동 패턴이 옮겨가는가'를 본다
    (원단위에서는 두 금형이 완전히 다른 운전 영역 — scripts/transfer_risk.py). 임계값은 학습 금형 안에서만 정한다
    (inner_threshold, shot 단위). 비교 기준 = 대상 금형 자체의 교차검증(같은 모델, 반복 평균)과 dummy."""
    ecfg = cfg["experiments"]
    models = [m for m in ecfg.get("cross_models", ecfg["ab_models"]) if m in cfg["models"]]
    budgets, seed, cvc = cfg["inspection_budgets"], cfg["seed"], cfg["cv"]
    own = metrics[metrics.scenario.eq("separate")] if "scenario" in metrics.columns else metrics  # dataset 열은 제품명
    rows = []
    for src in products:
        for dst in products:
            if src == dst:
                continue
            ls, ld = frames[src][0].reset_index(drop=True), frames[dst][0].reset_index(drop=True)
            Xs, ys = ls[feats], ls["y"].to_numpy()
            Xd, yd = ld[feats], ld["y"].to_numpy()
            gs = make_groups(ls, "shot", cvc.get("block_shots", 10))
            for name in models:
                thr = inner_threshold(make_model(name, seed), Xs, ys, gs, cvc["inner_splits"], seed + 100)
                m = make_model(name, seed).fit(Xs, ys)
                p = m.predict_proba(Xd)[:, 1]
                oof = pd.DataFrame({"y": yd, "p": p, "pred": (p >= thr).astype(int), "shot_id": ld["shot_id"].to_numpy()})
                met = evaluate_oof(oof, budgets)
                row = {"direction": f"{src}→{dst}", "train": src, "test": dst, "model": name, "threshold": thr,
                       "n_train_defect_rows": int(ys.sum()), "n_test_defect_rows": int(yd.sum())}
                for k in ["shot_ap", "row_ap", "shot_roc_auc", "row_roc_auc"] + \
                         [f"shot_recall@{int(round(b * 100))}%" for b in budgets] + ["shot_TP", "shot_FP", "shot_FN"]:
                    row[k] = met.get(k, np.nan)
                for mode in cvc["modes"]:  # 대상 금형이 '자기 데이터'로 얻는 성능 (기준)
                    o = own[own["product"].eq(dst) & own.model.eq(name) & own.cv_mode.eq(mode)]
                    row[f"own_cv_{mode}_shot_ap"] = float(o["shot_ap"].mean()) if len(o) else np.nan
                d = own[own["product"].eq(dst) & own.model.eq("dummy") & own.cv_mode.eq("shot")]
                row["dummy_shot_ap"] = float(d["shot_ap"].mean()) if len(d) else np.nan
                row["lift_vs_dummy"] = row["shot_ap"] / row["dummy_shot_ap"] if row["dummy_shot_ap"] else np.nan
                rows.append(row)
    t = pd.DataFrame(rows)
    return add_ceilings(t.rename(columns={"test": "product"}), ceil).rename(columns={"product": "test"}) \
        if len(t) else t


# 2) class_weight / 3) 기준선 -----------------------------------------------------------------
def exp_class_weight(frames, products, feats, metrics, oof_all, cfg, ceil):
    base = [m for m in cfg["experiments"]["class_weight_models"] if m in cfg["models"]]
    none = [f"{m}_none" for m in base]
    o, m = _run_separate(frames, products, feats, none, cfg)
    t_none = summarize(m, o, ceil)
    t_bal = summarize(*_main_subset(metrics, oof_all, base), ceil)
    t_none["class_weight"], t_none["model"] = "none", t_none["model"].map(lambda x: BASE_MODEL.get(x, x))
    t_bal["class_weight"] = t_bal["model"].map(lambda x: "balanced_subsample" if x == "rf" else "balanced")
    t = pd.concat([t_bal, t_none], ignore_index=True)
    return t[["class_weight"] + [c for c in t.columns if c != "class_weight"]], m, o


def exp_dummies(frames, products, feats, metrics, oof_all, cfg, ceil):
    extra = list(cfg["experiments"]["dummies"])
    o, m = _run_separate(frames, products, feats, extra, cfg)
    t = pd.concat([summarize(*_main_subset(metrics, oof_all, ["dummy"]), ceil), summarize(m, o, ceil)],
                  ignore_index=True)
    t["model"] = t["model"].replace({"dummy": "dummy (prior)"})
    return t


# 4) 임계값 trade-off / 검사비율 ----------------------------------------------------------------
def threshold_table(oof_sel: pd.DataFrame, grid) -> pd.DataFrame:
    """고정 임계값별 판정 결과. 개수는 반복당 평균, precision/recall 은 반복 합계로 계산."""
    rows = []
    reps = oof_sel.groupby("repeat")
    for t in grid:
        c = np.zeros(4)
        for _, g in reps:
            pred, y = g.p.to_numpy() >= t, g.y.to_numpy() == 1
            c += [(pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum()]
        tp, fp, fn, tn = c
        n = reps.ngroups
        rows.append({"threshold": round(float(t), 2), "TP": tp / n, "FP": fp / n, "FN": fn / n, "TN": tn / n,
                     "검사건수": (tp + fp) / n, "precision": tp / (tp + fp) if tp + fp else np.nan,
                     "recall": tp / (tp + fn) if tp + fn else np.nan})
    return pd.DataFrame(rows)


def budget_table(oof_sel: pd.DataFrame, budgets) -> pd.DataFrame:
    """점수 상위 b 비율만 검사할 때 (동점은 무작위로 끊었을 때의 기댓값). 반복당 평균."""
    rows = []
    for b in budgets:
        acc = []
        for _, g in oof_sel.groupby("repeat"):
            k = int(np.ceil(len(g) * b))
            tp = expected_topk(g.y.to_numpy(), g.p.to_numpy(), k)
            P = g.y.sum()
            acc.append((k, tp, k - tp, P - tp))
        k, tp, fp, fn = np.mean(acc, axis=0)
        rows.append({"budget": f"{b:.0%}", "검사건수": k, "TP": tp, "FP": fp, "FN": fn,
                     "precision": tp / k if k else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan})
    return pd.DataFrame(rows)


def exp_thresholds(oof_all, choice, product, cfg, ceil):
    g = cfg["experiments"]["threshold_grid"]
    grid = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 2)
    thr, bud = [], []
    for mode in cfg["cv"]["modes"]:
        o = oof_all[(oof_all.dataset == choice["dataset"]) & (oof_all.model == choice["model"])
                    & (oof_all["product"] == product) & (oof_all.cv_mode == mode)]
        thr.append(threshold_table(o, grid).assign(cv_mode=mode))
        bud.append(budget_table(o, cfg["inspection_budgets"]).assign(cv_mode=mode))
    lead = lambda t: t[["cv_mode"] + [c for c in t.columns if c != "cv_mode"]]  # noqa: E731
    out = {}
    for k, v in (("threshold", thr), ("budget", bud)):
        t = lead(pd.concat(v, ignore_index=True)).assign(product=product, model=choice["model"])
        out[k] = add_ceilings(t, ceil)
    return out


# 성능 범위 / holdout 시뮬레이션 --------------------------------------------------------------
def perf_range(table: pd.DataFrame, choices: dict, ceil: dict) -> pd.DataFrame:
    """선택된 모델의 shot CV(낙관) ~ block CV(비관) 범위."""
    rows = []
    for p, c in choices.items():
        t = table[(table["product"] == p) & (table.model == c["model"]) & (table.dataset == c["dataset"])]
        r = {"product": p, "model": c["model"]}
        for v in ("row_ap", "shot_ap", "row_roc_auc", "row_recall@10%", "row_f1@2%"):
            vals = t.set_index("cv_mode")[v]
            r[f"{v} (block~shot)"] = f"{vals.get('block', np.nan):.3f} ~ {vals.get('shot', np.nan):.3f}"
        rows.append(r)
    return add_ceilings(pd.DataFrame(rows), ceil)


def holdout_designs(frames, products, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """holdout 후보 설계별로 holdout 에 들어가는 불량 수 (무작위 설계는 seed 여러 개).
    반환: (설계별 요약표, 시행별 원자료 — 그림용)"""
    hc = cfg["experiments"]["holdout_sim"]
    frac, seeds, k = hc["frac"], range(hc["seeds"]), cfg["cv"].get("block_shots", 10)
    rows, trials = [], []
    for p in products:
        lab = frames[p][0].reset_index(drop=True)
        y = lab.y.to_numpy()
        blk = make_groups(lab, "block", k)
        order = lab.groupby("shot_id").row_id.min().sort_values()
        sidx = lab.shot_id.map(pd.Series(np.arange(len(order)), index=order.index)).to_numpy()
        n_sh, width = len(order), int(len(order) * frac)
        n_split = max(2, int(round(1 / frac)))

        def rec(design, te_list):
            d = np.array([int(y[te].sum()) for te in te_list])
            s = np.array([lab.shot_id.iloc[te][y[te] == 1].nunique() for te in te_list])
            trials.extend({"mold": p, "design": design, "trial": i, "holdout_defect_rows": int(v)}
                          for i, v in enumerate(d))
            rows.append({"mold": p, "design": design, "trials": len(te_list),
                         "holdout_rows": int(np.mean([len(t) for t in te_list])),
                         "holdout_defect_rows(min/med/max)": f"{d.min()}/{int(np.median(d))}/{d.max()}",
                         "holdout_defect_shots(min/med/max)": f"{s.min()}/{int(np.median(s))}/{s.max()}",
                         "P(holdout 불량 0)": float((d == 0).mean()), "dev_defect_rows_min": int(y.sum() - d.max())})
        rec("A. 시간순 마지막 25% (연속)", [np.where(sidx >= n_sh - width)[0]])
        rec(f"A'. 연속 25% 구간, 시작 위치 전부({k}샷 간격)",
            [np.where((sidx >= s0) & (sidx < s0 + width))[0] for s0 in range(0, n_sh - width + 1, k)])
        rec("B. 블록 단위 층화 25%",
            [next(StratifiedGroupKFold(n_split, shuffle=True, random_state=sd).split(y, y, blk))[1] for sd in seeds])
        rec("C. 블록 무작위 25% (층화 없음)",
            [next(GroupShuffleSplit(1, test_size=frac, random_state=sd).split(y, y, blk))[1] for sd in seeds])
        rec("D. shot 단위 층화 25% (참고: 이웃 샷 차단 없음)",
            [next(StratifiedGroupKFold(n_split, shuffle=True, random_state=sd).split(y, y, lab.shot_id))[1]
             for sd in seeds])
    return pd.DataFrame(rows), pd.DataFrame(trials)


# 5) 이전 베이스라인 비교 ---------------------------------------------------------------------
def baseline_v0_ladder(frames, products, feats, metrics, oof_all, cfg, ceil, root: Path, synthetic: bool):
    """archive/baseline_v0 의 수치(5-fold 1회, 블록 없음)에서 이번 수치로 한 단계씩 조건을 바꿔 본다."""
    bc = cfg["experiments"]["baseline_v0"]
    arch = None
    f = root / bc["archive_csv"]
    if f.exists() and not synthetic:
        arch = pd.read_csv(f)
    rows = []
    for p in products:
        lab = frames[p][0].reset_index(drop=True)
        y = lab.y.to_numpy()

        def run(label, change, model, mode, n_splits, n_repeats, splits_fn=None, groups=None):
            c = json.loads(json.dumps(cfg))
            c["cv"]["n_splits"], c["cv"]["n_repeats"] = n_splits, n_repeats
            o, m, _ = run_cv(lab, feats, [model], c, p, mode, log=_quiet, splits_fn=splits_fn, groups=groups)
            m["product"] = p
            s = summarize(m, o, ceil).iloc[0]
            rows.append({"product": p, "step": label, "change": change, "model": model, "cv": f"{mode} {n_splits}-fold",
                         "repeats": n_repeats, "row_ap": s.row_ap, "row_ap_sd": s.row_ap_sd, "shot_ap": s.shot_ap,
                         "row_ap_foldmean": s.row_ap_foldmean})

        if arch is not None:
            r = arch[(arch.experiment == "A") & (arch.dataset == p.upper()) & (arch.model == "LogisticRegression")
                     & (arch["class_weight"].isna() | arch["class_weight"].astype(str).eq("None"))]
            if len(r):
                rows.append({"product": p, "step": "0", "change": "archive 보고값 (baseline_v0, 5-fold 1회, 블록 없음)",
                             "model": "LogisticRegression(C=1, 가중치 없음)", "cv": "shot 5-fold", "repeats": 1,
                             "row_ap": float(r.pr_auc.iloc[0]), "row_ap_sd": np.nan, "shot_ap": np.nan,
                             "row_ap_foldmean": np.nan})
        # 1) 같은 설정 재현: 그룹 번호를 v0 처럼 '첫 등장 순서 정수'로 → StratifiedGroupKFold 분할까지 동일
        codes = pd.factorize(lab.shot_id)[0]
        run("1", "v0 설정 재현 (v0 와 같은 그룹 번호·seed)", "logreg_v0", "shot", 5, 1,
            splits_fn=lambda rep: group_splits(y, codes, 5, cfg["seed"] + rep), groups=codes)
        n_rep = cfg["cv"]["n_repeats"]
        run("2", f"분할 운 반영: 5-fold × {n_rep}회 (이번 파이프라인 분할)", "logreg_v0", "shot", 5, n_rep)
        run("3", "3-fold 로 변경 (블록 CV 에서 불량 0 fold 방지)", "logreg_v0", "shot", 3, n_rep)
        run("4", "block CV (연속 10샷 묶음, 이웃 샷 차단)", "logreg_v0", "block", 3, n_rep)
        for mode, step in (("shot", "5"), ("block", "6")):
            mm = metrics[metrics.scenario.eq("separate") & metrics.model.eq("logreg") & metrics["product"].eq(p)
                         & metrics.cv_mode.eq(mode)]
            oo = oof_all[oof_all.dataset.eq(p) & oof_all.model.eq("logreg") & oof_all.cv_mode.eq(mode)]
            if len(mm):
                s = summarize(mm, oo, ceil).iloc[0]
                rows.append({"product": p, "step": step, "change": f"이번 logreg (C=0.3, balanced), {mode} CV",
                             "model": "logreg", "cv": f"{mode} {cfg['cv']['n_splits']}-fold", "repeats": n_rep,
                             "row_ap": s.row_ap, "row_ap_sd": s.row_ap_sd, "shot_ap": s.shot_ap,
                             "row_ap_foldmean": s.row_ap_foldmean})
    return add_ceilings(pd.DataFrame(rows), ceil)


# 후보 코드(02) 누수 재현 ------------------------------------------------------------------------
def candidate02_leak(frames, products, feats, cfg) -> pd.DataFrame:
    """팀 후보 코드(02)의 설정을 그대로 재현해 누수 두 가지의 영향을 본다 (코드 원본은 저장소에 포함하지 않음).
    02 설정: StandardScaler + LogisticRegression(liblinear, balanced), StratifiedGroupKFold(5, seed 42),
             shot_id = '연속해서 공정값이 같은 행'만 묶음, X 에 position_in_shot(짝 안 기록 순서) 포함.
    단계: (1) 02 그대로 → (2) position_in_shot 제거 → (3) shot_id 를 전역 기준으로. 지표는 pooled OOF, fold 평균은 참고."""
    from sklearn.model_selection import StratifiedGroupKFold as SGKF
    rows = []
    for p in products:
        lab = frames[p][0].sort_values("row_id").reset_index(drop=True)
        y = lab.y.to_numpy()
        base = [c for c in feats]
        same_prev = lab[base].eq(lab[base].shift(1)).all(axis=1)
        consec = (~same_prev).cumsum().to_numpy() - 1
        pos = pd.Series(1, index=lab.index).groupby(consec).cumsum().to_numpy()   # 연속 묶음 안 순서 (1, 2)
        glob = pd.factorize(lab.shot_id)[0]
        steps = [("1", "02 그대로: 연속 행 묶음 + position_in_shot", consec, True),
                 ("2", "position_in_shot 제거 (연속 행 묶음은 그대로)", consec, False),
                 ("3", "shot_id 를 전역 기준으로 (이번 파이프라인과 같은 묶음)", glob, False)]
        for step, change, groups, use_pos in steps:
            X = lab[base].to_numpy(float)
            if use_pos:
                X = np.c_[X, pos]
            oof, fold_ap, fold_auc = np.zeros(len(y)), [], []
            for tr, va in SGKF(5, shuffle=True, random_state=cfg["seed"]).split(X, y, groups):
                m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000,
                                                                       solver="liblinear", random_state=cfg["seed"]))
                m.fit(X[tr], y[tr])
                oof[va] = m.predict_proba(X[va])[:, 1]
                if 0 < y[va].sum() < len(va):
                    fold_ap.append(average_precision_score(y[va], oof[va]))
                    fold_auc.append(roc_auc_score(y[va], oof[va]))
            split_pairs = int(pd.DataFrame({"g": groups, "s": lab.shot_id}).groupby("s").g.nunique().gt(1).sum())
            rows.append({"product": p, "step": step, "change": change, "n_groups": int(len(np.unique(groups))),
                         "global_pairs_split": split_pairs,
                         "row_ap (pooled OOF)": average_precision_score(y, oof),
                         "row_roc_auc (pooled OOF)": roc_auc_score(y, oof),
                         "row_ap_foldmean (참고)": float(np.mean(fold_ap)),
                         "row_roc_auc_foldmean (참고)": float(np.mean(fold_auc))})
    return pd.DataFrame(rows)


# 가상 선택 -----------------------------------------------------------------------------------
def hypothetical_selection(metrics, extra_metrics: dict, choices, sel, n_repeats) -> pd.DataFrame:
    """실험 모델(가중치 없음·통합)까지 후보에 넣고 같은 규칙(sel)으로 골랐다면? — 보고용, 적용하지 않음."""
    parts = [metrics[metrics.scenario.eq("separate")][["cv_mode", "product", "model", "repeat", sel["metric"]]]]
    for tag, m in extra_metrics.items():
        m = m[m["product"] != "all"][["cv_mode", "product", "model", "repeat", sel["metric"]]].copy()
        if tag == "pooled":
            m["model"] = m["model"] + "@pooled"
        parts.append(m)
    allm = pd.concat(parts, ignore_index=True)
    allm = allm[allm.cv_mode.eq(sel["cv_mode"]) & ~allm.model.str.startswith("dummy")]

    def cx(name):
        base, pooled = name.split("@")[0], "@pooled" in name
        base = BASE_MODEL.get(base, base)
        return (COMPLEXITY.index(base) if base in COMPLEXITY else 99) + (0.5 if pooled else 0)

    rows = []
    for p, c in choices.items():
        g = allm[allm["product"] == p].groupby("model")[sel["metric"]].agg(["mean", "std"])
        best = g["mean"].idxmax()
        se = g.loc[best, "std"] / np.sqrt(max(n_repeats, 1)) if sel.get("rule") == "one_se" else 0.0
        ok = g[g["mean"] >= g.loc[best, "mean"] - (se if np.isfinite(se) else 0)].copy()
        ok["cx"] = [cx(m) for m in ok.index]
        pick = ok.sort_values(["cx", "mean"], ascending=[True, False]).index[0]
        rows.append({"product": p, "actual_choice": c["model"], "hypothetical_choice": pick,
                     "hypothetical_score": g.loc[pick, "mean"], "best_candidate": best,
                     "best_score": g.loc[best, "mean"], "changes_final_model": pick != c["model"]})
    return pd.DataFrame(rows)


# 선택 확인 / 불량 유형별 탐지 -------------------------------------------------------------------
def _shot_ap(g):
    s = g.groupby("shot_id").agg(y=("y", "max"), p=("p", "max"))
    return average_precision_score(s.y, s.p) if s.y.sum() else np.nan


def selection_check(oof_all, oof_none, choice, product) -> dict | None:
    """최종 모델 vs 같은 모델의 가중치 없는 변형(<model>_none).
    - fold 단위 shot_ap 승/패/무 (block 은 불량 최다 fold 와 나머지로 나눔)
    - 사전 기준: block·shot 양쪽에서 불량이 있는 fold 의 과반을 변형이 이기면 교체 후보
    - 보정 후 Brier(shot 그룹 CV Platt), 검사비율 2%/10% 재현율"""
    from .evaluate import recall_at_budget
    from .final import fit_platt
    base, alt = choice["model"], f"{choice['model']}_none"
    b = oof_none[(oof_none.model == alt) & (oof_none["product"] == product)]
    a = oof_all[(oof_all.dataset == choice["dataset"]) & (oof_all.model == base) & (oof_all["product"] == product)]
    if b.empty or a.empty:
        return None
    rows = []
    for (mode, rep, fold), ga in a.groupby(["cv_mode", "repeat", "fold"]):
        gb = b[(b.cv_mode == mode) & (b.repeat == rep) & (b.fold == fold)]
        rows.append({"cv_mode": mode, "repeat": rep, "fold": fold, "valid_defect_rows": int(ga.y.sum()),
                     f"shot_ap[{base}]": _shot_ap(ga), f"shot_ap[{alt}]": _shot_ap(gb)})
    f = pd.DataFrame(rows)
    d = f[f"shot_ap[{alt}]"] - f[f"shot_ap[{base}]"]
    f["result"] = np.select([d > 1e-12, d < -1e-12], [f"{alt} 승", f"{base} 승"], "무")
    mx = f[f.cv_mode == "block"].valid_defect_rows.max() if (f.cv_mode == "block").any() else None
    f["group"] = np.where(f.cv_mode != "block", f.cv_mode + ": 전체",
                          np.where(f.valid_defect_rows == mx, f"block: 불량 최다 fold ({mx}행)", "block: 나머지 fold"))
    has = f[f.valid_defect_rows > 0]
    summ = has.groupby("group").agg(folds=("result", "size"),
                                    alt_wins=("result", lambda r: int((r == f"{alt} 승").sum())),
                                    base_wins=("result", lambda r: int((r == f"{base} 승").sum())),
                                    ties=("result", lambda r: int((r == "무").sum())),
                                    **{f"mean shot_ap[{base}]": (f"shot_ap[{base}]", "mean"),
                                       f"mean shot_ap[{alt}]": (f"shot_ap[{alt}]", "mean")}).reset_index()
    verdict = {}
    for mode, g in has.groupby("cv_mode"):
        w = int((g.result == f"{alt} 승").sum())
        verdict[mode] = {"folds": len(g), "alt_wins": w, "majority": bool(w > len(g) / 2)}
    swap = bool(verdict) and all(v["majority"] for v in verdict.values())
    pooled = {}  # 반복별로 fold 를 이어 붙인 shot_ap 의 평균 (2-5·2-8 과 같은 방식)
    for mode in f.cv_mode.unique():
        pooled[mode] = {name: float(np.mean([_shot_ap(r) for _, r in o[o.cv_mode == mode].groupby("repeat")]))
                        for name, o in ((base, a), (alt, b))}
    cal_rows = []
    for name, o in ((base, a), (alt, b)):
        for mode, g in o.groupby("cv_mode"):
            om = g.groupby("row").agg(y=("y", "first"), p=("p", "mean"), shot_id=("shot_id", "first"))
            _, cal = fit_platt(om.p.to_numpy(), om.y.to_numpy(), om.shot_id.to_numpy())
            cal_rows.append({"cv_mode": mode, "model": name, "brier_calibrated_cv": float(np.mean((cal - om.y) ** 2)),
                             "brier_constant": float(om.y.mean() * (1 - om.y.mean())),
                             **{f"row_recall@{int(bb * 100)}%": float(np.mean([recall_at_budget(r.y, r.p, bb)
                                                                                 for _, r in g.groupby("repeat")]))
                                for bb in (0.02, 0.10)}})
    return {"folds": f, "summary": summ, "calib": pd.DataFrame(cal_rows),
            "verdict": {"base": base, "alt": alt, "by_mode": verdict, "swap_by_rule": swap, "pooled": pooled}}


def defect_type_detection(oof_all, choice, product, types: pd.Series, budgets=(0.02, 0.05, 0.10),
                          threshold=0.5) -> tuple[pd.DataFrame, dict]:
    """불량 유형별로 몇 행이 잡히는가 (반복당 평균). 상위 k% = 점수 순위 ≤ ceil(n·k), 임계값 = 모델 점수 ≥ threshold."""
    o = oof_all[(oof_all.dataset == choice["dataset"]) & (oof_all.model == choice["model"])
                & (oof_all["product"] == product)].copy()
    o["type"] = o.shot_id.map(types).fillna("normal")
    rows = []
    for mode, g in o.groupby("cv_mode"):
        caught = []
        for rep, gr in g.groupby("repeat"):
            rank = gr.p.rank(ascending=False, method="average")
            d = gr[gr.y == 1].assign(**{f"top{int(b * 100)}%": rank[gr.y == 1] <= int(np.ceil(len(gr) * b))
                                        for b in budgets},
                                     **{f"score≥{threshold:g}": gr.p[gr.y == 1] >= threshold})
            caught.append(d.assign(repeat=rep))
        c = pd.concat(caught)
        cols = [f"top{int(b * 100)}%" for b in budgets] + [f"score≥{threshold:g}"]
        for t, gt in c.groupby("type"):
            r = {"cv_mode": mode, "defect_type": t, "shots": gt.shot_id.nunique(), "defect_rows": gt.row_id.nunique()}
            for col in cols:
                r[f"TP rows {col}"] = float(gt.groupby("repeat")[col].sum().mean())
            rows.append(r)
    order = o.groupby("shot_id").row_id.min().sort_values()
    idx = pd.Series(np.arange(len(order)), index=order.index)
    pw = sorted(int(idx[s]) for s in types[types.eq("process_wide")].index if s in idx.index)
    info = {"process_wide_shot_idx": pw, "consecutive": bool(pw) and pw == list(range(pw[0], pw[0] + len(pw)))}
    return pd.DataFrame(rows), info


# 실행 묶음 -----------------------------------------------------------------------------------
def run_cv_experiments(frames, products, feats, metrics, oof_all, cfg, audit, log=print) -> dict:
    ceil = ceilings(audit)
    res, extra = {}, {}
    log("  [실험 1] 제품별(A) vs 통합(B)")
    res["exp_ab"], extra["pooled"] = exp_separate_vs_pooled(frames, products, feats, metrics, oof_all, cfg, ceil)
    res["exp_domain_auc"] = domain_classifier_auc(frames, products, feats, cfg["seed"])
    log("  [실험 1b] 교차 금형 (CN7→RG3, RG3→CN7)")
    res["exp_cross_mold"] = exp_cross_mold(frames, products, feats, metrics, cfg, ceil)
    log("  [실험 2] class_weight None vs balanced")
    res["exp_class_weight"], extra["none"], extra_oof_none = exp_class_weight(frames, products, feats, metrics,
                                                                            oof_all, cfg, ceil)
    log("  [실험 3] 기준선 3종")
    res["exp_dummies"] = exp_dummies(frames, products, feats, metrics, oof_all, cfg, ceil)
    return {"tables": res, "extra_metrics": extra, "extra_oof": {"none": extra_oof_none}}


def run_post_selection(frames, products, feats, metrics, oof_all, table, choices, cfg, audit, extra_metrics,
                       fold_check, root: Path, synthetic: bool, log=print, extra_oof=None, types_by_product=None) -> dict:
    ceil = ceilings(audit)
    res = {}
    log("  [선택 확인] 최종 모델 vs 가중치 없는 변형 (fold 단위)")
    for p, c in choices.items():
        sc = selection_check(oof_all, (extra_oof or {}).get("none", pd.DataFrame(columns=["model", "product"])), c, p)
        if sc:
            res[f"selection_check_{p}_folds"], res[f"selection_check_{p}_summary"] = sc["folds"], sc["summary"]
            res[f"selection_check_{p}_calib"], res[f"_selection_verdict_{p}"] = sc["calib"], sc["verdict"]
        if types_by_product and p in types_by_product:
            t, info = defect_type_detection(oof_all, c, p, types_by_product[p])
            res[f"defect_type_detection_{p}"], res[f"_defect_type_info_{p}"] = t, info
    focus = cfg["experiments"].get("threshold_products", ["cn7"])
    log("  [실험 4] 임계값 trade-off / 검사비율")
    for p in focus:
        if p in choices:
            t = exp_thresholds(oof_all, choices[p], p, cfg, ceil)
            res[f"exp_threshold_{p}"], res[f"exp_budget_{p}"] = t["threshold"], t["budget"]
    log("  [실험 5] 이전 베이스라인 비교")
    res["exp_baseline_v0"] = baseline_v0_ladder(frames, products, feats, metrics, oof_all, cfg, ceil, root, synthetic)
    res["perf_range"] = perf_range(table, choices, ceil)
    log("  [누수 재현] 후보 코드(02) 설정")
    res["candidate02_leak"] = candidate02_leak(frames, products, feats, cfg)
    res["holdout_designs"], res["holdout_trials"] = holdout_designs(frames, products, cfg)
    res["hypothetical_selection"] = hypothetical_selection(metrics, extra_metrics, choices, cfg["selection"],
                                                           cfg["cv"]["n_repeats"])
    fc = fold_check
    if fc is not None and len(fc) and "valid_defect_rows" in fc:
        r = fc[(fc.mold == "cn7") & (fc.cv_mode == "block") & (fc.n_splits == cfg["cv"]["n_splits"])
               & (fc.repeat == 0)]
        if len(r):
            res["_cn7_block_alloc"] = {"valid": sorted(r.valid_defect_rows.astype(int).tolist()),
                                       "train_for_largest": int(r.loc[r.valid_defect_rows.idxmax(), "train_defect_rows"])}
    return res
