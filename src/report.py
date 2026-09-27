"""결과를 outputs/report.md 한 장으로 정리 (보고서 작성용 근거 자료)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import known_facts
from .evaluate import TYPE_KO
from .models import MODEL_DESC


def md(df: pd.DataFrame, floatfmt=3, index=False) -> str:
    if df is None or len(df) == 0:
        return "_(없음)_\n"
    d = df.reset_index() if index else df.copy()

    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows]) + "\n"


def _pivot(table, cols, modes):
    key = ["product", "dataset", "model"]
    pv = table.pivot_table(index=key, columns="cv_mode", values=cols, aggfunc="first")
    pv.columns = [f"{c}[{m}]" for c, m in pv.columns]
    order = [f"{c}[{m}]" for c in cols for m in modes if f"{c}[{m}]" in pv.columns]
    return pv[order].reset_index()


def fold_check_summary(fold_check: pd.DataFrame) -> pd.DataFrame:
    """fold 수(n_splits)별: 불량이 든 그룹 수, 검증 fold 의 불량 행 수 범위, 불량 0개 fold 수."""
    if fold_check is None or not len(fold_check) or "valid_defect_rows" not in fold_check:
        return pd.DataFrame()
    fc = fold_check.dropna(subset=["valid_defect_rows"])
    g = fc.groupby(["mold", "cv_mode", "n_splits"])
    t = g.agg(groups_with_defect=("groups_with_defect_total", "first"),
              valid_defects_min=("valid_defect_rows", "min"), valid_defects_max=("valid_defect_rows", "max"),
              folds=("fold", "size"), folds_without_defect=("valid_defect_rows", lambda v: int((v == 0).sum())))
    t["repeat별 검증 불량 배분"] = g.apply(lambda d: " / ".join(sorted({str(sorted(x.valid_defect_rows.tolist()))
                                                              for _, x in d.groupby("repeat")})))
    return t.reset_index()


def ceiling_text(audit) -> str:
    return "행 단위 성능 상한(짝 두 행은 예측이 같아 넘을 수 없음): " + ", ".join(
        f"{p.upper()} 정밀도 {a['row_precision_ceiling']:.3f} / F1 {a['row_f1_ceiling']:.3f}" for p, a in audit.items())


def _cols(t, keep):
    return t[[c for c in keep if c in t.columns]]


def write(path, cfg, audit, dropped, table, ci, choices, final_summ, err, type_tab, manifest, holdout_info=None,
          fold_check=None, ablation=None, exp=None):
    exp = exp or {}
    modes = cfg["cv"]["modes"]
    CEIL = f"> {ceiling_text(audit)}\n"
    sel = cfg["selection"]
    L = [f"# 사출성형 품질불량 사전예측 — 실행 결과 ({cfg['run_name']})\n",
         "> 자동 생성 문서. 숫자는 `run_all.py` 실행 시점의 결과이며 보고서 작성의 근거 자료로 사용.\n"]

    # 1 ------------------------------------------------------------------
    L.append("## 1. 데이터 이해 및 진단\n")
    for p, a in audit.items():
        sc = a["scale_check"]
        L.append(f"### {p.upper()}\n")
        L.append(f"- labeled {a['rows']:,}행, 불량 {a['positives']}행 ({a['positive_rate']:.2%}), unlabeled {a['unlabeled_rows']:,}행")
        L.append(f"- 샷(LH/RH 짝) {a['shots']:,}개, 짝 크기 분포 {a['shot_size_hist']}, 짝이 row_id상 연속인 비율 {a['pair_adjacent_ratio']:.1%}")
        L.append(f"- 불량 샷 {a['defect_shots']}개 = **공정 전체 이상(양쪽 불량) {a['process_wide_shots(양쪽 불량)']}개** + "
                 f"**캐비티 국소(한쪽만 불량) {a['cavity_local_shots(한쪽만 불량)']}개**"
                 + (f" + 짝 없는 행 {a['single_row_defects']}개" if a["single_row_defects"] else ""))
        L.append(f"- **행 단위 성능 상한**: 짝은 입력이 같아 예측도 같음 → 피할 수 없는 FP {a['row_FP_unavoidable']}건, "
                 f"정밀도 상한 {a['row_precision_ceiling']:.3f}, F1 상한 {a['row_f1_ceiling']:.3f}")
        if "first_defect_shot_idx" in a:
            L.append(f"- 불량 위치(샷 순서 {a['n_shots']}개 중): {a['first_defect_shot_idx']}번째 ~ {a['last_defect_shot_idx']}번째 "
                     f"(전체 순서의 {a['defect_span_share']:.1%} 구간), 앞 10% 안에 있는 불량 샷 비율 {a['defects_in_first_10pct_shots']:.0%}")
        L.append(f"- 순서(10분위)별 불량 행 수: {a['positives_by_order_decile']}")
        if a.get("cavity_local_pairs"):
            L.append(f"- 한쪽만 불량인 짝 {a['cavity_local_pairs']}개에서 불량 행이 짝의 앞 행인 비율: "
                     f"{a['defect_is_first_row_share']:.0%} (50%에서 크게 벗어나면 행 순서에 불량 정보가 섞인 것. "
                     "원본 연결 결과 기록 순서 누수로 확인되어 변수로 쓰지 않음)")
        L.append(f"- 표준화 여부: {a['looks_standardized']} / 상수열(labeled): {a['constant_in_labeled']} / 결측 {a['missing_values']}")
        L.append(f"- labeled↔unlabeled 분포 차이: adversarial AUC {a['adversarial_auc_labeled_vs_unlabeled']} "
                 f"(0.5=같음), KS 상위 {a['ks_top5']}")
        L.append(f"- **눈금(표준화 기준) 점검**: 두 파일 모두 평균0·표준편차1 = {sc['both_files_mean0_std1']}, "
                 f"labeled 값이 unlabeled에 똑같이 존재하는 비율 평균 {sc['value_overlap_mean']:.1%} → {sc['verdict']}")
        L.append(f"- labeled 와 **z값이 똑같은** 행이 unlabeled 에 있는 수: {a['labeled_rows_same_z_in_unlabeled']}행 "
                 f"/ unlabeled 짝 크기 분포 {a['unlabeled_shot_size_hist']}")
        L.append("  - ⚠ 0이어도 '겹침 없음'이 아님: 파일별 표준화로 같은 행도 z값이 다름. 공개 원본 검증 결과 "
                 f"**대회 unlabeled 안에 labeled 행이 들어 있음** ({known_facts.overlap_text()}, 사실 10) "
                 "→ unlabeled 로 학습하거나 unlabeled 예측을 평가에 쓰지 않음")
        L.append(f"  - {known_facts.OVERLAP_CRITERION}\n")
        L.append(f"![](figures/audit_{p}.png)\n")
    L.append(f"제거한 상수열: {dropped}\n")

    # 2 ------------------------------------------------------------------
    L.append("## 2. 모델 비교\n")
    L.append("**검증 설계**\n")
    L.append("- 반복 층화 그룹 교차검증: "
             f"{cfg['cv']['n_splits']}-fold × {cfg['cv']['n_repeats']}회 반복, 평균±표준편차")
    L.append("- `shot` CV: LH/RH 짝을 같은 fold로 묶음 (짝 누수 차단) — 같은 생산 구간 안에서의 성능 (낙관적)")
    L.append(f"- `block` CV: 연속 {cfg['cv'].get('block_shots', 10)}샷을 한 덩어리로 묶음 (이웃 샷까지 차단) — 처음 보는 구간에서의 성능 (보수적)")
    L.append("- 판정 임계값은 학습 fold 내부 OOF에서 F1 최대값으로 결정 (검증 fold 미사용)")
    L.append("- 분할표: `fold_id.csv` (mold, row_id, shot_id, block_id, fold_<cv_mode>_r<반복>) — 팀원 모델도 같은 분할 사용")
    if holdout_info:
        L.append(f"- 홀드아웃: 제품별로 연속 블록 기준 {holdout_info['frac']:.0%}를 처음부터 떼어 둠 {holdout_info['counts']}")
    L.append("")
    fcs = fold_check_summary(fold_check)
    if len(fcs):
        L.append(f"**fold 수 점검 — 왜 {cfg['cv']['n_splits']}-fold 인가** (검증 fold 에 불량이 0개면 AP·재현율을 잴 수 없음)\n")
        L.append(md(fcs))
        cn = fcs[(fcs.mold == "cn7") & (fcs.cv_mode == "block")]
        if len(cn):
            L.append("\n- CN7 은 불량 17행이 연속 구간(샷 순번 24~60)에 몰려 불량이 든 블록이 "
                     f"{int(cn.groups_with_defect.iloc[0])}개뿐 → 블록 CV 를 5-fold 로 하면 검증 fold 에 불량 0개가 생김"
                     f" (위 표 `folds_without_defect`). 그래서 모든 제품·CV 방식을 3-fold 로 통일")
            L.append("- 블록 수가 적어 반복(repeat)을 바꿔도 불량 블록의 배분은 거의 같음 → CN7 블록 CV 의 반복 간 표준편차는 "
                     "'분할 운'의 일부(정상 블록 배분)만 반영함")
        al = exp.get("_cn7_block_alloc")
        if al:
            L.append(f"- **CN7 block CV 는 매우 비관적인 조건**: 검증 fold 별 불량 배분이 {al['valid']} 이라, "
                     f"불량이 가장 많은 fold 는 **불량 {al['train_for_largest']}행만으로 학습해 {max(al['valid'])}행을 맞혀야** 함. "
                     "그래서 성능은 shot CV(낙관) ~ block CV(비관) 사이의 **범위**로 제시함 (2-1 아래 '성능 범위' 표)")
        L.append("")
    hd = exp.get("holdout_designs")
    if hd is not None and len(hd):
        L.append("**holdout 을 쓰지 않은 이유** — 25% 를 떼어 둘 때 holdout 에 들어가는 불량 수 "
                 f"(무작위 설계는 seed {cfg['experiments']['holdout_sim']['seeds']}개)\n")
        L.append(md(hd, floatfmt=2))
        L.append("\n- CN7 은 불량이 한 구간에 몰려 있어 시간순 holdout 은 불량 0개, 블록 층화 holdout 은 불량 2행뿐, "
                 "블록 무작위는 불량 0개일 확률이 높거나 반대로 거의 전부가 빠져 학습 쪽에 1행만 남기도 함 → "
                 "어떤 설계도 '평가할 수 있는 크기'의 holdout 이 되지 않음")
        L.append("- 그래서 holdout 대신 labeled 전 행의 교차검증 예측(`predictions_labeled_cv.csv`)을 테스트 예측으로 사용 "
                 "(`holdout.enabled=false`, 팀 확정 전까지 유지)\n")
    L.append("\n".join(f"- `{k}`: {v}" for k, v in MODEL_DESC.items() if k in cfg["models"]) + "\n")
    L.append("### 2-1. 핵심 지표 (샷 CV vs 블록 CV)\n")
    L.append("`row_f1` = 학습 fold 내부 F1 최적 임계값 기준, `row_f1@2%` = 위험 상위 2%만 불량 판정했을 때 "
             "(블록 CV에서는 임계값이 불안정하므로 `@2%` 쪽이 운영 판단에 가까움)\n")
    pv = _pivot(table, ["shot_ap", "shot_ap_sd", "row_ap", "row_f1", "row_f1@2%", "row_recall@10%"], modes)
    sort_col = f"shot_ap[{sel['cv_mode']}]" if f"shot_ap[{sel['cv_mode']}]" in pv.columns else pv.columns[3]
    L.append(md(pv.sort_values(["product", sort_col], ascending=[True, False])))
    L.append(CEIL)
    pr = exp.get("perf_range")
    if pr is not None and len(pr):
        L.append("**성능 범위 — 선택 모델의 block CV(비관) ~ shot CV(낙관)** (pooled OOF, 반복 평균)\n")
        L.append(md(pr))
        L.append("\n실제 현장 성능은 이 범위 안 어딘가로 봐야 함: 처음 보는 생산 구간이면 block 쪽, "
                 "이미 겪은 운전 상태가 반복되면 shot 쪽에 가까움.\n")
    L.append("\n> ※ `dummy` 의 ROC-AUC 가 0.5 에서 벗어나는 것(예: 블록 CV 에서 0.3 안팎)은 **fold 별 상수 예측을 합친 착시**입니다. "
             "dummy 는 fold 마다 학습 불량률 하나만 예측하는데, fold 마다 그 값이 달라서 OOF 를 이어 붙이면 "
             "'불량이 적은 학습 fold 에서 나온 점수'가 불량이 많은 검증 fold 에 붙는 식의 인위적 순서가 생깁니다. "
             "dummy 는 PR-AUC(≈불량률)·재현율@K(≈K) 기준선으로만 보세요.\n")
    L.append("\n### 2-2. 최종 모델 선택\n")
    L.append(f"선택 규칙: `{sel['cv_mode']}` CV의 `{sel['metric']}` 기준, "
             + ("1-SE 규칙(최고 성능과 1 표준오차 이내면 더 단순한 모델)" if sel.get("rule") == "one_se" else "최고값") + "\n")
    for p, c in choices.items():
        L.append(f"- **{p.upper()}** → 시나리오 `{c['scenario']}`, 모델 `{c['model']}` "
                 f"({sel['metric']} = {c['score']:.3f}; 최고 `{c['best_model']}` {c['best_score']:.3f}; "
                 f"무작위 기준선 {c['dummy']:.3f}의 **{c['lift_vs_dummy']:.2f}배**)")
        L.append("  - 95% 부트스트랩 신뢰구간(샷 재표본): " + ", ".join(f"{k} [{v[0]:.3f}, {v[1]:.3f}]" for k, v in ci[p].items()))
        if not c["beats_chance"]:
            L.append(f"  - ⚠ **{p.upper()}는 어떤 모델도 무작위 기준선을 의미 있게 넘지 못함** → "
                     "공정변수로 설명되지 않는 불량(캐비티/금형 요인, 라벨 품질 등). 예측 성능 대신 원인 진단과 한계로 서술.")
    # 1-SE 주석: 반복 간 불량 배분이 같으면 반복 간 표준편차(=SE 의 근거)가 분할 운을 덜 반영
    if fold_check is not None and len(fold_check) and "valid_defect_rows" in fold_check:
        fc = fold_check[(fold_check.cv_mode == sel["cv_mode"]) & (fold_check.n_splits == cfg["cv"]["n_splits"])]
        for p, g in fc.groupby("mold"):
            allocs = {tuple(sorted(x.valid_defect_rows.astype(int))) for _, x in g.groupby("repeat")}
            if len(allocs) == 1 and p in choices:
                L.append(f"  - ※ 1-SE 주석 ({p.upper()}): `{sel['cv_mode']}` CV 는 반복마다 검증 fold 의 불량 배분이 "
                         f"{list(next(iter(allocs)))} 으로 같음 → 반복 간 표준편차가 분할 운을 일부만 반영해 **SE 가 과소추정**됨 "
                         "(1-SE 허용 폭이 실제보다 좁음)")
    for p, c in choices.items():
        v = exp.get(f"_selection_verdict_{p}")
        if not v:
            continue
        base, alt = v["base"], v["alt"]
        L.append(f"\n**선택 확인 ({p.upper()}): `{base}` vs 가중치 없는 `{alt}` — fold 단위** "
                 "(각 fold 의 검증 shot_ap 를 직접 비교)\n")
        L.append(md(exp[f"selection_check_{p}_summary"]))
        L.append("\n**보정 후 Brier · 검사비율 재현율** (Brier 는 shot 그룹 CV 로 보정한 확률, 작을수록 좋음)\n")
        L.append(md(exp[f"selection_check_{p}_calib"], floatfmt=4))
        bm = v["by_mode"]
        rule = ", ".join(f"{m} {x['alt_wins']}/{x['folds']}" for m, x in bm.items())
        L.append(f"\n- 사전 기준: block·shot 양쪽에서 불량이 있는 fold 의 과반을 `{alt}` 가 이기면 교체 → {rule} 승 → "
                 + ("**기준 충족**" if v["swap_by_rule"] else f"**기준 미충족, `{base}` 유지**"))
        pm = v.get("pooled", {}).get(sel["cv_mode"], {})
        if not v["swap_by_rule"] and pm and pm.get(alt, 0) > pm.get(base, 0):
            L.append(f"- pooled OOF(2-5·2-8)에서는 `{alt}` 가 높게 나오는데 ({sel['cv_mode']} shot_ap {pm[alt]:.3f} vs {pm[base]:.3f}), pooled OOF 는 fold 마다 다른 모델의 점수를 이어 붙여 "
                     "계산하므로 **fold 간 점수 눈금 차이가 섞일 수 있음** (가설, 미검증). 최종 선택은 미리 정한 기준(fold 단위 승패)과 "
                     "현장 운영 지표인 검사비율 재현율로 판단함.")
    L.append("\n![](figures/model_comparison.png)\n")

    if ablation is not None and len(ablation):
        L.append("### 2-3. Ablation: 짝 안 위치 변수 (⚠ 누수 변수 — 비교용)\n")
        L.append("짝(LH/RH) 두 행 중 row_id 가 앞선 행이면 1인 변수(`pair_first`)를 켰을 때와 껐을 때를 **같은 분할**로 비교. "
                 "이 변수는 캐비티가 아니라 원본 **기록 순서**이며 불량 여부와 얽혀 있음 → 모델 선택·예측 파일에 쓰지 않음.\n")
        L.append(f"- 근거 ({known_facts.SOURCE}): {known_facts.pair_order_leak_text()}\n")
        show = ["note", "product", "cv_mode", "model"] + [c for c in ablation.columns
                                                       if c.startswith(("shot_ap", "row_ap", "row_roc_auc"))]
        L.append(md(ablation[show].sort_values(["product", "cv_mode", "model"])))
        L.append("\n켰을 때 오르는 만큼이 '정답이 새어 들어와 생긴 가짜 성능'입니다.\n")

    if exp:
        L.append("### 추가 실험 공통 사항\n")
        L.append("- 모든 실험은 `fold_id.csv` 와 같은 분할(shot·block), 지표는 **pooled OOF**(반복마다 fold 를 이어 붙여 계산 → "
                 "반복 평균·표준편차). `*_foldmean` 은 fold 별 값의 평균으로 **참고용**")
        L.append("- 모델 선택 규칙(block · shot_ap · 1-SE)은 그대로. 실험 모델은 선택 후보가 아님 (2-8 에서 '넣었다면'만 보고)\n")
    keep = ["experiment", "class_weight", "product", "cv_mode", "model", "row_ap", "row_ap_sd", "shot_ap", "shot_ap_sd",
            "row_roc_auc", "row_recall@10%", "row_f1@2%", "row_ap_foldmean", "row_P_ceiling", "row_F1_ceiling"]
    t = exp.get("exp_ab")
    if t is not None and len(t):
        L.append("### 2-4. 제품별(A) vs 통합(B, 제품 구분 변수 `is_rg3`)\n")
        L.append("통합 모델도 제품별 fold k 를 합친 같은 분할로 학습·평가하고, 성능은 제품별로 나눠 계산.\n")
        L.append(md(_cols(t, keep).sort_values(["product", "cv_mode", "model", "experiment"])))
        d = exp.get("exp_domain_auc")
        if d is not None and len(d):
            L.append("\n**도메인 분류기 AUC** — 공정변수만으로 CN7 과 RG3 를 구분할 수 있는가 (5-fold, shot 단위 그룹)\n")
            L.append(md(d))
            auc = d.set_index("domain_classifier")["roc_auc"]
            tree_max = auc[[i for i in auc.index if i.startswith("트리")]].max()
            lin = auc[[i for i in auc.index if i.startswith("선형")]].max()
            L.append(f"\n트리 AUC {tree_max:.3f}: 공정값의 **조합·분포 모양**으로 두 제품이 완전히 갈림. 선형 AUC {lin:.3f}: "
                     "파일마다 따로 표준화돼 평균·분산 차이가 지워져 선형 경계로는 구분이 안 됨(0.5 아래는 분할에 따른 흔들림). "
                     "→ 두 제품은 같은 분포로 볼 수 없어 **제품별 모델이 기본**, 통합은 비교 실험으로만.")
            ab = t.pivot_table(index=["product", "cv_mode", "model"], columns="experiment", values="shot_ap")
            if ab.shape[1] == 2:
                a_col, b_col = sorted(ab.columns)
                per = (ab[b_col] > ab[a_col]).groupby(level="product").agg(["sum", "size"])
                L.append("- shot_ap 기준 통합(B)이 제품별(A)보다 높은 조합 수: "
                         + ", ".join(f"{p.upper()} {int(r['sum'])}/{int(r['size'])}" for p, r in per.iterrows())
                         + " (신호가 없는 제품은 둘 다 기준선 수준이라 우열이 의미 없음)\n")
    t = exp.get("exp_class_weight")
    if t is not None and len(t):
        L.append("### 2-5. class_weight: None vs balanced (SMOTE 미사용)\n")
        L.append(md(_cols(t, keep).sort_values(["product", "cv_mode", "model", "class_weight"])))
        cw = t.pivot_table(index=["product", "cv_mode", "model"], columns="class_weight", values="shot_ap")
        if "none" in cw.columns:
            other = [c for c in cw.columns if c != "none"]
            bal = cw[other].bfill(axis=1).iloc[:, 0]
            diff = (cw["none"] - bal).rename("Δ shot_ap (none − balanced)").reset_index()
            L.append("\n**가중치를 뺐을 때 shot_ap 변화**\n")
            L.append(md(diff))
            for (p, mode), g in diff.groupby(["product", "cv_mode"]):
                col = "Δ shot_ap (none − balanced)"
                up = g.loc[g[col] > 0.01, "model"].tolist()
                down = g.loc[g[col] < -0.01, "model"].tolist()
                L.append(f"- {p.upper()} {mode}: 가중치 없음이 높음 {up or '없음'} / balanced 가 높음 {down or '없음'} "
                         "(|Δ| ≤ 0.01 은 차이 없음으로 봄)")
        L.append("\n- 가중치는 확률의 크기만이 아니라 **학습된 경계(점수 순서)도 바꿈** → AP 도 달라짐. 방향은 모델마다 다름")
        L.append("- 이 비교는 보고용. 본 모델 후보(config.models)는 balanced 그대로이며, 선택에 미치는 영향은 2-8 참고\n")
    t = exp.get("exp_dummies")
    if t is not None and len(t):
        L.append("### 2-6. 기준선 3종\n")
        L.append(md(_cols(t, keep).sort_values(["product", "cv_mode", "model"])))
        L.append("\n- 세 기준선 모두 PR-AUC ≈ 불량률. most_frequent·stratified 는 재현율@10% ≈ 0.1 (무작위 검사와 같음)")
        L.append("- prior(dummy) 만 block CV 에서 ROC-AUC·재현율이 0.5·0.1 에서 벗어나는 것은 fold 별 상수 예측을 합친 착시(2-1 주석)"
                 " → '무작위 기준선'으로는 most_frequent·stratified 를 보는 것이 정확\n")
    t = exp.get("exp_baseline_v0")
    if t is not None and len(t):
        L.append("### 2-7. 이전 베이스라인(archive/baseline_v0)과 비교 — 왜 0.476 → 0.19~0.36 인가\n")
        L.append(md(_cols(t, ["product", "step", "change", "model", "cv", "repeats", "row_ap", "row_ap_sd", "shot_ap",
                             "row_ap_foldmean", "row_P_ceiling", "row_F1_ceiling"])))
        c = t[t["product"] == "cn7"].set_index("step")
        if {"1", "2", "3", "4", "5", "6"} <= set(c.index):
            r = c["row_ap"]
            same = "0" in c.index and abs(r["1"] - r["0"]) < 1e-6
            L.append(f"\n- 1단계: v0 설정을 그대로 재현하면 CN7 row PR-AUC {r['1']:.3f}"
                     + (" — archive 보고값과 **정확히 같음** → 차이는 코드가 아니라 검증 조건·모델 설정에서 옴" if same else ""))
            L.append(f"- 2단계 (분할 운): 같은 5-fold 라도 분할을 바꿔 {c.loc['2', 'repeats']}회 반복하면 "
                     f"{r['2']:.3f} ± {c.loc['2', 'row_ap_sd']:.3f} ({r['2'] - r['1']:+.3f})")
            L.append(f"- 3단계 (3-fold): {r['3']:.3f} ({r['3'] - r['2']:+.3f}) — fold 마다 학습 데이터가 줄어듦")
            L.append(f"- 4단계 (block CV): {r['4']:.3f} ({r['4'] - r['3']:+.3f}) — 이웃 샷까지 막으면 연속 불량 구간을 '옆에서 보고' "
                     "맞히는 효과가 사라짐")
            L.append(f"- 5·6단계 (모델 설정 C=1·가중치 없음 → C=0.3·balanced): shot {r['5']:.3f} ({r['5'] - r['3']:+.3f}), "
                     f"block {r['6']:.3f} ({r['6'] - r['4']:+.3f}) — **검증 조건만큼 큰 하락이 모델 설정에서 옴** (2-5·2-8 참고)")
            L.append(f"- 정리: 0.476 → 이번 row PR-AUC {r['6']:.3f}(block) ~ {r['5']:.3f}(shot). "
                     f"shot 단위 AP 로는 {c.loc['6', 'shot_ap']:.3f} ~ {c.loc['5', 'shot_ap']:.3f}. "
                     "'0.19~0.36' 은 block shot_ap(0.19)와 shot row PR-AUC(0.36)를 섞은 표현이라, "
                     "같은 지표끼리 비교할 때는 위 두 범위를 쓸 것\n")
    t = exp.get("hypothetical_selection")
    if t is not None and len(t):
        L.append("### 2-8. 실험 모델까지 후보에 넣었다면? (같은 규칙, 보고용 — 적용 안 함)\n")
        L.append(md(t))
        if t.changes_final_model.any():
            L.append("\n⚠ 일부 제품에서 선택이 달라짐 → 팀 합의 후에만 반영\n")
        else:
            L.append("\n실험 모델을 넣어도 최종 선택은 같음.\n")

    # 3 ------------------------------------------------------------------
    L.append("## 3. 영향요인 및 오류분석\n")
    for p, c in choices.items():
        dt, info = exp.get(f"defect_type_detection_{p}"), exp.get(f"_defect_type_info_{p}")
        if dt is None or not len(dt) or not info or not info["process_wide_shot_idx"]:
            continue
        L.append(f"### 3-0. 핵심 발견 ({p.upper()}): 공정변수는 'shot 전체 이상'은 잡고 '캐비티 국소 불량'은 못 잡는다\n")
        L.append(f"`{c['model']}` 의 교차검증 예측에서 불량 유형별로 잡힌 행 수 (반복당 평균). "
                 "`process_wide` = 짝(LH/RH) 둘 다 불량, `cavity_local` = 한쪽만 불량\n")
        L.append(md(dt, floatfmt=1))
        pw = info["process_wide_shot_idx"]
        blk = dt[(dt.cv_mode == "block") & (dt.defect_type == "process_wide")]
        L.append(f"\n- block CV(처음 보는 구간)에서 상위 2%·5%·점수 0.5 이상으로 잡힌 불량은 **짝 둘 다 불량인 shot "
                 f"{len(pw)}개({int(blk.defect_rows.iloc[0]) if len(blk) else '?'}행)뿐**이고, 한쪽만 불량인 행은 잡히지 않음 "
                 "→ 4-1 표에서 TP 가 약 6행에서 멈추는 이유")
        L.append("- shot CV 에서 한쪽만 불량인 행이 일부 잡히는 것은 같은 불량 구간의 이웃 shot 이 학습에 들어간 경우로 보임 "
                 "(block CV 에서는 사라짐)")
        L.append(f"- **한계**: 근거가 되는 '짝 둘 다 불량' shot 은 {len(pw)}개(순번 {pw[0]}~{pw[-1]}"
                 + (", 연속" if info["consecutive"] else "") + ")뿐 — 한 번의 공정 이상 구간일 수 있어 일반화에 주의")
        only_local = [q for q, a in audit.items() if a["defect_shots"] and not a["process_wide_shots(양쪽 불량)"]]
        link = ("; ".join(f"{q.upper()} 는 불량 shot {audit[q]['defect_shots']}개가 모두 '한쪽만 불량'" for q in only_local)
                + " → 신호가 없는 것과 같은 이유로 연결됨. ") if only_local else ""
        if cfg.get("external", {}).get("enabled"):
            cav = {q: f["defects_by_cavity"] for q, f in known_facts.PUBLIC_VERIFIED.items()}
            one_side = [q.upper() for q, v in cav.items() if min(v.values()) == 0]
            L.append(f"- {link}공개 원본 연결 결과(해석용, {known_facts.SOURCE}): 불량의 캐비티 분포 "
                     + ", ".join(f"{q.upper()} {v}" for q, v in cav.items())
                     + (f" — {', '.join(one_side)} 는 불량이 한 캐비티에만 있음. " if one_side else ". ")
                     + "**공정변수는 shot 전체 이상을 알려주지만 캐비티 국소 불량은 알려주지 못한다.**\n")
        else:
            L.append(f"- {link}**공정변수는 shot 전체 이상을 알려주지만 캐비티 국소 불량은 알려주지 못한다.** "
                     "(캐비티 LH/RH 분포는 공개 원본 연결로만 확인 가능 — external.enabled=false 라 인용하지 않음)\n")
    L.append("### 3-1. 불량 유형별 탐지율 (샷 단위, 반복 평균)\n")
    L.append("`pw` = " + TYPE_KO["process_wide"] + ", `cl` = " + TYPE_KO["cavity_local"]
             + ". `thr` = 판정 임계값 기준, `@5%` = 위험 상위 5% 샷만 검사할 때\n")
    if len(type_tab):
        tt = type_tab.rename(columns=lambda c: c.replace("process_wide_", "pw_").replace("cavity_local_", "cl_"))
        keep = [k for k in ["pw_recall_thr", "pw_recall@5%", "cl_recall_thr", "cl_recall@5%", "cl_recall@10%"]
                if k in tt.columns]
        piv = tt.pivot_table(index=["product", "dataset", "model"], columns="cv_mode", values=keep, aggfunc="first")
        piv.columns = [f"{c}[{m}]" for c, m in piv.columns]
        piv = piv[[f"{c}[{m}]" for c in keep for m in modes if f"{c}[{m}]" in piv.columns]]
        n_info = tt.groupby("product")[["pw_n", "cl_n"]].first()
        L.append("불량 샷 수: " + ", ".join(f"{p.upper()} pw {int(r.pw_n)} / cl {int(r.cl_n)}" for p, r in n_info.iterrows()) + "\n")
        L.append(md(piv.reset_index()))
    for p, e in err.items():
        s, imp, bins, fn_cases, inter, by_type = e
        c = choices[p]
        L.append(f"\n### 3-2. {p.upper()} 상세 (`{c['model']}`, {s['cv_mode']} CV, 반복 0)\n")
        if not c["beats_chance"]:
            L.append("> 이 제품은 무작위 기준선을 넘지 못했으므로 아래 중요도·구간 분석은 '신호 없음'의 근거로만 사용.\n")
        L.append("**불량 유형별 판정 결과**\n")
        L.append(md(by_type, index=True))
        L.append(f"\n- 판정 결과: {s['outcomes']}")
        L.append(f"- FP 분해: {s['fp_breakdown']}  ← `twin_FP`는 짝의 다른 쪽이 불량이라 입력만으로는 구분 불가능한 FP\n")
        L.append(f"![](figures/timeline_{p}.png)\n")
        L.append("**영향 변수 (permutation importance, AP 감소량)**\n")
        L.append(md(imp.head(10), index=True))
        L.append("\n**FN/FP가 몰리는 공정조건 (상위 변수 5분위, 표준화 단위)**\n")
        L.append(md(bins))
        L.append("\n**상위 2개 변수 상호작용 (3×3 구간)**\n")
        L.append(md(inter))
        L.append("\n**미탐(FN) 사례: 정상 중앙값 대비 편차 (표준화 단위)**\n")
        L.append(md(fn_cases))
        L.append(f"\n![](figures/errors_{p}.png)\n")

    # 4 ------------------------------------------------------------------
    L.append("## 4. 현장 활용 — 검사 우선순위와 예측 파일\n")
    L.append(md(pd.DataFrame(final_summ)))
    L.append(f"\n- 판정 기준(`pred_label`): {final_summ[0]['decision_rule'] if final_summ else ''}. "
             "임계값은 교차검증 점수 분포에서 정하므로 새 샷을 바로 판정할 수 있음")
    L.append(f"- 위험 등급: 교차검증 점수 기준 상위 {cfg['risk_tiers']['inspect_now_top']:.0%}에 해당하는 점수 이상 `즉시검사`, "
             f"상위 {cfg['risk_tiers']['priority_top']:.0%} 이상 `우선검사`, 나머지 `일반`. "
             "unlabeled에서 등급 비율이 2%/10%와 크게 다르면 운전 분포가 달라졌다는 신호")
    L.append("- **운영 제안 (3-0 근거)**: 공정 데이터 모델은 'shot 전체 이상(공정 조건 이탈)' 선별에 쓰고, "
             "캐비티 국소 불량(한쪽만 불량)은 금형·캐비티별 점검이나 비전 검사 등 **별도 수단**이 필요함")
    L.append("- 확률 보정(Platt)의 성능 확인용 내부 CV 도 shot 단위 그룹 분할 (LH/RH 짝이 갈라지지 않게)")
    L.append("- 모든 예측 파일에 `mold`(cn7/rg3) 열과 `uncertainty` 열: labeled CV 행은 반복별 OOF 점수의 표준편차, "
             "holdout·unlabeled 행은 fold 모델 전체 예측의 표준편차 (`uncertainty_n` = 개수)")
    L.append("- `predictions_labeled_cv.csv`: labeled 전 행의 교차검증 예측 (정답 포함 → 테스트 예측 파일로 사용 가능)")
    if holdout_info:
        L.append("- `predictions_holdout.csv`: 처음부터 떼어 둔 홀드아웃 블록 예측과 성능(위 표 test_*)")
    L.append("- `predictions/pipeline_<모델>.csv`: 모든 모델의 교차검증 예측을 팀 공통 형식"
             "(mold, product, model, cv_mode, repeat, fold, row_id, shot_id, y_true, y_score, y_pred)으로 저장 → "
             "`python run_evaluation.py` 로 같은 기준 채점·검산")
    L.append("- `predictions_unlabeled/predictions_unlabeled.csv`: unlabeled 예측 (정답이 없어 평가 폴더와 분리). "
             "`in_labeled_range`=False인 행은 학습 범위 밖 → 신뢰도 낮음")
    L.append(f"  - ⚠ 대회 unlabeled 안에 labeled 행이 들어 있음 ({known_facts.overlap_text()}). "
             "`in_labeled_train_row`(행 자체) / `in_labeled_train_shot`(같은 shot 의 짝 포함)=True 행은 학습에 쓰인 행이라 "
             "uncertainty 가 과소평가될 수 있음. "
             "대회 파일만으로는 판정할 수 없어 기본값은 NaN (external.enabled=true + 검증 결과가 있을 때만 채움). "
             "자세한 경고는 `predictions_unlabeled/predictions_unlabeled_meta.json`")
    if any(a["scale_check"]["both_files_mean0_std1"] and a["scale_check"]["value_overlap_mean"] < 0.5 for a in audit.values()):
        L.append("  - ⚠ labeled/unlabeled가 파일별로 따로 표준화된 것으로 보임 → unlabeled 예측은 참고용")
    for p in cfg.get("experiments", {}).get("threshold_products", []):
        tt, bt = exp.get(f"exp_threshold_{p}"), exp.get(f"exp_budget_{p}")
        if tt is None:
            continue
        mdl = tt.model.iloc[0]
        L.append(f"\n### 4-1. {p.upper()} 최종 모델(`{mdl}`) 판정 기준별 결과 — 교차검증 OOF (개수는 반복당 평균)\n")
        L.append("**고정 임계값 (모델 점수 기준, 보정 전)**\n")
        for mode in modes:
            L.append(f"\n`{mode}` CV\n")
            L.append(md(_cols(tt[tt.cv_mode == mode], ["threshold", "TP", "FP", "FN", "TN", "검사건수",
                                                       "precision", "recall"]), floatfmt=2))
        L.append("\n**검사비율(budget): 점수 상위 k% 만 검사**\n")
        for mode in modes:
            L.append(f"\n`{mode}` CV\n")
            L.append(md(_cols(bt[bt.cv_mode == mode], ["budget", "검사건수", "TP", "FP", "FN", "precision", "recall"]),
                        floatfmt=2))
        L.append("\n- block CV 에서는 임계값·검사비율을 바꿔도 TP 가 6행 근처에서 멈춤 — 처음 보는 구간에서는 일부 불량 유형만 잡힘 "
                 "(유형별로는 3-1 표 참고)")
        L.append("- 고정 임계값은 모델 점수(보정 전) 기준이라 0.5 가 '불량 확률 50%'를 뜻하지 않음. 운영 판정은 검사비율 기준(config.decision)")
        L.append(f"\n{CEIL}")
    L.append("\n![](figures/inspection_curve.png)\n")
    for p in choices:
        L.append(f"![](figures/calibration_{p}.png)")

    # 5 ------------------------------------------------------------------
    L.append("\n## 5. 재현성\n")
    L.append("```json\n" + json.dumps(manifest, ensure_ascii=False, indent=1) + "\n```\n")
    path.write_text("\n".join(L), encoding="utf-8")
