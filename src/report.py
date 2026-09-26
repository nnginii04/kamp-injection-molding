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


def write(path, cfg, audit, dropped, table, ci, choices, final_summ, err, type_tab, manifest, holdout_info=None,
          fold_check=None, ablation=None):
    modes = cfg["cv"]["modes"]
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
                 "→ unlabeled 로 학습하거나 unlabeled 예측을 평가에 쓰지 않음\n")
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
                     "'분할 운'의 일부(정상 블록 배분)만 반영함\n")
    L.append("\n".join(f"- `{k}`: {v}" for k, v in MODEL_DESC.items() if k in cfg["models"]) + "\n")
    L.append("### 2-1. 핵심 지표 (샷 CV vs 블록 CV)\n")
    L.append("`row_f1` = 학습 fold 내부 F1 최적 임계값 기준, `row_f1@2%` = 위험 상위 2%만 불량 판정했을 때 "
             "(블록 CV에서는 임계값이 불안정하므로 `@2%` 쪽이 운영 판단에 가까움)\n")
    pv = _pivot(table, ["shot_ap", "shot_ap_sd", "row_ap", "row_f1", "row_f1@2%", "row_recall@10%"], modes)
    sort_col = f"shot_ap[{sel['cv_mode']}]" if f"shot_ap[{sel['cv_mode']}]" in pv.columns else pv.columns[3]
    L.append(md(pv.sort_values(["product", sort_col], ascending=[True, False])))
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

    # 3 ------------------------------------------------------------------
    L.append("## 3. 영향요인 및 오류분석\n")
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
    L.append("- 확률 보정(Platt)의 성능 확인용 내부 CV 도 shot 단위 그룹 분할 (LH/RH 짝이 갈라지지 않게)")
    L.append("- 모든 예측 파일에 `mold`(cn7/rg3) 열과 `uncertainty` 열: labeled CV 행은 반복별 OOF 점수의 표준편차, "
             "holdout·unlabeled 행은 fold 모델 전체 예측의 표준편차 (`uncertainty_n` = 개수)")
    L.append("- `predictions_labeled_cv.csv`: labeled 전 행의 교차검증 예측 (정답 포함 → 테스트 예측 파일로 사용 가능)")
    if holdout_info:
        L.append("- `predictions_holdout.csv`: 처음부터 떼어 둔 홀드아웃 블록 예측과 성능(위 표 test_*)")
    L.append("- `predictions_unlabeled.csv`: unlabeled 예측. `in_labeled_range`=False인 행은 학습 범위 밖 → 신뢰도 낮음")
    L.append(f"  - ⚠ 대회 unlabeled 안에 labeled 행이 들어 있음 ({known_facts.overlap_text()}). "
             "`in_labeled_train`=True 행은 학습에 쓰인 행이라 uncertainty 가 과소평가될 수 있음. "
             "대회 파일만으로는 판정할 수 없어 기본값은 NaN (external.enabled=true + 검증 결과가 있을 때만 채움). "
             "자세한 경고는 `predictions_unlabeled_meta.json`")
    if any(a["scale_check"]["both_files_mean0_std1"] and a["scale_check"]["value_overlap_mean"] < 0.5 for a in audit.values()):
        L.append("  - ⚠ labeled/unlabeled가 파일별로 따로 표준화된 것으로 보임 → unlabeled 예측은 참고용")
    L.append("\n![](figures/inspection_curve.png)\n")
    for p in choices:
        L.append(f"![](figures/calibration_{p}.png)")

    # 5 ------------------------------------------------------------------
    L.append("\n## 5. 재현성\n")
    L.append("```json\n" + json.dumps(manifest, ensure_ascii=False, indent=1) + "\n```\n")
    path.write_text("\n".join(L), encoding="utf-8")
