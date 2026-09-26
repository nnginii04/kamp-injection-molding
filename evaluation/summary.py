"""save_experiment_result(): 예측 CSV 폴더를 읽어 결과표를 자동으로 만든다.

만드는 파일 (out_dir, 기본 outputs/evaluation/)
  experiment_summary.csv   모델 × 제품 × (cv_mode) × 단위(행/샷)마다 한 줄: 평균±편차 지표 + Recall@K
  experiment_summary.md    보고서에 붙일 요약표
  recall_at_k_detail.csv   K별 상세 (무작위 기준선, 잡은/놓친 불량 수, 유형별 재현율)
  threshold_detail.csv     임계값·검사 비율별 F1/Recall/Precision/FN/FP
  evaluation_manifest.json 입력 파일 해시·행 수·설정·버전 (재현성)
"""
from __future__ import annotations

import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import GROUP_KEYS, load_predictions, pairs_split_across_folds, sha256
from .metrics import evaluate_model
from .recall_at_k import DEFAULT_KS, calculate_recall_at_k
from .threshold import DEFAULT_BUDGETS, DEFAULT_THRESHOLDS, evaluate_thresholds

LEAD = ["file", *GROUP_KEYS, "level", "decision", "std_over", "n_runs", "n", "n_pos", "prevalence",
        "pr_auc_mean", "pr_auc_std", "pr_auc_lift", "roc_auc_mean", "roc_auc_std", "f1_mean", "f1_std",
        "precision_mean", "precision_std", "recall_mean", "recall_std", "fn_mean", "fp_mean", "tp_mean",
        "flag_rate_mean"]


def _fmt(m, s=None, d=3):
    if m is None or pd.isna(m):
        return "-"
    return f"{m:.{d}f}" if s is None or pd.isna(s) else f"{m:.{d}f} ± {s:.{d}f}"


def _s(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) == "nan" else str(v)


def _markdown(summary: pd.DataFrame, ks, notes, n_files) -> str:
    L = ["# 실험 결과 요약 (자동 생성)", "",
         f"- 생성: {time.strftime('%Y-%m-%d %H:%M')} · 예측 파일 {n_files}개",
         "- 평균 ± 편차 = 교차검증 반복(repeat)별 값의 평균과 표준편차 (없으면 fold별, 그것도 없으면 1회 값)",
         "- 무작위 기준선: PR-AUC ≈ 불량률, Recall@K(R@K) ≈ K. 괄호 안 배수 = PR-AUC ÷ 불량률",
         "- FN·FP: repeat 단위면 전체 1회분의 평균, fold 단위면 fold 합계 (둘 다 '데이터 전체를 한 번 예측했을 때' 기준)", ""]
    labels = [f"{k * 100:g}%" for k in ks]
    for level, title in (("shot", "샷 단위 (LH/RH 짝 = 한 샷, 짝 중 하나라도 불량이면 불량 샷)"),
                         ("row", "행 단위 (제품 하나 = 한 행)")):
        sub = summary[summary["level"].eq(level)]
        if sub.empty:
            continue
        keys = [c for c in GROUP_KEYS if c in sub.columns and sub[c].notna().any() and c != "model"]
        if "dataset" in keys and "product" in sub.columns and (
                sub["dataset"].isna() | sub["dataset"].astype(str).eq(sub["product"].astype(str))).all():
            keys.remove("dataset")  # 제품별 학습이면 dataset = product라 생략
        L += [f"## {title}", ""]
        head = [*keys, "model", "불량/전체", "PR-AUC (배수)", "ROC-AUC", "판정", "F1", "FN", "FP",
                *[f"R@{x}" for x in labels]]
        L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for _, r in sub.iterrows():
            cells = [_s(r[k]) for k in keys] + [str(r["model"]), f"{int(r['n_pos'])}/{int(r['n'])}",
                     f"{_fmt(r['pr_auc_mean'], r['pr_auc_std'])} ({r['pr_auc_lift']:.1f}x)",
                     _fmt(r["roc_auc_mean"]), str(r["decision"]), _fmt(r["f1_mean"], r["f1_std"]),
                     _fmt(r["fn_mean"], d=1), _fmt(r["fp_mean"], d=1)]
            cells += [_fmt(r.get(f"recall@{x}_mean")) for x in labels]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
    if notes:
        L += ["## 확인 필요", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(L)


def save_experiment_result(pred_dir="outputs/predictions", out_dir="outputs/evaluation", *, files=None,
                           levels=("row", "shot"), ks=DEFAULT_KS, threshold=None, budget=None,
                           thresholds=DEFAULT_THRESHOLDS, budgets=DEFAULT_BUDGETS, shot_map="auto",
                           run_col="auto", verbose=True) -> pd.DataFrame:
    """pred_dir의 *.csv(또는 files 목록)를 모두 평가해 out_dir에 결과표를 쓰고, 요약 DataFrame을 돌려준다.

    threshold / budget : F1·FN·FP를 계산할 판정 기준 (둘 다 없으면 파일의 y_pred, 없으면 점수 ≥ 0.5)
    shot_map           : 'auto'면 out_dir 상위 폴더의 predictions_labeled_cv.csv(run_all.py 결과)를 찾아
                         row_id로 shot_id·불량 유형을 붙인다 (예측 파일에 shot_id가 없을 때)
    """
    out_dir = Path(out_dir)
    paths = [Path(p) for p in files] if files else sorted(Path(pred_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"예측 CSV가 없습니다: {pred_dir}")
    if isinstance(shot_map, str) and shot_map == "auto":
        cand = out_dir.parent / "predictions_labeled_cv.csv"
        shot_map = cand if cand.exists() else None
    out_dir.mkdir(parents=True, exist_ok=True)

    summ, kdet, tdet, mani, notes = [], [], [], [], []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for path in paths:
            df = load_predictions(path, shot_map=shot_map)
            keys = [c for c in GROUP_KEYS if c in df.columns]
            groups = df.groupby(keys, sort=True, dropna=False) if keys else [((), df)]
            n_split = 0
            for kv, g in groups:
                kv = kv if isinstance(kv, tuple) else (kv,)
                ident = {"file": path.name, **dict(zip(keys, kv))}
                split = pairs_split_across_folds(g)
                n_split = max(n_split, split)
                for level in levels:
                    if level == "shot" and "shot_id" not in g.columns:
                        continue
                    m = evaluate_model(g, threshold=threshold, budget=budget, level=level, run_col=run_col)
                    rk = calculate_recall_at_k(g, ks=ks, level=level, run_col=run_col)
                    th = evaluate_thresholds(g, thresholds=thresholds, budgets=budgets, level=level,
                                             run_col=run_col)
                    row = {**ident, **m, "pairs_split_across_folds": split}
                    for _, r in rk.iterrows():
                        row[f"recall@{r.k_label}_mean"] = r.recall_mean
                        row[f"recall@{r.k_label}_std"] = r.recall_std
                        row[f"lift@{r.k_label}"] = r.lift
                    summ.append(row)
                    kdet.append(rk.assign(**ident))
                    tdet.append(th.assign(**ident))
            if "shot_id" not in df.columns:
                notes.append(f"{path.name}: shot_id가 없어 샷 단위 평가를 건너뜀 (row_id를 넣으면 run_all.py 결과로 자동 연결)")
            if n_split:
                notes.append(f"{path.name}: LH/RH 짝이 서로 다른 fold로 나뉜 경우가 있음(최대 {n_split}샷) → 같은 입력이 "
                             "학습·검증에 동시에 들어가 성능이 부풀려질 수 있음. shot_id로 묶어 나누기(StratifiedGroupKFold) 권장")
            mani.append({"file": str(path), "sha256": sha256(path), "rows": int(len(df)),
                         "columns": [str(c) for c in df.columns if c != "source_file"]})
    for w in caught:
        msg = str(w.message)
        if msg not in notes:
            notes.append(msg)

    summary = pd.DataFrame(summ)
    lead = [c for c in LEAD if c in summary.columns]
    summary = summary[lead + [c for c in summary.columns if c not in lead]]
    sort = [c for c in ("product", "level", "cv_mode") if c in summary.columns]
    summary = summary.sort_values(sort + ["pr_auc_mean"], ascending=[True] * len(sort) + [False],
                                  kind="stable").reset_index(drop=True)
    summary.to_csv(out_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(kdet, ignore_index=True).to_csv(out_dir / "recall_at_k_detail.csv", index=False, encoding="utf-8-sig")
    pd.concat(tdet, ignore_index=True).to_csv(out_dir / "threshold_detail.csv", index=False, encoding="utf-8-sig")
    (out_dir / "experiment_summary.md").write_text(_markdown(summary, ks, notes, len(paths)), encoding="utf-8")
    import sklearn
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "settings": {"levels": list(levels), "ks": list(ks), "threshold": threshold, "budget": budget,
                             "thresholds": [float(t) for t in thresholds], "budgets": list(budgets),
                             "run_col": run_col, "shot_map": str(shot_map) if shot_map is not None else None},
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
                "inputs": mani, "notes": notes}
    (out_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
    if verbose:
        print(f"[평가] 예측 파일 {len(paths)}개 → {out_dir / 'experiment_summary.csv'} ({len(summary)}줄)")
        for n in notes:
            print("  ! " + n)
    return summary
