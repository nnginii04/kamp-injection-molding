"""팀 공통 평가 실행 — 실행 버튼만 누르면 됩니다.

읽는 것
  outputs/predictions/*.csv   ← run_all.py 의 교차검증 예측(pipeline_<모델>.csv) + 팀원 모델의 예측 파일
                                 (형식: evaluation/README.md). y_true 열이 없는 CSV 는 건너뛰고 이름을 알려 줌
                                 unlabeled 예측은 정답이 없으므로 outputs/predictions_unlabeled/ 에 따로 있고 여기서 읽지 않음
만드는 것 (outputs/evaluation/)
  experiment_summary.csv / .md (cv_mode 별로 표를 나눔), recall_at_k_detail.csv, threshold_detail.csv,
  evaluation_manifest.json, crosscheck.csv (run_all.py 결과와 report.md 수치 검산)

판정 기준 (F1·FN·FP 계산용) — 둘 중 하나만 지정
  --threshold T   점수 ≥ T 이면 불량 판정
  --budget B      점수 상위 B 비율(예: 0.02)만 불량 판정
  (둘 다 없으면) 파일의 y_pred(모델이 낸 판정; run_all.py 는 학습 fold 내부 F1 최적 임계값) → y_pred 가 없으면 점수 ≥ 0.5

옵션 예시
  python run_evaluation.py --budget 0.02
  python run_evaluation.py --no-pipeline                    # run_all.py 결과(pipeline_*.csv)는 빼고 팀 예측 파일만
  python run_evaluation.py --outputs outputs/synthetic      # 가짜 데이터 실행 결과로 동작 확인
경로 옵션(--outputs, --pred-dir, --out-dir)의 상대경로는 모두 이 파일이 있는 폴더(저장소 루트) 기준.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from evaluation import save_experiment_result  # noqa: E402
from evaluation.summary import NO_CV_MODE, lift_text  # noqa: E402

DEFAULT_RULE = "파일의 y_pred(모델 판정) — y_pred 가 없으면 점수 ≥ 0.5"
PIPELINE_PREFIX = "pipeline_"
# (단위, evaluation 열, report.md 의 근거인 outputs/tables/cv_summary.csv 열)
CROSS = [("row", "pr_auc_mean", "row_ap"), ("row", "pr_auc_std", "row_ap_sd"), ("row", "roc_auc_mean", "row_roc_auc"),
         ("row", "f1_mean", "row_f1"), ("row", "recall@10%_mean", "row_recall@10%"),
         ("shot", "pr_auc_mean", "shot_ap"), ("shot", "pr_auc_std", "shot_ap_sd"), ("shot", "f1_mean", "shot_f1"),
         ("shot", "recall@10%_mean", "shot_recall@10%")]
F1_METRICS = {"row_f1", "shot_f1"}


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).name


def crosscheck(summ: pd.DataFrame, cv_summary: Path, out_dir: Path, custom_rule: bool) -> pd.DataFrame | None:
    """pipeline_*.csv 평가 결과 ↔ run_all.py 가 report.md 에 쓴 수치(cv_summary.csv) 비교."""
    if not cv_summary.exists():
        return None
    cs = pd.read_csv(cv_summary)
    if "scenario" in cs.columns:
        cs = cs[cs.scenario.eq("separate")]
    ev = summ[summ["file"].astype(str).str.startswith(PIPELINE_PREFIX)]
    rows = []
    for _, r in ev.iterrows():
        ref = cs[cs.cv_mode.eq(r.get("cv_mode")) & cs["product"].eq(r.get("product")) & cs.model.eq(r.get("model"))]
        if ref.empty:
            continue
        for level, ecol, rcol in CROSS:
            if r["level"] != level or ecol not in r.index or rcol not in ref.columns:
                continue
            if custom_rule and rcol in F1_METRICS:  # 판정 기준을 바꾸면 F1 은 report 와 다른 기준
                continue
            a, b = float(ref[rcol].iloc[0]), float(r[ecol])
            rows.append({"product": r["product"], "cv_mode": r["cv_mode"], "model": r["model"], "metric": rcol,
                         "report(cv_summary)": a, "run_evaluation": b, "abs_diff": abs(a - b)})
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(out_dir / "crosscheck.csv", index=False, encoding="utf-8-sig")
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description="팀 공통 평가: 예측 CSV → 결과표")
    ap.add_argument("--outputs", default="outputs", help="결과 폴더 (기본 outputs, 저장소 루트 기준)")
    ap.add_argument("--pred-dir", default=None, help="예측 CSV 폴더 (기본 <outputs>/predictions)")
    ap.add_argument("--out-dir", default=None, help="결과 저장 폴더 (기본 <outputs>/evaluation)")
    ap.add_argument("--no-pipeline", action="store_true", help=f"run_all.py 결과({PIPELINE_PREFIX}*.csv)는 빼기")
    rule = ap.add_mutually_exclusive_group()
    rule.add_argument("--threshold", type=float, default=None, help="판정 기준: 점수 ≥ 이 값")
    rule.add_argument("--budget", type=float, default=None, help="판정 기준: 점수 상위 이 비율 (예: 0.02)")
    ap.add_argument("--levels", default="row,shot", help="평가 단위 (row, shot)")
    args = ap.parse_args()

    out_root = resolve(args.outputs)
    pred_dir = resolve(args.pred_dir) if args.pred_dir else out_root / "predictions"
    out_dir = resolve(args.out_dir) if args.out_dir else out_root / "evaluation"
    files = sorted(pred_dir.glob("*.csv")) if pred_dir.exists() else []
    if args.no_pipeline:
        files = [f for f in files if not f.name.startswith(PIPELINE_PREFIX)]
    if not files:
        pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"평가할 예측 파일이 없습니다.\n- 팀 예측 CSV를 {pred_dir} 에 넣거나\n- run_all.py를 먼저 실행하세요.")
        return 0

    if args.threshold is not None:
        rule_txt = f"점수 ≥ {args.threshold:g}"
    elif args.budget is not None:
        rule_txt = f"점수 상위 {args.budget * 100:g}%"
    else:
        rule_txt = f"{DEFAULT_RULE} (기본값: --threshold / --budget 미지정)"
    print(f"판정 기준(F1·FN·FP): {rule_txt}")
    print("평가 대상:")
    for f in files:
        print("  -", rel(f))
    shot_map = out_root / "predictions_labeled_cv.csv"
    summ = save_experiment_result(files=files, out_dir=out_dir, threshold=args.threshold, budget=args.budget,
                                  levels=tuple(x.strip() for x in args.levels.split(",") if x.strip()),
                                  shot_map=shot_map if shot_map.exists() else None, verbose=False)
    skipped = summ.attrs.get("skipped_files", [])
    if skipped:
        print("건너뛴 파일 (y_true 없음 — 정답이 없는 예측은 평가 대상이 아님): " + ", ".join(skipped))
    if not shot_map.exists():
        print(f"  ! {shot_map.name} 가 없어 전역 shot_id 를 붙일 수 없음 → 샷 단위 평가는 파일에 shot_id 가 있을 때만")

    cvm = summ["cv_mode"].fillna(NO_CV_MODE).astype(str) if "cv_mode" in summ.columns \
        else pd.Series(NO_CV_MODE, index=summ.index)
    keys = [c for c in ("product", "level", "model", "file") if c in summ.columns]
    with pd.option_context("display.max_rows", 300, "display.width", 220, "display.max_colwidth", 40):
        for mode in sorted(cvm.unique(), key=lambda m: (m == NO_CV_MODE, m)):
            sub = summ[cvm.eq(mode)]
            warn = "  ⚠ 검증 방식을 알 수 없음 — 다른 표와 직접 비교 금지" if mode == NO_CV_MODE else ""
            print(f"\n=== cv_mode = {mode}{warn} ===")
            view = sub[keys].copy()
            view["PR-AUC"] = [f"{m:.3f}" + ("" if pd.isna(s) else f" ± {s:.3f}")
                              for m, s in zip(sub.pr_auc_mean, sub.pr_auc_std)]
            view["배수"] = sub.pr_auc_lift.map(lift_text)
            for c in ("recall@10%_mean", "recall@20%_mean"):
                if c in sub.columns:
                    view[c.replace("_mean", "")] = sub[c].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
            view["F1"] = sub.f1_mean.map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
            view["FN"] = sub.fn_mean.map(lambda v: "-" if pd.isna(v) else f"{v:.1f}")
            print(view.to_string(index=False))
    for n in summ.attrs.get("notes", []):
        print("  ! " + n)
    print(f"\n자세한 표: {rel(out_dir / 'experiment_summary.md')}")

    cc = crosscheck(summ, out_root / "tables" / "cv_summary.csv", out_dir,
                    custom_rule=args.threshold is not None or args.budget is not None)
    if cc is not None and len(cc):
        mx = float(cc.abs_diff.max())
        ok = mx < 1e-9
        print(f"\n[검산] run_all.py report 수치(cv_summary.csv) vs run_evaluation: {len(cc)}개 값 비교, "
              f"최대 차이 {mx:.2e} → {'일치' if ok else '불일치 — crosscheck.csv 확인'}  ({rel(out_dir / 'crosscheck.csv')})")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
