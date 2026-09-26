"""팀 공통 평가 실행 — 실행 버튼만 누르면 됩니다.

읽는 것
  1) outputs/predictions/*.csv          ← A·B 모델의 예측 파일 (형식: evaluation/README.md)
  2) outputs/tables/oof_predictions.csv ← run_all.py의 교차검증 예측 (있으면 같이 평가 → report.md 숫자와 검산)
만드는 것 (outputs/evaluation/)
  experiment_summary.csv / .md, recall_at_k_detail.csv, threshold_detail.csv, evaluation_manifest.json

옵션 예시
  python run_evaluation.py --budget 0.02        # F1·FN·FP를 '상위 2% 검사' 기준으로
  python run_evaluation.py --threshold 0.3      # F1·FN·FP를 '점수 ≥ 0.3' 기준으로
  python run_evaluation.py --no-pipeline        # run_all.py 결과는 빼고 팀 예측 파일만
  python run_evaluation.py --outputs outputs/synthetic   # 가짜 데이터 실행 결과로 동작 확인
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from evaluation import save_experiment_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="팀 공통 평가: 예측 CSV → 결과표")
    ap.add_argument("--outputs", default="outputs", help="결과 폴더 (기본 outputs)")
    ap.add_argument("--pred-dir", default=None, help="예측 CSV 폴더 (기본 <outputs>/predictions)")
    ap.add_argument("--out-dir", default=None, help="결과 저장 폴더 (기본 <outputs>/evaluation)")
    ap.add_argument("--no-pipeline", action="store_true", help="run_all.py 교차검증 예측은 빼기")
    ap.add_argument("--threshold", type=float, default=None, help="판정 기준: 점수 ≥ 이 값")
    ap.add_argument("--budget", type=float, default=None, help="판정 기준: 점수 상위 이 비율 (예: 0.02)")
    ap.add_argument("--levels", default="row,shot", help="평가 단위 (row, shot)")
    args = ap.parse_args()

    out_root = ROOT / args.outputs
    pred_dir = Path(args.pred_dir) if args.pred_dir else out_root / "predictions"
    out_dir = Path(args.out_dir) if args.out_dir else out_root / "evaluation"
    files = sorted(pred_dir.glob("*.csv")) if pred_dir.exists() else []
    oof = out_root / "tables" / "oof_predictions.csv"
    if not args.no_pipeline and oof.exists():
        files.append(oof)
    if not files:
        pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"평가할 예측 파일이 없습니다.\n- A·B 예측 CSV를 {pred_dir} 에 넣거나\n- run_all.py를 먼저 실행하세요.")
        return

    print("평가 대상:")
    for f in files:
        print("  -", f.relative_to(ROOT) if f.is_relative_to(ROOT) else f)
    summ = save_experiment_result(files=files, out_dir=out_dir, threshold=args.threshold, budget=args.budget,
                                  levels=tuple(x.strip() for x in args.levels.split(",") if x.strip()),
                                  shot_map=out_root / "predictions_labeled_cv.csv"
                                  if (out_root / "predictions_labeled_cv.csv").exists() else None)

    keys = [c for c in ("file", "product", "cv_mode", "level", "model") if c in summ.columns]
    view = summ[keys].copy()
    view["PR-AUC"] = [f"{m:.3f}" + ("" if pd.isna(s) else f" ± {s:.3f}")
                      for m, s in zip(summ.pr_auc_mean, summ.pr_auc_std)]
    view["배수"] = summ.pr_auc_lift.map("{:.1f}x".format)
    for c in ("recall@10%_mean", "recall@20%_mean"):
        if c in summ.columns:
            view[c.replace("_mean", "")] = summ[c].map("{:.2f}".format)
    view["F1"] = summ.f1_mean.map("{:.3f}".format)
    view["FN"] = summ.fn_mean.map("{:.1f}".format)
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.max_colwidth", 40):
        print(view.to_string(index=False))
    print(f"\n자세한 표: {out_dir / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
