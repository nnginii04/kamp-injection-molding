"""전체 파이프라인 한 번에 실행 (재현성 제출용).

    python run_all.py                 # 전처리 → 분할 → 베이스라인 학습 → 점검 지표
    python run_all.py --skip-train    # 전처리·분할만 (fold_id.csv 재생성)
    python run_all.py --config configs/config.yaml

각 단계는 개별 실행도 된다:
    python -m src.data.preprocess
    python -m src.data.split
    python -m src.models.train
    python -m src.evaluation.metrics
"""
import argparse
import random
import time

import numpy as np

from src.data.loading import load_config
from src.data.preprocess import preprocess
from src.data.split import make_folds, save_folds
from src.evaluation.metrics import evaluate_baselines
from src.models.train import train_baselines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    t0 = time.time()

    print("=" * 70, "\n[1/4] preprocess")
    lab, _ = preprocess(cfg)
    print("=" * 70, "\n[2/4] split")
    save_folds(make_folds(lab, cfg), cfg)
    if args.skip_train:
        return
    print("=" * 70, "\n[3/4] train baselines")
    train_baselines(cfg)
    print("=" * 70, "\n[4/4] sanity metrics")
    evaluate_baselines(cfg)
    print(f"\n[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
