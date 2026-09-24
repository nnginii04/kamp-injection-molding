"""전처리 → 학습 → 평가 → 설명 → 추론을 한 번에 실행."""
import random
import numpy as np

from src.preprocess import load_config, preprocess
from src import train, evaluate, explain, predict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg["seed"])
    preprocess(cfg)
    train.run(cfg)
    evaluate.run(cfg)
    explain.run(cfg)
    predict.run(cfg)
    print("done. see outputs/")
