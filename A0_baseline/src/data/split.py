"""분할 규칙 — 전원이 이 파일이 만든 fold_id.csv 하나만 쓴다.

- 단위: shot (같은 shot의 두 행은 절대 갈라지지 않는다)
- 홀드아웃 20%: shot 단위, (금형 × 불량 여부)로 층화. 모델 선택·임계값 결정에 쓰지 않는다
- 개발셋: StratifiedGroupKFold(5등분) × 5반복(시드만 다름), 역시 shot 단위 · 층화
- 결과: data/processed/fold_id.csv
    row_id, shot_id, mold, cavity_pos, holdout(0/1), fold_rep0 ... fold_rep4 (홀드아웃 행은 -1)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.loading import load_config


def make_folds(lab: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    y = cfg["data"]["target"]
    seed = cfg["seed"]
    n_splits = cfg["split"]["n_splits"]
    n_repeats = cfg["split"]["n_repeats"]
    holdout_frac = cfg["split"]["holdout_frac"]

    # shot 단위 라벨(한쪽이라도 불량이면 불량 shot)로 층화해야 쌍이 갈라지지 않으면서 불량이 고르게 퍼진다
    shot_y = lab.groupby("shot_id")[y].transform("max")
    strat = lab["mold"] + "_" + shot_y.astype(str)
    groups = lab["shot_id"].to_numpy()

    folds = lab[["row_id", "shot_id", "mold", "cavity_pos"]].copy()

    # 1) 홀드아웃: n_splits_h = round(1/holdout_frac) 등분 후 첫 조각
    n_h = int(round(1 / holdout_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_h, shuffle=True, random_state=seed)
    _, hold_idx = next(sgkf.split(lab, strat, groups))
    holdout = np.zeros(len(lab), dtype=int)
    holdout[hold_idx] = 1
    folds["holdout"] = holdout

    # 2) 개발셋 반복 K-fold
    dev_mask = holdout == 0
    dev = lab[dev_mask]
    for rep in range(n_repeats):
        col = np.full(len(lab), -1, dtype=int)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        dev_pos = np.where(dev_mask)[0]
        for k, (_, te) in enumerate(sgkf.split(dev, strat[dev_mask], groups[dev_mask])):
            col[dev_pos[te]] = k
        folds[f"fold_rep{rep}"] = col

    _check(folds, lab, y, n_repeats)
    return folds


def _check(folds: pd.DataFrame, lab: pd.DataFrame, y: str, n_repeats: int) -> None:
    # 쌍이 갈라지지 않았는지
    for col in ["holdout"] + [f"fold_rep{r}" for r in range(n_repeats)]:
        per_shot = folds.groupby("shot_id")[col].nunique()
        assert (per_shot == 1).all(), f"{col}: 같은 shot이 서로 다른 fold에 들어감"
    # 불량 분포 출력
    d = folds.merge(lab[["row_id", y]], on="row_id")
    print(f"[split] holdout rows={int(d.holdout.sum())} defect rows={int(d.loc[d.holdout == 1, y].sum())} | "
          f"dev rows={int((d.holdout == 0).sum())} defect rows={int(d.loc[d.holdout == 0, y].sum())}")
    dev = d[d.holdout == 0]
    for r in range(n_repeats):
        cnt = dev.groupby(f"fold_rep{r}")[y].sum().astype(int).tolist()
        print(f"[split] rep{r} fold별 불량 행 수: {cnt}")


def save_folds(folds: pd.DataFrame, cfg: dict) -> Path:
    out = Path(cfg["paths"]["processed"]) / "fold_id.csv"
    folds.to_csv(out, index=False)
    print(f"[split] saved {out}")
    return out


def load_folds(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(Path(cfg["paths"]["processed"]) / "fold_id.csv")


def iter_folds(folds: pd.DataFrame, rep: int):
    """(fold 번호, train row_id 배열, test row_id 배열)을 차례로 낸다. 홀드아웃은 제외."""
    col = f"fold_rep{rep}"
    dev = folds[folds["holdout"] == 0]
    for k in sorted(dev[col].unique()):
        te = dev.loc[dev[col] == k, "row_id"].to_numpy()
        tr = dev.loc[dev[col] != k, "row_id"].to_numpy()
        yield int(k), tr, te


if __name__ == "__main__":
    cfg = load_config()
    lab = pd.read_csv(Path(cfg["paths"]["processed"]) / "labeled.csv")
    save_folds(make_folds(lab, cfg), cfg)
