"""전처리.

- 상수열 제거 (Clamp_Open_Position)
- shot_id 부여: 같은 금형에서 공정 변수 24개가 완전히 같은 연속 행 = 같은 shot (2캐비티 LH/RH)
- cavity_pos: shot 안에서 몇 번째 행인지 (1 또는 2). 짝 없는 행은 1, shot_size=1
- mold_cn7: 금형 이진 변수 (통합 모델용)
- is_idle (unlabeled만): 6개 센서가 동시에 각 열의 최솟값이면 미기록(대기) 행
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loading import feature_columns, load_config, load_labeled, load_unlabeled


def assign_shots(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """파일 순서대로 보면서 직전 행과 (금형 + 공정값)이 완전히 같으면 같은 shot으로 묶는다."""
    df = df.copy()
    vals = df[feat_cols].to_numpy()
    molds = df["mold"].to_numpy()
    same_as_prev = np.zeros(len(df), dtype=bool)
    same_as_prev[1:] = (vals[1:] == vals[:-1]).all(axis=1) & (molds[1:] == molds[:-1])

    shot_id = np.empty(len(df), dtype=int)
    cavity_pos = np.empty(len(df), dtype=int)
    sid, pos = -1, 0
    for i, same in enumerate(same_as_prev):
        if same and pos == 1:          # 직전 행과 같고 그 행이 shot의 첫 행이면 → 두 번째 캐비티
            pos = 2
        else:                          # 새 shot 시작 (3행 이상 같은 값이 이어져도 2행씩 끊는다)
            sid += 1
            pos = 1
        shot_id[i] = sid
        cavity_pos[i] = pos

    df["shot_id"] = shot_id
    df["cavity_pos"] = cavity_pos
    df["shot_size"] = df.groupby("shot_id")["shot_id"].transform("size")
    return df


def add_mold_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mold_cn7"] = (df["mold"] == "cn7").astype(int)
    return df


def flag_idle(df: pd.DataFrame, idle_cols: list[str]) -> pd.DataFrame:
    """6개 센서가 동시에 열 최솟값이면 미기록 행. 금형별로 최솟값이 다르므로 금형 안에서 판단."""
    df = df.copy()
    idle = np.ones(len(df), dtype=bool)
    for mold, g in df.groupby("mold"):
        m = np.ones(len(g), dtype=bool)
        for c in idle_cols:
            m &= np.isclose(g[c].to_numpy(), g[c].min())
        idle[g.index.to_numpy()] = m
    df["is_idle"] = idle.astype(int)
    return df


def preprocess(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)

    lab = load_labeled(cfg)
    unl = load_unlabeled(cfg)
    for c in cfg["data"]["drop_cols"]:
        lab = lab.drop(columns=c, errors="ignore")
        unl = unl.drop(columns=c, errors="ignore")

    feat = feature_columns(lab, cfg)
    feat = [c for c in feat if c != "mold_cn7"]

    lab = assign_shots(lab, feat)
    lab = add_mold_feature(lab)
    unl = assign_shots(unl, feat)
    unl = add_mold_feature(unl)
    unl = flag_idle(unl, cfg["data"]["idle_sensor_cols"])

    lab.to_csv(out / "labeled.csv", index=False)
    unl.to_csv(out / "unlabeled.csv", index=False)

    y = cfg["data"]["target"]
    n_shots = lab["shot_id"].nunique()
    n_pairs = int((lab.drop_duplicates("shot_id")["shot_size"] == 2).sum())
    print(f"[preprocess] labeled rows={len(lab)} shots={n_shots} (pairs={n_pairs}, singles={n_shots - n_pairs}) "
          f"defect rows={int(lab[y].sum())} defect shots={lab.groupby('shot_id')[y].max().sum()}")
    print(f"[preprocess] unlabeled rows={len(unl)} shots={unl['shot_id'].nunique()} idle rows={int(unl['is_idle'].sum())} "
          f"({unl['is_idle'].mean() * 100:.1f}%)")
    print(f"[preprocess] features={len(feat)} -> {out / 'labeled.csv'}, {out / 'unlabeled.csv'}")
    return lab, unl


if __name__ == "__main__":
    preprocess(load_config())
