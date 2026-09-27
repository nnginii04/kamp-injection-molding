"""원본 CSV 4개 적재. 라벨/비라벨을 각각 하나의 DataFrame으로 합치고 금형(mold) 열을 붙인다."""
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read(raw_dir: Path, fname: str, mold: str, index_col: str) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / fname)
    df = df.rename(columns={index_col: "orig_index"})
    df.insert(0, "mold", mold)
    return df


def load_labeled(cfg: dict) -> pd.DataFrame:
    raw = Path(cfg["paths"]["raw"])
    frames = [_read(raw, f, mold, cfg["data"]["index_col"]) for mold, f in cfg["data"]["labeled"].items()]
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "row_id", range(len(df)))
    return df


def load_unlabeled(cfg: dict) -> pd.DataFrame:
    raw = Path(cfg["paths"]["raw"])
    frames = [_read(raw, f, mold, cfg["data"]["index_col"]) for mold, f in cfg["data"]["unlabeled"].items()]
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "row_id", range(len(df)))
    return df


def feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """모델 입력 열 = 공정 변수 전부 (메타·타깃 제외, 상수열 제외)."""
    meta = {"row_id", "orig_index", "mold", "shot_id", "cavity_pos", "shot_size", "is_idle", cfg["data"]["target"]}
    return [c for c in df.columns if c not in meta and c not in cfg["data"]["drop_cols"]]
