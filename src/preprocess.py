"""전처리: 중복 제거, 상수열 제거, 금형 병합."""
import pandas as pd
from pathlib import Path


def load_config(path="configs/config.yaml"):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess(cfg):
    raw = Path(cfg["paths"]["raw"])
    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for mold, fname in cfg["data"]["labeled"].items():
        df = pd.read_csv(raw / fname)
        df["mold"] = mold
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df = df.drop(columns=[c for c in cfg["data"]["drop_cols"] if c in df.columns])
    if cfg["data"]["drop_duplicates"]:
        before = len(df)
        df = df.drop_duplicates()
        print(f"[preprocess] duplicates removed: {before} -> {len(df)}")

    df.to_csv(out / "labeled.csv", index=False)
    print(f"[preprocess] saved {out / 'labeled.csv'} shape={df.shape}")
    return df


if __name__ == "__main__":
    preprocess(load_config())
