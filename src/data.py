"""데이터 로드, 스키마 검증, 샷(shot) 그룹 부여.

핵심 가정
- 한 번의 사출(shot)에서 LH/RH 두 제품이 동시에 나오며, 두 행은 공정값이 완전히 같다.
  → 공정값이 완전히 같은 행을 같은 shot_id로 묶는다. 교차검증에서 같은 shot이 학습/검증에
    동시에 들어가면 '정답을 보고 푸는' 누수가 생기므로 반드시 그룹 단위로 분할한다.
- 라벨 1 = 불량(소수 클래스)으로 가정한다. 가이드북으로 확인 필요 (config.positive_label).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. KAMP 공지의 '1. 사출성형기 AI 데이터셋.zip' 압축을 풀어 "
            f"data/ 폴더에 넣으세요 (README 참고)."
        )
    for enc in ("utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path} 인코딩을 읽을 수 없습니다.")


def load_all(cfg: dict, data_dir: Path) -> dict:
    """제품별 labeled/unlabeled DataFrame과 메타정보를 반환."""
    id_col, label_col = cfg["id_col"], cfg["label_col"]
    out = {"labeled": {}, "unlabeled": {}, "hashes": {}, "feature_cols": None}
    feature_sets = []
    for product, files in cfg["products"].items():
        lab_path, unl_path = data_dir / files["labeled"], data_dir / files["unlabeled"]
        lab, unl = _read(lab_path), _read(unl_path)
        out["hashes"][files["labeled"]] = sha256(lab_path)
        out["hashes"][files["unlabeled"]] = sha256(unl_path)

        assert label_col in lab.columns, f"{files['labeled']}에 {label_col} 열이 없습니다."
        if id_col not in lab.columns:  # 인덱스 열 이름이 다른 경우 대비
            lab.insert(0, id_col, np.arange(len(lab)))
        if id_col not in unl.columns:
            unl.insert(0, id_col, np.arange(len(unl)))

        feats = [c for c in lab.columns if c not in (id_col, label_col)]
        missing_in_unl = set(feats) - set(unl.columns)
        assert not missing_in_unl, f"unlabeled에 없는 열: {missing_in_unl}"
        feature_sets.append(tuple(feats))

        labels = set(lab[label_col].unique())
        assert labels <= {0, 1}, f"라벨 값이 0/1이 아닙니다: {labels}"

        lab = lab.rename(columns={id_col: "row_id"})
        unl = unl.rename(columns={id_col: "row_id"})
        lab["y"] = (lab[label_col] == cfg["positive_label"]).astype(int)
        lab = lab.drop(columns=[label_col])
        lab["product"], unl["product"] = product, product
        lab["source"], unl["source"] = "labeled", "unlabeled"
        out["labeled"][product] = lab
        out["unlabeled"][product] = unl[["row_id", *feats, "product", "source"]]

    assert len(set(feature_sets)) == 1, "제품별 파일의 열 구성이 다릅니다."
    out["feature_cols"] = list(feature_sets[0])
    return out


def assign_shots(df: pd.DataFrame, feature_cols: list[str], prefix: str) -> pd.DataFrame:
    """공정값이 완전히 같은 행을 같은 shot으로 묶는다 (LH/RH 짝)."""
    df = df.copy()
    key = pd.util.hash_pandas_object(df[feature_cols].round(10), index=False)
    codes, _ = pd.factorize(key)
    df["shot_id"] = [f"{prefix}_{c}" for c in codes]
    df["shot_size"] = df.groupby("shot_id")["row_id"].transform("size")
    return df


def shot_table(df: pd.DataFrame) -> pd.DataFrame:
    """shot 단위 라벨: 짝 중 하나라도 불량이면 불량 샷."""
    g = df.groupby("shot_id")
    t = g.agg(product=("product", "first"), n_rows=("y", "size"), n_pos=("y", "sum"),
              first_row=("row_id", "min"), last_row=("row_id", "max"))
    t["y_shot"] = (t["n_pos"] > 0).astype(int)
    t["conflict"] = (t["n_pos"] > 0) & (t["n_pos"] < t["n_rows"])
    return t.reset_index()
