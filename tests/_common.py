"""테스트 공통 준비: 설정, 합성 데이터(항상), 대회 데이터(있으면)."""
from __future__ import annotations

import json
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import assign_shots, load_all  # noqa: E402
from src.synthetic import generate  # noqa: E402

RAW = ROOT / "data" / "raw"


def load_cfg() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def has_real_data() -> bool:
    cfg = load_cfg()
    return all((RAW / f[k]).exists() for f in cfg["products"].values() for k in ("labeled", "unlabeled"))


def _prepare(data_dir: Path) -> tuple[dict, dict]:
    cfg = load_cfg()
    data = load_all(cfg, data_dir)
    for p in data["labeled"]:
        data["labeled"][p] = assign_shots(data["labeled"][p], data["feature_cols"], p).reset_index(drop=True)
        data["unlabeled"][p] = assign_shots(data["unlabeled"][p], data["feature_cols"], f"{p}u")
    return cfg, data


@lru_cache(maxsize=1)
def synthetic():
    tmp = Path(tempfile.mkdtemp(prefix="kamp_test_"))
    return _prepare(generate(tmp / "data"))


@lru_cache(maxsize=1)
def real():
    return _prepare(RAW)
