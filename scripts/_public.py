"""KAMP 공개 원본 접근 도우미 (scripts/ 전용, 읽기 전용).

- 경로: config.json 의 external.public_path (zip, 압축 푼 폴더, 또는 CSV 파일). `~` 허용.
- 허용: external.enabled=true 이거나 실행 시 --enable 을 줬을 때만 읽는다 (기본 끔).
- zip 은 풀지 않고 안의 파일을 바로 읽는다. 어떤 파일도 쓰거나 저장소로 복사하지 않는다.
- ⚠ 대회 규정 확인 전까지 공개 원본은 모델 입력 금지. 해석·검증 용도로만 쓴다.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "cp949")


def public_path(cfg: dict, override: str | None = None) -> Path:
    raw = override or cfg.get("external", {}).get("public_path")
    if not raw:
        raise SystemExit("config.json 의 external.public_path 가 비어 있습니다.")
    return Path(raw).expanduser()


def allowed(cfg: dict, enable_flag: bool) -> bool:
    if cfg.get("external", {}).get("enabled", False) or enable_flag:
        return True
    print("외부(KAMP 공개 원본) 데이터 사용이 꺼져 있습니다 (config.json external.enabled=false).\n"
          "해석·검증용으로 이번 한 번만 쓰려면 --enable 을 붙여 실행하세요. 모델 입력으로는 쓰지 않습니다.")
    return False


def _zip_member(z: zipfile.ZipFile, name: str) -> str:
    hits = [m for m in z.namelist() if m == name or m.endswith("/" + name)]
    if not hits:
        raise FileNotFoundError(f"zip 안에 {name} 이 없습니다: {z.filename}")
    return hits[0]


def has(src: Path, name: str) -> bool:
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            return any(m == name or m.endswith("/" + name) for m in z.namelist())
    if src.is_dir():
        return (src / name).exists()
    return src.name == name and src.exists()


def read_csv(src: Path, name: str, **kw) -> pd.DataFrame:
    """src(zip/폴더/CSV) 안의 name 을 읽는다. 인코딩은 순서대로 시도."""
    kw.setdefault("low_memory", False)
    if not src.exists():
        raise FileNotFoundError(f"공개 원본이 없습니다: {src}")
    for enc in ENCODINGS:
        try:
            if src.suffix.lower() == ".zip":
                with zipfile.ZipFile(src) as z, z.open(_zip_member(z, name)) as f:
                    return pd.read_csv(f, encoding=enc, **kw)
            path = src / name if src.is_dir() else src
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{src}:{name} 인코딩을 읽을 수 없습니다.")
