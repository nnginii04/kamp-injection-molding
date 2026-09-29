"""가이드북 준지도 실습(단계 ⑥, [코드 48])의 평가 세트를 그대로 재현해 가이드북 결과표([표 6]~[표 13])와 대조한다.

가이드북 [코드 48]: moldset_labeled_<제품>.csv 를 읽고 StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42).
가이드북 혼동행렬(p.84)의 평가 세트 구성: CN7 불량 56 · 양품 359, RG3 불량 83 · 양품 347.

확인하는 것
1) 같은 코드로 나눈 평가 세트의 불량·양품 수 → 양품 수가 가이드북 표와 정확히 같으면 같은 분할이다.
2) 평가 세트 불량이 파일 전체 불량보다 많은지 → 많으면 불량 행을 복제·증강한 뒤 평가했다는 뜻 (코드에는 없음).
3) 평가 세트 불량 행과 공정값이 완전히 같은 행이 학습 세트에 있는 비율 (LH/RH 쌍둥이 누수).

실행: python scripts/guidebook_split_check.py   → outputs/tables/guidebook_split_check.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
GUIDEBOOK_TABLE = {"cn7": {"test_defect": 56, "test_normal": 359}, "rg3": {"test_defect": 83, "test_normal": 347}}


def main():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    rows = []
    for p, files in cfg["products"].items():
        d = pd.read_csv(ROOT / cfg["data_dir"] / files["labeled"], low_memory=False, index_col=False)
        d = d.drop(columns=["Unnamed: 0"])
        X, y = d.loc[:, d.columns != "PassOrFail"], d["PassOrFail"]
        tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42).split(X, y))
        key = lambda df: df.round(8).astype(str).agg("|".join, axis=1)  # noqa: E731
        train_keys = set(key(X.iloc[tr]))
        te_def = X.iloc[te][y.iloc[te].to_numpy() == 1]
        g = GUIDEBOOK_TABLE[p]
        n_def_te = int(y.iloc[te].sum())
        rows.append({
            "product": p, "file_rows": len(d), "file_defects": int(y.sum()),
            "code48_test_rows": len(te), "code48_test_defects": n_def_te,
            "code48_test_normals": int((y.iloc[te] == 0).sum()),
            "guidebook_test_defects": g["test_defect"], "guidebook_test_normals": g["test_normal"],
            "normals_match": int((y.iloc[te] == 0).sum()) == g["test_normal"],
            "defect_inflation_x": round(g["test_defect"] / n_def_te, 1),
            "test_defects_exceed_file_defects": g["test_defect"] > int(y.sum()),
            "test_defect_rows_with_identical_train_row": round(float(key(te_def).isin(train_keys).mean()), 2),
        })
    t = pd.DataFrame(rows)
    out = ROOT / cfg["output_dir"] / "tables"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "guidebook_split_check.csv", index=False, encoding="utf-8-sig")
    print(t.T.to_string())


if __name__ == "__main__":
    main()
