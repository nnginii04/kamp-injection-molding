"""3장 보조 분석 — 불량 shot 의 공정값이 재질 권장 가공 범위 안이었는가 (모델·실험 아님, 기술 통계만).

질문: 불량이 '권장 범위를 벗어난 운전' 때문인가, 아니면 범위 안에서도 생기는가?
      범위 안에서도 생긴다면 → 현재 기록 변수로 설명되지 않는 원인(정보 공백)의 근거.

재질 (외부 조사, 2026-09-30 현빈): CN7 = ASA (실물 각인), RG3 = ABS (실물 각인). 등급(grade)은 모른다.
권장 범위 (등급을 모르므로 대표 데이터시트 여러 개의 범위를 함께 적고, 가장 넓은 범위로 판정):
  ASA  용융 240~280 °C  : INEOS Styrolution Luran S 757G 데이터시트
       금형  40~80 °C   : Luran S 777K 데이터시트 (757G 는 60 °C 단일값)
  ABS  용융 220~260 °C  : Terluran GP-35 220~260, Toray Toyolac 일반 220~240 / 내열 240~260, LG HF380 200~230
       금형  30~80 °C   : Terluran GP-35 30~80, Toyolac·HF380 40~80
  건조 80 °C 내외 (두 재질 공통) — Hopper_Temperature 가 건조기 온도인지 확인되지 않아 판정하지 않고 값만 보고.
변수 대응 (가정, 보고서에 명시):
  - Barrel_Temperature_1~6 은 1 이 노즐 쪽(가장 높음), 6 이 호퍼 쪽. 용융온도 판정은 앞쪽 1~3 구역으로 한다.
    구역 온도는 실린더 벽 온도라서 실제 용융 수지 온도와 다를 수 있다.
  - Mold_Temperature_3·4 의 측정 위치(금형 표면/냉각수)는 알 수 없다.
판정 기준은 결과를 보기 전에 위와 같이 고정했다.

대상: 대회 기간 labeled (본), 대회 이후 기간 (CN7 10/27~11/03, RG3 11/04~11/06; 탐색적, 외부 데이터).
단위: shot (LH/RH 는 공정값이 같다). 불량 shot = 두 부품 중 하나라도 불량.

실행: python scripts/material_window_check.py [--public <원본 폴더 또는 zip>]
결과: outputs/tables/material_window_summary.csv (제품 × 기간 × 변수 요약)
      outputs/tables/material_window_defect_shots.csv (불량 shot 별 값·범위 판정·정상 분포 안 위치)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _public  # noqa: E402
import external_experiments as X  # noqa: E402
from src.data import assign_shots, load_all  # noqa: E402

MATERIAL = {"cn7": "ASA", "rg3": "ABS"}
WINDOW = {  # (하한, 상한, 근거)
    "ASA": {"melt": (240, 280, "Luran S 757G 240~280"), "mold": (40, 80, "Luran S 777K 40~80")},
    "ABS": {"melt": (220, 260, "Terluran GP-35 220~260 (Toyolac 내열 240~260)"),
            "mold": (30, 80, "Terluran GP-35 30~80 (Toyolac·HF380 40~80)")},
}
CHECK = {  # 변수 → 판정 종류 (None = 값만 보고)
    "Barrel_Temperature_1": "melt", "Barrel_Temperature_2": "melt", "Barrel_Temperature_3": "melt",
    "Barrel_Temperature_4": None, "Barrel_Temperature_5": None, "Barrel_Temperature_6": None,
    "Mold_Temperature_3": "mold", "Mold_Temperature_4": "mold", "Hopper_Temperature": None,
}


def shot_level(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby("shot_id", sort=False)
    S = g[cols].first()
    S["y"] = g["y"].max()
    S["n_defect_parts"] = g["y"].sum()
    S["reason"] = g["Reason"].agg(lambda r: "/".join(sorted({str(x) for x in r if isinstance(x, str)})))
    S["side"] = g.apply(lambda h: "/".join(sorted(h.loc[h["y"].eq(1), "PART_NAME"].str.extract(r"\b(LH|RH)\b")[0]
                                                  .dropna().unique())), include_groups=False)
    S["ts"] = g["TimeStamp"].first()
    return S.reset_index()


def summarize(S: pd.DataFrame, product: str, period: str) -> tuple[list[dict], pd.DataFrame]:
    mat = MATERIAL[product]
    norm, dfc = S[S["y"].eq(0)], S[S["y"].eq(1)]
    rows, flags = [], dfc[["shot_id", "ts", "reason", "side", "n_defect_parts"]].copy()
    for v, kind in CHECK.items():
        lo, hi, src = WINDOW[mat][kind] if kind else (np.nan, np.nan, "판정 안 함")
        inw = (lambda x: x.between(lo, hi)) if kind else (lambda x: pd.Series(np.nan, index=x.index))
        nv = np.sort(norm[v].to_numpy(float))
        pct = dfc[v].apply(lambda x: 100 * (np.searchsorted(nv, x, "left") + np.searchsorted(nv, x, "right")) / 2 / len(nv))
        rows.append(dict(product=product, material=mat, period=period, variable=v,
                         window=f"{lo:g}~{hi:g}" if kind else "", source=src,
                         normal_shots=len(norm), defect_shots=len(dfc),
                         normal_median=norm[v].median(), normal_min=norm[v].min(), normal_max=norm[v].max(),
                         defect_median=dfc[v].median(), defect_min=dfc[v].min(), defect_max=dfc[v].max(),
                         normal_in_window=inw(norm[v]).mean() if kind else np.nan,
                         defect_in_window=inw(dfc[v]).mean() if kind else np.nan,
                         defect_pct_in_normal_median=pct.median(),
                         defect_outside_normal_1_99=float(((pct < 1) | (pct > 99)).mean()) if len(dfc) else np.nan))
        flags[v] = dfc[v].to_numpy()
        flags[f"{v}_in_window"] = inw(dfc[v]).to_numpy() if kind else np.nan
        flags[f"{v}_pct_in_normal"] = pct.round(1).to_numpy()
    flags.insert(0, "period", period)
    flags.insert(0, "product", product)
    return rows, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=None)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    src = _public.public_path(cfg, args.public)
    pub = _public.read_csv(src, "moldset_labeled.csv")
    ld = _public.read_csv(src, "labeled_data.csv")
    out = ROOT / cfg["output_dir"] / "tables"
    out.mkdir(parents=True, exist_ok=True)
    data = load_all(cfg, ROOT / cfg["data_dir"])
    feats = data["feature_cols"]
    cols = list(CHECK)
    summ, flags = [], []
    for p in data["labeled"]:
        L = X.link_side(assign_shots(data["labeled"][p], feats, p), pub, feats)
        C = L[["shot_id", "y", "Reason", "PART_NAME", "TimeStamp"]].join(L[[f"raw_{c}" for c in cols]]
                                                                        .rename(columns=lambda c: c[4:]))
        r, f = summarize(shot_level(C, cols), p, "대회 기간")
        summ += r
        flags.append(f)
        # 대회 이후 기간: future_rows 는 (원단위 − mu)/sd 를 돌려주므로 mu=0, sd=1 을 주면 원단위(온도 변수는 보정 없음)
        zero, one = pd.Series(0.0, index=feats), pd.Series(1.0, index=feats)
        F = X.future_rows(ld, L, p, feats, zero, one)
        if len(F):
            r, f = summarize(shot_level(F, cols), p, "이후 기간(탐색)")
            summ += r
            flags.append(f)
    S = pd.DataFrame(summ).round(3)
    S.to_csv(out / "material_window_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(flags).round(2).to_csv(out / "material_window_defect_shots.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    show = ["product", "period", "variable", "window", "normal_shots", "defect_shots", "normal_median", "defect_median",
            "normal_in_window", "defect_in_window", "defect_pct_in_normal_median", "defect_outside_normal_1_99"]
    print(S[show].to_string(index=False))


if __name__ == "__main__":
    main()
