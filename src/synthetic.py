"""실제 대회 파일과 같은 형식·구조의 '가짜' 데이터 생성기 (파이프라인 동작 확인용).

실제 데이터에서 확인된 구조를 흉내 낸다:
- 공개 원본(labeled_data.csv 형식: 원 단위 값, TimeStamp, PART_NAME LH/RH, Y/N, 완전 중복 행 포함)
- 대회 labeled = 원본을 중복 제거·S14·제품별로 자른 뒤 '파일별로 표준화' (CN7 1,211행 / RG3 1,182행)
- CN7: 가동 초기 1시간 안에 불량 몰림. 양쪽 불량 3샷(사출속도 극단값) + 한쪽 불량 11샷(RH 10, LH 1)
- RG3: 한쪽(RH)만 불량 25샷, 공정값과 무관하게 흩어짐
- unlabeled = 다른 기간·운전 범위, 짝 일부만 유지, 파일별로 따로 표준화
※ 여기서 나온 성능 숫자는 의미 없음. 실제 데이터로 다시 돌려야 함.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

FEATURES = ["Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time", "Clamp_Close_Time",
            "Cushion_Position", "Plasticizing_Position", "Clamp_Open_Position", "Max_Injection_Speed",
            "Max_Screw_RPM", "Average_Screw_RPM", "Max_Injection_Pressure", "Max_Switch_Over_Pressure",
            "Max_Back_Pressure", "Average_Back_Pressure", "Barrel_Temperature_1", "Barrel_Temperature_2",
            "Barrel_Temperature_3", "Barrel_Temperature_4", "Barrel_Temperature_5", "Barrel_Temperature_6",
            "Hopper_Temperature", "Mold_Temperature_3", "Mold_Temperature_4"]
EXTRA = ["Switch_Over_Position", "Barrel_Temperature_7", "Mold_Temperature_1", "Mold_Temperature_2"]

# name: (평균, 흔들림 표준편차, 기록 해상도)
SPEC = {
    "Injection_Time": (9.59, 0.02, 0.01), "Filling_Time": (4.47, 0.02, 0.01),
    "Plasticizing_Time": (16.2, 0.35, 0.01), "Cycle_Time": (59.5, 0.25, 0.01),
    "Clamp_Close_Time": (7.13, 0.0, 0.01), "Cushion_Position": (653.4, 0.03, 0.01),
    "Plasticizing_Position": (68.4, 0.04, 0.01), "Clamp_Open_Position": (0.0, 0.0, 0.1),
    "Max_Injection_Speed": (75.0, 0.4, 0.1), "Max_Screw_RPM": (33.0, 0.25, 0.1),
    "Average_Screw_RPM": (292.0, 0.8, 0.1), "Max_Injection_Pressure": (141.8, 0.25, 0.1),
    "Max_Switch_Over_Pressure": (136.2, 0.35, 0.1), "Max_Back_Pressure": (36.4, 0.35, 0.1),
    "Average_Back_Pressure": (58.8, 0.25, 0.1), "Barrel_Temperature_1": (286.0, 0.6, 0.1),
    "Barrel_Temperature_2": (285.0, 0.6, 0.1), "Barrel_Temperature_3": (280.0, 0.6, 0.1),
    "Barrel_Temperature_4": (275.0, 0.6, 0.1), "Barrel_Temperature_5": (270.0, 0.6, 0.1),
    "Barrel_Temperature_6": (265.0, 0.6, 0.1), "Hopper_Temperature": (65.0, 1.2, 0.1),
    "Mold_Temperature_3": (24.5, 0.5, 0.1), "Mold_Temperature_4": (24.8, 0.5, 0.1),
    "Switch_Over_Position": (0.0, 0.0, 0.1), "Barrel_Temperature_7": (0.0, 0.0, 0.1),
    "Mold_Temperature_1": (0.0, 0.0, 0.1), "Mold_Temperature_2": (0.0, 0.0, 0.1),
}
RG3_OFFSET = {"Injection_Time": -1.4, "Filling_Time": -0.9, "Cycle_Time": -4.0, "Clamp_Close_Time": -0.3,
              "Cushion_Position": -12.0, "Max_Injection_Speed": -8.0, "Max_Injection_Pressure": 6.0}
PART = {"cn7": ("CN7 W/S SIDE MLD'G LH", "CN7 W/S SIDE MLD'G RH"),
        "rg3": ("RG3 MOLD'G W/SHLD, LH", "RG3 MOLD'G W/SHLD, RH")}


def _raw_shots(n, rng, product, shift=None):
    shift = shift or {}
    out = {}
    for f in FEATURES + EXTRA:
        mu, sd, _ = SPEC[f]
        mu = mu + (RG3_OFFSET.get(f, 0.0) if product == "rg3" else 0.0) + shift.get(f, 0.0)
        if sd > 0:
            a = 0.92
            e = lfilter([sd * np.sqrt(1 - a * a)], [1, -a], rng.normal(size=n)) + rng.normal(0, sd * 0.3, n)
        else:
            e = np.zeros(n)
        out[f] = mu + e
    return pd.DataFrame(out)


def _quantize(df):
    for f in FEATURES + EXTRA:
        res = SPEC[f][2]
        df[f] = np.round(np.round(df[f] / res) * res, 6)
    for f in ("Max_Injection_Pressure", "Max_Switch_Over_Pressure"):  # 원본의 float32 흔적 (141.800003 등)
        df[f] = df[f].astype(np.float32).astype(float)
    return df


def _day(product, date, start, n_shots, rng, episode=None):
    t0 = pd.Timestamp(f"{date} {start}")
    X = _raw_shots(n_shots, rng, product)
    X["TimeStamp"] = t0 + pd.to_timedelta(np.cumsum(rng.normal(60.5, 1.2, n_shots)) - 60.5, unit="s")
    X["label_LH"], X["label_RH"] = "Y", "Y"
    X["reason_LH"], X["reason_RH"] = None, None
    if episode:
        episode(X, rng)
    return _quantize(X)


def _cn7_day1(X, rng):
    ep = slice(20, 61)                                             # 가동 초기 불안정 구간
    X.loc[ep, "Clamp_Close_Time"] += 0.10
    X.loc[ep, "Max_Injection_Speed"] -= 0.7
    X.loc[ep, "Hopper_Temperature"] -= 2.5
    X.loc[ep, "Cushion_Position"] -= 0.05
    for i in (4, 9, 15):                                            # 공정 전체 이상: 양쪽 불량
        X.loc[i, "Max_Injection_Speed"] += 7.0
        X.loc[i, ["label_LH", "label_RH"]] = "N"
        X.loc[i, ["reason_LH", "reason_RH"]] = "미성형"
    for i in (24, 40, 48, 50, 51, 52, 53, 54, 55, 56, 57):          # 캐비티 국소: 한쪽만 불량
        side = "LH" if i == 40 else "RH"
        X.loc[i, f"label_{side}"] = "N"
        X.loc[i, f"reason_{side}"] = rng.choice(["가스", "미성형"])


def _cn7_day3(X, rng):
    X.loc[100:150, "Clamp_Close_Time"] += 0.20                     # 다른 설정 구간 (불량 없음)


def _rg3_defects(n_bad):
    def f(X, rng):
        X.loc[: len(X) // 3, "Clamp_Close_Time"] += 0.10          # 설정 변경 구간 (불량과 무관)
        idx = rng.choice(np.arange(5, len(X)), size=n_bad, replace=False)
        X.loc[idx, "label_RH"] = "N"
        X.loc[idx, "reason_RH"] = rng.choice(["가스", "미성형", "크랙"], size=n_bad)
    return f


def _rows(product, day, rng, singleton_first=False):
    rows = []
    for i, r in day.iterrows():
        # 실제 데이터처럼 짝의 앞 행이 대부분 RH (90%)
        sides = ["RH"] if (singleton_first and i == 0) else (["RH", "LH"] if rng.random() < 0.9 else ["LH", "RH"])
        for side in sides:
            d = {f: r[f] for f in FEATURES + EXTRA}
            d.update({"TimeStamp": r.TimeStamp, "PART_NAME": PART[product][0 if side == "LH" else 1],
                      "PassOrFail": r[f"label_{side}"], "Reason": r[f"reason_{side}"]})
            rows.append(d)
    return rows


def _later(product, date, n, rng, n_bad, side):
    X = _day(product, date, "06:00:00", n, rng)
    idx = rng.choice(np.arange(n), size=n_bad, replace=False)
    X.loc[idx, f"label_{side}"] = "N"
    X.loc[idx, f"reason_{side}"] = "미성형"
    return X


def _standardize(df):
    X = df[FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0  # StandardScaler와 같은 처리
    return pd.DataFrame((X - mu) / sd, columns=FEATURES)


def _unlabeled(product, n_shots, rng):
    parts = []
    for k, shift in enumerate([{"Max_Injection_Speed": 5.0, "Barrel_Temperature_1": 3.0},
                               {"Cycle_Time": -1.5, "Hopper_Temperature": 4.0, "Clamp_Close_Time": 0.27},
                               {"Max_Injection_Speed": -2.0, "Plasticizing_Time": 0.8, "Clamp_Close_Time": 0.47}]):
        X = _raw_shots(n_shots // 3, rng, product, shift)
        X["Clamp_Open_Position"] = np.where(rng.random(len(X)) < 0.05, 650.0, 0.0)
        parts.append(X)
    X = _quantize(pd.concat(parts, ignore_index=True))
    keep_pair = rng.random(len(X)) < 0.22
    X = X.loc[np.repeat(X.index, np.where(keep_pair, 2, 1))].reset_index(drop=True)
    Z = _standardize(X)
    Z.insert(0, "Unnamed: 0", np.arange(len(Z)))
    return Z


def generate(out_dir: Path, seed=7, n_unl_shots=6000):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cn7_days = [("2020-10-16", "04:57:47", 188, _cn7_day1, True), ("2020-10-19", "08:02:10", 36, None, False),
                ("2020-10-20", "06:31:05", 382, _cn7_day3, False)]
    rg3_days = [("2020-10-21", "13:05:00", 41, None), ("2020-10-22", "05:10:30", 300, _rg3_defects(15)),
                ("2020-10-23", "05:21:40", 250, _rg3_defects(10))]
    for date, start, n, ep, single in cn7_days:
        rows += _rows("cn7", _day("cn7", date, start, n, rng, ep), rng, singleton_first=single)
    for date, start, n, ep in rg3_days:
        rows += _rows("rg3", _day("rg3", date, start, n, rng, ep), rng)
    rows += _rows("cn7", _later("cn7", "2020-10-27", 60, rng, 5, "LH"), rng)   # 대회 labeled 밖 (원본에만)
    rows += _rows("rg3", _later("rg3", "2020-11-04", 40, rng, 5, "LH"), rng)
    pub = pd.DataFrame(rows).sort_values("TimeStamp", kind="stable").reset_index(drop=True)
    pub.insert(0, "_id", np.arange(len(pub)) + 100000)
    pub["PART_FACT_PLAN_DATE"] = pub.TimeStamp.dt.strftime("%Y-%m-%d 00:00:00")
    pub["PART_FACT_SERIAL"] = "1"
    pub["EQUIP_CD"], pub["EQUIP_NAME"] = "S14", "650톤-우진2호기"
    other = pub.iloc[:3].copy()
    other["EQUIP_CD"], other["PART_NAME"], other["_id"] = "S15", "SP2 CVR ROOF RACK", [1, 2, 3]
    dups = pub.sample(frac=0.35, random_state=seed)                            # 원본의 완전 중복 행
    pub = pd.concat([pub, other, dups]).sort_values(["TimeStamp", "_id"], kind="stable").reset_index(drop=True)
    pub["TimeStamp"] = pub.TimeStamp.dt.strftime("%Y-%m-%d %H:%M:%S")
    meta = ["_id", "TimeStamp", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "PART_NAME", "EQUIP_CD", "EQUIP_NAME",
            "PassOrFail", "Reason"]
    pub = pub[meta + FEATURES + EXTRA]
    pub.to_csv(out_dir / "labeled_data.csv", index=False, encoding="utf-8-sig")

    # 대회 labeled: 중복 제거 → S14 → 제품별 처음 N행(시간순) → 파일별 표준화
    clean = pub.drop_duplicates()
    clean = clean[clean.EQUIP_CD.eq("S14")].sort_values(["TimeStamp", "_id"], kind="stable")
    offset = 0
    for product, n_rows in (("cn7", 1211), ("rg3", 1182)):
        sub = clean[clean.PART_NAME.str.startswith(product.upper())].iloc[:n_rows]
        Z = _standardize(sub)
        Z.insert(0, "PassOrFail", sub.PassOrFail.eq("N").astype(int).to_numpy())
        Z.insert(0, "Unnamed: 0", np.arange(offset, offset + len(Z)))
        offset += len(Z)
        Z.to_csv(out_dir / f"moldset_labeled_{product}.csv", index=False)
        _unlabeled(product, n_unl_shots, rng).to_csv(out_dir / f"moldset_unlabeled_{product}.csv", index=False)
    return out_dir
