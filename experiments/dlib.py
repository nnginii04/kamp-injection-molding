"""D 시리즈 공통: 공개 원본(원 단위 + 시각)을 대회 행 번호로 연결하고, 도메인 가설 피처를 만든다.
- 공정변수 23개는 대회 파일과 같은 열(원 단위). 대회 파일 = 이 값을 파일별로 표준화한 것(확인됨) → StandardScaler 를 쓰면 모델 결과 동일.
- 샷 = 같은 제품에서 공정값 전체가 같은 행 묶음 (레포 shot_id 와 같은 규칙).
- 시간 피처는 '생산 시점에 알 수 있는 값'만: 직전 샷까지의 시각 정보. 라벨·불량 사유는 쓰지 않음.
사전 등록(결과 보기 전 고정):
  재가동 = 직전 샷과 간격 > 5분 (정상 사이클 약 60초)
  D1 가동 초기: log1p(재가동 후 경과 분), log1p(재가동 후 샷 수), log1p(직전 정지 분)
  D2 체류시간: log1p(직전 샷과 간격 초), 최근 10샷 평균 간격(초, log1p)
"""
import os, sys
import numpy as np
import pandas as pd

# 공개 원본 경로: 실행 시 인자로 주거나 환경변수 KAMP_PUBLIC_CSV 로 지정 (레포 data/ 에 넣지 않음 — test_data_only_competition_csvs)
PUB = os.environ.get("KAMP_PUBLIC_CSV") or (sys.argv[1] if len(sys.argv) > 1 else None)
assert PUB and os.path.exists(PUB), "공개 원본 moldset_labeled.csv 경로를 인자로 주세요: python experiments/d1_d3_domain.py <경로>"
PROC = ['Injection_Time', 'Filling_Time', 'Plasticizing_Time', 'Cycle_Time', 'Clamp_Close_Time', 'Cushion_Position',
        'Plasticizing_Position', 'Clamp_Open_Position', 'Max_Injection_Speed', 'Max_Screw_RPM', 'Average_Screw_RPM',
        'Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Max_Back_Pressure', 'Average_Back_Pressure',
        'Barrel_Temperature_1', 'Barrel_Temperature_2', 'Barrel_Temperature_3', 'Barrel_Temperature_4',
        'Barrel_Temperature_5', 'Barrel_Temperature_6', 'Hopper_Temperature', 'Mold_Temperature_3', 'Mold_Temperature_4']
D1 = ["min_since_restart", "shots_since_restart", "prev_stop_min"]
D2 = ["gap_prev_sec", "gap_mean10_sec"]
RESTART_SEC = 300


def load_public():
    d = pd.read_csv(PUB).rename(columns={"Unnamed: 0": "row_id"})
    d["t"] = pd.to_datetime(d["TimeStamp"])
    d["mold"] = d["PART_NAME"].str[:3].str.lower()
    d["side"] = d["PART_NAME"].str[-2:]
    d["date"] = d["TimeStamp"].str[:10]
    d["shot_key"] = d.groupby("mold")[PROC].transform(lambda s: s.round(4).astype(str)).agg("|".join, axis=1)
    d["shot_id"] = d["mold"] + "_" + d.groupby("mold")["shot_key"].transform(lambda s: pd.factorize(s)[0]).astype(str)
    d = d.drop(columns="shot_key")
    # 샷 단위 시각 = 두 행 중 이른 시각
    sh = d.groupby("shot_id").agg(mold=("mold", "first"), t=("t", "min")).sort_values(["mold", "t"])
    sh["gap"] = sh.groupby("mold")["t"].diff().dt.total_seconds()
    sh["restart"] = sh["gap"].isna() | (sh["gap"] > RESTART_SEC)
    sh["seg"] = sh.groupby("mold")["restart"].cumsum()
    g = sh.groupby(["mold", "seg"])
    sh["min_since_restart"] = np.log1p((sh["t"] - g["t"].transform("min")).dt.total_seconds() / 60)
    sh["shots_since_restart"] = np.log1p(g.cumcount())
    stop = sh["gap"].where(sh["restart"]).fillna(24 * 3600).clip(upper=24 * 3600) / 60   # 첫 가동은 24시간 정지로 간주
    sh["prev_stop_min"] = np.log1p(stop.groupby([sh["mold"], sh["seg"]]).transform("first"))
    gap_in = sh["gap"].where(~sh["restart"])                     # 재가동 간격은 체류시간이 아니라 정지 → 제외
    sh["gap_prev_sec"] = np.log1p(gap_in.fillna(gap_in.median()))
    sh["gap_mean10_sec"] = np.log1p(sh.groupby(["mold", "seg"])["gap_prev_sec"].transform(
        lambda s: np.expm1(s).shift(1).rolling(10, min_periods=1).mean()).fillna(np.expm1(sh["gap_prev_sec"])))
    sh["shot_order"] = sh.groupby("mold").cumcount()
    d = d.merge(sh[D1 + D2 + ["seg", "shot_order"]], left_on="shot_id", right_index=True, how="left")
    return d.sort_values("row_id").reset_index(drop=True)
