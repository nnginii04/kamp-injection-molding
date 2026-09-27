"""피처 구성.

- 상수열 제거 (labeled에서 상수인 열은 학습 신호가 없음)
- (선택) 순서 기반 피처: 직전 샷 대비 변화량(d1_), 직전 N샷 평균 대비 편차(devN_)
  → 가동 초기·조건 변경 직후처럼 '공정이 흔들리는 구간'을 포착. 과거 샷의 공정값만 쓰므로
    실제 운영에서도 계산 가능(라벨 미사용).
- (선택, 기본 끔) unlabeled로 학습한 Isolation Forest 점수(if_score)
  ※ v1 진단: labeled/unlabeled가 파일별로 따로 표준화되어 눈금이 달라 labeled에 적용하면 무의미
- pooled 시나리오: 제품 구분 변수(is_rg3) 추가

금지 변수 (FORBIDDEN_FEATURES, check_feature_list 로 자동 검사)
- ID·정답: Unnamed: 0(row_id), PassOrFail(y)
- labeled 에서 상수: Clamp_Open_Position
- 그룹 번호: shot_id, block_id, shot_size (분할용 식별자일 뿐 공정 정보가 아님)
- 짝 안 위치: pair_first / pair_position / position_in_shot / cavity_pos
  → 짝 안 순서는 캐비티가 아니라 원본 기록 순서이고 불량 여부와 얽혀 있음(누수, 사실 6).
    ablation(비교용)에서만 켤 수 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


FORBIDDEN_FEATURES = {
    "Unnamed: 0", "row_id", "PassOrFail", "y", "Clamp_Open_Position",
    "shot_id", "block_id", "shot_size", "group",
    "pair_first", "pair_position", "position_in_shot", "cavity_pos",
}
LEAKY_ABLATION_FEATURES = {"pair_first"}  # ablation 에서만 허용 (결과표에 '누수 변수 — 비교용' 표시)


def check_feature_list(feats: list[str], allow: set[str] | frozenset = frozenset()) -> list[str]:
    """모델 입력 변수 목록에 금지 변수가 있으면 멈춘다."""
    bad = sorted((set(feats) & FORBIDDEN_FEATURES) - set(allow))
    if bad:
        raise AssertionError(f"모델 입력에 금지 변수가 있습니다: {bad} (src/features.py FORBIDDEN_FEATURES 참고)")
    return list(feats)


def validate_feature_config(fcfg: dict) -> None:
    """누수로 확인된 옵션을 본 실험에서 켜지 못하게 막는다."""
    if fcfg.get("pair_position"):
        raise ValueError("features.pair_position 은 누수 변수(사실 6)라 본 실험에서 쓸 수 없습니다. "
                         "비교가 필요하면 config.json 의 ablation.pair_position.enabled 를 쓰세요.")


def select_base_features(data: dict, feats: list[str], drop_constant: bool) -> tuple[list[str], list[str]]:
    if not drop_constant:
        return feats, []
    dropped = [c for c in feats if all(lab[c].nunique() <= 1 for lab in data["labeled"].values())]
    return [c for c in feats if c not in dropped], dropped


def add_iforest_scores(data: dict, base: list[str], params: dict, seed: int) -> dict:
    models = {}
    for product in data["labeled"]:
        unl = data["unlabeled"][product]
        m = IsolationForest(n_estimators=params["n_estimators"],
                            max_samples=min(params["max_samples"], len(unl)),
                            contamination=params["contamination"], random_state=seed, n_jobs=-1)
        m.fit(unl[base].to_numpy(float))
        for key in ("labeled", "unlabeled"):
            df = data[key][product]
            df["if_score"] = -m.score_samples(df[base].to_numpy(float))  # 클수록 이상
        models[product] = m
    return models


def add_pair_position(df: pd.DataFrame) -> pd.DataFrame:
    """짝(같은 샷 2행) 안에서 row_id가 앞선 행이면 1, 뒤면 0, 짝 없는 행이면 0.5.
    ⚠ 사용 금지 (Ablation 비교용으로만): 원본 연결 결과 앞 행 = 원본에 먼저 기록된 행이고, 앞 행이 RH인 비율은
      CN7 65.8%·RG3 56.3%뿐이라 캐비티를 나타내지 않는다. 그런데 한쪽만 불량인 짝의 불량 행은 91%·92%가 앞 행
      → 기록 순서가 불량 여부와 얽힌 누수. 켜면 성능이 오르지만 정답이 새어 들어간 결과다."""
    df = df.copy()
    g = df.groupby("shot_id")["row_id"]
    first = g.rank(method="first").eq(1)
    size = g.transform("size")
    df["pair_first"] = np.where(size == 2, first.astype(float), 0.5)
    return df


def sequence_columns(base: list[str], window: int) -> list[str]:
    return [f"d1_{c}" for c in base] + [f"dev{window}_{c}" for c in base]


def add_sequence_features(df: pd.DataFrame, base: list[str], window: int) -> pd.DataFrame:
    """샷 순서(row_id 첫 등장 순)대로 직전 샷 대비 변화량과 직전 window샷 평균 대비 편차를 계산.
    같은 샷의 두 행(LH/RH)은 같은 값을 받는다. 파일(제품·labeled/unlabeled)마다 따로 계산."""
    df = df.copy()
    shots = (df.sort_values("row_id").drop_duplicates("shot_id")[["shot_id", *base]]
             .reset_index(drop=True))
    prev = shots[base].shift(1)
    roll = shots[base].shift(1).rolling(window, min_periods=1).mean()
    d1 = (shots[base] - prev).fillna(0.0)
    d1.columns = [f"d1_{c}" for c in base]
    dev = (shots[base] - roll).fillna(0.0)
    dev.columns = [f"dev{window}_{c}" for c in base]
    seq = pd.concat([shots[["shot_id"]], d1, dev], axis=1).set_index("shot_id")
    for c in seq.columns:
        df[c] = df["shot_id"].map(seq[c]).to_numpy()
    return df


def scenario_frames(data: dict, scenario: str, key: str = "labeled") -> dict:
    """scenario별 (이름 → labeled df, unlabeled df) 반환."""
    if scenario == "separate":
        return {p: (data[key][p], data["unlabeled"][p]) for p in data[key]}
    if scenario == "pooled":
        lab = pd.concat(data[key].values(), ignore_index=True)
        unl = pd.concat(data["unlabeled"].values(), ignore_index=True)
        for df in (lab, unl):
            df["is_rg3"] = (df["product"] == "rg3").astype(int)
        return {"pooled": (lab, unl)}
    raise ValueError(scenario)


def model_features(base: list[str], scenario: str, use_if: bool, seq_cols: list[str] | None = None,
                   extra: list[str] | None = None) -> list[str]:
    cols = list(base)
    if seq_cols:
        cols += seq_cols
    if extra:
        cols += extra
    if use_if:
        cols.append("if_score")
    if scenario == "pooled":
        cols.append("is_rg3")
    return cols
