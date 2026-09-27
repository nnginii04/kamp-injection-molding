"""평가 모듈 공통 도구: 예측 파일 읽기, 열 이름 맞추기, 반복(run) 나누기, 샷 단위 변환.

팀 공통 예측 CSV 형식 (자세한 설명은 evaluation/README.md)
  필수  y_true  : 정답 (1 = 불량, 0 = 양품)
        y_score : 불량 의심 점수 (클수록 불량 가능성이 높음. 확률이 아니어도 됨)
  권장  product(또는 mold: cn7/rg3), model, cv_mode(shot/block), repeat, fold, row_id(대회 파일의 'Unnamed: 0')
  선택  y_pred  : 모델이 내린 판정 (0/1)
        shot_id : 넣어도 무시한다. row_id 로 run_all.py 의 전역 shot_id(공정값 전체가 같은 행 = 한 shot)를 항상 붙인다
                  (파일마다 shot 을 다르게 묶으면 — 예: 연속 행만 묶기 — 짝이 갈라져 누수·집계 차이가 생기므로)
열 이름이 조금 달라도(y, p, pred, label, prob ...) 자동으로 맞춘다.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 표준 열 이름 → 자동으로 인식할 다른 이름들 (앞에 있는 이름이 우선)
ALIASES = {
    "y_true": ("y_true", "y", "label", "target", "passorfail", "true_label"),
    "y_score": ("y_score", "score", "p", "prob", "proba", "y_prob", "y_proba", "pred_proba", "probability",
                "risk_score", "cv_risk_score", "defect_prob", "cv_defect_prob"),
    "y_pred": ("y_pred", "pred", "pred_label", "cv_pred_label", "prediction"),
    "row_id": ("row_id", "unnamed: 0", "id"),
    "product": ("product", "mold"),
}
# 결과표를 나누는 기준 열 (파일에 있는 것만 사용)
GROUP_KEYS = ("model", "product", "cv_mode", "dataset", "scenario")


class MissingRequiredColumns(ValueError):
    """y_true / y_score 가 없는 파일 (예: 정답이 없는 unlabeled 예측) → 평가에서 건너뛴다."""


def read_csv_any(path) -> pd.DataFrame:
    """한글 인코딩(utf-8/cp949)을 자동으로 맞춰 읽는다."""
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last = e
    raise last


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """열 이름을 표준 이름으로 바꾸고 값이 올바른지 확인한다."""
    df = df.copy()
    lower = {str(c).lower().strip(): c for c in df.columns}
    for canon, names in ALIASES.items():
        if canon in df.columns:
            continue
        for n in names:
            if n in lower and lower[n] in df.columns and lower[n] not in ALIASES:
                df = df.rename(columns={lower[n]: canon})
                break
    miss = [c for c in ("y_true", "y_score") if c not in df.columns]
    if miss:
        raise MissingRequiredColumns(f"필수 열이 없습니다: {miss}. 현재 열: {list(df.columns)}\n"
                         "정답은 y_true(1=불량), 점수는 y_score(클수록 불량 의심)로 저장해 주세요.")
    y = pd.to_numeric(df["y_true"], errors="coerce")
    if y.isna().any() or not set(y.unique()) <= {0, 1}:
        raise ValueError(f"y_true는 0/1이어야 합니다 (1 = 불량). 현재 값: {sorted(map(str, df['y_true'].unique()))[:10]}")
    df["y_true"] = y.astype(int)
    s = pd.to_numeric(df["y_score"], errors="coerce")
    if s.isna().any():
        warnings.warn(f"y_score가 비어 있는 행 {int(s.isna().sum())}개를 평가에서 뺍니다.")
        df, s = df[s.notna()].copy(), s[s.notna()]
    df["y_score"] = s.astype(float)
    if "y_pred" in df.columns:
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    for c in ("product", "model"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def attach_shot_id(df: pd.DataFrame, shot_map) -> pd.DataFrame:
    """row_id(대회 파일의 'Unnamed: 0')로 run_all.py 결과의 전역 shot_id·defect_type을 붙인다.
    파일에 shot_id 가 있어도 무시하고 덮어쓴다 (다르게 묶인 짝 수는 경고로 알림).
    shot_map: outputs/predictions_labeled_cv.csv (경로 또는 DataFrame)"""
    sm = shot_map if isinstance(shot_map, pd.DataFrame) else read_csv_any(shot_map)
    sm = sm.rename(columns={c: "row_id" for c in sm.columns if str(c).lower() in ("unnamed: 0",)})
    if not {"row_id", "shot_id"} <= set(sm.columns) or "row_id" not in df.columns:
        return df
    cols = ["row_id", "shot_id"] + (["defect_type"] if "defect_type" in sm.columns else [])
    ref = sm[cols].drop_duplicates("row_id")
    if "shot_id" in df.columns:  # 파일이 가져온 shot 묶음이 전역 기준과 다른지 확인
        own = df[["row_id", "shot_id"]].drop_duplicates("row_id").merge(ref[["row_id", "shot_id"]], on="row_id",
                                                                          suffixes=("_file", "_global"))
        a = own.groupby("shot_id_file")["shot_id_global"].nunique()
        b = own.groupby("shot_id_global")["shot_id_file"].nunique()
        if (a > 1).any() or (b > 1).any():
            warnings.warn(f"파일의 shot_id 가 전역 기준과 다름 (전역 shot 이 갈라진 경우 {int((b > 1).sum())}개, "
                          f"서로 다른 shot 이 합쳐진 경우 {int((a > 1).sum())}개) → 파일의 shot_id 를 무시하고 전역 shot_id 사용")
    out = df.drop(columns=[c for c in ("shot_id", "defect_type") if c in df.columns]).merge(
        ref, on="row_id", how="left")
    miss = out["shot_id"].isna()
    if miss.any():
        warnings.warn(f"shot_id를 찾지 못한 행 {int(miss.sum())}개는 짝 없는 단독 행으로 취급합니다.")
        out.loc[miss, "shot_id"] = "row_" + out.loc[miss, "row_id"].astype(str)
    return out


def load_predictions(path_or_df, shot_map=None) -> pd.DataFrame:
    """예측 CSV(또는 DataFrame)를 읽어 표준 형식으로 만든다. model 열이 없으면 파일 이름을 모델 이름으로 쓴다."""
    if isinstance(path_or_df, pd.DataFrame):
        df, src = standardize(path_or_df), "(DataFrame)"
    else:
        path = Path(path_or_df)
        df, src = standardize(read_csv_any(path)), path.name
    if "model" not in df.columns:
        df["model"] = Path(src).stem if src != "(DataFrame)" else "model"
    if shot_map is not None and "row_id" in df.columns:
        df = attach_shot_id(df, shot_map)  # 파일의 shot_id 는 항상 전역 기준으로 덮어씀
    df["source_file"] = src
    return df


def as_frame(pred, y_score=None, y_pred=None) -> pd.DataFrame:
    """함수 입력을 통일: DataFrame / CSV 경로 / (y_true 배열, y_score 배열) 모두 허용."""
    if isinstance(pred, pd.DataFrame):
        return standardize(pred) if not {"y_true", "y_score"} <= set(pred.columns) else pred
    if isinstance(pred, (str, Path)):
        return load_predictions(pred)
    if y_score is None:
        raise ValueError("배열로 넣을 때는 evaluate_model(y_true, y_score) 처럼 점수도 함께 넣어 주세요.")
    d = {"y_true": np.asarray(pred), "y_score": np.asarray(y_score)}
    if y_pred is not None:
        d["y_pred"] = np.asarray(y_pred)
    return standardize(pd.DataFrame(d))


def run_column(df: pd.DataFrame, run_col="auto"):
    """평균±편차를 낼 단위. auto: repeat(교차검증 반복) → 없으면 fold → 없으면 1회."""
    if run_col == "auto":
        for c in ("repeat", "fold"):
            if c in df.columns:
                return c
        return None
    if run_col in (None, "none"):
        return None
    if run_col not in df.columns:
        raise ValueError(f"run_col='{run_col}' 열이 없습니다.")
    return run_col


def iter_runs(df: pd.DataFrame, rc):
    if rc is None:
        yield 0, df
    else:
        yield from df.groupby(rc, sort=True)


def to_level(df: pd.DataFrame, level: str, rc=None) -> pd.DataFrame:
    """level='shot'이면 LH/RH 짝을 한 샷으로 합친다 (정답 = 짝 중 하나라도 불량, 점수·판정 = 둘 중 큰 값)."""
    if level == "row":
        return df
    if level != "shot":
        raise ValueError("level은 'row' 또는 'shot'")
    if "shot_id" not in df.columns:
        raise ValueError("샷 단위 평가에는 shot_id 열이 필요합니다 (row_id가 있으면 shot_map으로 붙일 수 있음).")
    # fold 단위일 때는 fold로 나누지 않고 샷을 합친다 (짝이 다른 fold로 갈라졌어도 샷은 하나로 셈).
    by = [c for c in GROUP_KEYS if c in df.columns] + ([rc] if rc and rc != "fold" else []) + ["shot_id"]
    agg = {"y_true": ("y_true", "max"), "y_score": ("y_score", "max")}
    if "y_pred" in df.columns:
        agg["y_pred"] = ("y_pred", "max")
    if "defect_type" in df.columns:
        agg["defect_type"] = ("defect_type", "first")
    if rc == "fold":
        agg["fold"] = ("fold", "first")
    return df.groupby(by, sort=False, dropna=False).agg(**agg).reset_index()


def pairs_split_across_folds(df: pd.DataFrame) -> int:
    """같은 샷(LH/RH 짝)이 서로 다른 fold로 나뉜 개수. 0이 아니면 같은 입력이 학습·검증에 동시에 들어간 것(누수)."""
    if not {"shot_id", "fold"} <= set(df.columns):
        return 0
    by = [c for c in (*GROUP_KEYS, "repeat") if c in df.columns] + ["shot_id"]
    return int((df.groupby(by, dropna=False)["fold"].nunique() > 1).sum())


def quantile_higher(s, q: float) -> float:
    """점수 분위수 (실제 점수 값 중 하나를 고름 → 상위 k% 이상이 반드시 찍힘)."""
    try:
        return float(np.quantile(s, q, method="higher"))
    except TypeError:  # numpy < 1.22
        return float(np.quantile(s, q, interpolation="higher"))


def binary_counts(y, flag) -> dict:
    """판정(flag)에 대한 Precision/Recall/F1/TP/FP/FN/TN/찍은 비율.
    찍은 게 하나도 없으면 Precision = 0 (sklearn 기본값과 같음 → flag_rate 0으로 구분)."""
    y, flag = np.asarray(y).astype(int), np.asarray(flag).astype(bool)
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "flag_rate": float(flag.mean()) if len(flag) else np.nan}


def expected_hits(hit, score, k: int) -> float:
    """점수 상위 k개를 검사할 때 hit(0/1)의 기대 개수. 경계의 동점은 무작위로 끊는다고 보고 기댓값으로 계산.
    (파일 순서로 동점을 끊으면 불량이 파일 앞쪽에 몰린 CN7에서 결과가 부풀려진다)"""
    hit, score = np.asarray(hit, dtype=float), np.asarray(score, dtype=float)
    if k <= 0 or len(score) == 0:
        return 0.0
    k = min(k, len(score))
    thr = np.sort(score)[::-1][k - 1]
    above, tie = score > thr, score == thr
    return float(hit[above].sum() + (k - above.sum()) * hit[tie].mean())


def pct(k: float) -> str:
    v = k * 100
    return f"{v:.0f}%" if abs(v - round(v)) < 1e-9 else f"{v:g}%"
