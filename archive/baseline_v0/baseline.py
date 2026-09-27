#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
baseline.py
============================================================================
2026 제6회 K-인공지능 제조데이터 분석 경진대회
사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정
— "신뢰할 수 있고 재현 가능한 베이스라인" 자동 실행 스크립트
============================================================================

실행 방법
    python baseline.py
    (data/ 폴더에 moldset_labeled_cn7.csv, moldset_labeled_rg3.csv,
     moldset_unlabeled_cn7.csv, moldset_unlabeled_rg3.csv 가 있어야 한다.)

모든 결과는 outputs/ 아래에 저장된다.
    outputs/eda/                 데이터 진단·분포 분석 표와 그래프
    outputs/models/              최종 모델(joblib)과 메타데이터, feature importance
    outputs/errors/              validation / OOF 오류 분석용 데이터(FN/FP 분석에 사용)
    outputs/thresholds/          threshold 변화에 따른 trade-off 표
    outputs/predictions/         unlabeled 데이터 예측 결과(+ baseline inspection_rank)
    outputs/model_comparison.csv 실험 A/B × 모델 × class_weight 성능 비교표 (random holdout)
    outputs/model_comparison_all_validations.csv  검증 방식(random/group/ordered)별 비교표
    outputs/cv_group5fold_comparison.csv          group-aware 5-fold CV (OOF) 비교표

설계 원칙
    * 외부 패키지 없이 pandas / numpy / scipy / scikit-learn / matplotlib 만으로 실행된다.
    * 모든 전처리는 sklearn Pipeline 안에서 train 데이터에만 fit 된다(데이터 누수 방지).
    * 모든 random seed 는 42 로 고정한다.
    * 확인되지 않은 사실은 가정하지 않고 TODO / 주석으로 남긴다.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경에서도 그래프 저장 가능
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0. 설정값 (하드코딩은 여기에만 모아 둔다)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def _find_data_dir(project_dir: Path, marker: str = "moldset_labeled_cn7.csv") -> Path:
    """원본 CSV 가 있는 폴더를 찾는다.

    project/data/ → 저장소 루트 data/ → 그 하위 폴더(data/raw 등) 순서로 marker 파일을 찾는다.
    환경변수 KAMP_DATA_DIR 가 있으면 그것을 최우선으로 쓴다.
    """
    env = os.environ.get("KAMP_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    for base in (project_dir / "data", project_dir.parent / "data"):
        if (base / marker).exists():
            return base
        if base.exists():
            hits = sorted(base.rglob(marker))
            if hits:
                return hits[0].parent
    return project_dir / "data"


DATA_DIR = _find_data_dir(PROJECT_DIR)
OUTPUT_DIR = PROJECT_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
MODEL_DIR = OUTPUT_DIR / "models"
ERROR_DIR = OUTPUT_DIR / "errors"
THRESHOLD_DIR = OUTPUT_DIR / "thresholds"
PRED_DIR = OUTPUT_DIR / "predictions"

FILES = {
    "labeled_cn7": "moldset_labeled_cn7.csv",
    "labeled_rg3": "moldset_labeled_rg3.csv",
    "unlabeled_cn7": "moldset_unlabeled_cn7.csv",
    "unlabeled_rg3": "moldset_unlabeled_rg3.csv",
}
SOURCES = ["CN7", "RG3"]

TARGET_COL = "PassOrFail"
ID_COL = "Unnamed: 0"          # 원본 CSV 의 첫 번째 컬럼(이름 없는 인덱스 컬럼)
SOURCE_COL = "source"          # 실험 B 에서 추가하는 제품(금형) 구분 컬럼
SOURCE_FEATURE = "source_is_RG3"  # 실험 B 모델에 실제로 들어가는 0/1 인코딩 컬럼

# 현재 1을 불량(Fail) 클래스로 가정하였으며 공식 데이터 설명 확인 필요.
# 근거: 두 labeled 파일 모두 1 이 소수 클래스(1.4%, 2.1%)로, 일반적인 사출 공정의
#       불량률 수준과 부합한다. 원본 KAMP 데이터의 PassOrFail 은 Y/N 이었던 것으로
#       알려져 있으나 이 파일에서는 0/1 로 변환되어 있어 매핑을 데이터만으로는
#       단정할 수 없다.  TODO: 대회 제공 데이터 설명서에서 0/1 의미 확인.
POSITIVE_CLASS = 1

TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.5
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)   # 0.10 ~ 0.90
N_CV_FOLDS = 5
NEAR_CONSTANT_SHARE = 0.99   # 최빈값 비율이 이 이상이면 '거의 상수' 로 분류
TOP_K_IMPORTANCE = 15

PRIMARY_VALIDATION = "random_holdout"
VALIDATION_STRATEGIES = ["random_holdout", "group_holdout", "ordered_holdout"]
SELECTION_BASIS = "cv_group5fold"   # 최종 모델 선정에 사용할 평가 결과(가장 보수적인 검증)

# 최종 모델을 수동으로 지정하고 싶으면 아래 형식으로 바꾼다.
#   {"CN7": ("A", "RandomForest", "balanced_subsample"), "RG3": ("B", "LogisticRegression", "balanced")}
FINAL_SELECTION = "auto"

MODEL_COMPLEXITY = {          # 성능이 비슷할 때 단순한 모델을 우선하기 위한 순서(작을수록 단순)
    "Dummy": 0,
    "LogisticRegression": 1,
    "RandomForest": 2,
    "HistGradientBoosting": 3,
}
PRAUC_TIE_TOLERANCE = 0.02    # PR-AUC 차이가 이 값 이내이면 '비슷한 성능' 으로 보고 단순 모델 우선


# ---------------------------------------------------------------------------
# 1. 공통 유틸리티
# ---------------------------------------------------------------------------
def banner(title: str, char: str = "=") -> None:
    print("\n" + char * 78)
    print(title)
    print(char * 78)


def make_dirs() -> None:
    for d in (EDA_DIR, MODEL_DIR, ERROR_DIR, THRESHOLD_DIR, PRED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@dataclass
class Dataset:
    """모델 학습에 필요한 요소를 한 묶음으로 보관한다."""

    name: str                      # "CN7", "RG3", "CN7+RG3"
    experiment: str                # "A" 또는 "B"
    X: pd.DataFrame                # feature 행렬 (ID / target 제외)
    y: pd.Series                   # 0/1 target
    ids: pd.Series                 # 원본 Unnamed: 0 값 (예측 결과와 원본을 잇는 키)
    source: pd.Series              # 각 행의 제품(CN7/RG3)
    groups: pd.Series              # feature 값이 완전히 같은 행끼리 같은 번호 (shot 단위 추정)
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.feature_names = list(self.X.columns)


# ---------------------------------------------------------------------------
# 2. 데이터 로딩 및 기본 진단
# ---------------------------------------------------------------------------
def load_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """4개 CSV 를 읽어 dict 로 돌려준다."""
    dfs = {}
    for key, fname in FILES.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
        dfs[key] = pd.read_csv(path)
        print(f"[load] {key:14s} {str(dfs[key].shape):14s} <- {path.name}")
    return dfs


def get_process_features(df: pd.DataFrame) -> list[str]:
    """ID / target 을 제외한 공정변수 목록."""
    return [c for c in df.columns if c not in (ID_COL, TARGET_COL, SOURCE_COL, SOURCE_FEATURE)]


def inspect_dataset(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """shape, dtype, 결측, 중복, 클래스 비율, 기술통계, unique 수, 상수 컬럼을 출력하고 표로 돌려준다."""
    banner(f"[1] 기본 진단 — {name}", "-")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    dup_all = int(df.duplicated().sum())
    feats = get_process_features(df)
    dup_feat = int(df.duplicated(subset=feats).sum())
    print(f"결측치 총합: {int(df.isna().sum().sum())}")
    print(f"중복 행 수 (모든 컬럼 기준): {dup_all}")
    print(f"중복 행 수 (ID 제외, 공정변수 기준): {dup_feat}  ({dup_feat / len(df):.1%})")
    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts(dropna=False).sort_index()
        print(f"{TARGET_COL} class count: {vc.to_dict()}")
        print(f"{TARGET_COL} class ratio: {(vc / len(df)).round(4).to_dict()}")

    rows = []
    for c in df.columns:
        s = df[c]
        top_share = s.value_counts(normalize=True, dropna=False).iloc[0]
        row = {
            "dataset": name,
            "column": c,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "missing_ratio": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
            "top_value_share": float(top_share),
            "is_constant": bool(s.nunique(dropna=True) <= 1),
            "is_near_constant": bool(top_share >= NEAR_CONSTANT_SHARE),
        }
        if pd.api.types.is_numeric_dtype(s):
            row.update(
                mean=float(s.mean()), std=float(s.std()), min=float(s.min()),
                q25=float(s.quantile(0.25)), median=float(s.median()),
                q75=float(s.quantile(0.75)), max=float(s.max()),
                skew=float(s.skew()),
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    const_cols = summary.loc[summary.is_constant, "column"].tolist()
    near_const = summary.loc[summary.is_near_constant & ~summary.is_constant, "column"].tolist()
    print("상수 컬럼:", const_cols if const_cols else "없음")
    print(f"거의 상수 컬럼(최빈값 비율 >= {NEAR_CONSTANT_SHARE:.0%}):", near_const if near_const else "없음")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary[["column", "dtype", "n_unique", "top_value_share", "mean", "std", "min", "max"]]
              .round(3).to_string(index=False))
    return summary


def check_column_consistency(dfs: dict[str, pd.DataFrame]) -> None:
    """labeled 와 unlabeled 의 컬럼 일치 여부 확인."""
    banner("[1] labeled / unlabeled 컬럼 일치 여부", "-")
    for s in SOURCES:
        lab = dfs[f"labeled_{s.lower()}"]
        unl = dfs[f"unlabeled_{s.lower()}"]
        lab_feats = set(lab.columns) - {TARGET_COL}
        unl_feats = set(unl.columns)
        print(f"{s}: labeled(타깃 제외) == unlabeled ? {lab_feats == unl_feats} "
              f"| labeled 에만: {sorted(lab_feats - unl_feats)} | unlabeled 에만: {sorted(unl_feats - lab_feats)}")
    same = list(dfs["labeled_cn7"].columns) == list(dfs["labeled_rg3"].columns)
    print(f"CN7 vs RG3 labeled 컬럼 순서/이름 동일 ? {same}")


def check_scaling(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """각 파일이 이미 표준화(평균 0, 표준편차 1)된 데이터인지 파일별로 실제 확인한다."""
    banner("[1] 공정변수 스케일 확인 (파일별 평균/표준편차)", "-")
    rows = []
    for key, df in dfs.items():
        feats = get_process_features(df)
        m = df[feats].mean()
        sd = df[feats].std(ddof=0)
        rows.append({
            "dataset": key,
            "n_features": len(feats),
            "max_abs_mean": float(m.abs().max()),
            "n_std_close_to_1": int(((sd - 1).abs() < 1e-3).sum()),
            "n_std_zero": int((sd < 1e-12).sum()),
            "std_zero_columns": ", ".join(sd.index[sd < 1e-12].tolist()),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("-> 모든 파일에서 평균≈0, 표준편차≈1 이면 '파일별로 각각 표준화된 데이터' 이다.")
    print("   (이 경우 CN7/RG3, labeled/unlabeled 사이의 절대적인 크기 비교는 불가능하고,")
    print("    분포의 '모양' 차이만 비교할 수 있다. 원본 단위 데이터는 제공되지 않았다.)")
    out.to_csv(EDA_DIR / "scaling_check.csv", index=False)
    return out


def analyze_id_column(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """'Unnamed: 0' 이 단순 행 번호인지, 시간/생산순서 정보를 담는지 점검한다."""
    banner("[1] 'Unnamed: 0' 컬럼 분석", "-")
    rows = []
    for key, df in dfs.items():
        s = df[ID_COL]
        d = s.diff().dropna()
        feats = [c for c in get_process_features(df) if df[c].nunique() > 1]   # 상수 컬럼은 자기상관 정의 불가
        # 주어진 순서대로의 lag-1 자기상관 vs 무작위로 섞은 뒤의 자기상관
        ac_ordered = df[feats].apply(lambda col: col.autocorr(1)).mean()
        ac_shuffled = (df[feats].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
                       .apply(lambda col: col.autocorr(1)).mean())
        row = {
            "dataset": key,
            "min": int(s.min()), "max": int(s.max()), "n": len(s), "n_unique": int(s.nunique()),
            "is_unique": bool(s.is_unique),
            "monotonic_increasing": bool(s.is_monotonic_increasing),
            "all_consecutive": bool((d == 1).all()),
            "share_diff_eq_1": float((d == 1).mean()),
            "max_gap": int(d.max()),
            "mean_lag1_autocorr_given_order": float(ac_ordered),
            "mean_lag1_autocorr_shuffled": float(ac_shuffled),
        }
        if TARGET_COL in df.columns:
            pos_ids = df.loc[df[TARGET_COL] == POSITIVE_CLASS, ID_COL]
            q = pd.qcut(s, 5, labels=False)
            row["positive_ids_min"] = int(pos_ids.min())
            row["positive_ids_max"] = int(pos_ids.max())
            row["positives_per_id_quintile"] = str(df.groupby(q, observed=True)[TARGET_COL].sum().tolist())
            row["corr_id_vs_target"] = float(np.corrcoef(s, df[TARGET_COL])[0, 1])
        rows.append(row)
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string(index=False))

    # 파일 간 ID 겹침 여부
    keys = list(dfs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ov = len(set(dfs[keys[i]][ID_COL]) & set(dfs[keys[j]][ID_COL]))
            print(f"ID 겹침 {keys[i]} & {keys[j]}: {ov}")
    # unlabeled 에서 CN7/RG3 가 ID 순서상 섞여 있는지
    u = pd.concat([
        dfs["unlabeled_cn7"][[ID_COL]].assign(src="CN7"),
        dfs["unlabeled_rg3"][[ID_COL]].assign(src="RG3"),
    ]).sort_values(ID_COL)
    n_runs = int((u["src"] != u["src"].shift()).sum())
    print(f"unlabeled CN7/RG3 를 ID 순으로 합쳤을 때 제품이 바뀌는 구간 수: {n_runs} "
          f"(1 이면 완전히 분리, 클수록 번갈아 생산)")
    out.to_csv(EDA_DIR / "id_column_analysis.csv", index=False)

    print("\n[해석 가이드]")
    print(" * labeled 는 0부터 연속된 행 번호(CN7 0~1210, RG3 1211~2392)로 '단순 인덱스' 형태이지만,")
    print("   주어진 순서에서 인접 행의 공정변수 자기상관이 매우 높고(섞으면 0 근처),")
    print("   동일한 값의 행이 인접해 나타나므로 '기록(시간) 순서' 를 보존한 인덱스로 보는 것이 합리적이다.")
    print(" * 따라서 feature 로는 쓰지 않고 ID 로만 보존하며, 시간순 holdout 의 기준으로 활용한다.")
    print(" * TODO: 원본 데이터의 TimeStamp 컬럼이 있다면 실제 생산 순서와 대조 필요.")
    return out


def analyze_duplicates(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """공정변수가 완전히 같은 행(중복)의 구조와, 중복 쌍 내부의 라벨 불일치를 분석한다."""
    banner("[1] 중복 행 구조 분석 (공정변수 기준)", "-")
    rows = []
    for key, df in dfs.items():
        feats = get_process_features(df)
        g = df.groupby(feats, sort=False).ngroup()
        gsize = g.value_counts()
        tmp = df.assign(_grp=g, _pos=np.arange(len(df)))
        pairs = tmp[tmp["_grp"].isin(gsize[gsize == 2].index)].groupby("_grp")["_pos"].agg(["min", "max"])
        row = {
            "dataset": key,
            "n_rows": len(df),
            "n_unique_feature_rows": int(g.nunique()),
            "group_size_counts": str(gsize.value_counts().sort_index().to_dict()),
            "share_size2_groups_adjacent": float(((pairs["max"] - pairs["min"]) == 1).mean()) if len(pairs) else np.nan,
        }
        if TARGET_COL in df.columns:
            lab = tmp.groupby("_grp")[TARGET_COL].agg(["size", "sum"])
            row["label_pattern_(size,n_pos)_counts"] = str({str(k): int(v) for k, v in lab.value_counts().sort_index().items()})
            n_pos = int((df[TARGET_COL] == POSITIVE_CLASS).sum())
            n_pos_with_neg_twin = int(((lab["size"] == 2) & (lab["sum"] == 1)).sum())
            n_pos_clean = n_pos - n_pos_with_neg_twin      # 쌍 전체가 불량이거나 단독인 양성
            row["n_positive"] = n_pos
            row["n_positive_with_identical_negative_twin"] = n_pos_with_neg_twin
            # 어떤 모델이든 같은 feature 에는 같은 예측을 하므로 아래 상한이 생긴다.
            row["bound_recall_at_zero_FP"] = n_pos_clean / n_pos if n_pos else np.nan
            row["bound_precision_at_full_recall"] = n_pos / (n_pos + n_pos_with_neg_twin) if n_pos else np.nan
            p_full = row["bound_precision_at_full_recall"]
            row["bound_f1_at_full_recall"] = 2 * p_full / (1 + p_full) if n_pos else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.to_string(index=False))
    out.to_csv(EDA_DIR / "duplicate_structure.csv", index=False)
    print("\n[해석 가이드]")
    print(" * labeled 데이터의 거의 모든 행은 '공정변수가 완전히 같은 인접 2행' 쌍을 이룬다.")
    print("   불량(1) 행의 대부분은 같은 공정값을 가진 정상(0) 행과 쌍을 이루므로,")
    print("   공정변수만으로는 두 행을 구분할 수 없다 → 어떤 모델도 이 쌍에서는 (TP+FP) 또는 (FN+TN) 이 된다.")
    print(" * 가설: 1 shot(1회 사출) 에서 2개 부품(예: LH/RH)이 나오고 부품별로 검사 결과가 붙은 형태.")
    print("   TODO: 원본 데이터의 PART_NAME 등 메타데이터로 확인 필요.")
    print(" * 결론: 무작위 split 은 같은 쌍을 train/validation 에 나눠 넣어 성능을 왜곡할 수 있으므로")
    print("   쌍(group) 단위 split 을 함께 평가한다.")
    print(" * 성능 상한(bound_*): FP 0 으로 잡을 수 있는 최대 Recall, 모든 불량을 잡을 때의 최대 Precision/F1.")
    return out


def analyze_target(df: pd.DataFrame, name: str) -> dict:
    """target 분포(정상/불량 수와 비율)."""
    vc = df[TARGET_COL].value_counts().sort_index()
    n = len(df)
    info = {
        "dataset": name,
        "n_rows": n,
        "n_class0": int(vc.get(0, 0)),
        "n_class1": int(vc.get(1, 0)),
        "ratio_class1": float(vc.get(1, 0) / n),
        "positive_class_assumed_defect": POSITIVE_CLASS,
    }
    print(f"[target] {name}: 0 -> {info['n_class0']} ({1 - info['ratio_class1']:.2%}), "
          f"1 -> {info['n_class1']} ({info['ratio_class1']:.2%})  "
          f"(불균형 비율 약 1:{info['n_class0'] / max(info['n_class1'], 1):.0f})")
    return info


def plot_target_distribution(target_infos: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = [t["dataset"] for t in target_infos]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [t["n_class0"] for t in target_infos], width=0.4, label="PassOrFail = 0")
    ax.bar(x + 0.2, [t["n_class1"] for t in target_infos], width=0.4, label="PassOrFail = 1 (assumed defect)")
    for i, t in enumerate(target_infos):
        ax.text(i + 0.2, t["n_class1"] + 10, f"{t['n_class1']} ({t['ratio_class1']:.1%})", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("rows")
    ax.set_title("Target distribution (labeled data)")
    ax.legend(fontsize=8)
    save_fig(fig, EDA_DIR / "target_distribution.png")


def plot_positives_along_index(dfs: dict[str, pd.DataFrame]) -> None:
    """불량이 생산순서(ID) 상 어디에 몰려 있는지 시각화."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=False)
    for ax, s in zip(axes, SOURCES):
        df = dfs[f"labeled_{s.lower()}"]
        pos = df.loc[df[TARGET_COL] == POSITIVE_CLASS, ID_COL]
        ax.vlines(df[ID_COL], 0, 0.3, color="lightgray", lw=0.5)
        ax.vlines(pos, 0, 1, color="crimson", lw=1.2)
        ax.set_yticks([])
        ax.set_xlim(df[ID_COL].min() - 5, df[ID_COL].max() + 5)
        ax.set_title(f"{s}: rows (gray) and PassOrFail=1 rows (red) along '{ID_COL}' order — n_pos={len(pos)}")
    axes[-1].set_xlabel(ID_COL)
    save_fig(fig, EDA_DIR / "positives_along_index.png")


# ---------------------------------------------------------------------------
# 3. 데이터 분포 분석
# ---------------------------------------------------------------------------
def analyze_class_conditional(df: pd.DataFrame, name: str, feats: list[str], top_k: int = 6) -> pd.DataFrame:
    """정상/불량별 변수 분포 차이(평균 차이, KS 통계량)를 계산하고 상위 변수를 그래프로 저장한다."""
    pos = df[df[TARGET_COL] == POSITIVE_CLASS]
    neg = df[df[TARGET_COL] != POSITIVE_CLASS]
    rows = []
    for f in feats:
        ks = ks_2samp(pos[f], neg[f])
        rows.append({
            "dataset": name, "feature": f,
            "mean_normal": neg[f].mean(), "mean_defect": pos[f].mean(),
            "mean_diff_defect_minus_normal": pos[f].mean() - neg[f].mean(),
            "std_normal": neg[f].std(), "std_defect": pos[f].std(),
            "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue,
        })
    out = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
    print(f"\n[class-conditional] {name}: 정상 vs 불량 분포 차이가 큰 변수 (KS 통계량 순)")
    print(out.head(10)[["feature", "mean_normal", "mean_defect", "mean_diff_defect_minus_normal", "ks_stat", "ks_pvalue"]]
          .round(3).to_string(index=False))
    out.to_csv(EDA_DIR / f"class_conditional_{name}.csv", index=False)

    top = out.head(top_k)["feature"].tolist()
    fig, axes = plt.subplots(1, len(top), figsize=(2.6 * len(top), 3.4))
    for ax, f in zip(np.atleast_1d(axes), top):
        ax.boxplot([neg[f].values, pos[f].values], widths=0.5,
                   showfliers=True, flierprops=dict(marker=".", markersize=3))
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["normal(0)", "defect(1)"])
        ax.set_title(f, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{name}: top-{top_k} features by KS(normal vs defect) — values are file-wise z-scores", fontsize=9)
    save_fig(fig, EDA_DIR / f"class_conditional_top_features_{name}.png")
    return out


def compare_domain_distribution(cn7: pd.DataFrame, rg3: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """CN7 vs RG3 의 동일 feature 분포 차이: mean, difference, SMD, KS. 그리고 '도메인 분류기' 로 구분 가능성 점검."""
    banner("[2] CN7 vs RG3 분포 비교", "-")
    rows = []
    for f in feats:
        m1, m2 = cn7[f].mean(), rg3[f].mean()
        s1, s2 = cn7[f].std(), rg3[f].std()
        pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
        ks = ks_2samp(cn7[f], rg3[f])
        rows.append({
            "feature": f, "CN7_mean": m1, "RG3_mean": m2, "difference": m1 - m2,
            "standardized_mean_diff": (m1 - m2) / pooled if pooled > 0 else np.nan,
            "CN7_std": s1, "RG3_std": s2, "CN7_skew": cn7[f].skew(), "RG3_skew": rg3[f].skew(),
            "CN7_n_unique": cn7[f].nunique(), "RG3_n_unique": rg3[f].nunique(),
            "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue,
        })
    out = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[["feature", "CN7_mean", "RG3_mean", "difference", "standardized_mean_diff",
                   "CN7_n_unique", "RG3_n_unique", "ks_stat", "ks_pvalue"]].round(3).to_string(index=False))
    print("\n[주의] 두 파일이 각각 표준화되어 있어 평균 차이/SMD 는 0 근처가 '만들어져' 있다.")
    print("       분포 '모양' 차이는 KS 통계량(0=같음, 1=완전히 다름)으로 본다.")
    out.to_csv(EDA_DIR / "domain_comparison_cn7_vs_rg3.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(out["feature"][::-1], out["ks_stat"][::-1])
    ax.set_xlabel("KS statistic (CN7 vs RG3)")
    ax.set_title("Distribution-shape difference between CN7 and RG3 (per feature)")
    ax.tick_params(labelsize=7)
    save_fig(fig, EDA_DIR / "domain_comparison_ks.png")

    # 도메인 분류기: feature 만으로 CN7/RG3 를 맞출 수 있으면 두 도메인은 사실상 다른 분포이다.
    X = pd.concat([cn7[feats], rg3[feats]], ignore_index=True)
    d = np.r_[np.zeros(len(cn7)), np.ones(len(rg3))]
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    auc_rf = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
                             X, d, cv=cv, scoring="roc_auc").mean()
    auc_lr = cross_val_score(Pipeline([("scaler", StandardScaler()),
                                       ("lr", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
                             X, d, cv=cv, scoring="roc_auc").mean()
    print(f"\n[도메인 분류기] feature 만으로 CN7/RG3 구분 — RandomForest AUC={auc_rf:.3f}, LogisticRegression AUC={auc_lr:.3f}")
    print("   AUC 가 1 에 가까우면 트리 모델은 source 컬럼이 없어도 제품을 완벽히 구분할 수 있다.")
    print("   → 실험 B 에서 source 컬럼은 '새 정보' 가 아니라 이미 feature 에 들어 있는 정보의 명시적 표기이다.")
    pd.DataFrame([{"domain_classifier": "RandomForest", "roc_auc": auc_rf},
                  {"domain_classifier": "LogisticRegression", "roc_auc": auc_lr}]
                 ).to_csv(EDA_DIR / "domain_classifier_auc.csv", index=False)
    return out


def compare_labeled_unlabeled(dfs: dict[str, pd.DataFrame], feats: list[str]) -> pd.DataFrame:
    """같은 제품의 labeled vs unlabeled 분포 모양 비교 (파일별 표준화 때문에 위치 비교는 불가)."""
    banner("[2] labeled vs unlabeled 분포 비교 (같은 제품)", "-")
    rows = []
    for s in SOURCES:
        lab = dfs[f"labeled_{s.lower()}"]
        unl = dfs[f"unlabeled_{s.lower()}"]
        for f in feats:
            ks = ks_2samp(lab[f], unl[f])
            rows.append({"source": s, "feature": f, "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue,
                         "labeled_min": lab[f].min(), "labeled_max": lab[f].max(),
                         "unlabeled_min": unl[f].min(), "unlabeled_max": unl[f].max(),
                         "labeled_n_unique": lab[f].nunique(), "unlabeled_n_unique": unl[f].nunique()})
    out = pd.DataFrame(rows)
    summ = out.groupby("source")["ks_stat"].agg(["mean", "median", "max"]).round(3)
    print("feature 별 KS 통계량 요약 (labeled vs unlabeled):")
    print(summ.to_string())
    print("\n[주의] labeled 와 unlabeled 도 각각 따로 표준화되어 있다. 같은 z-score 값이 같은 물리량을 뜻한다는")
    print("       보장이 없으므로 unlabeled 예측은 '상대적 순위(ranking)' 용도로만 해석해야 한다.")
    print("       TODO: 원본 단위 데이터 또는 스케일링 파라미터(평균/표준편차) 확보 필요.")
    out.to_csv(EDA_DIR / "labeled_vs_unlabeled_ks.csv", index=False)
    return out


def analyze_correlation(df: pd.DataFrame, name: str, feats: list[str], thresh: float = 0.9) -> pd.DataFrame:
    """feature 상관행렬 저장 + 강한 상관 쌍 출력 + heatmap 저장."""
    corr = df[feats].corr()
    corr.to_csv(EDA_DIR / f"correlation_{name}.csv")
    pairs = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            if abs(corr.loc[a, b]) >= thresh:
                pairs.append({"dataset": name, "feature_a": a, "feature_b": b, "corr": corr.loc[a, b]})
    pairs_df = pd.DataFrame(pairs)
    print(f"[correlation] {name}: |r| >= {thresh} 인 변수 쌍 {len(pairs_df)}개")
    if len(pairs_df):
        print(pairs_df.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=6)
    ax.set_yticklabels(feats, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(f"Feature correlation — {name}")
    save_fig(fig, EDA_DIR / f"correlation_heatmap_{name}.png")
    return pairs_df


# ---------------------------------------------------------------------------
# 4. feature 준비 / train-validation split
# ---------------------------------------------------------------------------
def select_feature_columns(dfs: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    """labeled 두 파일 모두에서 상수인 컬럼을 제외한 공정변수 목록을 만든다."""
    feats_all = get_process_features(dfs["labeled_cn7"])
    dropped = []
    for f in feats_all:
        if all(dfs[f"labeled_{s.lower()}"][f].nunique() <= 1 for s in SOURCES):
            dropped.append(f)
    feats = [f for f in feats_all if f not in dropped]
    print(f"[features] 공정변수 {len(feats_all)}개 중 상수 컬럼 {dropped} 제외 → {len(feats)}개 사용")
    print("           (ID 컬럼 'Unnamed: 0' 은 feature 에서 제외하고 ID 로만 보존)")
    return feats, dropped


def prepare_features(df: pd.DataFrame, feats: list[str], source: str, experiment: str) -> Dataset:
    """단일 제품 데이터(실험 A)를 Dataset 으로 변환한다."""
    X = df[feats].copy()
    y = df[TARGET_COL].astype(int).copy()
    groups = pd.Series(df.groupby(feats, sort=False).ngroup().values, index=df.index, name="dup_group")
    return Dataset(name=source, experiment=experiment, X=X, y=y,
                   ids=df[ID_COL].copy(), source=pd.Series(source, index=df.index, name=SOURCE_COL),
                   groups=groups)


def prepare_combined_features(cn7: pd.DataFrame, rg3: pd.DataFrame, feats: list[str]) -> Dataset:
    """실험 B: CN7+RG3 를 합치고 source 컬럼(0/1 인코딩)을 추가한다."""
    df = pd.concat([cn7.assign(**{SOURCE_COL: "CN7"}), rg3.assign(**{SOURCE_COL: "RG3"})], ignore_index=True)
    X = df[feats].copy()
    X[SOURCE_FEATURE] = (df[SOURCE_COL] == "RG3").astype(int)
    y = df[TARGET_COL].astype(int).copy()
    groups = pd.Series(df.groupby(feats + [SOURCE_COL], sort=False).ngroup().values, index=df.index, name="dup_group")
    return Dataset(name="CN7+RG3", experiment="B", X=X, y=y,
                   ids=df[ID_COL].copy(), source=df[SOURCE_COL].copy(), groups=groups)


def split_data(ds: Dataset, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """train / validation 의 위치 인덱스를 돌려준다.

    random_holdout  : 층화(stratified) 무작위 holdout (기본, 과제 명세)
    group_holdout   : 공정변수가 완전히 같은 행(중복 쌍)은 반드시 같은 쪽에 두는 층화 holdout
                      (StratifiedGroupKFold 의 첫 fold ≈ 20%)
    ordered_holdout : 제품별로 ID(기록 순서) 기준 뒤쪽 20% 를 validation 으로 (시간순 holdout 근사)
    """
    idx = np.arange(len(ds.y))
    y = ds.y.values
    if strategy == "random_holdout":
        tr, va = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    elif strategy == "group_holdout":
        sgkf = StratifiedGroupKFold(n_splits=int(round(1 / TEST_SIZE)), shuffle=True, random_state=RANDOM_STATE)
        tr, va = next(iter(sgkf.split(ds.X, y, groups=ds.groups.values)))
    elif strategy == "ordered_holdout":
        tr_list, va_list = [], []
        for s in ds.source.unique():
            pos = idx[ds.source.values == s]
            pos = pos[np.argsort(ds.ids.values[pos], kind="stable")]
            n_va = int(round(len(pos) * TEST_SIZE))
            tr_list.append(pos[:-n_va])
            va_list.append(pos[-n_va:])
        tr, va = np.concatenate(tr_list), np.concatenate(va_list)
    else:
        raise ValueError(f"알 수 없는 split 전략: {strategy}")
    return np.sort(tr), np.sort(va)


def describe_split(ds: Dataset, strategy: str, tr: np.ndarray, va: np.ndarray) -> dict:
    """split 의 크기, 클래스 수, 그리고 validation 행의 '완전히 같은 행' 이 train 에 있는지(누수 진단)."""
    tr_groups = set(ds.groups.values[tr])
    leak = np.isin(ds.groups.values[va], list(tr_groups)).mean()
    info = {
        "dataset": ds.name, "experiment": ds.experiment, "validation": strategy,
        "n_train": len(tr), "n_val": len(va),
        "n_pos_train": int(ds.y.values[tr].sum()), "n_pos_val": int(ds.y.values[va].sum()),
        "share_val_rows_with_identical_row_in_train": float(leak),
    }
    return info


# ---------------------------------------------------------------------------
# 5. 모델 정의 / 평가 / threshold / 오류 분석 / feature importance
# ---------------------------------------------------------------------------
def build_models() -> dict[tuple[str, str], Pipeline]:
    """(모델명, class_weight) -> Pipeline.

    * 전처리는 모두 Pipeline 안에 있으므로 fit 시 train 데이터에만 적합된다(누수 방지).
    * 결측치는 현재 없지만 재현성과 안전을 위해 SimpleImputer 를 포함한다.
    * 강한 hyperparameter tuning 은 하지 않는다 (baseline).
    """
    models: dict[tuple[str, str], Pipeline] = {}
    models[("Dummy", "most_frequent")] = Pipeline([
        ("clf", DummyClassifier(strategy="most_frequent"))])
    models[("Dummy", "stratified")] = Pipeline([
        ("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE))])
    for cw in (None, "balanced"):
        models[("LogisticRegression", str(cw))] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight=cw, random_state=RANDOM_STATE)),
        ])
    for cw in (None, "balanced_subsample"):
        models[("RandomForest", str(cw))] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight=cw,
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ])
    for cw in (None, "balanced"):
        models[("HistGradientBoosting", str(cw))] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(class_weight=cw, random_state=RANDOM_STATE)),
        ])
    return models


def predict_proba_positive(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """불량(POSITIVE_CLASS) 확률 벡터."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if POSITIVE_CLASS in classes:
        return proba[:, classes.index(POSITIVE_CLASS)]
    return np.zeros(len(X))   # train 에 양성이 하나도 없던 경우


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC + confusion matrix 항목."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_val": int(len(y_true)), "n_pos_val": int(y_true.sum()),
    }
    # ROC-AUC / PR-AUC 는 양성과 음성이 모두 있어야 정의된다.
    if 0 < y_true.sum() < len(y_true):
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def analyze_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds=THRESHOLDS) -> pd.DataFrame:
    """threshold 를 바꿔 가며 Precision/Recall/F1/FN/FP 를 계산 (trade-off 분석용, 최적화용 아님)."""
    rows = []
    for t in thresholds:
        m = evaluate_model(y_true, y_prob, t)
        rows.append({"threshold": t, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                     "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
                     "n_flagged": m["fp"] + m["tp"]})
    return pd.DataFrame(rows)


def create_error_dataframe(ds: Dataset, va: np.ndarray, y_prob: np.ndarray,
                           threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """validation 표본별 원본 ID, source, y_true, y_pred, 확률, error_type + 모든 공정 feature."""
    y_true = ds.y.values[va]
    y_pred = (y_prob >= threshold).astype(int)
    err = np.where((y_true == 1) & (y_pred == 1), "TP",
          np.where((y_true == 0) & (y_pred == 0), "TN",
          np.where((y_true == 0) & (y_pred == 1), "FP", "FN")))
    out = pd.DataFrame({
        "original_index": ds.ids.values[va],
        "source": ds.source.values[va],
        "dup_group": ds.groups.values[va],
        "y_true": y_true, "y_pred": y_pred,
        "defect_probability": y_prob, "error_type": err,
    })
    feats = [c for c in ds.feature_names if c != SOURCE_FEATURE]
    out = pd.concat([out, ds.X.iloc[va][feats].reset_index(drop=True)], axis=1)
    return out


def extract_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame | None:
    """RandomForest: feature_importances_, LogisticRegression: coefficient. 그 외는 None."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_,
                             "kind": "feature_importances_"}).sort_values("importance", ascending=False)
    if hasattr(clf, "coef_"):
        coef = clf.coef_.ravel()
        return pd.DataFrame({"feature": feature_names, "importance": coef, "abs_importance": np.abs(coef),
                             "kind": "coefficient(standardized X)"}).sort_values("abs_importance", ascending=False)
    return None


def plot_feature_importance(imp: pd.DataFrame, title: str, path: Path, top_k: int = TOP_K_IMPORTANCE) -> None:
    top = imp.head(top_k)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["crimson" if v < 0 else "steelblue" for v in top["importance"]]
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.set_title(title + "\n(importance ≠ causality; SHAP analysis is a next step)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.axvline(0, color="black", lw=0.5)
    save_fig(fig, path)


def slug(*parts) -> str:
    return "_".join(str(p).replace("+", "_").replace(" ", "") for p in parts)


# ---------------------------------------------------------------------------
# 6. 실험 실행 (holdout / CV)
# ---------------------------------------------------------------------------
def run_holdout_experiment(ds: Dataset, models: dict, strategy: str, save_artifacts: bool) -> tuple[list[dict], dict]:
    """하나의 Dataset 에 대해 split 전략 하나로 모든 모델을 학습·평가한다.

    실험 B(CN7+RG3) 는 전체 validation 성능과 함께 source 별 성능도 따로 기록한다
    (dataset 이름: 'CN7+RG3[CN7]', 'CN7+RG3[RG3]') → 실험 A 와 제품별 비교 가능.
    """
    tr, va = split_data(ds, strategy)
    info = describe_split(ds, strategy, tr, va)
    print(f"\n[{ds.experiment}/{ds.name}] validation={strategy}: train {info['n_train']} (pos {info['n_pos_train']}), "
          f"val {info['n_val']} (pos {info['n_pos_val']}), "
          f"val 행 중 train 에 완전히 같은 행이 있는 비율 {info['share_val_rows_with_identical_row_in_train']:.1%}")
    if info["n_pos_val"] == 0:
        print("   !! validation 에 양성(불량)이 하나도 없어 Recall/F1/AUC 를 해석할 수 없다. (표에는 남기되 참고용)")

    rows = []
    for (mname, cw), model in models.items():
        model.fit(ds.X.iloc[tr], ds.y.iloc[tr])
        prob = predict_proba_positive(model, ds.X.iloc[va])
        metrics = evaluate_model(ds.y.values[va], prob)
        base = {"experiment": ds.experiment, "dataset": ds.name, "model": mname, "class_weight": cw,
                "validation": strategy, **metrics, **{k: v for k, v in info.items()
                                                      if k in ("n_train", "n_pos_train",
                                                               "share_val_rows_with_identical_row_in_train")}}
        rows.append(base)
        if ds.experiment == "B":   # 제품별로 쪼갠 성능
            for s in SOURCES:
                mask = ds.source.values[va] == s
                m_s = evaluate_model(ds.y.values[va][mask], prob[mask])
                rows.append({**base, "dataset": f"{ds.name}[{s}]", **m_s})

        if save_artifacts and mname != "Dummy":
            tag = slug(ds.experiment, ds.name, mname, cw, strategy)
            create_error_dataframe(ds, va, prob).to_csv(ERROR_DIR / f"{tag}_validation_errors.csv", index=False)
            analyze_thresholds(ds.y.values[va], prob).to_csv(THRESHOLD_DIR / f"{tag}_threshold_performance.csv",
                                                             index=False)
            imp = extract_importance(model, ds.feature_names)
            if imp is not None:
                imp.to_csv(MODEL_DIR / f"{tag}_feature_importance.csv", index=False)
                plot_feature_importance(imp, f"{mname} (class_weight={cw}) — {ds.experiment}/{ds.name} [{strategy}]",
                                        MODEL_DIR / f"{tag}_feature_importance.png")
    return rows, info


def cross_validate_oof(ds: Dataset, models: dict, n_splits: int = N_CV_FOLDS) -> tuple[list[dict], dict]:
    """group-aware 층화 K-fold 로 모든 행에 대해 out-of-fold(OOF) 불량 확률을 만든다.

    holdout 은 validation 양성 수가 3~5개뿐이라 지표가 매우 불안정하므로,
    모든 양성이 한 번씩 validation 에 포함되는 OOF 지표를 '보조 평가' 로 사용한다.
    (튜닝에는 사용하지 않는다.)
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = list(sgkf.split(ds.X, ds.y.values, groups=ds.groups.values))
    rows, oof_store = [], {}
    for (mname, cw), model in models.items():
        oof = np.zeros(len(ds.y))
        fold_f1 = []
        for tr, va in folds:
            model.fit(ds.X.iloc[tr], ds.y.iloc[tr])
            oof[va] = predict_proba_positive(model, ds.X.iloc[va])
            fold_f1.append(evaluate_model(ds.y.values[va], oof[va])["f1"])
        metrics = evaluate_model(ds.y.values, oof)
        base = {"experiment": ds.experiment, "dataset": ds.name, "model": mname, "class_weight": cw,
                "validation": f"cv_group{n_splits}fold", **metrics,
                "fold_f1_mean": float(np.mean(fold_f1)), "fold_f1_std": float(np.std(fold_f1))}
        rows.append(base)
        if ds.experiment == "B":
            for s in SOURCES:
                mask = ds.source.values == s
                rows.append({**base, "dataset": f"{ds.name}[{s}]",
                             **evaluate_model(ds.y.values[mask], oof[mask]),
                             "fold_f1_mean": np.nan, "fold_f1_std": np.nan})
        oof_store[(mname, cw)] = oof
        if mname != "Dummy":
            tag = slug(ds.experiment, ds.name, mname, cw, f"cv_group{n_splits}fold")
            create_error_dataframe(ds, np.arange(len(ds.y)), oof).to_csv(ERROR_DIR / f"{tag}_oof_errors.csv", index=False)
            analyze_thresholds(ds.y.values, oof).to_csv(THRESHOLD_DIR / f"{tag}_threshold_performance.csv", index=False)
    return rows, oof_store


def print_comparison(df: pd.DataFrame, title: str) -> None:
    cols = ["experiment", "dataset", "model", "class_weight", "accuracy", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "tn", "fp", "fn", "tp", "n_pos_val"]
    banner(title, "-")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(df.sort_values(["f1", "pr_auc"], ascending=False)[cols].round(3).to_string(index=False))


def plot_threshold_tradeoff(thr: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    ax1.plot(thr["threshold"], thr["precision"], marker="o", ms=3, label="precision")
    ax1.plot(thr["threshold"], thr["recall"], marker="o", ms=3, label="recall")
    ax1.plot(thr["threshold"], thr["f1"], marker="o", ms=3, label="f1")
    ax1.set_xlabel("threshold")
    ax1.set_ylabel("score")
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    ax2.bar(thr["threshold"] - 0.008, thr["fn"], width=0.016, alpha=0.35, color="crimson", label="FN")
    ax2.bar(thr["threshold"] + 0.008, thr["fp"], width=0.016, alpha=0.35, color="gray", label="FP")
    ax2.set_ylabel("count (FN / FP)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax1.set_title(title, fontsize=9)
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# 7. 최종 baseline 모델 선정 / 재학습 / unlabeled 예측
# ---------------------------------------------------------------------------
def select_final_models(all_results: pd.DataFrame, target_infos: list[dict]) -> dict[str, dict]:
    """제품(CN7/RG3)별로 최종 baseline 모델을 고른다.

    규칙(auto):
      1) SELECTION_BASIS 검증 결과(기본: group-aware 5-fold OOF)만 사용한다 — 중복 쌍 누수가 없는 가장 보수적인 평가.
      2) Dummy 제외. 먼저 실험 A(해당 제품 단독 학습) 후보 중에서 고른다:
         PR-AUC(threshold 와 무관한 '불량 순위 품질') 가 가장 높은 후보를 고르되, PR-AUC 차이가
         PRAUC_TIE_TOLERANCE 이내인 후보끼리는 F1 → Recall → 단순한 모델 순으로 우선한다.
      3) 실험 B(CN7+RG3 병합, 해당 제품 부분 성능) 의 최고 PR-AUC 가 A 의 선택보다 PRAUC_TIE_TOLERANCE 를
         넘게 높을 때만 B 로 바꾼다. (근거 없는 병합을 피한다는 원칙: 병합은 이득이 분명할 때만.)
      4) 선택된 모델의 PR-AUC 가 불량률(=Dummy 의 PR-AUC)의 1.5배 미만이고 ROC-AUC < 0.6 이면
         '유의미한 신호 없음' 으로 표시한다. (예측 파일은 형식상 만들되 해석 주의)
    단순 최고 F1 로 고르지 않는 이유: threshold 0.5 의 F1 은 양성 17~25개에서 매우 불안정하고,
    검사 우선순위 활용에는 확률 순위 품질(PR-AUC)이 더 중요하기 때문이다.
    """
    chosen: dict[str, dict] = {}
    basis = all_results[(all_results["validation"] == SELECTION_BASIS) & (all_results["model"] != "Dummy")].copy()
    basis["complexity"] = basis["model"].map(MODEL_COMPLEXITY)
    base_rate = {t["dataset"]: t["ratio_class1"] for t in target_infos}

    def pick(cand: pd.DataFrame) -> pd.Series:
        near = cand[cand["pr_auc"] >= cand["pr_auc"].max() - PRAUC_TIE_TOLERANCE]
        return near.sort_values(["f1", "recall", "complexity", "pr_auc"], ascending=[False, False, True, False]).iloc[0]

    for s in SOURCES:
        cand_a = basis[(basis["experiment"] == "A") & (basis["dataset"] == s)]
        cand_b = basis[(basis["experiment"] == "B") & (basis["dataset"] == f"CN7+RG3[{s}]")]
        banner(f"[최종 baseline 선정] {s}", "-")
        cols = ["experiment", "dataset", "model", "class_weight", "pr_auc", "roc_auc", "f1", "precision", "recall", "fn", "fp"]
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(pd.concat([cand_a, cand_b]).sort_values("pr_auc", ascending=False)[cols].round(3).to_string(index=False))
        best_a = pick(cand_a)
        top = best_a
        reason = "실험 A(제품별 단독 학습) 후보 중 PR-AUC 최상위군에서 F1/Recall/단순성 우선"
        if FINAL_SELECTION != "auto":
            exp, mname, cw = FINAL_SELECTION[s]
            top = basis[(basis["experiment"] == exp) & (basis["model"] == mname) & (basis["class_weight"] == cw) &
                        (basis["dataset"] == (s if exp == "A" else f"CN7+RG3[{s}]"))].iloc[0]
            reason = "FINAL_SELECTION 수동 지정"
        elif len(cand_b):
            best_b = pick(cand_b)
            if best_b["pr_auc"] > best_a["pr_auc"] + PRAUC_TIE_TOLERANCE:
                top, reason = best_b, f"실험 B 가 PR-AUC 를 {PRAUC_TIE_TOLERANCE} 넘게 개선하여 병합 채택"
            else:
                print(f"   실험 B 최고 PR-AUC {best_b['pr_auc']:.3f} vs 실험 A 선택 {best_a['pr_auc']:.3f} "
                      f"→ 병합 이득이 분명하지 않아 실험 A 유지")
        best_prauc_all = pd.concat([cand_a, cand_b])["pr_auc"].max() if len(cand_b) else cand_a["pr_auc"].max()
        best_roc_all = pd.concat([cand_a, cand_b])["roc_auc"].max() if len(cand_b) else cand_a["roc_auc"].max()
        no_signal = bool((best_prauc_all < 1.5 * base_rate[s]) and (best_roc_all < 0.6))
        if no_signal and FINAL_SELECTION == "auto":
            # 어떤 모델도 Dummy 보다 낫지 않으면 순위를 다투는 것이 무의미하다.
            # 가장 단순한 비가중 모델을 '자리표시자(placeholder)' 로 두고, 신호 없음을 명시한다.
            top = cand_a[(cand_a["model"] == "LogisticRegression") & (cand_a["class_weight"] == "None")].iloc[0]
            reason = "신호 없음(모든 모델이 Dummy 수준) → 가장 단순한 비가중 모델을 자리표시자로 선택"
        chosen[s] = {"choice": (str(top["experiment"]), str(top["model"]), str(top["class_weight"])),
                     "reason": reason, "no_signal": no_signal, "metrics": top}
        print(f"→ 선택: 실험 {top['experiment']} / {top['model']} / class_weight={top['class_weight']} "
              f"(PR-AUC {top['pr_auc']:.3f}, ROC-AUC {top['roc_auc']:.3f}, F1 {top['f1']:.3f}, "
              f"Recall {top['recall']:.3f}, FN {top['fn']}, FP {top['fp']})  — {reason}")
        if no_signal:
            print(f"   !! {s}: 모든 모델이 Dummy 수준(최고 PR-AUC {best_prauc_all:.3f} vs 불량률 {base_rate[s]:.3f}, "
                  f"최고 ROC-AUC {best_roc_all:.3f}).")
            print("      현재 공정변수만으로는 불량을 구분할 신호가 확인되지 않는다. 예측 파일은 형식상 생성하며 해석에 주의.")
    return chosen


def train_final_model(choice: tuple[str, str, str], datasets: dict[str, Dataset]) -> tuple[Pipeline, Dataset]:
    """선정된 모델을 labeled 전체 데이터(실험 A: 해당 제품 / 실험 B: CN7+RG3)로 재학습한다."""
    exp, mname, cw = choice
    models = build_models()
    model = models[(mname, cw)]
    ds = datasets[exp]
    model.fit(ds.X, ds.y)
    return model, ds


def predict_unlabeled(model: Pipeline, ds_train: Dataset, unlabeled: pd.DataFrame, source: str,
                      feats: list[str], choice: tuple[str, str, str], threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """unlabeled 데이터의 불량 확률·라벨·baseline inspection_rank 를 만든다.

    inspection_rank 는 defect_probability 내림차순 1부터의 순위일 뿐, 최종 검사 우선순위 알고리즘이 아니다.
    """
    X = unlabeled[feats].copy()
    if SOURCE_FEATURE in ds_train.feature_names:
        X[SOURCE_FEATURE] = int(source == "RG3")
    X = X[ds_train.feature_names]
    prob = predict_proba_positive(model, X)
    out = pd.DataFrame({
        "original_index": unlabeled[ID_COL].values,
        "source": source,
        "predicted_label": (prob >= threshold).astype(int),
        "defect_probability": prob,
    })
    out["inspection_rank"] = out["defect_probability"].rank(method="first", ascending=False).astype(int)
    # 모델/threshold 등 메타정보는 행마다 반복하지 않고 predictions/prediction_metadata.json 에 따로 기록한다.
    meta = {"source": source, "experiment": choice[0], "model": choice[1], "class_weight": choice[2],
            "threshold_for_predicted_label": threshold, "n_rows": int(len(out)),
            "note": "inspection_rank is a baseline ranking (defect_probability descending), not a final inspection "
                    "priority algorithm. labeled/unlabeled files were standardized separately, so treat probabilities "
                    "as relative scores (see README)."}
    meta_path = PRED_DIR / "prediction_metadata.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    existing[source] = meta
    meta_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    return out.sort_values("inspection_rank").reset_index(drop=True)


def save_final_model(model: Pipeline, ds: Dataset, source: str, choice: tuple[str, str, str],
                     metrics_row: pd.Series | None, no_signal: bool = False) -> None:
    """모델(joblib) + 메타데이터(json) 저장. SHAP 분석은 이 파일들로 바로 이어서 할 수 있게 구조만 준비한다."""
    path = MODEL_DIR / f"final_model_{source}.joblib"
    joblib.dump(model, path)
    meta = {
        "source": source, "experiment": choice[0], "model": choice[1], "class_weight": choice[2],
        "trained_on": ds.name, "n_train_rows": int(len(ds.y)), "n_train_pos": int(ds.y.sum()),
        "feature_names": ds.feature_names, "positive_class": POSITIVE_CLASS, "threshold": DEFAULT_THRESHOLD,
        "selection_basis": SELECTION_BASIS,
        "no_meaningful_signal_vs_dummy": no_signal,
        "selection_metrics": {k: (None if pd.isna(v) else float(v)) for k, v in metrics_row.items()
                              if k in ("pr_auc", "roc_auc", "f1", "precision", "recall", "fn", "fp")} if metrics_row is not None else None,
        "shap_todo": "shap.TreeExplainer(model.named_steps['clf']) 또는 shap.LinearExplainer 로 "
                     "model[:-1].transform(X) 결과에 대해 분석 (imputer/scaler 는 pipeline 앞단).",
    }
    (MODEL_DIR / f"final_model_{source}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def sanity_check_unlabeled_predictions(pred: pd.DataFrame, labeled_ratio: float,
                                       labeled: pd.DataFrame, unlabeled: pd.DataFrame, feats: list[str]) -> dict:
    """예측 불량률이 labeled 불량률과 얼마나 다른지, unlabeled 값이 학습 범위를 벗어나는지(외삽) 점검."""
    rate = float(pred["predicted_label"].mean())
    lo, hi = labeled[feats].min(), labeled[feats].max()
    outside = ((unlabeled[feats] < lo) | (unlabeled[feats] > hi))
    info = {"source": pred["source"].iloc[0], "n_unlabeled": len(pred), "predicted_defect_rate@0.5": rate,
            "labeled_defect_ratio": labeled_ratio, "mean_defect_probability": float(pred["defect_probability"].mean()),
            "p90_defect_probability": float(pred["defect_probability"].quantile(0.9)),
            "share_rows_with_any_feature_outside_labeled_range": float(outside.any(axis=1).mean()),
            "mean_n_features_outside_labeled_range": float(outside.sum(axis=1).mean())}
    print(f"[unlabeled] {info['source']}: {len(pred)}행, 예측 불량률(0.5 기준) {rate:.2%} "
          f"(labeled 불량률 {labeled_ratio:.2%}), 평균 확률 {info['mean_defect_probability']:.3f}, "
          f"학습 범위 밖 값을 하나라도 가진 행 {info['share_rows_with_any_feature_outside_labeled_range']:.1%}")
    if rate > labeled_ratio * 2:
        print("   !! 예측 불량률이 labeled 불량률의 2배를 넘는다 → labeled/unlabeled 가 따로 표준화된 영향(스케일 불일치)")
        print("      또는 실제 분포 변화 가능성. 원본 단위 확인 전까지는 확률의 절대값보다 순위(inspection_rank)만 참고.")
    return info


# ---------------------------------------------------------------------------
# 8. 전체 실행
# ---------------------------------------------------------------------------
def main() -> dict:
    make_dirs()
    results: dict = {}

    # ---------------- 1. 로딩 / 진단 ----------------
    banner("[1] 데이터 로딩 및 기본 진단")
    dfs = load_data()
    results["dfs"] = dfs
    check_column_consistency(dfs)
    inspections = pd.concat([inspect_dataset(k, df) for k, df in dfs.items()], ignore_index=True)
    inspections.to_csv(EDA_DIR / "inspection_summary.csv", index=False)
    results["scaling"] = check_scaling(dfs)
    results["id_analysis"] = analyze_id_column(dfs)
    results["duplicates"] = analyze_duplicates(dfs)

    banner("[1] target 분포", "-")
    target_infos = [analyze_target(dfs[f"labeled_{s.lower()}"], s) for s in SOURCES]
    pd.DataFrame(target_infos).to_csv(EDA_DIR / "target_distribution.csv", index=False)
    plot_target_distribution(target_infos)
    plot_positives_along_index(dfs)
    results["target_infos"] = target_infos

    # ---------------- 2. 분포 분석 ----------------
    banner("[2] 데이터 분포 분석")
    feats, dropped = select_feature_columns(dfs)
    results["features"], results["dropped_features"] = feats, dropped
    cn7, rg3 = dfs["labeled_cn7"], dfs["labeled_rg3"]
    results["class_conditional"] = {s: analyze_class_conditional(dfs[f"labeled_{s.lower()}"], s, feats) for s in SOURCES}
    results["correlation_pairs"] = pd.concat([analyze_correlation(dfs[f"labeled_{s.lower()}"], s, feats) for s in SOURCES],
                                             ignore_index=True)
    results["domain_comparison"] = compare_domain_distribution(cn7, rg3, feats)
    results["labeled_vs_unlabeled"] = compare_labeled_unlabeled(dfs, feats)

    # ---------------- 3~10. 실험 A / B ----------------
    banner("[3~10] 실험 A (CN7, RG3 각각) / 실험 B (CN7+RG3 + source)")
    datasets = {
        "A_CN7": prepare_features(cn7, feats, "CN7", "A"),
        "A_RG3": prepare_features(rg3, feats, "RG3", "A"),
        "B": prepare_combined_features(cn7, rg3, feats),
    }
    results["datasets"] = datasets
    # source 컬럼이 target 의 강한 proxy 인지: 제품별 불량률 + 도메인 분류기(위) 로 점검
    print("[source proxy 점검] 제품별 불량률:", {t["dataset"]: round(t["ratio_class1"], 4) for t in target_infos},
          "→ 차이는 약 0.7%p 로, source 는 약한 사전확률 정보일 뿐 target 을 직접 드러내지 않는다.")

    models = build_models()
    rows_all, split_infos = [], []
    for strategy in VALIDATION_STRATEGIES:
        for key, ds in datasets.items():
            rows, info = run_holdout_experiment(ds, models, strategy, save_artifacts=(strategy == PRIMARY_VALIDATION))
            rows_all.extend(rows)
            split_infos.append(info)
    pd.DataFrame(split_infos).to_csv(EDA_DIR / "split_summary.csv", index=False)

    banner("[보조] group-aware 5-fold CV (모든 행의 OOF 확률)")
    cv_rows, oof_store = [], {}
    for key, ds in datasets.items():
        r, oof = cross_validate_oof(ds, models)
        cv_rows.extend(r)
        oof_store[key] = oof

    all_results = pd.DataFrame(rows_all + cv_rows)
    spec_cols = ["experiment", "dataset", "model", "class_weight", "accuracy", "precision", "recall", "f1",
                 "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]
    extra_cols = ["n_val", "n_pos_val", "validation"]
    primary = all_results[all_results["validation"] == PRIMARY_VALIDATION][spec_cols + extra_cols]
    primary.sort_values(["dataset", "f1"], ascending=[True, False]).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    all_results.to_csv(OUTPUT_DIR / "model_comparison_all_validations.csv", index=False)
    all_results[all_results["validation"] == SELECTION_BASIS].to_csv(OUTPUT_DIR / "cv_group5fold_comparison.csv", index=False)
    results["all_results"] = all_results

    print_comparison(primary, f"[12] 결과 비교표 — {PRIMARY_VALIDATION} (F1 내림차순)")
    print_comparison(all_results[all_results["validation"] == "group_holdout"], "[12] 결과 비교표 — group_holdout (중복 쌍 분리)")
    print_comparison(all_results[all_results["validation"] == "ordered_holdout"], "[12] 결과 비교표 — ordered_holdout (기록 순서 뒤 20%)")
    print_comparison(all_results[all_results["validation"] == SELECTION_BASIS], f"[12] 결과 비교표 — {SELECTION_BASIS} (OOF)")

    # ---------------- 11. 최종 모델 / unlabeled 예측 ----------------
    banner("[11] 최종 baseline 모델 재학습 및 unlabeled 예측")
    chosen_full = select_final_models(all_results, target_infos)
    chosen = {s: chosen_full[s]["choice"] for s in SOURCES}
    results["chosen"], results["chosen_full"] = chosen, chosen_full
    results["final"] = {}
    for s in SOURCES:
        choice = chosen[s]
        exp_key = f"A_{s}" if choice[0] == "A" else "B"
        model, ds_train = train_final_model(choice, {"A": datasets[exp_key], "B": datasets["B"]})
        basis_rows = all_results[(all_results["validation"] == SELECTION_BASIS) & (all_results["model"] == choice[1]) &
                                 (all_results["class_weight"] == choice[2]) & (all_results["experiment"] == choice[0]) &
                                 (all_results["dataset"] == (s if choice[0] == "A" else f"CN7+RG3[{s}]"))]
        save_final_model(model, ds_train, s, choice, basis_rows.iloc[0] if len(basis_rows) else None,
                         no_signal=chosen_full[s]["no_signal"])
        pred = predict_unlabeled(model, ds_train, dfs[f"unlabeled_{s.lower()}"], s, feats, choice)
        pred.to_csv(PRED_DIR / f"prediction_{s.lower()}.csv", index=False)
        labeled_ratio = next(t["ratio_class1"] for t in target_infos if t["dataset"] == s)
        results["final"][s] = {"choice": choice, "model": model,
                               "sanity": sanity_check_unlabeled_predictions(
                                   pred, labeled_ratio, dfs[f"labeled_{s.lower()}"], dfs[f"unlabeled_{s.lower()}"], feats)}
        imp = extract_importance(model, ds_train.feature_names)
        if imp is not None:
            imp.to_csv(MODEL_DIR / f"final_model_{s}_feature_importance.csv", index=False)
            plot_feature_importance(imp, f"Final baseline for {s}: {choice[1]} (cw={choice[2]}), trained on all labeled {ds_train.name}",
                                    MODEL_DIR / f"final_model_{s}_feature_importance.png")
        # 선정 모델의 threshold trade-off 그래프 (OOF 기준)
        key = exp_key if choice[0] == "A" else "B"
        oof = oof_store[key][(choice[1], choice[2])]
        ds = datasets[key]
        mask = ds.source.values == s
        thr = analyze_thresholds(ds.y.values[mask], oof[mask])
        thr.to_csv(THRESHOLD_DIR / f"final_{s}_threshold_performance_oof.csv", index=False)
        plot_threshold_tradeoff(thr, f"{s}: threshold trade-off of final baseline "
                                     f"({choice[0]}/{choice[1]}/cw={choice[2]}, group 5-fold OOF)",
                                THRESHOLD_DIR / f"final_{s}_threshold_tradeoff_oof.png")
        results["final"][s]["threshold_table"] = thr
    pd.DataFrame([results["final"][s]["sanity"] for s in SOURCES]).to_csv(PRED_DIR / "unlabeled_prediction_sanity.csv", index=False)

    summarize(results)
    return results


def summarize(results: dict) -> None:
    """[15] 최종 요약 출력."""
    banner("[15] 최종 요약")
    dfs = results["dfs"]
    ti = {t["dataset"]: t for t in results["target_infos"]}
    all_results = results["all_results"]
    print("1. 데이터 구조")
    for k, df in dfs.items():
        print(f"   - {k}: {df.shape}")
    print(f"   - 사용 feature {len(results['features'])}개 (상수 컬럼 {results['dropped_features']} 제외), ID='Unnamed: 0', target='PassOrFail'")
    print("   - 모든 파일이 각각 표준화(z-score)되어 있으며, labeled 는 '동일 공정값 인접 2행' 쌍 구조(약 50% 중복)")
    print("2. 정상/불량 비율 (1 = 불량 가정)")
    for s in SOURCES:
        print(f"   - {s}: 정상 {ti[s]['n_class0']} / 불량 {ti[s]['n_class1']} ({ti[s]['ratio_class1']:.2%})")
    dc = results["domain_comparison"]
    print("3. CN7/RG3 분포 차이")
    print(f"   - 평균 차이는 표준화 때문에 0 (비교 불가). KS 통계량 평균 {dc['ks_stat'].mean():.3f}, "
          f"최대 {dc['ks_stat'].max():.3f} ({dc.iloc[0]['feature']}) → 분포 모양이 뚜렷이 다름")
    dca = pd.read_csv(EDA_DIR / "domain_classifier_auc.csv")
    print(f"   - 도메인 분류기 AUC: RF {dca.iloc[0]['roc_auc']:.3f} / LR {dca.iloc[1]['roc_auc']:.3f}")
    print("4. 각 모델 성능: outputs/model_comparison.csv (random holdout), "
          "outputs/model_comparison_all_validations.csv, outputs/cv_group5fold_comparison.csv 참고")
    basis = all_results[all_results["validation"] == SELECTION_BASIS]
    for s in SOURCES:
        sub = basis[((basis["experiment"] == "A") & (basis["dataset"] == s)) |
                    ((basis["experiment"] == "B") & (basis["dataset"] == f"CN7+RG3[{s}]"))]
        print(f"   [{s}, {SELECTION_BASIS}]")
        print(sub.sort_values("pr_auc", ascending=False)[["experiment", "model", "class_weight", "f1", "precision", "recall",
                                                          "pr_auc", "roc_auc", "fn", "fp"]].round(3).to_string(index=False))
    print("5~6. best baseline model 과 선정 이유")
    for s in SOURCES:
        c = results["chosen"][s]
        cf = results["chosen_full"][s]
        print(f"   - {s}: 실험 {c[0]} / {c[1]} / class_weight={c[2]}  — {cf['reason']} (근거 데이터: {SELECTION_BASIS})")
        if cf["no_signal"]:
            print(f"     !! {s} 는 모든 모델이 Dummy 수준: 현재 feature 로는 불량 신호가 확인되지 않음")
    print("7~8. FN / FP 개수 (선정 모델, threshold 0.5)")
    for s in SOURCES:
        c = results["chosen"][s]
        for v in (PRIMARY_VALIDATION, SELECTION_BASIS):
            r = all_results[(all_results["validation"] == v) & (all_results["experiment"] == c[0]) &
                            (all_results["model"] == c[1]) & (all_results["class_weight"] == c[2]) &
                            (all_results["dataset"] == (s if c[0] == "A" else f"CN7+RG3[{s}]"))]
            if len(r):
                r = r.iloc[0]
                print(f"   - {s} [{v}]: FN={r['fn']} FP={r['fp']} TP={r['tp']} TN={r['tn']} (양성 {r['n_pos_val']}개 중)")
    print("9. threshold 변화에 따른 trade-off (선정 모델, OOF)")
    for s in SOURCES:
        thr = results["final"][s]["threshold_table"]
        sel = thr[thr["threshold"].isin([0.1, 0.3, 0.5, 0.7, 0.9])]
        print(f"   [{s}]")
        print(sel[["threshold", "precision", "recall", "f1", "fn", "fp", "n_flagged"]].round(3).to_string(index=False))
    print("10. 다음 단계에서 개선할 부분")
    for line in (
        "shot(중복 쌍) 단위로 라벨을 집계(any-defect)한 문제 정의 검토 — 현재 행 단위 라벨은 동일 feature 에 상반된 라벨이 붙어 성능 상한이 있음",
        "원본 단위 데이터/스케일링 파라미터 확보 → labeled/unlabeled 스케일 불일치 해소 (없으면 unlabeled 는 순위 용도로만)",
        "PassOrFail 0/1 의미 공식 확인 (현재 1=불량 가정)",
        "FN/FP 오류 파일(outputs/errors/*_oof_errors.csv)로 불량이 집중되는 공정조건 분석 + SHAP",
        "threshold 는 별도 검증(예: nested CV)으로 결정 — Recall 우선 운영점(검사 여력 대비) 설계",
        "CN7 은 불량이 한 시기(ID 47~120)에 몰려 있어 시간순 검증이 불가능 → 추가 기간 데이터 필요",
        "class_weight 외 불균형 처리(SMOTE, threshold moving, cost-sensitive) 및 간단한 튜닝은 group-aware CV 안에서만",
    ):
        print(f"   - {line}")


if __name__ == "__main__":
    warnings.filterwarnings("default")
    main()
