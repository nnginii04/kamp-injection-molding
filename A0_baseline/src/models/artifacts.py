"""저장된 모델 artifact 불러오기 (SHAP·오류분석에서 B·C가 쓰는 진입점).

파일 규칙 (outputs/model/):
    {model}__{feature_set}__rep{r}_fold{k}.pkl   fold 모델 (기본은 rep0의 5개만 저장, config models.save_all_folds=true면 25개)
    {model}__{feature_set}__final.pkl            개발셋 전체로 학습한 최종 모델 (홀드아웃 예측에 쓴 것)
    features__{feature_set}.json                 그 모델이 받은 입력 열 이름·순서

사용 예:
    from src.models.artifacts import load_model, load_features, get_X
    m = load_model("random_forest", "process+cavity")            # final
    m0 = load_model("random_forest", "process+cavity", rep=0, fold=0)
    feats = load_features("process+cavity")
    X = get_X(lab, "process+cavity")                              # lab = data/processed/labeled.csv
    import shap; shap.TreeExplainer(m).shap_values(X)

주의: fold 모델로 SHAP을 계산할 때는 그 fold의 test 행(fold_id.csv)에만 적용해야 학습에 본 행이 섞이지 않는다.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data.loading import load_config


def _model_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["paths"]["outputs"]) / "model"


def load_model(model: str, feature_set: str, rep: int | None = None, fold: int | None = None, cfg: dict | None = None):
    d = _model_dir(cfg)
    name = f"{model}__{feature_set}__final.pkl" if rep is None else f"{model}__{feature_set}__rep{rep}_fold{fold}.pkl"
    return joblib.load(d / name)


def load_features(feature_set: str, cfg: dict | None = None) -> list[str]:
    with open(_model_dir(cfg) / f"features__{feature_set}.json", encoding="utf-8") as f:
        return json.load(f)


def get_X(lab: pd.DataFrame, feature_set: str, cfg: dict | None = None) -> pd.DataFrame:
    """labeled.csv에서 모델이 학습한 열만, 학습 때와 같은 순서로 뽑는다. index는 row_id."""
    return lab.set_index("row_id")[load_features(feature_set, cfg)]
