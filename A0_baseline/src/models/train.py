"""fold별 학습 → 예측 저장 → 모델 artifact 저장.

예측 CSV 스키마 (B·C가 그대로 읽는다):
    row_id, shot_id, mold, cavity_pos, fold_rep, fold, y_true, y_prob, model, feature_set
저장 위치:
    outputs/predictions/{model}__{feature_set}__oof.csv      개발셋 OOF (모든 rep × fold, 한 행이 한 (row, rep))
    outputs/predictions/{model}__{feature_set}__holdout.csv  개발셋 전체로 학습한 최종 모델의 홀드아웃 예측 (fold_rep=-1, fold=-1)
모델 artifact (SHAP용):
    outputs/model/{model}__{feature_set}__rep0_fold{k}.pkl   rep0의 fold 모델 5개
    outputs/model/{model}__{feature_set}__final.pkl          개발셋 전체 모델
    outputs/model/features__{feature_set}.json               입력 열 순서
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.loading import feature_columns, load_config
from src.data.split import iter_folds, load_folds
from src.models.registry import build_model

SCHEMA = ["row_id", "shot_id", "mold", "cavity_pos", "fold_rep", "fold", "y_true", "y_prob", "model", "feature_set"]


def get_features(lab: pd.DataFrame, cfg: dict, feature_set: str) -> list[str]:
    base = [c for c in feature_columns(lab, cfg) if c != "mold_cn7"]
    if feature_set == "process":
        return base + ["mold_cn7"]
    if feature_set == "process+cavity":
        return base + ["mold_cn7", "cavity_pos"]
    raise ValueError(feature_set)


def _pred_frame(lab: pd.DataFrame, rows: np.ndarray, prob: np.ndarray, rep: int, fold: int,
                model: str, feature_set: str, y: str) -> pd.DataFrame:
    sub = lab.set_index("row_id").loc[rows]
    return pd.DataFrame({
        "row_id": rows, "shot_id": sub["shot_id"].to_numpy(), "mold": sub["mold"].to_numpy(),
        "cavity_pos": sub["cavity_pos"].to_numpy(), "fold_rep": rep, "fold": fold,
        # 소수 10자리로 반올림: n_jobs 병렬 합산 순서 때문에 1e-16 수준으로 흔들리는 값을 없애 재실행 시 파일이 바이트 단위로 같게 한다
        "y_true": sub[y].to_numpy(), "y_prob": np.round(prob, 10), "model": model, "feature_set": feature_set,
    })[SCHEMA]


def train_one(model_name: str, feature_set: str, lab: pd.DataFrame, folds: pd.DataFrame, cfg: dict) -> None:
    y = cfg["data"]["target"]
    feats = get_features(lab, cfg, feature_set)
    seed = cfg["seed"]
    n_rep = cfg["split"]["n_repeats"]
    save_all = cfg["models"].get("save_all_folds", False)
    out_pred = Path(cfg["paths"]["outputs"]) / "predictions"
    out_model = Path(cfg["paths"]["outputs"]) / "model"
    out_pred.mkdir(parents=True, exist_ok=True)
    out_model.mkdir(parents=True, exist_ok=True)

    X = lab.set_index("row_id")[feats]
    Y = lab.set_index("row_id")[y]
    tag = f"{model_name}__{feature_set}"

    oof = []
    for rep in range(n_rep):
        for k, tr, te in iter_folds(folds, rep):
            m = build_model(model_name, cfg, seed + rep * 100 + k)
            m.fit(X.loc[tr], Y.loc[tr])
            prob = m.predict_proba(X.loc[te])[:, 1]
            oof.append(_pred_frame(lab, te, prob, rep, k, model_name, feature_set, y))
            if save_all or rep == 0:
                joblib.dump(m, out_model / f"{tag}__rep{rep}_fold{k}.pkl")
    oof = pd.concat(oof, ignore_index=True)
    oof.to_csv(out_pred / f"{tag}__oof.csv", index=False)

    dev_rows = folds.loc[folds["holdout"] == 0, "row_id"].to_numpy()
    hold_rows = folds.loc[folds["holdout"] == 1, "row_id"].to_numpy()
    final = build_model(model_name, cfg, seed)
    final.fit(X.loc[dev_rows], Y.loc[dev_rows])
    hold = _pred_frame(lab, hold_rows, final.predict_proba(X.loc[hold_rows])[:, 1], -1, -1, model_name, feature_set, y)
    hold.to_csv(out_pred / f"{tag}__holdout.csv", index=False)
    joblib.dump(final, out_model / f"{tag}__final.pkl")
    with open(out_model / f"features__{feature_set}.json", "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)
    print(f"[train] {tag}: oof rows={len(oof)} ({n_rep} reps) holdout rows={len(hold)} -> {out_pred}")


def train_baselines(cfg: dict) -> None:
    lab = pd.read_csv(Path(cfg["paths"]["processed"]) / "labeled.csv")
    folds = load_folds(cfg)
    for feature_set in cfg["models"].get("feature_sets", ["process", "process+cavity"]):
        for name in cfg["models"]["baseline"]:
            train_one(name, feature_set, lab, folds, cfg)


if __name__ == "__main__":
    train_baselines(load_config())
