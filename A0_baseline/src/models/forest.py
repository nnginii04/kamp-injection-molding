"""베이스라인 2: 랜덤 포레스트."""
from sklearn.ensemble import RandomForestClassifier


def build(cfg: dict, seed: int) -> RandomForestClassifier:
    p = cfg["models"]["random_forest"]
    return RandomForestClassifier(n_estimators=p["n_estimators"], min_samples_leaf=p["min_samples_leaf"],
                                  class_weight=p["class_weight"], n_jobs=p["n_jobs"], random_state=seed)
