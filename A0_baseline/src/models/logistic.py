"""베이스라인 1: 로지스틱 회귀. 데이터가 이미 표준화돼 있어 스케일링은 생략."""
from sklearn.linear_model import LogisticRegression


def build(cfg: dict, seed: int) -> LogisticRegression:
    p = cfg["models"]["logistic"]
    return LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=p["max_iter"], random_state=seed)
