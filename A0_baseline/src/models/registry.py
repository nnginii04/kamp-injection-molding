"""모델 이름 → 생성 함수. 새 모델(boosting.py 등)은 여기에 한 줄 추가."""
from src.models import forest, logistic

_BUILDERS = {
    "logistic": logistic.build,
    "random_forest": forest.build,
}


def build_model(name: str, cfg: dict, seed: int):
    if name not in _BUILDERS:
        raise KeyError(f"unknown model '{name}'. available: {list(_BUILDERS)}")
    return _BUILDERS[name](cfg, seed)
