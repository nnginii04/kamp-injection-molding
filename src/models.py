"""모델 정의. 모두 sklearn 인터페이스(fit / predict_proba).

지도학습(logreg, rf, lgbm, ensemble)은 '불량과 닮은 샷'을 찾고,
이상탐지(spc_maxz, iforest_normal)는 '평소 정상 공정과 다른 샷'을 찾는다.
v1 결과에서 공정 전체 이상(양쪽 캐비티 불량)은 극단적인 공정값으로 나타났으므로,
처음 보는 이상에도 반응하는 이상탐지와 지도학습을 결합한 hybrid를 둔다.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:  # KAMP-NOTE 등 lightgbm이 없는 환경 대비
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


def ecdf_score(ref_sorted: np.ndarray, s, ref_sd: float) -> np.ndarray:
    """학습 데이터 점수 분포 기준 백분위(0~1, 양 끝 제외). 학습 범위를 벗어난 값도 순서가 유지되게
    범위 밖에서는 tanh로 부드럽게 늘린다 (범위 밖 값끼리 동점이 되지 않게)."""
    s = np.asarray(s, dtype=float)
    n = len(ref_sorted)
    sd = ref_sd if ref_sd > 0 else 1.0
    out = (np.searchsorted(ref_sorted, s, side="right") + 0.5) / (n + 1)
    hi, lo = s > ref_sorted[-1], s < ref_sorted[0]
    out[hi] = (n + 0.5 + 0.5 * np.tanh((s[hi] - ref_sorted[-1]) / sd)) / (n + 1)
    out[lo] = (0.5 - 0.5 * np.tanh((ref_sorted[0] - s[lo]) / sd)) / (n + 1)
    return out


class _ScoreClassifier(ClassifierMixin, BaseEstimator):  # 순서 중요: sklearn>=1.6 태그 인식
    """원점수(raw)를 학습 기준 백분위로 바꿔 predict_proba로 내보내는 공통 틀."""

    def _set_ref(self, raw):
        self.classes_ = np.array([0, 1])
        self.ref_ = np.sort(np.asarray(raw, dtype=float))
        self.ref_sd_ = float(np.std(self.ref_)) or 1.0

    def predict_proba(self, X):
        p = ecdf_score(self.ref_, self.raw_score(X), self.ref_sd_)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class UnsupervisedScore(_ScoreClassifier):
    """unlabeled로 학습한 Isolation Forest 점수(if_score 열)를 그대로 위험 점수로 사용.
    ※ labeled/unlabeled 눈금이 다르면 무의미 (v2 기본 비활성)."""

    def __init__(self, column: str = "if_score"):
        self.column = column

    def raw_score(self, X):
        return np.asarray(X[self.column], dtype=float)

    def fit(self, X, y=None):
        self._set_ref(self.raw_score(X))
        return self


class SPCMaxZ(_ScoreClassifier):
    """관리도(Shewhart) 방식 기준선: 학습 fold의 '정상 샘플' 평균·표준편차로 변수별 z를 구하고,
    가장 크게 벗어난 변수의 |z|를 점수로 사용 → '어느 한 변수라도 평소 범위를 벗어나면 경보'.
    현장에서 이미 쓰는 규칙과 같은 논리라서 AI 모델이 넘어야 할 기준선 역할."""

    def __init__(self, sd_floor: float = 0.01):
        self.sd_floor = sd_floor

    def raw_score(self, X):
        Z = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        return np.max(np.abs(Z), axis=1)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        normal = X[y == 0] if (y == 0).sum() > 1 else X
        self.mu_ = normal.mean(axis=0)
        self.sd_ = np.maximum(normal.std(axis=0), self.sd_floor)
        self._set_ref(self.raw_score(normal))
        return self


class IForestNormal(_ScoreClassifier):
    """정상(양품) 샘플만으로 학습한 Isolation Forest (novelty detection).
    불량 라벨은 '정상 샘플을 고르는 데'만 쓰므로, 학습 때 본 적 없는 유형의 공정 이상에도 반응한다."""

    def __init__(self, n_estimators: int = 300, random_state: int = 0):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def raw_score(self, X):
        return -self.if_.score_samples(np.asarray(X, dtype=float))

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        normal = X[y == 0] if (y == 0).sum() > 1 else X
        self.if_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state,
                                   n_jobs=1).fit(normal)
        self._set_ref(self.raw_score(normal))
        return self


class Hybrid(ClassifierMixin, BaseEstimator):
    """2단계 결합.
    1단계(관리도 경보): 정상 샘플 기준 최대 |z|가 학습 정상 샘플의 상위 (1-gate) 밖이면 강한 경보
                   → 변수 하나가 극단값으로 튀는 '공정 전체 이상' 담당 (라벨 불필요, 처음 보는 이상에도 반응)
                   ※ Isolation Forest는 여러 변수가 조금씩 벗어난 샷을 더 이상하게 보고, 변수 하나만 극단인
                     샷은 놓치는 경향이 있어(검증에서 확인) 관리도 방식을 쓴다
    2단계(지도학습): 앙상블 위험점수 → 미세한 패턴의 불량 담당
    최종 점수 = max(2단계 백분위, 1단계 백분위[gate를 넘은 경우만])  (OR 규칙)"""

    def __init__(self, seed: int = 42, gate: float = 0.99):
        self.seed = seed
        self.gate = gate

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.anom_ = SPCMaxZ().fit(X, y)
        self.sup_ = make_model("ensemble", self.seed).fit(X, y)
        s = self.sup_.predict_proba(X)[:, 1]
        self.ref_ = np.sort(s)
        self.ref_sd_ = float(np.std(s)) or 1.0
        return self

    def predict_proba(self, X):
        pa = self.anom_.predict_proba(X)[:, 1]
        ps = ecdf_score(self.ref_, self.sup_.predict_proba(X)[:, 1], self.ref_sd_)
        p = np.where(pa >= self.gate, np.maximum(ps, pa), ps)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class DeterministicRF(RandomForestClassifier):
    """학습은 병렬, 예측은 단일 스레드. sklearn 은 병렬 예측 때 트리별 확률을 스레드 순서대로 더해
    실행마다 끝자리(~1e-16)가 달라진다 → 결과 파일 해시가 매번 달라지는 것을 막는다."""

    def predict_proba(self, X):
        n_jobs = self.n_jobs
        self.n_jobs = 1
        try:
            return super().predict_proba(X)
        finally:
            self.n_jobs = n_jobs


def _lgbm(seed, class_weight="balanced"):
    if HAS_LGBM:
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
                              min_child_samples=10, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=5.0, class_weight=class_weight,
                              random_state=seed, n_jobs=2, verbose=-1,
                              deterministic=True, force_row_wise=True)  # 멀티스레드에서도 실행마다 같은 결과
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          min_samples_leaf=10, l2_regularization=5.0,
                                          class_weight=class_weight, random_state=seed)


class BalancedXGB(ClassifierMixin, BaseEstimator):  # 순서 중요: sklearn>=1.6 태그 인식
    """XGBoost + fit 시점의 불량 비율로 scale_pos_weight 를 정함 (= class_weight balanced 와 같은 효과).
    fold 마다 양성 수가 달라서 고정값 대신 fit 때 계산한다. xgboost 가 없으면 HistGradientBoosting 으로 대체."""

    def __init__(self, seed: int = 42, balanced: bool = True):
        self.seed, self.balanced = seed, balanced

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        spw = float((y == 0).sum() / max((y == 1).sum(), 1)) if self.balanced else 1.0
        if HAS_XGB:
            self.model_ = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=5,
                                        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                                        scale_pos_weight=spw, random_state=self.seed, n_jobs=2,
                                        tree_method="hist", eval_metric="aucpr", verbosity=0)
        else:  # 대체: sklearn 부스팅 (수치는 달라짐)
            self.model_ = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                         min_samples_leaf=10, l2_regularization=5.0,
                                                         class_weight="balanced" if self.balanced else None,
                                                         random_state=self.seed)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _catboost(seed, balanced=True):
    if HAS_CAT:
        return CatBoostClassifier(iterations=300, learning_rate=0.03, depth=3, l2_leaf_reg=5.0,
                                  auto_class_weights="Balanced" if balanced else None,
                                  random_seed=seed, thread_count=2, verbose=0, allow_writing_files=False)
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=10,
                                          l2_regularization=5.0, class_weight="balanced" if balanced else None,
                                          random_state=seed)


def make_model(name: str, seed: int):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    # --- 추가 실험(Phase 3) 전용 변형: 본 모델 목록(config.models)에 없으면 모델 선택에 쓰이지 않음 ---
    if name == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")
    if name == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "logreg_none":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight=None, max_iter=5000))
    if name == "rf_none":
        return DeterministicRF(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                      class_weight=None, random_state=seed, n_jobs=-1)
    if name == "lgbm_none":
        return _lgbm(seed, class_weight=None)
    if name == "xgb_none":
        return BalancedXGB(seed=seed, balanced=False)
    if name == "catboost_none":
        return _catboost(seed, balanced=False)
    if name == "logreg_v0":  # archive/baseline_v0 의 LogisticRegression(class_weight=None) 재현
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    if name == "spc_maxz":
        return SPCMaxZ()
    if name == "iforest_normal":
        return IForestNormal(random_state=seed)
    if name == "iforest_unsup":
        return UnsupervisedScore()
    if name == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))
    if name == "rf":
        return DeterministicRF(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                      class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    if name == "lgbm":
        return _lgbm(seed)
    if name == "xgb":
        return BalancedXGB(seed=seed)
    if name == "catboost":
        return _catboost(seed)
    if name == "ensemble":
        return VotingClassifier([("logreg", make_model("logreg", seed)),
                                 ("rf", make_model("rf", seed)),
                                 ("lgbm", make_model("lgbm", seed))], voting="soft")
    if name == "hybrid":
        return Hybrid(seed=seed)
    raise ValueError(f"알 수 없는 모델: {name}")


# 1-SE 규칙에서 '더 단순한 모델'을 고르기 위한 복잡도 순서
COMPLEXITY = ["dummy", "spc_maxz", "iforest_normal", "iforest_unsup", "logreg", "rf", "lgbm", "xgb", "catboost",
              "ensemble", "hybrid"]
# 추가 실험 변형의 복잡도는 기반 모델과 같게 본다 (가상 선택 계산용)
BASE_MODEL = {"logreg_none": "logreg", "logreg_v0": "logreg", "rf_none": "rf", "lgbm_none": "lgbm",
              "xgb_none": "xgb", "catboost_none": "catboost",
              "dummy_most_frequent": "dummy", "dummy_stratified": "dummy"}

ANOMALY_MODELS = {"spc_maxz", "iforest_normal", "iforest_unsup"}

MODEL_DESC = {
    "dummy": "불량률만 아는 기준선 (정보 없음)",
    "spc_maxz": "관리도 기준선: 정상 평균에서 가장 크게 벗어난 변수의 |z| (현장 규칙과 같은 논리)",
    "iforest_normal": "이상탐지: 정상 샘플만으로 학습한 Isolation Forest (처음 보는 이상에도 반응)",
    "iforest_unsup": "이상탐지: unlabeled로 학습한 Isolation Forest (눈금 불일치로 기본 비활성)",
    "logreg": "로지스틱 회귀 (선형 베이스라인, 클래스 가중치)",
    "rf": "랜덤포레스트 (비선형, 소표본에 강건)",
    "lgbm": "LightGBM (부스팅, 얕은 트리+강한 규제)",
    "xgb": "XGBoost (부스팅, 얕은 트리+강한 규제, scale_pos_weight 로 클래스 균형)",
    "catboost": "CatBoost (부스팅, 대칭 트리, auto_class_weights=Balanced)",
    "xgb_none": "XGBoost, 클래스 가중치 없음",
    "catboost_none": "CatBoost, 클래스 가중치 없음",
    "ensemble": "소프트 보팅 앙상블 (logreg+rf+lgbm)",
    "hybrid": "2단계 결합: 관리도 강한 이탈(정상 기준 최대 |z| 상위 1% 밖) OR 지도학습 앙상블 위험점수",
    "dummy_most_frequent": "기준선: 항상 다수 클래스(양품) — 점수가 모두 같음",
    "dummy_stratified": "기준선: 불량률에 맞춰 무작위로 0/1",
    "logreg_none": "로지스틱 회귀, 클래스 가중치 없음",
    "rf_none": "랜덤포레스트, 클래스 가중치 없음",
    "lgbm_none": "LightGBM, 클래스 가중치 없음",
    "logreg_v0": "이전 베이스라인(v0)의 로지스틱 회귀 (C=1, 가중치 없음)",
}
