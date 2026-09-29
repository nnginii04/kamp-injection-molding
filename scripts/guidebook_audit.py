"""가이드북 공식 방법 재현 + 검증 방식별 성능 비교 (실험 G) · 순서 피처 효과 점검 (실험 S).

왜 하는가
- KAMP 가이드북(『사출성형기 AI 데이터셋 분석실습 가이드북』)은 두 방법을 제시한다.
  G1 잡음제거 오토인코더: 양품만 학습 → 복원 오차가 크면 불량. 데이터를 행 단위로 6:4 무작위 분할, 임계값 3~5σ.
  G2 준지도 자기학습(self-training): labeled 로 학습한 분류기(SVM·GNB·RF)가 unlabeled 중 확신이 높은 상위 10% 에
     가짜 라벨을 붙여 학습에 추가하는 것을 반복. labeled 를 행 단위 층화 7:3 분할.
- 두 방법 모두 **행 단위 무작위 분할**을 쓴다. 이 데이터는 LH/RH 두 행이 공정값이 완전히 같으므로(한 shot)
  행 단위로 나누면 같은 shot 이 학습·평가에 갈라져 들어가 성능이 부풀려진다.
- 그래서 같은 모델을 세 가지 검증 방식으로 잰다.
    row   : 가이드북 방식 (행 단위 층화 3-fold) — 누수 있음
    shot  : shot 단위 그룹 3-fold (우리 파이프라인, 같은 구간 안 성능 = 낙관)
    block : 연속 10 shot 묶음 3-fold (처음 보는 생산 구간 = 보수)
  shot·block 분할은 src.cv.cv_splits 그대로라 outputs/fold_id.csv 와 같다.

- 실험 S: 문헌(쿠션·계량의 shot 간 변동 = 충전량 불일치 신호)에 근거해 순서 피처(직전 shot 대비 변화량,
  직전 5 shot 평균 대비 편차)를 넣었을 때 성능이 바뀌는지 같은 분할에서 비교. 순서 피처는 row_id(기록 순서)에
  의존하지만 라벨은 쓰지 않는다 (src.features.add_sequence_features).

- 실험 C: 지도(logreg) 점수 + 양품 전용 이상 점수(AE / Isolation Forest / 관리도)의 백분위 평균(src.models.SupAnomAvg).

실행:  python scripts/guidebook_audit.py [--fast] [--only G|S|C]
결과:  outputs/tables/exp_guidebook_protocol.csv, exp_sequence_features.csv, exp_sup_ae_combo.csv (+ *_raw.csv)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.semi_supervised import SelfTrainingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import features  # noqa: E402
from src.cv import cv_splits, make_groups, run_cv  # noqa: E402
from src.data import assign_shots, load_all  # noqa: E402

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- 가이드북 방법 재현
class GuidebookAE(BaseEstimator, ClassifierMixin):
    """G1 잡음제거 오토인코더 (가이드북 [코드 16] 구조를 sklearn 으로 옮김).
    입력 Dropout 0.3 → Dense 15 → 5 → 15 (ReLU) → 출력, MSE·Adam. 양품(y=0)만 학습.
    TensorFlow 대신 MLPRegressor 를 쓰고, Dropout 은 학습 입력의 30% 를 0 으로 가리는 잡음으로 대신했다.
    점수 = 복원 오차(MSE). predict_proba 의 두 번째 열은 오차를 학습 양품 오차의 5σ 로 나눈 값(0~1 로 자름)이라
    0.5 가 대략 '가이드북 5σ 임계값의 절반' — 순위 지표(AP·검사비율별 재현율)에는 영향 없음."""

    def __init__(self, seed: int = 42, noise: float = 0.3, n_copies: int = 3):
        self.seed, self.noise, self.n_copies = seed, noise, n_copies

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        Xn = np.asarray(X, float)[np.asarray(y) == 0]
        self.scaler_ = MinMaxScaler().fit(Xn)
        Z = self.scaler_.transform(Xn)
        Zin = np.vstack([Z * (rng.random(Z.shape) >= self.noise) for _ in range(self.n_copies)])
        Zout = np.vstack([Z] * self.n_copies)
        self.ae_ = MLPRegressor(hidden_layer_sizes=(15, 5, 15), activation="relu", solver="adam",
                                learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                                validation_fraction=0.1, n_iter_no_change=15, random_state=self.seed)
        self.ae_.fit(Zin, Zout)
        err = self._err(Xn)
        self.thr5_ = float(err.mean() + 5 * err.std())  # 가이드북 예시의 5σ 임계값
        self.classes_ = np.array([0, 1])
        return self

    def _err(self, X):
        Z = self.scaler_.transform(np.asarray(X, float))
        return ((self.ae_.predict(Z) - Z) ** 2).mean(axis=1)

    def predict_proba(self, X):
        s = np.clip(self._err(X) / (2 * self.thr5_), 0, 1)
        return np.c_[1 - s, s]

    def guidebook_predict(self, X):
        return (self._err(X) > self.thr5_).astype(int)


class GuidebookSelfTraining(BaseEstimator, ClassifierMixin):
    """G2 준지도 자기학습 (가이드북 [코드 50]): RF 를 labeled 로 학습 → unlabeled 에서 확신 높은 상위 10% 에
    가짜 라벨 → 합쳐서 재학습, unlabeled 사용분을 다 쓸 때까지 반복. unlabeled 는 학습에만 쓴다.
    속도를 위해 unlabeled 는 제품별로 unl_n 행만 무작위 추출(가이드북의 unlabeled_usage 인자에 해당).
    주의: 대회 unlabeled 안에는 labeled 행이 들어 있어(README 사실 10) 이 방법은 원래부터 평가 행을 간접적으로 본다."""

    unlabeled_pool: dict = {}

    def __init__(self, product: str = "cn7", seed: int = 42, unl_n: int = 1000, pct: float = 0.10):
        self.product, self.seed, self.unl_n, self.pct = product, seed, unl_n, pct

    def fit(self, X, y):
        U = self.unlabeled_pool[self.product]
        rng = np.random.default_rng(self.seed)
        U = U[rng.choice(len(U), size=min(self.unl_n, len(U)), replace=False)]
        Xa = np.vstack([np.asarray(X, float), U])
        ya = np.r_[np.asarray(y), -np.ones(len(U), dtype=int)]
        base = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=self.seed, n_jobs=-1)
        self.m_ = SelfTrainingClassifier(base, criterion="k_best", k_best=max(1, int(len(U) * self.pct)),
                                         max_iter=int(1 / self.pct) + 1).fit(Xa, ya)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(np.asarray(X, float))


EXTRA = {"guidebook_ae": GuidebookAE, "guidebook_selftrain": GuidebookSelfTraining}


def patch_make_model(product: str):
    """src.cv.run_cv 가 부르는 make_model 에 가이드북 모델 두 개를 끼워 넣는다 (다른 모델은 그대로)."""
    import src.cv as cvmod
    from src.models import make_model as orig

    def mm(name, seed):
        if name == "guidebook_ae":
            return GuidebookAE(seed=seed)
        if name == "guidebook_selftrain":
            return GuidebookSelfTraining(product=product, seed=seed)
        return orig(name, seed)

    cvmod.make_model = mm


def row_splits_fn(lab: pd.DataFrame, cfg: dict):
    """가이드북 방식: shot 무시, 행 단위 층화 3-fold (반복마다 seed 변경)."""
    y = lab["y"].to_numpy()

    def fn(rep):
        skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"] + rep)
        return list(skf.split(np.zeros(len(y)), y))
    return fn


def summarize(metrics: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = ["shot_ap", "row_ap", "shot_recall@5%", "shot_recall@10%", "row_recall@5%", "row_f1"]
    cols = [c for c in cols if c in metrics.columns]
    g = metrics.groupby(keys)[cols]
    out = g.mean().round(3)
    sd = g.std().round(3).add_suffix("_sd")
    return pd.concat([out, sd], axis=1).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="반복 2회 (동작 확인용)")
    ap.add_argument("--only", choices=["G", "S", "C"], default=None)
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if args.fast:
        cfg["cv"]["n_repeats"] = 2
    out_tab = ROOT / cfg["output_dir"] / "tables"
    out_tab.mkdir(parents=True, exist_ok=True)

    data = load_all(cfg, ROOT / cfg["data_dir"])
    raw = data["feature_cols"]
    for p in data["labeled"]:
        data["labeled"][p] = assign_shots(data["labeled"][p], raw, p)
        data["unlabeled"][p] = assign_shots(data["unlabeled"][p], raw, f"{p}u")
    base, _ = features.select_base_features(data, raw, True)
    GuidebookSelfTraining.unlabeled_pool = {p: data["unlabeled"][p][base].to_numpy(float) for p in data["unlabeled"]}
    quiet = lambda *_: None  # noqa: E731
    t0 = time.time()

    # ---------------- 실험 G: 가이드북 방법 × 검증 방식
    if args.only in (None, "G"):
        models = ["guidebook_ae", "guidebook_selftrain", "iforest_normal", "logreg", "rf"]
        rows = []
        for p, lab in data["labeled"].items():
            lab = lab.reset_index(drop=True)
            patch_make_model(p)
            for mode in ("row", "shot", "block"):
                if mode == "row":
                    groups = np.arange(len(lab)).astype(str)  # 가이드북: 행마다 독립 (내부 임계값도 행 단위)
                    _, met, _ = run_cv(lab, base, models, cfg, p, "block", log=quiet,
                                       splits_fn=row_splits_fn(lab, cfg), groups=groups)
                else:
                    _, met, _ = run_cv(lab, base, models, cfg, p, mode, log=quiet)
                met["cv_mode"], met["product"] = mode, p
                rows.append(met)
                print(f"[G] {p} {mode} 완료 ({time.time()-t0:.0f}s)", flush=True)
        raw_g = pd.concat(rows, ignore_index=True)
        raw_g.to_csv(out_tab / "exp_guidebook_protocol_raw.csv", index=False, encoding="utf-8-sig")
        tab = summarize(raw_g, ["product", "model", "cv_mode"])
        tab.to_csv(out_tab / "exp_guidebook_protocol.csv", index=False, encoding="utf-8-sig")
        piv = tab.pivot_table(index=["product", "model"], columns="cv_mode", values="shot_ap")[["row", "shot", "block"]]
        piv["inflation_row/block"] = (piv["row"] / piv["block"]).round(1)
        print("\n[G] shot 단위 AP (검증 방식별)\n", piv.to_string())

    # ---------------- 실험 S: 순서 피처 on/off
    if args.only in (None, "S"):
        w = int(cfg["features"].get("sequence_window", 5))
        seq = features.sequence_columns(base, w)
        models = ["logreg", "catboost", "rf"]
        rows = []
        for p, lab in data["labeled"].items():
            lab = features.add_sequence_features(lab, base, w).reset_index(drop=True)
            for mode in ("shot", "block"):
                lit = [c for c in ("Cushion_Position", "Plasticizing_Time", "Max_Injection_Pressure") if c in base]
                seq3 = [f"d1_{c}" for c in lit] + [f"dev{w}_{c}" for c in lit]  # 문헌 근거 3변수만 (불량 수 대비 과적합 방지)
                for tag, feats in (("base", base), ("base+seq", base + seq), ("base+seq_lit3", base + seq3)):
                    _, met, _ = run_cv(lab, feats, models, cfg, p, mode, log=quiet)
                    met["cv_mode"], met["product"], met["features"] = mode, p, tag
                    rows.append(met)
                print(f"[S] {p} {mode} 완료 ({time.time()-t0:.0f}s)", flush=True)
        raw_s = pd.concat(rows, ignore_index=True)
        raw_s.to_csv(out_tab / "exp_sequence_features_raw.csv", index=False, encoding="utf-8-sig")
        tab = summarize(raw_s, ["product", "model", "cv_mode", "features"])
        tab.to_csv(out_tab / "exp_sequence_features.csv", index=False, encoding="utf-8-sig")
        piv = tab.pivot_table(index=["product", "model", "cv_mode"], columns="features", values="shot_ap")
        print("\n[S] shot 단위 AP (순서 피처 유무)\n", piv.to_string())
    # ---------------- 실험 C: 지도 + 양품전용 이상점수 결합 (src.models 의 logreg_ae 와 대조군)
    if args.only == "C":
        models = ["logreg", "ae_normal", "logreg_ae", "logreg_if", "logreg_spc"]
        rows = []
        for p, lab in data["labeled"].items():
            lab = lab.reset_index(drop=True)
            patch_make_model(p)
            for mode in ("shot", "block"):
                _, met, _ = run_cv(lab, base, models, cfg, p, mode, log=quiet)
                met["cv_mode"], met["product"] = mode, p
                rows.append(met)
                print(f"[C] {p} {mode} 완료 ({time.time()-t0:.0f}s)", flush=True)
        raw_c = pd.concat(rows, ignore_index=True)
        raw_c.to_csv(out_tab / "exp_sup_ae_combo_raw.csv", index=False, encoding="utf-8-sig")
        tab = summarize(raw_c, ["product", "model", "cv_mode"])
        tab.to_csv(out_tab / "exp_sup_ae_combo.csv", index=False, encoding="utf-8-sig")
        print(tab[["product", "model", "cv_mode", "shot_ap", "row_ap", "row_recall@5%", "shot_recall@10%"]].to_string(index=False))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
