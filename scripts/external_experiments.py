"""공개 원본(외부 데이터)을 쓰는 실험 3종 — 운영진 확인(2026-09-30): 출처·이유·방법을 보고서에 밝히면 사용 가능.

출처: KAMP 「사출성형기 AI 데이터셋」 원본 (04. Dataset_Molding.zip) 의 moldset_labeled.csv, labeled_data.csv.
대회 파일과의 연결: 대회 labeled 의 Unnamed: 0 = 원본 moldset_labeled.csv 의 Unnamed: 0 (scripts/verify_public_link.py 로 검증).

X1  캐비티 위치(side, RH=1)를 모델 입력에 추가 — 같은 교차검증 분할(shot·block)에서 넣기 전/후 비교
    · 한 번 사출(shot)의 LH/RH 는 공정값이 같아 공정값만으로는 두 제품의 위험을 구분할 수 없다.
      현장에서 LH/RH 는 부품 번호로 이미 알고 있으므로 입력으로 쓸 수 있다.
    · 결합 모델(logreg_spc 등)의 이상 점수 쪽에는 side 를 넣지 않는다 (src.models.SIDE_COLS).
X2  검사 정책 비교 — 모델 점수만 / RH 먼저(무작위) / RH 먼저 + 모델 순 / side 를 넣은 모델
X3  미래 검증 — 대회 labeled 전체로 학습 → 대회 기간 이후 라벨(CN7 10/27~11/03, RG3 11/04~11/06)을 채점.
    · 미래 행은 대회 파일과 같은 눈금이 되도록 '대회 labeled 원단위의 제품별 평균·표준편차(ddof=0)'로 표준화
      (대회 파일의 z 값이 정확히 이 방식으로 만들어졌음을 먼저 확인한다).

실행: python scripts/external_experiments.py --public <원본 폴더 또는 zip> [--fast] [--only X1|X2|X3]
결과: outputs/external/*.csv (+ 콘솔 요약)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _public  # noqa: E402
from src import features  # noqa: E402
from src.cv import run_cv  # noqa: E402
from src.data import assign_shots, load_all  # noqa: E402
from src.evaluate import evaluate_oof, expected_topk  # noqa: E402
from src.models import make_model  # noqa: E402

warnings.filterwarnings("ignore")
BUDGETS = [0.02, 0.05, 0.10, 0.20]
EQUIP = "650톤-우진2호기"
PREFIX = {"cn7": "CN7", "rg3": "RG3"}
RPM_FIX = True
POOL_N = 5000


# --------------------------------------------------------------------------- 데이터 연결
def link_side(lab: pd.DataFrame, pub: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """대회 labeled 에 원본의 TimeStamp·PART_NAME·Reason·원단위 값을 붙이고 side_rh(RH=1) 를 만든다."""
    cols = ["Unnamed: 0", "TimeStamp", "PART_NAME", "Reason"] + feats
    m = lab.merge(pub[cols].rename(columns={f: f"raw_{f}" for f in feats}),
                  left_on="row_id", right_on="Unnamed: 0", how="left", validate="1:1")
    assert m["PART_NAME"].notna().all(), "원본 연결 실패 (Unnamed: 0 불일치)"
    m["side_rh"] = m["PART_NAME"].str.contains(r"\bRH\b", regex=True).astype(int)
    return m.drop(columns=["Unnamed: 0"])


def zscale_params(linked: pd.DataFrame, feats: list[str]) -> tuple[pd.Series, pd.Series, float]:
    """대회 파일 z = (원단위 − 평균) / 표준편차(ddof=0) 인지 확인하고, 그 평균·표준편차를 돌려준다."""
    raw = linked[[f"raw_{f}" for f in feats]].astype(float)
    raw.columns = feats
    mu, sd = raw.mean(), raw.std(ddof=0)
    z = (raw - mu) / sd.replace(0, 1.0)
    err = float(np.nanmax(np.abs(z.to_numpy() - linked[feats].to_numpy(float))))
    return mu, sd, err


def future_rows(ld: pd.DataFrame, linked: pd.DataFrame, product: str, feats: list[str],
                mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    """labeled_data.csv 에서 대회 기간 이후 같은 설비·같은 제품의 라벨 행 → 대회 눈금의 z 값."""
    d = ld[ld["EQUIP_NAME"].eq(EQUIP) & ld["PART_NAME"].str.startswith(PREFIX[product])].copy()
    d = d.drop_duplicates()
    d = d.drop_duplicates(subset=["TimeStamp", "PART_NAME"])
    t_end = pd.to_datetime(linked["TimeStamp"]).max()
    d = d[pd.to_datetime(d["TimeStamp"]) > t_end].copy()
    seen = set(zip(linked["TimeStamp"].astype(str), linked["PART_NAME"].astype(str)))
    d = d[[(t, n) not in seen for t, n in zip(d["TimeStamp"].astype(str), d["PART_NAME"].astype(str))]]
    raw = d[feats].astype(float)
    # 기록 눈금 변경 보정: Average_Screw_RPM 이 10/29 부터 약 1/10 로 기록됨 (CN7 292.5→29.2, RG3 290.5→29.0,
    # 같은 날 Max_Screw_RPM·다른 변수는 그대로) → 10배로 되돌린다. 보정 전후 모두 보고 (--no-rpm-fix).
    n_fix = 0
    if RPM_FIX and "Average_Screw_RPM" in raw:
        low = raw["Average_Screw_RPM"] < 100
        n_fix = int(low.sum())
        raw.loc[low, "Average_Screw_RPM"] *= 10
    z = (raw - mu) / sd.replace(0, 1.0)
    out = z.reset_index(drop=True)
    out.attrs["rpm_fixed_rows"] = n_fix
    out["y"] = d["PassOrFail"].eq("N").astype(int).to_numpy()
    out["Reason"] = d["Reason"].to_numpy()
    out["TimeStamp"] = d["TimeStamp"].to_numpy()
    out["PART_NAME"] = d["PART_NAME"].to_numpy()
    out["side_rh"] = out["PART_NAME"].str.contains(r"\bRH\b", regex=True).astype(int)
    out["shot_id"] = product + "f_" + out["TimeStamp"].astype(str)
    out["product"] = product
    return out.sort_values(["TimeStamp", "PART_NAME"]).reset_index(drop=True)


# --------------------------------------------------------------------------- X4: 과거 정상 운전 기록을 이상 점수 기준으로
class PoolSPC:
    """관리도 max|z| 의 기준(평균·표준편차)과 점수 분포를 labeled 가 아니라 '과거 정상 운전 기록'에서 잡는다.
    기록: 원본 unlabeled_data.csv 의 같은 설비·같은 제품 행 중 센서 미기록(idle) 행을 뺀 것,
    labeled 기간 시작 '이전' 최근 n 행만 (평가 행과 시간이 겹치지 않음, 라벨 불사용)."""

    def __init__(self, pool: pd.DataFrame, feats: list[str]):
        P = pool[feats].to_numpy(float)
        self.feats, self.mu, self.sd = feats, P.mean(axis=0), np.maximum(P.std(axis=0), 0.01)
        self.ref = np.sort(self.raw(pool))

    def raw(self, X):
        Z = (X[self.feats].to_numpy(float) - self.mu) / self.sd
        return np.abs(Z).max(axis=1)

    def pct(self, X):
        return (np.searchsorted(self.ref, self.raw(X), side="right") + 0.5) / (len(self.ref) + 1)


class SupPoolAvg:
    """logreg(학습 fold) 백분위 + PoolSPC 백분위 평균. sklearn 흉내만 (run_cv 가 clone 하므로 get_params 제공)."""
    _pools: dict = {}

    def __init__(self, product="cn7", seed=42):
        self.product, self.seed = product, seed

    def get_params(self, deep=True):
        return {"product": self.product, "seed": self.seed}

    def set_params(self, **k):
        for a, v in k.items():
            setattr(self, a, v)
        return self

    def fit(self, X, y):
        from src.models import ecdf_score  # noqa: F401
        self.sup_ = make_model("logreg", self.seed).fit(X, y)
        s = self.sup_.predict_proba(X)[:, 1]
        self.ref_, self.ref_sd_ = np.sort(s), float(np.std(s)) or 1.0
        self.pool_ = self._pools[self.product]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        from src.models import ecdf_score
        ps = ecdf_score(self.ref_, self.sup_.predict_proba(X)[:, 1], self.ref_sd_)
        p = 0.5 * ps + 0.5 * self.pool_.pct(X)
        return np.c_[1 - p, p]


def build_pool(un: pd.DataFrame, product: str, start: str, feats: list[str], mu, sd, n: int = 5000):
    d = un[un["EQUIP_NAME"].eq(EQUIP) & un["PART_NAME"].str.startswith(PREFIX[product])].copy()
    idle_cols = ["Max_Screw_RPM", "Average_Screw_RPM", "Average_Back_Pressure", "Barrel_Temperature_1",
                 "Mold_Temperature_3", "Mold_Temperature_4"]
    d = d[~(d[idle_cols] == 0).all(axis=1)]
    d = d[pd.to_datetime(d["TimeStamp"]) < pd.to_datetime(start)].sort_values("TimeStamp").tail(n)
    raw = d[feats].astype(float)
    z = (raw - mu) / sd.replace(0, 1.0)
    z.attrs["range"] = (d["TimeStamp"].min(), d["TimeStamp"].max())
    return z.reset_index(drop=True)


# --------------------------------------------------------------------------- 지표
def recall_at(y, s, b):
    y = np.asarray(y, float)
    return expected_topk(y, np.asarray(s, float), int(np.ceil(len(y) * b))) / y.sum() if y.sum() else np.nan


def rh_first_recall(y, side, s, b):
    """RH 제품 먼저(그 안은 점수 순), 남으면 LH 점수 순."""
    y, side, s = np.asarray(y, float), np.asarray(side), np.asarray(s, float)
    k, got = int(np.ceil(len(y) * b)), 0.0
    for v in (1, 0):
        m = side == v
        take = min(k, int(m.sum()))
        if take > 0:
            got += expected_topk(y[m], s[m], take)
        k -= take
        if k <= 0:
            break
    return got / y.sum() if y.sum() else np.nan


def ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if np.sum(y) else np.nan


# --------------------------------------------------------------------------- 실행
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--public", default=None, help="원본 폴더 또는 zip (기본: config external.public_path)")
    ap_.add_argument("--fast", action="store_true")
    ap_.add_argument("--only", choices=["X1", "X2", "X3", "X4"], default=None)
    ap_.add_argument("--models", default="logreg,logreg_spc,logreg_ae,catboost")
    ap_.add_argument("--no-rpm-fix", action="store_true", help="Average_Screw_RPM 눈금 보정을 끄고 채점")
    ap_.add_argument("--pool-n", type=int, default=5000, help="X4 기준 기록 행 수 (labeled 시작 직전 최근 n행)")
    args = ap_.parse_args()
    global RPM_FIX, POOL_N
    RPM_FIX, POOL_N = not args.no_rpm_fix, args.pool_n

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if args.fast:
        cfg["cv"]["n_repeats"] = 2
    src = _public.public_path(cfg, args.public)
    pub = _public.read_csv(src, "moldset_labeled.csv")
    ld = _public.read_csv(src, "labeled_data.csv")
    out = ROOT / cfg["output_dir"] / "external"
    out.mkdir(parents=True, exist_ok=True)

    data = load_all(cfg, ROOT / cfg["data_dir"])
    raw = data["feature_cols"]
    base, _ = features.select_base_features(data, raw, True)
    models = args.models.split(",")
    t0, log = time.time(), []

    linked, fut = {}, {}
    for p in data["labeled"]:
        lab = assign_shots(data["labeled"][p], raw, p).reset_index(drop=True)
        linked[p] = link_side(lab, pub, raw)
        mu, sd, err = zscale_params(linked[p], raw)
        fut[p] = future_rows(ld, linked[p], p, raw, mu, sd)
        L = linked[p]
        msg = (f"{p}: 대회 labeled {len(L)}행(불량 {int(L.y.sum())} = RH {int(L.y[L.side_rh.eq(1)].sum())} / "
               f"LH {int(L.y[L.side_rh.eq(0)].sum())}), 눈금 재현 최대 오차 {err:.2e} | 미래 {len(fut[p])}행 "
               f"({fut[p].TimeStamp.min()[:10]}~{fut[p].TimeStamp.max()[:10]}), 불량 {int(fut[p].y.sum())} "
               f"{fut[p].loc[fut[p].y.eq(1), 'Reason'].value_counts().to_dict()}, RPM 눈금 보정 {fut[p].attrs.get('rpm_fixed_rows', 0)}행")
        print(msg, flush=True)
        log.append(msg)

    # ---- X1 + X2: 교차검증
    if args.only in (None, "X1", "X2"):
        mets, pol = [], []
        for p, L in linked.items():
            for mode in ("block", "shot"):
                for tag, feats in (("base", base), ("base+side", base + ["side_rh"])):
                    oof, met, _ = run_cv(L, feats, models, cfg, p, mode, log=lambda *_: None)
                    met["product"], met["cv_mode"], met["features"] = p, mode, tag
                    mets.append(met)
                    oof = oof.merge(L[["row_id", "side_rh"]], on="row_id", how="left")
                    for (name, rep), g in oof.groupby(["model", "repeat"]):
                        for b in BUDGETS:
                            row = dict(product=p, cv_mode=mode, features=tag, model=name, repeat=rep, budget=b)
                            pol.append({**row, "policy": "model", "recall": recall_at(g.y, g.p, b)})
                            if tag == "base":
                                pol.append({**row, "policy": "rh_then_model",
                                            "recall": rh_first_recall(g.y, g.side_rh, g.p, b)})
                                pol.append({**row, "policy": "rh_random",
                                            "recall": rh_first_recall(g.y, g.side_rh, np.zeros(len(g)), b)})
                    print(f"[X1] {p} {mode} {tag} ({time.time()-t0:.0f}s)", flush=True)
        M = pd.concat(mets, ignore_index=True)
        M.to_csv(out / "x1_side_cv_raw.csv", index=False, encoding="utf-8-sig")
        keys = ["product", "cv_mode", "features", "model"]
        vals = ["row_ap", "shot_ap", "row_recall@2%", "row_recall@5%", "row_recall@10%"]
        t1 = M.groupby(keys)[vals].mean().round(3).join(M.groupby(keys)["row_ap"].std().round(3).rename("row_ap_sd"))
        t1.reset_index().to_csv(out / "x1_side_cv.csv", index=False, encoding="utf-8-sig")
        P = pd.DataFrame(pol)
        t2 = (P.groupby(["product", "cv_mode", "features", "model", "policy", "budget"])["recall"].mean().round(3)
              .unstack("budget").reset_index())
        t2.to_csv(out / "x2_policy.csv", index=False, encoding="utf-8-sig")
        pd.set_option("display.width", 250)
        print("\n[X1] 행 단위 지표 (반복 평균)\n", t1.to_string())
        print("\n[X2] 검사 정책별 재현율 (행 단위)\n", t2.to_string(index=False))

    # ---- X3: 미래 검증
    if args.only in (None, "X3"):
        rows, ranks = [], []
        for p, L in linked.items():
            F = fut[p]
            if not len(F) or F.y.sum() == 0:
                continue
            for tag, feats in (("base", base), ("base+side", base + ["side_rh"])):
                for name in models:
                    m = make_model(name, cfg["seed"]).fit(L[feats], L["y"].to_numpy())
                    s = m.predict_proba(F[feats])[:, 1]
                    r = dict(product=p, features=tag, model=name, n=len(F), defects=int(F.y.sum()),
                             row_ap=round(ap(F.y, s), 3))
                    for b in BUDGETS:
                        r[f"recall@{int(b*100)}%"] = round(recall_at(F.y, s, b), 3)
                        if tag == "base":
                            r[f"rh_then_model@{int(b*100)}%"] = round(rh_first_recall(F.y, F.side_rh, s, b), 3)
                    rows.append(r)
                    pct = pd.Series(s).rank(ascending=False, pct=True).to_numpy()
                    for i in np.flatnonzero(F.y.to_numpy() == 1):
                        ranks.append(dict(product=p, features=tag, model=name, TimeStamp=F.TimeStamp.iat[i],
                                          part=F.PART_NAME.iat[i], reason=F.Reason.iat[i],
                                          top_pct=round(float(pct[i]) * 100, 1)))
            for b in BUDGETS:  # 기준선
                rows.append(dict(product=p, features="-", model="rh_random", n=len(F), defects=int(F.y.sum()),
                                 **{f"recall@{int(b*100)}%": round(rh_first_recall(F.y, F.side_rh, np.zeros(len(F)), b), 3)}))
        R = pd.DataFrame(rows)
        R = R.groupby(["product", "features", "model"], as_index=False).first()
        sfx = "" if RPM_FIX else "_norpmfix"
        R.to_csv(out / f"x3_future{sfx}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(ranks).to_csv(out / f"x3_future_defect_ranks{sfx}.csv", index=False, encoding="utf-8-sig")
        print("\n[X3] 미래 검증 (대회 labeled 전체로 학습 → 이후 기간 채점)\n", R.to_string(index=False))
    # ---- X4: 과거 정상 운전 기록 기준 관리도 + logreg (교차검증 + 미래)
    if args.only == "X4":
        import src.cv as cvmod
        un = _public.read_csv(src, "unlabeled_data.csv")
        orig_make = cvmod.make_model
        rows4, fut4 = [], []
        for p, L in linked.items():
            mu, sd, _ = zscale_params(L, raw)
            pool = build_pool(un, p, L["TimeStamp"].min(), base, mu[base], sd[base], n=POOL_N)
            SupPoolAvg._pools[p] = PoolSPC(pool, base)
            msg = f"[X4] {p} 기준 기록 {len(pool)}행 {pool.attrs['range']}"
            print(msg, flush=True); log.append(msg)
            cvmod.make_model = lambda name, seed, _p=p: SupPoolAvg(_p, seed) if name == "logreg_pool" else orig_make(name, seed)
            for mode in ("block", "shot"):
                _, met, _ = run_cv(L, base, ["logreg", "logreg_spc", "logreg_pool"], cfg, p, mode, log=lambda *_: None)
                met["product"], met["cv_mode"] = p, mode
                rows4.append(met)
                print(f"[X4] {p} {mode} ({time.time()-t0:.0f}s)", flush=True)
            F = fut[p]
            if len(F) and F.y.sum():
                for name in ("logreg", "logreg_spc", "logreg_pool"):
                    m = (SupPoolAvg(p, cfg["seed"]) if name == "logreg_pool" else make_model(name, cfg["seed"])).fit(L[base], L["y"].to_numpy())
                    sc = m.predict_proba(F[base])[:, 1]
                    fut4.append(dict(product=p, model=name, row_ap=round(ap(F.y, sc), 3),
                                     **{f"recall@{int(b*100)}%": round(recall_at(F.y, sc, b), 3) for b in BUDGETS}))
        cvmod.make_model = orig_make
        M4 = pd.concat(rows4, ignore_index=True)
        t4 = M4.groupby(["product", "cv_mode", "model"])[["shot_ap", "row_ap", "row_recall@5%", "row_recall@10%"]].agg(["mean", "std"]).round(3)
        t4.to_csv(out / f"x4_pool{POOL_N}_cv.csv", encoding="utf-8-sig")
        pd.DataFrame(fut4).to_csv(out / f"x4_pool{POOL_N}_future.csv", index=False, encoding="utf-8-sig")
        print(t4.to_string()); print(pd.DataFrame(fut4).to_string(index=False))

    (out / "run_log.txt").write_text("\n".join(log) + f"\n총 {time.time()-t0:.0f}s\n", encoding="utf-8")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
