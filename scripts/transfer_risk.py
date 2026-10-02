"""
transfer_risk.py — CN7 ↔ RG3 전이 위험 스크리닝 (라벨 없이 계산, 해석·검증 전용)

"Screen Before You Reuse" (2026)의 라벨-프리 전이 위험 통계 3종을 우리 데이터에 맞게 재구현:
  (a) 변수별 Wasserstein-1 거리  — 어떤 변수가 두 금형 사이에서 얼마나 다른가 (표준화 단위)
  (b) RBF-MMD²(median heuristic) + 순열검정 p값 — 24변수 전체 분포가 다른가
  (c) 도메인 분류기 AUC (Logistic / RandomForest, 5-fold OOF) — 공정값만 보고 금형을 맞출 수 있나
  (+) 같은 금형을 반으로 갈라 계산한 "무이동(no-shift) 기준선" — 위 세 값의 바닥

입력은 반드시 원단위. 대회 배포본은 금형별로 따로 표준화돼 있어 두 금형이 인위적으로 같아 보임.
기본은 KAMP 공개 원본(config.external.public_path 의 zip)에서 moldset_labeled.csv 를 읽고,
PART_NAME 에 'CN7' / 'RG3' 가 들어간 행으로 금형을 나눈다. 원본은 읽기만 하고 저장소로 복사하지 않는다
(scripts/_public.py 규칙 그대로). 결과는 outputs/analysis/ 에만 저장.

실행)
  python scripts/transfer_risk.py                       # config.json 의 external.public_path 사용
  python scripts/transfer_risk.py --public "~/Downloads/04. Dataset_Molding.zip"
  python scripts/transfer_risk.py --competition-only    # 대회 labeled 2,393행에 해당하는 원본 행만
  python scripts/transfer_risk.py --input 어떤_원단위.csv --mold-col mold   # 직접 파일을 줄 때

출력)
  outputs/analysis/transfer_risk_per_variable.csv   변수별 W1(표준화·원단위), 금형별 중앙값/IQR, LR 계수 순위
  outputs/analysis/transfer_risk_summary.csv        MMD², p값, 도메인 AUC, 무이동 기준선
  outputs/analysis/transfer_risk_summary.json       위 내용 + 실행 설정
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _public  # noqa: E402  (scripts/_public.py — 공개 원본 읽기 전용 헬퍼)

NON_FEATURES = {
    "Unnamed: 0", "row_id", "shot_id", "shot_ts_id", "_id", "PassOrFail", "Reason", "y",
    "TimeStamp", "PART_NAME", "EQUIP_NAME", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL",
    "ERR_FACT_QTY", "mold", "product", "side", "cavity", "cavity_pos", "in_unlabeled", "in_labeled",
    "is_idle", "unl_orig_index", "fold", "fold_rep", "Clamp_Open_Position",
}


# ----------------------------------------------------------------------------- 데이터
def competition_features(cfg: dict) -> list[str]:
    """대회 labeled 파일의 열 = 우리가 쓰는 입력 변수 (id·라벨 제외). 원본에서 같은 열만 뽑는다."""
    p = next(iter(cfg["products"].values()))
    cols = pd.read_csv(ROOT / cfg["data_dir"] / p["labeled"], nrows=1).columns
    return [c for c in cols if c not in (cfg["id_col"], cfg["label_col"])]


def load_from_public(cfg: dict, public: str | None, feats: list[str], competition_only: bool):
    src = _public.public_path(cfg, public)
    if not src.exists():
        raise SystemExit(f"공개 원본을 찾지 못했습니다: {src}\n  --public 으로 zip 경로를 주거나 config.external.public_path 를 확인하세요.")
    usecols = [cfg["id_col"], "PART_NAME", "TimeStamp"] + feats
    df = _public.read_csv(src, "moldset_labeled.csv", usecols=lambda c: c in usecols)
    df["mold"] = df["PART_NAME"].astype(str).str.upper().str.extract(r"(CN7|RG3)", expand=False).str.lower()
    df = df.dropna(subset=["mold"])
    if competition_only:
        keep = []
        for name, files in cfg["products"].items():
            ids = set(pd.read_csv(ROOT / cfg["data_dir"] / files["labeled"], usecols=[cfg["id_col"]])[cfg["id_col"]])
            keep.append(df[(df["mold"] == name) & (df[cfg["id_col"]].isin(ids))])
        df = pd.concat(keep)
    print(f"[data] 공개 원본 {src.name} → moldset_labeled.csv {len(df)}행 "
          f"(cn7 {int((df['mold']=='cn7').sum())}, rg3 {int((df['mold']=='rg3').sum())})"
          + (" [대회 행만]" if competition_only else " [원본 전체: CN7 10/27 214행 포함]"))
    return df


def pick_features(df: pd.DataFrame, drop: set[str]) -> list[str]:
    feats = [c for c in df.columns if c not in NON_FEATURES | drop and pd.api.types.is_numeric_dtype(df[c])]
    if not feats:
        raise SystemExit("숫자형 입력 변수를 찾지 못했습니다.")
    return feats


def dedupe_shots(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """LH/RH 두 행은 공정값이 같으므로 한 shot 을 두 번 세지 않도록 제거."""
    return df.drop_duplicates(subset=feats).reset_index(drop=True)


# ----------------------------------------------------------------------------- (a) 변수별 W1
def per_variable_w1(A: pd.DataFrame, B: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    sd = pd.concat([A[feats], B[feats]]).std(ddof=0)
    rows = []
    for f in feats:
        if sd[f] == 0:
            continue
        w_raw = wasserstein_distance(A[f].to_numpy(), B[f].to_numpy())
        rows.append({"feature": f, "w1_std": w_raw / sd[f], "w1_raw": w_raw,
                     "median_A": float(A[f].median()), "median_B": float(B[f].median()),
                     "iqr_A": float(A[f].quantile(.75) - A[f].quantile(.25)),
                     "iqr_B": float(B[f].quantile(.75) - B[f].quantile(.25))})
    return pd.DataFrame(rows).sort_values("w1_std", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- (b) MMD
def _mmd2_from_kernel(K: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> float:
    Kxx, Kyy, Kxy = K[np.ix_(ix, ix)], K[np.ix_(iy, iy)], K[np.ix_(ix, iy)]
    m, n = len(ix), len(iy)
    return float((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1)) - 2 * Kxy.mean())


def mmd_rbf(X: np.ndarray, Y: np.ndarray, n_perm: int = 500, seed: int = 0, max_rows: int = 2000) -> dict:
    """비편향 MMD² (RBF, median heuristic) + 순열검정. X, Y 는 표준화된 행렬."""
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        X = X[rng.choice(len(X), max_rows, replace=False)]
    if len(Y) > max_rows:
        Y = Y[rng.choice(len(Y), max_rows, replace=False)]
    Z = np.vstack([X, Y])
    D = cdist(Z, Z, "euclidean")
    bw = float(np.median(D[np.triu_indices_from(D, k=1)]))
    K = np.exp(-(D ** 2) / (2 * bw ** 2))
    ix, iy = np.arange(len(X)), np.arange(len(X), len(Z))
    obs = _mmd2_from_kernel(K, ix, iy)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(Z))
        perm[i] = _mmd2_from_kernel(K, p[:len(X)], p[len(X):])
    return {"mmd2": obs, "mmd2_perm_p": float((np.sum(perm >= obs) + 1) / (n_perm + 1)),
            "mmd2_perm_95pct": float(np.quantile(perm, .95)), "mmd_bandwidth": bw,
            "n_x": int(len(X)), "n_y": int(len(Y))}


# ----------------------------------------------------------------------------- (c) 도메인 분류기
def domain_auc(XA: np.ndarray, XB: np.ndarray, feats: list[str], seed: int = 0):
    X = np.vstack([XA, XB]); y = np.r_[np.zeros(len(XA)), np.ones(len(XB))]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    p_lr = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
    p_rf = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
    out = {"domain_auc_lr": float(roc_auc_score(y, p_lr)), "domain_auc_rf": float(roc_auc_score(y, p_rf))}
    lr.fit(X, y)
    coef = pd.DataFrame({"feature": feats, "lr_abs_coef": np.abs(lr[-1].coef_[0])})
    coef["lr_coef_rank"] = coef["lr_abs_coef"].rank(ascending=False).astype(int)
    return out, coef


# ----------------------------------------------------------------------------- 기준선
def within_domain_baseline(df: pd.DataFrame, feats: list[str], seed: int, n_perm: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    h1, h2 = df.iloc[idx[:len(df) // 2]], df.iloc[idx[len(df) // 2:]]
    sc = StandardScaler().fit(df[feats])
    X1, X2 = sc.transform(h1[feats]), sc.transform(h2[feats])
    w = per_variable_w1(h1, h2, feats); m = mmd_rbf(X1, X2, n_perm=n_perm, seed=seed); a, _ = domain_auc(X1, X2, feats, seed)
    return {"baseline_w1_std_max": float(w["w1_std"].max()), "baseline_w1_std_mean": float(w["w1_std"].mean()),
            "baseline_mmd2": m["mmd2"], "baseline_mmd2_perm_p": m["mmd2_perm_p"],
            "baseline_domain_auc_lr": a["domain_auc_lr"], "baseline_domain_auc_rf": a["domain_auc_rf"]}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--public", default=None, help="공개 원본 zip/폴더 (기본 config.external.public_path)")
    ap.add_argument("--competition-only", action="store_true", help="대회 labeled 2,393행에 해당하는 원본 행만 사용")
    ap.add_argument("--input", help="공개 원본 대신 쓸 원단위 CSV (금형 열 포함)")
    ap.add_argument("--standardized", action="store_true",
                    help="대회 배포본(금형별 표준화)으로 계산 — 설정값 차이를 제거한 뒤 '운전 패턴'이 얼마나 다른지")
    ap.add_argument("--mold-col", default="mold"); ap.add_argument("--a", default="cn7"); ap.add_argument("--b", default="rg3")
    ap.add_argument("--drop", nargs="*", default=[], help="추가로 제외할 열")
    ap.add_argument("--no-dedupe", action="store_true", help="LH/RH 중복 행 제거하지 않음")
    ap.add_argument("--jitter", action="store_true",
                    help="각 변수의 기록 눈금(인접 고유값 최소 간격) ±½ 범위의 균등 잡음을 더해 '정확한 값' 지문을 지운 뒤 계산 — 지문 효과 확정용")
    ap.add_argument("--min-unique", type=int, default=0,
                    help="고유값 개수가 이 값 이하인(설정값처럼 이산적인) 변수를 제외 — 트리 모델이 이산값을 지문처럼 외우는지 진단용")
    ap.add_argument("--n-perm", type=int, default=500); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="출력 접두어 (기본 <output_dir>/analysis/transfer_risk)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.standardized:
        parts = []
        for name, files in cfg["products"].items():
            d = pd.read_csv(ROOT / cfg["data_dir"] / files["labeled"]); d["mold"] = name; parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        A, B = df[df["mold"] == "cn7"].copy(), df[df["mold"] == "rg3"].copy()
        feats = pick_features(df, set(args.drop))
        print(f"[data] 대회 배포본(금형별 표준화) labeled: cn7 {len(A)}행, rg3 {len(B)}행")
    elif args.input:
        df = pd.read_csv(args.input)
        key = df[args.mold_col].astype(str).str.lower()
        A, B = df[key.str.contains(args.a)].copy(), df[key.str.contains(args.b)].copy()
        feats = pick_features(df, set(args.drop))
    else:
        feats = [f for f in competition_features(cfg) if f not in NON_FEATURES | set(args.drop)]
        df = load_from_public(cfg, args.public, feats, args.competition_only)
        A, B = df[df["mold"] == "cn7"].copy(), df[df["mold"] == "rg3"].copy()
    if len(A) == 0 or len(B) == 0:
        raise SystemExit(f"한쪽 금형 행이 0개입니다 (A={len(A)}, B={len(B)}).")

    n_rows = (len(A), len(B))
    if not args.no_dedupe:
        A, B = dedupe_shots(A, feats), dedupe_shots(B, feats)
    nun = pd.DataFrame({"feature": feats, "nunique_A": [A[f].nunique() for f in feats], "nunique_B": [B[f].nunique() for f in feats]})
    if args.min_unique > 0:
        discrete = nun[nun[["nunique_A", "nunique_B"]].min(axis=1) <= args.min_unique]["feature"].tolist()
        feats = [f for f in feats if f not in discrete]
        print(f"[data] 고유값 ≤{args.min_unique} 인 이산 변수 {len(discrete)}개 제외: {discrete}")
    pooled_sd = pd.concat([A[feats], B[feats]]).std(ddof=0)
    const = [f for f in feats if pooled_sd[f] == 0]
    feats = [f for f in feats if f not in const]
    print(f"[data] 변수 {len(feats)}개{' (상수열 제외: ' + ', '.join(const) + ')' if const else ''} | "
          f"CN7 행 {n_rows[0]} → shot {len(A)} | RG3 행 {n_rows[1]} → shot {len(B)}")

    if args.jitter:
        rng = np.random.default_rng(args.seed)
        for f in feats:
            steps = [np.diff(np.unique(d[f].to_numpy())) for d in (A, B)]
            step = min(float(x[x > 0].min()) for x in steps if (x > 0).any())
            A[f] = A[f] + rng.uniform(-step / 2, step / 2, len(A))
            B[f] = B[f] + rng.uniform(-step / 2, step / 2, len(B))
        print(f"[data] jitter 적용: 변수별 기록 눈금 ±½ 균등 잡음 (seed {args.seed})")
    sc = StandardScaler().fit(pd.concat([A[feats], B[feats]]))
    XA, XB = sc.transform(A[feats]), sc.transform(B[feats])
    print("[a] 변수별 Wasserstein ..."); w1 = per_variable_w1(A, B, feats)
    print("[b] MMD + 순열검정 ..."); mmd = mmd_rbf(XA, XB, n_perm=args.n_perm, seed=args.seed)
    print("[c] 도메인 분류기 AUC ..."); auc, coef = domain_auc(XA, XB, feats, seed=args.seed)
    print("[+] 무이동 기준선 (CN7 반분 / RG3 반분) ...")
    base_A = within_domain_baseline(A, feats, args.seed, min(args.n_perm, 200))
    base_B = within_domain_baseline(B, feats, args.seed, min(args.n_perm, 200))

    per_var = w1.merge(coef, on="feature", how="left").merge(nun, on="feature", how="left")
    src_name = "대회 배포본 (금형별 표준화)" if args.standardized else (args.input or "KAMP 공개 원본 moldset_labeled.csv (원단위)")
    summary = {"A": "cn7", "B": "rg3", "source": src_name,
               "competition_only": bool(args.competition_only), "n_features": len(feats),
               "n_shots_A": int(len(A)), "n_shots_B": int(len(B)),
               "w1_std_max": float(w1["w1_std"].max()), "w1_std_mean": float(w1["w1_std"].mean()),
               "w1_top3": ", ".join(w1["feature"].head(3)), "n_features_w1_gt_1": int((w1["w1_std"] > 1).sum()),
               **mmd, **auc,
               **{f"{k}__cn7_half": v for k, v in base_A.items()}, **{f"{k}__rg3_half": v for k, v in base_B.items()},
               "config": {**vars(args), "config": Path(args.config).name,
                          "public": Path(args.public).name if args.public else None}}  # 로컬 경로는 적지 않음

    default_name = ("transfer_risk_standardized" if args.standardized else "transfer_risk") + ("_jitter" if args.jitter else "")
    out = Path(args.out) if args.out else ROOT / cfg["output_dir"] / "analysis" / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    per_var.to_csv(f"{out}_per_variable.csv", index=False)
    pd.DataFrame([{k: v for k, v in summary.items() if k != "config"}]).to_csv(f"{out}_summary.csv", index=False)
    Path(f"{out}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 전이 위험 스크리닝 요약 (CN7 vs RG3, {src_name}, shot 단위) ===")
    print(f"도메인 분류기 AUC  LR {auc['domain_auc_lr']:.3f} / RF {auc['domain_auc_rf']:.3f}"
          f"   (무이동 기준선 LR: CN7반분 {base_A['baseline_domain_auc_lr']:.3f}, RG3반분 {base_B['baseline_domain_auc_lr']:.3f})")
    print(f"MMD²              {mmd['mmd2']:.4f}  순열 p={mmd['mmd2_perm_p']:.3f}  (순열 95% {mmd['mmd2_perm_95pct']:.4f})"
          f"   (기준선 {base_A['baseline_mmd2']:.4f} / {base_B['baseline_mmd2']:.4f})")
    print(f"변수별 W1(표준화)  최대 {summary['w1_std_max']:.2f}, 평균 {summary['w1_std_mean']:.2f}, "
          f"W1>1 인 변수 {summary['n_features_w1_gt_1']}/{len(feats)}   (기준선 최대 {base_A['baseline_w1_std_max']:.2f})")
    print("\n두 금형을 가장 갈라놓는 변수 (W1 표준화 상위 8):")
    print(per_var[["feature", "w1_std", "median_A", "median_B", "lr_coef_rank", "nunique_A", "nunique_B"]].head(8).to_string(index=False))
    print("\n고유값이 적은(이산적인) 변수 — 트리 모델이 금형 지문으로 쓸 수 있는 것:")
    print(nun.assign(min_unique=nun[["nunique_A", "nunique_B"]].min(axis=1)).sort_values("min_unique").head(8).to_string(index=False))
    print(f"\n저장: {out}_per_variable.csv / _summary.csv / _summary.json")


if __name__ == "__main__":
    main()
