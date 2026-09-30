"""E1 — 같은 검사량(부품 수)에서 검사 방법 비교. 조건은 docs/protocol_E1_E3.md 에 실행 전 고정.

질문: 동일한 부품 검사 예산에서, 결합 모델(logreg_spc)이 단순 검사 방법보다 불량 부품을 더 많이 포착하는가?

- 점수는 고정된 것만 사용: 대회 기간 = run_all.py 교차검증 OOF (outputs/predictions/pipeline_*.csv),
  미래 기간 = 대회 labeled 전체로 학습한 같은 모델 (scripts/external_experiments.py 의 연결·표준화 함수 재사용).
- side(LH/RH)·미래 라벨은 공개 원본(외부 데이터)에서 온다 → 결과 표에 uses_external 로 표시.
- 동점은 고정 seed 0~49 무작위 순서로 끊는다 (라벨·행 순서 미사용).

실행: python scripts/e1_policy_comparison.py --public <원본 폴더 또는 zip>
결과: outputs/evaluation/policy_comparison.csv (요약), policy_comparison_raw.csv, policy_comparison_<제품>.png
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _public  # noqa: E402
import external_experiments as X  # noqa: E402
from src import features  # noqa: E402
from src.data import assign_shots, load_all  # noqa: E402
from src.models import make_model  # noqa: E402

warnings.filterwarnings("ignore")
BUDGETS = [0.02, 0.05, 0.10, 0.20, 0.50]
SEEDS = range(50)
MODELS = {"spc": "spc_maxz", "logreg": "logreg", "logreg_spc": "logreg_spc"}
METHODS = ["random", "rh_first", "spc", "logreg", "logreg_spc", "logreg_spc_rh"]
EXTERNAL_METHODS = {"rh_first", "logreg_spc_rh"}


# --------------------------------------------------------------------------- 검사 순서
def units(parts: pd.DataFrame, method: str, rng: np.random.Generator) -> list[np.ndarray]:
    """검사 단위(한 번에 검사하는 부품 묶음)의 순서. parts: shot, side_rh, s_<model> (위치 인덱스 0..n-1)."""
    n = len(parts)
    if method == "random":
        return [np.array([i]) for i in rng.permutation(n)]
    if method == "rh_first":
        rh = np.flatnonzero(parts["side_rh"].to_numpy() == 1)
        lh = np.flatnonzero(parts["side_rh"].to_numpy() == 0)
        return [np.array([i]) for i in np.r_[rng.permutation(rh), rng.permutation(lh)]]
    col = "s_" + ("logreg_spc" if method == "logreg_spc_rh" else method)
    g = parts.groupby("shot", sort=False)
    shot_score = g[col].max()
    shot_rows = g.indices
    tie = rng.random(len(shot_score))
    order = np.lexsort((tie, -shot_score.to_numpy()))  # 점수 내림차순, 동점은 무작위
    out = []
    side = parts["side_rh"].to_numpy()
    for j in order:
        rows = np.asarray(shot_rows[shot_score.index[j]])
        if method == "logreg_spc_rh":  # 순서 정의 (a): 같은 shot 안에서 RH → LH, 부품 단위 비용
            rows = rows[np.argsort(-side[rows], kind="stable")]
            out += [np.array([r]) for r in rows]
        else:
            out.append(rows)
    return out


def inspect(y: np.ndarray, seq: list[np.ndarray], k: int) -> dict:
    used, tp = 0, 0
    for u in seq:
        if used + len(u) > k:
            break
        used += len(u)
        tp += int(y[u].sum())
    P = int(y.sum())
    return {"k": k, "inspected": used, "unused": k - used, "TP": tp, "FP": used - tp, "FN": P - tp}


def need_for(y: np.ndarray, seq: list[np.ndarray], frac: float = 0.8) -> float:
    target, used, tp = int(np.ceil(frac * y.sum())), 0, 0
    if target == 0:
        return np.nan
    for u in seq:
        used += len(u)
        tp += int(y[u].sum())
        if tp >= target:
            return used / len(y)
    return np.nan


def evaluate(parts: pd.DataFrame, meta: dict, fold_col: str | None = None) -> list[dict]:
    """parts 한 묶음(한 반복 또는 미래 기간)을 모든 방법·예산·seed 로 평가. fold_col 이 있으면 fold 안 순위(ii)."""
    y = parts["y"].to_numpy()
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            if fold_col is None:
                seq = units(parts, method, rng)
                need80 = need_for(y, seq)
                for b in BUDGETS:
                    r = inspect(y, seq, int(np.floor(b * len(parts))))
                    rows.append({**meta, "rank_mode": "pooled", "method": method, "budget": b, "seed": seed,
                                 "need80": need80, **r})
            else:
                per_fold = []
                for _, fp in parts.groupby(fold_col, sort=True):
                    fp = fp.reset_index(drop=True)
                    per_fold.append((fp["y"].to_numpy(), units(fp, method, rng), len(fp)))
                for b in BUDGETS:
                    acc = {"k": 0, "inspected": 0, "unused": 0, "TP": 0, "FP": 0, "FN": 0}
                    for yf, seq, nf in per_fold:
                        r = inspect(yf, seq, int(np.floor(b * nf)))
                        for key in acc:
                            acc[key] += r[key]
                    rows.append({**meta, "rank_mode": "within_fold", "method": method, "budget": b, "seed": seed,
                                 "need80": np.nan, **acc})
    return rows


# --------------------------------------------------------------------------- 데이터
def oof_parts(pred_dir: Path, product: str, cv_mode: str, side: pd.Series) -> dict[int, pd.DataFrame]:
    out = {}
    frames = []
    for tag, m in MODELS.items():
        d = pd.read_csv(pred_dir / f"pipeline_{m}.csv")
        d = d[d["product"].eq(product) & d["cv_mode"].eq(cv_mode)]
        frames.append(d[["repeat", "fold", "row_id", "shot_id", "y_true", "y_score"]].rename(
            columns={"y_score": f"s_{tag}"}))
    m = frames[0]
    for f in frames[1:]:
        m = m.merge(f, on=["repeat", "fold", "row_id", "shot_id", "y_true"], how="inner", validate="1:1")
    m["side_rh"] = m["row_id"].map(side).astype(int)
    m = m.rename(columns={"y_true": "y", "shot_id": "shot"})
    for rep, g in m.groupby("repeat"):
        out[int(rep)] = g.sort_values("row_id").reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=None)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    src = _public.public_path(cfg, args.public)
    pub = _public.read_csv(src, "moldset_labeled.csv")
    ld = _public.read_csv(src, "labeled_data.csv")
    out = ROOT / cfg["output_dir"] / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = ROOT / cfg["output_dir"] / "predictions"

    data = load_all(cfg, ROOT / cfg["data_dir"])
    raw = data["feature_cols"]
    base, _ = features.select_base_features(data, raw, True)
    rows = []
    for p in data["labeled"]:
        lab = assign_shots(data["labeled"][p], raw, p).reset_index(drop=True)
        L = X.link_side(lab, pub, raw)
        side = L.set_index("row_id")["side_rh"]
        # 대회 기간 OOF
        for cv_mode in ("block", "shot"):
            for rep, parts in oof_parts(pred_dir, p, cv_mode, side).items():
                meta = dict(product=p, eval=f"{cv_mode}_cv", repeat=rep, N=len(parts), P=int(parts.y.sum()),
                            defect_shots=int(parts.groupby("shot").y.max().sum()))
                rows += evaluate(parts, meta)
                rows += evaluate(parts, meta, fold_col="fold")
        # 미래 기간 (대회 labeled 전체로 학습한 고정 점수)
        mu, sd, _ = X.zscale_params(L, raw)
        for fix in (True, False):
            X.RPM_FIX = fix
            F = X.future_rows(ld, L, p, raw, mu, sd)
            if not len(F) or F.y.sum() == 0:
                continue
            parts = pd.DataFrame({"shot": F["shot_id"], "side_rh": F["side_rh"], "y": F["y"]})
            for tag, m in MODELS.items():
                parts[f"s_{tag}"] = make_model(m, cfg["seed"]).fit(L[base], L["y"].to_numpy()).predict_proba(F[base])[:, 1]
            meta = dict(product=p, eval="future" if fix else "future_norpmfix", repeat=0, N=len(parts),
                        P=int(parts.y.sum()), defect_shots=int(parts.groupby("shot").y.max().sum()))
            rows += evaluate(parts, meta)
        X.RPM_FIX = True
        print(f"{p} 완료", flush=True)

    R = pd.DataFrame(rows)
    R["recall"] = R["TP"] / R["P"]
    R["precision"] = np.where(R["inspected"] > 0, R["TP"] / R["inspected"].clip(lower=1), np.nan)
    R["inspect_rate"] = R["inspected"] / R["N"]
    R["uses_external"] = R["method"].isin(EXTERNAL_METHODS) | R["eval"].str.startswith("future")
    R.to_csv(out / "policy_comparison_raw.csv", index=False, encoding="utf-8-sig")
    keys = ["product", "eval", "rank_mode", "method", "budget"]
    agg = R.groupby(keys).agg(
        N=("N", "first"), P=("P", "first"), defect_shots=("defect_shots", "first"),
        uses_external=("uses_external", "first"),
        inspect_rate=("inspect_rate", "mean"), unused=("unused", "mean"),
        TP=("TP", "mean"), FP=("FP", "mean"), FN=("FN", "mean"),
        recall=("recall", "mean"), recall_sd=("recall", "std"), recall_min=("recall", "min"),
        recall_max=("recall", "max"), precision=("precision", "mean"), need80=("need80", "mean"),
    ).round(4).reset_index()
    agg.to_csv(out / "policy_comparison.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    show = agg[agg.rank_mode.eq("pooled") & agg["eval"].isin(["block_cv", "future"])]
    print(show.pivot_table(index=["product", "eval", "method"], columns="budget", values="recall").round(3).to_string())
    print(show.groupby(["product", "eval", "method"])["need80"].first().round(3).to_string())
    plot(agg, out)




# --------------------------------------------------------------------------- 그림 (검사량–포착률)
def plot(agg: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if "NotoSansCJK" in f or "NanumGothic" in f or "AppleGothic" in f:
            font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = [n for n in ("Noto Sans CJK KR", "NanumGothic", "AppleGothic", "DejaVu Sans")
                                   if any(n == ff.name for ff in font_manager.fontManager.ttflist)][:1] or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    style = {  # 강조 = 결합 모델(파랑), 외부 메타정보 = 주황, 나머지 회색 계열
        "logreg_spc": dict(color="#2a78d6", lw=2.2, ls="-", label="logreg_spc (결합)"),
        "logreg_spc_rh": dict(color="#2a78d6", lw=1.4, ls=(0, (4, 2)), label="logreg_spc + RH 우선 [외부]"),
        "logreg": dict(color="#52514e", lw=1.6, ls="-", label="로지스틱"),
        "spc": dict(color="#9a9892", lw=1.6, ls="-", label="관리도"),
        "rh_first": dict(color="#eb6834", lw=1.6, ls="-", label="RH 우선 [외부]"),
        "random": dict(color="#c3c2b7", lw=1.4, ls=(0, (2, 2)), label="무작위"),
    }
    panels = [("block_cv", "pooled", "대회 기간 · block CV\n(전 fold 합산 순위)"),
              ("block_cv", "within_fold", "대회 기간 · block CV\n(fold 안 순위)"),
              ("future", "pooled", "대회 이후 기간 [외부 라벨]\n(대회 전체로 학습, 고정 점수)")]
    xs = [b * 100 for b in BUDGETS]
    for p in agg["product"].unique():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
        for ax, (ev, rm, title) in zip(axes, panels):
            d = agg[agg["product"].eq(p) & agg["eval"].eq(ev) & agg["rank_mode"].eq(rm)]
            if d.empty:
                ax.set_visible(False)
                continue
            n, P, S = int(d.N.iat[0]), int(d.P.iat[0]), int(d.defect_shots.iat[0])
            for m in ["random", "rh_first", "spc", "logreg", "logreg_spc_rh", "logreg_spc"]:
                g = d[d.method.eq(m)].sort_values("budget")
                ax.plot(xs, g.recall.to_numpy(), marker="o", ms=4, **style[m])
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}%" for x in xs])
            ax.set_ylim(0, 1.02)
            ax.grid(True, color="#e6e5e0", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_title(f"{title}\n부품 {n:,} · 불량 {P}개 · 불량 shot {S}개", fontsize=10, loc="left")
            ax.set_xlabel("검사한 부품 비율 (예산)")
        axes[0].set_ylabel("불량 부품 포착률")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9)
        fig.suptitle(f"{p.upper()} · 같은 검사량에서 방법별 불량 포착률 (seed 50개 평균, 동점 무작위)", x=0.01, ha="left", fontsize=12)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        fig.savefig(out / f"policy_comparison_{p}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__" and "--plot-only" in sys.argv:
    _cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    _out = ROOT / _cfg["output_dir"] / "evaluation"
    plot(pd.read_csv(_out / "policy_comparison.csv"), _out)


if __name__ == "__main__" and "--plot-only" not in sys.argv:
    main()
