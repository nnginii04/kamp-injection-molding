"""3장(영향요인 및 오류분석, 15점).

1) 불량 유형별 탐지율: 공정 전체 이상(양쪽 캐비티 불량) vs 캐비티 국소(한쪽만 불량)
   → '어떤 조건에서 작동하고 어떤 조건에서 실패하는가'의 핵심 답
2) 샷 CV vs 블록 CV 타임라인: 연속 불량 구간에서 이웃 샷에 기대는 정도
3) 영향 변수: 검증 fold 기준 permutation importance (AP 감소량)
4) FP 분해: 짝 때문에 구분 불가능한 FP(twin) vs 모델 FP
5) FN/FP가 몰리는 공정조건(구간별 집계)과 상위 2개 변수 상호작용
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .evaluate import TYPE_KO, type_recall
from .plots import plt, savefig

TYPE_COLORS = {"process_wide": "#c0392b", "cavity_local": "#e08e0b", "single_row": "#7d3c98"}


def importance(fitted: dict, mode: str, model_name: str, lab: pd.DataFrame, feats: list[str], seed: int) -> pd.DataFrame:
    rows = []
    for (m_mode, name, k), (m, va) in fitted.items():
        if name != model_name or m_mode != mode:
            continue
        Xv, yv = lab[feats].iloc[va], lab["y"].to_numpy()[va]
        if yv.sum() == 0:
            continue
        try:
            r = permutation_importance(m, Xv, yv, scoring="average_precision", n_repeats=5,
                                       random_state=seed, n_jobs=1)
        except Exception as e:  # 중요도 계산 실패가 전체 실행을 멈추지 않도록
            print(f"  (중요도 계산 건너뜀: {name} fold {k}: {e})")
            continue
        rows.append(pd.Series(r.importances_mean, index=feats, name=k))
    if not rows:
        return pd.DataFrame(columns=["mean", "std"])
    t = pd.concat(rows, axis=1)
    return pd.DataFrame({"mean": t.mean(axis=1), "std": t.std(axis=1)}).sort_values("mean", ascending=False)


def type_table(oof_all: pd.DataFrame, types_by_product: dict, budgets) -> pd.DataFrame:
    """모든 (cv_mode, dataset, model)에 대해 제품별 불량 유형 탐지율 (반복 평균)."""
    rows = []
    for (mode, dataset, model), g in oof_all.groupby(["cv_mode", "dataset", "model"]):
        for product, types in types_by_product.items():
            sub = g[g["product"] == product]
            if sub.empty:
                continue
            recs = [type_recall(r, types, budgets) for _, r in sub.groupby("repeat")]
            avg = pd.DataFrame(recs).mean().to_dict()
            rows.append({"cv_mode": mode, "dataset": dataset, "product": product, "model": model, **avg})
    return pd.DataFrame(rows)


def _timeline(product, dataset, model_name, lab, oof_all, types, cfg, fig_dir):
    """샷 순서별 위험 순위(백분위). 위쪽일수록 먼저 검사. 샷 CV와 블록 CV를 위아래로 비교."""
    order = lab[lab["product"] == product].groupby("shot_id")["row_id"].min().sort_values()
    idx = pd.Series(np.arange(len(order)), index=order.index)
    modes = [m for m in cfg["cv"]["modes"] if m in oof_all.cv_mode.unique()]
    fig, axes = plt.subplots(len(modes), 1, figsize=(11, 2.6 * len(modes)), squeeze=False, sharex=True)
    rt = cfg["risk_tiers"]
    for ax, mode in zip(axes[:, 0], modes):
        o = oof_all[(oof_all.dataset == dataset) & (oof_all.model == model_name)
                    & (oof_all["product"] == product) & (oof_all.cv_mode == mode)]
        s = o.groupby("shot_id").p.mean()
        pct = s.rank(pct=True) * 100
        t = types.reindex(pct.index).fillna("normal")
        x = idx.reindex(pct.index)
        nm = t.eq("normal")
        ax.scatter(x[nm], pct[nm], s=5, color="#b9c2cc", label="normal shot")
        for ty, col in TYPE_COLORS.items():
            m = t.eq(ty)
            if m.any():
                ax.scatter(x[m], pct[m], s=46, color=col, edgecolors="black", linewidths=0.4,
                           label=f"{ty} ({int(m.sum())})", zorder=3)
        for q, lab_txt in ((1 - k, f"top {k:.0%}") for k in sorted(rt)):
            ax.axhline(q * 100, color="#555", lw=0.8, ls="--")
            ax.text(len(order) * 1.005, q * 100, lab_txt, fontsize=7, va="center")
        ax.set(ylabel="risk rank (pct)", ylim=(-2, 102),
               title=f"{product.upper()} {model_name} — {mode} CV: where each shot ranks (higher = inspect first)")
        ax.legend(fontsize=7, loc="lower right", ncol=4)
    axes[-1, 0].set_xlabel("shot index (production order)")
    savefig(fig, fig_dir / f"timeline_{product}.png")


def analyze(product, dataset, model_name, cv_mode, lab, feats, oof_all, fitted, types, cfg, fig_dir):
    ea = cfg["error_analysis"]
    o = oof_all[(oof_all.dataset == dataset) & (oof_all.model == model_name) & (oof_all.repeat == 0)
                & (oof_all.cv_mode == cv_mode)]
    o = o[o["product"] == product]
    df = lab.reset_index(drop=True).iloc[o.row.to_numpy()].copy()
    df["p"], df["pred"] = o.p.to_numpy(), o.pred.to_numpy()
    df["defect_type"] = df.shot_id.map(types).fillna("normal")
    df["outcome"] = np.select([(df.y == 1) & (df.pred == 1), (df.y == 0) & (df.pred == 1),
                               (df.y == 1) & (df.pred == 0)], ["TP", "FP", "FN"], "TN")
    twin_pos = df.groupby("shot_id").y.transform("max").astype(bool)
    df["fp_type"] = np.where(df.outcome.eq("FP") & twin_pos, "twin_FP(구분 불가)",
                             np.where(df.outcome.eq("FP"), "model_FP", ""))

    # 불량 유형별 샷 판정 (반복 0)
    shot = df.groupby("shot_id").agg(type=("defect_type", "first"), pred=("pred", "max"))
    by_type = (shot[shot.type.ne("normal")].groupby("type").pred.agg(["size", "sum"])
               .rename(columns={"size": "defect_shots", "sum": "detected"}))
    by_type["detect_rate"] = by_type.detected / by_type.defect_shots
    by_type.index = [f"{i} = {TYPE_KO.get(i, i)}" for i in by_type.index]

    imp = importance(fitted, cv_mode, model_name, lab.reset_index(drop=True), feats, cfg["seed"])
    top = [c for c in imp.index if c not in ("is_rg3",)][: ea["top_features"]]
    if not top:
        top = [c for c in feats if c not in ("is_rg3",)][: ea["top_features"]]

    bins = []
    for c in top:
        try:
            b = pd.qcut(df[c].rank(method="first"), ea["n_bins"], labels=False)
        except ValueError:
            continue
        g = df.groupby(b)
        t = pd.DataFrame({
            "feature": c, "bin": range(g.ngroups),
            "range": g[c].agg(lambda s: f"[{s.min():.2f}, {s.max():.2f}]").values,
            "rows": g.size().values, "defects": g.y.sum().values,
            "TP": g.outcome.agg(lambda s: (s == "TP").sum()).values,
            "FN": g.outcome.agg(lambda s: (s == "FN").sum()).values,
            "FP": g.outcome.agg(lambda s: (s == "FP").sum()).values,
        })
        t["defect_rate"] = t.defects / t.rows
        t["FP_rate"] = t.FP / (t.rows - t.defects).clip(lower=1)
        t["FN_share"] = t.FN / max(int(df.outcome.eq("FN").sum()), 1)
        bins.append(t)
    bin_table = pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()

    normal_med = df.loc[df.y == 0, top].median()
    fn_cases = df.loc[df.outcome.eq("FN"), ["row_id", "shot_id", "defect_type", "p", *top]].copy()
    for c in top:
        fn_cases[c] = (fn_cases[c] - normal_med[c]).round(2)

    # 그림: 중요도 + 상위 2개 변수 위 오류 위치(불량 유형 표시)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    it = imp.head(10).iloc[::-1]
    if len(it):
        axes[0].barh(it.index, it["mean"], xerr=it["std"], color="#3b6fb6")
    axes[0].set(title=f"{product.upper()} permutation importance ({model_name}, {cv_mode} CV)", xlabel="AP drop")
    if len(top) >= 2:
        a, b = top[0], top[1]
        rng = np.random.default_rng(0)
        jit = lambda n: rng.normal(0, 0.03, n)  # noqa: E731
        colors = {"TN": "#cfd6df", "FP": "#e0a030", "TP": "#2a9d55", "FN": "#d1453b"}
        for oc in ["TN", "FP", "TP", "FN"]:
            s = df[df.outcome == oc]
            axes[1].scatter(s[a] + jit(len(s)), s[b] + jit(len(s)), s=10 if oc == "TN" else 34,
                            c=colors[oc], alpha=0.6 if oc == "TN" else 0.95, edgecolors="none",
                            label=f"{oc} ({len(s)})")
        pw = df[df.defect_type.eq("process_wide")]
        if len(pw):
            axes[1].scatter(pw[a], pw[b], s=110, facecolors="none", edgecolors="black", marker="s",
                            linewidths=1.0, label="process-wide defect")
        axes[1].set(title="errors on top-2 features", xlabel=a, ylabel=b)
        axes[1].legend(fontsize=7)
    savefig(fig, fig_dir / f"errors_{product}.png")
    _timeline(product, dataset, model_name, lab, oof_all, types, cfg, fig_dir)

    inter = pd.DataFrame()
    if len(top) >= 2:
        a, b = top[0], top[1]
        qa = pd.qcut(df[a].rank(method="first"), 3, labels=["low", "mid", "high"])
        qb = pd.qcut(df[b].rank(method="first"), 3, labels=["low", "mid", "high"])
        inter = df.groupby([qa, qb], observed=False).agg(rows=("y", "size"), defects=("y", "sum"),
                                                         FN=("outcome", lambda s: (s == "FN").sum()),
                                                         FP=("outcome", lambda s: (s == "FP").sum()))
        inter["defect_rate"] = (inter.defects / inter.rows).round(4)
        inter.index.names = [a, b]
        inter = inter.reset_index()

    summary = {
        "cv_mode": cv_mode,
        "outcomes": df.outcome.value_counts().to_dict(),
        "fp_breakdown": df.loc[df.outcome.eq("FP"), "fp_type"].value_counts().to_dict(),
        "top_features": top,
    }
    return summary, imp, bin_table, fn_cases, inter, by_type
