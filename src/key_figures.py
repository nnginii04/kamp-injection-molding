"""보고서·발표용 핵심 그림 6개 → outputs/figures/key<번호>_<이름>.png

라벨은 영문 (한글 폰트가 없는 PC 에서도 깨지지 않게). 설명은 report.md 에 한국어로.
색은 역할별로 고정: CV 방식(block 파랑 / shot 주황), 불량 유형(양쪽 불량 보라 / 한쪽 불량 청록),
ablation(OFF 보라 / ON 노랑). 팔레트는 색각이상 구분 검증을 통과한 조합이며,
대비가 낮은 색(청록·노랑)은 값 라벨·범례를 함께 둔다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import recall_at_budget
from .plots import plt, savefig

C = {"block": "#2a78d6", "shot": "#eb6834", "folds": ["#2a78d6", "#eb6834", "#1baf7a"],
     "process_wide": "#4a3aa7", "cavity_local": "#1baf7a", "off": "#4a3aa7", "on": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8984", "base": "#d9d8d3", "surface": "#fcfcfb"}
TYPE_LABEL = {"process_wide": "both cavities defective (process-wide)",
              "cavity_local": "one cavity defective (cavity-local)"}
MODE_LABEL = {"shot": "shot CV (optimistic)", "block": "block CV (pessimistic)"}


def _style(ax):
    ax.set_facecolor(C["surface"])
    ax.tick_params(colors=C["ink2"], labelsize=8)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C["base"])
    ax.title.set_color(C["ink"])


def _shot_index(lab: pd.DataFrame) -> pd.Series:
    order = lab.groupby("shot_id").row_id.min().sort_values()
    return pd.Series(np.arange(len(order)), index=order.index)


def fig_defect_timeline(frames, types_by_product, path):
    ps = list(frames)
    fig, axes = plt.subplots(len(ps), 1, figsize=(10, 1.55 * len(ps) + 0.9), squeeze=False)
    for ax, p in zip(axes[:, 0], ps):
        lab = frames[p][0]
        idx, types = _shot_index(lab), types_by_product[p]
        ax.axhline(0, color=C["base"], lw=6, solid_capstyle="butt", zorder=0)
        for t in ("process_wide", "cavity_local"):
            sids = [s for s in types[types.eq(t)].index if s in idx.index]
            if sids:
                ax.scatter(idx[sids], np.zeros(len(sids)), marker="|", s=420, lw=2.2, color=C[t],
                           label=f"{TYPE_LABEL[t]} ({len(sids)})", zorder=2)
        pos = idx[[s for s in types[types.ne("normal")].index if s in idx.index]]
        span = f" (between shot {pos.min()} and {pos.max()})" if len(pos) else ""
        if len(pos) and (pos.max() - pos.min()) < 0.15 * len(idx):  # 좁은 구간에 몰리면 확대 그림
            lo, hi = max(pos.min() - 8, 0), pos.max() + 8
            ins = ax.inset_axes([0.30, 0.34, 0.36, 0.50])
            ins.set_facecolor(C["surface"])
            ins.axhline(0, color=C["base"], lw=5, solid_capstyle="butt", zorder=0)
            for t in ("process_wide", "cavity_local"):
                sids = [s for s in types[types.eq(t)].index if s in idx.index]
                if sids:
                    ins.scatter(idx[sids], np.zeros(len(sids)), marker="|", s=260, lw=2.2, color=C[t], zorder=2)
            ins.set(xlim=(lo, hi), ylim=(-1, 1), yticks=[])
            ins.tick_params(labelsize=7, colors=C["ink2"])
            ins.set_title(f"zoom: shots {lo}–{hi}", fontsize=7, color=C["ink2"], loc="left")
            ins.grid(False)
            ax.indicate_inset_zoom(ins, edgecolor=C["muted"])
        ax.set(yticks=[], xlim=(-5, len(idx) + 5), ylim=(-1, 1))
        ax.set_title(f"{p.upper()}: {int(types.ne('normal').sum())} defect shots among {len(idx)} shots{span}",
                     fontsize=9, loc="left")
        ax.grid(False)
        _style(ax)
        ax.legend(fontsize=7, loc="upper right", frameon=False, markerscale=0.35)
    axes[-1, 0].set_xlabel("shot index (production order)", fontsize=8, color=C["ink2"])
    fig.suptitle("Where defects occur along production order", fontsize=11, x=0.01, ha="left", color=C["ink"])
    savefig(fig, path)


def fig_cv_concept(fold_id: pd.DataFrame, lab: pd.DataFrame, types: pd.Series, path, n_show=120, product="cn7"):
    f = fold_id[fold_id.mold == product]
    idx = _shot_index(lab)
    shots = f.groupby("shot_id").first()
    shots["i"] = idx.reindex(shots.index)
    shots = shots[shots.i < n_show].sort_values("i")
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    rows = [("block", 0.0), ("shot", 1.2)]
    for mode, y0 in rows:
        col = f"fold_{mode}_r0"
        for _, r in shots.iterrows():
            ax.add_patch(plt.Rectangle((r.i + 0.08, y0), 0.84, 0.8, color=C["folds"][int(r[col]) % 3], lw=0))
        ax.text(-2, y0 + 0.4, MODE_LABEL[mode], ha="right", va="center", fontsize=8, color=C["ink"])
    for t, mk in (("process_wide", "v"), ("cavity_local", "v")):
        sids = [s for s in types[types.eq(t)].index if s in shots.index]
        if sids:
            ax.scatter(shots.loc[sids, "i"] + 0.5, np.full(len(sids), 2.3), marker=mk, s=38, color=C[t],
                       label=TYPE_LABEL[t], zorder=3)
    for k in range(3):
        ax.scatter([], [], marker="s", s=60, color=C["folds"][k], label=f"validation fold {k}")
    ax.set(xlim=(-1, n_show + 1), ylim=(-0.3, 2.7), yticks=[])
    ax.set_xlabel(f"{product.upper()} shot index (first {n_show} shots, repeat 0)", fontsize=8, color=C["ink2"])
    ax.grid(False)
    _style(ax)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=5, frameon=False)
    ax.set_title("Shot CV keeps LH/RH pairs together; block CV also keeps 10 neighbouring shots together",
                 fontsize=10, loc="left")
    savefig(fig, path)


def fig_recall_range(oof_all, choices, path, budgets=np.linspace(0.01, 0.20, 39)):
    ps = list(choices)
    fig, axes = plt.subplots(1, len(ps), figsize=(5.2 * len(ps), 3.9), squeeze=False)
    for ax, p in zip(axes[0], ps):
        c = choices[p]
        curves = {}
        for mode in ("shot", "block"):
            o = oof_all[(oof_all.dataset == c["dataset"]) & (oof_all.model == c["model"])
                        & (oof_all["product"] == p) & (oof_all.cv_mode == mode)]
            if o.empty:
                continue
            curves[mode] = np.array([np.mean([recall_at_budget(g.y, g.p, b) for _, g in o.groupby("repeat")])
                                     for b in budgets])
        x = budgets * 100
        if len(curves) == 2:
            ax.fill_between(x, curves["block"], curves["shot"], color=C["base"], alpha=0.6, lw=0,
                            label="performance range")
        for mode, yv in curves.items():
            ax.plot(x, yv, color=C[mode], lw=2, label=MODE_LABEL[mode])
            for b in (0.02, 0.05, 0.10):
                j = int(np.argmin(np.abs(budgets - b)))
                ax.plot(x[j], yv[j], "o", ms=5, color=C[mode], mec=C["surface"], mew=1.5, zorder=3)
            ax.annotate(f"{yv[-1]:.2f}", (x[-1], yv[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=C["ink2"], va="center")
        ax.plot(x, budgets, "--", color=C["muted"], lw=1.2, label="random inspection")
        ax.set(xlim=(0, 22), ylim=(0, 1.02))
        ax.set_xlabel("inspected share of production (%)", fontsize=8, color=C["ink2"])
        ax.set_ylabel("defects caught (row recall)", fontsize=8, color=C["ink2"])
        ax.set_title(f"{p.upper()} — {c['model']} (dots at 2/5/10%)", fontsize=9, loc="left")
        _style(ax)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("Recall by inspection budget: block CV (pessimistic) ~ shot CV (optimistic)", fontsize=11,
                 x=0.01, ha="left", color=C["ink"])
    savefig(fig, path)


def fig_defect_type_detection(dt: pd.DataFrame, product, model, path):
    cols = [c for c in dt.columns if c.startswith("TP rows top")]
    modes = [m for m in ("block", "shot") if m in set(dt.cv_mode)]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.0 * len(modes), 3.6), squeeze=False, sharey=True)
    w = 0.36
    for ax, mode in zip(axes[0], modes):
        d = dt[dt.cv_mode == mode].set_index("defect_type")
        xs = np.arange(len(cols))
        for j, t in enumerate(("process_wide", "cavity_local")):
            if t not in d.index:
                continue
            share = d.loc[t, cols].to_numpy(float) / d.loc[t, "defect_rows"]
            bars = ax.bar(xs + (j - 0.5) * (w + 0.02), share, w, color=C[t], label=f"{TYPE_LABEL[t]} "
                          f"({int(d.loc[t, 'defect_rows'])} rows)")
            for b, v in zip(bars, share):
                ax.annotate(f"{v:.0%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=7, color=C["ink2"])
        ax.set_xticks(xs, [c.replace("TP rows ", "") for c in cols], fontsize=8)
        ax.set(ylim=(0, 1.1))
        ax.set_title(MODE_LABEL[mode], fontsize=9, loc="left")
        _style(ax)
    axes[0, 0].set_ylabel("share of defect rows caught", fontsize=8, color=C["ink2"])
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=8, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{product.upper()} ({model}): share of defect rows caught, by defect type and inspection budget",
                 fontsize=10, x=0.01, ha="left", color=C["ink"])
    savefig(fig, path)


def fig_ablation(ab: pd.DataFrame, path):
    metrics = [("row_roc_auc", "row ROC-AUC"), ("row_ap", "row PR-AUC")]
    d = ab.sort_values(["product", "cv_mode", "model"]).reset_index(drop=True)
    labels = [f"{r['product'].upper()} {r.cv_mode} {r.model}" for _, r in d.iterrows()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * len(d) + 1.6), sharey=True)
    ys = np.arange(len(d))[::-1]
    for ax, (m, name) in zip(axes, metrics):
        off, on = d[f"{m}[OFF]"].to_numpy(), d[f"{m}[ON 누수]"].to_numpy()
        for y, a, b in zip(ys, off, on):
            ax.plot([a, b], [y, y], color=C["base"], lw=2, zorder=1)
        ax.scatter(off, ys, s=46, color=C["off"], label="pair-position OFF (used)", zorder=2,
                   edgecolor=C["surface"], linewidth=1.5)
        ax.scatter(on, ys, s=46, color=C["on"], label="pair-position ON (leak, comparison only)", zorder=3,
                   edgecolor=C["surface"], linewidth=1.5)
        for y, b in zip(ys, on):
            ax.annotate(f"{b:.2f}", (b, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7,
                        color=C["ink2"])
        ax.set_title(name, fontsize=9, loc="left")
        _style(ax)
    axes[0].set_yticks(ys, labels, fontsize=8)
    axes[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False)
    fig.suptitle("Leak check: within-pair row order as a feature (ON = leak, comparison only)", fontsize=11,
                 x=0.01, ha="left", color=C["ink"])
    savefig(fig, path)


def fig_holdout(trials: pd.DataFrame, totals: dict, path):
    short = {d: d.split(".")[0] for d in trials.design.unique()}
    ms = list(trials.mold.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(5.2 * len(ms), 3.7), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, p in zip(axes[0], ms):
        t = trials[trials.mold == p]
        designs = list(dict.fromkeys(t.design))
        ticks = []
        for i, dsg in enumerate(designs):
            v = t[t.design == dsg].holdout_defect_rows.to_numpy()
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=12, color=C["block"], alpha=0.35, lw=0)
            ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=C["ink"], lw=2)
            ticks.append(f"{short[dsg]}\nP0 {np.mean(v == 0):.0%}")
        ax.axhline(totals.get(p, np.nan), color=C["muted"], ls="--", lw=1)
        ax.annotate(f"all defects = {totals.get(p)}", (len(designs) - 0.5, totals.get(p, 0)), xytext=(0, 3),
                    textcoords="offset points", ha="right", fontsize=7, color=C["ink2"])
        ax.set_xticks(range(len(designs)), ticks, fontsize=7)
        ax.set_ylabel("defect rows in 25% holdout", fontsize=8, color=C["ink2"])
        ax.set_title(f"{p.upper()} (dots = seeds, bar = median)", fontsize=9, loc="left")
        _style(ax)
    fig.suptitle("Why no holdout: A time-last, A' sliding window, B block-stratified, C block-random, "
                 "D shot-stratified\n(P0 = share of trials whose holdout contains no defect)", fontsize=9.5,
                 x=0.01, ha="left", color=C["ink"])
    savefig(fig, path)


def make_all(fig_dir, frames, types_by_product, fold_id, oof_all, choices, exp, ablation) -> list[str]:
    """만든 그림 파일 이름 목록을 돌려준다 (자료가 없으면 해당 그림은 건너뜀)."""
    made = []

    def go(name, fn, *a):
        fn(*a, fig_dir / name)
        made.append(name)

    go("key1_defect_timeline.png", fig_defect_timeline, frames, types_by_product)
    if fold_id is not None and "cn7" in frames:
        go("key2_cv_concept.png", fig_cv_concept, fold_id, frames["cn7"][0], types_by_product["cn7"])
    go("key3_recall_range.png", fig_recall_range, oof_all, choices)
    dt = exp.get("defect_type_detection_cn7")
    if dt is not None and len(dt) and "cn7" in choices:
        go("key4_defect_type_detection.png", fig_defect_type_detection, dt, "cn7", choices["cn7"]["model"])
    if ablation is not None and len(ablation):
        go("key5_ablation_leak.png", fig_ablation, ablation)
    tr = exp.get("holdout_trials")
    if tr is not None and len(tr):
        totals = {p: int(frames[p][0].y.sum()) for p in frames}
        go("key6_holdout_simulation.png", fig_holdout, tr, totals)
    return made
