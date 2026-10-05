"""E5 — 바닥–모델–천장. 조건은 docs/protocol_E5.md 에 실행 전 고정 (커밋 df930da).

- 바닥: 학습 없는 단순 규칙(세션 초반 순번, 직전 정지 시간) + E1 의 random·rh_first·spc
- 모델: freeze 된 최종 로지스틱 (대회 기간 = pipeline_logreg.csv OOF, 미래 = 대회 labeled 전체 학습)
- 천장: 라벨을 아는 shot 단위 최적 순서(oracle) — 같은 검사량에서 가능한 최대 포착률. 행 단위 정밀도 상한은 outputs/audit.json 인용.
- Q2: 날짜별 shot 라벨 순환 이동 블록 순열검정 (block CV shot AP, 미래 고정 점수 shot AP)

실행: python scripts/e5_floor_ceiling.py --public <원본 폴더 또는 zip> [--perms 200] [--skip-perm]
결과: outputs/e5/ (freeze 된 outputs 의 다른 폴더는 건드리지 않음)
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
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

import _public  # noqa: E402
import e1_policy_comparison as E1  # noqa: E402
import external_experiments as X  # noqa: E402
from src import features  # noqa: E402
from src.data import assign_shots, load_all  # noqa: E402
from src.models import make_model  # noqa: E402

warnings.filterwarnings("ignore")
G_MAIN = 5.0
G_SENS = (2.0, 10.0, 30.0)
METHODS = ["random", "rh_first", "session_start", "stop_gap", "spc", "logreg", "oracle"]
EXTERNAL = {"rh_first", "session_start", "stop_gap"}
LABEL_USING = {"oracle"}


# --------------------------------------------------------------------------- 단순 규칙
def rule_scores(ts: pd.Series, gap_min: float) -> pd.DataFrame:
    """행별 TimeStamp → 행별 (session_start, stop_gap) 점수. 라벨 미사용, shot = 같은 TimeStamp."""
    t = pd.to_datetime(ts)
    u = pd.Series(sorted(t.unique()))
    gap = u.diff().dt.total_seconds().div(60).fillna(gap_min + 1e-9)  # 기간 첫 shot = 세션 시작
    new = gap > gap_min
    sess = new.cumsum()
    pos = u.groupby(sess).cumcount() + 1
    per = pd.DataFrame({"t": u, "s_session_start": -pos.to_numpy(), "s_stop_gap": gap.to_numpy()})
    per.loc[0, "s_stop_gap"] = gap_min  # 프로토콜: 첫 shot 간격 = G
    m = pd.DataFrame({"t": t.to_numpy()}).merge(per, on="t", how="left")
    return m[["s_session_start", "s_stop_gap"]]


def add_oracle(parts: pd.DataFrame) -> pd.DataFrame:
    """천장: 불량 부품이 많은 shot 부터 (라벨 사용, 상한 계산 전용)."""
    parts = parts.copy()
    parts["s_oracle"] = parts.groupby("shot")["y"].transform("sum").astype(float)
    return parts


def evaluate(parts, meta, methods, fold_col=None):
    old = E1.METHODS
    E1.METHODS = methods
    try:
        return E1.evaluate(parts, meta, fold_col)
    finally:
        E1.METHODS = old


# --------------------------------------------------------------------------- 순열검정
def rotate_labels(shot_tab: pd.DataFrame, rng) -> pd.Series:
    """shot_tab: shot, day, t, pattern(tuple, 행 순서의 라벨). 날짜마다 무작위 오프셋 순환 이동 → shot→pattern."""
    out = {}
    tab = shot_tab.assign(size=shot_tab["pattern"].map(len))
    for _, g in tab.sort_values("t").groupby(["day", "size"], sort=False):  # 행 수가 같은 shot 끼리만
        pats = list(g["pattern"])
        n = len(pats)
        off = int(rng.integers(1, n)) if n > 1 else 0
        rolled = pats[-off:] + pats[:-off] if off else pats
        for s, p in zip(g["shot"], rolled):
            out[s] = p
    return pd.Series(out)


def apply_patterns(rows: pd.DataFrame, shot_col: str, mapping: pd.Series) -> np.ndarray:
    """shot→패턴(행 순서 라벨). rotate_labels 가 같은 행 수끼리만 옮기므로 길이가 항상 맞다."""
    y = np.zeros(len(rows), dtype=int)
    for s, idx in rows.groupby(shot_col, sort=False).indices.items():
        p = mapping[s]
        assert len(p) == len(idx)
        y[idx] = p
    return y


def shot_ap(y, s, shot):
    d = pd.DataFrame({"y": y, "s": s, "shot": shot}).groupby("shot").max()
    return average_precision_score(d.y, d.s) if d.y.sum() else np.nan


def patterns(rows: pd.DataFrame, shot_col: str, y_col: str, t_col: str) -> pd.DataFrame:
    g = rows.groupby(shot_col, sort=False)
    tab = pd.DataFrame({"shot": list(g.indices.keys())})
    tab["pattern"] = [tuple(rows[y_col].to_numpy()[i]) for i in g.indices.values()]
    tab["t"] = pd.to_datetime([rows[t_col].iat[i[0]] for i in g.indices.values()])
    tab["day"] = tab["t"].dt.date
    return tab


class _SafeModel:
    """순열 후 학습 fold 에 불량이 0개면 모델을 적합할 수 없다 → 그 fold 는 상수 점수(정보 없음)로 둔다.
    프로토콜 보완 2026-10-05 (docs/protocol_E5.md '변경 기록')."""
    n_degenerate = 0

    def __init__(self, make):
        self.m, self.const = make(), None

    def fit(self, X, y):
        if len(np.unique(y)) < 2:
            _SafeModel.n_degenerate += 1
            self.const = float(np.mean(y))
            return self
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.const is not None:
            return np.column_stack([np.full(len(X), 1 - self.const), np.full(len(X), self.const)])
        return self.m.predict_proba(X)


def perm_block_cv(product, L, n_perm, log):
    import harness as H
    df, feats = H.load(product)
    tmap = L.set_index("row_id")["TimeStamp"]
    df["TimeStamp"] = df["row_id"].map(tmap)
    assert df["TimeStamp"].notna().all()
    obs = H.run_cv(df, feats, H.mk, "block")["shot_ap"][0]
    tab = patterns(df, "shot_id", H.Y, "TimeStamp")
    rng = np.random.default_rng(0)
    null = []
    t0 = time.time()
    for i in range(n_perm):
        d = df.copy()
        d[H.Y] = apply_patterns(d, "shot_id", rotate_labels(tab, rng))
        assert d[H.Y].sum() == df[H.Y].sum()
        null.append(H.run_cv(d, feats, lambda: _SafeModel(H.mk), "block")["shot_ap"][0])
        if (i + 1) % 25 == 0:
            log(f"  [perm block_cv {product}] {i+1}/{n_perm} ({time.time()-t0:.0f}s)")
    log(f"  [perm block_cv {product}] 학습 fold 불량 0개로 상수 점수 처리: {_SafeModel.n_degenerate}회 / {n_perm*15} fold")
    _SafeModel.n_degenerate = 0
    return obs, np.array(null)


def perm_future(F, score, n_perm):
    obs = shot_ap(F["y"].to_numpy(), score, F["shot_id"].to_numpy())
    tab = patterns(F, "shot_id", "y", "TimeStamp")
    rng = np.random.default_rng(0)
    null = [shot_ap(apply_patterns(F, "shot_id", rotate_labels(tab, rng)), score, F["shot_id"].to_numpy())
            for _ in range(n_perm)]
    return obs, np.array(null)


# --------------------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=None)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--skip-perm", action="store_true")
    ap.add_argument("--products", default="cn7,rg3")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    src = _public.public_path(cfg, args.public)
    pub = _public.read_csv(src, "moldset_labeled.csv")
    ld = _public.read_csv(src, "labeled_data.csv")
    out = ROOT / cfg["output_dir"] / "e5"
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = ROOT / cfg["output_dir"] / "predictions"
    logs, t0 = [], time.time()

    def log(s):
        print(s, flush=True)
        logs.append(s)

    E1.MODELS = {"spc": "spc_maxz", "logreg": "logreg"}
    data = load_all(cfg, ROOT / cfg["data_dir"])
    raw = data["feature_cols"]
    base, _ = features.select_base_features(data, raw, True)
    rows, sens, perm_rows, null_rows = [], [], [], []
    for p in args.products.split(","):
        lab = assign_shots(data["labeled"][p], raw, p).reset_index(drop=True)
        L = X.link_side(lab, pub, raw)
        side = L.set_index("row_id")["side_rh"]
        ts = L.set_index("row_id")["TimeStamp"]
        rule_comp = {g: pd.concat([L[["row_id"]], rule_scores(L["TimeStamp"], g)], axis=1).set_index("row_id")
                     for g in (G_MAIN, *G_SENS)}
        # 대회 기간
        for cv_mode in ("block", "shot"):
            for rep, parts in E1.oof_parts(pred_dir, p, cv_mode, side).items():
                parts = add_oracle(parts)
                assert parts["shot"].map(parts.groupby("shot")["row_id"].apply(lambda r: ts[r].nunique())).eq(1).all()
                rc = rule_comp[G_MAIN].loc[parts["row_id"]].reset_index(drop=True)
                parts = pd.concat([parts, rc], axis=1)
                meta = dict(product=p, eval=f"{cv_mode}_cv", repeat=rep, N=len(parts), P=int(parts.y.sum()),
                            defect_shots=int(parts.groupby("shot").y.max().sum()), G=G_MAIN)
                rows += evaluate(parts, meta, METHODS)
                rows += evaluate(parts, meta, METHODS, fold_col="fold")
                if cv_mode == "block" and rep == 0:
                    for g in G_SENS:
                        pg = parts.drop(columns=["s_session_start", "s_stop_gap"])
                        pg = pd.concat([pg, rule_comp[g].loc[pg["row_id"]].reset_index(drop=True)], axis=1)
                        sens += evaluate(pg, {**meta, "G": g}, ["session_start", "stop_gap"])
        log(f"{p} 대회 기간 완료 ({time.time()-t0:.0f}s)")
        # 미래 기간
        mu, sd, _ = X.zscale_params(L, raw)
        for fix in (True, False):
            X.RPM_FIX = fix
            F = X.future_rows(ld, L, p, raw, mu, sd)
            if not len(F) or F.y.sum() == 0:
                continue
            parts = pd.DataFrame({"shot": F["shot_id"], "side_rh": F["side_rh"], "y": F["y"]})
            for tag, m in E1.MODELS.items():
                parts[f"s_{tag}"] = make_model(m, cfg["seed"]).fit(L[base], L["y"].to_numpy()).predict_proba(F[base])[:, 1]
            parts = add_oracle(parts)
            ev = "future" if fix else "future_norpmfix"
            meta = dict(product=p, eval=ev, repeat=0, N=len(parts), P=int(parts.y.sum()),
                        defect_shots=int(parts.groupby("shot").y.max().sum()), G=G_MAIN)
            parts = pd.concat([parts, rule_scores(F["TimeStamp"], G_MAIN)], axis=1)
            rows += evaluate(parts, meta, METHODS)
            if fix:
                for g in G_SENS:
                    pg = pd.concat([parts.drop(columns=["s_session_start", "s_stop_gap"]),
                                    rule_scores(F["TimeStamp"], g)], axis=1)
                    sens += evaluate(pg, {**meta, "G": g}, ["session_start", "stop_gap"])
                if not args.skip_perm:
                    obs, null = perm_future(F, parts["s_logreg"].to_numpy(), args.perms)
                    perm_rows.append(dict(product=p, eval="future", metric="shot_ap", observed=obs,
                                          null_mean=null.mean(), null_p95=np.quantile(null, .95),
                                          p=(1 + (null >= obs).sum()) / (len(null) + 1), n_perm=len(null)))
                    null_rows += [dict(product=p, eval="future", i=i, shot_ap=v) for i, v in enumerate(null)]
        X.RPM_FIX = True
        log(f"{p} 미래 기간 완료 ({time.time()-t0:.0f}s)")
        if not args.skip_perm:
            obs, null = perm_block_cv(p, L, args.perms, log)
            perm_rows.append(dict(product=p, eval="block_cv", metric="shot_ap(pooled, 5반복 평균)", observed=obs,
                                  null_mean=null.mean(), null_p95=np.quantile(null, .95),
                                  p=(1 + (null >= obs).sum()) / (len(null) + 1), n_perm=len(null)))
            null_rows += [dict(product=p, eval="block_cv", i=i, shot_ap=v) for i, v in enumerate(null)]
            log(f"{p} 순열검정 완료 ({time.time()-t0:.0f}s): {perm_rows[-1]}")

    R = finish(pd.DataFrame(rows))
    R.to_csv(out / "floor_ceiling_raw.csv", index=False, encoding="utf-8-sig")
    agg = summarize(R)
    audit = json.loads((ROOT / cfg["output_dir"] / "audit.json").read_text(encoding="utf-8"))
    for p in agg["product"].unique():
        c = audit.get(p, {})
        agg.loc[agg["product"].eq(p), "row_precision_ceiling"] = c.get("row_precision_ceiling", np.nan)
        agg.loc[agg["product"].eq(p), "row_f1_ceiling"] = c.get("row_f1_ceiling", np.nan)
    agg.to_csv(out / "floor_ceiling.csv", index=False, encoding="utf-8-sig")
    if sens:
        summarize(finish(pd.DataFrame(sens)), extra=["G"]).to_csv(out / "rule_sensitivity.csv", index=False, encoding="utf-8-sig")
    if perm_rows:
        pd.DataFrame(perm_rows).to_csv(out / "permutation_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(null_rows).to_csv(out / "permutation_null.csv", index=False, encoding="utf-8-sig")
    show = agg[agg.rank_mode.eq("pooled") & agg["eval"].isin(["block_cv", "future"]) & agg.budget.le(0.2)]
    log(show.pivot_table(index=["product", "eval", "method"], columns="budget", values="recall").round(3).to_string())
    if perm_rows:
        log(pd.DataFrame(perm_rows).round(4).to_string())
    log(f"총 {time.time()-t0:.0f}s")
    (out / "run_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    plot(agg, out)


def finish(R):
    R["recall"] = R["TP"] / R["P"]
    R["precision"] = np.where(R["inspected"] > 0, R["TP"] / R["inspected"].clip(lower=1), np.nan)
    R["inspect_rate"] = R["inspected"] / R["N"]
    R["uses_external"] = R["method"].isin(EXTERNAL) | R["eval"].str.startswith("future")
    R["uses_label"] = R["method"].isin(LABEL_USING)
    return R


def summarize(R, extra=()):
    keys = ["product", "eval", "rank_mode", *extra, "method", "budget"]
    return R.groupby(keys).agg(
        N=("N", "first"), P=("P", "first"), defect_shots=("defect_shots", "first"),
        uses_external=("uses_external", "first"), uses_label=("uses_label", "first"),
        inspect_rate=("inspect_rate", "mean"), TP=("TP", "mean"), FP=("FP", "mean"), FN=("FN", "mean"),
        recall=("recall", "mean"), recall_sd=("recall", "std"), recall_min=("recall", "min"),
        recall_max=("recall", "max"), precision=("precision", "mean"),
    ).round(4).reset_index()


# --------------------------------------------------------------------------- 그림
def plot(agg: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if "NotoSansCJK" in f or "NanumGothic" in f or "AppleGothic" in f or "AppleSDGothic" in f:
            try:
                font_manager.fontManager.addfont(f)
            except Exception:
                pass
    names = {ff.name for ff in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [n for n in ("Noto Sans CJK KR", "NanumGothic", "AppleGothic", "Apple SD Gothic Neo")
                                   if n in names][:1] or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    floor = ["random", "rh_first", "session_start", "stop_gap"]
    label = {"random": "무작위", "rh_first": "RH 우선", "session_start": "세션 초반 순번",
             "stop_gap": "직전 정지 시간", "spc": "관리도", "logreg": "최종 모델(로지스틱)",
             "oracle": "천장(라벨을 아는 최적 순서)"}
    panels = [("block_cv", "pooled", "대회 기간 · block CV"), ("future", "pooled", "대회 이후 기간 [외부 라벨, 탐색적]")]
    for p in agg["product"].unique():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        for ax, (ev, rm, title) in zip(axes, panels):
            d = agg[agg["product"].eq(p) & agg["eval"].eq(ev) & agg["rank_mode"].eq(rm)]
            if d.empty:
                ax.set_visible(False)
                continue
            piv = d.pivot_table(index="budget", columns="method", values="recall").sort_index()
            xs = piv.index.to_numpy() * 100
            lo, hi = piv[floor].min(axis=1), piv[floor].max(axis=1)
            ax.fill_between(xs, lo, hi, color="#d9d8d2", alpha=0.7, lw=0, label="바닥(학습 없는 규칙 범위)")
            for m, st in [("session_start", dict(color="#8a8984", lw=1.2, ls=(0, (3, 2)))),
                          ("stop_gap", dict(color="#8a8984", lw=1.2, ls=(0, (1, 2)))),
                          ("spc", dict(color="#9a9892", lw=1.4)),
                          ("logreg", dict(color="#2a78d6", lw=2.6)),
                          ("oracle", dict(color="#1f1e1c", lw=1.2, ls=(0, (6, 3))))]:
                ax.plot(xs, piv[m].to_numpy(), marker="o", ms=3.5, label=label[m], **st)
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}%" for x in xs])
            ax.set_ylim(0, 1.02)
            ax.grid(True, color="#e6e5e0", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            n, P, S = int(d.N.iat[0]), int(d.P.iat[0]), int(d.defect_shots.iat[0])
            ax.set_title(f"{title}\n부품 {n:,} · 불량 {P}개 · 불량 shot {S}개", fontsize=10, loc="left")
            ax.set_xlabel("검사한 부품 비율 (예산)")
        axes[0].set_ylabel("불량 부품 포착률")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
        fig.suptitle(f"{p.upper()} · 바닥–모델–천장 (같은 검사량, seed 50개 평균)", x=0.01, ha="left", fontsize=12)
        fig.tight_layout(rect=(0, 0.12, 1, 0.95))
        fig.savefig(out / f"floor_ceiling_{p}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__" and "--plot-only" in sys.argv:
    _cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    _o = ROOT / _cfg["output_dir"] / "e5"
    plot(pd.read_csv(_o / "floor_ceiling.csv"), _o)
elif __name__ == "__main__":
    main()
