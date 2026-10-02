"""검사 우선순위 (4장·5장): 불확실성 → 3구역 규칙 → 순위표 → Budget Curve → Ablation.

질문: 같은 검사량에서, 모델 점수만 쓸 때(S0)보다 추가 점수를 더하면 불량 부품을 더 많이 잡는가?

조건 (결과를 보기 전에 고정):
- 점수: run_all.py 의 block 교차검증 OOF (outputs/predictions/pipeline_<모델>.csv, cv_mode == "block").
  제품별 모델은 --models 로 지정 (기본 cn7=logreg, rg3=lgbm).
- 판단 단위는 shot (한 shot 을 고르면 두 부품을 모두 검사). 검사량 = 검사한 부품 수 / 전체 부품 수,
  포착률 = 찾은 불량 부품 수 / 전체 불량 부품 수. shot 점수 = 그 shot 행 점수의 최댓값.
- 불확실성(Uncertainty) = 같은 행에 대한 반복 5회 OOF 예측의 표준편차 (라벨을 쓰지 않음).
- 결합 방법: 각 점수를 제품 안 백분위로 바꿔 같은 가중치로 평균 (가중치를 튜닝하지 않음).
    S0 모델 점수만 / S1 +불확실성 / S2 +Novelty / S3 +Ambiguity / S4 전부  (파일이 없는 항은 건너뜀)
- 3구역: 점수 상위 2 % = High(반드시 검사), 2~10 % = Review(사람 판단·여유 있을 때 검사), 나머지 = Low.
- 동점은 고정 seed 0~19 의 무작위 순서로 끊고 평균한다. 무작위 검사 기준선의 포착률 = 검사량.
- 수치는 반복 5회 각각 계산한 뒤 평균±표준편차.

실행: python scripts/inspection_priority.py [--ambiguity <csv>] [--novelty <csv>] [--models cn7=logreg rg3=lgbm]
      추가 점수 csv 는 점수 열(이름에 ambig / novel 포함)과 제품 열(mold 또는 product)이 있어야 한다. 두 형식을 받는다:
        (a) shot 단위·반복별: shot_id, repeat, (fold) — 그 반복의 학습 fold 로만 계산한 값. fold 열이 있으면 예측 파일과 대조한다.
        (b) 행 단위·고정: row_id — 모든 반복에 같은 값을 쓴다.
결과 (outputs/inspection/): uncertainty.csv, ranking.csv, zone_summary.csv, budget_curve.csv, ablation.csv,
      budget_curve_<제품>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
ZONES = {"High": 0.02, "Review": 0.10}  # 누적 상위 비율
N_TIE_SEEDS = 20
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def pct_rank(s: pd.Series) -> pd.Series:
    """제품 안 백분위 (0~1, 클수록 위험). 동점은 평균 순위."""
    return s.rank(method="average", pct=True)


def load_scores(pred_dir: Path, product: str, model: str) -> pd.DataFrame:
    p = pred_dir / f"pipeline_{model}.csv"
    assert p.exists(), f"{p} 가 없음 → run_all.py 를 먼저 실행"
    d = pd.read_csv(p)
    d = d[d["product"].eq(product) & d["cv_mode"].eq("block")]
    assert len(d), f"{p} 에 {product} block 예측이 없음"
    assert d.groupby("row_id")["repeat"].nunique().eq(d["repeat"].nunique()).all(), "행마다 반복 수가 다름"
    return d[["row_id", "shot_id", "repeat", "fold", "y_true", "y_score"]].copy()


def load_extra(path: str | None, key: str, product: str, d: pd.DataFrame) -> pd.DataFrame | None:
    """추가 점수 csv → shot × 반복 표 (없으면 None). d = 그 제품의 예측(행·shot·반복·fold)."""
    if not path:
        return None
    e = pd.read_csv(path)
    col = next((c for c in e.columns if key in c.lower()), None)
    assert col, f"{path}: 이름에 '{key}' 가 들어간 점수 열이 필요"
    for c in ("mold", "product"):
        if c in e.columns:
            e = e[e[c].astype(str).str.lower().eq(product)]
            break
    reps = sorted(d["repeat"].unique())
    shots = d["shot_id"].drop_duplicates()
    if {"shot_id", "repeat"} <= set(e.columns) and e["shot_id"].notna().all():  # (a) shot 단위·반복별
        assert not e.duplicated(["shot_id", "repeat"]).any(), f"{path}: (shot_id, repeat) 중복"
        if "fold" in e.columns:  # 같은 분할로 계산했는지 대조
            ref = d.drop_duplicates(["shot_id", "repeat"]).set_index(["shot_id", "repeat"])["fold"]
            got = e.set_index(["shot_id", "repeat"])["fold"]
            assert got.reindex(ref.index).eq(ref).all(), f"{path}: fold 배정이 예측 파일과 다름 (다른 분할로 계산된 점수)"
        W = e.pivot(index="shot_id", columns="repeat", values=col).reindex(index=shots, columns=reps)
    else:  # (b) 행 단위·고정
        assert "row_id" in e.columns and e["row_id"].notna().all(), f"{path}: shot_id+repeat 또는 row_id 가 필요"
        r = d.drop_duplicates("row_id").set_index("row_id")["shot_id"]
        v = e.groupby("row_id")[col].mean().reindex(r.index).groupby(r).max().reindex(shots)
        W = pd.DataFrame({k: v for k in reps})
    assert W.notna().all().all(), f"{path}: {product} 에 점수가 없는 shot 이 있음"
    return W


def shot_table(d: pd.DataFrame, extras: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """반복별 shot 표: 점수(P), 불확실성(U), 추가 점수, 부품 수, 불량 부품 수."""
    wide = d.pivot(index="row_id", columns="repeat", values="y_score")
    row = d.drop_duplicates("row_id").set_index("row_id")[["shot_id", "y_true"]]
    row["U"] = wide.std(axis=1, ddof=0)
    row["P_mean"] = wide.mean(axis=1)
    g = row.groupby("shot_id", sort=False)
    S = pd.DataFrame({"parts": g.size(), "defect_parts": g["y_true"].sum(), "U": g["U"].max(), "P_mean": g["P_mean"].max()})
    for k, W in extras.items():  # 대표값(순위표용) = 반복 평균, 반복별 값은 따로 보관
        S[k] = W.mean(axis=1).reindex(S.index)
        for r in W.columns:
            S[f"{k}_r{r}"] = W[r].reindex(S.index)
    for r in wide.columns:
        S[f"P_r{r}"] = wide[r].groupby(row["shot_id"]).max()
    return S


def curve(score: np.ndarray, parts: np.ndarray, defects: np.ndarray, seed: int):
    """점수 내림차순(동점은 seed 무작위)으로 shot 을 검사할 때의 누적 검사량·포착률."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(score)), -score))
    x = np.cumsum(parts[order]) / parts.sum()
    y = np.cumsum(defects[order]) / defects.sum()
    return x, y


def at_budgets(score, parts, defects):
    """검사 예산별 포착률·실제 검사량(동점 seed 평균)과 곡선 면적."""
    n = len(score)
    rec, ins, auc = [], [], []
    for seed in range(N_TIE_SEEDS):
        x, y = curve(score, parts, defects, seed)
        k = [max(int(np.ceil(b * n)), 1) - 1 for b in BUDGETS]
        rec.append(y[k]); ins.append(x[k])
        auc.append(np.trapezoid(np.r_[0, y], np.r_[0, x]) if hasattr(np, "trapezoid") else np.trapz(np.r_[0, y], np.r_[0, x]))
    return np.mean(rec, 0), np.mean(ins, 0), float(np.mean(auc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["cn7=logreg", "rg3=lgbm"])
    ap.add_argument("--ambiguity", default=None)
    ap.add_argument("--novelty", default=None)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8")) if (ROOT / "config.json").exists() else {}
    out_root = ROOT / cfg.get("output_dir", "outputs")
    out = out_root / "inspection"
    out.mkdir(parents=True, exist_ok=True)
    models = dict(m.split("=") for m in args.models)

    unc_rows, rank_rows, zone_rows, abl_rows, curve_rows = [], [], [], [], []
    for product, model in models.items():
        d = load_scores(out_root / "predictions", product, model)
        extras = {k: v for k, v in {"A": load_extra(args.ambiguity, "ambig", product, d),
                                    "N": load_extra(args.novelty, "novel", product, d)}.items() if v is not None}
        S = shot_table(d, extras)
        reps = sorted(d["repeat"].unique())
        parts, defects = S["parts"].to_numpy(), S["defect_parts"].to_numpy()
        # 방법 정의: 백분위 평균 (튜닝 없음)
        methods = {"S0 모델 점수만": ["P"], "S1 +불확실성": ["P", "U"]}
        if "N" in extras:
            methods["S2 +Novelty"] = ["P", "N"]
        if "A" in extras:
            methods["S3 +Ambiguity"] = ["P", "A"]
        if len(extras) >= 1:
            methods["S4 전부"] = ["P", "U"] + sorted(extras)
        def combined(r, terms):
            comp = {"P": pct_rank(S[f"P_r{r}"]), "U": pct_rank(S["U"]), **{k: pct_rank(S[f"{k}_r{r}"]) for k in extras}}
            return np.mean([comp[t] for t in terms], axis=0)

        # ---- Ablation + Budget Curve (반복별) ----
        for name, terms in methods.items():
            R, I, A = [], [], []
            for r in reps:
                rec, ins, auc = at_budgets(combined(r, terms), parts, defects)
                R.append(rec); I.append(ins); A.append(auc)
                if name.startswith(("S0", "S1")):
                    x, y = curve(combined(r, terms), parts, defects, 0)
                    grid = np.linspace(0, 1, 201)
                    curve_rows += [dict(product=product, model=model, method=name, repeat=r, inspect_rate=g, recall=v)
                                   for g, v in zip(grid, np.interp(grid, np.r_[0, x], np.r_[0, y]))]
            R, I = np.array(R), np.array(I)
            for j, b in enumerate(BUDGETS):
                abl_rows.append(dict(product=product, model=model, method=name, terms="+".join(terms), budget_shots=b,
                                     inspect_rate_parts=round(I[:, j].mean(), 4), recall_parts=round(R[:, j].mean(), 4),
                                     recall_sd=round(R[:, j].std(ddof=1), 4), random_recall=round(I[:, j].mean(), 4),
                                     missed_defect_parts=round(float(defects.sum() * (1 - R[:, j].mean())), 2),
                                     defect_parts=int(defects.sum()), parts=int(parts.sum()), shots=len(S),
                                     curve_area=round(float(np.mean(A)), 4), curve_area_sd=round(float(np.std(A, ddof=1)), 4)))

        # ---- 3구역 (S0 기준, 반복별 → 평균) ----
        zr = []
        for r in reps:
            rk = S[f"P_r{r}"].rank(method="first", ascending=False) / len(S)
            zone = np.where(rk <= ZONES["High"], "High", np.where(rk <= ZONES["Review"], "Review", "Low"))
            for z in ("High", "Review", "Low"):
                m = zone == z
                zr.append(dict(zone=z, shots=m.sum(), parts=parts[m].sum(), defect_parts=defects[m].sum()))
        Z = pd.DataFrame(zr).groupby("zone", sort=False).mean()
        for z, v in Z.iterrows():
            zone_rows.append(dict(product=product, model=model, zone=z, shots=round(v.shots, 1), parts=round(v.parts, 1),
                                  parts_share=round(v.parts / parts.sum(), 4), defect_parts=round(v.defect_parts, 2),
                                  defect_share=round(v.defect_parts / defects.sum(), 4),
                                  defect_rate_in_zone=round(v.defect_parts / v.parts, 4),
                                  lift_vs_overall=round((v.defect_parts / v.parts) / (defects.sum() / parts.sum()), 2)))

        # ---- 순위표 (대표 점수 = 반복 5회 평균) ----
        T = S.assign(product=product, model=model).reset_index()
        T["rank"] = T["P_mean"].rank(method="first", ascending=False).astype(int)
        T["top_pct"] = (T["rank"] / len(T)).round(4)
        T["zone"] = np.where(T["top_pct"] <= ZONES["High"], "High", np.where(T["top_pct"] <= ZONES["Review"], "Review", "Low"))
        T["uncertainty_pct"] = pct_rank(T["U"]).round(4)
        T = T.rename(columns={"P_mean": "score", "U": "uncertainty", "A": "ambiguity", "N": "novelty"})
        keep = ["product", "model", "shot_id", "rank", "top_pct", "zone", "score", "uncertainty", "uncertainty_pct"] + \
               [c for c in ("ambiguity", "novelty") if c in T.columns] + ["parts", "defect_parts"]
        rank_rows.append(T.sort_values("rank")[keep])
        u = d.drop_duplicates("row_id")[["row_id", "shot_id"]].assign(product=product, model=model)
        u["uncertainty"] = u["row_id"].map(d.pivot(index="row_id", columns="repeat", values="y_score").std(axis=1, ddof=0))
        unc_rows.append(u)
        print(f"{product} ({model}) 완료: shot {len(S)}, 불량 부품 {int(defects.sum())}", flush=True)

    enc = dict(index=False, encoding="utf-8-sig")
    pd.concat(unc_rows).to_csv(out / "uncertainty.csv", **enc)
    pd.concat(rank_rows).round(6).to_csv(out / "ranking.csv", **enc)
    ZS = pd.DataFrame(zone_rows); ZS.to_csv(out / "zone_summary.csv", **enc)
    AB = pd.DataFrame(abl_rows); AB.to_csv(out / "ablation.csv", **enc)
    C = pd.DataFrame(curve_rows)
    Cm = C.groupby(["product", "model", "method", "inspect_rate"])["recall"].agg(["mean", "min", "max"]).reset_index()
    Cm.round(4).to_csv(out / "budget_curve.csv", **enc)
    plot(Cm, AB, out)

    pd.set_option("display.width", 250)
    print("\n## Ablation: 검사 예산별 불량 부품 포착률 (반복 5회 평균)")
    print(AB.pivot_table(index=["product", "method"], columns="budget_shots", values="recall_parts", sort=False).round(3).to_string())
    print("\n## 곡선 면적 (클수록 좋음, 무작위 = 0.5)")
    print(AB.groupby(["product", "method"], sort=False)[["curve_area", "curve_area_sd"]].first().to_string())
    print("\n## 3구역 (S0 기준)")
    print(ZS.to_string(index=False))


def plot(Cm: pd.DataFrame, AB: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f2328", "#6b7280", "#e5e7eb", "#fcfcfb"
    SER = [("S0 모델 점수만", "#2a78d6", "o"), ("S1 +불확실성", "#eb6834", "s")]
    for product, g in Cm.groupby("product", sort=False):
        fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=BG)
        ax.set_facecolor(BG)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls=(0, (4, 3)), label="무작위 검사")
        for name, col, mk in SER:
            h = g[g["method"] == name]
            if h.empty:
                continue
            ax.fill_between(h["inspect_rate"], h["min"], h["max"], color=col, alpha=0.12, lw=0)
            ax.plot(h["inspect_rate"], h["mean"], color=col, lw=2, label=name)
            a = AB[(AB["product"] == product) & (AB["method"] == name)]
            ax.plot(a["inspect_rate_parts"], a["recall_parts"], ls="", marker=mk, ms=7, color=col,
                    markeredgecolor=BG, markeredgewidth=1.5)
            if name.startswith("S0"):
                for _, r in a[a["budget_shots"].isin([0.02, 0.10])].iterrows():
                    ax.annotate(f"검사 {r.inspect_rate_parts:.0%} → {r.recall_parts:.0%}", (r.inspect_rate_parts, r.recall_parts),
                                xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK)
        model = g["model"].iloc[0]
        ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.02)
        ax.set_xlabel("검사량 (검사한 부품 / 전체 부품)", color=INK); ax.set_ylabel("불량 부품 포착률", color=INK)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
        ax.set_title(f"{product.upper()} 검사 예산 곡선 ({model}, block 교차검증, 띠 = 반복 5회 최소~최대)", fontsize=10.5, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.8); ax.tick_params(colors=MUTED, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out / f"budget_curve_{product}.png", dpi=160, facecolor=BG)
        plt.close(fig)


if __name__ == "__main__":
    main()
