"""1장(데이터 이해 및 진단, 15점)용 진단.

- 행/불량 수, LH·RH 짝 구조, 짝 안 라벨 충돌 → 불량 유형(공정 전체 이상 vs 캐비티 국소)
- 짝 구조 때문에 생기는 '행 단위 성능 상한' (공정값이 같으면 예측도 같을 수밖에 없음)
- 불량이 순서상 어디에 몰려 있는가 (연속 불량 구간 여부)
- 상수열·결측·이산화 정도·표준화 여부
- labeled vs unlabeled: 분포 차이(KS, adversarial AUC)와 '눈금' 일치 여부(파일별 표준화 진단)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import shot_table
from .evaluate import shot_types
from .plots import plt, savefig

TYPE_COLORS = {"process_wide": "#c0392b", "cavity_local": "#e08e0b", "single_row": "#7d3c98"}


def row_level_ceiling(shots: pd.DataFrame) -> dict:
    """같은 shot 두 행은 입력이 같아 예측도 같다 → 행 단위 F1 상한.
    불량 shot을 전부 '불량'으로 찍는 것이 최적이며, 충돌 짝의 양품 쪽은 반드시 FP가 된다."""
    P = int(shots["n_pos"].sum())
    fp = int((shots.loc[shots.y_shot == 1, "n_rows"] - shots.loc[shots.y_shot == 1, "n_pos"]).sum())
    return {"row_TP_max": P, "row_FP_unavoidable": fp,
            "row_precision_ceiling": P / (P + fp) if P + fp else np.nan,
            "row_f1_ceiling": 2 * P / (2 * P + fp) if P else np.nan}


def adversarial_auc(a: pd.DataFrame, b: pd.DataFrame, cols: list[str], seed: int) -> dict:
    """labeled(a)와 unlabeled(b)를 구분할 수 있는가? 0.5면 같은 분포, 1에 가까우면 다른 분포.
    - tree: 비선형. 파일마다 따로 표준화되어 '값의 눈금(격자)'이 다르면 그것만으로 1.0 근처가 나올 수 있음
    - linear: 평균·분산 수준의 차이만 봄 → tree만 높고 linear가 0.5 근처면 '눈금 차이' 가능성"""
    n = min(len(b), 5 * len(a))
    b = b.sample(n=n, random_state=seed)
    X = pd.concat([a[cols], b[cols]], ignore_index=True)
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    tree = HistGradientBoostingClassifier(max_depth=3, max_iter=150, random_state=seed)
    lin = LogisticRegression(max_iter=2000)
    return {k: round(float(roc_auc_score(y, cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1])), 3)
            for k, m in (("tree", tree), ("linear", lin))}


def _grid_step(v: np.ndarray) -> float:
    u = np.unique(np.round(v, 8))
    if len(u) < 3:
        return np.nan
    d = np.diff(u)
    return float(np.percentile(d, 10))


def scale_check(lab: pd.DataFrame, unl: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, dict]:
    """두 파일이 같은 기준으로 표준화됐는지 점검.
    - 두 파일 모두 평균≈0, 표준편차≈1 이면 '파일별로 따로 표준화'된 것
    - 같은 물리값이면 같은 z값이어야 하므로, labeled 값이 unlabeled에 '똑같이' 존재하는 비율(value_overlap)이
      낮으면 눈금이 다르다는 뜻 (연속값이 아니라 이산값이라 이 검사가 가능)"""
    rows = []
    for c in feats:
        a, b = lab[c].to_numpy(float), unl[c].to_numpy(float)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        ub = set(np.round(b, 6))
        rows.append({"feature": c, "lab_mean": a.mean(), "lab_std": a.std(), "unl_mean": b.mean(),
                     "unl_std": b.std(), "value_overlap": float(np.isin(np.round(a, 6), list(ub)).mean()),
                     "grid_step_ratio(unl/lab)": _grid_step(b) / _grid_step(a) if _grid_step(a) else np.nan})
    t = pd.DataFrame(rows)
    sep = bool(len(t) and t.unl_mean.abs().max() < 0.05 and (t.unl_std - 1).abs().max() < 0.05
               and t.lab_mean.abs().max() < 0.05)
    summ = {"both_files_mean0_std1": sep,
            "value_overlap_mean": float(t.value_overlap.mean()) if len(t) else np.nan,
            "verdict": ("파일별로 따로 표준화된 것으로 보임 → 같은 z값이 두 파일에서 다른 물리값. "
                        "labeled로 학습한 모델을 unlabeled에 그대로 적용하면 눈금이 어긋남")
            if sep and len(t) and t.value_overlap.mean() < 0.5 else
            ("눈금이 같을 가능성" if len(t) and t.value_overlap.mean() >= 0.5 else "판단 보류")}
    return t, summ


def defect_sequence(lab: pd.DataFrame, types: pd.Series) -> dict:
    """샷을 row_id 순서로 늘어놓았을 때 불량 샷이 어디에 있는가."""
    order = lab.groupby("shot_id")["row_id"].min().sort_values()
    idx = pd.Series(np.arange(len(order)), index=order.index)
    t = types.reindex(order.index)
    pos = idx[t.ne("normal")].to_numpy()
    if len(pos) == 0:
        return {"n_shots": len(order)}
    return {"n_shots": len(order), "first_defect_shot_idx": int(pos.min()), "last_defect_shot_idx": int(pos.max()),
            "defect_span_share": float((pos.max() - pos.min() + 1) / len(order)),
            "defects_in_first_10pct_shots": float((pos < 0.1 * len(order)).mean())}


def pair_position(lab: pd.DataFrame, types: pd.Series) -> dict:
    """캐비티 국소 불량 짝에서 불량 행이 짝의 '앞 행'(row_id가 작은 쪽)인 비율.
    0 또는 1에 가까우면 행 순서가 캐비티(LH/RH)를 나타낼 가능성 (원본 연결로 확인 필요)."""
    cl = types[types.eq("cavity_local")].index
    sub = lab[lab.shot_id.isin(cl)]
    firsts = []
    for _, g in sub.groupby("shot_id"):
        if len(g) != 2:
            continue
        g = g.sort_values("row_id")
        firsts.append(int(g.y.iloc[0] == 1))
    return {"cavity_local_pairs": len(firsts), "defect_is_first_row_share": float(np.mean(firsts)) if firsts else np.nan}


def run_audit(data: dict, feats: list[str], seed: int, fig_dir) -> tuple[dict, dict]:
    res, tables = {}, {}
    for product, lab in data["labeled"].items():
        unl = data["unlabeled"][product]
        shots = shot_table(lab)
        types = shot_types(lab)
        tc = types.value_counts()
        r = {
            "rows": len(lab), "positives": int(lab.y.sum()), "positive_rate": float(lab.y.mean()),
            "unlabeled_rows": len(unl), "shots": len(shots),
            "shot_size_hist": {int(k): int(v) for k, v in shots.n_rows.value_counts().sort_index().items()},
            "defect_shots": int(shots.y_shot.sum()),
            "process_wide_shots(양쪽 불량)": int(tc.get("process_wide", 0)),
            "cavity_local_shots(한쪽만 불량)": int(tc.get("cavity_local", 0)),
            "single_row_defects": int(tc.get("single_row", 0)),
            "missing_values": int(lab[feats].isna().sum().sum() + unl[feats].isna().sum().sum()),
            "constant_in_labeled": [c for c in feats if lab[c].nunique() <= 1],
            "constant_in_unlabeled": [c for c in feats if unl[c].nunique() <= 1],
            "looks_standardized": bool(abs(lab[[c for c in feats if lab[c].nunique() > 1]].mean()).max() < 1e-6
                                       and np.allclose(lab[[c for c in feats if lab[c].nunique() > 1]].std(ddof=0),
                                                       1, atol=1e-3)),
        }
        pairs = shots[shots.n_rows == 2]
        r["pair_adjacent_ratio"] = float(((pairs.last_row - pairs.first_row) == 1).mean()) if len(pairs) else np.nan
        r.update(row_level_ceiling(shots))
        r.update(defect_sequence(lab, types))
        r.update(pair_position(lab, types))

        unl_key = pd.util.hash_pandas_object(unl[feats].round(10), index=False)
        vc = unl_key.value_counts()
        r["unlabeled_shot_size_hist"] = {int(k): int(v) for k, v in vc.value_counts().sort_index().items()}
        lab_key = pd.util.hash_pandas_object(lab[feats].round(10), index=False)
        r["labeled_rows_found_in_unlabeled"] = int(lab_key.isin(set(unl_key)).sum())

        use = [c for c in feats if c not in r["constant_in_labeled"]]
        ks = pd.Series({c: ks_2samp(lab[c], unl[c]).statistic for c in use}).sort_values(ascending=False)
        r["ks_top5"] = {k: round(float(v), 3) for k, v in ks.head(5).items()}
        r["adversarial_auc_labeled_vs_unlabeled"] = adversarial_auc(lab, unl, use, seed)
        sc_table, sc_summ = scale_check(lab, unl, use)
        r["scale_check"] = sc_summ
        tables[f"{product}_scale_check"] = sc_table.round(4)

        lab_sorted = lab.sort_values("row_id").reset_index(drop=True)
        dec = pd.qcut(np.arange(len(lab_sorted)), 10, labels=False)
        r["positives_by_order_decile"] = lab_sorted.groupby(dec).y.sum().astype(int).tolist()

        tables[f"{product}_feature_profile"] = pd.DataFrame({
            "unique_labeled": lab[feats].nunique(), "unique_unlabeled": unl[feats].nunique(),
            "mean_labeled": lab[feats].mean(), "mean_unlabeled": unl[feats].mean(),
            "std_labeled": lab[feats].std(), "std_unlabeled": unl[feats].std(),
            "ks_stat": ks.reindex(feats),
            "defect_mean": lab.loc[lab.y == 1, feats].mean(), "normal_mean": lab.loc[lab.y == 0, feats].mean(),
        }).round(4)
        res[product] = r

        # 그림: 불량 샷 위치(순서) + 분포 차이
        order = lab.groupby("shot_id")["row_id"].min().sort_values()
        idx = pd.Series(np.arange(len(order)), index=order.index)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
        ax = axes[0]
        ax.axhline(0, color="#cfd6df", lw=6, solid_capstyle="butt", zorder=0)
        for t, col in TYPE_COLORS.items():
            sids = types[types.eq(t)].index
            if len(sids):
                ax.scatter(idx[sids], np.zeros(len(sids)), marker="|", s=900, color=col, lw=2.2,
                           label=f"{t} ({len(sids)})")
        ax.set(title=f"{product.upper()}: defect shots along production order", xlabel="shot index (row order)",
               yticks=[], xlim=(-5, len(order) + 5))
        ax.legend(fontsize=7, loc="upper right", markerscale=0.3)
        ks.head(10).iloc[::-1].plot.barh(ax=axes[1], color="#c8793a")
        axes[1].set(title="labeled vs unlabeled shift (KS)", xlabel="KS statistic")
        savefig(fig, fig_dir / f"audit_{product}.png")
    return res, tables
