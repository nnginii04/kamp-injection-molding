"""E3 — 모델 선택과 분할에 대한 점검 (중첩 교차검증). 조건은 docs/protocol_E1_E3.md (변경 기록 포함).

- 방법 셋만: logreg, spc_maxz, logreg_spc.
- 바깥: 연속 block(크기 5·10·20 shot) 단위 3-fold × 5 반복 (StratifiedGroupKFold, 기존 분할 규칙과 같은 seed 체계).
- 안쪽: 바깥 학습 부분에서 같은 block 크기로 3-fold × 3 반복 → 세 방법의 fold 안 shot AP 평균(불량 있는 fold 만)으로
  선택, 최고와 1 표준오차 이내면 단순한 쪽 (spc_maxz < logreg < logreg_spc). 오경보 임계값도 안쪽 OOF 상위 10 %.
- 바깥 라벨은 마지막 채점에만 쓴다. 평가 fold 에 불량이 없으면 AP 는 '평가 불가'(평균에서 제외, 행은 남김),
  학습 쪽에 불량이 없으면 '학습 불가'.
- block 분할은 앞뒤 구간이 학습에 들어가므로 '미관측 구간 검증'이다 (미래 예측 검증이 아님).

실행: python scripts/e3_robustness.py
결과: outputs/model/robustness_summary.csv (block 크기 × 방법 요약), robustness_folds.csv (바깥 fold 별),
      robustness_predictions.csv (바깥 fold 예측)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import features  # noqa: E402
from src.cv import group_splits, make_groups  # noqa: E402
from src.data import assign_shots, load_all  # noqa: E402
from src.evaluate import expected_topk  # noqa: E402
from src.models import make_model  # noqa: E402

warnings.filterwarnings("ignore")
METHODS = ["spc_maxz", "logreg", "logreg_spc"]  # 단순한 순서
BLOCKS = [5, 10, 20]
OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS, INNER_REPEATS = 3, 5, 3, 3


def shot_ap(y, s, shots):
    d = pd.DataFrame({"y": y, "s": s, "shot": shots}).groupby("shot").max()
    return float(average_precision_score(d.y, d.s)) if d.y.sum() else np.nan


def recall_at(y, s, b=0.10):
    y = np.asarray(y, float)
    return expected_topk(y, np.asarray(s, float), int(np.ceil(len(y) * b))) / y.sum() if y.sum() else np.nan


def fit_score(method, Xtr, ytr, Xte, seed):
    return make_model(method, seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def inner_select(lab, feats, groups, seed):
    """안쪽 CV: 방법별 fold 안 shot AP 평균과 표준오차, 1-SE 선택, 방법별 상위 10 % 임계값(안쪽 OOF)."""
    y = lab["y"].to_numpy()
    X = lab[feats]
    fold_ap = {m: [] for m in METHODS}
    oof = {m: [] for m in METHODS}
    for r in range(INNER_REPEATS):
        try:
            splits = group_splits(y, groups, INNER_SPLITS, seed + 1000 + r)
        except ValueError:
            continue
        for tr, va in splits:
            if y[tr].sum() == 0:
                continue
            for m in METHODS:
                s = fit_score(m, X.iloc[tr], y[tr], X.iloc[va], seed)
                oof[m].append(s)
                if y[va].sum():
                    fold_ap[m].append(shot_ap(y[va], s, lab["shot_id"].to_numpy()[va]))
    stats = {m: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan, len(v))
             for m, v in fold_ap.items() if v}
    if not stats:
        return None, {}, {}
    best = max(stats, key=lambda m: stats[m][0])
    bar = stats[best][0] - (stats[best][1] if np.isfinite(stats[best][1]) else 0)
    chosen = next(m for m in METHODS if m in stats and stats[m][0] >= bar)
    thr = {m: float(np.quantile(np.concatenate(v), 0.90)) for m, v in oof.items() if v}
    return chosen, stats, thr


def main():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = cfg["seed"]
    out = ROOT / cfg["output_dir"] / "model"
    out.mkdir(parents=True, exist_ok=True)
    data = load_all(cfg, ROOT / cfg["data_dir"])
    raw = data["feature_cols"]
    base, _ = features.select_base_features(data, raw, True)
    rows, preds = [], []
    for p in data["labeled"]:
        lab = assign_shots(data["labeled"][p], raw, p).reset_index(drop=True)
        y = lab["y"].to_numpy()
        for B in BLOCKS:
            groups = make_groups(lab, "block", B)
            for rep in range(OUTER_REPEATS):
                try:
                    outer = group_splits(y, groups, OUTER_SPLITS, seed + rep)
                except ValueError as e:
                    rows.append(dict(product=p, block=B, repeat=rep, fold=-1, status=f"분할 불가: {e}"))
                    continue
                for k, (tr, te) in enumerate(outer):
                    base_row = dict(product=p, block=B, repeat=rep, fold=k, n_test=len(te),
                                    test_defect_parts=int(y[te].sum()), train_defect_parts=int(y[tr].sum()),
                                    test_defect_shots=int(lab.iloc[te].groupby("shot_id").y.max().sum()))
                    if y[tr].sum() == 0:
                        rows.append({**base_row, "status": "학습 불가"})
                        continue
                    sub = lab.iloc[tr].reset_index(drop=True)
                    chosen, stats, thr = inner_select(sub, base, groups[tr], seed + rep * 10 + k)
                    for m in METHODS:
                        s = fit_score(m, lab[base].iloc[tr], y[tr], lab[base].iloc[te], seed)
                        yt = y[te]
                        neg = (yt == 0).sum()
                        fa = float(((s >= thr[m]) & (yt == 0)).sum() / neg) if m in thr and neg else np.nan
                        evaluable = yt.sum() > 0
                        rows.append({**base_row, "method": m, "chosen": m == chosen, "selected_method": chosen,
                                     "inner_ap": stats.get(m, (np.nan,))[0],
                                     "status": "평가" if evaluable else "평가 불가(불량 없음)",
                                     "row_ap": float(average_precision_score(yt, s)) if evaluable else np.nan,
                                     "shot_ap": shot_ap(yt, s, lab["shot_id"].to_numpy()[te]) if evaluable else np.nan,
                                     "recall@10%": recall_at(yt, s) if evaluable else np.nan,
                                     "false_alarm_rate": fa, "inspect_rate_at_thr": float((s >= thr[m]).mean()) if m in thr else np.nan})
                        preds.append(pd.DataFrame({"product": p, "block": B, "repeat": rep, "fold": k, "method": m,
                                                   "row_id": lab["row_id"].to_numpy()[te], "shot_id": lab["shot_id"].to_numpy()[te],
                                                   "y_true": yt, "y_score": s}))
            print(f"{p} block {B} 완료", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "robustness_folds.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds).to_csv(out / "robustness_predictions.csv", index=False, encoding="utf-8-sig")
    ev = R[R["status"].eq("평가")]
    summ = ev.groupby(["product", "block", "method"]).agg(
        folds_evaluated=("row_ap", "size"), row_ap=("row_ap", "mean"), row_ap_sd=("row_ap", "std"),
        shot_ap=("shot_ap", "mean"), recall_at_10=("recall@10%", "mean"),
        false_alarm_rate=("false_alarm_rate", "mean"), chosen_rate=("chosen", "mean")).round(3).reset_index()
    cnt = (R.drop_duplicates(["product", "block", "repeat", "fold"]).groupby(["product", "block"])["status"]
           .value_counts().unstack(fill_value=0))
    cnt.to_csv(out / "robustness_fold_status.csv", encoding="utf-8-sig")
    fa_nodef = R[R["status"].eq("평가 불가(불량 없음)")].groupby(["product", "block", "method"])["false_alarm_rate"].mean().round(3)
    summ = summ.merge(fa_nodef.rename("false_alarm_rate_no_defect_folds").reset_index(), how="left",
                      on=["product", "block", "method"])
    # 짝지은 비교: 같은 바깥 fold 에서 logreg_spc − logreg
    pv = ev.pivot_table(index=["product", "block", "repeat", "fold"], columns="method", values="row_ap")
    pv["diff_spc_vs_logreg"] = pv["logreg_spc"] - pv["logreg"]
    paired = pv.groupby(["product", "block"])["diff_spc_vs_logreg"].agg(
        mean="mean", wins=lambda d: int((d > 0).sum()), n="size").round(3)
    summ = summ.merge(paired.reset_index(), how="left", on=["product", "block"])
    summ.to_csv(out / "robustness_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(cnt.to_string())


if __name__ == "__main__":
    main()
