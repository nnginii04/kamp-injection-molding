"""D 시리즈 실험. 평가 두 가지:
 (A) 레포 fold_id 교차검증 (labeled 안) — 기존 E 시리즈와 같은 기준
 (B) 날짜 단위 홀드아웃 (학습에 쓰지 않은 생산일에서 채점) — 사전 등록한 3개 분할
     CN7: 10/16·10/19·10/20 학습 → 10/27 채점 (대회 파일에 없는 날)
     RG3: 10/21·10/22 학습 → 10/23 채점 / 10/21·10/23 학습 → 10/22 채점
모델: logreg C=0.3 balanced (CN7 최종 모델 설정). 피처 세트: base(공정 23개), +D1, +D2, +D1+D2.
D3(비지도 관리도): 고정 기준선(학습일 정상 샷 평균·표준편차) vs 이동 기준선(직전 20샷, σ 하한 = 0.25 × 학습 정상 σ)."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from dlib import load_public, PROC, D1, D2

import harness as H

Y = "PassOrFail"
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))
pub = load_public()
FSETS = {"base": [], "+D1": D1, "+D2": D2, "+D1+D2": D1 + D2}


def feats_for(mold, extra):
    base = [c for c in PROC if pub.loc[pub.mold == mold, c].nunique() > 1]
    return base + extra


def shot_table(df, score):
    t = pd.DataFrame({"shot": df["shot_id"].values, "y": df[Y].values, "s": score, "t": df["t"].values})
    return t.groupby("shot").agg(y=("y", "max"), s=("s", "max"), t=("t", "min")).sort_values("t")


def holdout_metrics(sh):
    n, P = len(sh), int(sh.y.sum())
    r = sh.s.rank(ascending=False, method="min")
    out = {"test_shots": n, "defect_shots": P, "shot_ap": average_precision_score(sh.y, sh.s),
           "random_ap": P / n, "defect_ranks": ",".join(str(int(x)) for x in sorted(r[sh.y == 1]))}
    for k in (0.05, 0.10):
        m = int(np.ceil(k * n)); top = sh.sort_values("s", ascending=False, kind="mergesort").head(m)
        out[f"recall@{int(k*100)}%({m}샷)"] = top.y.sum() / P
    return out


# ---------- (A) 레포 분할 CV ----------
rowsA = []
for mold in ("cn7", "rg3"):
    lab = pub[pub.row_id.isin(H.FOLD[H.FOLD.mold == mold].row_id)].copy()
    lab = lab.merge(H.FOLD[H.FOLD.mold == mold].drop(columns=["mold", "shot_id"]), on="row_id").reset_index(drop=True)
    base_folds = {}
    for name, extra in FSETS.items():
        f = feats_for(mold, extra)
        for mode in ("block", "shot"):
            s, fo = H.run_cv(lab, f, mk, mode, per_fold=True)
            row = {"mold": mold, "features": name, "cv": mode, **{k: v[0] for k, v in s.items()}, "shot_ap_sd": s["shot_ap"][1]}
            if name == "base": base_folds[mode] = fo
            else: row.update(H.fold_compare(base_folds[mode], fo))
            rowsA.append(row)
A = pd.DataFrame(rowsA)

# ---------- (B) 날짜 홀드아웃 ----------
SPLITS = [("cn7", ["2020-10-16", "2020-10-19", "2020-10-20"], "2020-10-27"),
          ("rg3", ["2020-10-21", "2020-10-22"], "2020-10-23"),
          ("rg3", ["2020-10-21", "2020-10-23"], "2020-10-22")]
rowsB = []
for mold, tr_days, te_day in SPLITS:
    tr = pub[(pub.mold == mold) & pub.date.isin(tr_days)]
    te = pub[(pub.mold == mold) & (pub.date == te_day)]
    for name, extra in FSETS.items():
        f = feats_for(mold, extra)
        m = mk().fit(tr[f].values, tr[Y].values)
        rowsB.append({"mold": mold, "test_day": te_day, "model": f"logreg {name}",
                      **holdout_metrics(shot_table(te, m.predict_proba(te[f].values)[:, 1]))})
    # D3: 비지도 관리도
    f = feats_for(mold, [])
    nrm = tr[tr[Y] == 0][f]; mu, sd = nrm.mean(), nrm.std(ddof=0).replace(0, 1e-9)
    rowsB.append({"mold": mold, "test_day": te_day, "model": "D3 관리도: 고정 기준선",
                  **holdout_metrics(shot_table(te, ((te[f] - mu) / sd).abs().max(axis=1).values))})
    allm = pub[pub.mold == mold]
    shv = allm.groupby("shot_id").agg({**{c: "first" for c in f}, "t": "min"}).sort_values("t")
    prev = shv[f].shift(1).rolling(20, min_periods=5)
    z = ((shv[f] - prev.mean()).abs() / prev.std(ddof=0).clip(lower=0.25 * sd, axis=1)).max(axis=1).fillna(0)
    rowsB.append({"mold": mold, "test_day": te_day, "model": "D3 관리도: 이동 기준선(직전 20샷)",
                  **holdout_metrics(shot_table(te, te["shot_id"].map(z).values))})
B = pd.DataFrame(rowsB)

A.to_csv(os.path.join(H.OUT, "d_results_cv.csv"), index=False); B.to_csv(os.path.join(H.OUT, "d_results_dayholdout.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print("(A) 레포 분할 CV"); print(A[["mold","features","cv","shot_ap","shot_ap_sd","row_ap","shot_recall@5%","shot_recall@10%","folds","alt_wins","base_wins","ties"]].round(3).to_string())
print("\n(B) 날짜 홀드아웃"); print(B.round(3).to_string())
