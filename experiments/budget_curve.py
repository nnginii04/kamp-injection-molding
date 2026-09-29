"""검사 규칙 Budget Curve: 규칙 조합별 '검사 비율(검사한 부품/전체 부품) 대비 잡은 불량 부품 비율'.
단위: 부품(행). 샷을 시간순으로 흘려보내며 온라인으로 판단(미래 정보 없음).
 ① 모델: 샷 점수 ≥ 임계값이면 두 부품 검사. 임계값 = 교차검증(OOF) 점수의 상위 k% 분위수 (레포 판정 규칙과 같은 방식)
 ② 재가동: 재가동(간격 > 5분) 직후 N샷은 두 부품 검사
 ③ 추적: 검사에서 불량을 찾으면 같은 캐비티(LH/RH)의 다음 M샷 검사 (추적 중 또 찾으면 연장)
평가 세트(사전 고정):
 CN7-CV   : 대회 labeled(10/16~20), 점수 = block CV OOF (반복 5회 곡선 평균)
 CN7-10/27: labeled 전체로 학습 → 10/27 채점, 임계값 = labeled OOF 분위수 (처음 보는 날)
 RG3-CV   : 대회 labeled, 점수 = block CV OOF (logreg; 레포 최종은 lgbm 이지만 둘 다 무작위 수준)
격자: k ∈ {0,1,2,3,5,7,10,15,20,30}%, N ∈ {0,3,5,10}, M ∈ {0,1,3,5}"""
import os, sys
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from dlib import load_public, PROC
import harness as H

Y = "PassOrFail"
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))
KS = [0, .01, .02, .03, .05, .07, .10, .15, .20, .30]; NS = [0, 3, 5, 10]; MS = [0, 1, 3, 5]
pub = load_public()
restart_shots = set(pub.loc[pub.shots_since_restart == 0, "shot_id"])


def oof_scores(lab, feats, rep):
    col = f"fold_block_r{rep}"; s = np.full(len(lab), np.nan)
    for f in sorted(lab[col].unique()):
        tr, va = lab[lab[col] != f], lab[lab[col] == f]
        s[va.index] = mk().fit(tr[feats].values, tr[Y].values).predict_proba(va[feats].values)[:, 1]
    return s


def simulate(df, score, thr_by_k):
    """df: 한 평가 세트의 행(시간순 정렬 전), score: 행 점수. 반환: (k,N,M)별 검사율·재현율."""
    d = df.assign(s=score)
    shots = d.groupby("shot_id").agg(t=("t", "min"), s=("s", "max"), seg=("seg", "first"),
                                    ssr=("shots_since_restart", "first")).sort_values("t")
    shots["ssr_n"] = np.round(np.expm1(shots.ssr)).astype(int)
    parts = dict(zip(zip(d.shot_id, d.side), d[Y]))
    assert len(parts) == len(d), "샷 안에 같은 캐비티 행이 2개"
    total_rows, total_def = len(d), int(d[Y].sum())
    out = []
    for k, thr in thr_by_k.items():
        flag1 = shots.s >= thr if k > 0 else pd.Series(False, index=shots.index)
        for N in NS:
            flag2 = shots.ssr_n < N
            for M in MS:
                follow = {"LH": 0, "RH": 0}; insp = caught = 0
                for sid, f1, f2 in zip(shots.index, flag1, flag2):
                    for side in ("LH", "RH"):
                        if (sid, side) not in parts: continue
                        if f1 or f2 or follow[side] > 0:
                            insp += 1; bad = parts[(sid, side)]
                            caught += bad
                            follow[side] = M if (bad and M > 0) else max(follow[side] - 1, 0)
                        else:
                            follow[side] = max(follow[side] - 1, 0)
                out.append({"k": k, "N": N, "M": M, "inspect_rate": insp / total_rows, "recall": caught / total_def,
                            "inspected": insp, "caught": caught, "defects": total_def, "rows": total_rows})
    return pd.DataFrame(out)


res = []
for mold in ("cn7", "rg3"):
    feats = [c for c in PROC if pub.loc[pub.mold == mold, c].nunique() > 1]
    fold = H.FOLD[H.FOLD.mold == mold].drop(columns=["mold", "shot_id"])
    lab = pub[pub.row_id.isin(fold.row_id)].merge(fold, on="row_id").reset_index(drop=True)
    reps = []
    for rep in range(5):
        s = oof_scores(lab, feats, rep)
        sh = pd.Series(s).groupby(lab.shot_id).max()
        thr = {k: (np.quantile(sh, 1 - k) if k > 0 else np.inf) for k in KS}
        r = simulate(lab, s, thr); r["rep"] = rep; reps.append(r)
        if mold == "cn7" and rep == 0:
            thr0 = thr
    cv = pd.concat(reps).groupby(["k", "N", "M"], as_index=False)[["inspect_rate", "recall", "inspected", "caught", "defects", "rows"]].mean()
    cv["set"] = f"{mold.upper()}-CV(10/16~20)" if mold == "cn7" else "RG3-CV(10/21~23)"
    res.append(cv)
    if mold == "cn7":
        te = pub[(pub.mold == "cn7") & (pub.date == "2020-10-27")].reset_index(drop=True)
        m = mk().fit(lab[feats].values, lab[Y].values)
        r = simulate(te, m.predict_proba(te[feats].values)[:, 1], thr0); r["set"] = "CN7-10/27(처음 보는 날)"
        res.append(r)

R = pd.concat(res)[["set", "k", "N", "M", "inspect_rate", "recall", "inspected", "caught", "defects", "rows"]]
R.to_csv(os.path.join(H.OUT, "budget_curve.csv"), index=False)
pd.set_option("display.width", 250)
for s, g in R.groupby("set", sort=False):
    print("\n##", s)
    sel = g[((g.N == 0) & (g.M == 0)) | ((g.k == 0.02) & (g.M.isin([0, 3])) ) | ((g.k == 0) & (g.M == 3))]
    print(sel.round(3).to_string(index=False))
