"""캐비티(LH/RH) 분석 + Budget Curve 에 ④ RH 우선 검사 추가.
(1) 캐비티별 불량률: 제품 × 날짜 × 캐비티 × 불량 사유, LH vs RH 차이 검정(Fisher)
(2) 검사 규칙 비교 (부품 단위, 샷을 시간순으로 흘리며 적용, 미래 정보 없음)
    ① 모델 상위 k% (두 부품)                 : 기존과 같음 (block CV OOF 점수, 10/27 은 labeled 로 학습)
    ⑤ 균등 샘플링 (두 부품, 매 1/s 샷)       : 모델·지식 없이 검사량만 맞춘 기준선
    ④ RH 우선 샘플링 (RH 부품만, 매 1/s 샷)  : 도메인 가설 1 (RH 캐비티 금형 요인)
    ③ 추적: 불량을 찾으면 같은 캐비티 다음 3샷
    샘플링 규칙(④⑤)은 시작 위치를 바꿔 가며 평균한 기대값을 쓴다.
    ④ 규칙의 'RH' 는 다른 날짜에서 먼저 관측된 사실에서 정한다 (날짜 홀드아웃에서 순환논리 방지):
      RG3 10/22 → 10/23 채점, 10/23 → 10/22 채점, CN7 10/16 → 10/27 채점
"""
import os
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from dlib import load_public, PROC
import harness as H

Y = "PassOrFail"
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))
pub = load_public(); pub["Reason"] = pub["Reason"].fillna("")

# ---------- (1) 캐비티별 불량률 ----------
rows = []
for (mold, date), g in pub.groupby(["mold", "date"]):
    n = g.groupby("side").size()
    for reason in ["가스", "미성형", "초기허용불량", "전체"]:
        m = g[Y] == 1 if reason == "전체" else g["Reason"] == reason
        c = g[m].groupby("side").size().reindex(["LH", "RH"], fill_value=0)
        if reason != "전체" and c.sum() == 0: continue
        p = fisher_exact([[c["LH"], n["LH"] - c["LH"]], [c["RH"], n["RH"] - c["RH"]]])[1]
        rows.append({"mold": mold, "date": date, "reason": reason, "LH_rows": n["LH"], "LH_defect": c["LH"],
                     "LH_rate%": 100 * c["LH"] / n["LH"], "RH_rows": n["RH"], "RH_defect": c["RH"],
                     "RH_rate%": 100 * c["RH"] / n["RH"], "RH_share": c["RH"] / c.sum() if c.sum() else np.nan,
                     "fisher_p": p})
CAV = pd.DataFrame(rows)
CAV.to_csv(os.path.join(H.OUT, "cavity_defect_rates.csv"), index=False)


# ---------- (2) 규칙 시뮬레이션 ----------
def simulate_once(df, score, thr, s_both, s_rh, M, off):
    d = df.assign(s=score)
    shots = d.groupby("shot_id").agg(t=("t", "min"), s=("s", "max")).sort_values("t")
    parts = dict(zip(zip(d.shot_id, d.side), d[Y]))
    follow = {"LH": 0, "RH": 0}; insp = caught = 0
    step_b = int(round(1 / s_both)) if s_both > 0 else 0
    step_r = int(round(1 / s_rh)) if s_rh > 0 else 0
    for i, (sid, sc) in enumerate(zip(shots.index, shots.s)):
        f1 = sc >= thr
        fb = step_b > 0 and (i + off) % step_b == 0
        fr = step_r > 0 and (i + off) % step_r == 0
        for side in ("LH", "RH"):
            if (sid, side) not in parts: continue
            if f1 or fb or (fr and side == "RH") or follow[side] > 0:
                insp += 1; bad = parts[(sid, side)]; caught += bad
                follow[side] = M if (bad and M > 0) else max(follow[side] - 1, 0)
            else:
                follow[side] = max(follow[side] - 1, 0)
    return insp / len(d), caught / d[Y].sum()


def simulate(df, score, thr, s_both, s_rh, M):
    """샘플링 규칙은 시작 위치(offset)에 따라 결과가 달라지므로 가능한 모든 offset 의 평균을 쓴다 (기대값)."""
    step = max(int(round(1 / s_both)) if s_both > 0 else 1, int(round(1 / s_rh)) if s_rh > 0 else 1)
    r = np.array([simulate_once(df, score, thr, s_both, s_rh, M, o) for o in range(step)])
    return r[:, 0].mean(), r[:, 1].mean()


def oof(lab, feats, rep=0):
    col = f"fold_block_r{rep}"; s = np.full(len(lab), np.nan)
    for f in sorted(lab[col].unique()):
        tr, va = lab[lab[col] != f], lab[lab[col] == f]
        s[va.index] = mk().fit(tr[feats].values, tr[Y].values).predict_proba(va[feats].values)[:, 1]
    return s


SETS = []
for mold in ("cn7", "rg3"):
    feats = [c for c in PROC if pub.loc[pub.mold == mold, c].nunique() > 1]
    fold = H.FOLD[H.FOLD.mold == mold].drop(columns=["mold", "shot_id"])
    lab = pub[pub.row_id.isin(fold.row_id)].merge(fold, on="row_id").reset_index(drop=True)
    s = oof(lab, feats); shs = pd.Series(s).groupby(lab.shot_id).max()
    thr = {k: np.quantile(shs, 1 - k) for k in (.01, .02, .05, .10, .20)}
    if mold == "cn7":
        SETS.append(("CN7-CV(10/16~20)", lab, s, thr))
        te = pub[(pub.mold == "cn7") & (pub.date == "2020-10-27")].reset_index(drop=True)
        SETS.append(("CN7-10/27(처음 보는 날)", te, mk().fit(lab[feats].values, lab[Y].values).predict_proba(te[feats].values)[:, 1], thr))
    else:
        SETS.append(("RG3-CV(10/21~23)", lab, s, thr))
        for tr_d, te_d in [(["2020-10-21", "2020-10-22"], "2020-10-23"), (["2020-10-21", "2020-10-23"], "2020-10-22")]:
            tr = pub[(pub.mold == "rg3") & pub.date.isin(tr_d)]; te = pub[(pub.mold == "rg3") & (pub.date == te_d)].reset_index(drop=True)
            m = mk().fit(tr[feats].values, tr[Y].values)
            st = pd.Series(m.predict_proba(tr[feats].values)[:, 1]).groupby(tr.shot_id.values).max()
            SETS.append((f"RG3-{te_d[5:].replace('-', '/')}(날짜 홀드아웃)", te, m.predict_proba(te[feats].values)[:, 1],
                         {k: np.quantile(st, 1 - k) for k in (.01, .02, .05, .10, .20)}))

S = [.05, .10, .20, .50, 1.0]
out = []
for name, df, sc, thr in SETS:
    for k, t in thr.items():
        for M in (0, 3):
            ir, rc = simulate(df, sc, t, 0, 0, M); out.append((name, f"① 모델 상위 {k:.0%}" + (" + ③" if M else ""), "①" + ("+③" if M else ""), k, M, ir, rc))
    for s in S:
        for M in (0, 3):
            ir, rc = simulate(df, sc, np.inf, s, 0, M); out.append((name, f"⑤ 균등 {s:.0%}" + (" + ③" if M else ""), "⑤" + ("+③" if M else ""), s, M, ir, rc))
            ir, rc = simulate(df, sc, np.inf, 0, s, M); out.append((name, f"④ RH {s:.0%}" + (" + ③" if M else ""), "④" + ("+③" if M else ""), s, M, ir, rc))
    for s in S:
        ir, rc = simulate(df, sc, thr[.02], 0, s, 3); out.append((name, f"① 2% + ④ RH {s:.0%} + ③", "①2%+④+③", s, 3, ir, rc))
R = pd.DataFrame(out, columns=["set", "rule", "family", "param", "M", "inspect_rate", "recall"])
R.to_csv(os.path.join(H.OUT, "budget_curve_cavity.csv"), index=False)

pd.set_option("display.width", 250)
print("## 캐비티별 불량률"); print(CAV.round(3).to_string(index=False))
for n, g in R.groupby("set", sort=False):
    print("\n##", n)
    print(g[g.rule.isin(["① 모델 상위 2%", "① 모델 상위 2% + ③", "⑤ 균등 10%", "④ RH 20%", "⑤ 균등 50%", "④ RH 100%",
                         "④ RH 20% + ③", "① 2% + ④ RH 20% + ③", "① 2% + ④ RH 100% + ③"])][["rule", "inspect_rate", "recall"]].round(3).to_string(index=False))
