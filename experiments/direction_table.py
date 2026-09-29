"""변수 방향 검증표: 문헌상 원인 방향 vs 데이터(불량 샷 − 같은 날 정상 샷, 원 단위) vs CN7 모델 계수 방향.
사전 기준: '데이터에서 차이 있음' = |효과| ≥ 0.5 정상σ 이고 Mann-Whitney p < 0.05 (샷 단위). 불량 샷 수가 적어 p 는 참고용."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from dlib import load_public, PROC

# 문헌 기대 방향: +1 = 높을수록 해당 불량, -1 = 낮을수록. (출처 키)
LIT = {
 "가스": {"Barrel_Temperature_*": (+1, "E: 높은 용융온도"), "Average_Screw_RPM": (+1, "E: 높은 스크루 회전"),
          "Max_Screw_RPM": (+1, "E: 높은 스크루 회전"), "Max_Injection_Speed": (+1, "E: 고속 사출 마찰열"),
          "Cycle_Time": (+1, "H: 체류시간(대리변수)")},
 "미성형": {"Barrel_Temperature_*": (-1, "E,F: 낮은 용융온도"), "Mold_Temperature_*": (-1, "E: 낮은 금형온도"),
          "Max_Injection_Pressure": (-1, "E,F: 사출압 부족"), "Max_Injection_Speed": (-1, "E: 낮은 사출속도"),
          "Injection_Time": (+1, "E: 사출시간 지연")},
}
def lit(reason, col):
    for k, v in LIT[reason].items():
        if k == col or (k.endswith("*") and col.startswith(k[:-1])): return v
    return (0, "")

d = load_public(); d["Reason"] = d["Reason"].fillna("")
rows = []
for mold, days in [("cn7", ["2020-10-16"]), ("rg3", ["2020-10-22", "2020-10-23"])]:
    x = d[(d.mold == mold) & d.date.isin(days)]
    feats = [c for c in PROC if x[c].nunique() > 1]
    sh = x.groupby("shot_id").agg({**{c: "first" for c in feats}, "Reason": lambda s: set(s) - {""}})
    norm = sh[sh.Reason.apply(len) == 0]
    lab = d[(d.mold == mold) & (d.date < "2020-10-27")]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)).fit(lab[feats], lab.PassOrFail)
    coef = dict(zip(feats, m[-1].coef_[0]))
    for reason in ("가스", "미성형"):
        dfc = sh[sh.Reason.apply(lambda r: reason in r)]
        for c in feats:
            sdn = norm[c].std()
            eff = (dfc[c].median() - norm[c].median()) / sdn if sdn > 0 else np.nan
            p = mannwhitneyu(dfc[c], norm[c]).pvalue if sdn > 0 else np.nan
            e, src = lit(reason, c)
            obs = int(np.sign(eff)) if (abs(eff) >= 0.5 and p < 0.05) else 0
            rows.append({"mold": mold, "reason": reason, "n_defect_shots": len(dfc), "n_normal_shots": len(norm), "variable": c,
                         "normal_median": norm[c].median(), "defect_median": dfc[c].median(), "effect_sigma": eff, "p_mw": p,
                         "lit_dir": e, "lit_src": src, "obs_dir": obs,
                         "model_coef(cn7 logreg, 전체 불량)": coef[c] if mold == "cn7" else np.nan})
T = pd.DataFrame(rows)
def verdict(r):
    if r.lit_dir == 0: return "문헌 기준 없음" if r.obs_dir == 0 else "데이터만 차이"
    if r.obs_dir == 0: return "데이터 차이 없음"
    return "일치" if r.obs_dir == r.lit_dir else "반대"
T["verdict"] = T.apply(verdict, axis=1)
import harness as H
T.to_csv(os.path.join(H.OUT, "direction_table.csv"), index=False)
pd.set_option("display.width", 250)
show = T[(T.lit_dir != 0) | (T.obs_dir != 0)]
print(show[["mold","reason","n_defect_shots","variable","normal_median","defect_median","effect_sigma","p_mw","lit_dir","obs_dir","model_coef(cn7 logreg, 전체 불량)","verdict"]].round(3).to_string())
