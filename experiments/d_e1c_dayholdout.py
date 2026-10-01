"""B안 보고 문장 점검: RG3 E1c(직전 샷 Δ1 + 직전 5샷 중앙값 편차)가 날짜 홀드아웃에서도 유지되는가."""
import os
import numpy as np, pandas as pd
import harness as H
from d1_d3_domain import pub, mk, feats_for, shot_table, holdout_metrics, Y
rows=[]
for tr_days, te_day in [(["2020-10-21","2020-10-22"],"2020-10-23"),(["2020-10-21","2020-10-23"],"2020-10-22")]:
    m = pub[pub.mold=="rg3"].copy(); f = feats_for("rg3", [])
    shv = m.groupby("shot_id").agg({**{c:"first" for c in f},"t":"min"}).sort_values("t")
    d1 = shv[f].diff().fillna(0); d1.columns=[c+"__d1" for c in f]
    dv = (shv[f]-shv[f].shift(1).rolling(5,min_periods=1).median()).fillna(0); dv.columns=[c+"__dev5" for c in f]
    m = m.merge(pd.concat([d1,dv],axis=1), left_on="shot_id", right_index=True)
    fe = f + list(d1.columns) + list(dv.columns)
    tr, te = m[m.date.isin(tr_days)], m[m.date==te_day]
    for name, ff in [("base", f), ("E1c", fe)]:
        mod = mk().fit(tr[ff], tr[Y].values)
        r = holdout_metrics(shot_table(te, mod.predict_proba(te[ff])[:,1]))
        rows.append({"test_day":te_day,"features":name,"shot_ap":round(r["shot_ap"],3),"random_ap":round(r["random_ap"],3)})
R = pd.DataFrame(rows); R.to_csv(os.path.join(H.OUT, "d_e1c_dayholdout.csv"), index=False)
print(R.to_string(index=False))
