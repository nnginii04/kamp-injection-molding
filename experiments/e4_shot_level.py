"""E4: shot 단위 학습. 짝 두 행(공정값 동일)을 한 샷으로 합치고, 라벨 = 짝 중 하나라도 불량.
행 단위 학습은 같은 입력에 0과 1이 동시에 붙는 짝(CN7 11개, RG3 25개)이 있어 학습 신호가 서로 상쇄됨.
모델 설정은 baseline 과 같음(logreg C=0.3 balanced). 점수는 두 행에 그대로 복사해 같은 방식으로 채점.
E4+E1c(RG3): E1 채택 후보와 결합했을 때도 확인."""
from harness import *
from seqfeat import add_seq, mk

def shot_fit(feats):
    def fit_fn(tr, va):
        t = tr.groupby("shot_id").agg({**{c: "first" for c in feats}, Y: "max"})
        m = mk().fit(t[feats], t[Y].values)
        return m.predict_proba(va[feats])[:, 1]
    return fit_fn

rows=[]
for mold in ["cn7","rg3"]:
    df, feats = load(mold)
    sets = [("row 학습(baseline)", df, feats, None), ("E4 shot 학습", df, feats, "shot")]
    if mold == "rg3":
        d2, f2 = add_seq(df, feats, "both")
        sets += [("E1c row 학습", d2, f2, None), ("E1c + E4 shot 학습", d2, f2, "shot")]
    for mode in ["block","shot"]:
        base=None
        for name, d, f, kind in sets:
            s, fo = run_cv(d, f, mk, mode, fit_fn=shot_fit(f) if kind else None, per_fold=True)
            row={"mold":mold,"cv":mode,"variant":name,**{k:v[0] for k,v in s.items()},"shot_ap_sd":s["shot_ap"][1]}
            if base is None: base=fo
            else: row.update(fold_compare(base, fo))
            rows.append(row)
r=pd.DataFrame(rows); r.to_csv(os.path.join(OUT, "e4_results.csv"),index=False)
pd.set_option("display.width",250)
print(r[["mold","cv","variant","shot_ap","shot_ap_sd","row_ap","row_auc","shot_recall@2%","shot_recall@5%","shot_recall@10%","folds","alt_wins","base_wins","ties"]].round(3).to_string())
