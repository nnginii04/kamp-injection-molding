from harness import *
from seqfeat import add_seq, mk
rng = np.random.default_rng(1)
df, feats = load("rg3")
d2, f2 = add_seq(df, feats, "both")
real = run_cv(d2, f2, mk, "block")["shot_ap"][0]
null=[]
for i in range(200):
    d = df.copy(); perm = rng.permutation(d.shot_idx.max()+1)
    d["shot_idx"] = d["shot_idx"].map(dict(enumerate(perm)))
    null.append(run_cv(*add_seq(d, feats, "both"), mk, "block")["shot_ap"][0])
null=np.array(null)
print("rg3 E1c block shot_ap real", round(real,4), "| null mean", null.mean().round(4), "p95", np.quantile(null,.95).round(4), "p99", np.quantile(null,.99).round(4), "max", null.max().round(4), "| p =", ((null>=real).sum()+1)/(len(null)+1))
np.save(os.path.join(OUT, "e1_null_rg3.npy"), null)
# 기여 변수: 전체 데이터로 학습한 표준화 계수 상위
m = mk().fit(d2[f2].values, d2[Y].values)
coef = pd.Series(m[-1].coef_[0], index=f2).sort_values(key=abs, ascending=False)
print(coef.head(12).round(3).to_string())
