"""E2: 가동 구간(최근 N샷) 기준선 관리도. 라벨·학습 없음(비지도).
사전 등록: N=20샷, 표준편차 하한 0.25(대회 z 단위) — 10/27 분석에서 '평소 거의 안 움직이는 변수가 조금만 움직여도 7σ'가 된 문제를 막는 하한.
점수 = 변수별 |x - 직전 N샷 평균| / max(직전 N샷 표준편차, 하한) 의 최대값 (spc_maxz 와 같은 모양, 기준선만 이동).
비교 대상: 레포 spc_maxz (학습 fold 정상 샘플의 평균·표준편차 고정 기준선)."""
from harness import *

def spc_static(df, feats, mode):
    def fit_fn(tr, va):
        n = tr[tr[Y] == 0][feats]; mu, sd = n.mean(), n.std(ddof=0).replace(0, 1e-9)
        return ((va[feats] - mu) / sd).abs().max(axis=1).values
    return run_cv(df, feats, None, mode, fit_fn=fit_fn, per_fold=True)

def spc_rolling(df, feats, mode, N=20, floor=0.25):
    shot = df.groupby("shot_idx")[feats].first().sort_index()
    prev = shot.shift(1).rolling(N, min_periods=5)
    z = (shot - prev.mean()).abs() / prev.std(ddof=0).clip(lower=floor)
    sc = z.max(axis=1).fillna(0.0)
    s_row = df["shot_idx"].map(sc).values
    fit_fn = lambda tr, va: s_row[va.index]
    return run_cv(df, feats, None, mode, fit_fn=fit_fn, per_fold=True)

rows=[]
for mold in ["cn7","rg3"]:
    df, feats = load(mold)
    for mode in ["block","shot"]:
        b, bf = spc_static(df, feats, mode)
        rows.append({"mold":mold,"cv":mode,"variant":"spc_maxz(고정 기준선, 레포)",**{k:v[0] for k,v in b.items()}})
        for N in [20]:
            s, sf = spc_rolling(df, feats, mode, N)
            rows.append({"mold":mold,"cv":mode,"variant":f"E2 이동 기준선 N={N}",**{k:v[0] for k,v in s.items()}, **fold_compare(bf, sf)})
r=pd.DataFrame(rows); r.to_csv(os.path.join(OUT, "e2_results.csv"),index=False)
pd.set_option("display.width",250)
print(r[["mold","cv","variant","shot_ap","row_ap","row_auc","shot_recall@2%","shot_recall@5%","shot_recall@10%","folds","alt_wins","base_wins","ties"]].round(3).to_string())
