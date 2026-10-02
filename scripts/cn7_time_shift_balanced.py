"""Equal-date sampling and date-group robustness; no baseline changes."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import transfer_risk as tr
from cn7_time_shift import split_dates


def balanced_sample(df, n, seed):
    if df.groupby('production_date').size().min() < n:
        raise ValueError('Every date must supply n distinct shots')
    return pd.concat([g.sample(n=n, replace=False, random_state=seed) for _, g in df.groupby('production_date')]).reset_index(drop=True)


def grouped_auc(df, feats, early):
    X = df[feats].to_numpy(); y = (~df.production_date.isin(early)).astype(int).to_numpy()
    models = {'lr': tr.make_pipeline(StandardScaler(), tr.LogisticRegression(max_iter=2000)),
              'rf': tr.RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)}
    pred = {k: np.zeros(len(df)) for k in models}
    # Leave an entire date out: 12 fits, all validation shots from unseen dates.
    for date in sorted(df.production_date.unique()):
        test = df.production_date.eq(date).to_numpy()
        for name, model in models.items():
            fit = clone(model).fit(X[~test], y[~test])
            pred[name][test] = fit.predict_proba(X[test])[:, 1]
    out = df[['production_date']].copy(); out['period'] = y
    for name in models: out[name] = pred[name]
    return {f'group_auc_{k}': float(roc_auc_score(y, p)) for k,p in pred.items()}, out


def metrics(df, feats, early, seed):
    A=df[df.production_date.isin(early)]; B=df[~df.production_date.isin(early)]
    sc=StandardScaler().fit(df[feats])
    w=tr.per_variable_w1(A,B,feats)
    m=tr.mmd_rbf(sc.transform(A[feats]),sc.transform(B[feats]),n_perm=500,seed=seed)
    a,_=tr.domain_auc(sc.transform(A[feats]),sc.transform(B[feats]),feats,seed)
    return dict(w1_mean=float(w.w1_std.mean()),**m,**a),w


def date_permutation(df, feats, early):
    Z=StandardScaler().fit_transform(df[feats]); D=cdist(Z,Z)
    bw=float(np.median(D[np.triu_indices_from(D,1)])); K=np.exp(-D**2/(2*bw**2))
    days=sorted(df.production_date.unique())
    def stat(selected):
        mask=df.production_date.isin(selected).to_numpy()
        return tr._mmd2_from_kernel(K,np.flatnonzero(mask),np.flatnonzero(~mask))
    obs=stat(early); values=np.array([stat(list(c)) for c in combinations(days,6)])
    return dict(mmd2=obs,exact_date_permutation_p=float(np.mean(values>=obs-1e-12)),n_assignments=len(values))


def main():
    root=tr.ROOT; prefix=root/'outputs/analysis/cn7_time_split_balanced'
    prior=json.loads((root/'outputs/analysis/cn7_time_split_summary.json').read_text(encoding='utf-8'))
    feats=prior['original_features']; early=prior['early_dates']; late=prior['late_dates']
    d=tr._public.read_csv(Path(prior['source']),'unlabeled_data.csv',usecols=lambda c:c in ['TimeStamp','PART_NAME']+feats)
    d=d[d.PART_NAME.astype(str).str.upper().str.extract(r'(CN7|RG3)',expand=False).eq('CN7')]
    d,_,e,l=split_dates(d); assert e==early and l==late
    d=tr.dedupe_shots(d[d.production_date.isin(early+late)],feats)
    counts=d.groupby('production_date').size(); n=int(counts.min())
    records=[]; ws=[]
    for seed in range(42,52):
        sample=balanced_sample(d,n,seed)
        r,w=metrics(sample,feats,early,seed); r['seed']=seed
        records.append(r); w['seed']=seed; ws.append(w)
        print('balanced seed',seed,flush=True)
    sample=balanced_sample(d,n,42)
    group,pred=grouped_auc(sample,feats,early)
    exact=date_permutation(sample,feats,early)
    removed=[]
    for day in sorted(counts.index):
        sub=sample[~sample.production_date.eq(day)].reset_index(drop=True)
        r,_=metrics(sub,feats,early,42)
        g,_=grouped_auc(sub,feats,early)
        removed.append(dict(removed_date=day,**r,**g))
        print('removed',day,flush=True)
    # All original shots inform each date's mean; each date receives one vote.
    means=d.groupby('production_date')[feats].mean().reset_index()
    mean_exact=date_permutation(means,feats,early)
    daily=counts.rename('available_shots').to_frame(); daily['sampled_shots']=n
    daily.to_csv(f'{prefix}_daily_counts.csv')
    pd.DataFrame(records).to_csv(f'{prefix}_repeats.csv',index=False)
    pd.concat(ws).to_csv(f'{prefix}_per_variable.csv',index=False)
    pd.DataFrame(removed).to_csv(f'{prefix}_leave_one_date_out.csv',index=False)
    pred.to_csv(f'{prefix}_date_oof.csv',index=False)
    result=dict(n_per_date=n,n_early=6*n,n_late=6*n,seeds=list(range(42,52)),sampling='without replacement; exactly 8 distinct shots per date; no date excluded',
                infeasible_caps=[30,50,100],reason='minimum date has 8 shots; larger equal counts require duplication or exclusion',
                primary=records[0],date_group_oof=group,date_block_permutation=exact,full_data_date_mean_permutation=mean_exact,
                early_dates=early,late_dates=late,limitations=['12 dates only; random date-label exchangeability is an assumption',
                'Repeated samples are not independent datasets','Date-centroid analysis measures mean shifts, not entire distributions'])
    Path(f'{prefix}_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
