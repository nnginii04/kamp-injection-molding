"""Descriptive local process-state audit; no fitted prediction/covariance models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import _public

ROOT=Path(__file__).resolve().parents[1]


def standardized_difference(values, controls):
    center=controls.median(); sd=controls.std(ddof=1)
    # Constant controls have no estimable z; never silently divide by an epsilon.
    return (values-center)/sd.replace(0,np.nan),center,sd


def main():
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    feats=[c for c in pd.read_csv(ROOT/'data/raw/moldset_labeled_cn7.csv',nrows=0).columns if c not in ['Unnamed: 0','PassOrFail','Clamp_Open_Position']]
    raw=_public.read_csv(_public.public_path(cfg),'labeled_data.csv',usecols=lambda c:c in feats+['TimeStamp','PART_NAME','PassOrFail','Reason'])
    raw=raw[raw.PART_NAME.str.contains('CN7') & raw.TimeStamp.str.startswith('2020-10-27')].copy()
    raw['TimeStamp']=pd.to_datetime(raw.TimeStamp)
    assert raw.groupby('TimeStamp')[feats].nunique(dropna=False).le(1).all().all()
    shots=raw.sort_values('TimeStamp').drop_duplicates('TimeStamp').set_index('TimeStamp')
    status=raw.groupby('TimeStamp').PassOrFail.agg(lambda s:(s=='N').any())
    shots['defect']=status
    d=shots[shots.defect]; assert len(d)==5
    old=pd.read_csv(ROOT/'outputs/analysis/audit_oct27_defects.csv')
    assert set(d.index)==set(pd.to_datetime(old.TimeStamp))
    assert raw[raw.TimeStamp.isin(d.index)].Reason.eq('초기허용불량').all()
    results=[]; timelines=[]; summaries=[]
    for n in [10,5]:
        before=shots[(shots.index<d.index.min()) & ~shots.defect].tail(n)
        after=shots[(shots.index>d.index.max()) & ~shots.defect].head(n)
        assert len(before)==len(after)==n
        controls=pd.concat([before,after]); selected=pd.concat([before,d,after]).sort_index()
        z,med,sd=standardized_difference(selected[feats],controls[feats])
        dz=z.loc[d.index]
        table=pd.DataFrame({'feature':feats,'normal_median':med.values,'normal_sd':sd.values,
            'before_median':before[feats].median().values,'after_median':after[feats].median().values,
            'defect_median':d[feats].median().values,'median_abs_z':dz.abs().median().values,
            'all_above_center':d[feats].gt(med).all().values,'all_below_center':d[feats].lt(med).all().values,
            'all_abs_z_ge2':dz.abs().ge(2).all().values,'constant_control':sd.eq(0).values,
            'constant_control_changed':(sd.eq(0) & d[feats].ne(med).any()).values})
        for j,t in enumerate(d.index,1):
            table[f'defect_{j}_value']=d.loc[t,feats].to_numpy()
            table[f'defect_{j}_z']=dz.loc[t].to_numpy()
        table['n_each_side']=n; table=table.sort_values('median_abs_z',ascending=False)
        results.append(table)
        # Standardized Euclidean norm, descriptive and correlation-unadjusted.
        valid=sd[sd.gt(0)].index
        distance=np.sqrt(z[valid].pow(2).sum(axis=1))
        ts=selected[feats].copy(); ts['phase']=['before' if t<d.index.min() else 'after' if t>d.index.max() else 'defect' for t in ts.index]
        ts['gap_sec']=shots.index.to_series().diff().dt.total_seconds().reindex(ts.index)
        ts['distance']=distance; ts['n_each_side']=n
        timelines.append(ts.reset_index())
        byphase=ts.groupby('phase').distance.agg(['min','median','max']).to_dict('index')
        summaries.append(dict(n_each_side=n,distance_dimensions=len(valid),zero_sd_features=sd[sd.eq(0)].index.tolist(),
            phase_distances=byphase,normal_distance_max=float(distance.loc[controls.index].max()),
            defect_distance_min=float(distance.loc[d.index].min()),
            common_direction_features=table[table.all_above_center|table.all_below_center].feature.tolist(),
            common_direction_and_all_abs_z_ge2=table[(table.all_above_center|table.all_below_center)&table.all_abs_z_ge2].feature.tolist()))
    out=ROOT/'outputs/analysis/audit_oct27_features'
    pd.concat(results).to_csv(f'{out}_comparison.csv',index=False)
    pd.concat(timelines).to_csv(f'{out}_timeline.csv',index=False)
    summary=dict(source='dataset/labeled_data.csv',defect_times=[str(t) for t in d.index],
        standardization='(value - pooled local normal median) / pooled local normal sample SD (ddof=1)',
        distance='Euclidean norm of standardized differences; no covariance fit; correlated features count separately',
        mahalanobis_not_used='Existing implementation requires fitting StandardScaler and LedoitWolf; no model fitting requested',
        analyses=summaries)
    Path(f'{out}_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(results[0][['feature','normal_median','normal_sd','defect_median','median_abs_z','all_above_center','all_below_center']].to_string(index=False))
    print(timelines[0][['TimeStamp','phase','distance']].to_string(index=False))


if __name__=='__main__': main()
