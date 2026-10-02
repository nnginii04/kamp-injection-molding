"""Descriptive competition one-side matched audit. No prediction model training."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _public
from audit_oct27_features import standardized_difference
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data import assign_shots


def match_controls(shots,t,n=5):
    normal=shots[(shots.index.date==t.date()) & shots.normal]
    before=normal[normal.index<t].tail(n); after=normal[normal.index>t].head(n)
    if len(before)<n or len(after)<n:
        return None
    return pd.concat([before,after])


def effects(long):
    rows=[]
    for feature,g in long.groupby('feature'):
        valid=g.z.dropna()
        rows.append(dict(feature=feature,n_defects=len(g),n_finite_z=len(valid),
            defect_median=g.defect_value.median(),matched_normal_median=g.control_median.median(),
            median_paired_difference=g.difference.median(),median_z=valid.median(),median_abs_z=valid.abs().median(),
            fraction_above=float(g.difference.gt(0).mean()),fraction_below=float(g.difference.lt(0).mean()),
            fraction_abs_z_ge2=float(valid.abs().ge(2).mean()) if len(valid) else None,
            zero_sd_changed=int(g.zero_sd_changed.sum())))
    return pd.DataFrame(rows).sort_values('median_abs_z',ascending=False)


def main():
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8')); src=_public.public_path(cfg)
    mold=_public.read_csv(src,'moldset_labeled.csv')
    raw=_public.read_csv(src,'labeled_data.csv')
    comps={p:pd.read_csv(ROOT/cfg['data_dir']/cfg['products'][p]['labeled']) for p in ['cn7','rg3']}
    allfeats=[c for c in comps['cn7'] if c not in [cfg['id_col'],cfg['label_col']]]
    base=[c for c in allfeats if c!='Clamp_Open_Position']; assert len(base)==23
    derived=['Mold_Temperature_3_minus_4','Injection_minus_Switch_Pressure']
    feats=base+derived
    unl=_public.read_csv(src,'unlabeled_data.csv',usecols=lambda c:c in base+['TimeStamp','PART_NAME'])
    pairs=[]; levels=[]; deltas=[]; summaries=[]; audits={}
    out=ROOT/'outputs/analysis/oneside_matched_'
    for product,comp in comps.items():
        mapped=assign_shots(comp.rename(columns={cfg['id_col']:'row_id'}),allfeats,product)
        m=mapped.merge(mold,left_on='row_id',right_on=cfg['id_col'],suffixes=('','_raw'),validate='one_to_one')
        assert m.PART_NAME.str.upper().str.contains(product.upper()).all()
        assert m.PassOrFail.eq(m.PassOrFail_raw).all()
        assert m.groupby('shot_id').TimeStamp.nunique().eq(1).all()
        # Re-standardize matching raw population, verify values independently of labels.
        x=m[[f+'_raw' for f in allfeats]].to_numpy(float); sd=x.std(axis=0); sd[sd==0]=1
        assert np.allclose((x-x.mean(axis=0))/sd,m[allfeats].to_numpy(float),atol=0.002)
        target=m.groupby('shot_id').agg(n=('PassOrFail','size'),bad=('PassOrFail','sum'),TimeStamp=('TimeStamp','first'))
        target=target[(target.bad>0)&(target.bad<target.n)].copy(); target.TimeStamp=pd.to_datetime(target.TimeStamp)
        assert len(target)=={'cn7':11,'rg3':25}[product]
        dates=set(target.TimeStamp.dt.strftime('%Y-%m-%d'))
        r=raw[raw.PART_NAME.str.upper().str.contains(product.upper()) & raw.TimeStamp.str[:10].isin(dates)].copy()
        assert r.groupby(['TimeStamp','PART_NAME'])[['PassOrFail','Reason']].nunique(dropna=False).le(1).all().all()
        r=r.drop_duplicates(['TimeStamp','PART_NAME']); r.TimeStamp=pd.to_datetime(r.TimeStamp)
        assert r.groupby('TimeStamp')[base].nunique(dropna=False).le(1).all().all()
        status=r.groupby('TimeStamp').agg(n=('PART_NAME','size'),normal=('PassOrFail',lambda v:v.eq('Y').all()),bad=('PassOrFail',lambda v:v.eq('N').sum()))
        # Include unlabeled observed shots only for predecessor values, never as known-normal controls.
        u=unl[unl.PART_NAME.str.upper().str.contains(product.upper()) & unl.TimeStamp.str[:10].isin(dates)].copy()
        u.TimeStamp=pd.to_datetime(u.TimeStamp)
        timeline=pd.concat([r[['TimeStamp']+base],u[['TimeStamp']+base]],ignore_index=True)
        spread=timeline.groupby('TimeStamp')[base].max()-timeline.groupby('TimeStamp')[base].min()
        assert spread.le(0.00005).all().all()  # verified float32 versus decimal storage; prefer labeled source
        timeline=timeline.drop_duplicates('TimeStamp').sort_values('TimeStamp').set_index('TimeStamp')
        timeline[derived[0]]=timeline.Mold_Temperature_3-timeline.Mold_Temperature_4
        timeline[derived[1]]=timeline.Max_Injection_Pressure-timeline.Max_Switch_Over_Pressure
        timeline['previous_time']=timeline.index.to_series().groupby(timeline.index.date).shift()
        timeline['gap_sec']=(timeline.index.to_series()-timeline.previous_time).dt.total_seconds()
        delta=timeline[feats].groupby(timeline.index.date).diff()
        shots=timeline.join(status); shots['normal']=shots.normal.fillna(False) & shots.n.eq(2)
        exclusions=[]
        for sid,row in target.iterrows():
            t=row.TimeStamp; assert shots.loc[t,'bad']==1 and shots.loc[t,'n']==2
            control=match_controls(shots,t)
            if control is None:
                exclusions.append(dict(shot_id=sid,reason='fewer than 5 normal shots on either side on same date')); continue
            for ct in control.index:
                pairs.append(dict(product=product,defect_shot_id=sid,defect_time=t,control_time=ct,
                    side='before' if ct<t else 'after',offset_sec=(ct-t).total_seconds(),control_previous_time=shots.loc[ct,'previous_time'],control_previous_gap_sec=shots.loc[ct,'gap_sec']))
            def summarize_values(values, controls, dest, kind):
                z,med,sd=standardized_difference(values.to_frame().T,controls)
                for f in feats:
                    dest.append(dict(product=product,shot_id=sid,TimeStamp=t,feature=f,defect_value=values[f],
                        control_median=med[f],difference=values[f]-med[f],z=z.iloc[0][f],control_sd=sd[f],
                        zero_sd_changed=bool(sd[f]==0 and values[f]!=med[f]),n_controls=int(controls[f].notna().sum())))
                return z.iloc[0],sd
            z,sd=summarize_values(shots.loc[t,feats].astype(float),control[feats],levels,'level')
            summarize_values(delta.loc[t],delta.loc[control.index],deltas,'delta')
            bz,_,_=standardized_difference(control[base],control[base]); valid=[f for f in base if sd[f]>0]
            dist=float(np.sqrt(z[valid].pow(2).sum())); nd=np.sqrt(bz[valid].pow(2).sum(axis=1))
            summaries.append(dict(product=product,shot_id=sid,TimeStamp=t,n_controls=len(control),
                distance=dist,normal_distance_max=float(nd.max()),normal_distance_median=float(nd.median()),
                distance_ratio_to_normal_max=dist/nd.max(),distance_dimensions=len(valid),
                zero_sd_changed_features=';'.join(f for f in base if sd[f]==0 and shots.loc[t,f]!=control[f].median()),
                previous_time=shots.loc[t,'previous_time'],previous_gap_sec=shots.loc[t,'gap_sec'],
                max_control_offset_sec=max(abs((ct-t).total_seconds()) for ct in control.index)))
        audits[product]=dict(target_count=len(target),exclusions=exclusions,dates=sorted(dates),
                            predecessor_source='labeled_data + unlabeled_data, within same product/date; observed previous shot')
    pairs=pd.DataFrame(pairs); level=pd.DataFrame(levels); delta=pd.DataFrame(deltas); defect=pd.DataFrame(summaries)
    pairs.to_csv(f'{out}matched_pairs.csv',index=False)
    level.to_csv(f'{out}defect_feature_details.csv',index=False);delta.to_csv(f'{out}defect_delta_details.csv',index=False)
    defect.to_csv(f'{out}defect_level_summary.csv',index=False)
    for product in ['cn7','rg3']:
        ef=effects(level[level['product'].eq(product)]);de=effects(delta[delta['product'].eq(product)])
        ef.to_csv(f'{out}feature_effects_{product}.csv',index=False);de.to_csv(f'{out}previous_delta_{product}.csv',index=False)
        p=pairs[pairs['product'].eq(product)];audits[product].update(analyzed=int(defect['product'].eq(product).sum()),control_pairs=len(p),unique_control_shots=p.control_time.nunique())
        top=ef[ef.feature.isin(base) & ef.n_finite_z.ge(ef.n_defects*.8)].head(5).feature.tolist()
        fig,axes=plt.subplots(1,5,figsize=(16,4))
        for ax,f in zip(axes,top):
            g=level[(level['product']==product)&level.feature.eq(f)]
            for _,r in g.iterrows(): ax.plot([0,1],[r.control_median,r.defect_value],color='#718096',alpha=.55)
            ax.scatter(np.zeros(len(g)),g.control_median,color='#267f99',s=15);ax.scatter(np.ones(len(g)),g.defect_value,color='#d45b39',s=15)
            ax.set_xticks([0,1],['Local normal\nmedian','Defect']);ax.set_title(f.replace('_','\n'),fontsize=9)
        fig.suptitle(product.upper()+' one-side: matched values (raw units)');fig.tight_layout();fig.savefig(f'{out}{product}_top_features.png',dpi=160);plt.close(fig)
        print(product,'TOP',ef[ef.feature.isin(base)].head(5).to_string(index=False))
        print(product,'DELTA',de.head(5).to_string(index=False))
    fig,axes=plt.subplots(2,1,figsize=(13,7))
    for ax,p in zip(axes,['cn7','rg3']):
        g=defect[defect['product'].eq(p)].sort_values('TimeStamp');ax.bar(g.shot_id,g.distance_ratio_to_normal_max,color='#267f99');ax.axhline(1,color='#d45b39',linestyle='--');ax.set_title(p.upper());ax.tick_params(axis='x',rotation=60);ax.set_ylabel('Distance / local normal max')
    fig.tight_layout();fig.savefig(f'{out}distance.png',dpi=160);plt.close(fig)
    result=dict(products=audits,base_features=base,derived_definitions={derived[0]:'Mold_Temperature_3 - Mold_Temperature_4',derived[1]:'Max_Injection_Pressure - Max_Switch_Over_Pressure'},
       method='5 preceding + 5 following known-normal two-cavity shots; same product/day; control reuse allowed',
       effect='per defect: (value - local control median)/local sample SD; aggregate paired differences and z by median',
       distance='Reuse audit_oct27_features.standardized_difference; Euclidean norm across nonconstant base features; no covariance/model fit',
       ranking='Top base features require finite z for at least 80% of matched defects; remaining effects retained with coverage counts',
       limitations=['Controls reused, observations not independent','Constant controls yield undefined z; raw changes flagged separately',
                    'Distance uses controls for center/scale and reference; descriptive, not calibrated detection performance',
                    'Deltas include preceding abnormal shots; predecessor times/gaps are recorded','CN7 defects share one production date'])
    Path(f'{out}summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(audits,indent=2));print(defect.to_string(index=False))


if __name__=='__main__':main()


