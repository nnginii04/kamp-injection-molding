"""Train-fold-only, class-distance ambiguity. Run: python -m src.analysis.ambiguity"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from src.data import assign_shots, sha256

ROOT=Path(__file__).resolve().parents[2]
FEATURES=['Injection_Time','Filling_Time','Plasticizing_Time','Cycle_Time','Clamp_Close_Time','Cushion_Position','Plasticizing_Position','Max_Injection_Speed','Max_Screw_RPM','Average_Screw_RPM','Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure','Average_Back_Pressure',*[f'Barrel_Temperature_{i}' for i in range(1,7)],'Hopper_Temperature','Mold_Temperature_3','Mold_Temperature_4']


def ambiguity(d0,d1):
    total=d0+d1
    score=np.divide(2*np.minimum(d0,d1),total,out=np.full_like(total,np.nan),where=total>0)
    score[total==0]=1.0
    return score


def score_fold(reference,query,k=10):
    """query labels/side are ignored; reference labels define the two classes only."""
    if reference['product'].nunique()!=1 or query['product'].nunique()!=1 or reference['product'].iloc[0]!=query['product'].iloc[0]:
        raise ValueError('Mold mixing forbidden')
    if set(reference.shot_id)&set(query.shot_id): raise ValueError('Shot overlap')
    if not set(reference.y.unique())<={0,1}: raise ValueError('Invalid reference label')
    for frame in [reference,query]:
        if not frame.groupby('shot_id')[FEATURES].nunique(dropna=False).le(1).all().all(): raise ValueError('Inconsistent within-shot features')
    train=reference.drop_duplicates('shot_id').sort_values('shot_id').copy()
    fraction=reference.groupby('shot_id').y.mean().reindex(train.shot_id).to_numpy()
    sc=StandardScaler().fit(train[FEATURES])
    distances=cdist(sc.transform(query[FEATURES]),sc.transform(train[FEATURES]))
    result=query[['product','row_id','shot_id']].reset_index(drop=True).copy()
    for label,name in [(0,'normal'),(1,'defect')]:
        ids=set(reference.loc[reference.y.eq(label),'shot_id']); mask=train.shot_id.isin(ids).to_numpy()
        if not mask.any():
            result[f'nearest_{name}_distance']=np.nan;result[f'nearest_{name}_shot_id']=None
        else:
            sub=distances[:,mask]; idx=sub.argmin(axis=1)
            result[f'nearest_{name}_distance']=sub[np.arange(len(query)),idx]
            result[f'nearest_{name}_shot_id']=train.loc[mask,'shot_id'].to_numpy()[idx]
    ratios=[];counts=[];ids=[];mixed=[]
    for ds in distances:
        boundary=np.sort(ds)[min(k,len(ds))-1]
        use=ds<=boundary+1e-12  # include all boundary ties, equal shot weight
        ratios.append(float(fraction[use].mean()));counts.append(int(use.sum()))
        ids.append(';'.join(train.shot_id.to_numpy()[use]));mixed.append(int(((fraction[use]>0)&(fraction[use]<1)).sum()))
    result['local_defect_ratio']=ratios;result['local_neighbor_count']=counts
    result['local_neighbor_shot_ids']=ids;result['local_mixed_label_shots']=mixed
    result['ambiguity_score']=ambiguity(result.nearest_normal_distance.to_numpy(),result.nearest_defect_distance.to_numpy())
    result['nearest_classes_same_shot']=result.nearest_normal_shot_id.eq(result.nearest_defect_shot_id)
    result['reference_shots']=len(train);result['reference_defect_shots']=int((fraction>0).sum())
    return result,dict(mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),train_shot_ids=train.shot_id.tolist())


def main():
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    foldpath=ROOT/'outputs/fold_id.csv';fold_hash=sha256(foldpath)
    folds=pd.read_csv(foldpath).rename(columns={'mold':'product'})
    foldcols=sorted(c for c in folds if c.startswith('fold_shot_r'))
    if not foldcols: raise ValueError('Missing saved shot folds')
    rows=[];audit=[];input_hashes={}
    for product,files in cfg['products'].items():
        path=ROOT/cfg['data_dir']/files['labeled'];input_hashes[str(path.relative_to(ROOT))]=sha256(path)
        data=pd.read_csv(path).rename(columns={cfg['id_col']:'row_id',cfg['label_col']:'y'})
        original=[c for c in data if c not in ['row_id','y']]
        assert set(original)-{'Clamp_Open_Position'}==set(FEATURES)
        data=assign_shots(data,original,product);data['product']=product
        f=folds[folds['product'].eq(product)]
        data=data.merge(f[['row_id','shot_id',*foldcols]],on=['row_id','shot_id'],how='outer',validate='one_to_one',indicator=True)
        if not data._merge.eq('both').all(): raise ValueError('Saved population mismatch')
        for col in foldcols:
            if not data.groupby('shot_id')[col].nunique().eq(1).all(): raise ValueError('Shot split across folds')
            for fold in sorted(data[col].unique()):
                train=data[data[col]!=fold];query=data[data[col]==fold]
                scores,params=score_fold(train,query)
                # Metadata attached only AFTER computing scores.
                truth=query.set_index('row_id');counts=data.groupby('shot_id').y.agg(['sum','size'])
                types=np.select([counts['sum'].eq(0),counts['sum'].eq(counts['size'])],['normal','both-side'],default='one-side')
                type_map=dict(zip(counts.index,types))
                scores['y_true']=scores.row_id.map(truth.y);scores['shot_type']=scores.shot_id.map(type_map)
                scores['repeat']=int(col.split('_r')[-1]);scores['fold']=int(fold);scores['cv_mode']='shot';rows.append(scores)
                audit.append(dict(product=product,column=col,fold=int(fold),query_shot_ids=query.shot_id.unique().tolist(),**params))
    out=ROOT/'outputs/analysis';out.mkdir(parents=True,exist_ok=True)
    scores=pd.concat(rows,ignore_index=True);scores.to_csv(out/'ambiguity_scores.csv',index=False)
    keys=['product','row_id','shot_id','y_true','shot_type'];metrics=['nearest_normal_distance','nearest_defect_distance','local_defect_ratio','ambiguity_score']
    agg=scores.groupby(keys)[metrics].agg(['mean','median']);agg.columns=['_'.join(c) for c in agg.columns];agg=agg.reset_index()
    agg['n_repeats']=scores.groupby(keys).size().to_numpy();assert agg.n_repeats.eq(len(foldcols)).all()
    agg.to_csv(out/'ambiguity_row_summary.csv',index=False)
    summaries=[]
    for product,g in agg.groupby('product'):
        for name,part in [('row_normal',g[g.y_true.eq(0)]),('row_defect',g[g.y_true.eq(1)]),
                          *[(typ,g[g.shot_type.eq(typ)].drop_duplicates('shot_id')) for typ in ['normal','one-side','both-side']]]:
            v=part.ambiguity_score_mean
            summaries.append(dict(product=product,group=name,unit='row' if name.startswith('row_') else 'shot',n=len(v),
                mean=v.mean(),median=v.median(),q25=v.quantile(.25),q75=v.quantile(.75),min=v.min(),max=v.max(),fraction_ge_08=v.ge(.8).mean()))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'ambiguity_summary.csv',index=False)
    top=agg.drop_duplicates(['product','shot_id']).sort_values(['product','ambiguity_score_mean','shot_id'],ascending=[True,False,True]).groupby('product').head(20).drop(columns='y_true')
    top.to_csv(out/'ambiguity_top_cases.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for i,p in enumerate(['cn7','rg3']):
        g=agg[agg['product']==p]
        for label in [0,1]:
            v=np.sort(g[g.y_true.eq(label)].ambiguity_score_mean);axes[i,0].step(v,np.arange(1,len(v)+1)/len(v),label='normal row' if label==0 else 'defect row')
        for typ in ['normal','one-side','both-side']:
            v=np.sort(g[g.shot_type.eq(typ)].drop_duplicates('shot_id').ambiguity_score_mean)
            if len(v): axes[i,1].step(v,np.arange(1,len(v)+1)/len(v),label=f'{typ} (n={len(v)})')
        for ax in axes[i]:ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean OOF ambiguity (5 repeats)',ylabel='Empirical CDF',title=p.upper());ax.legend(fontsize=8)
    fig.tight_layout();eda=ROOT/'outputs/eda';eda.mkdir(exist_ok=True);fig.savefig(eda/'ambiguity_distribution.png',dpi=170);plt.close(fig)
    assert sha256(foldpath)==fold_hash
    meta=dict(features=FEATURES,fold_sha256=fold_hash,input_sha256=input_hashes,fold_columns=foldcols,k=10,
        definition='2*min(d_normal,d_defect)/(d_normal+d_defect); both zero -> 1; missing class -> NaN',
        scaling='StandardScaler fit on unique TRAIN shots only',class_reference='row labels; a mixed-label training shot belongs to both class reference sets',
        local_ratio='Mean train-shot defect-row fraction across 10 nearest unique shots, all distance-boundary ties included',
        caveats=['Relative equidistance is not calibrated uncertainty or defect probability','One-side reference shots occur in both classes and can force score=1',
                 'Rare defect reference density differs; large distance to both classes can also yield high ambiguity','Repeat results are dependent'],
        nearest_classes_same_shot_fraction=float(scores.nearest_classes_same_shot.mean()),fold_audit=audit)
    (out/'ambiguity_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print('same reference shot fraction',meta['nearest_classes_same_shot_fraction'])


if __name__=='__main__':main()

