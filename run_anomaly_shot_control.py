"""Matched IF/SPC/Mahalanobis normal-shot control, preserving all prior files."""
import json
from pathlib import Path
import subprocess
import importlib.metadata
import numpy as np
import pandas as pd
from evaluation import save_experiment_result
from src.data import assign_shots,load_all,sha256
from src.features import select_base_features,model_features,check_feature_list
from src.anomaly_shot_control import run_control,MODELS
from run_mahalanobis import baseline_predictions,table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/experiments/anomaly_shot_control'


def hashes():
    files={ROOT/p for p in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if (ROOT/p).is_file()}
    files.update(p for d in ('outputs','data/raw') for p in (ROOT/d).rglob('*') if p.is_file() and OUT not in p.parents)
    files.update(p for d in ('src','tests') for p in (ROOT/d).glob('*.py'))
    files.update(ROOT.glob('*.py'))
    return {str(p.relative_to(ROOT)):sha256(p) for p in files}


def save(df,name):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,encoding='utf-8-sig')


def analyze(pred,shots):
    groups=pred.groupby(['product','repeat','model','shot_id'])
    assert groups.fold.nunique().eq(1).all() and groups.y_score.nunique().eq(1).all()
    ranks=groups.agg(y=('y_true','max'),score=('y_score','max')).reset_index()
    ranks=ranks.merge(shots[['product','shot_id','defect_type']],on=['product','shot_id'],validate='many_to_one')
    ranked=[]
    for _,f in ranks.groupby(['product','repeat','model']):
        f=f.copy();s=f.score
        f['rank']=s.rank(ascending=False,method='average')
        f['rank_min']=s.rank(ascending=False,method='min');f['rank_max']=s.rank(ascending=False,method='max')
        for k in (5,10,20):
            f[f'top{k}_weight']=((int(np.ceil(len(f)*k/100))-f.rank_min+1)/(f.rank_max-f.rank_min+1)).clip(0,1)
        ranked.append(f)
    ranks=pd.concat(ranked,ignore_index=True);defects=ranks[ranks.y.eq(1)]
    save(ranks,'all_shot_ranks.csv');save(defects,'defect_ranks_long.csv')
    wide=defects.pivot(index=['product','repeat','shot_id','defect_type'],columns='model',values=['rank','top5_weight','top10_weight','top20_weight'])
    wide.columns=[f'{m}_{v}' for v,m in wide.columns];save(wide.reset_index(),'defect_shot_ranks.csv')
    types=[]
    for (p,m,t),f in defects.groupby(['product','model','defect_type']):
        types.append(dict(product=p,model=m,defect_type=t,n_defects=f.shot_id.nunique(),rank_mean=f['rank'].mean(),rank_median=f['rank'].median(),
                          **{f'recall{k}':f[f'top{k}_weight'].mean() for k in (5,10,20)}))
    types=pd.DataFrame(types);save(types,'defect_type_summary.csv')
    exclusive=[];overlap=[]
    for scale in ('ecdf','raw'):
        subset=defects[defects.model.isin([f'{m}_shot_{scale}' for m in MODELS])]
        for (p,r),f in subset.groupby(['product','repeat']):
            for k in (5,10,20):
                w=f.pivot(index='shot_id',columns='model',values=f'top{k}_weight')
                for m in MODELS:
                    own=w[f'{m}_shot_{scale}'];others=w.drop(columns=f'{m}_shot_{scale}')
                    for sid in w.index:
                        if own.loc[sid]>0 and others.loc[sid].lt(1).all():
                            exclusive.append(dict(product=p,repeat=r,score_scale=scale,budget_pct=k,model=m,shot_id=sid,
                                definite=bool(own.loc[sid]==1 and others.loc[sid].eq(0).all()),
                                own_weight=own.loc[sid],other_max_weight=others.loc[sid].max()))
                for a,b in [('iforest','spc'),('iforest','mahalanobis'),('spc','mahalanobis')]:
                    x,y=w[f'{a}_shot_{scale}'],w[f'{b}_shot_{scale}']
                    overlap.append(dict(product=p,repeat=r,score_scale=scale,budget_pct=k,model_a=a,model_b=b,
                        both=float((x*y).sum()),a_only=float((x*(1-y)).sum()),b_only=float((y*(1-x)).sum()),neither=float(((1-x)*(1-y)).sum())))
    ex=pd.DataFrame(exclusive,columns=['product','repeat','score_scale','budget_pct','model','shot_id','definite','own_weight','other_max_weight'])
    save(ex,'exclusive_defect_detections.csv')
    o=pd.DataFrame(overlap);save(o,'overlap_by_repeat.csv')
    save(o.groupby(['product','score_scale','budget_pct','model_a','model_b'])[['both','a_only','b_only','neither']].mean().reset_index(),'overlap_summary.csv')
    focus=defects[defects.shot_id.isin(['cn7_55','cn7_56','cn7_57'])]
    save(focus,'cn7_55_56_57_by_repeat.csv')
    return types,ex


def main():
    before=hashes();cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    if cfg['cv']['n_repeats']!=5 or cfg['cv']['n_splits']!=3 or cfg['holdout']['enabled']:raise ValueError('Requires unchanged saved folds')
    if any(cfg['features'].get(k,False) for k in ('sequence_features','unlabeled_iforest_score','pair_position')):raise ValueError('No extra features')
    data=load_all(cfg,ROOT/cfg['data_dir'])
    if data['hashes']!=json.loads((ROOT/'outputs/manifest.json').read_text(encoding='utf-8'))['input_sha256']:raise ValueError('Changed baseline data')
    base,_=select_base_features(data,data['feature_cols'],cfg['features']['drop_constant'])
    features=check_feature_list(model_features(base,'separate',False));assert len(features)==23
    lab=pd.concat([assign_shots(f,data['feature_cols'],p) for p,f in data['labeled'].items()],ignore_index=True)
    folds=pd.read_csv(ROOT/'outputs/fold_id.csv')
    references=baseline_predictions(folds,lab)
    references=references[references.model.isin(['iforest_normal','spc_maxz'])].copy()
    references.model=references.model+'_row_reference'
    pred,shots,audit,ids=run_control(lab,folds,features,cfg)
    assert audit.groupby(['product','repeat','fold']).normal_shot_ids_sha256.nunique().eq(1).all()
    save(audit,'fit_audit.csv');(OUT/'fit_shot_ids.json').write_text(json.dumps(ids,indent=2),encoding='utf-8')
    combined=pd.concat([pred,references],ignore_index=True)
    files=[]
    for m,f in combined.groupby('model'):
        save(f,f'predictions/{m}.csv');files.append(OUT/f'predictions/{m}.csv')
    save(lab[['product','row_id','shot_id']],'predictions_labeled_cv.csv')
    summary=save_experiment_result(files=files,out_dir=OUT/'evaluation',levels=('shot',),
        shot_map=OUT/'predictions_labeled_cv.csv',ks=(.05,.1,.2),verbose=False)
    summary['comparison_role']=np.select([summary.model.str.endswith('_shot_ecdf'),summary.model.str.endswith('_shot_raw')],['primary_common_ecdf','sensitivity_raw'],default='reference_product_row')
    save(summary,'comparison_all.csv')
    primary=summary[summary.comparison_role.eq('primary_common_ecdf')];save(primary,'comparison_primary.csv')
    raw=summary[summary.comparison_role.eq('sensitivity_raw')];save(raw,'comparison_raw.csv')
    ref=summary[summary.comparison_role.eq('reference_product_row')];save(ref,'comparison_row_reference.csv')
    types,exclusive=analyze(combined,shots)
    prior=pd.read_csv(ROOT/'outputs/experiments/mahalanobis/predictions/mahalanobis_lw.csv')
    fresh=pred[pred.model.eq('mahalanobis_shot_raw')]
    keys=['product','repeat','fold','row_id','shot_id','y_true']
    check=fresh.merge(prior,on=keys,how='outer',validate='one_to_one',indicator=True,suffixes=('_control','_prior'))
    assert check._merge.eq('both').all()
    diff=float((check.y_score_control-check.y_score_prior).abs().max());assert diff<1e-12
    assert before==hashes()
    manifest=dict(features=features,fold_sha256=sha256(ROOT/'outputs/fold_id.csv'),normal='unique shots with max(y)==0',
        fit_sets_identical=True,fit_count=90,no_new_splits=True,primary_score='common train-normal ECDF',sensitivity_score='raw native score',
        training_normalization='All fitted statistics use exactly the same normal training shots; algorithm-internal scaling retained',
        previous_mahalanobis_raw_max_abs_diff=diff,protected_files_unchanged=True,protected_files_sha256=before,
        versions={k:importlib.metadata.version(k) for k in ('numpy','pandas','scikit-learn')})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    cols=['product','model','pr_auc_mean','recall@5%_mean','recall@10%_mean','recall@20%_mean']
    lines=['# 정상 shot 학습 control ablation','',
        '3-fold × 5 repeat의 기존 block fold, 기존 23개 feature, shot당 1행, y_shot=0인 정상 train shot을 세 모델에 동일하게 전달했다. 학습·score 보정에 validation을 사용하지 않았다.',
        'IF는 baseline과 같은 300 trees/seed=42, SPC는 같은 sd_floor=0.01이다. Mahalanobis는 기존 StandardScaler+LedoitWolf를 재사용했다. 전처리는 각 방법 고유의 정의를 유지하되 모든 통계의 학습 모집단은 같다.',
        '주 비교는 세 모델 모두 train 정상 raw score로 동일 ecdf_score 함수를 fit/적용한다. raw 결과도 사전에 정한 sensitivity로 함께 보고한다. 기준은 train 내부이며 validation score로 정규화하지 않는다. ECDF는 fold 내부 순서를 보존하지만 fold 사이 상대 순서를 바꿀 수 있다.', '',
        '## Primary: 공통 ECDF (repeat 평균)','',table(primary,cols),'','## Sensitivity: 원점수','',table(raw,cols),'',
        '## Reference: 기존 product-row IF/SPC','',table(ref,cols),'',
        'reference와 shot ECDF 간 비교는 모델·보정법을 유지하면서 정상 학습 단위를 바꾼 것이다. 중복 제거와 one-side 정상 짝 제외 효과가 함께 들어가며 둘을 별도로 분리한 실험은 아니다.', '',
        '## CN7 유형별 primary','',table(types[types['product'].eq('cn7')&types.model.str.endswith('_shot_ecdf')],['defect_type','model','rank_mean','rank_median','recall5','recall10','recall20']),'',
        '## 파일과 해석 범위','',
        '- defect_shot_ranks.csv: 모든 불량 shot의 모델별 순위 및 검사 예산 포함 비율. raw·ECDF·row reference 구분.',
        '- exclusive_defect_detections.csv: 나머지 두 shot 모델이 놓치고 한 모델만 잡는 관측. definite=False는 경계 동점 후보.',
        '- cn7_55_56_57_by_repeat.csv: 세 관심 shot의 repeat별 순위 및 포함 여부.',
        '- overlap_summary.csv: 동일 예산 탐지 겹침. 양쪽 경계 동점은 독립 균등 해소의 기대값. 탐지 합집합을 성능 향상으로 해석하지 않는다.',
        '- fit_audit.csv / fit_shot_ids.json: 모델별 동일 학습 정상 shot ID 해시 및 전체 목록.',
        f'- 이전 Mahalanobis raw 최대 차이 {diff:.3g}. 기존 파일 {len(before)}개 hash 보존.',
        '동일 shot 정상 학습에서 남는 차이는 모델 방식과 그 고유 전처리를 포함한다. 작은 불량 표본과 반복 CV의 재사용 때문에 일반화 성능이나 인과 효과를 확정하지 않는다.']
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PRIMARY');print(table(primary,cols));print('RAW');print(table(raw,cols));print('TYPES');print(types.to_string(index=False))


if __name__=='__main__':main()
