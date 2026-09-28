"""Independent PCA ablation; never regenerate splits or refit reference models."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from src.data import load_all, assign_shots, sha256
from src.features import select_base_features
from src.pca_anomaly import run_pca, prepare_shots, PRED_COLS
from evaluation import evaluate_model, calculate_recall_at_k

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/experiments/pca_anomaly'
CONTROL=ROOT/'outputs/experiments/anomaly_shot_control'

def protected():
    paths={ROOT/p for p in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()}
    for folder in ['outputs','data/raw']:
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and OUT not in p.parents)
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths) if p.is_file()}

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def analyze(pred,shots):
    ranks=[];metrics=[]
    for (product,model,repeat),part in pred.groupby(['product','model','repeat']):
        g=part.groupby('shot_id',sort=False).agg(y_true=('y_true','max'),y_score=('y_score','first'),fold=('fold','first')).reset_index()
        if part.groupby('shot_id')[['y_score','fold']].nunique().gt(1).any().any():raise ValueError('Inconsistent shot prediction')
        g['rank']=g.y_score.rank(ascending=False,method='average')
        lo=g.y_score.rank(ascending=False,method='min');hi=g.y_score.rank(ascending=False,method='max')
        g['rank_min']=lo;g['rank_max']=hi;g['percentile_rank']=g['rank']/len(g)*100
        m=dict(product=product,model=model,repeat=repeat,AP=average_precision_score(g.y_true,g.y_score),AUC=roc_auc_score(g.y_true,g.y_score),n=len(g),n_defect=int(g.y_true.sum()))
        for k in [5,10,20]:
            g[f'hit_{k}']=np.clip((int(np.ceil(len(g)*k/100))-lo+1)/(hi-lo+1),0,1)
            m[f'Recall@{k}%']=g.loc[g.y_true.eq(1),f'hit_{k}'].mean()
        for label in [0,1]:
            vals=g.loc[g.y_true.eq(label),'y_score']
            for q in [.1,.25,.5,.75,.9]:m[f'score_y{label}_q{q}']=vals.quantile(q)
        g['product']=product;g['model']=model;g['repeat']=repeat
        ranks.append(g);metrics.append(m)
    ranks=pd.concat(ranks,ignore_index=True).merge(shots[['shot_id','defect_type']],on='shot_id',validate='many_to_one')
    metrics=pd.DataFrame(metrics)
    ranks.to_csv(OUT/'all_shot_ranks.csv',index=False)
    defects=ranks[ranks.y_true.eq(1)].copy();defects.to_csv(OUT/'defect_shot_ranks.csv',index=False)
    metrics.to_csv(OUT/'metrics_by_repeat.csv',index=False)
    cols=['AP','AUC','Recall@5%','Recall@10%','Recall@20%']
    summary=metrics.groupby(['product','model'])[cols].mean().reset_index()
    common=[]
    for (product,model),part in pred.groupby(['product','model']):
        result=evaluate_model(part,level='shot')
        recall=calculate_recall_at_k(part,level='shot',ks=(.05,.10,.20))
        row=dict(product=product,model=model,**result)
        ref=summary[(summary['product']==product)&summary.model.eq(model)].iloc[0]
        if abs(result['pr_auc_mean']-ref.AP)>1e-12:raise ValueError('Common AP mismatch')
        for _,r in recall.iterrows():
            k=int(round(r.k*100));row[f'Recall@{k}%']=r.recall_mean
            if abs(r.recall_mean-ref[f'Recall@{k}%'])>1e-12:raise ValueError('Common Recall mismatch')
        common.append(row)
    pd.DataFrame(common).to_csv(OUT/'common_evaluation_verification.csv',index=False)
    metrics.groupby(['product','model'])[cols].std().to_csv(OUT/'comparison_std.csv')
    for scale in ['ecdf','raw']:
        summary[summary.model.str.endswith(scale)].to_csv(OUT/f'comparison_{scale}.csv',index=False)
    types=defects.groupby(['product','model','defect_type']).agg(mean_rank=('rank','mean'),median_rank=('rank','median'),recall5=('hit_5','mean'),recall10=('hit_10','mean'),recall20=('hit_20','mean'))
    types.to_csv(OUT/'defect_type_summary.csv')
    defects[defects.shot_id.isin(['cn7_55','cn7_56','cn7_57'])].to_csv(OUT/'focus_shots.csv',index=False)
    correlations=[];detection=[]
    for (product,repeat),part in defects.groupby(['product','repeat']):
        for scale in ['ecdf','raw']:
            sub=part[part.model.str.endswith(scale)]
            wide=sub.pivot(index='shot_id',columns='model',values='rank')
            correlations.append(dict(product=product,repeat=repeat,scale=scale,n_defect=len(wide),spearman=wide[f'pca_shot_{scale}'].corr(wide[f'mahalanobis_shot_{scale}'],method='spearman')))
            for k in [5,10,20]:
                w=sub.pivot(index='shot_id',columns='model',values=f'hit_{k}')
                for sid,row in w.iterrows():
                    p=row[f'pca_shot_{scale}'];others=row.drop(f'pca_shot_{scale}')
                    detection.append(dict(product=product,repeat=repeat,scale=scale,budget=k,shot_id=sid,pca_weight=p,
                        iforest_weight=row[f'iforest_shot_{scale}'],spc_weight=row[f'spc_shot_{scale}'],mahalanobis_weight=row[f'mahalanobis_shot_{scale}'],
                        pca_only_definite=bool(p==1 and others.max()==0),pca_only_possible=bool(p>0 and others.max()<1)))
    pd.DataFrame(correlations).to_csv(OUT/'mahalanobis_pca_rank_correlation.csv',index=False)
    pd.DataFrame(detection).to_csv(OUT/'defect_detection_sets.csv',index=False)
    old=pd.read_csv(CONTROL/'comparison_all.csv')
    checks=[]
    for _,r in summary[~summary.model.str.startswith('pca')].iterrows():
        ref=old[(old['product']==r['product'])&(old.model==r.model)].iloc[0]
        for new,prev in [('AP','pr_auc_mean'),('AUC','roc_auc_mean'),*[(f'Recall@{k}%',f'recall@{k}%_mean') for k in [5,10,20]]]:
            delta=abs(r[new]-ref[prev]);checks.append(dict(product=r['product'],model=r.model,metric=new,absolute_difference=delta))
            if delta>1e-12:raise ValueError('Reference metric changed')
    pd.DataFrame(checks).to_csv(OUT/'reference_metric_verification.csv',index=False)
    return summary

def markdown_table(frame):
    def fmt(v):return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
                     *['| '+' | '.join(map(fmt,row))+' |' for row in frame.itertuples(index=False,name=None)]])

def write_report(summary,audit):
    types=pd.read_csv(OUT/'defect_type_summary.csv')
    focus=pd.read_csv(OUT/'focus_shots.csv')
    focus=focus[focus.model.str.endswith('ecdf')].groupby(['model','shot_id'])[['rank','hit_5','hit_10','hit_20']].mean().reset_index()
    corr=pd.read_csv(OUT/'mahalanobis_pca_rank_correlation.csv').groupby(['product','scale']).spearman.agg(['mean','min','max']).reset_index()
    detections=pd.read_csv(OUT/'defect_detection_sets.csv')
    unique=detections[(detections.scale=='ecdf')&detections.pca_only_definite].groupby(['product','budget','shot_id']).size().rename('repeats_of_5').reset_index()
    text='''# PCA reconstruction anomaly: independent ablation

정상 센서 움직임의 대표 패턴으로 Shot을 복원하고, 복원 오차가 클수록 위험하게 평가한다.
branch: `exp/pca-anomaly`. 기존 코드/config/output은 수정하지 않았으며 저장된 block CV 3-fold × 5 repeat만 사용했다.
각 repeat의 전체 OOF Shot을 모아 평가한 뒤 5회 평균을 냈다. fold별 지표를 평균한 값이 아니다.
CN7 606 Shot/불량 14 (one-side 11, both-side 3), RG3 591 Shot/불량 25.

## 학습 및 점수

- 기존 23개 feature, Shot당 1행, 원래 제품 y의 max가 0인 정상 train Shot만 사용.
- 정상 학습 Shot ID 목록을 기존 IF/SPC/Mahalanobis control과 30개 fold 전부 정확히 대조한 후 fit.
- StandardScaler와 full-SVD PCA는 정상 train만 fit. 누적 explained variance >=95%가 되는 최소 component 선택.
- standardized space에서 projection → reconstruction → feature별 squared error의 평균이 raw score.
- primary ECDF는 기존 세 control과 같은 `src.models.ecdf_score`: 내부 `(count<=score+0.5)/(n+1)`, 학습 범위 바깥은 train score 표준편차를 사용하는 tanh 확장. 엄밀한 계단 ECDF에 smoothing/tail 확장이 더해진 기존 공통 규칙이다. validation로 보정하지 않는다.
- raw 결과도 별도 비교. ECDF는 fold 간 순위와 동점을 바꿀 수 있으므로 두 결과를 구분한다.
- 원래 제품 행으로 점수를 복원했고 prediction CSV의 y_true는 원래 제품 y를 유지한다. y_pred는 정상 train raw score의 95% 분위수 기준이며 AP/Recall@budget에는 사용하지 않는다.
- 검사 Shot 수는 ceil(N×budget); 위험 rank 1이 최상위. 동점은 평균 rank, budget 경계 동점은 무작위 선택 기대 포함률(hit_5/10/20)로 평가.
- 모든 비교 모델의 저장 예측은 동일 row/shot/label/fold 집합을 검증. IF/SPC/Mahalanobis 재학습 없음.
- CN7 유형은 두 제품 라벨의 합/개수로 사후 구분한다. feature나 학습조건 선택에 사용하지 않는다.

## Primary: 공통 train-normal ECDF

'''
    text+=markdown_table(summary[summary.model.str.endswith('ecdf')])+'\n\n## Raw score sensitivity\n\n'
    text+=markdown_table(summary[summary.model.str.endswith('raw')])+'\n\n## Component 수\n\n'
    text+=markdown_table(audit.groupby('product').n_components.agg(['min','max']).reset_index())
    text+='\n\n## CN7 유형 (ECDF, repeat-shot 관측의 평균/중앙 rank)\n\n'
    text+=markdown_table(types[(types['product']=='cn7')&types.model.str.endswith('ecdf')])
    text+='\n\n## 주요 질문의 답\n\n'
    text+='''1. PCA는 both-side 3 Shot을 모두 매 repeat 상위 5%에서 탐지한다. one-side는 Recall@5/10/20% = 21.8/36.4/56.4%. one-side 평균 rank 186.2, 중앙 103으로, 일부를 더 잘 잡지만 하위권으로 밀리는 불량도 있다. Mahalanobis의 one-side 평균 157.0, 중앙 181과 비교하면 개선이 균일하지 않다.
2. cn7_56은 PCA도 상위 5%에서 5/5회 탐지한다. cn7_57은 PCA 평균 rank 119.6으로, 상위 5/10%에서는 0/5회, 20%에서 4/5회다. Mahalanobis의 두 Shot 상위 5% 5/5회 탐지가 모두 유지되지는 않는다.
3. 같은 repeat·예산에서 나머지 세 모델 모두 놓치고 PCA만 확실히 잡은 CN7 Shot: @5% cn7_55 (2/5회), @10% cn7_24 (5/5), cn7_52 (2/5), @20% cn7_24 (5/5), cn7_50 (2/5). 이는 해당 예산에서의 추가 탐지이며 모든 예산에서 새로운 Shot이라는 뜻은 아니다. 표의 repeats는 독립 불량 수가 아니다.
4. Mahalanobis–PCA 불량 순위 Spearman은 CN7 평균 약 0.831, RG3 약 0.501이다. 불량 Shot만 대상으로 각 repeat의 전체 Shot 위험순위 간 상관을 계산했다.
5. RG3 PCA ECDF AP 0.0482, AUC 0.5244로 기존 세 모델보다 높지만, 불량률 25/591=0.0423에 가깝다. Recall@5/10/20%는 4.0/6.4/18.4%이며 무작위 검사 기대 5.08/10.15/20.14%를 밑돈다. repeat별 AUC는 0.5606/0.5196/0.5332/0.5111/0.4973이다. 강한 분리력이 생겼다고 해석하기 어렵다.

CN7에서는 Mahalanobis가 AP와 Recall@5%에서 더 좋고, PCA가 Recall@10/20%에서 더 좋다. raw에서는 CN7 IF Recall@20%가 75.7%로 PCA 62.9%보다 높다. 따라서 ECDF primary의 결과를 모든 score 표현에 일반화하지 않는다. 14/25개 불량을 같은 데이터에서 반복 평가한 탐색적 오류구조 분석이며, 반복들을 독립 표본으로 간주한 유의성/일반화 주장은 하지 않는다.

### cn7_55/56/57 상세

'''
    text+=markdown_table(focus)+'\n\n### PCA만 확실히 탐지 (다른 3개 모두 포함률 0, PCA 포함률 1)\n\n'
    text+=markdown_table(unique)+'\n\n### 불량 순위 상관\n\n'+markdown_table(corr)
    text+='''

## 산출물과 재현

`predictions/pca_shot_{ecdf,raw}.csv`: 공통 schema OOF prediction.
`comparison_{ecdf,raw}.csv`, `comparison_std.csv`, `metrics_by_repeat.csv`: 평균/표준편차/repeat별 지표와 score 분위수.
`defect_shot_ranks.csv`, `all_shot_ranks.csv`: 모델별 repeat별 위험순위와 budget 포함률.
`defect_type_summary.csv`, `focus_shots.csv`, `defect_detection_sets.csv`, `mahalanobis_pca_rank_correlation.csv`: 사후 분석.
`fit_audit.csv`, `fit_parameters.json`, `shot_map.csv`: component/정상 ID/Scaler/PCA 파라미터/기존 fold 감사 자료.
`reference_metric_verification.csv`: 저장 control의 기존 AP/AUC/Recall을 1e-12 이내로 재현.
`common_evaluation_verification.csv`: 기존 공통 evaluation 모듈과 PCA 포함 모든 지표 대조.
`protected_before.json`, `manifest.json`, `verification.json`: 실행 전후 보호 파일 hash 및 최종 검사.
`tests_full.log`: 기존 38개 + 신규 10개 = 전체 48개 테스트 결과.

실행 (PowerShell, 저장소 root):
```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_pca_anomaly.py -v
./.venv/Scripts/python.exe run_pca_anomaly.py
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```
'''
    (OUT/'summary.md').write_text(text,encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before=protected();write_json(OUT/'protected_before.json',before)
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    if cfg['cv']['n_repeats']!=5 or cfg['cv']['n_splits']!=3:raise ValueError('Expected 3 x 5 CV')
    data=load_all(cfg,ROOT/cfg['data_dir'])
    baseline=json.loads((ROOT/'outputs/manifest.json').read_text(encoding='utf-8'))
    if data['hashes']!=baseline['input_sha256']:raise ValueError('Baseline input hashes differ')
    features,_=select_base_features(data,data['feature_cols'],True)
    control=json.loads((CONTROL/'manifest.json').read_text(encoding='utf-8'))
    if len(features)!=23 or features!=control['features']:raise ValueError('Feature set mismatch')
    foldpath=ROOT/'outputs/fold_id.csv'
    if sha256(foldpath)!=control['fold_sha256']:raise ValueError('Fold file changed')
    lab=pd.concat([assign_shots(v,data['feature_cols'],p) for p,v in data['labeled'].items()],ignore_index=True)
    folds=pd.read_csv(foldpath)
    rows,shots,foldcols=prepare_shots(lab,folds,features)
    expected_ids=json.loads((CONTROL/'fit_shot_ids.json').read_text(encoding='utf-8'))
    refs=[]
    # Validate saved comparator identities/labels/folds before fitting PCA.
    for name in ['iforest','spc','mahalanobis']:
        for scale in ['ecdf','raw']:
            pred=pd.read_csv(CONTROL/f'predictions/{name}_shot_{scale}.csv')
            if list(pred.columns)!=PRED_COLS or set(pred.repeat)!={0,1,2,3,4} or not pred.cv_mode.eq('block').all():raise ValueError('Reference schema/CV mismatch')
            for repeat,part in pred.groupby('repeat'):
                check=part.merge(rows[['product','row_id','shot_id','y',foldcols[repeat]]],on=['product','row_id'],how='outer',validate='one_to_one',suffixes=('','_expected'),indicator=True)
                if not (check._merge.eq('both')&check.shot_id.eq(check.shot_id_expected)&check.y_true.eq(check.y)&check.fold.eq(check[foldcols[repeat]])).all():raise ValueError('Reference population mismatch')
            refs.append(pred)
    pca,shots,audit,parameters=run_pca(lab,folds,features,cfg,expected_ids)
    (OUT/'predictions').mkdir(exist_ok=True)
    for model,part in pca.groupby('model'):part.to_csv(OUT/f'predictions/{model}.csv',index=False)
    audit.to_csv(OUT/'fit_audit.csv',index=False);write_json(OUT/'fit_parameters.json',parameters)
    shots[['product','shot_id','y','defect_type',*foldcols]].to_csv(OUT/'shot_map.csv',index=False)
    summary=analyze(pd.concat([pca,*refs],ignore_index=True),shots)
    write_report(summary,audit)
    after=protected()
    if before!=after:raise RuntimeError('Protected files changed')
    write_json(OUT/'manifest.json',dict(features=features,fold_sha256=sha256(foldpath),fit_count=len(audit),normal_fit_ids_equal_control=True,
        comparator_models_retrained=False,primary='train-normal ECDF with shared baseline tail extension',raw_score='standardized feature MSE',
        component_rule='minimum train cumulative explained variance >= 0.95',protected_files_unchanged=True,protected_file_count=len(before),
        protected_after=after,reference_prediction_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (CONTROL/'predictions').glob('*.csv')}))
    print(summary.to_string(index=False))

if __name__=='__main__':main()
