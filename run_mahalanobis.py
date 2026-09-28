"""Independent Mahalanobis experiment. Existing baseline files are read-only."""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from evaluation.metrics import evaluate_model
from evaluation.recall_at_k import calculate_recall_at_k
from src.data import load_all, assign_shots, sha256
from src.features import select_base_features, model_features, check_feature_list
from src.mahalanobis import PRED_COLS, run_oof

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/experiments/mahalanobis'
BASELINES = ('logreg','iforest_normal','spc_maxz')


def protected_hashes():
    names = subprocess.check_output(['git','ls-files'], cwd=ROOT, text=True).splitlines()
    paths = {ROOT/n for n in names if (ROOT/n).is_file()}
    paths.update(p for folder in ('outputs','data/raw') for p in (ROOT/folder).rglob('*')
                 if p.is_file() and OUT not in p.parents)
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def save(frame, name):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def table(frame, cols):
    lines = ['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for _,r in frame.iterrows():
        lines.append('| '+' | '.join(f'{r[c]:.4f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols)+' |')
    return '\n'.join(lines)


def baseline_predictions(folds, lab):
    """Validate saved baselines before fitting the new estimator."""
    result=[]
    truth=lab[['product','row_id','shot_id','y']].rename(columns={'y':'y_true'})
    for name in BASELINES:
        f=pd.read_csv(ROOT/f'outputs/predictions/pipeline_{name}.csv')
        f=f[f.cv_mode.eq('block')].copy()
        if set(f.repeat)!={0,1,2,3,4} or set(f.model)!={name}:
            raise ValueError(f'Unexpected saved baseline: {name}')
        for rep,part in f.groupby('repeat'):
            check=part.merge(truth,on=['product','row_id','shot_id','y_true'],how='outer',validate='one_to_one',indicator=True)
            if not check._merge.eq('both').all():
                raise ValueError('Saved baseline labels/rows differ')
            check=part.merge(folds.rename(columns={'mold':'product'}),on=['product','row_id','shot_id'],validate='one_to_one')
            if len(check)!=len(lab) or not check.fold.eq(check[f'fold_block_r{rep}']).all():
                raise ValueError('Saved baseline fold mismatch')
        result.append(f[PRED_COLS])
    return pd.concat(result,ignore_index=True)


def analyze(predictions, shots):
    grouped=predictions.groupby(['product','model','repeat','shot_id'])
    if not grouped.fold.nunique().eq(1).all() or not grouped.y_score.nunique().eq(1).all():
        raise ValueError('Inconsistent predictions within shot')
    s=grouped.agg(y_true=('y_true','max'),y_score=('y_score','max'),fold=('fold','first')).reset_index()
    s=s.merge(shots[['product','shot_id','defect_type']],on=['product','shot_id'],validate='many_to_one')
    metric_rows=[]; rank_frames=[]
    for (product,model,rep),f in s.groupby(['product','model','repeat']):
        f=f.copy()
        ev=evaluate_model(f,level='row',run_col=None)
        recall=calculate_recall_at_k(f,ks=(.05,.1,.2),level='row',run_col=None).set_index('k_label')
        metric_rows.append(dict(product=product,model=model,repeat=rep,n_shots=len(f),n_defects=int(f.y_true.sum()),
            ap=ev['pr_auc_mean'],auc=ev['roc_auc_mean'],**{f'recall{k}':recall.loc[f'{k}%','recall_mean'] for k in (5,10,20)}))
        f['rank']=f.y_score.rank(ascending=False,method='average')
        f['rank_min']=f.y_score.rank(ascending=False,method='min')
        f['rank_max']=f.y_score.rank(ascending=False,method='max')
        f['top_percent']=100*f['rank']/len(f)
        for k in (5,10,20):
            w=((math.ceil(len(f)*k/100)-f.rank_min+1)/(f.rank_max-f.rank_min+1)).clip(0,1)
            f[f'top{k}_weight']=w
            f[f'top{k}_status']=np.select([w.eq(0),w.eq(1)],['excluded','included'],default='boundary_tie')
            assert abs(w.sum()-math.ceil(len(f)*k/100))<1e-10
        rank_frames.append(f)
    ranks=pd.concat(rank_frames,ignore_index=True)
    metrics=pd.DataFrame(metric_rows)
    cols=['ap','auc','recall5','recall10','recall20']
    g=metrics.groupby(['product','model'])
    summary=g[cols].mean().join(g[cols].std().add_suffix('_sd')).reset_index()
    defect=ranks[ranks.y_true.eq(1)].copy()
    index=['product','repeat','shot_id','defect_type']
    wide=defect.pivot(index=index,columns='model',values=['rank','y_score','top5_weight','top10_weight','top20_weight'])
    wide.columns=[f'{model}_{col}' for col,model in wide.columns]
    wide=wide.reset_index()
    for name in BASELINES:
        wide[f'mahal_rank_minus_{name}']=wide.mahalanobis_lw_rank-wide[f'{name}_rank']
    type_rows=[]
    for (product,typ,model),f in defect.groupby(['product','defect_type','model']):
        type_rows.append(dict(product=product,defect_type=typ,model=model,n_defects=f.shot_id.nunique(),
            rank_mean=f['rank'].mean(),rank_median=f['rank'].median(),
            **{f'recall{k}':f[f'top{k}_weight'].mean() for k in (5,10,20)}))
    types=pd.DataFrame(type_rows)
    overlaps=[]; novel=[]
    for (product,rep),f in wide.groupby(['product','repeat']):
        for k in (5,10,20):
            m=f[f'mahalanobis_lw_top{k}_weight']
            for base in BASELINES:
                b=f[f'{base}_top{k}_weight']
                # If both have boundary ties, independent random tie resolution
                # defines these expectations. No scores/rules are optimized.
                overlaps.append(dict(product=product,repeat=rep,budget_pct=k,baseline=base,n_defects=len(f),
                    both=float((m*b).sum()),baseline_only=float((b*(1-m)).sum()),mahal_only=float((m*(1-b)).sum()),
                    neither=float(((1-m)*(1-b)).sum()),
                    mahal_only_definite_ids=';'.join(f.loc[m.eq(1)&b.eq(0),'shot_id']),
                    baseline_only_definite_ids=';'.join(f.loc[b.eq(1)&m.eq(0),'shot_id'])))
            old=f[[f'{b}_top{k}_weight' for b in BASELINES]]
            for idx,r in f.iterrows():
                if m.loc[idx]>0 and old.loc[idx].lt(1).all():
                    novel.append(dict(product=product,repeat=rep,shot_id=r.shot_id,defect_type=r.defect_type,budget_pct=k,
                        mahal_rank=r.mahalanobis_lw_rank,logreg_rank=r.logreg_rank,iforest_rank=r.iforest_normal_rank,spc_rank=r.spc_maxz_rank,
                        definite=bool(m.loc[idx]==1 and old.loc[idx].eq(0).all())))
    overlap=pd.DataFrame(overlaps)
    overlap_summary=overlap.groupby(['product','budget_pct','baseline'])[['both','baseline_only','mahal_only','neither']].mean().reset_index()
    novel=pd.DataFrame(novel,columns=['product','repeat','shot_id','defect_type','budget_pct','mahal_rank','logreg_rank','iforest_rank','spc_rank','definite'])
    reference=pd.read_csv(ROOT/'outputs/tables/cv_summary.csv')
    errors=[]
    for _,r in summary[summary.model.isin(BASELINES)].iterrows():
        ref=reference[reference['product'].eq(r['product'])&reference.model.eq(r.model)&reference.cv_mode.eq('block')&reference.scenario.eq('separate')].iloc[0]
        for col,old in [('ap','shot_ap'),('auc','shot_roc_auc'),('recall5','shot_recall@5%'),('recall10','shot_recall@10%'),('recall20','shot_recall@20%')]:errors.append(abs(r[col]-ref[old]))
    assert max(errors)<1e-12
    for name,f in {'metrics_by_repeat.csv':metrics,'comparison.csv':summary,'all_shot_ranks.csv':ranks,
        'defect_shot_ranks.csv':wide,'defect_type_summary.csv':types,'detection_overlap_by_repeat.csv':overlap,
        'detection_overlap_summary.csv':overlap_summary,'new_defects_vs_all_baselines.csv':novel}.items():save(f,name)
    return summary,types,overlap_summary,novel,max(errors)


def main():
    before=protected_hashes()
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    if cfg['cv']['n_repeats']!=5 or cfg['cv']['n_splits']!=3 or cfg['holdout']['enabled']:
        raise ValueError('Requires saved baseline 3-fold x 5 repeat without holdout')
    if any(cfg['features'].get(k,False) for k in ('sequence_features','unlabeled_iforest_score','pair_position')):
        raise ValueError('Only unchanged baseline features are supported')
    data=load_all(cfg,ROOT/cfg['data_dir'])
    original=json.loads((ROOT/'outputs/manifest.json').read_text(encoding='utf-8'))
    if data['hashes']!=original['input_sha256']:
        raise ValueError('Data differ from saved baseline manifest')
    base,dropped=select_base_features(data,data['feature_cols'],cfg['features']['drop_constant'])
    feats=check_feature_list(model_features(base,'separate',False))
    if len(feats)!=23:raise ValueError('Expected exactly 23 unchanged baseline features')
    lab=pd.concat([assign_shots(f,data['feature_cols'],p) for p,f in data['labeled'].items()],ignore_index=True)
    folds=pd.read_csv(ROOT/'outputs/fold_id.csv')
    baseline=baseline_predictions(folds,lab)
    pred,shots,diag,params=run_oof(lab,folds,feats,cfg)
    OUT.mkdir(parents=True,exist_ok=True)
    all_pred=pd.concat([baseline,pred],ignore_index=True)
    for model,f in all_pred.groupby('model'):save(f,f'predictions/{model}.csv')
    save(lab[['product','row_id','shot_id']],'predictions_labeled_cv.csv')
    save(diag,'fold_diagnostics.csv')
    (OUT/'fit_parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    summary,types,overlap,novel,error=analyze(all_pred,shots)
    after=protected_hashes()
    if before!=after:raise RuntimeError('Protected baseline files changed')
    manifest=dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        features=feats,dropped_features=dropped,fold_sha256=sha256(ROOT/'outputs/fold_id.csv'),
        input_sha256=data['hashes'],baseline_files_unchanged=True,baseline_files_sha256=before,
        method='StandardScaler(train normal shots) -> LedoitWolf(train normal shots) -> sqrt Mahalanobis distance',
        normal_definition='max(original row y)==0; one row per shot',score='raw distance, higher is riskier; no ECDF or validation calibration',
        y_pred_rule='distance >= 95th percentile of training normal distances; not used for primary ranking metrics',
        comparisons='saved baseline OOF only; no retraining of Logistic, IF or SPC',baseline_metric_max_error=error,
        python=platform.python_version(),versions={k:importlib.metadata.version(k) for k in ('numpy','pandas','scikit-learn','scipy')},
        source_sha256={f:sha256(ROOT/f) for f in ('run_mahalanobis.py','src/mahalanobis.py','tests/test_mahalanobis.py')})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Mahalanobis 정상 shot 이상탐지 ablation','',
        '기존 block CV 3-fold × 5 repeat의 fold를 그대로 재사용했다. 기존 세 모델은 저장된 OOF이고 Mahalanobis만 새로 학습했다. '+
        'AP는 average precision, Recall은 ceil(N×예산) 검사 및 경계 동점 기대값이다. 모든 주요 지표는 repeat별 pooled OOF shot 기준의 5회 평균이다.', '',
        '## 학습·거리 정의','',
        '- 기존 23개 feature만 사용. 정상은 y_shot=max(y)=0인 shot으로 정의하고 shot당 한 행만 학습한다. one-side 정상 짝도 제외한다.',
        '- 각 train fold 정상 shot에만 StandardScaler와 LedoitWolf를 fit한다. shrinkage 강도는 LedoitWolf의 학습 데이터 기반 추정값이며 검증 튜닝이 없다.',
        '- z=StandardScaler(x), d=sqrt((z−μ)^T precision (z−μ)). 거리 원값을 위험 점수로 사용한다. validation fit·새 split·score 반전은 없다.',
        '- 스케일링은 수축 공분산의 단위 의존성을 줄이기 위해 train 정상 shot만으로 적용한다. fold별 원거리의 눈금 차이는 남을 수 있으며 ECDF로 사후 조정하지 않는다.',
        '- baseline IF/SPC는 정상 제품 행으로 학습했으므로 one-side shot의 정상 짝이 학습에 포함될 수 있다. 이번 Mahalanobis는 완전 정상 shot만 사용한다. 따라서 차이를 covariance 방식 하나만의 효과로 해석하지 않는다.',
        '- prediction의 y_true는 원래 제품 행 라벨이며 shot 점수를 원래 행에 매핑한다. 편의상 y_pred는 train 정상 거리 95% 분위수로 정했으며 AP·Recall@budget에는 사용하지 않는다.', '',
        '## Primary: shot-level 비교','',table(summary,['product','model','ap','recall5','recall10','recall20']), '',
        '반복 표준편차는 comparison.csv의 *_sd 열에 있다. 순위는 각 제품·repeat의 전체 shot 기준이며 평균 동점 순위를 사용한다.', '',
        '## CN7 유형별 차이','',table(types[types['product'].eq('cn7')],['defect_type','model','n_defects','rank_mean','rank_median','recall5','recall10','recall20']), '',
        '## 동일 검사 예산의 탐지 집합 (repeat 평균 shot 수)','',table(overlap,['product','budget_pct','baseline','both','baseline_only','mahal_only','neither']), '',
        'both/only/neither는 Mahalanobis와 각 기준 모델 사이의 분류다. 양쪽 경계에 동점이 있으면 독립 균등 tie-break 기대값을 사용한다. 합집합은 검사량이 늘어나므로 성능 향상으로 제시하지 않는다.', '',
        '## 기존 세 모델 모두 놓쳤지만 Mahalanobis가 잡은 관측','',
        table(novel,list(novel.columns)) if len(novel) else '해당 관측 없음.', '',
        'definite=True는 모든 동점 해소 방식에서 새로 잡히는 경우다. False는 경계 동점에 따라 달라지는 후보다.', '',
        '## RG3 분리력 참고','',table(summary[summary['product'].eq('rg3')],['model','ap','auc','recall5','recall10','recall20']), '',
        'RG3 불량률은 25/591=0.0423이며 무작위 검사 기대 Recall은 검사 예산과 거의 같다. 반복들은 같은 불량을 재사용하므로 독립적인 성능 개선 증거가 아니다.', '',
        '## 검증 및 파일','',
        f'- 실행 전후 기존 baseline 등 보호 파일 {len(before)}개 SHA-256 동일. 기존 3개 모델의 지표 재계산 최대 오차 {error:.3g}.',
        '- predictions/: 공통 schema의 4개 모델 OOF CSV. 기존 CSV는 읽기만 했다.',
        '- defect_shot_ranks.csv: 모든 defect shot의 repeat별 4개 모델 순위·점수·예산별 포함 비율.',
        '- detection_overlap_by_repeat.csv: repeat별 겹침 및 각 모델만 잡은 shot ID.',
        '- new_defects_vs_all_baselines.csv: 세 기준 모델 모두 놓친 신규 포착 관측.',
        '- fold_diagnostics.csv / fit_parameters.json: 정상 학습 shot ID, scaler, covariance, precision, shrinkage 근거.',
        '- manifest.json: feature 목록·환경·입력 및 보호 파일 해시.']
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(table(summary,['product','model','ap','auc','recall5','recall10','recall20']))
    print(table(types[types['product'].eq('cn7')],['defect_type','model','rank_mean','recall5','recall10','recall20']))
    print('New detections:',len(novel))


if __name__=='__main__':main()
