"""Exact CN7 block-OOF Logistic reproduction and linear contribution analysis.

Reuses shap_oof_explainer.attribution and the saved shap_case_inventory output.
No tuning, new feature, alternative SHAP estimator, or other-model fit.
"""
import io
import os
import json
import hashlib
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.special import logit, expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data import load_all, assign_shots, sha256
from src.features import select_base_features, check_feature_list
from src.models import make_model
from src.cv import cv_splits, make_groups
from src.analysis.shap_oof_explainer import attribution
from src.analysis.audit_b_final_metadata import append_notes, CLAMP_NOTE, REASON_NOTE

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/analysis/shap_final_cn7'
ARCHIVE=Path(os.environ.get('KAMP_DELIVERED_ARCHIVE',ROOT.parent/'제조데이터.zip'))  # 전달본 zip 위치(환경변수), 로컬 경로 하드코딩 금지
ZIPBASE='제조데이터/kamp-injection-molding/'


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def inventory_bytes(relative):
    local=ROOT/relative
    if local.exists():return local.read_bytes(),str(relative)
    with zipfile.ZipFile(ARCHIVE) as z:return z.read(ZIPBASE+relative),f'{ARCHIVE.name}::{ZIPBASE}{relative}'


def table(d):
    def fmt(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |',
                      *['| '+' | '.join(map(fmt,r))+' |' for r in d.itertuples(index=False,name=None)]])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'outputs').rglob('*') if p.is_file() and OUT not in p.parents}
    ref=git('rev-parse','origin/main').decode().strip()
    cfg=json.loads(git('show',f'{ref}:config.json'))
    assert cfg['cv']['n_splits']==3 and cfg['cv']['n_repeats']==5
    assert not any(cfg['features'].get(k,False) for k in ['sequence_features','unlabeled_iforest_score','pair_position'])
    # Verify split/data/feature code is identical to the selected current main.
    for name in ['src/cv.py','src/data.py','src/features.py']:
        assert (ROOT/name).read_bytes().replace(b'\r\n',b'\n')==git('show',f'{ref}:{name}').replace(b'\r\n',b'\n'),name
    data=load_all(cfg,ROOT/cfg['data_dir'])
    baseline=json.loads((ROOT/'outputs/manifest.json').read_text(encoding='utf-8'))
    assert data['hashes']==baseline['input_sha256']
    features,_=select_base_features(data,data['feature_cols'],cfg['features']['drop_constant'])
    check_feature_list(features);assert len(features)==23
    assert not set(features)&{'side','side_rh','LH','RH','y','y_true','PassOrFail','Reason','shot_id','row_id'}
    lab=assign_shots(data['labeled']['cn7'],data['feature_cols'],'cn7').reset_index(drop=True)
    X=lab[features];y=lab.y.to_numpy()
    folds=pd.read_csv(ROOT/'outputs/fold_id.csv');folds=folds[folds.mold.eq('cn7')].set_index('row_id').loc[lab.row_id]
    assert np.array_equal(folds.shot_id,lab.shot_id)
    assert np.array_equal(make_groups(lab,'block',cfg['cv']['block_shots']),folds.block_id)
    pred=pd.read_csv(ROOT/'outputs/predictions/pipeline_logreg.csv',float_precision='round_trip')
    pred=pred[pred['product'].eq('cn7')&pred.cv_mode.eq('block')].copy()
    assert len(pred)==len(lab)*5 and not pred.duplicated(['repeat','row_id']).any()
    assert set(pred.repeat)==set(range(5))
    fits=[];checks=[];foldaudit=[]
    for repeat in range(5):
        col=f'fold_block_r{repeat}'
        assert folds.groupby('shot_id')[col].nunique().eq(1).all()
        assert folds.groupby('block_id')[col].nunique().eq(1).all()
        for fold,(tr,va) in enumerate(cv_splits(lab,'block',cfg,repeat)):
            assert np.array_equal(np.sort(va),np.flatnonzero(folds[col].to_numpy()==fold))
            assert set(lab.iloc[tr].shot_id).isdisjoint(lab.iloc[va].shot_id)
            assert set(folds.iloc[tr].block_id).isdisjoint(folds.iloc[va].block_id)
            model=make_model('logreg',cfg['seed'])
            lr=model.steps[-1][1]
            assert lr.C==.3 and lr.class_weight=='balanced' and lr.max_iter==5000 and lr.solver=='lbfgs'
            model.fit(X.iloc[tr],y[tr])
            actual=model.predict_proba(X.iloc[va])[:,1]
            saved=pred[pred.repeat.eq(repeat)&pred.fold.eq(fold)].set_index('row_id').loc[lab.iloc[va].row_id]
            assert len(saved)==len(va) and np.array_equal(saved.shot_id,lab.iloc[va].shot_id)
            assert np.array_equal(saved.y_true,y[va])
            exact=actual==saved.y_score.to_numpy()
            for j,pos in enumerate(va):
                checks.append(dict(repeat=repeat,fold=fold,row_id=int(lab.row_id.iloc[pos]),shot_id=lab.shot_id.iloc[pos],
                    saved_score=float(saved.y_score.iloc[j]),reproduced_score=float(actual[j]),absolute_error=float(abs(actual[j]-saved.y_score.iloc[j])),exact_match=bool(exact[j])))
            if not exact.all():
                pd.DataFrame(checks).to_csv(OUT/'oof_reproduction_check.csv',index=False)
                dump('verification.json',dict(oof_exact_match=False,stopped_before_attribution=True,repeat=repeat,fold=fold,max_error=float(abs(actual-saved.y_score.to_numpy()).max())))
                (OUT/'summary.md').write_text('OOF exact reproduction failed. Contribution analysis stopped. See verification.json and oof_reproduction_check.csv.',encoding='utf-8')
                raise RuntimeError('OOF is not exactly equal; explanation stopped')
            fits.append((repeat,fold,tr,va,model))
            foldaudit.append(dict(repeat=repeat,fold=fold,train_rows=len(tr),validation_rows=len(va),
                train_shot_ids=lab.iloc[tr].shot_id.unique().tolist(),validation_shot_ids=lab.iloc[va].shot_id.unique().tolist()))
    checks=pd.DataFrame(checks);checks.to_csv(OUT/'oof_reproduction_check.csv',index=False)
    print(f'EXACT PASS: {len(checks)} predictions, 15 CN7 folds, max error {checks.absolute_error.max()}',flush=True)

    # Reuse existing common-error inventory. Verify its CN7 Logistic and hybrid
    # membership against saved OOF data before selecting cases (never fit hybrid).
    invbytes,invsource=inventory_bytes('outputs/analysis/error_case_inventory.csv')
    inv=pd.read_csv(io.BytesIO(invbytes));inv=inv[inv['product'].eq('cn7')].copy()
    hybytes,hysource=inventory_bytes('outputs/predictions/pipeline_logreg_spc.csv')
    hy=pd.read_csv(io.BytesIO(hybytes),float_precision='round_trip')
    hy=hy[hy['product'].eq('cn7')&hy.cv_mode.eq('block')]
    ids=['repeat','fold','row_id','shot_id','y_true']
    pd.testing.assert_frame_equal(pred[ids].sort_values(ids).reset_index(drop=True),hy[ids].sort_values(ids).reset_index(drop=True))
    for name,df in [('logreg',pred),('logreg_spc',hy)]:
        s=df.groupby(['shot_id','repeat']).agg(score=('y_score','max'),prediction=('y_pred','max'),truth=('y_true','max'))
        agg=s.groupby('shot_id').agg(score=('score','mean'),positive_fraction=('prediction','mean'),truth=('truth','max'))
        rows=inv.set_index('shot_id').loc[agg.index]
        np.testing.assert_allclose(rows[name+'_score'],agg.score,rtol=0,atol=1e-12)
        np.testing.assert_allclose(rows[name+'_positive_repeat_fraction'],agg.positive_fraction,rtol=0,atol=1e-12)
        assert np.array_equal(rows.y_true,agg.truth)
        error=np.where(agg.truth.eq(1),np.where(agg.positive_fraction.gt(.5),'TP','FN'),np.where(agg.positive_fraction.gt(.5),'FP','TN'))
        assert np.array_equal(rows[name+'_error_type'],error)
    assert inv.set_index('shot_id').loc['cn7_58','logreg_error_type']=='TP'
    assert inv.set_index('shot_id').loc['cn7_48','logreg_error_type']=='FN'
    fp=inv[inv.error_type.eq('FP/FP')].sort_values(['mean_percentile','shot_id'],ascending=[False,True]).iloc[0].shot_id
    cases={'TP':'cn7_58','FN':'cn7_48','FP':fp}
    inv[inv.error_type.eq('FP/FP')].sort_values('mean_percentile',ascending=False).to_csv(OUT/'common_fp_selection.csv',index=False)
    allparts=[];additivity=[];coefs=[]
    for repeat,fold,tr,va,model in fits:
        scaler,lr=model.steps[0][1],model.steps[-1][1]
        phi,base,method,_=attribution(model,X.iloc[va],X.iloc[tr],features,'logreg',cfg['seed'],32)
        z=scaler.transform(X.iloc[va]);contribution=z*lr.coef_[0]
        score=model.predict_proba(X.iloc[va])[:,1];decision=model.decision_function(X.iloc[va])
        total=lr.intercept_[0]+contribution.sum(axis=1)
        assert ((score>0)&(score<1)).all()
        np.testing.assert_allclose(total,decision,rtol=0,atol=1e-10)
        np.testing.assert_allclose(total,logit(score),rtol=0,atol=1e-9)
        np.testing.assert_allclose(phi+base[:,None]/len(features),contribution+lr.intercept_[0]/len(features),rtol=0,atol=1e-10)
        for j,f in enumerate(features):coefs.append(dict(repeat=repeat,fold=fold,feature=f,coefficient=lr.coef_[0,j],raw_slope=lr.coef_[0,j]/scaler.scale_[j]))
        for i,pos in enumerate(va):
            additivity.append(dict(repeat=repeat,fold=fold,row_id=int(lab.row_id.iloc[pos]),shot_id=lab.shot_id.iloc[pos],
                intercept=lr.intercept_[0],sum_contributions=contribution[i].sum(),sum_plus_intercept=total[i],logit_saved_score=logit(score[i]),
                absolute_error=abs(total[i]-logit(score[i])),train_mean_shap_error=abs(base[i]+phi[i].sum()-decision[i])))
        # Equal shot weight in global summary; identical process values can have
        # different LH/RH labels but are never counted twice here.
        local=lab.iloc[va].reset_index(drop=True);unique=~local.shot_id.duplicated()
        for i in np.flatnonzero(unique):
            for j,f in enumerate(features):
                allparts.append(dict(repeat=repeat,fold=fold,shot_id=local.shot_id.iloc[i],feature=f,raw_value=X.iloc[va[i],j],
                    standardized_value=z[i,j],coefficient=lr.coef_[0,j],contribution=contribution[i,j],train_mean_shap=phi[i,j],
                    intercept=lr.intercept_[0],score=score[i]))
    contributions=pd.DataFrame(allparts);a=pd.DataFrame(additivity);coef=pd.DataFrame(coefs)
    a.to_csv(OUT/'additivity_check.csv',index=False);coef.to_csv(OUT/'fold_coefficients.csv',index=False)
    contributions.to_csv(OUT/'all_oof_contributions.csv',index=False)
    g=contributions.assign(abs_contribution=contributions.contribution.abs(),positive=contributions.contribution.gt(0)).groupby('feature').agg(
        mean_abs_contribution=('abs_contribution','mean'),mean_contribution=('contribution','mean'),positive_contribution_fraction=('positive','mean'))
    c=coef.assign(positive=coef.coefficient.gt(0)).groupby('feature').agg(coefficient_mean=('coefficient','mean'),coefficient_min=('coefficient','min'),coefficient_max=('coefficient','max'),positive_coefficient_fold_fraction=('positive','mean'))
    g=g.join(c);g['direction']=np.where(g.coefficient_min.gt(0),'higher value increases defect log-odds',np.where(g.coefficient_max.lt(0),'higher value decreases defect log-odds','direction varies by fold'))
    g=g.sort_values('mean_abs_contribution',ascending=False).reset_index();g.insert(0,'importance_rank',np.arange(1,len(g)+1))
    g.to_csv(OUT/'global_importance.csv',index=False)
    summary=[]
    for kind,sid in cases.items():
        part=contributions[contributions.shot_id.eq(sid)]
        part.to_csv(OUT/f'{kind.lower()}_contributions.csv',index=False)
        refcase=inv[inv.shot_id.eq(sid)].iloc[0]
        why={'TP':'Requested candidate; confirmed positive in all 5 saved OOF decisions','FN':'Requested candidate; confirmed defect missed in all 5 saved OOF decisions','FP':'Highest existing mean_percentile among CN7 common FP; no contribution-based selection'}[kind]
        for repeat,pp in part.groupby('repeat'):
            rr=pred[pred.shot_id.eq(sid)&pred.repeat.eq(repeat)]
            pos=pp[pp.contribution.gt(0)].nlargest(5,'contribution');neg=pp[pp.contribution.lt(0)].nsmallest(5,'contribution')
            summary.append(dict(case_type=kind,shot_id=sid,selection_reason=why,repeat=int(repeat),fold=int(rr.fold.iloc[0]),
                y_true=int(rr.y_true.max()),product_row_labels=';'.join(f'{int(r.row_id)}:{int(r.y_true)}' for _,r in rr.iterrows()),
                score=float(rr.y_score.iloc[0]),saved_y_pred=int(rr.y_pred.max()),representative_figure=bool(repeat==0),
                intercept=float(pp.intercept.iloc[0]),sum_contributions=float(pp.contribution.sum()),logit_score=float(logit(rr.y_score.iloc[0])),
                positive_features=';'.join(pos.feature),negative_features=';'.join(neg.feature),
                side_pattern=refcase.side_pattern,defect_reason=refcase.defect_reason))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'case_summary.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11,9));plot=g.iloc[::-1]
    colors=plot.direction.map({'higher value increases defect log-odds':'#bd4b4b','higher value decreases defect log-odds':'#287990','direction varies by fold':'#8c8698'})
    ax.barh(plot.feature,plot.mean_abs_contribution,color=colors)
    ax.set_xlabel('Mean |standardized input × coefficient| (log-odds)')
    ax.set_title('CN7 Logistic | 606 OOF shots × 5 repeats',loc='left',weight='bold')
    fig.text(.04,.018,'Color: red = higher value raises log-odds; blue = lowers; gray = sign varies across 15 fitted folds.\nImportance uses actual contributions, not coefficient magnitude. Equal weight per shot and repeat.',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'global_importance.png',dpi=180);plt.close(fig)
    for kind,sid in cases.items():
        pp=contributions[contributions.shot_id.eq(sid)&contributions.repeat.eq(0)].sort_values('contribution')
        info=summary[summary.shot_id.eq(sid)&summary.repeat.eq(0)].iloc[0]
        fig,ax=plt.subplots(figsize=(12,9))
        labels=[f'{r.feature}  [raw={r.raw_value:.5g}]' for r in pp.itertuples()]
        ax.barh(labels,pp.contribution,color=np.where(pp.contribution.gt(0),'#bd4b4b','#287990'));ax.axvline(0,color='#444',lw=.8)
        ax.set_xlabel('Contribution to defect log-odds (standardized input × coefficient)')
        ax.set_title(f'{kind} | {sid} | repeat 0, fold {info.fold} | y={info.y_true} | OOF score={info.score:.6f}',loc='left',weight='bold')
        fig.text(.035,.016,f'Intercept {info.intercept:.6f} + contributions {info.sum_contributions:.6f} = logit(score) {info.logit_score:.6f}\nRed raises the score; blue lowers it relative to the training mean. All 23 original feature values shown.',fontsize=10)
        fig.tight_layout(rect=[0,.075,1,1]);fig.savefig(OUT/f'{kind.lower()}_contributions.png',dpi=180);plt.close(fig)
    note=['# 최종 CN7 Logistic OOF feature contribution 분석',
          f'기준: origin/main `{ref}`의 3-fold × 5 repeat block CV. 기존 Logistic(StandardScaler + C=0.3, balanced, max_iter=5000)을 CN7 15개 fold에서만 재현했다. 튜닝·새 feature·다른 모델 fit은 하지 않았다.',
          f'OOF 재현: **{len(checks)}개 제품 행 예측 전부 exact match**, 최대 절대오차 **{checks.absolute_error.max()}**. CSV는 round_trip float 파서로 읽었고 허용오차로 불일치를 통과시키지 않았다. 전체 재현 검증을 마친 뒤 contribution 계산을 시작했다.',
          '## 방법과 검증',
          '기존 shap_oof_explainer.attribution의 exact train-mean linear log-odds 설명을 재사용했다. 주 contribution은 z_j×β_j이며 z=(원본값−해당 train 평균)/train 표준편차다. 기존 train-mean SHAP은 (z−train z 평균)×β이며 train z 평균이 수치적으로 0이라 같은 설명이다. all_oof_contributions.csv에 두 값을 함께 저장했다. 별도의 SHAP 추정 방법은 추가하지 않았다.',
          f'intercept + Σ contribution = logit(OOF score) 최대 절대오차 **{a.absolute_error.max():.3g}**; 기존 attribution의 합 복원 최대오차 **{a.train_mean_shap_error.max():.3g}**. log-odds를 sigmoid로 변환한 것이 확률이다. contribution을 확률 증감으로 직접 읽으면 안 된다.',
          '각 validation Shot과 block은 해당 fold train에서 제외된다. Scaler와 계수는 train 제품 행만 fit한다. 23개 공정변수만 입력하며 y/side/Reason/ID는 입력에 없다. side·불량유형은 기존 inventory의 사후 설명 정보로만 사용했다.',
          '`raw_value`와 그림의 raw는 대회 labeled CSV에 저장된 입력값, 즉 fold StandardScaler 적용 전 값이다. 이 대회 데이터 자체가 표준화된 값이므로 초/도/RPM 등 물리 단위의 원시 센서값으로 해석하지 않는다. 별도 원본과의 역변환을 추정하지 않았다.',
          '## Global',
          '전체 606 Shot ×5 repeat의 mean absolute contribution으로 중요도를 계산했다. 동일 Shot LH/RH 중복 가중을 피했다. 방향은 각 fitted model에서 다른 값을 고정했을 때 값 증가가 log-odds에 미치는 계수 부호이며, fold마다 부호가 바뀌면 고정 방향으로 단정하지 않는다.',
          table(g.head(10)[['importance_rank','feature','mean_abs_contribution','direction','positive_coefficient_fold_fraction']]),
          '![Global](global_importance.png)',
          '## 사례 선정',
          f'기존 shap_case_inventory의 저장 inventory({invsource})를 사용하고 CN7 전체의 Logistic 및 logreg_spc OOF score·label·fold·저장 판정을 재검증했다. TP/FN/FP는 저장 y_pred의 repeat 과반수이며 score 0.5로 새 판정을 만들지 않았다. common FP 중 기존 두 모델 평균 percentile 최고인 {fp}를 골랐다. common_fp_selection.csv에 전체 후보와 순서를 저장했다.',
          '그림은 사전에 고정한 repeat0만 표시한다. CSV는 모든 5 repeat를 포함한다. repeat 평균 score의 logit을 하나의 fold 기여 합처럼 설명하지 않는다. y_true는 shot max이며 제품 행별 원래 label도 case_summary.csv에 기록했다.']
    for kind,sid in cases.items():
        case=summary[summary.shot_id.eq(sid)];rep=case[case.repeat.eq(0)].iloc[0]
        pp=contributions[contributions.shot_id.eq(sid)&contributions.repeat.eq(0)]
        tops=pd.concat([pp[pp.contribution.gt(0)].nlargest(5,'contribution'),pp[pp.contribution.lt(0)].nsmallest(5,'contribution')])
        interpretation={'TP':'양쪽 불량인 cn7_58은 Max_Injection_Speed의 큰 양의 기여가 음의 intercept와 반대 방향 기여를 넘어 높은 score를 만든다. 이는 미래 E2 event0이 아니라 별개의 대회 OOF 사례다.',
                        'FN':'한쪽 불량인 cn7_48은 Clamp_Close_Time 등 양의 기여가 있으나 Barrel_Temperature_2·Cushion_Position 등의 음의 기여가 상쇄한다. 기여 합은 양수 1.427이지만 intercept -5.829를 넘지 못해 score가 낮다. 양의 feature가 없어서 놓친 사례라고 설명하면 안 된다.',
                        'FP':'정상 cn7_5는 Clamp_Close_Time의 큰 양의 기여에 금형 온도·Plasticizing_Position 등이 더해져 높은 score를 만든다. 모델이 정상 공정 조합을 불량 방향으로 평가한 사례이며 실제 불량이나 공정 원인을 입증하는 것은 아니다.'}[kind]
        note.extend([f'## {kind}: {sid}',rep.selection_reason,interpretation,
            table(case[['repeat','fold','y_true','score','saved_y_pred','product_row_labels']]),
            f'repeat0 해석: 양의 기여 상위 {rep.positive_features}는 train 평균 대비 log-odds를 올리고, 음의 기여 상위 {rep.negative_features}는 낮춘다. intercept {rep.intercept:.6f}에 기여 합 {rep.sum_contributions:.6f}를 더하면 logit {rep.logit_score:.6f}, OOF score {rep.score:.6f}를 복원한다.',
            table(tops[['feature','raw_value','standardized_value','coefficient','contribution']]),
            f'![{kind}]({kind.lower()}_contributions.png)',
            '해석 한계: 이 값들은 해당 fitted Logistic의 판단 경로이며 불량의 물리적 원인을 입증하지 않는다. 상관된 공정변수 사이에 계수/기여가 분산될 수 있다. 동일 machine-level X의 LH/RH outcome 차이나 미관측 cavity 변수를 설명하지 못한다. FP의 양의 기여도 실제 불량 증거로 해석하지 않는다.'])
    note.extend(['## 산출물과 재현','global_importance.csv, case_summary.csv, tp/fn/fp_contributions.csv, 각 PNG, oof_reproduction_check.csv, additivity_check.csv, fold_coefficients.csv, all_oof_contributions.csv, common_fp_selection.csv, verification.json.',
                 '실행: `.venv\\Scripts\\python.exe -m src.analysis.shap_final_cn7`'])
    (OUT/'summary.md').write_text('\n\n'.join(note),encoding='utf-8')
    append_notes(OUT/'summary.md',CLAMP_NOTE,REASON_NOTE)
    assert all(sha256(ROOT/p)==h for p,h in protected.items())
    dump('verification.json',dict(oof_exact_match=True,max_absolute_score_error=float(checks.absolute_error.max()),predictions_checked=len(checks),
        fits=15,models_fitted=['logreg'],product='cn7',origin_main_commit=ref,fold_scheme='3-fold x 5 repeat block',
        feature_names=features,feature_count=23,scaler_train_only=True,shot_and_block_disjoint=True,side_label_feature_leakage=False,
        additivity_max_error=float(a.absolute_error.max()),additivity_tolerance=1e-9,existing_attribution_max_error=float(a.train_mean_shap_error.max()),
        global_units='606 unique shots x 5 repeats',cases=cases,case_inventory_source=invsource,case_inventory_sha256=hashlib.sha256(invbytes).hexdigest(),
        hybrid_reference_source=hysource,hybrid_reference_sha256=hashlib.sha256(hybytes).hexdigest(),
        protected_outputs_unchanged=True,protected_output_count=len(protected),fold_sha256=sha256(ROOT/'outputs/fold_id.csv'),
        prediction_sha256=sha256(ROOT/'outputs/predictions/pipeline_logreg.csv'),numpy=np.__version__,sklearn=sklearn.__version__,fold_audit=foldaudit))
    print(g.head(10).to_string(index=False));print(summary[summary.representative_figure].to_string(index=False))


if __name__=='__main__':main()
