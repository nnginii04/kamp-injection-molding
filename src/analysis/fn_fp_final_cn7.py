"""Descriptive final CN7 Logistic error conditions; no fit or threshold change."""
import io
import json
from pathlib import Path
import hashlib
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data import load_all, assign_shots, sha256
from src.features import select_base_features, check_feature_list
from src.analysis.shap_final_cn7 import inventory_bytes, table

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/analysis/fn_fp_final_cn7'
SHAP=ROOT/'outputs/analysis/shap_final_cn7'
ORDER=['TP','TN','FP','FN']


def compare(d,features,target,reference,common=False):
    pool=d[d.common_same_class] if common else d
    a=pool[pool.group.eq(target)];b=pool[pool.group.eq(reference)]
    rows=[]
    for f in features:
        x=a[f].to_numpy();y=b[f].to_numpy()
        assert len(x)>0 and len(y)>0
        aq=np.quantile(x,[.05,.25,.5,.75,.95]);bq=np.quantile(y,[.05,.25,.5,.75,.95])
        delta=float(np.sign(x[:,None]-y[None,:]).mean())
        diff=float(aq[2]-bq[2]);scale=(bq[3]-bq[1])/1.349
        lo=max(aq[1],bq[1]);hi=min(aq[3],bq[3]);den=min(aq[3]-aq[1],bq[3]-bq[1])
        agreement=[np.sign(np.median(np.delete(x,i))-bq[2])==np.sign(diff) for i in range(len(x))] if len(x)>1 else []
        dates=[np.sign(part[f].median()-bq[2])==np.sign(diff) for _,part in a.groupby('date')]
        rows.append(dict(comparison=target+'_vs_'+reference,scope='common_subset' if common else 'final_logreg',feature=f,n_target=len(x),n_reference=len(y),
            target_median=aq[2],target_q25=aq[1],target_q75=aq[3],reference_median=bq[2],reference_q25=bq[1],reference_q75=bq[3],
            median_difference=diff,reference_robust_scale=scale,robust_standardized_difference=diff/scale if scale>1e-12 else np.nan,
            reference_zero_iqr=bool(scale<=1e-12),cliffs_delta=delta,abs_cliffs_delta=abs(delta),
            target_fraction_in_reference_iqr=float(((x>=bq[1])&(x<=bq[3])).mean()),
            target_fraction_in_reference_central90=float(((x>=bq[0])&(x<=bq[4])).mean()),
            target_fraction_in_reference_range=float(((x>=y.min())&(x<=y.max())).mean()),
            iqr_overlap_fraction= max(0.,hi-lo)/den if den>1e-12 else np.nan,
            ks_distance=float(ks_2samp(x,y).statistic),loo_direction_agreement=float(np.mean(agreement)) if agreement else np.nan,
            n_target_dates=len(dates),date_direction_agreement=float(np.mean(dates)),
            interpretation_limit='Very small defect group; descriptive only' if min(len(x),len(y))<10 else 'Observational association; dates and correlated settings not controlled'))
    return pd.DataFrame(rows).sort_values(['abs_cliffs_delta','feature'],ascending=[False,True]).reset_index(drop=True)


def plots(d,stats,fp,fn):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.3,1]})
    structure=pd.crosstab(d.group,d.side_pattern).reindex(index=ORDER,columns=['normal','one-side','both-side'],fill_value=0)
    bottom=np.zeros(4)
    for col,color in [('normal','#97adb9'),('one-side','#eab15a'),('both-side','#b45459')]:
        axes[0].bar(ORDER,structure[col],bottom=bottom,label=col,color=color)
        bottom+=structure[col].to_numpy()
    for i,n in enumerate(bottom):axes[0].text(i,n+6,str(int(n)),ha='center',weight='bold')
    axes[0].set_ylim(0,470);axes[0].set_ylabel('Unique shots');axes[0].set_title('All CN7 shots (n=606)',loc='left')
    axes[0].legend(frameon=False,fontsize=10)
    defect=structure.loc[['TP','FN'],['one-side','both-side']]
    bottom=np.zeros(2)
    for col,color in [('one-side','#eab15a'),('both-side','#b45459')]:
        vals=defect[col].to_numpy();axes[1].bar(['TP','FN'],vals,bottom=bottom,color=color)
        for i,v in enumerate(vals):
            if v:axes[1].text(i,bottom[i]+v/2,f'{col}\n{v}',ha='center',va='center',fontsize=10)
        bottom+=vals
    axes[1].set_ylim(0,14);axes[1].set_ylabel('Defect shots');axes[1].set_title('Defect structure (n=14)',loc='left')
    fig.suptitle('Figure 1 | CN7 Logistic: saved OOF majority decisions',x=.06,ha='left',weight='bold',fontsize=14)
    fig.text(.06,.015,'Positive = saved shot y_pred is positive in at least 3 of 5 repeats. No new score threshold.\nBoth FN are one-side; 9 of 11 one-side shots are TP. This does not identify a cavity-local cause.',fontsize=10)
    fig.tight_layout(rect=[.02,.10,.98,.94]);fig.savefig(OUT/'figure1_error_structure.png',dpi=180);plt.close(fig)
    # Three strongest univariate effects per comparison; no outcome rule created.
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for row,(effects,target,reference) in enumerate([(fp,'FP','TN'),(fn,'FN','TP')]):
        for ax,(_,effect) in zip(axes[row],effects.head(3).iterrows()):
            f=effect.feature;groups=[d[d.group.eq(reference)][f].to_numpy(),d[d.group.eq(target)][f].to_numpy()]
            boxes=ax.boxplot(groups,tick_labels=[reference,target],showfliers=False,patch_artist=True,widths=.5)
            for patch,color in zip(boxes['boxes'],['#97adb9','#cc795e']):patch.set_facecolor(color)
            for i,values in enumerate(groups):
                # Display all target points; sample-free distribution box for large reference.
                if len(values)<=15:ax.scatter(i+1+np.linspace(-.08,.08,len(values)),values,s=28,c='#273f51',zorder=4)
            ax.set_title(f'{f}\nCliff delta={effect.cliffs_delta:+.2f}',fontsize=11)
            ax.set_ylabel('Supplied standardized scale')
            ax.text(.02,.02,f'{target} in {reference} central 90%: {effect.target_fraction_in_reference_central90:.0%}',transform=ax.transAxes,fontsize=9,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':2})
            ax.grid(axis='y',alpha=.18)
    fig.suptitle('Figure 2 | Error-group feature differences and overlap',x=.06,ha='left',weight='bold',fontsize=14)
    fig.text(.06,.012,'Top 3 features by |Cliff delta| per comparison. Boxes: median and IQR; whiskers: 1.5 IQR.\nFP=172 vs TN=420; FN=2 vs TP=12. FN points shown individually. Values are not seconds / degrees / RPM.',fontsize=10)
    fig.tight_layout(rect=[.02,.09,.98,.94],h_pad=2.7);fig.savefig(OUT/'figure2_feature_comparison.png',dpi=180);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'outputs').rglob('*') if p.is_file() and OUT not in p.parents}
    check=json.loads((SHAP/'verification.json').read_text(encoding='utf-8'))
    assert check['oof_exact_match'] and check['max_absolute_score_error']==0
    assert sha256(ROOT/'outputs/predictions/pipeline_logreg.csv')==check['prediction_sha256']
    assert sha256(ROOT/'outputs/fold_id.csv')==check['fold_sha256']
    cfg=json.loads(subprocess.check_output(['git','show',check['origin_main_commit']+':config.json']))
    data=load_all(cfg,ROOT/cfg['data_dir']);features,_=select_base_features(data,data['feature_cols'],True)
    assert features==check['feature_names'] and len(features)==23;check_feature_list(features)
    base=json.loads((ROOT/'outputs/manifest.json').read_text(encoding='utf-8'));assert data['hashes']==base['input_sha256']
    raw,source=inventory_bytes('outputs/analysis/error_case_inventory.csv')
    assert hashlib.sha256(raw).hexdigest()==check['case_inventory_sha256']
    inv=pd.read_csv(io.BytesIO(raw),keep_default_na=False);inv=inv[inv['product'].eq('cn7')].copy()
    hyraw,hysource=inventory_bytes('outputs/predictions/pipeline_logreg_spc.csv')
    assert hashlib.sha256(hyraw).hexdigest()==check['hybrid_reference_sha256']
    hy=pd.read_csv(io.BytesIO(hyraw),float_precision='round_trip')
    lr=pd.read_csv(ROOT/'outputs/predictions/pipeline_logreg.csv',float_precision='round_trip')
    repeat_frames=[]
    for model,p in [('logreg',lr),('logreg_spc',hy)]:
        p=p[p['product'].eq('cn7')&p.cv_mode.eq('block')]
        assert len(p)==6055 and not p.duplicated(['repeat','row_id']).any()
        assert set(p.y_pred)<= {0,1} and set(p.repeat)==set(range(5))
        s=p.groupby(['shot_id','repeat']).agg(y_true=('y_true','max'),y_pred=('y_pred','max'),score=('y_score','max'),fold=('fold','first'),n_folds=('fold','nunique')).reset_index()
        assert s.n_folds.eq(1).all()
        a=s.groupby('shot_id').agg(y_true=('y_true','max'),positive_fraction=('y_pred','mean'),mean_score=('score','mean'),n_repeats=('repeat','nunique'))
        v=inv.set_index('shot_id').loc[a.index]
        assert a.n_repeats.eq(5).all() and len(a)==len(inv)==606
        assert np.array_equal(a.y_true,v.y_true)
        np.testing.assert_array_equal(a.positive_fraction,v[model+'_positive_repeat_fraction'])
        np.testing.assert_allclose(a.mean_score,v[model+'_score'],rtol=0,atol=1e-12)
        errors=np.where(a.y_true.eq(1),np.where(a.positive_fraction.gt(.5),'TP','FN'),np.where(a.positive_fraction.gt(.5),'FP','TN'))
        assert np.array_equal(errors,v[model+'_error_type'])
        assert np.array_equal(a.positive_fraction.gt(.5).astype(int),v[model+'_pred'])
        s['model']=model;repeat_frames.append(s)
    assert inv.error_type.eq(inv.logreg_error_type+'/'+inv.logreg_spc_error_type).all()
    lab=assign_shots(data['labeled']['cn7'],data['feature_cols'],'cn7')
    assert lab.groupby('shot_id')[features].nunique().le(1).all().all()
    target=lab.groupby('shot_id').y.agg(['max','sum','size'])
    rows=inv.set_index('shot_id').loc[target.index]
    structure=np.where(target['sum'].eq(0),'normal',np.where(target['sum'].eq(2)&target['size'].eq(2),'both-side',np.where(target['sum'].eq(1)&target['size'].eq(2),'one-side','unpaired')))
    assert np.array_equal(rows.side_pattern,structure) and np.array_equal(rows.y_true,target['max'])
    d=inv.merge(lab.drop_duplicates('shot_id')[['shot_id',*features]],on='shot_id',validate='one_to_one')
    d['group']=d.logreg_error_type;d['date']=pd.to_datetime(d.timestamp).dt.date.astype(str)
    d['common_same_class']=d.logreg_error_type.eq(d.logreg_spc_error_type)
    d['stable_all_repeats']=d.logreg_positive_repeat_fraction.isin([0,1])
    d['positive_repeats']=(d.logreg_positive_repeat_fraction*5).round().astype(int)
    d.to_csv(OUT/'error_cases.csv',index=False,encoding='utf-8-sig')
    pd.concat(repeat_frames).to_csv(OUT/'repeat_decisions.csv',index=False)
    gs=[]
    for group in ORDER:
        g=d[d.group.eq(group)]
        gs.append(dict(group=group,n_shots=len(g),normal=int(g.side_pattern.eq('normal').sum()),one_side=int(g.side_pattern.eq('one-side').sum()),both_side=int(g.side_pattern.eq('both-side').sum()),
            common_same_class=int(g.common_same_class.sum()),model_disagreements=int((~g.common_same_class).sum()),stable_5_of_5=int(g.stable_all_repeats.sum()),
            unstable_repeats=int((~g.stable_all_repeats).sum()),positive_repeat_counts=json.dumps(g.positive_repeats.value_counts().sort_index().to_dict()),
            reasons=json.dumps(g.defect_reason.replace('','normal/no defect').value_counts().to_dict(),ensure_ascii=False)))
    gs=pd.DataFrame(gs);gs.to_csv(OUT/'group_summary.csv',index=False,encoding='utf-8-sig')
    d.groupby(['group','side_pattern','defect_reason']).size().rename('n_shots').to_csv(OUT/'reason_structure.csv',encoding='utf-8-sig')
    stats=[]
    for group,g in d.groupby('group'):
        for f in features:
            q=g[f].quantile([.05,.25,.5,.75,.95]);stats.append(dict(group=group,feature=f,n_shots=len(g),q05=q.loc[.05],q25=q.loc[.25],median=q.loc[.5],q75=q.loc[.75],q95=q.loc[.95],iqr=q.loc[.75]-q.loc[.25],minimum=g[f].min(),maximum=g[f].max(),units='competition supplied standardized scale; not physical units'))
    stats=pd.DataFrame(stats);stats.to_csv(OUT/'feature_group_stats.csv',index=False)
    fn=compare(d,features,'FN','TP');fp=compare(d,features,'FP','TN')
    fn.to_csv(OUT/'fn_tp_comparison.csv',index=False);fp.to_csv(OUT/'fp_tn_comparison.csv',index=False)
    common=pd.concat([compare(d,features,'FN','TP',True),compare(d,features,'FP','TN',True)])
    common.to_csv(OUT/'common_subset_comparison.csv',index=False)
    compare(d,features,'FN','TN').to_csv(OUT/'fn_tn_overlap.csv',index=False)
    # Marginal distribution coverage only; no fitted normal/anomaly model.
    normal=d[d.y_true.eq(0)];overlap=[]
    for _,r in d[d.group.eq('FN')].iterrows():
        for f in features:
            q=normal[f].quantile([.05,.25,.75,.95])
            overlap.append(dict(shot_id=r.shot_id,feature=f,value=r[f],normal_q05=q.loc[.05],normal_q25=q.loc[.25],normal_q75=q.loc[.75],normal_q95=q.loc[.95],
                in_normal_iqr=bool(q.loc[.25]<=r[f]<=q.loc[.75]),in_normal_central90=bool(q.loc[.05]<=r[f]<=q.loc[.95]),in_normal_range=bool(normal[f].min()<=r[f]<=normal[f].max())))
    overlap=pd.DataFrame(overlap);overlap.to_csv(OUT/'fn_normal_overlap.csv',index=False)
    # Representative location is descriptive, not a new classifier/error rule.
    reps=[]
    for sid in ['cn7_48','cn7_5']:
        r=d[d.shot_id.eq(sid)].iloc[0];pool=d[d.group.eq(r.group)]
        for f in features:
            q=pool[f].quantile([.05,.25,.75,.95]);rank=(pool[f].lt(r[f]).sum()+.5*pool[f].eq(r[f]).sum())/len(pool)*100
            reps.append(dict(shot_id=sid,group=r.group,group_size=len(pool),feature=f,value=r[f],group_percentile=rank,
                in_group_iqr=bool(q.loc[.25]<=r[f]<=q.loc[.75]),in_group_central90=bool(q.loc[.05]<=r[f]<=q.loc[.95]),group_median=pool[f].median()))
    reps=pd.DataFrame(reps);reps.to_csv(OUT/'case_representativeness.csv',index=False)
    contrib=pd.read_csv(SHAP/'all_oof_contributions.csv')
    assert len(contrib)==606*5*23 and not contrib.duplicated(['shot_id','repeat','feature']).any()
    contrib=contrib.merge(d[['shot_id','group']],on='shot_id',validate='many_to_one')
    shapgroup=contrib.assign(abs_contribution=contrib.contribution.abs()).groupby(['group','feature']).agg(mean_contribution=('contribution','mean'),mean_abs_contribution=('abs_contribution','mean')).reset_index()
    shapgroup.to_csv(OUT/'shap_group_contributions.csv',index=False)
    plots(d,stats,fp,fn)
    assert all(sha256(ROOT/p)==h for p,h in before.items())
    verification=dict(definition_verified=True,definition='shot y_true=max(row labels); shot repeat y_pred=max(saved row y_pred); positive when >50% of 5 repeats',
        primary='existing inventory logreg_error_type, all CN7 shots',common_subset='same majority class as logreg_spc; preserved as auxiliary',
        no_new_score_threshold=True,inventory_source=source,inventory_sha256=hashlib.sha256(raw).hexdigest(),
        existing_shap_exact_match=True,existing_shap_prediction_hash_matches=True,counts=dict(zip(gs.group,gs.n_shots.astype(int))),
        features=features,no_fit=True,no_tuning=True,no_new_classifier=True,labels_and_side_posthoc_only=True,
        prior_verified_train_only_scaling=check['scaler_train_only'],prior_verified_shot_block_disjoint=check['shot_and_block_disjoint'],
        prior_side_label_feature_leakage=check['side_label_feature_leakage'],
        shap_verification_sha256=sha256(SHAP/'verification.json'),prediction_sha256=sha256(ROOT/'outputs/predictions/pipeline_logreg.csv'),
        protected_outputs_unchanged=True,protected_output_count=len(before))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2,ensure_ascii=False),encoding='utf-8')
    report=['# 최종 CN7 Logistic TP/TN/FP/FN 조건분석',
        '## 기존 오류 정의와 정합성',
        '기존 shap_case_inventory.py의 정의를 그대로 사용했다. Shot 정답은 제품 행 y_true의 max, repeat별 Shot 판정은 저장 y_pred의 max다. 5 repeat 중 3회 이상 양성이면 최종 양성이다. score 0.5/top2/top5를 새로 적용하지 않았다. final Logistic의 logreg_error_type을 사용하며 score 평균은 기술적 요약일 뿐 판정 기준이 아니다.',
        '기존 fn_fp_conditions.py는 Logistic과 logreg_spc 모두 같은 오류인 FN/FN·FP/FP 및 TP/TP·TN/TN만 비교했다. 이는 전체 Logistic 오류 중 부분집합이다. 이번 primary는 기존 inventory의 Logistic 단독 오류 전체이며, 공통 부분집합 비교도 common_subset_comparison.csv에 별도로 보존했다. 따라서 이전 common FP 수와 이번 FP 수 차이는 새로운 오류 기준 때문이 아니라 모델 간 일치 필터 범위 차이다.',
        'CN7 606 Shot의 두 모델 inventory 판정·repeat 양성비율·score를 저장 OOF와 다시 검증했다. 방금 생성한 SHAP verification의 OOF exact match 및 prediction/fold hash를 확인했다. 새로운 학습·추론·threshold tuning은 없었다.',
        '## A. 오류군 구조',table(gs),
        '![오류 구성](figure1_error_structure.png)',
        '## B. 공정변수 비교',
        '23개 feature 모두 Shot당 1표본으로 median/Q25/Q75/IQR/Q05/Q95/min/max를 저장했다. 값은 대회 제공 표준화 눈금으로 물리 단위가 아니다. 비교 우선순위는 기존 분석처럼 |Cliff delta|이며, delta=P(target>reference)−P(target<reference)이다. robust 차이는 median 차이를 reference IQR/1.349로 나누되 IQR=0은 NA다.',
        '겹침은 target의 reference IQR/중앙90%/관측범위 포함 비율, IQR 구간 겹침 길이/더 좁은 IQR 길이, KS 거리로 함께 확인했다. 이는 단변량 기술통계이며 다변량 공정영역의 동일함이나 정상 확률을 뜻하지 않는다. IQR=0인 구간 overlap 비율은 NA다. 작은 FN 표본과 날짜·상관변수 영향을 분리하지 못한다.',
        '### FP vs TN',table(fp.head(6)[['feature','n_target','n_reference','target_median','reference_median','cliffs_delta','target_fraction_in_reference_central90','target_fraction_in_reference_range']]),
        '### FN vs TP',table(fn.head(6)[['feature','n_target','n_reference','target_median','reference_median','cliffs_delta','target_fraction_in_reference_central90','target_fraction_in_reference_range']]),
        '![변수 비교](figure2_feature_comparison.png)',
        '## C. FN의 정상 분포와 겹침',table(overlap.groupby('shot_id')[['in_normal_iqr','in_normal_central90','in_normal_range']].sum().reset_index()),
        '정상 reference는 y_true=0인 592 Shot이며 TN만이 아니라 FP도 포함한다. label로 그룹을 만든 사후 분석이고 학습 또는 온라인 판정에 넣지 않았다.',
        '## D. 반복 일관성과 기존 B파트 해석',
        'FN은 cn7_40·cn7_48 두 Shot으로 모두 one-side·가스 불량이며 5회 모두 음성 판정이다. 두 Shot은 같은 2020-10-16의 가까운 시각(05:37:29, 05:45:26)에 있어 독립적인 여러 공정기간의 반복 실패라고 볼 수 없다. 전체 one-side 11개 중 9개는 TP, both-side 3개는 모두 TP다. 따라서 FN이 one-side에 집중됐다는 관찰은 맞지만 one-side 전체가 탐지 불가능하거나 cavity-local cause라고 단정할 근거는 아니다.',
        'FP 172개 중 두 모델 common FP는 84개다. 5회 모두 FP인 Shot은 28개이며 117개는 3/5회, 27개는 4/5회 양성이다. TN도 420개 중 355개가 일부 repeat에서 양성이다. TP12개와 FN2개는 5회 모두 판정이 같다. 따라서 정상 Shot의 FP/TN 구분은 fold별 저장 threshold/학습 변화에 민감한 부분이 크며 172개 모두가 반복마다 같은 FP라는 뜻은 아니다. 같은 데이터를 5회 평가한 것이므로 독립 표본 5배로 세지 않는다.',
        'FN vs TP: 두 FN의 Barrel_Temperature_2·Filling_Time은 TP보다 높고 Max_Injection_Speed는 낮다(|Cliff delta|=1). 이는 2개 FN과 12개 TP 안에서의 완전한 순서 분리이며, 신규 데이터의 완전한 분리 또는 유의한 일반 패턴을 보장하지 않는다. 정상592 Shot과 비교하면 cn7_40은 20/23개, cn7_48은 17/23개 feature가 정상 중앙90% 안이고, 두 Shot 모두 23/23개가 정상 관측범위 안이다. 기존 “정상 영역과 많이 겹친다”는 관찰을 단변량 수준에서 지지하되 정상 중심과 완전히 같거나 다변량으로 구별할 수 없다고 확대하지 않는다.',
        'FP vs TN: Mold_Temperature_3/4가 높고 Average_Back_Pressure가 낮은 쪽으로 이동한다(Cliff delta +0.416/+0.408/−0.355). 그러나 해당 feature의 FP 중 각각 89.5%/98.8%/96.5%가 TN 중앙90% 안에 있다. 특정 강한 하나의 공정 패턴이 전체 FP를 분명하게 나눈다는 주장은 지지되지 않는다. common subset에서도 상위 세 변수는 같아 방향은 유지되지만 표본 수는 다르다.',
        '## E. 대표 사례의 위치',
        'cn7_48: 전체 FN 2개 중 하나로, 다른 FN cn7_40과 one-side·가스·5회 일관된 FN·낮은 Max_Injection_Speed 및 높은 Filling_Time을 공유한다. 이 작은 오류군의 구조를 보여 주지만 일반 FN 모집단의 대표라고 할 수 없다. 평균 OOF score는 cn7_48 0.0221, cn7_40 0.2100으로 차이가 있고 Barrel_Temperature_2도 두 Shot 간 차이가 크다. FN 표본2개에서 IQR/중앙90%는 두 점 사이 보간이라 자기 사례가 구간 밖이라는 사실을 이상치 근거로 쓰지 않는다.',
        'cn7_5: FP172개 중 평균 Logistic score 0.9925로 2위이고 5/5회 양성인 안정적인 고점수 FP다. Clamp_Close_Time은 FP 내부 midrank percentile 약97.97%로 중앙90% 밖이다. 나머지22개 feature는 FP 중앙90% 안이며 15/23개는 IQR 안이다. 따라서 다수 변수는 FP군과 겹치지만 이 사례의 강한 Clamp_Close_Time 기여와 극단 score는 전체 FP의 전형으로 일반화하기 어렵다. 기존 common FP 선정은 두 모델 평균 percentile 기준 1위였으며 Logistic 단독 score 1위와 혼동하지 않는다.',
        '## F. SHAP와 일치하는 점과 긴장 관계',
        '기존 contribution 파일만 재집계했다. FP의 Mold_Temperature_3/4 평균 기여는 +0.524/+0.516, TN은 −0.336/−0.288로 금형 온도 차이와 판단 방향이 일치한다. 하지만 Average_Back_Pressure는 그룹 분포 차이가 비교적 커도 FP 평균 기여 −0.020, TN +0.025로 작다. 분포 차이가 큰 변수와 예측 기여가 큰 변수는 같지 않다.',
        'FN의 Filling_Time은 TP보다 높지만 평균 기여는 FN +0.511, TP −0.237이다. 즉 이 변수는 FN score를 올리는 방향이며 미포착의 원인이라고 해석하면 모순된다. 반대로 FN Max_Injection_Speed 평균 기여 −1.035(TP +2.392), Barrel_Temperature_2 −2.497(TP −0.751)는 낮은 점수와 부합한다. Barrel_Temperature_2는 이전 global SHAP에서 fold별 계수 부호가 바뀌므로 전체에서 “값이 높으면 항상 score를 낮춘다”라고 단정하지 않는다.',
        'Clamp_Close_Time은 global mean absolute contribution 1위이고 cn7_5의 큰 양의 기여 변수지만, 전체 FP 평균 기여는 +0.210이다. 한 사례의 강한 판단 패턴, 전체 중요도, 오류군 분포 차이는 서로 다른 질문이다. 이 차이를 숨기지 않고 구분한다.',
        '## 해석 한계',
        '모든 비교는 사후 오류군의 관측 연관이며 불량의 물리적 원인, cavity-local cause, 온라인 탐지 규칙을 제시하지 않는다. group/side/Reason/label은 통계 집계에만 사용했다. 제공 변수의 구분 신호와 겹침을 기술했으며 날짜 효과·상관관계·희소 불량의 영향을 분리하지 못했다. 새 threshold, FN classifier, anomaly method는 만들지 않았다.',
        '## 파일',
        'group_summary.csv; reason_structure.csv; feature_group_stats.csv; error_cases.csv; repeat_decisions.csv; fn_tp_comparison.csv; fp_tn_comparison.csv; common_subset_comparison.csv; fn_tn_overlap.csv; fn_normal_overlap.csv; case_representativeness.csv; shap_group_contributions.csv; figure1/2 PNG; verification.json.',
        '재현: `.venv\\Scripts\\python.exe -m src.analysis.fn_fp_final_cn7`']
    (OUT/'summary.md').write_text('\n\n'.join(report),encoding='utf-8')
    print(gs.to_string(index=False))
    print('FN vs TP',fn.head(6)[['feature','cliffs_delta','target_fraction_in_reference_central90']].to_string(index=False))
    print('FP vs TN',fp.head(6)[['feature','cliffs_delta','target_fraction_in_reference_central90']].to_string(index=False))
    print('FN normal overlap',overlap.groupby('shot_id')[['in_normal_iqr','in_normal_central90','in_normal_range']].sum().to_string())
    print('Representativeness',reps.groupby('shot_id')[['in_group_iqr','in_group_central90']].sum().to_string())


if __name__=='__main__':main()
