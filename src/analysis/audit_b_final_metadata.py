"""Read-only arithmetic audit of public provenance; no estimator fit/replay."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import _public
import external_experiments as X
from src.data import load_all,assign_shots,sha256
from src.features import select_base_features
from src.models import ecdf_score

REASON_NOTE='Reason 출처: 가스·미성형 등의 Reason은 대회 표준화 CSV 자체가 아니라 KAMP 공개 원본 데이터의 Reason 열에서 가져왔다. 공개 원본은 TimeStamp, PART_NAME, Reason 등 구조·오류해석용 메타데이터 확인에 사용했다. 대회 labeled의 row_id와 공개 moldset_labeled.csv의 Unnamed: 0 대응 1,211행 및 라벨 일치를 검증했다. 해당 정보를 새로운 학습 feature로 사용하지 않았다.'
SPC_NOTE='''## SPC 입력·driver 검증 및 해석 제한 (예진 검토 반영)

E2는 x3_future.csv 자체를 입력으로 읽지 않고 external_experiments.future_rows의 기본 RPM_FIX=True 경로를 호출한다. 이는 x3_future.csv를 생성하는 Average_Screw_RPM<100 → ×10 보정 경로와 같다. x3_future_norpmfix.csv 경로가 아니다. 현재 작업 폴더에 두 X3 요약 CSV 자체는 없으므로 파일명을 실행 provenance인 것처럼 주장하지 않고 코드와 저장 score를 대조했다.

모델 fit 없이 기존 SPC의 정상 행 평균/SD(sd_floor=0.01), max|z|, ECDF 공식을 산술 검산했다. 보정 후 shot1173 score=0.992887029288703 → 저장0.99289, rank=30/1389이고 전체1,389개 저장 SPC 점수(소수5자리) 및 rank가 일치한다. 보정 전은 score=1.0/rank1로 다르다. 실제 최대 |z| feature는 **Average_Screw_RPM**, signed z=−5.890276099357472이다. [근거: cn7_e2_spc_input_verification.csv의 rpm_fix=True/False 행; cn7_e2_spc_driver_verification.csv의 Average_Screw_RPM 행; b_final_metadata_verification.json.]

**SPC는 shot1173을 고위험으로 평가했으나 최대 이탈 변수는 Average_Screw_RPM이었다. 해당 변수는 10/29 이후 기록 눈금 변화가 확인된 변수이므로, 이 SPC 신호를 불량의 직접 공정 신호로 해석하지 않는다. ×10 보정 후에도 기록 수준/정밀도 변화의 영향이 남을 수 있어 보조 결과로만 사용한다.** 날짜별 원본 기록 범위는 cn7_public_rpm_daily_audit.csv에 남겼다. 잔여 영향의 가능성을 말하는 것이며 물리적 불량 원인을 확정하지 않는다.

SPC의 요청·room·실제 검사/포착 비교는 원본 로그 관측으로 유지한다. 이 조건별 운영 결과를 SPC가 불량의 공정 신호를 찾아냈다는 성공 근거로 강조하지 않는다. Logistic의 약0 score, rank823/1389, 요청0, room2/4, prediction/selection miss 결론은 그대로다.
'''
CLAMP_NOTE='''## Feature contribution의 시간·레시피 의존성

공개 원본 moldset_labeled.csv와 연결한 **CN7 competition 구간**의 Clamp_Close_Time은 약7.12/7.13/7.14의 세 값뿐인 준-레시피 변수다. 공개 데이터의 모든 CN7 기간에 값이 세 개뿐이라고 범위를 확대하지 않는다. CN7 competition 불량은 모두 2020-10-16 05:21:38–05:57:20에 집중되어 있다(17개 제품 행/14 Shot). [근거: ../cn7_public_metadata_audit.csv, ../b_final_metadata_verification.json.]

따라서 Clamp_Close_Time 기여도 1위는 불량의 물리적 원인이라기보다 특정 생산기간/레시피를 대리한 결과일 가능성이 있다. 이런 시간/조건 의존적 과적합을 줄이기 위해 random split 대신 인접 Shot을 묶어 제외하는 block CV를 사용했다. 다만 block CV도 시간과 레시피의 공선성을 완전히 제거하지는 못하므로 contribution은 **모델 판단 근거**로만 해석한다. 해당 변수가 실제 원인 또는 시간 대리변수라고 확정하지 않는다.
'''
CAPTION_NOTE='본 TP/TN/FP/FN은 5-repeat majority 판정을 이용한 오류구조 분석 결과이며, 양성 판정은 184/606 shot(30.4%)이다. 고정 검사예산에서의 성능은 제4장 E1을 참조한다. 14 defect 중 12 TP라는 결과와 검사 10%에서 포착률 약 0.61은 서로 다른 평가조건이다. 이 작업에서 E1 수치를 재계산하지 않았다.'


def append_notes(path,*notes):
    text=path.read_text(encoding='utf-8')
    for note in notes:
        if note not in text:text+='\n\n'+note+'\n'
    path.write_text(text,encoding='utf-8')


def apply_saved_notes():
    """Apply verified text/captions without invoking any analysis main/fit/replay."""
    from src.analysis.update_e2_detail_report import figures
    from src.analysis.fn_fp_final_cn7 import plots
    out=ROOT/'outputs/analysis'
    e2=out/'cn7_logreg_e2_shot1173_analysis.md'
    fn=out/'fn_fp_final_cn7/summary.md'
    sh=out/'shap_final_cn7/summary.md'
    mutable=[e2,fn,sh,out/'fig_cn7_logreg_e2_q1_score_rank.png',out/'fig_cn7_logreg_e2_q2_decision_flow.png',
             out/'fn_fp_final_cn7/figure1_error_structure.png']
    verification=out/'b_final_revision_verification.json'
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'outputs').rglob('*')
               if p.is_file() and p not in mutable and p!=verification}
    audit=json.loads((out/'b_final_metadata_verification.json').read_text(encoding='utf-8'))
    assert sha256(ROOT/'outputs/inspection/delay_replay_log_full.csv')==audit['spc_saved_log_sha256']
    assert audit['spc_input_checks'][0]['max_z_feature']=='Average_Screw_RPM'
    gs=pd.read_csv(out/'fn_fp_final_cn7/group_summary.csv').set_index('group')
    positive=int(gs.loc[['TP','FP'],'n_shots'].sum());total=int(gs.n_shots.sum())
    assert (positive,total)==(184,606) and round(100*positive/total,1)==30.4
    meta=pd.read_csv(out/'cn7_public_metadata_audit.csv',keep_default_na=False)
    cases=pd.read_csv(out/'fn_fp_final_cn7/error_cases.csv',keep_default_na=False)
    for _,r in cases.iterrows():
        linked=meta[meta.shot_id.eq(r.shot_id)]
        assert set(linked.row_id)==set(map(int,str(r.row_ids).split(';')))
        bad=linked[linked.PassOrFail.astype(int).eq(1)]
        reason=' | '.join(sorted(bad.Reason[bad.Reason.ne('')].unique()))
        assert reason==r.defect_reason, r.shot_id
    old_e2=e2.read_text(encoding='utf-8')
    q34=old_e2[old_e2.index('## Q3.'):old_e2.index('## 보조모델 비교')]
    e2text=old_e2
    warning='SPC 최대 이탈 변수는 ×10 보정 후에도 Average_Screw_RPM이다. 아래 포착·미포착은 저장된 검사운영 결과이며, 기록 눈금/정밀도 변화의 영향 때문에 불량의 직접 공정 신호를 탐지했다는 근거로 해석하지 않는다. 상세 driver 검증은 문서 마지막 절에 기록했다.'
    if warning not in e2text:e2text=e2text.replace('## 보조모델 비교 — Logistic과 구분','## 보조모델 비교 — Logistic과 구분\n\n'+warning,1)
    e2.write_text(e2text,encoding='utf-8')
    append_notes(e2,SPC_NOTE,REASON_NOTE)
    text=fn.read_text(encoding='utf-8')
    if CAPTION_NOTE not in text:text=text.replace('![오류 구성](figure1_error_structure.png)','![오류 구성](figure1_error_structure.png)\n\n'+CAPTION_NOTE,1)
    fn.write_text(text,encoding='utf-8');append_notes(fn,REASON_NOTE)
    append_notes(sh,CLAMP_NOTE,REASON_NOTE)
    # The evidence CSVs are existing immutable chart inputs; only captions change.
    target=pd.concat([pd.read_csv(out/'cn7_logreg_e2_q1_prediction_evidence.csv',keep_default_na=False),
                      pd.read_csv(out/'cn7_e2_auxiliary_decision_evidence.csv',keep_default_na=False)],ignore_index=True)
    assert len(target)==192
    figures(target)
    plots(cases,None,None,None,figure1_only=True)
    assert all(sha256(ROOT/p)==h for p,h in protected.items())
    new_e2=e2.read_text(encoding='utf-8')
    assert new_e2[new_e2.index('## Q3.'):new_e2.index('## 보조모델 비교')]==q34
    check=dict(positive_shots=positive,total_shots=total,positive_percent=100*positive/total,
        protected_output_files=len(protected),protected_outputs_sha256=protected,numeric_csv_files_unchanged=True,
        plot_data_unchanged=True,q3_q4_text_unchanged=True,reason_inventory_matches_public=True,
        training_executed=False,threshold_changed=False,replay_executed=False,
        updated_files=[str(p.relative_to(ROOT)) for p in mutable])
    verification.write_text(json.dumps(check,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in check.items() if k!='protected_outputs_sha256'},indent=2,ensure_ascii=False))


def main():
    out=ROOT/'outputs/analysis'
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    data=load_all(cfg,ROOT/cfg['data_dir']);raw=data['feature_cols'];features,_=select_base_features(data,raw,True)
    archive=Path('C:/Users/82102/Downloads/04. Dataset_Molding.zip')
    pub=_public.read_csv(archive,'moldset_labeled.csv');ld=_public.read_csv(archive,'labeled_data.csv')
    lab=assign_shots(data['labeled']['cn7'],raw,'cn7')
    linked=X.link_side(lab,pub,raw);mu,sd,error=X.zscale_params(linked,raw)
    assert error<1e-8
    matched=pub.set_index('Unnamed: 0').loc[lab.row_id]
    assert np.array_equal(matched.PassOrFail.to_numpy().astype(int),lab.y)
    # Formula audit of the already-defined SPC baseline, not model.fit().
    normal=linked.loc[linked.y.eq(0),features].to_numpy(float)
    center=normal.mean(axis=0);scale=np.maximum(normal.std(axis=0),.01)
    ref=np.sort(np.max(abs((normal-center)/scale),axis=1));refsd=float(np.std(ref)) or 1.
    full=pd.read_csv(ROOT/'outputs/inspection/delay_replay_log_full.csv',keep_default_na=False)
    saved=full[full['product'].eq('cn7')&full.model.eq('spc_maxz')&full.threshold.eq('top2')&full.delay.eq(0)&full.cap_per10.eq(2)&full.rule.eq('model')].sort_values('t')
    checks=[];driver=None
    for fixed in [True,False]:
        X.RPM_FIX=fixed
        future=X.future_rows(ld,linked,'cn7',raw,mu,sd)
        z=(future[features].to_numpy(float)-center)/scale
        scores=ecdf_score(ref,np.max(abs(z),axis=1),refsd)
        shots=future[['shot_id','TimeStamp']].assign(score=scores).groupby('shot_id').agg(ts=('TimeStamp','min'),score=('score','max')).sort_values('ts').reset_index()
        shots['rank']=shots.score.rank(ascending=False,method='min').astype(int)
        assert np.array_equal(shots.shot_id,saved.shot_id)
        target=shots.iloc[1173];delta=np.abs(shots.score.round(5).to_numpy()-saved.score.to_numpy())
        q=future[future.shot_id.eq(target.shot_id)]
        zs=(q[features].to_numpy(float)-center)/scale
        assert np.allclose(zs,zs[:1])
        top=int(np.argmax(abs(zs[0])))
        checks.append(dict(rpm_fix=fixed,equivalent_X3_output='x3_future.csv' if fixed else 'x3_future_norpmfix.csv',shot_id=target.shot_id,
            score=float(target.score),score_rounded5=round(float(target.score),5),rank=int(target['rank']),future_shots=len(shots),
            max_z_feature=features[top],signed_z=float(zs[0,top]),saved_score_matches=int((delta<1e-12).sum()),total_scores=len(delta),max_saved_score_error=float(delta.max())))
        if fixed:
            assert round(float(target.score),5)==.99289 and target['rank']==30 and np.all(delta<1e-12)
            assert np.array_equal(shots['rank'],saved.score_rank)
            driver=pd.DataFrame(dict(feature=features,competition_scaled_value=q[features].iloc[0].to_numpy(),normal_mean=center,normal_sd_floored=scale,signed_z=zs[0],abs_z=abs(zs[0])))
    X.RPM_FIX=True
    assert checks[0]['max_z_feature']=='Average_Screw_RPM'
    driver.sort_values('abs_z',ascending=False).to_csv(out/'cn7_e2_spc_driver_verification.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'cn7_e2_spc_input_verification.csv',index=False)
    # Public metadata audit, matched on competition row ID; no new model inputs.
    meta=matched[['TimeStamp','PART_NAME','Reason','PassOrFail','Clamp_Close_Time']].reset_index().rename(columns={'Unnamed: 0':'row_id'})
    meta['shot_id']=lab.shot_id.to_numpy()
    assert meta.groupby('shot_id').Clamp_Close_Time.nunique().eq(1).all()
    vals=sorted(meta.Clamp_Close_Time.unique().tolist());assert len(vals)==3
    defect=meta[meta.PassOrFail.astype(int).eq(1)]
    prod=ld[ld.EQUIP_NAME.eq(X.EQUIP)&ld.PART_NAME.str.startswith('CN7')].drop_duplicates(['TimeStamp','PART_NAME']).copy()
    prod['date']=pd.to_datetime(prod.TimeStamp).dt.date.astype(str)
    daily=prod.groupby('date').Average_Screw_RPM.agg(['count','min','median','max']);daily.to_csv(out/'cn7_public_rpm_daily_audit.csv')
    meta.to_csv(out/'cn7_public_metadata_audit.csv',index=False,encoding='utf-8-sig')
    result=dict(public_archive=str(archive),public_members=['dataset/moldset_labeled.csv','dataset/labeled_data.csv'],
        matched_rows=len(meta),labels_match=True,standardization_max_error=error,clamp_close_time_unique_values=vals,
        clamp_scope='CN7 competition-linked rows of public moldset_labeled.csv',
        defect_rows=len(defect),defect_shots=defect.shot_id.nunique(),defect_start=defect.TimeStamp.min(),defect_end=defect.TimeStamp.max(),
        defect_dates=sorted(pd.to_datetime(defect.TimeStamp).dt.date.astype(str).unique().tolist()),
        spc_input_checks=checks,spc_saved_log_sha256=sha256(ROOT/'outputs/inspection/delay_replay_log_full.csv'),
        e2_input_note='E2 calls future_rows with default RPM_FIX=True directly; x3_future.csv is a summary output, not the E2 input file. Same corrected preprocessing path.',
        training_executed=False,replay_executed=False,spc_validation_method='numpy arithmetic of existing mean/std/max|z|/ECDF formulas; no estimator fit')
    (out/'b_final_metadata_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
