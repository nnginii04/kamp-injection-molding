"""Read-only E2 missed-case audit; never import or fit prediction models."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import _public


def main():
    out=ROOT/'outputs/analysis';cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    source=_public.public_path(cfg)
    if not source.exists():source=Path.home()/'Downloads/04. Dataset_Molding.zip'
    c=pd.read_csv(ROOT/cfg['data_dir']/cfg['products']['cn7']['labeled'])
    feats=[f for f in c if f not in ['Unnamed: 0','PassOrFail','Clamp_Open_Position']];assert len(feats)==23
    raw=_public.read_csv(source,'labeled_data.csv')
    raw=raw[raw.EQUIP_NAME.eq('650톤-우진2호기') & raw.PART_NAME.str.startswith('CN7')].copy()
    assert raw.groupby(['TimeStamp','PART_NAME'])[['PassOrFail','Reason']].nunique(dropna=False).le(1).all().all()
    raw=raw.drop_duplicates(['TimeStamp','PART_NAME']);raw.TimeStamp=pd.to_datetime(raw.TimeStamp)
    target=pd.Timestamp('2020-11-03 04:44:17');pair=raw[raw.TimeStamp.eq(target)]
    assert len(pair)==2 and pair[feats].nunique(dropna=False).eq(1).all()
    assert pair.loc[pair.PART_NAME.str.endswith('LH'), 'PassOrFail'].tolist()==['N']
    assert pair.loc[pair.PART_NAME.str.endswith('LH'), 'Reason'].tolist()==['미성형']
    assert pair.loc[pair.PART_NAME.str.endswith('RH'), 'PassOrFail'].tolist()==['Y']
    print('PAIR',pair[['TimeStamp','PART_NAME','PassOrFail','Reason']].to_string(index=False))
    assert raw.groupby('TimeStamp')[feats].nunique(dropna=False).le(1).all().all()
    shots=raw.sort_values('TimeStamp').drop_duplicates('TimeStamp').set_index('TimeStamp')
    shots['normal']=raw.groupby('TimeStamp').PassOrFail.agg(lambda s:s.eq('Y').all())
    shots['n_parts']=raw.groupby('TimeStamp').size()
    log=pd.read_csv(ROOT/'outputs/inspection/delay_replay_log.csv');events=pd.read_csv(ROOT/'outputs/inspection/delay_replay_events.csv');overall=pd.read_csv(ROOT/'outputs/inspection/delay_replay.csv')
    log=log[log['product'].eq('cn7')]; events=events[events['product'].eq('cn7')]; overall=overall[overall['product'].eq('cn7')]
    one=log[log.t.eq(1173)];assert one.ts.eq(str(target)).all()
    ev=events[events.event.eq(1)].merge(overall,on=['product','model','threshold','delay','cap_per10','rule'],suffixes=('','_overall'),validate='one_to_one')
    cols=['product','model','threshold','delay','cap_per10','rule','event','start_shot','start_ts','defect_parts_found','thr']
    case=ev[cols].copy();case['target_score']=np.where(case.model.eq('logreg_spc'),one.score.iloc[0],np.nan)
    case['score_source']=np.where(case.model.eq('logreg_spc'),'stored replay log (rounded 5 decimals)','not persisted for this model')
    case['LH_label']='N';case['LH_reason']='미성형';case['RH_label']='Y';case['same_machine_X']=True
    case['score_minus_threshold']=case.target_score-case.thr
    case['detail_log_available']=case.model.eq('logreg_spc') & case.threshold.eq('top2') & case.cap_per10.eq(2)
    case.to_csv(out/'e2_missed_oneside_case.csv',index=False)
    times0=pd.to_datetime(events[events.event.eq(0)].start_ts.unique());assert len(times0)==1
    t0=times0[0];d0=raw[(raw.TimeStamp.dt.date==t0.date()) & raw.PassOrFail.eq('N')].TimeStamp.sort_values().unique();assert len(d0)==5
    comparisons=[];rows=[]
    for event,times in [(0,pd.DatetimeIndex(d0)),(1,pd.DatetimeIndex([target]))]:
        for n in [10,5]:
            pool=shots[(shots.index.date==times[0].date()) & shots.normal & shots.n_parts.eq(2)]
            before=pool[pool.index<times.min()].tail(n);after=pool[pool.index>times.max()].head(n);assert len(before)==len(after)==n
            control=pd.concat([before,after]);med=control[feats].median();sd=control[feats].std(ddof=1)
            z=(shots.loc[times,feats]-med)/sd.replace(0,np.nan)
            nz=(control[feats]-med)/sd.replace(0,np.nan);valid=sd.gt(0)
            dist=np.sqrt(z.loc[:,valid].pow(2).sum(axis=1));nd=np.sqrt(nz.loc[:,valid].pow(2).sum(axis=1))
            for i,t in enumerate(times):
                sl=log[log.ts.eq(str(t))];sc=float(sl.score.iloc[0])
                comparisons.append(dict(event=event,TimeStamp=t,n_each_side=n,score_logreg_spc=sc,distance=float(dist.loc[t]),normal_distance_max=float(nd.max()),normal_distance_median=float(nd.median()),distance_ratio=float(dist.loc[t]/nd.max()),n_dimensions=int(valid.sum()),max_abs_z=float(z.loc[t].abs().max()),normal_before_start=before.index.min(),normal_before_end=before.index.max(),normal_after_start=after.index.min(),normal_after_end=after.index.max()))
                for f in feats:
                    rows.append(dict(event=event,TimeStamp=t,n_each_side=n,feature=f,shot_value=shots.loc[t,f],control_median=med[f],difference=shots.loc[t,f]-med[f],control_sd=sd[f],local_z=z.loc[t,f],abs_local_z=abs(z.loc[t,f]),zero_sd=sd[f]==0,zero_sd_changed=bool(sd[f]==0 and shots.loc[t,f]!=med[f]),before_median=before[f].median(),after_median=after[f].median()))
    feature=pd.DataFrame(rows);compare=pd.DataFrame(comparisons)
    feature.sort_values(['event','n_each_side','abs_local_z'],ascending=[True,False,False]).to_csv(out/'e2_missed_oneside_feature_compare.csv',index=False)
    compare.to_csv(out/'e2_event_comparison.csv',index=False)
    thresholds=[]
    for model in ['logreg','logreg_spc','spc_maxz']:
        p=pd.read_csv(ROOT/f'outputs/predictions/pipeline_{model}.csv');p=p[p['product'].eq('cn7') & p.cv_mode.eq('block') & p['repeat'].eq(0)]
        for name,q in [('top2',.98),('top5',.95)]:
            exact=float(p.groupby('shot_id').y_score.max().quantile(q));stored=float(overall[(overall.model==model)&(overall.threshold==name)].thr.iloc[0]);assert abs(exact-stored)<=.0000051
            thresholds.append(f'{model} {name}: {exact:.8f} (stored {stored:.5f})')
    a=pd.read_csv(out/'ambiguity_row_summary.csv');am=json.loads((out/'ambiguity_manifest.json').read_text(encoding='utf-8'))
    pub=_public.read_csv(source,'moldset_labeled.csv');mapped=c[['Unnamed: 0']].merge(pub[['Unnamed: 0','TimeStamp']],on='Unnamed: 0',validate='one_to_one')
    no_target=not pd.to_datetime(mapped.TimeStamp).eq(target).any();assert no_target
    stats=case.groupby(['model','threshold']).defect_parts_found.agg(['count','sum'])
    top=feature[(feature.event==1)&feature.n_each_side.eq(10)].sort_values('abs_local_z',ascending=False).head(8)
    lines=['# E2 CN7 one-side 미성형 누락 사례 분석','', '## 사실',
      '- 대상: 미래기간 시간순 0-based shot 1173 = cn7f_2020-11-03 04:44:17. competition shot ID와 다른 체계이다.',
      '- 공개 원본 labeled_data.csv에서 같은 설비·제품을 선택하고 TimeStamp + PART_NAME 중복 제거. LH=N/미성형, RH=Y/Reason 없음. 23개 machine feature가 모두 정확히 같다.',
      '- E2 조건별 포착 수(각 행의 sum은 반복 조건에서 포착된 횟수이며 독립 사건 수가 아님):','```',stats.to_string(),'```',
      '- 저장 상세 로그는 logreg_spc/top2/cap=2의 delay 0,1,3,5 × 4개 rule에만 존재한다. 해당 16조건 모두 score=0.49665, reason 없음, inspected_parts=0, found=0. 따라서 이 조건에서는 용량 거절보다 앞 단계인 모델/추적 요청 부재가 직접적인 누락 경로이다.',
      '- block CV repeat=0의 OOF shot max 점수 분위수로 임계값을 재계산하여 기존 요약과 반올림 오차 이내 일치 확인:','```',*thresholds,'```',
      '- E2 scripts/e2_delay_replay.py는 competition 전체로 각 모델을 fit한 후 미래기간을 predict한다. pipeline_*.csv는 미래점수가 아니라 OOF 임계값 기준 자료이다. 이번 audit는 fit/predict/replay를 실행하지 않았다.',
      '- external_experiments.future_rows: 동일 설비/제품, competition 종료 이후, TimeStamp+PART_NAME 중복 제거, competition 원단위 평균/SD로 변환. Average_Screw_RPM<100이면 ×10 보정. E2 모델 입력 base에는 side 없음.',
      '- logreg: StandardScaler + balanced LogisticRegression(C=0.3). spc_maxz: 정상 train 행의 평균/SD(sd_floor=0.01) 기준 max|z|의 ECDF. logreg_spc: train 분포 기준 logreg 백분위와 SPC 점수를 0.5씩 평균. 상세 정의는 src/models.py.',
      '- 단독 logreg/SPC 미래 점수 및 결합 성분 점수는 현재 저장 파일에 없어 정확한 값을 복원하지 않았다. SPC 일부 조건에서의 포착은 기존 event 결과로 확인하며 낮은 국소거리와 모순되지 않는다(과거 정상 기준과 당일 국소 기준이 다름).',
      '- 같은 날짜, episode 전후 양쪽 정상 10 shot씩(민감도 5씩), 각 shot에서 LH/RH를 하나로 합침. local_z=(값−대조군 중앙값)/대조군 표본 SD. zero SD는 NaN과 변경 여부 별도 표시. 거리=유효 feature z의 유클리드 norm, covariance 학습 없음.',
      '### shot 1173 주요 편차','```',top[['feature','shot_value','control_median','difference','local_z']].to_string(index=False),'```',
      '### event 비교','```',compare[compare.n_each_side.eq(10)][['event','TimeStamp','score_logreg_spc','distance','normal_distance_max','distance_ratio']].to_string(index=False),'```',
      f'- 기존 Ambiguity는 competition 2393행 shot-fold OOF이며 CN7 연결 기간은 {mapped.TimeStamp.min()} ~ {mapped.TimeStamp.max()}. 목표 시각은 포함되지 않는다. 미래 shot1173을 competition row_id=1173 등으로 강제 연결하지 않았다.',
      '- SPC 포착 조건은 top2/top5 각각 cap_per10=4, rule=model 또는 model_track, 모든 delay(0/1/3/5)이다. 다른 조건은 미포착이다.',
      '- 전후 정상 5개씩으로 줄이면 event1 거리/정상최대는 1.072배(5.736/5.350), 10개씩에서는 0.688배(4.483/6.511)이다. 작은 편차의 평가는 window에 민감하다.',
      '- event0의 마지막 shot은 국소거리 이탈에도 결합 score=0.54702로 임계값 미만이다. 큰 국소 이탈과 모델 포착을 동일시할 수 없다.',
      '- SD=0인 변수는 거리에서 제외했다(event0 22차원, event1 21차원). event0 두 shot의 Clamp_Close_Time은 정상 고정값 7.12에서 7.18로 변했으며 별도 플래그에 기록했다. 거리 크기는 공분산·측정해상도를 보정한 검정통계량이 아니다.',
      '## 해석',
      '- event0은 초기허용불량 5 shot의 큰 국소 공정 이탈 사례이다. event1은 같은 machine X를 공유하는 LH/RH의 품질 결과가 다른 사례이며, 국소 정상 중심 거리와 각 변수 편차는 위 표와 같다.',
      '- 현재 machine-level 변수만으로는 해당 한쪽 미성형 outcome을 LH/RH 사이에서 구분할 정보가 관측되지 않았다. 이는 cavity-local 원인의 확정이나 모든 잠재 공정신호의 부재를 뜻하지 않는다.',
      '## 한계',
      '- event1은 단일 사건이며 전후 정상은 사후 설명용이다. 지역 기준 거리와 과거 학습 기준 score는 다른 척도이다. 정상 최대를 넘지 않는다고 정상임을 증명하지 않는다.',
      '- 미래 score 상세 로그는 일부 조건/모델만 저장됐으며 score는 소수점 5자리이다. SPC의 개별 누락이 용량/추적/RH 우선 중 무엇인지 로그 없이 확정하지 않았다.',
      '- future_rows의 RPM 기록눈금 보정은 E2 전처리 가정이다. 여기의 국소 비교는 같은 날 원단위를 사용해 전후 동일 단위로 비교했다.',
      '- 과거 Ambiguity는 미래기간 점수가 아니다. 원본 결합 메타정보와 다른 모집단에 대한 점수 재사용은 하지 않았다.',
      '- E2 자체가 탐색적 미래기간 검증이며 독립 최종 테스트가 아니다. 모델 재학습·전체 파이프라인 재실행·타 영역 코드 수정 없음.']
    (out/'e2_error_analysis.md').write_text('\n\n'.join(lines),encoding='utf-8')
    print(stats.to_string());print(top[['feature','shot_value','control_median','difference','local_z']].to_string(index=False));print(compare.to_string(index=False))

if __name__=='__main__':main()
