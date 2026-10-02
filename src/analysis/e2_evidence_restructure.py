"""Reorganize existing E2 evidence; no model or replay execution."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]

def main():
 out=ROOT/'outputs/analysis';ins=ROOT/'outputs/inspection'
 log=pd.read_csv(ins/'delay_replay_log.csv');log['csv_line']=np.arange(len(log))+2
 ev=pd.read_csv(ins/'delay_replay_events.csv');ev['event_csv_line']=np.arange(len(ev))+2
 ev=ev[ev['product'].eq('cn7')&ev.event.eq(1)];assert ev.start_shot.eq(1173).all() and ev.end_shot.eq(1173).all()
 old=pd.read_csv(out/'e2_shot1173_detection_detail.csv');old=old[old.shot.eq(1173)]
 rows=[];keys=['model','threshold','delay','cap_per10','rule']
 for _,e in ev.iterrows():
  match=log['product'].eq('cn7')
  om=pd.Series(True,index=old.index)
  for k in keys:match &= log[k].eq(e[k]);om &= old[k].eq(e[k])
  condition=log[match];target=condition[condition.t.eq(1173)];prior=condition[condition.t.between(1164,1172)]
  r={k:e[k] for k in keys};r.update(shot=1173,timestamp=e.start_ts,event_defect_parts_found=e.defect_parts_found,event_source='outputs/inspection/delay_replay_events.csv',event_csv_line=e.event_csv_line,saved_target_log_available=len(target)==1,capacity_direct_field_present=False)
  if len(target):
   assert len(target)==1 and set(prior.t)==set(range(1164,1173))
   t=target.iloc[0];used=int(prior.inspected_parts.sum());r.update(inspected=bool(t.inspected_parts),inspected_parts=t.inspected_parts,inspected_side='none' if t.inspected_parts==0 else 'not explicitly stored',found=t.found,request_reason=t.reason,score=t.score,target_log_csv_line=t.csv_line,prior_used_from_saved_log=used,capacity_derived_from_saved_log=int(e.cap_per10)-used,prior_log_evidence='; '.join(f'line {int(p.csv_line)}:t={int(p.t)},inspected_parts={int(p.inspected_parts)}' for _,p in prior.iterrows()),capacity_evidence_status='derived from complete saved t-9..t-1 inspection counts using replay window rule')
  else:r.update(inspected=np.nan,inspected_parts=np.nan,inspected_side='unknown from saved logs',found=np.nan,capacity_evidence_status='not provable from saved logs: condition detailed log absent')
  o=old[om];assert len(o)==1
  r.update(previous_reconstruction_side=o.inspected_side.iloc[0],previous_reconstruction_capacity=o.room_before.iloc[0],reconstruction_is_direct_log_evidence=False)
  rows.append(r)
 cap=pd.DataFrame(rows);cap.to_csv(out/'e2_shot1173_capacity_evidence.csv',index=False,encoding='utf-8-sig')
 loc=pd.read_csv(out/'e2_event_comparison.csv');feat=pd.read_csv(out/'e2_missed_oneside_feature_compare.csv');names=sorted(feat.feature.unique());assert len(names)==23
 loc['n_normal_controls']=2*loc.n_each_side;loc['reference_definition']='same CN7 production day; 5 or 10 normal shots before episode and after episode; both parts normal'
 loc['standardization']='(raw value - control median) / control sample SD ddof=1; zero SD excluded'
 loc['distance_definition']='sqrt(sum(local_z squared)) on nonzero-SD features'
 loc['ratio_definition']='defect distance / maximum distance of controls to same center'
 loc['source']='outputs/analysis/e2_event_comparison.csv';loc['source_csv_line']=np.arange(len(loc))+2
 loc.to_csv(out/'e2_shot1173_local_distance_evidence.csv',index=False,encoding='utf-8-sig')
 text=['# E2 shot1173 오류분석 — 질문과 증거를 따라가는 검증','이 문서는 기존 결과를 재구성한다. 새로운 모델 학습·점수 계산·replay 실행은 하지 않았다. 저장 원본 로그, 기존 분석 결과, 규칙 재구성을 구분한다.']
 def section(title,q,h,method,source,result,verdict,interp):
  text.extend(['## '+title,'### 질문',q,'### 초기 가설',h+' (검증을 위한 가설이며 사실 진술이 아님)','### 확인 방법',method,'### 사용 데이터/파일',source,'### 실제 관측 결과',result,'### 가설 판정',verdict,'### 해석',interp])
 section('1. 어떤 불량이며 LH/RH의 입력은 다른가?','1173이 한쪽 미성형이며, 모델 입력에 side를 구분할 차이가 있었는가?','LH/RH 공정값 차이로 품질 차이를 설명할 수 있다.','기존 원본 audit의 timestamp/제품/side 라벨과 23개 공정값 일치 검증을 확인한다.','공개 labeled_data.csv; e2_missed_oneside.py; e2_missed_oneside_case.csv','2020-11-03 04:44:17, CN7, LH=N/미성형, RH=Y. 23개 machine-level 값은 동일. 1173은 미래 replay의 0-based 순번으로 competition shot_id와 다르다.','해당 입력 차이 가설은 지지되지 않는다.','현재 machine-level 변수만으로 해당 한쪽 미성형 outcome을 LH/RH 사이에서 구분할 정보가 관측되지 않았다. cavity-local 원인을 확정하지 않는다.')
 section('2. 결합모델은 검사 요청을 만들었는가?','결합모델의 누락은 점수 미달인가, 검사 요청 이후 문제인가?','모델이 요청했지만 capacity 때문에 검사하지 못했다.','저장된 상세 로그에서 score/reason/inspected_parts/found와 OOF 임계값을 확인한다.','delay_replay_log.csv; delay_replay.csv; pipeline_logreg_spc.csv; e2_shot1173_capacity_evidence.csv','logreg_spc/top2/cap2의 16조건에서 score=0.49665, reason 비어 있음, inspected_parts=0, found=0. top2 임계값=0.96231866, top5=0.88875581.','저장된 16조건에서는 요청 후 거절 가설보다 모델·추적 요청 부재가 직접 관측된다.','이 조건의 누락을 capacity만으로 설명하면 안 된다. top5 및 다른 cap의 상세 결정 로그는 별도로 저장되지 않았다.')
 section('3. SPC도 shot 자체를 놓쳤는가?','SPC 일부 조건 포착이 실제로 존재하는가?','모든 모델이 해당 shot을 감지하지 못했다.','event1이 단일 shot임을 확인하고 조건별 defect_parts_found를 읽는다. 기존 점수 재구성은 별도 증거 수준으로 표시한다.','delay_replay_events.csv; e2_shot1173_detection_detail.csv','SPC는 top2/top5 각각 32조건 중 8포착, 24미포착. 포착은 cap4, model/model_track, delay 0/1/3/5. 기존 통계/ECDF 재구성 score=0.992887, 최대변수 Average_Screw_RPM z=-5.890276. 이 값은 저장된 SPC 상세 로그가 아니다.','모든 모델 감지 실패라는 가설은 기각한다.','SPC는 과거 정상 기준 편차에 반응했을 가능성을 기존 재구성이 뒷받침한다. RPM 10배 보정 전처리 가정이 포함되며 미성형 원인 증거는 아니다.')
 shown=cap[cap.saved_target_log_available][['threshold','delay','cap_per10','rule','inspected_parts','found','target_log_csv_line','prior_used_from_saved_log','capacity_derived_from_saved_log']]
 section('4. Capacity와 side 선택이 누락을 설명하는가?','shot1173에서 capacity가 실제 0이었는가? RH만 검사하고 LH를 놓쳤는가?','SPC flag 후 용량 소진 또는 RH 단독 검사 때문에 놓쳤다.','(1) event 조건과 상세 log를 product/model/threshold/delay/cap/rule로 연결. (2) t=1173 상세 행의 inspected_parts/found/reason 확인. (3) replay의 used=sum(n for s,n in recent if s>t-10) 정의에 따라 t=1164..1172의 저장 inspected_parts 합을 계산. room=cap-used. (4) 해당 9행이 모두 존재할 때만 산출. room 자체는 저장 로그 필드가 아니며 각 행 번호·t·검사수를 증거 CSV에 남겼다.','delay_replay_events.csv (조건별 사건 포착); delay_replay_log.csv (일부 조건만 상세 저장); scripts/e2_delay_replay.py (window와 side 순서 정의); e2_shot1173_capacity_evidence.csv (192조건, 1-based 헤더 포함 CSV 행 번호)','직접 저장된 상세 로그는 logreg_spc/top2/cap2의 16조건뿐이다.\n\n```\n'+shown.to_string(index=False)+'\n```\n\nSPC 64조건은 상세 log가 없다. event CSV의 defect_parts_found=0만으로 무검사와 정상 RH 단독검사를 구별할 수 없다. 포착 조건은 불량 LH가 검사됐다는 사실을 지지하지만 검사된 모든 side를 직접 기록하지 않는다. 이전 재구성의 room=0/none은 previous_reconstruction_* 열로만 보존하고 direct evidence로 사용하지 않는다.','SPC 미포착 48조건(top2/top5 합계)의 capacity=0과 RH 단독검사 부재는 현재 저장 원본 로그만으로 직접 증명할 수 없다. 기존 규칙 재구성에서는 room=0·무검사였고 event 포착수는 일치했지만 이는 상세 로그 관측과 동등하지 않다.','앞선 요약의 “모든 SPC 미포착은 capacity=0”은 재구성에 조건부인 결론으로 낮춘다. event 집계 일치는 내부 검사 경로의 유일성을 증명하지 않는다. 직접 확정하려면 기존 실행의 SPC 결정 로그에 used/room/chosen_side/want_model/want_track가 필요하다. 이번에는 새 replay를 실행하지 않았다.')
 section('5. 주변 정상에서 얼마나 벗어났는가?','event0과 shot1173은 같은 종류의 국소 공정 이탈인가?','shot1173도 startup episode처럼 주변 정상에서 크게 이탈한다.','같은 제품·생산일의 정상 reference를 정한다. event0은 연속 5불량 episode 전체 직전/직후 정상 각 10 shot(민감도 각 5), shot1173은 해당 shot 전후 정상 각 10/5 shot. ±10은 시간상의 모든 10 shot이 아니라 가장 가까운 정상 10개씩이며 timestamp 범위를 표에 기록한다. LH/RH 한 shot, 두 부품 정상인 reference만 사용. 23개 원단위 feature: '+', '.join(names)+'. 각 feature z=(값−reference 중앙값)/reference sample SD(ddof=1). robust MAD z가 아니다. SD=0 제외. distance=sqrt(sum(z²)), covariance 없는 표준화 유클리드 거리. 정상도 같은 중심/SD로 거리 계산. 비율=defect distance/정상 reference 최대거리. 1 초과는 선택한 정상 reference 중 최대보다 멀다는 기술적 의미만 갖는다.','e2_missed_oneside_feature_compare.csv; e2_event_comparison.csv; e2_shot1173_local_distance_evidence.csv; 기존 e2_missed_oneside.py','```\n'+loc[['event','TimeStamp','n_each_side','distance','normal_distance_max','distance_ratio','n_dimensions']].to_string(index=False)+'\n```\n\n전후10개 기준 event0 비율 3.56~41.63, shot1173 0.688. 전후5개 기준 shot1173 1.072. event0 22차원, shot1173 21차원. SD=0 제외로 차원이 다르고 작은 SD가 큰 거리값을 만들 수 있다.','shot1173이 event0과 같은 큰 국소 이탈이라는 가설은 지지되지 않는다. 작은 편차와 window 민감성은 남는다.','사후 정상 선택에 label과 이후 shot이 사용됐으므로 온라인 탐지 규칙이 아니다. 정상 기준 자기포함, reference 소표본, 상관변수 중복가중 때문에 검정통계량이나 이상 확률로 해석하면 안 된다. 날짜별 별도 scale이므로 절대 거리 간 직접 크기 비교보다 각 reference 대비 상대비율을 설명하되 동일 물리 변화량으로 해석하지 않는다.')
 section('6. Ambiguity와 단독 LR 점수로 보강 가능한가?','기존 파일만으로 이 미래 shot의 모든 모델 점수와 Ambiguity를 연결할 수 있는가?','competition row_id=1173 또는 결합점수로 누락값을 복원할 수 있다.','기존 연결 기간 및 score 저장 범위를 확인한다.','ambiguity_row_summary.csv; e2_shot1173_detection_detail.csv; pipeline_logreg.csv','Ambiguity의 CN7 competition 기간은 10/16~10/20로 대상 11/03과 다르다. LR 미래 score는 미저장. 결합점수는 LR 원확률과 SPC의 단순 평균이 아니다.','강제 ID 연결과 LR 역산은 불가.','관련 값은 미확인으로 유지한다. 저장 OOF 점수를 미래 학습모델의 점수라고 대체하지 않는다.')
 text+=['## 종합 결론','결합모델 상세 저장 조건에서는 점수 미달/요청 부재가 관측된다. SPC는 사건 포착 조건이 존재하며, 규칙 재구성은 감지 후 용량 소진을 설명하지만 해당 SPC capacity/side 경로는 저장 원본 상세 로그만으로 직접 확정되지 않는다. shot1173은 event0보다 국소 이탈이 작으며, 동일 machine X의 LH/RH outcome 차이는 이 입력만으로 구분되지 않는다. 원인 확정이나 일반화는 하지 않는다.']
 (out/'e2_error_analysis.md').write_text('\n\n'.join(text),encoding='utf-8')
 print(shown.to_string(index=False));print('rows',len(cap),'local',len(loc))
if __name__=='__main__':main()
