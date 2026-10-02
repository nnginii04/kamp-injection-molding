"""Descriptive per-product common OOF error conditions; no fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_all,assign_shots
from src.features import select_base_features
ROOT=Path(__file__).resolve().parents[2]

def main():
 out=ROOT/'outputs/analysis';inv=pd.read_csv(out/'error_case_inventory.csv');cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
 data=load_all(cfg,ROOT/cfg['data_dir']);features,_=select_base_features(data,data['feature_cols'],True);assert len(features)==23
 # Independently verify the existing repeat-majority inventory against saved OOF decisions.
 for model in ['logreg','logreg_spc']:
  p=pd.read_csv(ROOT/f'outputs/predictions/pipeline_{model}.csv');p=p[p.cv_mode.eq('block')]
  v=p.groupby(['product','shot_id','repeat']).y_pred.max().groupby(['product','shot_id']).mean()
  check=inv.set_index(['product','shot_id'])[model+'_positive_repeat_fraction'];assert np.allclose(v.reindex(check.index),check)
 tables=[];effects=[];locals_=[]
 for product,lab in data['labeled'].items():
  lab=assign_shots(lab,data['feature_cols'],product)
  assert lab.groupby('shot_id')[features].nunique().le(1).all().all()
  shots=lab.drop_duplicates('shot_id')[['shot_id',*features]]
  d=inv[inv['product'].eq(product)].merge(shots,on='shot_id',validate='one_to_one');assert len(d)==len(shots)
  d['date']=pd.to_datetime(d.timestamp).dt.date.astype(str)
  groups={'FN':d[d.error_type.eq('FN/FN')],'FP':d[d.error_type.eq('FP/FP')],'TP':d[d.error_type.eq('TP/TP')],'TN':d[d.error_type.eq('TN/TN')]}
  groups['one_side_FN']=groups['FN'][groups['FN'].side_pattern.eq('one-side')];groups['both_side_TP']=groups['TP'][groups['TP'].side_pattern.eq('both-side')]
  for group,g in groups.items():
   for f in features:
    q=g[f].quantile([.25,.5,.75]);tables.append(dict(product=product,group=group,feature=f,n_shots=len(g),n_dates=g.date.nunique(),median=q.loc[.5],q25=q.loc[.25],q75=q.loc[.75],iqr=q.loc[.75]-q.loc[.25],units='competition supplied scale (not physical units)'))
  for ga,gb in [('FN','TP'),('FP','TN'),('one_side_FN','both_side_TP')]:
   a,b=groups[ga],groups[gb]
   for f in features:
    x,y=a[f].to_numpy(),b[f].to_numpy();na,nb=len(x),len(y)
    if na and nb:
     diff=float(np.median(x)-np.median(y));scale=float((np.quantile(y,.75)-np.quantile(y,.25))/1.349)
     delta=float(np.sign(x[:,None]-y[None,:]).mean())
     loo=[np.sign(np.median(np.delete(x,i))-np.median(y))==np.sign(diff) for i in range(na)] if na>1 else []
     days=[np.sign(g[f].median()-np.median(y))==np.sign(diff) for _,g in a.groupby('date')]
     # Descriptive categories only, no significance or cause claim.
     tag='표본이 적어 판단 어려움' if min(na,nb)<10 else '반복적으로 관찰되는 패턴(표본 내)' if abs(delta)>=.33 and all(loo) and len(days)>=2 and all(days) else '약한 경향 또는 날짜별 불일치'
     effects.append(dict(product=product,comparison=ga+'_vs_'+gb,feature=f,n_target=na,n_reference=nb,target_median=np.median(x),reference_median=np.median(y),median_difference=diff,reference_robust_scale=scale,robust_standardized_difference=diff/scale if scale>1e-12 else np.nan,reference_zero_iqr=scale<=1e-12,cliffs_delta=delta,target_fraction_above_reference_median=float((x>np.median(y)).mean()),target_fraction_below_reference_median=float((x<np.median(y)).mean()),loo_direction_agreement=np.mean(loo) if loo else np.nan,n_target_dates=len(days),date_direction_agreement=np.mean(days),evidence_category=tag))
    else:effects.append(dict(product=product,comparison=ga+'_vs_'+gb,feature=f,n_target=na,n_reference=nb,evidence_category='비교군 없음'))
  # Label-based normal selection is retrospective only. Up to 5 before + 5 after, same date/product.
  normal=d[d.y_true.eq(0)].sort_values('timestamp')
  for _,r in d[d.error_type.isin(['FN/FN','FP/FP'])].iterrows():
   pool=normal[(normal.date==r.date)&normal.shot_id.ne(r.shot_id)]
   pre=pool[pool.timestamp<r.timestamp].tail(5);post=pool[pool.timestamp>r.timestamp].head(5);ctrl=pd.concat([pre,post])
   for f in features:
    med=ctrl[f].median();sd=ctrl[f].std(ddof=1)
    locals_.append(dict(product=product,shot_id=r.shot_id,error_type=r.error_type,feature=f,n_before=len(pre),n_after=len(post),balanced_complete=len(pre)==len(post)==5,value=r[f],control_median=med,difference=r[f]-med,local_standardized_difference=(r[f]-med)/sd if sd>1e-12 else np.nan))
 t=pd.DataFrame(tables);e=pd.DataFrame(effects);l=pd.DataFrame(locals_)
 t.to_csv(out/'fn_fp_condition_table.csv',index=False,encoding='utf-8-sig');e.to_csv(out/'fn_fp_feature_effects.csv',index=False,encoding='utf-8-sig');l.to_csv(out/'fn_fp_local_comparison.csv',index=False,encoding='utf-8-sig')
 lines=['# 공통 FN/FP 공정조건표 초안','## 기준 및 한계','CN7/RG3를 따로 계산. competition 제공 표준화 눈금이며 물리 단위 raw 값이 아니다. shot당 1표본, LH/RH 중복 제거. 기존 block OOF y_pred의 repeat 과반수 판정을 유지하고 prediction CSV로 재검증했다. TP/TN도 두 모델 공통 TP/TN만 사용한다. 모델 판단 불일치 사례는 비교군에서 제외한다.','median/IQR와 reference IQR/1.349로 나눈 중앙값 차이, Cliff delta(P(target>reference)-P(target<reference))를 기록. 0 IQR이면 표준화 차이를 NA로 두고 Cliff delta로 비교. 순위는 |Cliff delta| 기준. 표본 10개 미만은 판단 어려움, 그 외 |delta|>=0.33, target 한 건씩 제외해 방향 유지, 2개 이상 날짜의 target 중앙값 방향 일치일 때만 표본 내 반복 패턴으로 표시한다. 이는 검정/독립 반복 재현/인과 판정 기준이 아닌 보수적인 기술적 분류이다. 날짜별 비교도 전체 reference 기준이라 날짜 효과를 제거하지 못한다.']
 for product in ['cn7','rg3']:
  for comp,title in [('FN_vs_TP',f'{product.upper()} FN 특징'),('FP_vs_TN',f'{product.upper()} FP 특징'),('one_side_FN_vs_both_side_TP',f'{product.upper()} one-side FN / both-side TP')]:
   s=e[(e['product']==product)&(e.comparison==comp)].copy()
   if 'cliffs_delta' in s:s=s.assign(abs_delta=s.cliffs_delta.abs()).sort_values('abs_delta',ascending=False)
   cols=['feature','n_target','n_reference','target_median','reference_median','cliffs_delta','robust_standardized_difference','evidence_category']
   lines.extend(['## '+title,'```',s.head(5).reindex(columns=cols).to_string(index=False),'```'])
 lines+=['## Local comparison','같은 제품/날짜의 라벨 정상 shot에서 직전 5/직후 5개를 선택했다. FP 자신은 제외. 부족하면 다른 날짜를 섞지 않고 실제 수와 balanced_complete=False를 기록. local 차이는 전체 비교의 보조 설명이며 미래 정상 라벨/직후 shot을 사용한 사후 audit이다. 모델 재학습/예측에는 사용하지 않았다.','```',l.drop_duplicates(['product','shot_id']).groupby(['product','error_type']).balanced_complete.agg(['count','sum']).to_string(),'```','## 보고서용 안전한 문장','CN7 공통 FN은 2 shot에 불과해 특정 변수 영역을 일반적인 실패조건으로 결론낼 수 없다. RG3 및 FP의 변수 차이는 해당 OOF 오류 집단에서 관찰된 연관이며, 날짜·설정 조합·불량 종류의 영향을 분리한 결과는 아니다. one-side FN과 both-side TP의 차이는 side 효과뿐 아니라 불량 유형 및 공정기간 차이를 포함할 수 있다.','## 과도하게 말하면 안 되는 해석','변수 X 때문에 불량이 발생했다 / X 때문에 모델이 오판했다 / one-side의 원인은 cavity-local이다 / 높은 효과 크기가 독립 검증된 원인이다. RG3 both-side TP가 없으면 side간 비교 자체가 불가능하다. 모델별 SHAP 전에는 변수 차이를 예측에 대한 기여로 동일시하지 않는다.','## 검증','기존 두 prediction의 repeat 판정 재검증, shot X 일치, 연결 1:1 확인. 기존 팀 코드 수정, 모델 학습, run_all 실행 없음.']
 (out/'fn_fp_condition_summary.md').write_text('\n\n'.join(lines),encoding='utf-8')
 for p in ['cn7','rg3']:
  for c in ['FN_vs_TP','FP_vs_TN']:
   q=e[e['product'].eq(p)&e.comparison.eq(c)].copy();q['rank']=q.cliffs_delta.abs();print(p,c);print(q.sort_values('rank',ascending=False)[['feature','n_target','n_reference','cliffs_delta','evidence_category']].head(5).to_string(index=False))
if __name__=='__main__':main()
