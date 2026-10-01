"""Inventory of saved block OOF errors; no model training or inference."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import _public

def main():
 out=ROOT/'outputs/analysis'
 pub=_public.read_csv(Path.home()/'Downloads/04. Dataset_Molding.zip','moldset_labeled.csv')
 frames=[]
 for model in ['logreg','logreg_spc']:
  d=pd.read_csv(ROOT/f'outputs/predictions/pipeline_{model}.csv');d=d[d.cv_mode.eq('block')].copy()
  assert not d.duplicated(['product','repeat','row_id']).any()
  frames.append(d.assign(model=model))
 d=pd.concat(frames,ignore_index=True)
 keys=['product','repeat','fold','row_id','shot_id','y_true']
 assert set(map(tuple,frames[0][keys].to_numpy()))==set(map(tuple,frames[1][keys].to_numpy()))
 meta=d[['product','row_id','shot_id','y_true']].drop_duplicates().merge(pub[['Unnamed: 0','TimeStamp','PART_NAME','Reason','PassOrFail']],left_on='row_id',right_on='Unnamed: 0',validate='many_to_one')
 assert len(meta)==d[['product','row_id']].drop_duplicates().shape[0]
 assert (meta.y_true==meta.PassOrFail.astype(int)).all()
 rows=[]
 for (product,shot),g in meta.groupby(['product','shot_id']):
  assert g.TimeStamp.nunique()==1, 'feature-defined shot maps multiple timestamps'
  sides=g.PART_NAME.str.extract(r'\b(LH|RH)\b')[0]
  bad=g[g.y_true.eq(1)]
  pattern='normal' if bad.empty else 'both-side' if len(bad)==2 and len(g)==2 else 'one-side' if len(bad)==1 and len(g)==2 else 'unpaired'
  reasons=' | '.join(sorted(bad.Reason.dropna().unique()))
  rows.append(dict(product=product,shot_id=shot,timestamp=g.TimeStamp.iloc[0],side_pattern=pattern,side_labels='; '.join(f'{s}:{y}' for s,y in zip(sides,g.y_true)),defect_reason=reasons,reason_group='gas' if '가스' in reasons or 'gas' in reasons.lower() else 'short-shot' if '미성형' in reasons else 'other' if reasons else 'normal',y_true=int(g.y_true.max()),row_ids=';'.join(g.row_id.astype(str)),n_parts=len(g)))
 inv=pd.DataFrame(rows)
 for model in ['logreg','logreg_spc']:
  m=d[d.model.eq(model)].groupby(['product','shot_id','repeat']).agg(score=('y_score','max'),pred=('y_pred','max'),fold_count=('fold','nunique')).reset_index()
  assert m.fold_count.eq(1).all()
  s=m.groupby(['product','shot_id']).agg(**{model+'_score':('score','mean'),model+'_score_min':('score','min'),model+'_score_max':('score','max'),model+'_score_median':('score','median'),model+'_positive_repeat_fraction':('pred','mean'),model+'_n_repeats':('repeat','nunique')}).reset_index()
  inv=inv.merge(s,on=['product','shot_id'],validate='one_to_one')
  inv[model+'_pred']=inv[model+'_positive_repeat_fraction'].gt(.5).astype(int)
  inv[model+'_error_type']=np.where(inv.y_true.eq(1),np.where(inv[model+'_pred'].eq(1),'TP','FN'),np.where(inv[model+'_pred'].eq(1),'FP','TN'))
  inv[model+'_score_percentile']=inv.groupby('product')[model+'_score'].rank(pct=True)*100
 inv['error_type']=inv.logreg_error_type+'/'+inv.logreg_spc_error_type
 inv['common_error']=inv.error_type.isin(['FN/FN','FP/FP'])
 inv['models_disagree']=inv.logreg_pred.ne(inv.logreg_spc_pred)
 inv['mean_percentile']=(inv.logreg_score_percentile+inv.logreg_spc_score_percentile)/2
 inv.to_csv(out/'error_case_inventory.csv',index=False,encoding='utf-8-sig')
 chosen=[]
 def choose(pool,ascending,why):
  pool=pool[~pool.shot_id.isin([r['shot_id'] for r in chosen])].sort_values('mean_percentile',ascending=ascending)
  if len(pool):
   r=pool.iloc[0].to_dict();r['selection_reason']=why;r['priority']=len(chosen)+1;chosen.append(r)
 choose(inv[(inv['product']=='cn7')&(inv.error_type=='TP/TP')],False,'CN7 common TP; OOF high-score positive, not the external Oct27 episode')
 choose(inv[(inv.side_pattern=='one-side')&(inv.error_type=='FN/FN')],True,'Lowest joint score percentile among one-side common FN')
 choose(inv[inv.error_type=='FP/FP'],False,'Highest joint score percentile among common FP')
 # Product coverage and disagreement, without assuming a winning model.
 covered={r['product'] for r in chosen}
 choose(inv[(inv['product'].eq('cn7'))&(inv.side_pattern=='one-side')&(inv.error_type=='FN/FN')],True,'CN7 one-side common FN for cross-product comparison')
 choose(inv[inv.models_disagree],False,'Saved decisions disagree; compare explanations of both models')
 cand=pd.DataFrame(chosen);cand.to_csv(out/'shap_case_candidates.csv',index=False,encoding='utf-8-sig')
 counts=inv.groupby(['product','error_type','side_pattern','reason_group']).size().to_string()
 lines=['# SHAP 대표 오류 사례 후보','## 기준','기존 pipeline_logreg.csv와 pipeline_logreg_spc.csv의 block OOF만 사용. repeat별 shot 점수는 LH/RH y_score 최대, shot 정답/예측은 각각 y_true/y_pred 최대. 최종 점수는 repeat 평균(최소/최대/중앙값도 기록). TP/FN/FP는 저장 y_pred의 repeat 과반수로 정의하며 0.5 score 임계값을 새로 적용하지 않는다. 동률은 양성으로 처리하지 않는다. repeat 양성 비율로 불안정성 확인. 이는 후보 선정 요약이며 새 평가 결과가 아니다.','결합모델 score는 보정된 불량확률로 해석하지 않는다. score 크기가 다른 두 모델의 후보 순위는 제품 내 percentile 평균을 사용했다.','원본 moldset_labeled.csv의 ID로 TimeStamp/PART_NAME/Reason 연결. 모든 row 라벨 일치 및 shot의 단일 timestamp/동일 fold를 검증. side/Reason은 사후 해석에만 사용.','## 범위 제한','기존 process-wide 사례인 2020-10-27 event0과 2020-11-03 shot1173은 대회 이후 미래기간이다. 이 block OOF 파일에는 포함되지 않으므로 TP 후보와 동일 사건이라고 연결하지 않는다. CN7 common TP는 대체 후보이며 공정 전체 이상 여부는 SHAP/공정값 확인 전 확정하지 않는다.','## 사례 분포','```',counts,'```','## 추천 후보','```',cand[['priority','product','shot_id','timestamp','side_pattern','defect_reason','logreg_score','logreg_spc_score','error_type','selection_reason']].to_string(index=False),'```','## SHAP 실행 전 확인','최종 모델과 설명 대상 fitted fold/repeat를 확정해야 한다. repeat 평균 점수 자체를 하나의 fitted model SHAP으로 설명하지 않는다. 공통 FN/FP는 반복별 판정이 동일하다는 뜻이 아니므로 positive_repeat_fraction을 함께 확인한다. 정상 FP의 원인과 one-side의 cavity-local 원인은 현재 분류만으로 단정하지 않는다.','## 검증','두 모델 row/repeat/fold/shot/label 일치, 중복 OOF 없음, 원본 라벨 일치, 모든 shot의 단일 timestamp 및 fold 확인 통과. 모델 fit/predict 및 run_all 실행 없음.']
 (out/'error_case_selection.md').write_text('\n\n'.join(lines),encoding='utf-8')
 print(counts);print(cand[['priority','product','shot_id','timestamp','side_pattern','defect_reason','logreg_score','logreg_spc_score','error_type']].to_string(index=False))
 print('REPEATS', inv[['logreg_n_repeats','logreg_spc_n_repeats']].drop_duplicates().to_string(index=False))
if __name__=='__main__':main()


