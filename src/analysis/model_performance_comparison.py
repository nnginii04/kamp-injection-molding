"""Compare existing aligned block OOF predictions; no training."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from src.evaluate import expected_topk
ROOT=Path(__file__).resolve().parents[2]

def metrics(g):
 y=g.y_true.to_numpy();s=g.y_score.to_numpy();n=len(g);p=int(y.sum());k=int(np.ceil(n*.1))
 return dict(n_shots=n,n_defect=p,k=k,AP=average_precision_score(y,s) if p else np.nan,ROC_AUC=roc_auc_score(y,s) if 0<p<n else np.nan,Brier=brier_score_loss(y,s),hits=expected_topk(y,s,k),Recall10=expected_topk(y,s,k)/p if p else np.nan)

def main():
 out=ROOT/'outputs/analysis';paths=sorted((ROOT/'outputs/predictions').glob('pipeline_*.csv'));frames=[];sources=[];reference=None
 ft=pd.read_csv(ROOT/'outputs/fold_id.csv').set_index(['mold','row_id'])
 for path in paths:
  d=pd.read_csv(path);d=d[d.cv_mode.eq('block')].copy();name=path.stem.removeprefix('pipeline_');assert len(d) and d.model.eq(name).all()
  keys=['product','repeat','fold','row_id','shot_id','y_true'];key=d[keys].sort_values(keys).reset_index(drop=True)
  assert not d.duplicated(['product','repeat','row_id']).any();assert np.isfinite(d.y_score).all() and d.y_score.between(0,1).all()
  if reference is None:reference=key
  else:pd.testing.assert_frame_equal(reference,key)
  for rep,g in d.groupby('repeat'):
   fixed=ft.loc[list(zip(g['product'],g.row_id)),f'fold_block_r{rep}'].to_numpy();assert np.array_equal(fixed,g.fold)
  assert d.groupby(['product','repeat','shot_id']).fold.nunique().eq(1).all()
  frames.append(d.groupby(['product','model','repeat','fold','shot_id'],as_index=False).agg(y_true=('y_true','max'),y_score=('y_score','max')))
  sources.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),block_rows=len(d)))
 d=pd.concat(frames);foldrows=[];agg=[]
 for key,g in d.groupby(['product','model','repeat','fold']):foldrows.append(dict(zip(['product','model','repeat','fold'],key),**metrics(g)))
 f=pd.DataFrame(foldrows)
 repeat=[]
 for key,g in f.groupby(['product','model','repeat']):
  repeat.append(dict(zip(['product','model','repeat'],key),AP=g.AP.mean(),ROC_AUC=g.ROC_AUC.mean(),Brier=np.average(g.Brier,weights=g.n_shots),Recall10=g.hits.sum()/g.n_defect.sum(),n_valid_AP_folds=g.AP.notna().sum(),actual_budget=g.k.sum()/g.n_shots.sum()))
 for key,g in d.groupby(['product','model','repeat']):agg.append(dict(zip(['product','model','repeat'],key),**metrics(g)))
 r=pd.DataFrame(repeat);summary=r.groupby(['product','model']).agg(AP_mean=('AP','mean'),AP_std=('AP','std'),Recall10_mean=('Recall10','mean'),Recall10_std=('Recall10','std'),ROC_AUC_mean=('ROC_AUC','mean'),ROC_AUC_std=('ROC_AUC','std'),Brier_mean=('Brier','mean'),Brier_std=('Brier','std'),n_repeats=('repeat','nunique'),actual_budget_mean=('actual_budget','mean')).reset_index();summary['ranking']='within_fold';summary['unit']='shot';summary['AP_aggregation']='macro_fold_then_repeat';summary['Recall10_aggregation']='sum_fold_hits/sum_fold_defects_then_repeat'
 summary.to_csv(out/'model_performance_summary.csv',index=False)
 f.to_csv(out/'model_performance_folds.csv',index=False);r.to_csv(out/'model_performance_repeats.csv',index=False)
 pd.DataFrame(agg).assign(ranking='aggregate_OOF_per_repeat').to_csv(out/'model_performance_aggregate_SEPARATE.csv',index=False)
 pd.DataFrame(sources).to_csv(out/'model_performance_sources.csv',index=False)
 paired=[]
 for product,g in r.groupby('product'):
  baseline=g[g.model.eq('logreg')].set_index('repeat')
  for model,h in g.groupby('model'):
   for metric in ['AP','Recall10']:
    delta=h.set_index('repeat')[metric]-baseline[metric];paired.append(dict(product=product,model=model,metric=metric,mean_delta=delta.mean(),std_delta=delta.std(),wins=int((delta>1e-12).sum()),ties=int((delta.abs()<=1e-12).sum()),losses=int((delta< -1e-12).sum())))
 pair=pd.DataFrame(paired);pair.to_csv(out/'model_performance_vs_logreg.csv',index=False)
 for metric,filename in [('AP','model_AP_comparison.png'),('Recall10','model_Recall10_comparison.png')]:
  fig,axes=plt.subplots(1,2,figsize=(14,8),sharex=True)
  for ax,product in zip(axes,['cn7','rg3']):
   g=summary[summary['product'].eq(product)].sort_values(metric+'_mean');pos=np.arange(len(g));ax.barh(pos,g[metric+'_mean'],color=['#d88b32' if m=='logreg' else '#317b91' for m in g.model],alpha=.85);ax.errorbar(g[metric+'_mean'],pos,xerr=g[metric+'_std'],fmt='none',ecolor='#333333',capsize=3);ax.set_yticks(pos,g.model);ax.set_title(product.upper());ax.set_xlim(0,1.07);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Average precision' if metric=='AP' else 'Defect-shot recall at 10% shot budget')
  fig.suptitle('Block OOF | within-fold ranking | mean +/- SD across 5 repeats',fontsize=14);fig.text(.5,.015,'AP: fold macro average. Recall: pooled fold captures. Error bars are descriptive SD, not confidence intervals.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(out/filename,dpi=180);plt.close(fig)
 focus=summary[summary.model.isin(['logreg','logreg_spc','rf','xgb','lgbm','catboost','spc_maxz'])]
 lines=['# 기존 모델 성능 비교 — 동일 block OOF 기준','## 평가 정의','모든 모델에서 product/repeat/fold/row_id/shot_id/y_true가 정확히 일치하고 fold_id.csv와 일치함을 확인했다. CN7/RG3 별도, LH/RH를 shot 최대 score/최대 label로 묶었다. 모든 14개 prediction 모델을 포함한다.','주표/그래프는 within-fold ranking만 사용한다. 각 fold 상위 ceil(shot수×10%) 검사, 경계 동점은 기존 expected_topk의 무작위 순서 기대 포착수. Recall은 fold 포착수 합/전체 불량 shot수 후 repeat 평균. AP/AUC는 평가 가능한 fold의 동일가중 평균 후 repeat 평균, Brier는 shot수가중 평균. 불량 없는 fold AP, 단일 클래스 AUC는 NA. 실제 예산은 반올림 때문에 10%보다 조금 크며 CSV에 기록.','표준편차는 5 repeat 요약값 간 sample SD이며 독립 표본 CI가 아니다. fold별 자료도 별도 제공. aggregate OOF ranking은 진단용 별도 CSV에만 저장하고 주표와 섞지 않는다. 검사 단위는 shot이므로 E1 부품 검사예산과 직접 비교하지 않는다. Brier는 모든 score에 계산하지만 SPC/결합/이상점수는 보정된 불량확률이 아니므로 확률 보정 성능의 동등한 증거로 해석하지 않는다.','## 주요 모델 수치','```',focus[['product','model','AP_mean','AP_std','Recall10_mean','Recall10_std','ROC_AUC_mean','Brier_mean']].round(4).to_string(index=False),'```','## Logistic 대비 paired repeat 안정성','```',pair[pair.model.isin(['rf','xgb','lgbm','catboost','logreg_spc'])].round(4).to_string(index=False),'```','## 결과파일 탐색 및 제외','outputs/evaluation 및 outputs/model 디렉터리는 현재 없다. E1/E3 실행 스크립트는 있으나 예정된 policy_comparison/robustness 결과파일이 없으므로 수치를 보충하거나 추정하지 않았다. outputs/experiments/mahalanobis/evaluation/experiment_summary.csv 및 outputs/tables/cv_summary.csv는 존재하나 별도 실험/기존 집계이며 이번 주표에 결합하지 않았다. synthetic 결과도 제외한다.','## Freeze 판단에 필요한 사실','모델 우승을 사전에 정하지 않았다. 평균 차이 외 paired repeat 승/패와 fold 변동을 확인해야 한다. AP와 Recall@10%는 목적이 다르고 제품별 순위도 다를 수 있다. 같은 데이터의 반복분할로 독립 성능 우위를 증명할 수 없다. E1/E3 저장 결과 복구 및 실제 검사 단위/예산 확정 전 이 표만으로 최종 freeze 타당성을 단정하지 않는다.','## 재현 및 출처','실행: python -m src.analysis.model_performance_comparison. 기존 prediction만 읽고 모델 fit/predict/run_all 실행 없음. model_performance_sources.csv에 입력 경로/SHA256 기록.']
 (out/'model_performance_summary.md').write_text('\n\n'.join(lines),encoding='utf-8')
 print(focus[['product','model','AP_mean','AP_std','Recall10_mean','Recall10_std']].round(4).to_string(index=False));print(pair[pair.model.isin(['rf','xgb','lgbm','catboost','logreg_spc'])].round(4).to_string(index=False))
if __name__=='__main__':main()
