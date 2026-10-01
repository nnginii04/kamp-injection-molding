"""Reproduce selected block OOF folds and explain individual predictions.
Preview: python -m src.analysis.shap_oof_explainer
Final: python -m src.analysis.shap_oof_explainer --model logreg --global --cases cn7_58 rg3_80 rg3_44
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_all,assign_shots
from src.features import select_base_features
from src.cv import cv_splits
from src.models import make_model
ROOT=Path(__file__).resolve().parents[2]


def attribution(model, X, train, names, kind, seed, paths):
    """Train-mean linear log odds, or Monte Carlo interventional Shapley.
    Combined-model paths start at sampled TRAIN rows; never validation backgrounds.
    Correlated feature interventions can leave the observed process manifold.
    """
    sup=model if kind=='logreg' else model.sup_
    scaler,lr=sup.steps[0][1],sup.steps[-1][1]
    background=scaler.transform(train).mean(axis=0)
    linear=(scaler.transform(X)-background)*lr.coef_[0]
    linear_base=float(lr.intercept_[0]+background@lr.coef_[0])
    assert np.allclose(linear.sum(1)+linear_base,sup.decision_function(X),atol=1e-9)
    if kind=='logreg':return linear,np.full(len(X),linear_base),'exact_train_mean_log_odds',linear
    rng=np.random.default_rng(seed);values=[];bases=[]
    for x in X.to_numpy():
        sequences=[];orders=[]
        for _ in range(paths):
            b=train.iloc[int(rng.integers(len(train)))].to_numpy().copy()
            order=rng.permutation(len(names));seq=[b.copy()]
            for j in order:b[j]=x[j];seq.append(b.copy())
            sequences.extend(seq);orders.append(order)
        pred=model.predict_proba(pd.DataFrame(sequences,columns=names))[:,1].reshape(paths,len(names)+1)
        phi=np.zeros(len(names))
        for i,order in enumerate(orders):phi[order]+=np.diff(pred[i])/paths
        values.append(phi);bases.append(pred[:,0].mean())
    values=np.asarray(values);bases=np.asarray(bases)
    assert np.allclose(values.sum(1)+bases,model.predict_proba(X)[:,1],atol=1e-9)
    return values,bases,'MC_interventional_shapley_final_score',linear


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['both','logreg','logreg_spc'],default='both');ap.add_argument('--global',dest='global_run',action='store_true');ap.add_argument('--repeat',type=int,default=0);ap.add_argument('--cases',nargs='+');ap.add_argument('--paths',type=int,default=32);ap.add_argument('--tolerance',type=float,default=1e-8);a=ap.parse_args()
    assert a.paths>0
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'));out=ROOT/'outputs/analysis'
    assert not any(cfg['features'].get(k,False) for k in ['sequence_features','unlabeled_iforest_score']), 'Unsupported enabled feature stage; do not silently change pipeline'
    data=load_all(cfg,ROOT/cfg['data_dir']);raw=data['feature_cols'];base,_=select_base_features(data,raw,cfg['features']['drop_constant'])
    candidates=pd.read_csv(out/'shap_case_candidates.csv');cases=a.cases or candidates.shot_id.tolist();assert set(cases)<=set(candidates.shot_id)
    folds=pd.read_csv(ROOT/'outputs/fold_id.csv');checks=[];contributions=[];global_values=[];fits=0
    models=['logreg','logreg_spc'] if a.model=='both' else [a.model]
    for name in models:
      saved=pd.read_csv(ROOT/f'outputs/predictions/pipeline_{name}.csv');saved=saved[saved.cv_mode.eq('block')]
      for product,lab0 in data['labeled'].items():
        lab=assign_shots(lab0,raw,product).reset_index(drop=True);X=lab[base];y=lab.y.to_numpy()
        reps=range(cfg['cv']['n_repeats']) if a.global_run else [a.repeat]
        for rep in reps:
          for fold,(tr,va) in enumerate(cv_splits(lab,'block',cfg,rep)):
            selected=lab.iloc[va].shot_id.isin(cases).to_numpy()
            if not a.global_run and not selected.any():continue
            assert set(lab.iloc[tr].shot_id).isdisjoint(lab.iloc[va].shot_id)
            ft=folds[folds.mold.eq(product)].set_index('row_id').loc[lab.iloc[va].row_id]
            assert ft[f'fold_block_r{rep}'].eq(fold).all()
            assert len(np.unique(y[tr]))==2,'Original pipeline would skip this fold; no explanation produced'
            model=make_model(name,cfg['seed']).fit(X.iloc[tr],y[tr]);fits+=1
            pred=model.predict_proba(X.iloc[va])[:,1]
            old=saved[(saved['product']==product)&(saved['repeat']==rep)&(saved.fold==fold)].set_index('row_id').loc[lab.iloc[va].row_id]
            assert np.array_equal(old.shot_id.to_numpy(),lab.iloc[va].shot_id.to_numpy())
            errors=np.abs(pred-old.y_score.to_numpy());passed=errors<=a.tolerance
            for i,pos in enumerate(va):
              checks.append(dict(model=name,product=product,repeat=rep,fold=fold,row_id=lab.row_id.iloc[pos],shot_id=lab.shot_id.iloc[pos],is_case=bool(selected[i]),saved_score=old.y_score.iloc[i],reproduced_score=pred[i],absolute_error=errors[i],tolerance=a.tolerance,passed=bool(passed[i]),train_rows=len(tr)))
            if not passed.all():
              pd.DataFrame(checks).to_csv(out/'shap_reproduction_check.csv',index=False)
              raise RuntimeError('OOF reproduction failed; explanations stopped')
            explain=va if a.global_run else va[selected]
            phi,bias,method,linear=attribution(model,X.iloc[explain],X.iloc[tr],base,name,cfg['seed']+rep*100+fold,a.paths)
            for i,pos in enumerate(explain):
              if a.global_run:
                for j,f in enumerate(base):global_values.append(dict(model=name,product=product,repeat=rep,shot_id=lab.shot_id.iloc[pos],feature=f,abs_contribution=abs(phi[i,j]),method=method))
              if lab.shot_id.iloc[pos] not in cases:continue
              positive=np.argsort(-phi[i]);negative=np.argsort(phi[i])
              for j,f in enumerate(base):
                contributions.append(dict(model=name,product=product,repeat=rep,fold=fold,row_id=lab.row_id.iloc[pos],shot_id=lab.shot_id.iloc[pos],feature=f,value=X.iloc[pos,j],contribution=phi[i,j],base_value=bias[i],explained_output=bias[i]+phi[i].sum(),method=method,logistic_component_log_odds_contribution=linear[i,j],top_positive=bool(phi[i,j]>0 and j in positive[:5]),top_negative=bool(phi[i,j]<0 and j in negative[:5])))
    pd.DataFrame(checks).to_csv(out/'shap_reproduction_check.csv',index=False)
    pd.DataFrame(contributions).to_csv(out/('shap_case_contributions.csv' if a.global_run else 'shap_case_contributions_preview.csv'),index=False)
    if global_values:
      g=pd.DataFrame(global_values).groupby(['model','product','repeat','shot_id','feature','method']).abs_contribution.mean().reset_index()
      g.groupby(['model','product','feature','method']).abs_contribution.mean().rename('mean_abs_contribution_shot_repeat_weighted').reset_index().to_csv(out/'shap_global_importance.csv',index=False)
    (out/'shap_explainer_manifest.json').write_text(json.dumps(dict(args=vars(a),features=base,models=models,fit_count=fits,units={'logreg':'log odds','logreg_spc':'final score'},notes=['Preview uses repeat 0 by default, not averaged OOF','Final --global covers every block fold/repeat and equal-weights shots','Combined Shapley is Monte Carlo approximation; increase --paths to assess stability','No inner threshold fitting: checking score, not reproducing y_pred','No final-model winner assumed']),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS: {len(checks)} row predictions, {fits} fits, max error={pd.DataFrame(checks).absolute_error.max():.3g}, contribution rows={len(contributions)}')
if __name__=='__main__':main()
