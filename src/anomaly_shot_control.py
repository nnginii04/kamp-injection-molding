"""Matched normal-shot control; identical fit IDs and train-only ECDF for all models."""
import hashlib
import json
import numpy as np
import pandas as pd
from .mahalanobis import NormalShotMahalanobis, prepare_shots, PRED_COLS
from .models import IForestNormal, SPCMaxZ, ecdf_score

MODELS=('iforest','spc','mahalanobis')


def fit_models(normal, features, seed):
    x=normal[features]
    zero=np.zeros(len(x),dtype=int)
    return {'iforest':IForestNormal(random_state=seed).fit(x,zero),
            'spc':SPCMaxZ().fit(x,zero),
            'mahalanobis':NormalShotMahalanobis().fit(x)}


def raw_score(name, model, x):
    return model.score(x) if name=='mahalanobis' else model.raw_score(x)


def run_control(lab, folds, features, cfg, log=print):
    rows,shots,cols=prepare_shots(lab,folds,features,cfg['cv']['n_repeats'],cfg['cv']['n_splits'])
    output=[]; audit=[]; fit_ids={}
    for product,part in shots.groupby('product'):
        for rep,col in enumerate(cols):
            for fold in range(cfg['cv']['n_splits']):
                normal=part[part[col].ne(fold)&part.y.eq(0)]
                valid=part[part[col].eq(fold)]
                ids=normal.shot_id.tolist()
                ident=f'{product}_r{rep}_f{fold}'
                fit_ids[ident]=ids
                digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                if set(normal.shot_id)&set(valid.shot_id) or set(normal.block_id)&set(valid.block_id):
                    raise ValueError('Training/validation overlap')
                models=fit_models(normal,features,cfg['seed'])
                for name,model in models.items():
                    train=raw_score(name,model,normal[features])
                    raw=raw_score(name,model,valid[features])
                    ref=np.sort(train); sd=float(np.std(train)) or 1.
                    calibrated=ecdf_score(ref,raw,sd)
                    threshold=float(np.quantile(train,.95,method='higher'))
                    for scale,score in [('ecdf',calibrated),('raw',raw)]:
                        dest=rows[rows['product'].eq(product)&rows[col].eq(fold)].copy()
                        dest['y_score']=dest.shot_id.map(pd.Series(score,index=valid.shot_id))
                        if not np.isfinite(dest.y_score).all():raise ValueError('Invalid OOF mapping')
                        dest=dest.rename(columns={'y':'y_true'})
                        dest['mold']=product; dest['model']=f'{name}_shot_{scale}'
                        dest['cv_mode']='block';dest['repeat']=rep;dest['fold']=fold
                        flags=pd.Series((raw>=threshold).astype(int),index=valid.shot_id)
                        dest['y_pred']=dest.shot_id.map(flags)
                        output.append(dest[PRED_COLS])
                    audit.append(dict(product=product,repeat=rep,fold=fold,model=name,
                        normal_shots=len(ids),normal_shot_ids_sha256=digest,valid_shots=len(valid),
                        train_score_min=float(train.min()),train_score_max=float(train.max()),train_score_sd=sd,
                        shrinkage=float(model.covariance_.shrinkage_) if name=='mahalanobis' else np.nan))
            log(f'{product}: repeat {rep} complete (same normal shots for all three)')
    return pd.concat(output,ignore_index=True),shots,pd.DataFrame(audit),fit_ids
