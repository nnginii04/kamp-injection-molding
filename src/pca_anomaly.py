"""PCA reconstruction anomaly, trained only on complete normal training shots."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .features import check_feature_list
from .models import ecdf_score

PRED_COLS=['mold','product','model','cv_mode','repeat','fold','row_id','shot_id','y_true','y_score','y_pred']


class NormalShotPCA:
    def fit(self, normal):
        x=np.asarray(normal,dtype=float)
        if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
            raise ValueError('At least two finite training normal shots required')
        if not np.any(x.var(axis=0)>0):raise ValueError('All training features are constant')
        self.scaler_=StandardScaler().fit(x)
        z=self.scaler_.transform(x)
        # Fit the entire decomposition to TRAIN only, then take the smallest
        # prefix whose cumulative explained variance is >= .95 (including equality).
        self.pca_=PCA(svd_solver='full').fit(z)
        cumulative=np.cumsum(self.pca_.explained_variance_ratio_)
        self.n_components_=int(np.searchsorted(cumulative,.95,side='left')+1)
        self.components_=self.pca_.components_[:self.n_components_].copy()
        self.explained_variance_=float(cumulative[self.n_components_-1])
        self.previous_explained_variance_=float(cumulative[self.n_components_-2]) if self.n_components_>1 else 0.
        train=self.raw_score(x)
        self.ref_=np.sort(train);self.ref_sd_=float(np.std(train)) or 1.
        self.threshold_=float(np.quantile(train,.95,method='higher'))
        return self

    def reconstruct_standardized(self, features):
        x=np.asarray(features,dtype=float)
        if not np.isfinite(x).all():raise ValueError('Non-finite validation features')
        z=self.scaler_.transform(x)
        centered=z-self.pca_.mean_
        restored=(centered@self.components_.T)@self.components_+self.pca_.mean_
        return z,restored

    def raw_score(self, features):
        z,restored=self.reconstruct_standardized(features)
        return np.mean((z-restored)**2,axis=1)

    def score(self, features):
        return ecdf_score(self.ref_,self.raw_score(features),self.ref_sd_)


def prepare_shots(lab, folds, features, repeats=5, n_splits=3):
    check_feature_list(features)
    if set(features)&{'y_shot','n_pos','n_rows','conflict','defect_type'}:
        raise ValueError('Label-derived features forbidden')
    cols=[f'fold_block_r{r}' for r in range(repeats)]
    needed=['mold','row_id','shot_id','block_id',*cols]
    if not set(needed)<=set(folds):raise ValueError('Saved folds required; no split generation')
    f=folds[needed].copy()
    if f.isna().any().any() or lab[['product','shot_id','row_id','y']].isna().any().any():raise ValueError('Missing identity/label/fold')
    if lab.duplicated(['product','row_id']).any() or f.duplicated(['mold','row_id']).any():raise ValueError('Duplicate row ID')
    if not set(lab.y.unique())<={0,1}:raise ValueError('Binary labels required')
    for col in cols:
        if f.groupby('shot_id')[col].nunique().gt(1).any():raise ValueError('Shot in multiple folds')
        if f.groupby('block_id')[col].nunique().gt(1).any():raise ValueError('Block in multiple folds')
        for _,part in f.groupby('mold'):
            if set(part[col])!=set(range(n_splits)):raise ValueError('Invalid fold numbers')
    rows=lab.merge(f.rename(columns={'mold':'product','shot_id':'saved_shot_id'}),on=['product','row_id'],how='outer',validate='one_to_one',indicator=True)
    if not rows._merge.eq('both').all() or not rows.shot_id.eq(rows.saved_shot_id).all():raise ValueError('Saved identity mismatch')
    if rows.groupby('shot_id').product.nunique().gt(1).any():raise ValueError('Shot shared by products')
    if rows.groupby('shot_id')[features].nunique(dropna=False).gt(1).any().any():raise ValueError('Features differ within shot')
    if not np.isfinite(rows[features].to_numpy(float)).all():raise ValueError('Non-finite features')
    rows=rows.drop(columns=['_merge','saved_shot_id'])
    g=rows.groupby('shot_id').y.agg(['max','sum','size'])
    shots=rows.drop_duplicates('shot_id')[['product','shot_id',*features,'block_id',*cols]].copy()
    shots['y']=shots.shot_id.map(g['max']);n_pos=shots.shot_id.map(g['sum']);size=shots.shot_id.map(g['size'])
    shots['defect_type']=np.select([shots.y.eq(0),n_pos.eq(size)&size.ge(2),n_pos.gt(0)&n_pos.lt(size)],['normal','both-side','one-side'],default='single-row')
    for _,part in shots.groupby('product'):
        for col in cols:
            for fold in range(n_splits):
                if len(part[part[col].ne(fold)&part.y.eq(0)])<2:raise ValueError('Insufficient training normals')
    return rows.reset_index(drop=True),shots.reset_index(drop=True),cols


def run_pca(lab,folds,features,cfg,expected_ids,log=print):
    rows,shots,cols=prepare_shots(lab,folds,features,cfg['cv']['n_repeats'],cfg['cv']['n_splits'])
    # Validate every normal fitting population against the prior fair control
    # before any estimator is fit.
    actual={f'{p}_r{r}_f{k}':part.loc[part[col].ne(k)&part.y.eq(0),'shot_id'].tolist()
            for p,part in shots.groupby('product') for r,col in enumerate(cols) for k in range(cfg['cv']['n_splits'])}
    if actual!=expected_ids:raise ValueError('Normal training IDs differ from saved shot control')
    outputs=[];audit=[];parameters={}
    for product,part in shots.groupby('product'):
        for repeat,col in enumerate(cols):
            for fold in range(cfg['cv']['n_splits']):
                ident=f'{product}_r{repeat}_f{fold}'
                normal=part[part.shot_id.isin(actual[ident])]
                valid=part[part[col].eq(fold)]
                if set(normal.block_id)&set(valid.block_id):raise ValueError('Train/validation block overlap')
                model=NormalShotPCA().fit(normal[features])
                raw=model.raw_score(valid[features]);ecdf=ecdf_score(model.ref_,raw,model.ref_sd_)
                for scale,score in [('ecdf',ecdf),('raw',raw)]:
                    dest=rows[rows['product'].eq(product)&rows[col].eq(fold)].copy().rename(columns={'y':'y_true'})
                    dest['y_score']=dest.shot_id.map(pd.Series(score,index=valid.shot_id))
                    dest['y_pred']=dest.shot_id.map(pd.Series((raw>=model.threshold_).astype(int),index=valid.shot_id))
                    if not np.isfinite(dest.y_score).all():raise ValueError('Incomplete OOF')
                    dest['mold']=product;dest['model']=f'pca_shot_{scale}';dest['cv_mode']='block';dest['repeat']=repeat;dest['fold']=fold
                    outputs.append(dest[PRED_COLS])
                audit.append(dict(product=product,repeat=repeat,fold=fold,train_normal_shots=len(normal),valid_shots=len(valid),
                    n_components=model.n_components_,explained_variance=model.explained_variance_,
                    previous_explained_variance=model.previous_explained_variance_,train_score_min=float(model.ref_.min()),
                    train_score_max=float(model.ref_.max()),train_threshold95=model.threshold_))
                parameters[ident]=dict(normal_shot_ids=actual[ident],scaler_mean=model.scaler_.mean_.tolist(),
                    scaler_scale=model.scaler_.scale_.tolist(),pca_mean=model.pca_.mean_.tolist(),
                    components=model.components_.tolist(),explained_variance_ratio=model.pca_.explained_variance_ratio_.tolist(),
                    train_score_sorted=model.ref_.tolist(),train_score_sd=model.ref_sd_,train_threshold95=model.threshold_)
            log(f'{product}: PCA repeat {repeat} completed')
    return pd.concat(outputs,ignore_index=True),shots,pd.DataFrame(audit),parameters
