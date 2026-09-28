"""Independent normal-shot Mahalanobis ablation; saved block folds only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from .features import check_feature_list

PRED_COLS = ['mold','product','model','cv_mode','repeat','fold','row_id','shot_id','y_true','y_score','y_pred']
LABEL_FEATURES = {'y_shot','n_pos','n_rows','conflict','defect_type'}


class NormalShotMahalanobis:
    """StandardScaler + LedoitWolf, fitted on training normal shots only.

    score = sqrt((z-location) @ precision @ (z-location)).
    No ECDF, validation calibration, or tuning is performed.
    """
    def fit(self, normal_features):
        x = np.asarray(normal_features, dtype=float)
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError('At least two finite normal-shot feature rows are required')
        if not np.any(np.var(x, axis=0) > 0):
            raise ValueError('All normal-shot features are constant')
        self.scaler_ = StandardScaler().fit(x)
        z = self.scaler_.transform(x)
        self.covariance_ = LedoitWolf().fit(z)
        # Only a convenience y_pred for CSV interoperability; primary metrics
        # are rankings. This fixed training-normal quantile is not tuned.
        self.threshold_ = float(np.quantile(self.score(normal_features), .95, method='higher'))
        return self

    def score(self, features):
        x = np.asarray(features, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('Non-finite scoring features')
        squared = self.covariance_.mahalanobis(self.scaler_.transform(x))
        return np.sqrt(np.maximum(squared, 0.))


def prepare_shots(lab, saved_folds, features, repeats=5, n_splits=3):
    """Reject inconsistent assignments before fitting anything; retain original IDs."""
    check_feature_list(features)
    if set(features) & LABEL_FEATURES:
        raise ValueError('Label-derived information cannot be a feature')
    columns = [f'fold_block_r{r}' for r in range(repeats)]
    required = ['mold','row_id','shot_id','block_id',*columns]
    if not set(required) <= set(saved_folds):
        raise ValueError('Saved block fold columns are missing; no splits will be generated')
    folds = saved_folds[required].copy()
    if folds.isna().any().any() or lab[['product','row_id','shot_id','y']].isna().any().any():
        raise ValueError('Missing identity, label, or saved fold')
    if not set(lab.y.unique()) <= {0,1}:
        raise ValueError('Labels must be binary')
    if lab.duplicated(['product','row_id']).any() or folds.duplicated(['mold','row_id']).any():
        raise ValueError('Duplicate product/row_id')
    for col in columns:
        if folds.groupby('shot_id')[col].nunique().gt(1).any():
            raise ValueError(f'Shot assigned to multiple folds: {col}')
        if folds.groupby('block_id')[col].nunique().gt(1).any():
            raise ValueError(f'Block assigned to multiple folds: {col}')
        for _, part in folds.groupby('mold'):
            if set(part[col]) != set(range(n_splits)):
                raise ValueError(f'Invalid fold labels: {col}')
    rows = lab.merge(folds.rename(columns={'mold':'product','shot_id':'saved_shot_id'}),
                     on=['product','row_id'], how='outer', validate='one_to_one', indicator=True)
    if not rows._merge.eq('both').all() or not rows.shot_id.eq(rows.saved_shot_id).all():
        raise ValueError('Labeled rows/shot IDs do not match saved baseline folds')
    if rows.groupby('shot_id').product.nunique().gt(1).any():
        raise ValueError('Shot ID shared across products')
    if rows.groupby('shot_id')[features].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Features differ within shot')
    if not np.isfinite(rows[features].to_numpy(float)).all():
        raise ValueError('Non-finite features')
    rows = rows.drop(columns=['_merge','saved_shot_id'])
    g = rows.groupby('shot_id').y.agg(['max','size','sum'])
    shots = rows.drop_duplicates('shot_id')[['product','shot_id',*features,'block_id',*columns]].copy()
    shots['y'] = shots.shot_id.map(g['max'])
    shots['n_rows'] = shots.shot_id.map(g['size'])
    shots['n_pos'] = shots.shot_id.map(g['sum'])
    shots['defect_type'] = np.select([shots.y.eq(0), shots.n_pos.eq(shots.n_rows)&shots.n_rows.ge(2),
                                      shots.n_pos.gt(0)&shots.n_pos.lt(shots.n_rows)],
                                     ['normal','both-side','one-side'], default='single-row')
    for product, part in shots.groupby('product'):
        for col in columns:
            for fold in range(n_splits):
                if len(part[part[col].ne(fold)&part.y.eq(0)]) < 2:
                    raise ValueError(f'Insufficient train normal shots: {product}/{col}/{fold}')
    return rows.reset_index(drop=True), shots.reset_index(drop=True), columns


def fit_fold(shots, features, fold_column, fold):
    train = shots[shots[fold_column].ne(fold)]
    valid = shots[shots[fold_column].eq(fold)]
    normal = train[train.y.eq(0)]
    if set(train.shot_id) & set(valid.shot_id) or set(train.block_id) & set(valid.block_id):
        raise ValueError('Train/validation group overlap')
    model = NormalShotMahalanobis().fit(normal[features])
    score = pd.Series(model.score(valid[features]), index=valid.shot_id, name='score')
    return score, model, normal.shot_id.tolist()


def run_oof(lab, saved_folds, features, cfg, log=print):
    rows, shots, columns = prepare_shots(lab, saved_folds, features,
                                        cfg['cv']['n_repeats'], cfg['cv']['n_splits'])
    outputs, diagnostics, parameters = [], [], {}
    for product, part in shots.groupby('product'):
        for repeat, col in enumerate(columns):
            for fold in range(cfg['cv']['n_splits']):
                score, model, normal_ids = fit_fold(part, features, col, fold)
                valid = rows[rows['product'].eq(product)&rows[col].eq(fold)].copy()
                valid['y_score'] = valid.shot_id.map(score)
                if valid.y_score.isna().any():
                    raise ValueError('Missing OOF score mapping')
                valid = valid.rename(columns={'y':'y_true'})
                valid['mold'], valid['model'], valid['cv_mode'] = product, 'mahalanobis_lw', 'block'
                valid['repeat'], valid['fold'] = repeat, fold
                valid['y_pred'] = valid.y_score.ge(model.threshold_).astype(int)
                outputs.append(valid[PRED_COLS])
                cov = model.covariance_
                ident = f'{product}_r{repeat}_f{fold}'
                diagnostics.append(dict(product=product, repeat=repeat, fold=fold,
                    train_shots=int(part[col].ne(fold).sum()), train_normal_shots=len(normal_ids),
                    excluded_train_defect_shots=int((part[col].ne(fold)&part.y.eq(1)).sum()),
                    valid_shots=len(score), shrinkage=float(cov.shrinkage_), threshold_train_normal95=model.threshold_,
                    covariance_min_eigenvalue=float(np.linalg.eigvalsh(cov.covariance_).min())))
                parameters[ident] = dict(normal_shot_ids=normal_ids, scaler_mean=model.scaler_.mean_.tolist(),
                    scaler_scale=model.scaler_.scale_.tolist(), location=cov.location_.tolist(),
                    covariance=cov.covariance_.tolist(), precision=cov.precision_.tolist())
            log(f'{product}: saved block repeat {repeat} complete')
    prediction = pd.concat(outputs,ignore_index=True).sort_values(['product','repeat','row_id']).reset_index(drop=True)
    return prediction, shots, pd.DataFrame(diagnostics), parameters
