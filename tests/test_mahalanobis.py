"""Independent ablation: normal-shot fitting, fixed folds, covariance and OOF contracts."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from _common import load_cfg
from src.data import assign_shots
from src.mahalanobis import NormalShotMahalanobis, PRED_COLS, prepare_shots, fit_fold, run_oof


def fixture():
    rows=[]; folds=[]
    for product in ('cn7','rg3'):
        for shot in range(12):
            labels=([0,0],[0],[0,1],[1,1])[shot%4]
            for side,y in enumerate(labels):
                rid=(0 if product=='cn7' else 100)+2*shot+side
                rows.append(dict(product=product,row_id=rid,y=y,a=float(shot),b=float(shot**2+shot%2)))
                folds.append(dict(mold=product,row_id=rid,shot_id=f'{product}_{shot}',block_id=f'{product}_b{shot//4}',
                    **{f'fold_block_r{r}':(shot//4+r)%3 for r in range(5)}))
    raw=pd.DataFrame(rows)
    lab=pd.concat([assign_shots(raw[raw['product'].eq(p)],['a','b'],p) for p in ('cn7','rg3')],ignore_index=True)
    return lab,pd.DataFrame(folds),copy.deepcopy(load_cfg())


class TestMahalanobis(unittest.TestCase):
    def test_only_complete_normal_shots_fit_once(self):
        lab,folds,_=fixture()
        _,shots,_=prepare_shots(lab,folds,['a','b'])
        part=shots[shots['product'].eq('cn7')]
        _,model,ids=fit_fold(part,['a','b'],'fold_block_r0',0)
        self.assertEqual(set(ids),{'cn7_4','cn7_5','cn7_8','cn7_9'})
        self.assertEqual(len(ids),len(set(ids)))
        expected=part[part.shot_id.isin(ids)][['a','b']].to_numpy()
        np.testing.assert_allclose(model.scaler_.mean_,expected.mean(axis=0))
        np.testing.assert_allclose(model.scaler_.var_,expected.var(axis=0))
        self.assertEqual(int(model.scaler_.n_samples_seen_),4)

    def test_validation_and_train_defects_cannot_change_fit(self):
        lab,folds,_=fixture()
        _,shots,_=prepare_shots(lab,folds,['a','b'])
        part=shots[shots['product'].eq('cn7')].copy()
        _,first,ids=fit_fold(part,['a','b'],'fold_block_r0',0)
        altered=part.copy()
        altered.loc[altered.fold_block_r0.eq(0)|altered.y.eq(1),['a','b']]+=100000
        _,second,other=fit_fold(altered,['a','b'],'fold_block_r0',0)
        self.assertEqual(ids,other)
        np.testing.assert_array_equal(first.scaler_.mean_,second.scaler_.mean_)
        np.testing.assert_array_equal(first.covariance_.covariance_,second.covariance_.covariance_)
        np.testing.assert_array_equal(first.covariance_.location_,second.covariance_.location_)
        self.assertEqual(first.threshold_,second.threshold_)

    def test_distance_matches_quadratic_form(self):
        rng=np.random.default_rng(10)
        x=rng.normal(size=(100,3))
        model=NormalShotMahalanobis().fit(x)
        points=np.array([[1.,2.,3.],[-2.,0.,1.]])
        z=(points-model.scaler_.mean_)/model.scaler_.scale_-model.covariance_.location_
        expected=np.sqrt(np.einsum('ij,jk,ik->i',z,model.covariance_.precision_,z))
        np.testing.assert_allclose(model.score(points),expected)

    def test_correlation_changes_distance(self):
        rng=np.random.default_rng(42)
        a=rng.normal(size=1000)
        x=np.column_stack([a,a+rng.normal(scale=.03,size=1000)])
        model=NormalShotMahalanobis().fit(x)
        along,against=model.score([[2.,2.],[2.,-2.]])
        self.assertGreater(against,along*3)
        self.assertTrue(0<=model.covariance_.shrinkage_<=1)

    def test_collinear_columns_are_finite(self):
        a=np.arange(20,dtype=float)
        model=NormalShotMahalanobis().fit(np.column_stack([a,a,np.ones(20)]))
        self.assertTrue(np.isfinite(model.score([[30.,30.,1.],[30.,-30.,1.]])).all())

    def test_inconsistent_shot_or_block_fails_before_fit(self):
        lab,folds,cfg=fixture()
        for whole_shot,pattern in [(False,'Shot assigned'),(True,'Block assigned')]:
            bad=folds.copy()
            if whole_shot:bad.loc[bad.shot_id.eq('cn7_0'),'fold_block_r0']=1
            else:bad.loc[0,'fold_block_r0']=1
            with patch.object(NormalShotMahalanobis,'fit') as fit:
                with self.assertRaisesRegex(ValueError,pattern):run_oof(lab,bad,['a','b'],cfg,log=lambda _:None)
                fit.assert_not_called()

    def test_missing_and_mismatched_saved_folds_rejected(self):
        lab,folds,_=fixture()
        for bad in [folds.iloc[1:],folds.drop(columns='fold_block_r4'),pd.concat([folds,folds.iloc[:1]]),folds.assign(shot_id='wrong')]:
            with self.assertRaises(ValueError):prepare_shots(lab,bad,['a','b'])

    def test_no_label_derived_features(self):
        lab,folds,_=fixture()
        for feature in ('y_shot','n_pos','n_rows','conflict','defect_type'):
            with self.assertRaises(ValueError):prepare_shots(lab,folds,['a',feature])

    def test_fixed_folds_oof_mapping_and_no_new_split(self):
        lab,folds,cfg=fixture()
        original=lab.copy(deep=True); original_folds=folds.copy(deep=True)
        with patch('sklearn.model_selection.StratifiedGroupKFold.split',side_effect=AssertionError('No new split')):
            pred,shots,diag,parameters=run_oof(lab,folds,['a','b'],cfg,log=lambda _:None)
        self.assertEqual(list(pred.columns),PRED_COLS)
        self.assertEqual(len(pred),len(lab)*5)
        self.assertEqual(len(diag),30)
        self.assertEqual(len(parameters),30)
        self.assertFalse(pred.duplicated(['product','repeat','row_id']).any())
        for repeat,part in pred.groupby('repeat'):
            truth=part.merge(lab,on=['product','shot_id','row_id'],validate='one_to_one')
            self.assertTrue(truth.y_true.eq(truth.y).all())
            f=part.merge(folds,on=['row_id','shot_id'],validate='one_to_one')
            self.assertTrue(f.fold.eq(f[f'fold_block_r{repeat}']).all())
            self.assertTrue(part.groupby('shot_id').y_score.nunique().eq(1).all())
        pd.testing.assert_frame_equal(lab,original)
        pd.testing.assert_frame_equal(folds,original_folds)

    def test_insufficient_or_invalid_normal_data_rejected(self):
        for x in ([[1.,2.]],[[np.nan,1.],[2.,3.]],np.ones((10,3))):
            with self.assertRaises(ValueError):NormalShotMahalanobis().fit(x)


if __name__=='__main__':unittest.main()
