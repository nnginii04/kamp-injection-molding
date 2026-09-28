import unittest
from unittest.mock import patch
import numpy as np
from test_mahalanobis import fixture
from src.anomaly_shot_control import run_control,fit_models,raw_score
from src.mahalanobis import prepare_shots


class TestAnomalyShotControl(unittest.TestCase):
    def test_identical_normal_samples_and_no_split(self):
        lab,folds,cfg=fixture();cfg['cv']['n_repeats']=1
        seen=[]
        def spy(normal,features,seed):
            self.assertTrue(normal.y.eq(0).all())
            self.assertTrue(normal.shot_id.is_unique)
            seen.append(set(normal.shot_id))
            return fit_models(normal,features,seed)
        with patch('src.anomaly_shot_control.fit_models',side_effect=spy),patch('sklearn.model_selection.StratifiedGroupKFold.split',side_effect=AssertionError('No splits')):
            pred,shots,audit,ids=run_control(lab,folds,['a','b'],cfg,log=lambda _:None)
        self.assertEqual(len(seen),6)
        self.assertEqual(len(audit),18)
        self.assertTrue(audit.groupby(['product','repeat','fold']).normal_shot_ids_sha256.nunique().eq(1).all())
        self.assertEqual(len(pred),len(lab)*6)
        self.assertFalse(pred.duplicated(['model','product','repeat','row_id']).any())
        for _,part in pred.groupby('model'):
            merged=part.merge(lab,on=['product','row_id','shot_id'],validate='one_to_one')
            self.assertTrue(merged.y_true.eq(merged.y).all())
            self.assertTrue(part.groupby('shot_id').y_score.nunique().eq(1).all())

    def test_statistics_use_identical_normal_features(self):
        lab,folds,_=fixture();_,shots,_=prepare_shots(lab,folds,['a','b'])
        normal=shots[shots['product'].eq('cn7')&shots.fold_block_r0.ne(0)&shots.y.eq(0)]
        models=fit_models(normal,['a','b'],42)
        np.testing.assert_allclose(models['spc'].mu_,models['mahalanobis'].scaler_.mean_)
        self.assertEqual(models['iforest'].if_.max_samples_,len(normal))
        for name,model in models.items():
            self.assertTrue(np.isfinite(raw_score(name,model,normal[['a','b']])).all())

    def test_invalid_folds_rejected_before_models(self):
        lab,folds,cfg=fixture();folds.loc[0,'fold_block_r0']=1
        with patch('src.anomaly_shot_control.fit_models') as fit:
            with self.assertRaises(ValueError):run_control(lab,folds,['a','b'],cfg)
            fit.assert_not_called()
