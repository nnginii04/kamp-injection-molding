import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.pca_anomaly import NormalShotPCA, prepare_shots, run_pca, PRED_COLS
from src.models import ecdf_score

def fixture():
    lab=[];folds=[]
    for i in range(18):
        for j in range(2):
            lab.append(dict(product='cn7',row_id=2*i+j,shot_id=f'cn7_{i}',y=int(i==0 and j==0),a=float(i),b=float(i*i)))
            folds.append(dict(mold='cn7',row_id=2*i+j,shot_id=f'cn7_{i}',block_id=f'cn7_b{i//6}',fold_block_r0=i//6))
    return pd.DataFrame(lab),pd.DataFrame(folds)

class TestPCAAnomaly(unittest.TestCase):
    def test_minimum_components_and_scaler(self):
        x=np.random.default_rng(4).normal(size=(80,23))*np.arange(1,24)+5
        m=NormalShotPCA().fit(x)
        np.testing.assert_allclose(m.scaler_.mean_,x.mean(0))
        np.testing.assert_allclose(m.scaler_.scale_,x.std(0))
        self.assertGreaterEqual(m.explained_variance_,.95)
        self.assertLess(m.previous_explained_variance_,.95)

    def test_reconstruction_mse_and_ecdf(self):
        x=np.random.default_rng(1).normal(size=(100,8));x[:,7]=x[:,0]*2
        m=NormalShotPCA().fit(x);v=x[:4].copy();v[:,7]+=4
        z=m.scaler_.transform(v);c=z-m.pca_.mean_
        manual=np.mean((z-(c@m.components_.T)@m.components_-m.pca_.mean_)**2,axis=1)
        np.testing.assert_allclose(m.raw_score(v),manual)
        np.testing.assert_allclose(m.score(v),ecdf_score(np.sort(m.raw_score(x)),manual,m.ref_sd_))

    def test_validation_cannot_change_fit(self):
        x=np.random.default_rng(2).normal(size=(50,5));m=NormalShotPCA().fit(x)
        before=[v.copy() for v in [m.scaler_.mean_,m.scaler_.scale_,m.components_,m.ref_]]
        m.score(np.full((10,5),1e8))
        for a,b in zip(before,[m.scaler_.mean_,m.scaler_.scale_,m.components_,m.ref_]):np.testing.assert_array_equal(a,b)

    def test_out_of_subspace_anomaly(self):
        t=np.linspace(-2,2,80);x=np.column_stack([t,t])
        m=NormalShotPCA().fit(x)
        self.assertEqual(m.n_components_,1)
        self.assertGreater(m.raw_score([[2,-2]])[0],m.raw_score([[2,2]])[0]+1)

    def test_oof_and_normal_only(self):
        lab,folds=fixture();rows,shots,cols=prepare_shots(lab,folds,['a','b'],1,3)
        ids={f'cn7_r0_f{k}':shots.loc[shots.fold_block_r0.ne(k)&shots.y.eq(0),'shot_id'].tolist() for k in range(3)}
        with patch('sklearn.model_selection.StratifiedGroupKFold.split',side_effect=AssertionError('No new splits')):
            pred,_,audit,params=run_pca(lab,folds,['a','b'],{'cv':{'n_repeats':1,'n_splits':3}},ids,log=lambda _:None)
        self.assertEqual(list(pred.columns),PRED_COLS)
        self.assertEqual(len(pred),len(lab)*2)
        for _,g in pred.groupby('model'):
            np.testing.assert_array_equal(g.sort_values('row_id').y_true,lab.y)
            self.assertTrue(g.groupby('shot_id').y_score.nunique().eq(1).all())
        for key,p in params.items():
            self.assertNotIn('cn7_0',p['normal_shot_ids'])
            normal=shots[shots.shot_id.isin(p['normal_shot_ids'])]
            np.testing.assert_allclose(p['scaler_mean'],normal[['a','b']].mean())

    def test_fit_ids_checked_before_training(self):
        lab,folds=fixture()
        with patch.object(NormalShotPCA,'fit',side_effect=AssertionError('Must not fit')):
            with self.assertRaisesRegex(ValueError,'IDs differ'):
                run_pca(lab,folds,['a','b'],{'cv':{'n_repeats':1,'n_splits':3}},{})

    def test_shot_split_rejected(self):
        lab,folds=fixture();folds.loc[0,'fold_block_r0']=1
        with self.assertRaisesRegex(ValueError,'Shot in multiple'):prepare_shots(lab,folds,['a','b'],1,3)

    def test_missing_saved_row_rejected(self):
        lab,folds=fixture()
        with self.assertRaisesRegex(ValueError,'identity mismatch'):prepare_shots(lab,folds.iloc[1:],['a','b'],1,3)

    def test_label_feature_rejected(self):
        lab,folds=fixture()
        with self.assertRaises(ValueError):prepare_shots(lab,folds,['a','y_shot'],1,3)

    def test_degenerate_train_rejected(self):
        with self.assertRaises(ValueError):NormalShotPCA().fit(np.ones((5,3)))

if __name__=='__main__':unittest.main()
