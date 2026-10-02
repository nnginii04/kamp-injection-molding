import unittest
import numpy as np
import pandas as pd
from src.analysis.ambiguity import FEATURES,score_fold,ambiguity

class AmbiguityTests(unittest.TestCase):
    def frames(self):
        train=pd.DataFrame(np.arange(6.)[:,None]*np.ones((6,23)),columns=FEATURES)
        train['product']='cn7';train['row_id']=range(6);train['shot_id']=['s'+str(i) for i in range(6)];train['y']=[0,0,0,1,1,1]
        query=train.iloc[[0,1]].copy();query['shot_id']=['q0','q1'];return train,query
    def test_query_label_and_side_ignored(self):
        t,q=self.frames();a,_=score_fold(t,q);q['y']=1-q.y;q['side']=['LH','RH'];b,_=score_fold(t,q)
        pd.testing.assert_frame_equal(a,b)
        c,_=score_fold(t,q.drop(columns=['y','side']));pd.testing.assert_frame_equal(a,c)
    def test_overlap_rejected(self):
        t,q=self.frames();q.loc[q.index[0],'shot_id']='s0'
        with self.assertRaises(ValueError):score_fold(t,q)
    def test_mold_mixing_rejected(self):
        t,q=self.frames();q['product']='rg3'
        with self.assertRaises(ValueError):score_fold(t,q)
        q['product']='cn7';t.loc[0,'product']='rg3'
        with self.assertRaises(ValueError):score_fold(t,q)
    def test_train_only_scaling(self):
        t,q=self.frames();_,a=score_fold(t,q);q[FEATURES]=100000.;_,b=score_fold(t,q)
        self.assertEqual(a,b);self.assertEqual(a['mean'],[2.5]*23)
    def test_score_bounds_and_missing_class(self):
        np.testing.assert_allclose(ambiguity(np.array([0.,1.,0.]),np.array([0.,1.,2.])),[1,1,0])
        t,q=self.frames();t.y=0;s,_=score_fold(t,q);self.assertTrue(s.ambiguity_score.isna().all())
