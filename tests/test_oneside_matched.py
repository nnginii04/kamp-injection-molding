import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from oneside_matched import match_controls,effects

class MatchedTests(unittest.TestCase):
    def test_balanced_nearest_same_day_normal(self):
        idx=pd.date_range('2020-01-01 23:50',periods=35,freq='min')
        s=pd.DataFrame({'normal':True},index=idx);t=pd.Timestamp('2020-01-02 00:10')
        s.loc[t,'normal']=False;s.loc[t-pd.Timedelta(minutes=1),'normal']=False
        c=match_controls(s,t)
        self.assertEqual(len(c),10)
        self.assertTrue((c.index.date==t.date()).all())
        self.assertEqual(sum(c.index<t),5);self.assertEqual(sum(c.index>t),5)
        self.assertTrue(c.normal.all())
        self.assertEqual(c.index[c.index<t].max(),t-pd.Timedelta(minutes=2))
    def test_no_borrow_from_other_day(self):
        idx=pd.date_range('2020-01-01 23:50',periods=20,freq='min')
        self.assertIsNone(match_controls(pd.DataFrame({'normal':True},index=idx),pd.Timestamp('2020-01-02 00:02')))
    def test_direction_and_zero_variance_coverage(self):
        d=pd.DataFrame({'feature':['a']*3,'z':[2.,-3.,float('nan')],'defect_value':[3.,0.,2.],
          'control_median':[1.,1.,1.],'difference':[2.,-1.,1.],'zero_sd_changed':[False,False,True]})
        r=effects(d).iloc[0]
        self.assertEqual(r.n_finite_z,2);self.assertEqual(r.zero_sd_changed,1)
        self.assertAlmostEqual(r.fraction_above,2/3)
