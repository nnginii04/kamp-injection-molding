import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_oct27_features import standardized_difference

class FeatureAuditTests(unittest.TestCase):
    def test_uses_control_statistics_only(self):
        c=pd.DataFrame({'a':[1.,2.,3.]})
        z,median,sd=standardized_difference(pd.DataFrame({'a':[5.,100.]}),c)
        self.assertEqual(median.a,2.)
        self.assertEqual(sd.a,1.)
        self.assertEqual(z.a.tolist(),[3.,98.])
    def test_constant_not_infinite(self):
        z,_,sd=standardized_difference(pd.DataFrame({'a':[9.]}),pd.DataFrame({'a':[2.,2.,2.]}))
        self.assertEqual(sd.a,0.)
        self.assertTrue(np.isnan(z.a.iloc[0]))
