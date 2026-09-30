import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cn7_time_shift_balanced import balanced_sample, date_permutation

class BalancedTests(unittest.TestCase):
    def test_equal_unique_reproducible(self):
        d=pd.DataFrame({'production_date':['a']*8+['b']*40,'id':range(48)})
        a=balanced_sample(d,8,42)
        self.assertEqual(a.groupby('production_date').size().tolist(),[8,8])
        self.assertEqual(a.id.nunique(),16)
        pd.testing.assert_frame_equal(a,balanced_sample(d,8,42))
    def test_impossible_cap_rejected(self):
        d=pd.DataFrame({'production_date':['a']*8+['b']*40})
        with self.assertRaises(ValueError): balanced_sample(d,30,42)
    def test_exact_date_permutations(self):
        d=pd.DataFrame({'production_date':[str(i) for i in range(12)],'x':range(12)})
        r=date_permutation(d,['x'],[str(i) for i in range(6)])
        self.assertEqual(r['n_assignments'],924)
        self.assertGreaterEqual(r['exact_date_permutation_p'],2/924)
