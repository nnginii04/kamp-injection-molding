import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_oct27 import timeline

class AuditTests(unittest.TestCase):
    def test_sort_dedupe_gap_and_restart(self):
        t=pd.Series(['2020-10-27 00:11:00','2020-10-27 00:00:00','2020-10-27 00:10:00','2020-10-27 00:10:00'])
        s=timeline(t)
        self.assertEqual(len(s),3)
        self.assertEqual(s.gap_sec.iloc[1:].tolist(),[600,60])
        self.assertEqual(s.shot_since_candidate.tolist(),[1,1,2])
        self.assertFalse(s.restart_candidate.iloc[0])
    def test_threshold_strictly_greater(self):
        t=pd.Series(['2020-10-27 00:00:00','2020-10-27 00:05:00'])
        self.assertFalse(timeline(t).restart_candidate.any())
