import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cn7_time_shift import split_dates


class DateSplitTests(unittest.TestCase):
    def test_actual_days_not_calendar_window(self):
        days = pd.date_range('2020-01-01', periods=14, freq='3D')
        frame = pd.DataFrame({'TimeStamp': list(days[::-1]) + [days[0]]})
        _, all_days, early, late = split_dates(frame)
        self.assertEqual(len(all_days), 14)
        self.assertEqual(early, [d.strftime('%Y-%m-%d') for d in days[:6]])
        self.assertEqual(late, [d.strftime('%Y-%m-%d') for d in days[-6:]])
        self.assertFalse(set(early) & set(late))

    def test_insufficient_dates_rejected(self):
        with self.assertRaises(ValueError):
            split_dates(pd.DataFrame({'TimeStamp': pd.date_range('2020-01-01', periods=11)}))

    def test_missing_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            split_dates(pd.DataFrame({'TimeStamp': [None]}))
