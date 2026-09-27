"""사실 10 겹침 판정 기준(scripts/verify_public_link.overlap) 단위 검사 — 공개 원본 없이 작은 표로 확인."""
from __future__ import annotations

import sys
import unittest

import numpy as np
import pandas as pd

from _common import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from verify_public_link import overlap  # noqa: E402

FEATS = [f"f{i}" for i in range(24)]


def frame(rows):
    out = []
    for rid, ts, part, base in rows:
        r = {"row_id": rid, "TimeStamp": ts, "PART_NAME": part}
        r.update({f: base + i for i, f in enumerate(FEATS)})
        out.append(r)
    return pd.DataFrame(out)


class TestOverlapCriterion(unittest.TestCase):
    def test_rules(self):
        lab = frame([(0, "t1", "X LH", 24.8), (1, "t1", "X RH", 24.8),   # 짝
                     (2, "t2", "X LH", 30.0),                            # 단독 (짝은 unlabeled 에만)
                     (3, "t3", "X LH", 40.0)])                           # 값이 다름
        lab.loc[lab.row_id.isin([0, 1]), "f0"] = np.float32(24.8).item()  # 24.799999237... (float32)
        unl = frame([(100, "t1", "X LH", 24.8), (101, "t1", "X RH", 24.8),
                     (102, "t2", "X RH", 30.0),                          # 단독 행의 짝 → shot 만 일치
                     (103, "t3", "X LH", 40.5),                          # 키는 같지만 값이 다름
                     (104, "t9", "X LH", 50.0), (105, "t9", "X LH", 50.0)])  # 키 중복 → 1:1 아님, 제외
        flags, info = overlap(lab, unl, FEATS)
        f = flags.set_index("unlabeled_row_id")
        self.assertTrue(f.loc[100, "in_labeled_train_row"] and f.loc[101, "in_labeled_train_row"])
        self.assertEqual(f.loc[100, "labeled_row_id"], 0)
        self.assertFalse(f.loc[102, "in_labeled_train_row"])
        self.assertTrue(f.loc[102, "in_labeled_train_shot"])
        self.assertFalse(f.loc[103, "in_labeled_train_row"] or f.loc[103, "in_labeled_train_shot"])
        self.assertEqual(info["key_matched_1to1"], 3)            # 100, 101, 103 (102 는 PART_NAME 다름)
        self.assertEqual(info["key_matched_values_equal"], 2)
        self.assertEqual(info["key_matched_values_differ"], 1)
        self.assertEqual(info["unlabeled_dup_keys"], 2)
        self.assertLess(info["max_raw_diff_before_rounding"], 1e-5)  # float32 차이는 반올림으로 흡수


if __name__ == "__main__":
    unittest.main()
