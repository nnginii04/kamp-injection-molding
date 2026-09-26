"""실행 결과 파일 형식 검사 (run_all.py 실행 후).

`python run_all.py --synthetic --fast` 결과(outputs/synthetic/)가 있으면 그것을, 없으면 건너뛴다.
대회 데이터 실행 결과(outputs/)가 있으면 같은 검사를 한다.
"""
from __future__ import annotations

import json
import unittest

import pandas as pd

from _common import ROOT

DIRS = [d for d in (ROOT / "outputs" / "synthetic", ROOT / "outputs") if (d / "report.md").exists()]


class TestOutputs(unittest.TestCase):
    def setUp(self):
        if not DIRS:
            self.skipTest("run_all.py 결과가 없음 (python run_all.py --synthetic --fast 먼저 실행)")

    def test_prediction_files_have_mold_and_uncertainty(self):
        for d in DIRS:
            for name in ("predictions_labeled_cv.csv", "predictions_unlabeled.csv", "predictions_holdout.csv"):
                f = d / name
                if not f.exists():
                    continue
                t = pd.read_csv(f, nrows=2000)
                with self.subTest(file=str(f)):
                    self.assertIn("mold", t.columns)
                    self.assertTrue(t["mold"].isin(["cn7", "rg3"]).all())
                    self.assertIn("uncertainty", t.columns)
                    self.assertIn("uncertainty_n", t.columns)

    def test_unlabeled_overlap_flag_and_meta(self):
        for d in DIRS:
            t = pd.read_csv(d / "predictions_unlabeled.csv", usecols=["mold", "in_labeled_train"])
            meta = json.loads((d / "predictions_unlabeled_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(any("labeled 행이 들어 있음" in w for w in meta["warnings"]))
            self.assertTrue(any("과소평가" in w for w in meta["warnings"]))
            cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
            if not cfg["external"]["enabled"]:
                self.assertTrue(t["in_labeled_train"].isna().all(), "external 꺼짐인데 in_labeled_train 이 채워짐")

    def test_fold_id_file(self):
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        for d in DIRS:
            f = pd.read_csv(d / "fold_id.csv")
            self.assertTrue({"mold", "row_id", "shot_id", "block_id"} <= set(f.columns))
            fold_cols = [c for c in f.columns if c.startswith("fold_")]
            self.assertTrue(fold_cols)
            self.assertTrue({c.split("_")[1] for c in fold_cols} <= set(cfg["cv"]["modes"]))
            self.assertTrue(f.row_id.is_unique)
            for c in fold_cols:
                self.assertEqual(f.groupby("shot_id")[c].nunique().max(), 1, f"{c}: 짝이 다른 fold")
                if c.startswith("fold_block_"):
                    self.assertEqual(f.groupby("block_id")[c].nunique().max(), 1, f"{c}: block 이 다른 fold")
            self.assertTrue(f.shot_id.str.startswith(("cn7_", "rg3_")).all())

    def test_ablation_marked_as_leak(self):
        for d in DIRS:
            f = d / "tables" / "ablation_pair_position.csv"
            if f.exists():
                t = pd.read_csv(f)
                self.assertTrue((t["note"] == "누수 변수 — 비교용").all())
                self.assertIn("누수 변수 — 비교용", (d / "report.md").read_text(encoding="utf-8"))

    def test_experiment_tables_have_ceilings(self):
        for d in DIRS:
            for name in ("exp_ab", "exp_class_weight", "exp_dummies", "exp_baseline_v0", "perf_range",
                         "exp_threshold_cn7", "exp_budget_cn7"):
                f = d / "tables" / f"{name}.csv"
                if not f.exists():
                    continue
                t = pd.read_csv(f)
                with self.subTest(file=str(f)):
                    self.assertIn("row_P_ceiling", t.columns)
                    self.assertIn("row_F1_ceiling", t.columns)

    def test_baseline_v0_reproduced(self):
        d = ROOT / "outputs"
        f = d / "tables" / "exp_baseline_v0.csv"
        if not f.exists():
            self.skipTest("대회 데이터 실행 결과 없음")
        t = pd.read_csv(f, dtype={"step": str})
        c = t[t["product"] == "cn7"].set_index("step")
        if "0" in c.index:
            self.assertAlmostEqual(c.loc["1", "row_ap"], c.loc["0", "row_ap"], places=6,
                                   msg="v0 설정 재현값이 archive 보고값과 다름")

    def test_report_notes(self):
        for d in DIRS:
            r = (d / "report.md").read_text(encoding="utf-8")
            self.assertIn("fold 별 상수 예측을 합친 착시", r)
            self.assertIn("fold 수 점검", r)


if __name__ == "__main__":
    unittest.main()
