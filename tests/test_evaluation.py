"""run_evaluation.py / evaluation 모듈 동작 검사 (임시 폴더, 저장소 밖 cwd 에서 실행)."""
from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
import uuid

import numpy as np
import pandas as pd

from _common import ROOT


def run_eval(*args, cwd="/"):
    return subprocess.run([sys.executable, str(ROOT / "run_evaluation.py"), *args], cwd=cwd,
                          capture_output=True, text=True)


class TestRunEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rel = f"outputs/_test_eval_{uuid.uuid4().hex[:8]}"   # 저장소 루트 기준 상대경로 (git 제외 폴더)
        cls.root = ROOT / cls.rel
        pred = cls.root / "predictions"
        pred.mkdir(parents=True)
        rng = np.random.default_rng(0)
        n = 40  # 20 shot × 2행, 전역 shot 은 (0,1),(2,3)... 이지만 파일은 (1,2),(3,4)... 로 잘못 묶음
        row = np.arange(n)
        glob_shot = [f"cn7_{i // 2}" for i in row]
        y = np.zeros(n, int)
        y[[0, 1, 10, 21]] = 1
        pd.DataFrame({"row_id": row, "shot_id": glob_shot, "defect_type": "normal"}).to_csv(
            cls.root / "predictions_labeled_cv.csv", index=False)
        base = pd.DataFrame({"row_id": row, "product": "cn7", "y_true": y, "y_score": rng.random(n) + y,
                             "repeat": 0, "fold": row % 3})
        base.assign(model="good", cv_mode="block", shot_id=[f"bad_{(i + 1) // 2}" for i in row]).to_csv(
            pred / "good.csv", index=False)
        base.drop(columns=["y_true"]).assign(model="nolabel").to_csv(pred / "nolabel.csv", index=False)
        base.assign(model="nocv").to_csv(pred / "nocv.csv", index=False)
        base.assign(model="allneg", cv_mode="shot", y_true=0).to_csv(pred / "allneg.csv", index=False)
        cls.res = run_eval("--outputs", cls.rel)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_runs_from_other_cwd_with_relative_paths(self):
        self.assertEqual(self.res.returncode, 0, self.res.stderr)
        self.assertTrue((self.root / "evaluation" / "experiment_summary.csv").exists())

    def test_skips_file_without_y_true(self):
        self.assertIn("건너뛴 파일", self.res.stdout)
        self.assertIn("nolabel.csv", self.res.stdout)
        s = pd.read_csv(self.root / "evaluation" / "experiment_summary.csv")
        self.assertNotIn("nolabel", set(s.model))

    def test_default_rule_is_printed(self):
        self.assertIn("기본값", self.res.stdout)
        self.assertIn("y_pred", self.res.stdout)

    def test_threshold_and_budget_are_exclusive(self):
        r = run_eval("--outputs", self.rel, "--threshold", "0.5", "--budget", "0.02")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not allowed with", r.stderr)

    def test_tables_split_by_cv_mode_and_warn_missing(self):
        md = (self.root / "evaluation" / "experiment_summary.md").read_text(encoding="utf-8")
        self.assertIn("cv_mode = block", md)
        self.assertIn("cv_mode = shot", md)
        self.assertIn("(cv_mode 없음)", md)
        self.assertIn("nocv.csv: cv_mode 열이 없음", md)
        self.assertIn("cv_mode = (cv_mode 없음)", self.res.stdout)

    def test_lift_nan_shown_as_dash(self):
        md = (self.root / "evaluation" / "experiment_summary.md").read_text(encoding="utf-8")
        self.assertNotIn("nanx", md)
        self.assertNotIn("nanx", self.res.stdout)
        self.assertIn("(-)", md)

    def test_file_shot_id_overridden_by_global(self):
        md = (self.root / "evaluation" / "experiment_summary.md").read_text(encoding="utf-8")
        self.assertIn("파일의 shot_id 가 전역 기준과 다름", md)
        s = pd.read_csv(self.root / "evaluation" / "experiment_summary.csv")
        shot = s[(s.model == "good") & (s.level == "shot")].iloc[0]
        self.assertEqual(int(shot.n), 20)      # 전역 기준 20 shot (파일 기준이면 21)
        self.assertEqual(int(shot.n_pos), 3)   # 전역 shot 0, 5, 10 이 불량


if __name__ == "__main__":
    unittest.main()
