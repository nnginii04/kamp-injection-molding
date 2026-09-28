"""Saved-fold target ablation contracts; no generated CV assignments."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from _common import load_cfg
from evaluation._common import load_predictions, to_level
from src.data import assign_shots
from src.shot_experiment import PRED_COLS, prepare, run_ablation, summarize


def fixture():
    rows, folds = [], []
    for p in ("cn7", "rg3"):
        for shot in range(9):
            ys = ([0, 0], [0, 1], [1, 1])[shot % 3]
            if shot == 8:
                ys = [1]  # singleton
            for side, y in enumerate(ys):
                rid = (0 if p == "cn7" else 100) + shot * 2 + side
                rows.append(dict(product=p, row_id=rid, y=y, x=float(shot)))
                folds.append(dict(mold=p, row_id=rid, shot_id=f"{p}_{shot}", block_id=f"{p}_b{shot // 3}",
                                  fold_block_r0=shot // 3, fold_block_r1=(shot // 3 + 1) % 3))
    lab = pd.concat([assign_shots(pd.DataFrame(rows).query("product == @p"), ["x"], p)
                     for p in ("cn7", "rg3")], ignore_index=True)
    cfg = copy.deepcopy(load_cfg())
    cfg["cv"]["n_repeats"] = 2
    return lab, pd.DataFrame(folds), cfg


class TestShotExperiment(unittest.TestCase):
    def test_summary_matches_common_evaluator_including_ties(self):
        from evaluation.metrics import evaluate_model
        from evaluation.recall_at_k import calculate_recall_at_k
        lab, folds, cfg = fixture()
        pred, _ = run_ablation(lab, folds, ["x"], cfg, log=lambda _: None)
        pred["y_score"] = pred.y_score.round(1)  # explicit boundary ties
        _, summary = summarize(pred)
        for _, r in summary.iterrows():
            part = pred[pred["product"].eq(r["product"]) & pred.model.eq(r["model"])]
            m = evaluate_model(part, level=r.level)
            k = calculate_recall_at_k(part, ks=(0.05, 0.10, 0.20), level=r.level).set_index("k_label")
            self.assertAlmostEqual(r.ap_mean, m["pr_auc_mean"])
            self.assertAlmostEqual(r.ap_sd, m["pr_auc_std"])
            for budget in (5, 10, 20):
                self.assertAlmostEqual(r[f"recall@{budget}%_mean"], k.loc[f"{budget}%", "recall_mean"])
                self.assertAlmostEqual(r[f"recall@{budget}%_sd"], k.loc[f"{budget}%", "recall_std"])

    def test_any_defect_and_singleton(self):
        lab, folds, cfg = fixture()
        row, shots, _ = prepare(lab, folds, ["x"], 2, 3)
        self.assertEqual(len(shots), 18)
        self.assertTrue(shots.shot_id.is_unique)
        self.assertEqual(shots.set_index("shot_id").loc[["cn7_0", "cn7_1", "cn7_2", "cn7_8"], "y"].tolist(), [0, 1, 1, 1])
        pd.testing.assert_series_equal(row.y.reset_index(drop=True), lab.y.reset_index(drop=True))

    def test_inconsistent_shot_fails_before_fit(self):
        lab, folds, cfg = fixture()
        folds.loc[1, "fold_block_r1"] = 2
        with patch("src.shot_experiment.make_model") as model:
            with self.assertRaisesRegex(ValueError, "shot_id occurs in multiple folds"):
                run_ablation(lab, folds, ["x"], cfg)
            model.assert_not_called()

    def test_missing_or_mismatched_rows_and_shots_rejected(self):
        lab, folds, _ = fixture()
        for bad in (folds.iloc[1:], pd.concat([folds, folds.iloc[:1]]),
                    folds.assign(shot_id="wrong"), folds.drop(columns="fold_block_r1")):
            with self.subTest(kind=len(bad)):
                with self.assertRaises(ValueError):
                    prepare(lab, bad, ["x"], 2, 3)

    def test_block_split_rejected(self):
        lab, folds, _ = fixture()
        folds.loc[folds.shot_id.eq("cn7_0"), "fold_block_r0"] = 1
        with self.assertRaisesRegex(ValueError, "block_id"):
            prepare(lab, folds, ["x"], 2, 3)

    def test_label_derived_features_rejected(self):
        lab, folds, _ = fixture()
        for name in ("y_shot", "conflict", "n_pos", "n_rows", "defect_type"):
            with self.subTest(feature=name), self.assertRaises(ValueError):
                prepare(lab, folds, ["x", name], 2, 3)

    def test_fixed_splits_train_only_and_oof_mapping(self):
        from src.models import make_model
        lab, folds, cfg = fixture()
        original_lab, original_folds = lab.copy(deep=True), folds.copy(deep=True)
        seen = []

        def tracked(name, seed):
            m = make_model(name, seed)
            fit = m.fit
            def wrapped(X, y):
                result = fit(X, y)
                np.testing.assert_allclose(m.steps[0][1].mean_, X.mean().to_numpy())
                seen.append((X.x.tolist(), len(y)))
                return result
            m.fit = wrapped
            return m

        with patch("sklearn.model_selection.StratifiedGroupKFold.split", side_effect=AssertionError("No new splits")), \
                patch("src.shot_experiment.make_model", side_effect=tracked):
            pred, counts = run_ablation(lab, folds, ["x"], cfg, log=lambda _: None)
        self.assertEqual(list(pred.columns), PRED_COLS)
        self.assertEqual(len(pred), len(lab) * 2 * 2)
        self.assertFalse(pred.duplicated(["product", "model", "repeat", "row_id"]).any())
        expected = lab.set_index(["product", "row_id"]).y
        for (model, rep), part in pred.groupby(["model", "repeat"]):
            joined = part.merge(folds, on=["row_id", "shot_id"], suffixes=("", "_saved"))
            self.assertTrue(joined.fold.eq(joined[f"fold_block_r{rep}"]).all())
            self.assertTrue(part.set_index(["product", "row_id"]).y_true.sort_index().equals(expected.sort_index().rename("y_true")))
            self.assertTrue(part.groupby("shot_id").y_score.nunique().eq(1).all())
        for (_, c), (xs, n) in zip(counts.iterrows(), seen):
            excluded = (c["fold"] - c["repeat"]) % 3
            self.assertFalse(any(int(x) // 3 == excluded for x in xs))
            self.assertEqual(n, c.train_examples)
            if c.target == "shot":
                self.assertEqual(n, len(set(xs)))
        mapped = load_predictions(pred, shot_map=lab[["row_id", "shot_id"]])
        shot = to_level(mapped, "shot", "repeat")
        self.assertEqual(len(shot), 18 * 2 * 2)
        detail, summary = summarize(pred)
        self.assertEqual(len(summary), 8)
        self.assertTrue({"ap", "recall@5%", "recall@10%", "recall@20%"} <= set(detail))
        pd.testing.assert_frame_equal(lab, original_lab)
        pd.testing.assert_frame_equal(folds, original_folds)


if __name__ == "__main__":
    unittest.main()
