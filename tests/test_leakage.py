"""누수 방지 자동 검사.

    python -m unittest discover -s tests -v      (pytest 가 있으면 pytest tests/ 도 가능)

- 같은 shot 이 train/valid 에 동시에 있지 않음 (shot·block CV 모두)
- 블록 CV 에서 같은 block 이 train/valid 에 동시에 있지 않음, block 은 연속 샷
- shot_id 는 전역 기준 (공정값 전체가 같은 행) — 연속 행만 묶는 방식 금지
- 모델 입력에 금지 변수 없음 (Unnamed: 0 / PassOrFail / Clamp_Open_Position / shot_id / 위치 변수)
- 전처리·이상탐지 기준은 train fold 에만 fit, 판정 임계값은 train fold 내부에서만 결정
- 보정(Platt) 내부 CV 도 shot 단위, fold_id.csv 는 run_cv 와 같은 분할
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from _common import has_real_data, load_cfg, real, synthetic

from src import cv as cv_mod
from src import features
from src.final import platt_splits
from src.models import SPCMaxZ, make_model


def _base_feats(cfg, data):
    base, _ = features.select_base_features(data, data["feature_cols"], cfg["features"]["drop_constant"])
    return base


class TestConfigDefaults(unittest.TestCase):
    def test_leaky_options_off(self):
        cfg = load_cfg()
        f = cfg["features"]
        self.assertNotIn("pair_position", f, "pair_position 은 features 에서 제거하고 ablation 으로만")
        self.assertFalse(f["sequence_features"])
        self.assertFalse(f["unlabeled_iforest_score"])
        self.assertFalse(cfg["external"]["enabled"])
        self.assertFalse(cfg["holdout"]["enabled"])

    def test_pair_position_in_features_is_rejected(self):
        with self.assertRaises(ValueError):
            features.validate_feature_config({"pair_position": True})


class TestFeatureList(unittest.TestCase):
    def test_model_features_have_no_forbidden(self):
        cfg, data = synthetic()
        base = _base_feats(cfg, data)
        for scenario in ("separate", "pooled"):
            feats = features.model_features(base, scenario, cfg["features"]["unlabeled_iforest_score"])
            features.check_feature_list(feats)
            for bad in ("Unnamed: 0", "row_id", "PassOrFail", "y", "Clamp_Open_Position", "shot_id",
                        "block_id", "pair_first", "position_in_shot", "if_score"):
                self.assertNotIn(bad, feats)

    def test_check_feature_list_catches_leaks(self):
        for bad in ("pair_first", "shot_id", "Unnamed: 0", "Clamp_Open_Position", "PassOrFail"):
            with self.assertRaises(AssertionError):
                features.check_feature_list(["Injection_Time", bad])
        # ablation 에서만 명시적으로 허용
        features.check_feature_list(["Injection_Time", "pair_first"], allow=features.LEAKY_ABLATION_FEATURES)

    @unittest.skipUnless(has_real_data(), "data/raw 에 대회 CSV 가 없음")
    def test_constant_column_dropped_real(self):
        cfg, data = real()
        self.assertNotIn("Clamp_Open_Position", _base_feats(cfg, data))


class TestShotId(unittest.TestCase):
    @unittest.skipUnless(has_real_data(), "data/raw 에 대회 CSV 가 없음")
    def test_global_shot_counts_real(self):
        _, data = real()
        expect = {"cn7": (606, {1: 1, 2: 605}), "rg3": (591, {2: 591})}
        for p, (n, hist) in expect.items():
            lab = data["labeled"][p]
            self.assertEqual(lab.shot_id.nunique(), n)
            self.assertEqual(lab.groupby("shot_id").size().value_counts().to_dict(), hist)

    @unittest.skipUnless(has_real_data(), "data/raw 에 대회 CSV 가 없음")
    def test_separated_pairs_share_shot_id_real(self):
        """row_id 가 떨어져 있는 짝도 같은 shot_id (연속 행만 묶는 02 방식이면 갈라짐)."""
        _, data = real()
        for p in ("cn7", "rg3"):
            lab = data["labeled"][p]
            span = lab.groupby("shot_id").row_id.agg(lambda s: s.max() - s.min())
            self.assertGreater(int((span > 1).sum()), 0)
            feats = data["feature_cols"]
            key = pd.util.hash_pandas_object(lab[feats].round(10), index=False)
            self.assertEqual(key.nunique(), lab.shot_id.nunique())

    def test_same_process_values_same_shot(self):
        _, data = synthetic()
        feats = data["feature_cols"]
        for lab in data["labeled"].values():
            per_shot = lab.groupby("shot_id")[feats].nunique().max().max()
            self.assertEqual(per_shot, 1)


class TestSplits(unittest.TestCase):
    def _check(self, cfg, data):
        for p, lab in data["labeled"].items():
            for mode in cfg["cv"]["modes"]:
                groups = cv_mod.make_groups(lab, mode, cfg["cv"]["block_shots"])
                for rep in range(2):
                    for tr, va in cv_mod.cv_splits(lab, mode, cfg, rep):
                        self.assertFalse(set(lab.shot_id.iloc[tr]) & set(lab.shot_id.iloc[va]),
                                         f"{p}/{mode}: 같은 shot 이 train/valid 에 동시에")
                        self.assertFalse(set(groups[tr]) & set(groups[va]),
                                         f"{p}/{mode}: 같은 그룹이 train/valid 에 동시에")
                        self.assertEqual(len(set(tr) | set(va)), len(lab))

    def test_splits_disjoint_synthetic(self):
        self._check(*synthetic())

    @unittest.skipUnless(has_real_data(), "data/raw 에 대회 CSV 가 없음")
    def test_splits_disjoint_real(self):
        self._check(*real())

    def test_blocks_are_contiguous_shots(self):
        cfg, data = synthetic()
        k = cfg["cv"]["block_shots"]
        for lab in data["labeled"].values():
            blocks = pd.Series(cv_mod.make_groups(lab, "block", k), index=lab.index)
            order = lab.groupby("shot_id").row_id.min().sort_values()
            shot_block = lab.groupby("shot_id").apply(lambda g: blocks[g.index].unique().tolist())
            self.assertTrue((shot_block.map(len) == 1).all(), "한 shot 이 두 block 에 걸침")
            seq = shot_block.map(lambda b: b[0]).reindex(order.index).tolist()
            changes = [a != b for a, b in zip(seq, seq[1:])]
            self.assertEqual(sum(changes), len(set(seq)) - 1, "block 이 연속 샷이 아님")
            self.assertLessEqual(pd.Series(seq).value_counts().max(), k)

    def test_fold_table_matches_cv_splits(self):
        cfg, data = synthetic()
        lab = data["labeled"]["cn7"]
        ft = cv_mod.fold_table(lab, cfg).set_index("row_id")
        self.assertTrue({"mold", "shot_id", "block_id"} <= set(ft.columns))
        for mode in cfg["cv"]["modes"]:
            for rep in range(cfg["cv"]["n_repeats"]):
                col = f"fold_{mode}_r{rep}"
                self.assertIn(col, ft.columns)
                for k, (_, va) in enumerate(cv_mod.cv_splits(lab, mode, cfg, rep)):
                    self.assertTrue((ft.loc[lab.row_id.iloc[va], col] == k).all())
            # 같은 shot 은 모든 반복에서 같은 fold
            self.assertEqual(ft.groupby("shot_id")[f"fold_{mode}_r0"].nunique().max(), 1)


class TestExperimentSplits(unittest.TestCase):
    def test_pooled_splits_are_union_of_product_folds(self):
        from src import experiments as E
        cfg, data = synthetic()
        frames = {p: (lab, None) for p, lab in data["labeled"].items()}
        products = list(frames)
        lab, labs, offsets = E.pooled_frame(frames, products)
        for mode in cfg["cv"]["modes"]:
            pooled = E.pooled_splits_fn(labs, offsets, mode, cfg)(0)
            per = [cv_mod.cv_splits(x, mode, cfg, 0) for x in labs]
            for k, (tr, va) in enumerate(pooled):
                expect = set(np.concatenate([per[i][k][1] + offsets[i] for i in range(len(labs))]))
                self.assertEqual(set(va), expect)
                self.assertFalse(set(lab.shot_id.iloc[tr]) & set(lab.shot_id.iloc[va]))

    def test_experiment_models_not_selection_candidates(self):
        cfg = load_cfg()
        from src.models import BASE_MODEL
        self.assertFalse(set(BASE_MODEL) & set(cfg["models"]), "실험 변형 모델이 본 후보(config.models)에 들어감")


class TestFitOnTrainOnly(unittest.TestCase):
    def setUp(self):
        cfg, data = synthetic()
        self.cfg = cfg
        self.lab = data["labeled"]["cn7"]
        self.feats = _base_feats(cfg, data)
        self.tr, self.va = cv_mod.cv_splits(self.lab, "block", cfg, 0)[0]

    def test_scaler_fit_on_train(self):
        m = make_model("logreg", 0).fit(self.lab[self.feats].iloc[self.tr], self.lab.y.iloc[self.tr])
        scaler = m.steps[0][1]
        np.testing.assert_allclose(scaler.mean_, self.lab[self.feats].iloc[self.tr].mean().to_numpy())

    def test_spc_reference_is_train_normal(self):
        Xtr, ytr = self.lab[self.feats].iloc[self.tr], self.lab.y.iloc[self.tr].to_numpy()
        m = SPCMaxZ().fit(Xtr, ytr)
        np.testing.assert_allclose(m.mu_, Xtr[ytr == 0].mean().to_numpy())

    def test_inner_threshold_sees_train_rows_only(self):
        calls = []
        orig = cv_mod.inner_threshold

        def spy(model, X, y, groups, n_splits, seed):
            calls.append(set(X.index))
            return orig(model, X, y, groups, n_splits, seed)

        cfg = {**self.cfg, "cv": {**self.cfg["cv"], "n_repeats": 1}}
        cv_mod.inner_threshold = spy
        try:
            cv_mod.run_cv(self.lab, self.feats, ["logreg"], cfg, "cn7", "block", log=lambda *_: None)
        finally:
            cv_mod.inner_threshold = orig
        splits = cv_mod.cv_splits(self.lab.reset_index(drop=True), "block", cfg, 0)
        fitted = [(tr, va) for tr, va in splits if self.lab.y.iloc[tr].sum() > 0]
        self.assertEqual(len(calls), len(fitted))
        for seen, (tr, va) in zip(calls, fitted):
            self.assertEqual(seen, set(tr))
            self.assertFalse(seen & set(va))

    def test_platt_inner_cv_is_shot_grouped(self):
        lab = self.lab
        for tr, va in platt_splits(lab.y.to_numpy(), lab.shot_id.to_numpy()):
            self.assertFalse(set(lab.shot_id.iloc[tr]) & set(lab.shot_id.iloc[va]))


if __name__ == "__main__":
    unittest.main()
