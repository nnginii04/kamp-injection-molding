"""logreg_ae(지도 + 양품 오토인코더 결합) 점검: 점수가 채점 묶음에 의존하지 않는지(실시간 1건 채점과 같은지),
0~1 범위인지, 이상 점수는 양품만으로 학습되는지."""
import sys
import unittest
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import HAS_SMOTE, AENormal, make_model  # noqa: E402


def toy(seed=0, n=300, d=6):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"x{i}" for i in range(d)])
    y = (rng.random(n) < 0.05).astype(int)
    X.loc[y == 1, "x0"] += 3
    return X, y


class TestSupAnom(unittest.TestCase):
    def test_batch_independent(self):
        X, y = toy()
        for name in ("logreg_ae", "logreg_if", "logreg_spc", "ae_normal"):
            m = make_model(name, 42).fit(X, y)
            Xn, _ = toy(seed=1, n=50)  # 학습에 없던 새 행 (학습 행은 자기 점수가 기준 분포에 있어 부동소수 동점이 생김)
            full = m.predict_proba(Xn)[:, 1]
            one = np.array([m.predict_proba(Xn.iloc[[i]])[:, 1][0] for i in range(10)])
            np.testing.assert_allclose(full[:10], one, atol=1e-9, err_msg=name)
            self.assertTrue(((full >= 0) & (full <= 1)).all(), name)

    def test_ae_uses_normal_only(self):
        X, y = toy()
        a = AENormal(random_state=0).fit(X, y)
        X2 = X.copy()
        X2.loc[y == 1] = 1e3  # 불량 행 값을 바꿔도 정상만 쓰면 모델이 같아야 함
        b = AENormal(random_state=0).fit(X2, y)
        np.testing.assert_allclose(a.raw_score(X[y == 0]), b.raw_score(X[y == 0]), atol=1e-12)


class TestSmote(unittest.TestCase):
    @unittest.skipUnless(HAS_SMOTE, "imbalanced-learn 없음")
    def test_smote_only_inside_fit(self):
        """SMOTE 는 fit 에 들어온 학습 데이터에만 적용되고, 예측 시에는 입력 행 수 그대로 점수를 낸다."""
        X, y = toy()
        m = make_model("lgbm_smote", 42).fit(X, y)
        self.assertEqual(m.n_synthetic_, int((y == 0).sum() - (y == 1).sum()))  # 1:1 까지 합성
        self.assertEqual(m.predict_proba(X.iloc[:7]).shape, (7, 2))
        few = y.copy(); few[:] = 0; few[0] = 1  # 불량 1개면 SMOTE 없이 학습
        self.assertEqual(make_model("logreg_smote", 42).fit(X, few).n_synthetic_, 0)


if __name__ == "__main__":
    unittest.main()
