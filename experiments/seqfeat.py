"""시계열(직전 샷) 피처와 baseline logreg 정의. E1 변형 설명은 e1_sequence.py 참고."""
from harness import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def add_seq(df, feats, kind):
    shot = df.groupby("shot_idx")[feats].first().sort_index()   # 짝은 공정값이 같음 → 샷 단위 시계열
    new = {}
    if kind in ("d1", "both"):
        d1 = shot.diff(1).fillna(0.0); d1.columns = [f"{c}__d1" for c in feats]; new["d1"] = d1
    if kind in ("dev5", "both"):
        med = shot.shift(1).rolling(5, min_periods=1).median()
        dv = (shot - med).fillna(0.0); dv.columns = [f"{c}__dev5" for c in feats]; new["dev5"] = dv
    add = pd.concat(new.values(), axis=1)
    out = df.merge(add, left_on="shot_idx", right_index=True, how="left")
    out.index = df.index
    return out, feats + list(add.columns)

mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))
