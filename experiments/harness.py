"""실험 공통 하네스: 레포의 fold_id.csv 분할을 그대로 쓰고, 지표는 레포와 같은 방식(반복별 pooled OOF → 평균±표준편차)."""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp949 콘솔 대비
except Exception:
    pass
ID, Y = "Unnamed: 0", "PassOrFail"
import os, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # 레포 루트
FILES = {m: os.path.join(ROOT, "data", "raw", f"moldset_labeled_{m}.csv") for m in ("cn7", "rg3")}
_fold = [p for p in glob.glob(os.path.join(ROOT, "outputs", "**", "fold_id.csv"), recursive=True)]
assert _fold, "outputs/ 아래 fold_id.csv 가 없음 → run_all.py 를 먼저 실행"
FOLD = pd.read_csv(_fold[0], encoding="utf-8-sig")
N_REP = 5
SEED = 42

# ---------- 모델 선택 ----------
# KAMP_MODEL=logreg      : 기존 설정 (StandardScaler + LogisticRegression C=0.3 balanced) — 9/28~29 결과와 같은 정의
# KAMP_MODEL=<그 외 이름> : 레포 src/models.py 의 make_model(이름, seed=42) 를 그대로 사용 (예: logreg_spc)
MODEL = os.environ.get("KAMP_MODEL", "logreg")
OUT = os.path.join(ROOT, "outputs", "experiments", MODEL); os.makedirs(OUT, exist_ok=True)


class _RepoModel:
    """src.models.make_model 추정기 감싸기. DataFrame 으로 먼저 넣고, 안 되면 numpy 로 넣는다 (입력 형식 차이 흡수)."""
    def __init__(self, est):
        self.est, self.np_ = est, False

    def fit(self, X, y):
        y = np.asarray(y)
        try:
            self.est.fit(X, y)
        except Exception:
            self.np_ = True
            self.est.fit(np.asarray(X, dtype=float), y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(np.asarray(X, dtype=float) if self.np_ else X)


def mk():
    if MODEL == "logreg":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from src.models import make_model
    return _RepoModel(make_model(MODEL, SEED))


def pipeline_oof(mold, row_ids, y, rep, mode="block"):
    """run_all.py 가 만든 outputs/predictions/pipeline_<MODEL>.csv 의 OOF 점수를 row_ids 순서로 반환.
    파일이 없거나 행이 다 맞지 않으면 None (그때는 직접 학습). 라벨이 다르면 중단."""
    p = os.path.join(ROOT, "outputs", "predictions", f"pipeline_{MODEL}.csv")
    if MODEL == "logreg" or not os.path.exists(p):
        return None
    d = pd.read_csv(p, encoding="utf-8-sig")
    key = next((c for c in ("mold", "product") if c in d.columns and d[c].astype(str).str.lower().eq(mold).any()), None)
    if key is None:
        return None
    d = d[(d[key].astype(str).str.lower() == mold) & (d["cv_mode"] == mode) & (d["repeat"] == rep)]
    d = d.drop_duplicates("row_id").set_index("row_id")
    if not pd.Index(row_ids).isin(d.index).all():
        return None
    d = d.loc[list(row_ids)]
    assert (d["y_true"].values == np.asarray(y)).all(), "pipeline 예측 파일의 라벨이 실험 데이터와 다름"
    return d["y_score"].values.astype(float)


def load(mold):
    df = pd.read_csv(FILES[mold]).rename(columns={ID: "row_id"})
    fold = FOLD[FOLD["mold"] == mold].drop(columns="mold")
    df = df.merge(fold, on="row_id", how="left", validate="1:1")
    assert df["shot_id"].notna().all()
    feats = [c for c in df.columns if c not in ("row_id", Y, "shot_id", "block_id") and not c.startswith("fold_")]
    feats = [c for c in feats if df[c].nunique() > 1]  # drop_constant (Clamp_Open_Position)
    # 샷 순서 = 샷 안 최소 row_id (기록 순서). 라벨은 쓰지 않음
    order = df.groupby("shot_id")["row_id"].min().rank(method="first").astype(int) - 1
    df["shot_idx"] = df["shot_id"].map(order)
    return df, feats


def ap(y, s):
    return average_precision_score(y, s) if y.sum() > 0 else np.nan


def score_run(df, oof):
    """oof: row 순서의 점수. 행/샷 단위 AP·ROC-AUC·재현율@K."""
    d = pd.DataFrame({"y": df[Y].values, "s": oof, "shot": df["shot_id"].values})
    sh = d.groupby("shot").agg(y=("y", "max"), s=("s", "max"))
    out = {"row_ap": ap(d.y, d.s), "shot_ap": ap(sh.y, sh.s), "row_auc": roc_auc_score(d.y, d.s)}
    for k in (0.02, 0.05, 0.10):
        n = int(np.ceil(k * len(sh)))
        top = sh.sort_values("s", ascending=False, kind="mergesort").head(n)
        out[f"shot_recall@{int(k*100)}%"] = top.y.sum() / sh.y.sum()
    return out


def run_cv(df, feats, make_model, mode="block", fit_fn=None, per_fold=False):
    """make_model() → sklearn 추정기. fit_fn(model, tr, va) 가 있으면 그걸로 (tr,va)에 대한 점수 반환."""
    runs, folds = [], []
    for r in range(N_REP):
        col = f"fold_{mode}_r{r}"
        oof = np.full(len(df), np.nan)
        for f in sorted(df[col].unique()):
            tr, va = df[df[col] != f], df[df[col] == f]
            if fit_fn is not None:
                s = fit_fn(tr, va)
            else:
                m = make_model()
                m.fit(tr[feats], tr[Y].values)
                s = m.predict_proba(va[feats])[:, 1]
            oof[va.index] = s
            if per_fold:
                sv = pd.DataFrame({"y": va[Y].values, "s": s, "shot": va["shot_id"].values}).groupby("shot").max()
                folds.append({"repeat": r, "fold": f, "n_def_shots": int(sv.y.sum()), "shot_ap": ap(sv.y, sv.s)})
        runs.append(score_run(df, oof))
    res = pd.DataFrame(runs)
    summ = {k: (res[k].mean(), res[k].std(ddof=1)) for k in res.columns}
    return (summ, pd.DataFrame(folds)) if per_fold else summ


def fmt(summ, keys=("shot_ap", "row_ap", "row_auc", "shot_recall@2%", "shot_recall@5%")):
    return " | ".join(f"{k} {summ[k][0]:.3f}±{summ[k][1]:.3f}" for k in keys)


def fold_compare(fa, fb, label_a="base", label_b="alt", tol=1e-9):
    """fold 단위 승패 (레포 2-3 사전 기준과 같은 방식: 불량이 있는 fold 만)."""
    m = fa.merge(fb, on=["repeat", "fold", "n_def_shots"], suffixes=("_a", "_b"))
    m = m[m.n_def_shots > 0]
    d = m.shot_ap_b - m.shot_ap_a
    return {"folds": len(m), f"{label_b}_wins": int((d > tol).sum()), f"{label_a}_wins": int((d < -tol).sum()),
            "ties": int((d.abs() <= tol).sum())}
