"""점검: 이 실험 폴더에서 만든 모델(harness.mk)이 run_all.py 의 같은 모델과 같은 점수를 내는가.
공개 원본(원 단위)으로 block CV 반복 0 을 직접 학습 → outputs/predictions/pipeline_<모델>.csv 의 OOF 와 비교.
같으면(순위 상관 ≈ 1, AP 차이 ≈ 0) 날짜 홀드아웃 등 직접 학습하는 실험도 레포 모델과 같은 모델로 본다."""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from dlib import load_public, PROC
import harness as H

Y = "PassOrFail"
pub = load_public()
rows = []
for mold in ("cn7", "rg3"):
    feats = [c for c in PROC if pub.loc[pub.mold == mold, c].nunique() > 1]
    fold = H.FOLD[H.FOLD.mold == mold].drop(columns=["mold", "shot_id"])
    lab = pub[pub.row_id.isin(fold.row_id)].merge(fold, on="row_id").reset_index(drop=True)
    ref = H.pipeline_oof(mold, lab["row_id"], lab[Y], 0)
    if ref is None:
        rows.append({"mold": mold, "status": "pipeline 예측 파일 없음/행 불일치"}); continue
    own = np.full(len(lab), np.nan)
    for f in sorted(lab["fold_block_r0"].unique()):
        tr, va = lab[lab.fold_block_r0 != f], lab[lab.fold_block_r0 == f]
        own[va.index] = H.mk().fit(tr[feats], tr[Y].values).predict_proba(va[feats])[:, 1]
    sh = lambda s: pd.DataFrame({"y": lab[Y], "s": s, "g": lab.shot_id}).groupby("g").max()
    a, b = sh(ref), sh(own)
    rows.append({"mold": mold, "status": "비교", "rows": len(lab), "spearman": spearmanr(ref, own)[0],
                 "max_abs_diff": np.abs(ref - own).max(), "shot_ap_pipeline": H.ap(a.y, a.s), "shot_ap_here": H.ap(b.y, b.s)})
R = pd.DataFrame(rows)
R.to_csv(os.path.join(H.OUT, "check_pipeline_match.csv"), index=False)
print("모델:", H.MODEL); print(R.round(4).to_string(index=False))
