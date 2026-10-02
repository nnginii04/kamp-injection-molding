"""E3: logreg 규제 강도(C)·클래스 가중치를 중첩 CV 로 선택.
바깥: 레포 fold_id 분할. 안쪽: 학습 fold 안에서만 StratifiedGroupKFold(3) (block CV 는 block_id, shot CV 는 shot_id 로 묶음),
안쪽 pooled shot_ap 가 가장 높은 설정을 골라 학습 fold 전체로 다시 학습 → 검증 fold 채점. 검증 fold 는 선택에 쓰지 않음.
격자(사전 고정): C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3} × class_weight ∈ {balanced, None}
(E3 부터 추가한 채택 조건: block pooled shot_ap 상승폭 ≥ baseline 1 표준오차)"""
from harness import *
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from collections import Counter

GRID = [(C, cw) for C in (0.01, 0.03, 0.1, 0.3, 1, 3) for cw in ("balanced", None)]
mkp = lambda C, cw: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=cw, max_iter=5000))
chosen = Counter()

def nested(feats, mode, seed):
    gcol = "block_id" if mode == "block" else "shot_id"
    def fit_fn(tr, va):
        tr = tr.reset_index(drop=True)
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        best, best_s = None, -1
        for C, cw in GRID:
            oof = np.zeros(len(tr))
            for a, b in inner.split(tr, tr[Y], tr[gcol]):
                m = mkp(C, cw).fit(tr.loc[a, feats].values, tr.loc[a, Y].values)
                oof[b] = m.predict_proba(tr.loc[b, feats].values)[:, 1]
            sh = pd.DataFrame({"y": tr[Y], "s": oof, "g": tr.shot_id}).groupby("g").max()
            s = ap(sh.y, sh.s)
            if s > best_s + 1e-12: best, best_s = (C, cw), s
        chosen[(mode, best)] += 1
        return mkp(*best).fit(tr[feats].values, tr[Y].values).predict_proba(va[feats].values)[:, 1]
    return fit_fn

rows = []
for mold in ["cn7", "rg3"]:
    df, feats = load(mold)
    for mode in ["block", "shot"]:
        chosen.clear()
        b, bf = run_cv(df, feats, lambda: mkp(0.3, "balanced"), mode, per_fold=True)
        s, sf = run_cv(df, feats, None, mode, fit_fn=nested(feats, mode, 42), per_fold=True)
        rows.append({"mold": mold, "cv": mode, "variant": "baseline C=0.3 balanced", **{k: v[0] for k, v in b.items()}, "shot_ap_sd": b["shot_ap"][1]})
        rows.append({"mold": mold, "cv": mode, "variant": "E3 중첩 CV 튜닝", **{k: v[0] for k, v in s.items()}, "shot_ap_sd": s["shot_ap"][1], **fold_compare(bf, sf),
                     "chosen": "; ".join(f"C={k[1][0]},{k[1][1]}:{v}" for k, v in chosen.most_common())})
r = pd.DataFrame(rows); r.to_csv(os.path.join(OUT, "e3_results.csv"), index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 120)
print(r[["mold","cv","variant","shot_ap","shot_ap_sd","row_ap","shot_recall@2%","shot_recall@5%","shot_recall@10%","folds","alt_wins","base_wins","ties"]].round(3).to_string())
print(r[["mold","cv","chosen"]].dropna().to_string())
