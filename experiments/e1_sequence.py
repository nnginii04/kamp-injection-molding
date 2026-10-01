"""E1: 직전 샷 대비 변화량 피처. 라벨은 쓰지 않고 이전 샷의 '공정값'만 사용.
사전 등록한 변형 3개(결과 보기 전에 고정):
  E1a = 원변수 + 직전 1샷 대비 차이(Δ1)
  E1b = 원변수 + 직전 5샷 이동중앙값 대비 편차(dev5)   ← config.sequence_window=5
  E1c = 원변수 + Δ1 + dev5
채택 기준(레포 2-3 과 같음): block·shot 양쪽에서 불량이 있는 fold 의 과반을 이기고, block pooled shot_ap 가 오를 것."""
from harness import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from seqfeat import add_seq, mk
rows = []
for mold in ["cn7", "rg3"]:
    df, feats = load(mold)
    base = {m: run_cv(df, feats, mk, m, per_fold=True) for m in ("block", "shot")}
    for m in ("block", "shot"):
        rows.append({"mold": mold, "variant": f"baseline({MODEL})", "cv": m, **{k: v[0] for k, v in base[m][0].items()},
                     **{k+"_sd": v[1] for k, v in base[m][0].items()}})
    for name, kind in [("E1a Δ1", "d1"), ("E1b dev5", "dev5"), ("E1c Δ1+dev5", "both")]:
        d2, f2 = add_seq(df, feats, kind)
        for m in ("block", "shot"):
            s, fo = run_cv(d2, f2, mk, m, per_fold=True)
            cmp = fold_compare(base[m][1], fo)
            rows.append({"mold": mold, "variant": name, "cv": m, **{k: v[0] for k, v in s.items()},
                         **{k+"_sd": v[1] for k, v in s.items()}, **cmp})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "e1_results.csv"), index=False)
pd.set_option("display.width", 250)
print(res[["mold","variant","cv","shot_ap","shot_ap_sd","row_ap","row_auc","shot_recall@2%","shot_recall@5%","shot_recall@10%","folds","alt_wins","base_wins","ties"]].round(3).to_string())
