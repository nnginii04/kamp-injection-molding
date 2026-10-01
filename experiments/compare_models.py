"""두 모델의 재실행 결과 비교표 → outputs/experiments/compare_<A>_vs_<B>.md
사용: python experiments/compare_models.py [A=logreg] [B=logreg_spc]"""
import os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "outputs", "experiments")
A = sys.argv[1] if len(sys.argv) > 1 else "logreg"
B = sys.argv[2] if len(sys.argv) > 2 else "logreg_spc"


def md(d):
    """tabulate 없이 마크다운 표."""
    d = d.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    rows = [list(map(str, d.columns))] + d.astype(str).values.tolist()
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def rd(m, f):
    p = os.path.join(BASE, m, f)
    return pd.read_csv(p) if os.path.exists(p) else None


def side(f, keys, cols):
    a, b = rd(A, f), rd(B, f)
    if a is None or b is None:
        return f"_{f}: 한쪽 결과 없음_\n"
    m = a[keys + cols].merge(b[keys + cols], on=keys, suffixes=(f" [{A}]", f" [{B}]"))
    order = keys + [c + s for c in cols for s in (f" [{A}]", f" [{B}]")]
    return m[order].pipe(md) + "\n"


out = [f"# 실험 재실행 비교: {A} vs {B}\n",
       "같은 코드·같은 분할(fold_id.csv)·같은 데이터에서 모델만 바꿈. 숫자는 각 결과 csv 에서 그대로 옮김.\n"]
c = rd(B, "check_pipeline_match.csv")
if c is not None:
    out += ["## 0. 점검: 실험 폴더 모델 = run_all.py 모델?\n", c.pipe(md) + "\n"]
# E1: variant 이름의 baseline 표기가 모델마다 달라서 통일
for f, title in [("e1_results.csv", "1. 기존 E1 (직전 샷 변화량 피처) — 교차검증"), ("e4_results.csv", "2. 기존 E4 (샷 단위 학습) — 교차검증")]:
    a, b = rd(A, f), rd(B, f)
    if a is None or b is None:
        out.append(f"## {title}\n_한쪽 결과 없음_\n"); continue
    for d in (a, b):
        d["variant"] = d["variant"].str.replace(r"^baseline\(.*\)$", "baseline", regex=True)
    cols = [c for c in ["shot_ap", "shot_recall@10%", "alt_wins", "base_wins", "folds"] if c in a.columns and c in b.columns]
    m = a[["mold", "variant", "cv"] + cols].merge(b[["mold", "variant", "cv"] + cols], on=["mold", "variant", "cv"], suffixes=(f" [{A}]", f" [{B}]"))
    out.append(f"## {title}\n" + m.pipe(md) + "\n")
out += ["## 3. D 시리즈 (가동 초기 D1·체류시간 D2 피처) — 교차검증\n",
        side("d_results_cv.csv", ["mold", "features", "cv"], ["shot_ap", "shot_recall@10%"]),
        "## 4. 날짜 홀드아웃 (학습에 안 쓴 생산일)\n",
        side("d_results_dayholdout.csv", ["mold", "test_day", "model"], ["shot_ap", "random_ap", "defect_ranks"]),
        "## 5. RG3 E1c 날짜 홀드아웃\n",
        side("d_e1c_dayholdout.csv", ["test_day", "features"], ["shot_ap", "random_ap"])]
a, b = rd(A, "budget_curve.csv"), rd(B, "budget_curve.csv")
if a is not None and b is not None:
    sel = lambda d: d[((d.N == 0) & (d.M == 0) & d.k.isin([.02, .05, .10])) | ((d.k == .02) & (d.N == 0) & (d.M == 3)) | ((d.k == .02) & (d.N == 5) & (d.M == 3))]
    m = sel(a)[["set", "k", "N", "M", "inspect_rate", "recall"]].merge(sel(b)[["set", "k", "N", "M", "inspect_rate", "recall"]],
                                                                      on=["set", "k", "N", "M"], suffixes=(f" [{A}]", f" [{B}]"))
    out += ["## 6. Budget Curve (부품 단위 검사율·재현율; k=모델 상위, N=재가동 후 샷, M=같은 캐비티 추적 샷)\n", m.pipe(md) + "\n"]
a, b = rd(A, "budget_curve_cavity.csv"), rd(B, "budget_curve_cavity.csv")
if a is not None and b is not None:
    keep = ["① 모델 상위 2%", "① 모델 상위 10%", "① 모델 상위 2% + ③", "① 2% + ④ RH 20% + ③", "⑤ 균등 10%", "④ RH 20%"]
    m = a[a.rule.isin(keep)][["set", "rule", "inspect_rate", "recall"]].merge(b[b.rule.isin(keep)][["set", "rule", "inspect_rate", "recall"]],
                                                                              on=["set", "rule"], suffixes=(f" [{A}]", f" [{B}]"))
    out += ["## 7. 캐비티 규칙 (⑤ 균등·④ RH 샘플링은 모델과 무관 → 두 열이 같아야 정상)\n", m.pipe(md) + "\n"]
p = os.path.join(BASE, f"compare_{A}_vs_{B}.md")
open(p, "w", encoding="utf-8").write("\n".join(x for x in out if x))
print(p)
