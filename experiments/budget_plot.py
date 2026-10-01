"""budget_curve.csv → 그림 (제품·평가 세트별 작은 그래프 3개, 같은 축)."""
import os, sys
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import harness as H

for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
R = pd.read_csv(os.path.join(H.OUT, "budget_curve.csv"))
SER = [("① 모델 상위 k%", dict(N=0, M=0), "#2a78d6", "o"),
       ("① + ③ 불량 캐비티 추적(3샷)", dict(N=0, M=3), "#eb6834", "s"),
       ("① + ② 재가동 후 5샷", dict(N=5, M=0), "#1baf7a", "^"),
       ("① + ② + ③", dict(N=5, M=3), "#eda100", "D")]
sets = list(dict.fromkeys(R["set"]))
fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.8), sharey=True)
for ax, s in zip(axes, sets):
    g = R[R["set"] == s]
    ax.plot([0, .35], [0, .35], ls="--", lw=1, color="#9a9a96", label="무작위 검사")
    for name, cond, col, mk in SER:
        h = g[(g.N == cond["N"]) & (g.M == cond["M"])].sort_values("k")
        ax.plot(h.inspect_rate, h.recall, lw=2, color=col, marker=mk, ms=6, label=name,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2)
    p = g[(g.k == .02) & (g.N == 0) & (g.M == 0)].iloc[0]
    ax.annotate(f"① 상위 2%: 검사 {p.inspect_rate:.1%} → 불량 {p.recall:.0%}", (p.inspect_rate, p.recall),
                xytext=(0.97, 0.04), textcoords="axes fraction", ha="right", fontsize=9, color="#52514e")
    ax.set_title(s, fontsize=11, color="#0b0b0b"); ax.set_xlim(0, .35); ax.set_ylim(0, 1.03)
    ax.set_xlabel("검사 비율 (검사한 부품 / 전체 부품)", color="#52514e")
    ax.grid(color="#e8e7e2", lw=.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axes[0].set_ylabel("잡은 불량 부품 비율 (재현율)", color="#52514e")
axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)
fig.suptitle("검사 규칙 조합별 Budget Curve", fontsize=13, x=0.01, ha="left")
fig.tight_layout(); fig.patch.set_facecolor("#fcfcfb")
out = os.path.join(H.OUT, "budget_curve.png"); fig.savefig(out, dpi=160, facecolor=fig.get_facecolor()); print(out)
