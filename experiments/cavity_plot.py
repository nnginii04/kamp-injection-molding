"""budget_curve_cavity.csv → 그림 (평가 세트별 작은 그래프, 같은 축). 마커 모양으로 규칙을 함께 구분."""
import os
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import harness as H

for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
R = pd.read_csv(os.path.join(H.OUT, "budget_curve_cavity.csv"))
SER = [("①", "① 모델 상위 k%", "#2a78d6", "o"), ("⑤", "⑤ 균등 샘플링", "#eb6834", "s"),
       ("④", "④ RH 우선 샘플링", "#1baf7a", "^"), ("④+③", "④ RH 우선 + ③ 추적", "#eda100", "D"),
       ("①2%+④+③", "① 2% + ④ + ③", "#e87ba4", "v")]
sets = list(dict.fromkeys(R["set"]))
fig, axes = plt.subplots(1, len(sets), figsize=(4.3 * len(sets), 4.6), sharey=True)
for ax, s in zip(axes, sets):
    g = R[R["set"] == s]
    ax.plot([0, .6], [0, .6], ls="--", lw=1, color="#9a9a96", label="무작위 검사")
    for fam, lab, col, mk in SER:
        h = g[g.family == fam].sort_values("inspect_rate")
        ax.plot(h.inspect_rate, h.recall, lw=2, color=col, marker=mk, ms=6, label=lab, markeredgecolor="#fcfcfb", markeredgewidth=1.2)
    ax.set_title(s, fontsize=10.5); ax.set_xlim(0, .6); ax.set_ylim(0, 1.03)
    ax.set_xlabel("검사 비율 (검사 부품 / 전체)", color="#52514e")
    ax.grid(color="#e8e7e2", lw=.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axes[0].set_ylabel("잡은 불량 부품 비율", color="#52514e")
axes[-1].legend(loc="lower right", fontsize=8, frameon=False)
fig.suptitle("검사 규칙 비교: RH 우선 검사(④) 추가", fontsize=13, x=0.01, ha="left")
fig.tight_layout(); fig.patch.set_facecolor("#fcfcfb")
out = os.path.join(H.OUT, "budget_curve_cavity.png"); fig.savefig(out, dpi=150, facecolor=fig.get_facecolor()); print(out)
