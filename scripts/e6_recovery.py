"""E6b 정지 후 회복 곡선 + E6c 작업지시 단위 상관. 조건은 docs/protocol_E6.md 에 실행 전 고정.

- 설비 가동 로그(unlabeled_data.csv, 공개 원본 = 외부 데이터)의 생산 행(Average_Screw_RPM > 0)만 사용, 라벨 미사용.
- 정지 = 같은 설비·제품군에서 직전 생산 shot 과의 간격 ≥ 5분. 세션마다 31~130번째 shot 으로 기준 밴드를 잡고,
  1~30번째 shot 중 연속 5 shot 이 밴드 안에 들어오는 첫 순번을 회복 shot 으로 기록.
- 격리 규칙 N(정지 길이 구간) = 회복 shot 90 백분위 − 1, TimeStamp < 2020-10-16 (대회 labeled 시작 전) 데이터로만 계산.

실행: python scripts/e6_recovery.py --public <원본 폴더 또는 zip> [--perms 2000]
결과: outputs/e6/ (freeze 된 outputs 의 다른 폴더는 건드리지 않음)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import _public  # noqa: E402

G = 5.0
BINS = [5, 10, 30, 120, 720, np.inf]
BIN_LABELS = ["5-10분", "10-30분", "30-120분", "2-12시간", "12시간+"]
MIN_SESSION, BASE_FROM, BASE_TO, HORIZON, RUN = 60, 31, 130, 30, 5
MAIN_VARS = ["Filling_Time", "Injection_Time"]
AUX_VARS = ["Max_Injection_Pressure", "Cushion_Position", "Mold_Temperature_3"]
VARS = MAIN_VARS + AUX_VARS
CUTOFF = pd.Timestamp("2020-10-16")
MAIN_EQUIP, MAIN_FAM = "650톤-우진2호기", "CN7"
MIN_STOPS = 20


def shots_from_rows(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["Average_Screw_RPM"] > 0].copy()
    d["fam"] = d["PART_NAME"].astype(str).str.split().str[0]
    d["t"] = pd.to_datetime(d["TimeStamp"])
    S = d.groupby(["EQUIP_NAME", "fam", "t"], sort=False)[VARS].median().reset_index()
    return S.sort_values(["EQUIP_NAME", "fam", "t"]).reset_index(drop=True)


def recovery_shot(x: np.ndarray, lo: float, hi: float) -> int:
    ok = (x[:HORIZON + RUN - 1] >= lo) & (x[:HORIZON + RUN - 1] <= hi)
    for k in range(min(HORIZON, len(ok) - RUN + 1)):
        if ok[k:k + RUN].all():
            return k + 1
    return HORIZON + 1  # 미회복


def band(base: np.ndarray) -> tuple[float, float]:
    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med)))
    w = max(3 * 1.4826 * mad, 0.02 * abs(med), 0.01)
    return med - w, med + w


def sessions(S: pd.DataFrame, min_len: int = MIN_SESSION) -> tuple[pd.DataFrame, dict]:
    rows, curves = [], {}
    for (eq, fam), g in S.groupby(["EQUIP_NAME", "fam"], sort=False):
        g = g.reset_index(drop=True)
        gap = g["t"].diff().dt.total_seconds().div(60)
        new = gap >= G
        sid = new.cumsum()
        for s, h in g.groupby(sid, sort=False):
            if s == 0 or len(h) < min_len:  # 첫 세션은 정지 길이를 모름
                continue
            gm = float(gap.loc[h.index[0]])
            rec = dict(equip=eq, fam=fam, start=h["t"].iat[0], gap_min=round(gm, 2), n_shots=len(h))
            dev = {}
            for v in VARS:
                x = h[v].to_numpy(float)
                base = x[BASE_FROM - 1:BASE_TO]
                if v == "Mold_Temperature_3" and np.median(base) == 0:
                    rec[f"r_{v}"] = np.nan
                    continue
                lo, hi = band(base)
                rec[f"r_{v}"] = recovery_shot(x, lo, hi)
                mid, half = (lo + hi) / 2, (hi - lo) / 2
                dev[v] = (x[:HORIZON] - mid) / half  # 밴드 반폭 단위 이탈
            rec["r_main"] = max(rec["r_Filling_Time"], rec["r_Injection_Time"])
            rows.append(rec)
            curves[(eq, fam, rec["start"])] = dev
    R = pd.DataFrame(rows)
    R["gap_bin"] = pd.cut(R["gap_min"], BINS, right=False, labels=BIN_LABELS)
    return R, curves


def bin_table(R: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    def f(x):
        r = x["r_main"]
        return pd.Series(dict(stops=len(x), r_median=r.median(), r_p75=r.quantile(.75), r_p90=r.quantile(.9),
                              within5=(r <= 5).mean().round(3), not_recovered=(r > HORIZON).mean().round(3),
                              **{f"p90_{v}": x[f"r_{v}"].quantile(.9) for v in AUX_VARS}))
    return R.groupby(keys + ["gap_bin"], observed=True).apply(f).reset_index()


def hold_rule(R: pd.DataFrame) -> pd.DataFrame:
    m = R[R.equip.eq(MAIN_EQUIP) & R.fam.eq(MAIN_FAM) & (R.start < CUTOFF)]
    out, nxt = [], None
    for b in reversed(BIN_LABELS):  # 긴 구간부터: 표본 부족이면 바로 긴 쪽 값 사용
        x = m[m.gap_bin.eq(b)]["r_main"]
        if len(x) >= MIN_STOPS:
            n = int(max(np.ceil(x.quantile(.9)) - 1, 0))
            src = "자체"
        else:
            n = nxt if nxt is not None else int(max(np.ceil(m["r_main"].quantile(.9)) - 1, 0))
            src = "긴 구간 값" if nxt is not None else "전체 값"
        out.append(dict(gap_bin=b, stops=len(x), N_hold=n, source=src))
        nxt = n
    return pd.DataFrame(out[::-1])


def plan_corr(d: pd.DataFrame, n_perm: int, rng) -> pd.DataFrame:
    d = d[d["Average_Screw_RPM"] > 0].copy()
    d["fam"] = d["PART_NAME"].astype(str).str.split().str[0]
    d["t"] = pd.to_datetime(d["TimeStamp"])
    rows = []
    for (eq, fam, pdate, ser), g in d.groupby(["EQUIP_NAME", "fam", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL"]):
        if g["ERR_FACT_QTY"].nunique() != 1:
            continue
        ts = np.sort(g["t"].unique())
        if len(ts) < 200:
            continue
        gaps = np.diff(ts).astype("timedelta64[s]").astype(float) / 60
        rows.append(dict(equip=eq, fam=fam, plan=str(pdate), serial=ser, parts=len(g), shots=len(ts),
                         err=float(g["ERR_FACT_QTY"].iat[0]), stops=int((gaps >= G).sum())))
    P = pd.DataFrame(rows)
    P["err_rate"] = P["err"] / P["parts"]
    P["stops_per100"] = P["stops"] / P["shots"] * 100

    def spear(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    res = []
    for key, g in [("전체", P)] + [(f"{e}|{f}", h) for (e, f), h in P.groupby(["equip", "fam"]) if len(h) >= 15]:
        rho = spear(g["stops_per100"].to_numpy(), g["err_rate"].to_numpy())
        null = np.array([spear(rng.permutation(g["stops_per100"].to_numpy()), g["err_rate"].to_numpy())
                         for _ in range(n_perm)])
        res.append(dict(group=key, plans=len(g), spearman=round(rho, 3),
                        p_two_sided=round(float((np.abs(null) >= abs(rho)).mean()), 4)))
    return P, pd.DataFrame(res)


def plot(R, curves, rule, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for name in ("AppleGothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Serif CJK KR", "NanumGothic"):
        if any(name == f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    m = R[R.equip.eq(MAIN_EQUIP) & R.fam.eq(MAIN_FAM)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
    pos = np.arange(1, HORIZON + 1)
    for b, c in zip(BIN_LABELS, colors):
        keys = [(r.equip, r.fam, r.start) for r in m[m.gap_bin.eq(b)].itertuples()]
        if len(keys) < 5:
            continue
        D = np.array([np.max(np.abs([curves[k][v] for v in MAIN_VARS]), axis=0) for k in keys
                      if len(curves[k]["Filling_Time"]) == HORIZON])
        ax[0].plot(pos, np.median(D, 0), color=c, lw=2, label=f"{b} (n={len(D)})")
    ax[0].axhline(1, color="#c0392b", ls="--", lw=1)
    ax[0].text(HORIZON, 1.05, "정상 밴드 경계", color="#c0392b", ha="right", va="bottom")
    ax[0].set(xlabel="정지 후 shot 순번", ylabel="충전·사출시간 이탈 (밴드 반폭 단위, 중앙값)",
              title="CN7 · 정지 길이별 회복 곡선 (2020-03~10 설비 로그)")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].bar(rule["gap_bin"].astype(str), rule["N_hold"], color="#3182bd")
    for i, r in enumerate(rule.itertuples()):
        ax[1].text(i, r.N_hold + 0.2, f"{r.N_hold}\n(정지 {r.stops})", ha="center", va="bottom", fontsize=8)
    ax[1].set(ylabel="격리할 shot 수 N", title="정지 길이별 격리 규칙 (10/16 이전 로그로 결정)")
    ax[1].set_ylim(0, max(rule["N_hold"].max() + 3, 5))
    fig.tight_layout()
    fig.savefig(out / "recovery_curve_cn7.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=None)
    ap.add_argument("--perms", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    src = _public.public_path(cfg, a.public)
    out = ROOT / cfg["output_dir"] / "e6"
    out.mkdir(parents=True, exist_ok=True)
    d = _public.read_csv(src, "unlabeled_data.csv")
    S = shots_from_rows(d)
    R, curves = sessions(S)
    R.to_csv(out / "recovery_sessions.csv", index=False, encoding="utf-8-sig")
    T = bin_table(R, ["equip", "fam"])
    T.to_csv(out / "recovery_by_bin.csv", index=False, encoding="utf-8-sig")
    Tpre = bin_table(R[R.start < CUTOFF], ["equip", "fam"])
    Tpre.to_csv(out / "recovery_by_bin_pre1016.csv", index=False, encoding="utf-8-sig")
    rule = hold_rule(R)
    rule.to_csv(out / "hold_rule_cn7.csv", index=False, encoding="utf-8-sig")
    P, C = plan_corr(d, a.perms, np.random.default_rng(0))
    P.to_csv(out / "plan_err_vs_stops.csv", index=False, encoding="utf-8-sig")
    C.to_csv(out / "plan_corr.csv", index=False, encoding="utf-8-sig")
    plot(R, curves, rule, out)
    pd.set_option("display.width", 220)
    print(Tpre[Tpre.equip.eq(MAIN_EQUIP) & Tpre.fam.eq(MAIN_FAM)].to_string(index=False))
    print(rule.to_string(index=False))
    print("세션 수(설비·제품군별):\n", R.groupby(["equip", "fam"]).size().to_string())
    print(C.to_string(index=False))
    (out / "run_log_recovery.txt").write_text(f"e6_recovery.py {time.time() - t0:.1f}s, sessions={len(R)}, plans={len(P)}\n",
                                              encoding="utf-8")
    print(f"완료 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
