"""E6a 격리(hold) 운영안 재생. 조건은 docs/protocol_E6.md 에 실행 전 고정 (H5·H6 은 E6b 결과를 본 뒤 추가, 사후 추가 표기).

격리된 shot 은 LH·RH 두 부품을 나중에 검사한다 (용량 한도 없음). E2 의 고정 용량 재생(H0)과 같은 기간·같은 점수·같은 임계값.
- 미래 기간(주, 탐색적 날짜 검증): 대회 labeled 전체로 학습한 최종 로지스틱, 임계값 = 대회 기간 block OOF shot 점수 상위 2 %·5 %.
- 대회 기간(참고): pipeline_logreg.csv block 반복 0 OOF 점수 (같은 임계값).
- 정지 = 같은 평가 기간 labeled 기록의 TimeStamp 간격 ≥ 5분 (labeled 기록 기준 '기록 공백').
- H3 의 N = outputs/e6/hold_rule_cn7.csv (10/16 이전 설비 로그로 결정, e6_recovery.py 먼저 실행).

실행: python scripts/e6_hold_replay.py --public <원본 폴더 또는 zip>
결과: outputs/e6/hold_replay.csv (요약), hold_replay_events.csv (사건별), hold_replay_shots.csv (shot 별 격리 사유),
      future_stops_{cn7,rg3}.csv (미래 기간 정지별 공정값·경보·불량, 상위 2 % 임계값)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import _public  # noqa: E402
import e6_recovery as R6  # noqa: E402
import external_experiments as X  # noqa: E402
from src import features  # noqa: E402
from src.data import assign_shots, load_all  # noqa: E402
from src.models import make_model  # noqa: E402

warnings.filterwarnings("ignore")
G = 5.0
QUANTS = {"top2": 0.98, "top5": 0.95}
KS = [1, 3, 5]
MAIN_K = 3
FIXED_N = 5
REC_VARS = ["Filling_Time", "Injection_Time"]
PREV_N, REC_MAX = 100, 30


def shot_table(ts, side_rh, y, score, filling, injection) -> pd.DataFrame:
    d = pd.DataFrame(dict(t=pd.to_datetime(ts), side=side_rh, y=y, score=score,
                          Filling_Time=filling, Injection_Time=injection))
    g = d.groupby("t", sort=True)
    S = pd.DataFrame(dict(parts=g.size(), defects=g["y"].sum(), score=g["score"].max(),
                          Filling_Time=g["Filling_Time"].median(), Injection_Time=g["Injection_Time"].median(),
                          reasons=g["y"].apply(lambda s: int(s.sum())))).reset_index()
    S["gap_min"] = S["t"].diff().dt.total_seconds().div(60)
    return S


def bin_of(gap):
    for lo, hi, lab in zip(R6.BINS[:-1], R6.BINS[1:], R6.BIN_LABELS):
        if lo <= gap < hi:
            return lab
    return R6.BIN_LABELS[-1]


def rules(S: pd.DataFrame, thr: float, nrule: dict, first_gap: float | None, prev_tail: pd.DataFrame | None):
    n = len(S)
    alarm = (S["score"] >= thr).to_numpy()
    gap = S["gap_min"].to_numpy().copy()
    gap[0] = first_gap if first_gap is not None else np.inf
    stop = gap >= G
    H = {"H1_alarm": alarm.copy()}
    for k in KS:
        m = alarm.copy()
        for i in np.where(alarm)[0]:
            m[i:min(i + k + 1, n)] = True
        H[f"H2_alarm+{k}"] = m
    h3, h3f = np.zeros(n, bool), np.zeros(n, bool)
    for i in np.where(stop)[0]:
        N = nrule[bin_of(gap[i])]
        h3[i:min(i + N, n)] = True
        h3f[i:min(i + FIXED_N, n)] = True
    H["H3_stopN"] = h3
    H["H3_fixed5"] = h3f
    H["H4_alarm+3|stopN"] = H[f"H2_alarm+{MAIN_K}"] | h3
    # H5 회복 판정 (사후 추가): 직전 세션 마지막 100 shot 밴드, 연속 5 shot 안에 들어올 때까지 격리 (최대 30)
    h5 = np.zeros(n, bool)
    starts = list(np.where(stop)[0]) + [n]
    skipped = 0
    for a, b in zip(starts[:-1], starts[1:]):
        # 기준 = 정지 직전 100 shot (직전 세션이 짧으면 그 앞 세션까지 포함). 평가 기간 첫 정지는 이전 기간 마지막 100 shot.
        if a == 0:
            ref = prev_tail
        else:
            ref = S.iloc[max(0, a - PREV_N):a]
            if a < PREV_N and prev_tail is not None:
                ref = pd.concat([prev_tail.tail(PREV_N - a), ref])
        if ref is None or len(ref) < 20:
            skipped += 1
            continue
        ok = np.ones(b - a, bool)
        for v in REC_VARS:
            lo, hi = R6.band(ref[v].to_numpy(float))
            x = S[v].to_numpy(float)[a:b]
            ok &= (x >= lo) & (x <= hi)
        k_rec = min(REC_MAX, b - a)
        for j in range(b - a):
            if j + R6.RUN <= len(ok) and ok[j:j + R6.RUN].all():
                k_rec = j
                break
        h5[a:a + min(k_rec, REC_MAX)] = True
    H["H5_recovery*"] = h5
    H["H6_alarm+3|recovery*"] = H[f"H2_alarm+{MAIN_K}"] | h5
    H["ALL"] = np.ones(n, bool)
    return H, int(stop.sum()), skipped


def events(S: pd.DataFrame, gap: int = 3):
    idx = list(np.where(S["defects"].to_numpy() > 0)[0])
    ev, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > gap:
            ev.append(cur)
            cur = []
        cur.append(i)
    return ev + ([cur] if cur else [])


def stop_table(S: pd.DataFrame, H: dict, first_gap: float | None) -> pd.DataFrame:
    """정지(간격 ≥ 5분)마다: 첫 shot·첫 5 shot 충전/사출시간, 회복 판정 격리 shot 수, 첫 5 shot 경보·불량, 세션 불량."""
    gap = S["gap_min"].to_numpy().copy()
    gap[0] = first_gap if first_gap is not None else np.inf
    idx = list(np.where(gap >= G)[0]) + [len(S)]
    rows = []
    for a, b in zip(idx[:-1], idx[1:]):
        w = slice(a, min(a + 5, b))
        rows.append(dict(start=str(S["t"].iat[a]), gap_min=round(float(gap[a]), 1),
                         fill_first=round(float(S["Filling_Time"].iat[a]), 2),
                         fill_max5=round(float(S["Filling_Time"].iloc[w].max()), 2),
                         inj_max5=round(float(S["Injection_Time"].iloc[w].max()), 2),
                         h5_hold_shots=int(H["H5_recovery*"][a:b].sum()), alarms5=int(H["H1_alarm"][w].sum()),
                         defect_parts_first5=int(S["defects"].iloc[w].sum()),
                         defect_parts_session=int(S["defects"].iloc[a:b].sum()), session_shots=b - a))
    return pd.DataFrame(rows)


def summarize(S, H, period, p, thr_name, thr, n_stops, skipped):
    P, N = int(S["defects"].sum()), int(S["parts"].sum())
    EV = events(S)
    rows, ev_rows = [], []
    for name, m in H.items():
        held = int(S.loc[m, "parts"].sum())
        tp = int(S.loc[m, "defects"].sum())
        rows.append(dict(period=period, product=p, threshold=thr_name, thr=round(thr, 5), rule=name,
                         parts=N, P=P, held_parts=held, held_rate=round(held / N, 4), TP=tp, FN=P - tp,
                         recall=round(tp / P, 3) if P else np.nan, good_parts_held=held - tp,
                         stops=n_stops, h5_sessions_skipped=skipped, uses_external=True))
        for k, e in enumerate(EV):
            ev_rows.append(dict(period=period, product=p, threshold=thr_name, rule=name, event=k,
                                start=str(S["t"].iat[e[0]]), defect_shots=len(e),
                                defect_parts=int(S["defects"].iloc[e].sum()),
                                found_parts=int(S.loc[[i for i in e if m[i]], "defects"].sum())))
    return rows, ev_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=None)
    a = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    src = _public.public_path(cfg, a.public)
    pub = _public.read_csv(src, "moldset_labeled.csv")
    ld = _public.read_csv(src, "labeled_data.csv")
    out = ROOT / cfg["output_dir"] / "e6"
    rule_tab = pd.read_csv(out / "hold_rule_cn7.csv")
    nrule = dict(zip(rule_tab["gap_bin"], rule_tab["N_hold"]))
    pred = pd.read_csv(ROOT / cfg["output_dir"] / "predictions" / "pipeline_logreg.csv")
    data = load_all(cfg, ROOT / cfg["data_dir"])
    raw = data["feature_cols"]
    base, _ = features.select_base_features(data, raw, True)
    summ, evs, shots_out = [], [], []
    for p in data["labeled"]:
        lab = assign_shots(data["labeled"][p], raw, p).reset_index(drop=True)
        L = X.link_side(lab, pub, raw)
        mu, sd, _ = X.zscale_params(L, raw)
        oof = pred[pred["product"].eq(p) & pred["cv_mode"].eq("block") & pred["repeat"].eq(0)]
        shot_oof = oof.groupby("shot_id")["y_score"].max()
        # 대회 기간 (OOF)
        sc = L["row_id"].map(oof.set_index("row_id")["y_score"])
        Sc = shot_table(L["TimeStamp"], L["side_rh"], L["y"], sc.to_numpy(),
                        L["raw_Filling_Time"].astype(float), L["raw_Injection_Time"].astype(float))
        # 미래 기간
        F = X.future_rows(ld, L, p, raw, mu, sd)
        fs = make_model("logreg", cfg["seed"]).fit(L[base], L["y"].to_numpy()).predict_proba(F[base])[:, 1]
        Sf = shot_table(F["TimeStamp"], F["side_rh"], F["y"], fs,
                        F["Filling_Time"] * sd["Filling_Time"] + mu["Filling_Time"],
                        F["Injection_Time"] * sd["Injection_Time"] + mu["Injection_Time"])
        first_gap_f = (Sf["t"].iat[0] - Sc["t"].iat[-1]).total_seconds() / 60
        for qn, q in QUANTS.items():
            thr = float(shot_oof.quantile(q))
            for period, S, fg, tail in [("future", Sf, first_gap_f, Sc.tail(PREV_N)), ("competition", Sc, None, None)]:
                H, ns, sk = rules(S, thr, nrule, fg, tail)
                r, e = summarize(S, H, period, p, qn, thr, ns, sk)
                summ += r
                evs += e
                if qn == "top2" and period == "future":
                    stop_table(S, H, fg).to_csv(out / f"future_stops_{p}.csv", index=False, encoding="utf-8-sig")
                if qn == "top2":
                    T = S[["t", "parts", "defects", "score", "gap_min", "Filling_Time", "Injection_Time"]].copy()
                    for name in ("H1_alarm", f"H2_alarm+{MAIN_K}", "H3_stopN", "H5_recovery*"):
                        T[name] = H[name].astype(int)
                    shots_out.append(T.assign(period=period, product=p))
        print(f"{p} 완료", flush=True)
    Rr = pd.DataFrame(summ)
    Rr.to_csv(out / "hold_replay.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(evs).to_csv(out / "hold_replay_events.csv", index=False, encoding="utf-8-sig")
    pd.concat(shots_out).to_csv(out / "hold_replay_shots.csv", index=False, encoding="utf-8-sig")
    # H0 (E2 인용)
    e2 = ROOT / cfg["output_dir"] / "inspection" / "delay_replay.csv"
    if e2.exists():
        D = pd.read_csv(e2)
        D = D[D.model.eq("logreg") & D.rule.eq("model") & D.delay.eq(0)]
        print("H0 (E2 고정 용량, 지연 0, logreg):\n",
              D[["product", "threshold", "cap_per10", "inspect_rate", "TP", "P", "recall"]].to_string(index=False))
    pd.set_option("display.width", 250)
    print(Rr[Rr.threshold.eq("top2")][["period", "product", "rule", "held_parts", "held_rate", "TP", "P", "recall",
                                       "good_parts_held", "stops", "h5_sessions_skipped"]].to_string(index=False))


if __name__ == "__main__":
    main()
