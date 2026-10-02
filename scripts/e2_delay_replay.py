"""E2 — 검사 결과 지연을 반영한 시간순 재생. 조건은 docs/protocol_E1_E3.md 에 실행 전 고정.

재생 데이터: 대회 이후 기간 (CN7 10/27~11/03, RG3 11/04~11/06; 공개 원본 labeled_data.csv = 외부 데이터).
이 기간의 모델 점수 순위는 이미 확인했으므로 (X3) "탐색적 날짜 검증"이다. 독립 최종 테스트가 아니다.

각 shot t (생산 시각 순) 에서:
  1) 지금까지 도착한 검사 결과만 본다 (shot s 에서 검사한 결과는 d shot 이 더 생산된 뒤, 즉 shot s+d+1 을
     결정할 때부터 보임. 검사 안 한 제품의 라벨은 끝까지 모름)
  2) 현재 shot 의 점수(고정 임계값과 비교)와 도착한 결과로 검사 여부를 정한다
  3) 검사하면 결과 도착 시점을 예약한다
  4) 도착한 결과가 불량이면 그 시점의 shot 부터 3 shot 을 추적 검사 요청 (생산 시점의 정답은 쓰지 않음)
- 임계값: 대회 기간 block CV OOF (반복 0) shot 점수의 상위 2 % · 5 % 분위수. 재생 기간 점수로 다시 정하지 않는다.
- 용량: 최근 10 shot 동안 검사한 부품이 c 개 이상이면 새 검사 불가 → 미검사로 기록 (대기열 없음).
- 규칙: model / model_track / model_rh / model_rh_track
  · model 계열은 shot 의 두 부품을 함께 검사 (용량이 2 미만이면 그 shot 미검사)
  · rh 계열은 RH 부품 먼저, 용량이 남으면 LH (side 는 외부 메타정보)
  · 추적 요청이 모델 요청보다 우선
- 모델: logreg_spc (최종 후보) + 비교용 logreg, spc_maxz (E1 과 같은 세 방법).

실행: python scripts/e2_delay_replay.py --public <원본 폴더 또는 zip>
결과: outputs/inspection/delay_replay.csv (요약), delay_replay_events.csv (사건별), delay_replay_log.csv (결정 기록)
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _public  # noqa: E402
import external_experiments as X  # noqa: E402
from src import features  # noqa: E402
from src.data import assign_shots, load_all  # noqa: E402
from src.models import make_model  # noqa: E402

warnings.filterwarnings("ignore")
DELAYS = [0, 1, 3, 5]
CAPS = [2, 4]
WINDOW = 10
TRACK = 3
QUANTS = {"top2": 0.98, "top5": 0.95}
MODELS = ["logreg_spc", "logreg", "spc_maxz"]
RULES = ["model", "model_track", "model_rh", "model_rh_track"]


def shots_table(F: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    d = F.assign(score=score)
    d["ts"] = pd.to_datetime(d["TimeStamp"])
    g = d.groupby("shot_id", sort=False)
    S = pd.DataFrame({"ts": g["ts"].min(), "score": g["score"].max()}).sort_values("ts").reset_index()
    parts = {k: v.sort_values("side_rh", ascending=False)[["side_rh", "y"]].to_numpy() for k, v in g}
    S["parts"] = S["shot_id"].map(parts)  # 각 원소: [[side_rh, y], ...], RH 먼저
    S["score_rank"] = S["score"].rank(ascending=False, method="min").astype(int)  # 1 = 가장 위험
    S["score_pct"] = (S["score_rank"] / len(S)).round(4)  # 상위 몇 % 인지
    return S


def replay(S: pd.DataFrame, thr: float, delay: int, cap: int, rule: str):
    track_on, rh_first = "track" in rule, "_rh" in rule
    arrivals: dict[int, list[int]] = {}  # 도착 시점 → 불량 여부 목록
    track_from, track_until = 0, -1
    recent: list[tuple[int, int]] = []  # (shot, 검사 부품 수)
    log, inspected, tp, overflow = [], 0, 0, 0
    confirmed_at: dict[int, int] = {}  # 불량 shot → 확인 시점
    for t, row in enumerate(S.itertuples(index=False)):
        # 1) 도착한 결과 확인 → 불량이면 추적 시작
        for ydef in arrivals.pop(t, []):
            if ydef and track_on:  # 확인된 시점 다음 shot 부터 TRACK 개
                track_from, track_until = t, max(track_until, t + TRACK - 1)
        used = sum(n for s, n in recent if s > t - WINDOW)
        room = cap - used
        want_track = track_on and track_from <= t <= track_until
        want_model = row.score >= thr
        parts = row.parts
        chosen = []
        reason = "tracking" if want_track else ("model" if want_model else "")
        if want_track or want_model:
            if rh_first:
                chosen = [i for i in range(len(parts))][:max(room, 0)]
            else:
                chosen = list(range(len(parts))) if room >= len(parts) else []
            if len(chosen) < len(parts):
                overflow += 1
        n = len(chosen)
        if n:
            recent.append((t, n))
            inspected += n
            ys = [int(parts[i][1]) for i in chosen]
            tp += sum(ys)
            # 결과는 d shot 이 더 생산된 뒤 도착 → shot t+d+1 을 결정할 때부터 볼 수 있음 (d=0: 바로 다음 shot)
            arrivals.setdefault(t + delay + 1, []).append(int(any(ys)))
            if any(ys):
                confirmed_at[t] = t + delay + 1
        log.append(dict(t=t, shot_id=row.shot_id, ts=row.ts, score=round(float(row.score), 5), reason=reason,
                        inspected_parts=n, defect_parts=int(sum(p[1] for p in parts)),
                        found=int(sum(parts[i][1] for i in chosen)) if n else 0,
                        score_rank=int(row.score_rank), score_pct=float(row.score_pct), thr=round(float(thr), 5),
                        want_model=int(want_model), want_track=int(want_track), room=int(room),
                        inspected_sides="+".join("RH" if int(parts[i][0]) else "LH" for i in chosen),
                        defect_sides="+".join("RH" if int(q[0]) else "LH" for q in parts if int(q[1]))))
    return log, inspected, tp, overflow, confirmed_at


def events(S: pd.DataFrame, gap: int = 3) -> list[list[int]]:
    """불량 shot 을 사건으로 묶기: 불량 shot 사이 간격이 gap shot 이하이면 같은 사건."""
    idx = [i for i, p in enumerate(S["parts"]) if any(int(x[1]) for x in p)]
    ev, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > gap:
            ev.append(cur)
            cur = []
        cur.append(i)
    if cur:
        ev.append(cur)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=None)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    src = _public.public_path(cfg, args.public)
    pub = _public.read_csv(src, "moldset_labeled.csv")
    ld = _public.read_csv(src, "labeled_data.csv")
    out = ROOT / cfg["output_dir"] / "inspection"
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = ROOT / cfg["output_dir"] / "predictions"

    data = load_all(cfg, ROOT / cfg["data_dir"])
    raw = data["feature_cols"]
    base, _ = features.select_base_features(data, raw, True)
    summ, ev_rows, logs, logs_all = [], [], [], []
    for p in data["labeled"]:
        lab = assign_shots(data["labeled"][p], raw, p).reset_index(drop=True)
        L = X.link_side(lab, pub, raw)
        mu, sd, _ = X.zscale_params(L, raw)
        F = X.future_rows(ld, L, p, raw, mu, sd)
        if not len(F):
            continue
        for m in MODELS:
            oof = pd.read_csv(pred_dir / f"pipeline_{m}.csv")
            oof = oof[oof["product"].eq(p) & oof["cv_mode"].eq("block") & oof["repeat"].eq(0)]
            shot_oof = oof.groupby("shot_id")["y_score"].max()
            score = make_model(m, cfg["seed"]).fit(L[base], L["y"].to_numpy()).predict_proba(F[base])[:, 1]
            S = shots_table(F, score)
            EV = events(S)
            N, P = int(sum(len(x) for x in S["parts"])), int(F["y"].sum())
            for qn, q in QUANTS.items():
                thr = float(shot_oof.quantile(q))
                flag_rate = float((S["score"] >= thr).mean())
                for d in DELAYS:
                    for c in CAPS:
                        for rule in RULES:
                            log, ins, tp, ovf, conf = replay(S, thr, d, c, rule)
                            summ.append(dict(product=p, model=m, threshold=qn, thr=round(thr, 5),
                                             flagged_shot_rate=round(flag_rate, 4), delay=d, cap_per10=c, rule=rule,
                                             uses_external=True, N=N, P=P, defect_shots=sum(len(e) for e in EV),
                                             events=len(EV), inspect_rate=round(ins / N, 4), TP=tp, FN=P - tp,
                                             recall=round(tp / P, 4) if P else np.nan, overflow_requests=ovf))
                            for k, e in enumerate(EV):
                                hit = [conf[i] for i in e if i in conf]
                                first_conf = min(hit) if hit else None
                                ev_rows.append(dict(product=p, model=m, threshold=qn, delay=d, cap_per10=c, rule=rule,
                                                    event=k, start_shot=e[0], end_shot=e[-1],
                                                    start_ts=str(S["ts"].iat[e[0]]), defect_shots=len(e),
                                                    defect_parts=int(sum(sum(int(x[1]) for x in S["parts"].iat[i]) for i in e)),
                                                    first_confirm_shot=first_conf,
                                                    shots_to_confirm=(first_conf - e[0]) if first_conf is not None else None,
                                                    defect_parts_found=int(sum(r["found"] for r in log if e[0] <= r["t"] <= e[-1]))))
                            logs_all += [dict(product=p, model=m, threshold=qn, delay=d, cap_per10=c, rule=rule, **r)
                                         for r in log]
                            if m == "logreg_spc" and qn == "top2" and c == 2:
                                logs += [dict(product=p, model=m, threshold=qn, delay=d, cap_per10=c, rule=rule, **r)
                                         for r in log]
        print(f"{p} 완료", flush=True)
    R = pd.DataFrame(summ)
    R.to_csv(out / "delay_replay.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ev_rows).to_csv(out / "delay_replay_events.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(logs).to_csv(out / "delay_replay_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(logs_all).to_csv(out / "delay_replay_log_full.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    show = R[R.cap_per10.eq(2) & R.threshold.eq("top2")]
    print(show.pivot_table(index=["product", "model", "rule"], columns="delay", values=["recall", "inspect_rate"]).round(3).to_string())
    print(R.groupby(["product", "model", "threshold"])[["thr", "flagged_shot_rate"]].first().to_string())


if __name__ == "__main__":
    main()
