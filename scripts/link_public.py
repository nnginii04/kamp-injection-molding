"""KAMP 공개 원본(labeled_data.csv)과 대회 labeled 파일을 연결해 메타데이터를 복원한다.
복원 항목: 캐비티(LH/RH), 생산 시각, 불량 사유(Reason), 원 단위 공정값, 표준화 기준(평균·표준편차)

    python scripts/link_public.py --enable                 # config.external.public_path(zip/폴더) 사용
    python scripts/link_public.py --enable --public 경로   # zip, 압축 푼 폴더, labeled_data.csv 중 하나
    python scripts/link_public.py --synthetic              # run_all.py --synthetic 이 만든 가짜 데이터로 확인

기본 꺼짐: config.json external.enabled=false 이면 --enable 없이는 실행하지 않는다. 원본은 읽기만 한다.

출력 (outputs/link/ 또는 outputs/synthetic/link/)
    linked_labeled.csv  대회 labeled 각 행 ↔ 원본 행
    scale_params.json   제품별 표준화 기준 → z값을 원 단위로 되돌릴 때 사용
    link_report.md      매칭 결과, 캐비티·시간·사유별 불량, 원 단위 비교, 오류구간 원 단위 환산

⚠ 대회 규정의 외부(공개) 데이터 사용 허용 범위를 먼저 확인하세요. 이 스크립트의 기본 용도는 '해석'이며,
  복원한 정보를 모델 입력으로 쓰려면 별도 실험으로 분리하고 보고서에 출처와 방법을 명시하세요.

원리: 대회 파일은 원본 일부를 제품별로 표준화한 것으로 보인다. 원본에서 같은 행 수의 후보 구간을 잡아
      그 구간의 평균·표준편차로 대회 z값을 원 단위로 되돌렸을 때 원본 값과 정확히 일치하면 같은 데이터다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import _public  # scripts/_public.py (같은 폴더)

from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import expected_topk  # noqa: E402
from src.plots import plt, savefig  # noqa: E402

DEFAULT_TOP = ["Max_Injection_Speed", "Clamp_Close_Time", "Cushion_Position", "Hopper_Temperature",
               "Max_Switch_Over_Pressure", "Max_Injection_Pressure"]


def read_csv_any(path):
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path} 인코딩을 읽을 수 없습니다.")


def load_public(src: Path, feats: list[str]) -> pd.DataFrame:
    pub = _public.read_csv(src, "labeled_data.csv")
    need = {"_id", "TimeStamp", "PART_NAME", "PassOrFail", *feats}
    miss = need - set(pub.columns)
    if miss:
        raise ValueError(f"원본에 없는 열: {sorted(miss)}")
    n0 = len(pub)
    pub = pub.drop_duplicates().copy()
    pub["time"] = pd.to_datetime(pub["TimeStamp"])
    pub = pub.sort_values(["time", "_id"], kind="stable").reset_index(drop=True)
    pub.attrs["dropped_duplicates"] = n0 - len(pub)
    return pub


def _scale(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def try_match(comp: pd.DataFrame, cand: pd.DataFrame, feats: list[str]):
    """대회 z값을 후보 구간 기준으로 원 단위로 되돌려 원본과 비교. 성공 시 (방법, 대응 인덱스, mu, sd)."""
    X = cand[feats].to_numpy(float)
    mu, sd = _scale(X)
    C = comp[feats].to_numpy(float) * sd + mu
    kp, kc = np.round(X, 3), np.round(C, 3)
    if np.allclose(kp, kc, atol=2e-3):
        return "order", np.arange(len(comp)), mu, sd
    tp, tc = [tuple(r) for r in kp], [tuple(r) for r in kc]
    if Counter(tp) != Counter(tc):
        return None
    # 같은 공정값(짝) 안에서는 라벨로 대응: 대회 불량행 ↔ 원본 N행
    pub_groups, comp_groups = {}, {}
    for i, k in enumerate(tp):
        pub_groups.setdefault(k, []).append(i)
    for i, k in enumerate(tc):
        comp_groups.setdefault(k, []).append(i)
    mapping = np.empty(len(comp), dtype=int)
    yc = comp["y"].to_numpy()
    for k, comp_idx in comp_groups.items():
        pi = pub_groups[k]
        bad_p = [i for i in pi if cand.iloc[i]["PassOrFail"] == "N"]
        good_p = [i for i in pi if cand.iloc[i]["PassOrFail"] != "N"]
        for ci in sorted(comp_idx, key=lambda c: -yc[c]):
            mapping[ci] = (bad_p.pop(0) if (yc[ci] == 1 and bad_p) else (good_p.pop(0) if good_p else bad_p.pop(0)))
    return "multiset", mapping, mu, sd


def find_match(comp, pub, product, feats, log):
    n = len(comp)
    for equip in ("S14", None):
        sub = pub[pub.PART_NAME.astype(str).str.upper().str.startswith(product.upper())]
        if equip and "EQUIP_CD" in sub.columns:
            sub = sub[sub.EQUIP_CD.astype(str).eq(equip)]
        sub = sub.reset_index(drop=True)
        dates = sub.time.dt.date
        starts = [0] + list(np.flatnonzero(dates.ne(dates.shift()).to_numpy())[1:])
        for s in starts:
            cand = sub.iloc[s:s + n]
            if len(cand) < n:
                break
            res = try_match(comp, cand, feats)
            if res:
                log(f"  {product}: 매칭 성공 ({res[0]}) — 원본 {cand.time.min():%Y-%m-%d %H:%M} ~ "
                    f"{cand.time.max():%Y-%m-%d %H:%M}, 설비 필터 {equip or '없음'}")
                return cand.reset_index(drop=True), res
    log(f"  {product}: 매칭 실패 — 대회 파일이 원본의 연속 구간이 아니거나 전처리가 다름")
    return None, None


def shot_type_col(df, feats):
    key = df["TimeStamp"].astype(str) + "|" + pd.util.hash_pandas_object(df[feats].round(6), index=False).astype(str)
    g = df.groupby(key)["y"]
    size, s = g.transform("size"), g.transform("sum")
    return np.select([s == 0, (s == size) & (size >= 2), (s > 0) & (s < size)],
                     ["normal", "process_wide", "cavity_local"], "single_row"), key


def tail_prob(q, k: int) -> float:
    """각 행이 앞에 올 확률이 q_i일 때 k개 이상이 앞에 올 확률 (포아송-이항 분포의 꼬리, 정확 계산)."""
    dist = np.zeros(len(q) + 1)
    dist[0] = 1.0
    for qi in q:
        dist[1:] = dist[1:] * (1 - qi) + dist[:-1] * qi
        dist[0] *= 1 - qi
    return float(dist[k:].sum())


def md(df, fmt=3):
    if df is None or len(df) == 0:
        return "_(없음)_\n"
    d = df.reset_index()
    f = lambda v: (f"{v:.{fmt}f}" if isinstance(v, (float, np.floating)) and not np.isnan(v)  # noqa: E731
                   else ("" if isinstance(v, (float, np.floating)) else str(v)))
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "|".join(["---"] * len(d.columns)) + "|"]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--public", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--enable", action="store_true", help="external.enabled=false 여도 이번 한 번 공개 원본을 읽음")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.synthetic:  # 가짜 데이터: 외부 데이터가 아니므로 허용 확인 불필요
        data_dir = ROOT / "data" / "synthetic"
        out_root = ROOT / cfg["output_dir"] / "synthetic"
        pub_path = data_dir
    else:
        if not _public.allowed(cfg, args.enable):
            return
        data_dir = ROOT / cfg["data_dir"]
        out_root = ROOT / cfg["output_dir"]
        pub_path = _public.public_path(cfg, args.public)
    if not pub_path.exists() or not _public.has(pub_path, "labeled_data.csv"):
        print(f"공개 원본(labeled_data.csv)을 찾을 수 없습니다: {pub_path}\n"
              "config.json 의 external.public_path 를 KAMP '사출성형기 AI 데이터셋' zip 또는 폴더로 지정하세요.")
        return
    out = out_root / "link"
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    comp_all, feats = {}, None
    for p, files in cfg["products"].items():
        c = read_csv_any(data_dir / files["labeled"]).rename(columns={cfg["id_col"]: "row_id"})
        c["y"] = (c[cfg["label_col"]] == cfg["positive_label"]).astype(int)
        feats = [x for x in c.columns if x not in ("row_id", cfg["label_col"], "y")]
        comp_all[p] = c.sort_values("row_id").reset_index(drop=True)
    pub = load_public(pub_path, feats)
    log(f"원본 {len(pub) + pub.attrs['dropped_duplicates']:,}행 (완전 중복 {pub.attrs['dropped_duplicates']:,}행 제거 → {len(pub):,}행)")

    linked, scale, R = [], {}, ["# 공개 원본 연결 결과\n", "> ⚠ 대회 규정의 외부 데이터 허용 범위 확인 후 사용. 기본 용도는 해석.\n"]
    for p, comp in comp_all.items():
        cand, res = find_match(comp, pub, p, feats, log)
        if cand is None:
            R.append(f"## {p.upper()}: 매칭 실패\n")
            continue
        method, mapping, mu, sd = res
        scale[p] = {f: {"mean": float(m), "std": float(s)} for f, m, s in zip(feats, mu, sd)}
        m = cand.iloc[mapping].reset_index(drop=True)
        df = pd.DataFrame({"row_id": comp.row_id, "product": p, "y": comp.y, "PassOrFail": m.PassOrFail,
                           "TimeStamp": m.TimeStamp, "PART_NAME": m.PART_NAME,
                           "Reason": m.Reason if "Reason" in m.columns else None})
        df["cavity"] = df.PART_NAME.astype(str).str.extract(r"\b(LH|RH)\b", expand=False)
        t = pd.to_datetime(df.TimeStamp)
        df["date"] = t.dt.date.astype(str)
        df["minutes_from_day_start"] = ((t - t.groupby(df.date).transform("min")).dt.total_seconds() / 60).round(1)
        for f in feats:
            df[f"raw_{f}"] = m[f].to_numpy()
        df["shot_type"], key = shot_type_col(df.assign(**{f: m[f] for f in feats}), feats)
        label_ok = float((df.y == (df.PassOrFail == "N").astype(int)).mean())
        log(f"  {p}: 라벨 일치 {label_ok:.1%}")
        linked.append(df)

        # 짝 안 행 순서 ↔ 캐비티 (대회 파일의 행 순서가 캐비티를 담고 있는지)
        pairs = df.assign(k=key).sort_values("row_id").groupby("k").filter(lambda g: len(g) == 2)
        first_cav = pairs.groupby("k").cavity.first()
        cl = pairs[pairs.shot_type.eq("cavity_local")]
        cl_first_is_defect = cl.groupby("k").y.first().mean() if len(cl) else np.nan
        rh_first = float(first_cav.eq("RH").mean())
        dr = cl[cl.y.eq(1)]  # 한쪽 불량 짝의 불량 행 (짝마다 1행)
        # 행 순서가 캐비티만 반영한다면, 불량 행이 앞에 올 확률 = 그 행의 캐비티가 앞에 올 확률
        q = np.where(dr.cavity.eq("RH"), rh_first, 1 - rh_first) if len(dr) else np.array([])
        exp_first = float(q.mean()) if len(q) else np.nan
        n_obs = int(round(cl_first_is_defect * len(q))) if len(q) else 0
        p_val = tail_prob(q, n_obs) if len(q) else np.nan  # 캐비티만으로 이만큼 치우칠 확률
        R.append(f"## {p.upper()}\n")
        R.append(f"- 매칭: `{method}` / 원본 구간 {t.min():%Y-%m-%d %H:%M} ~ {t.max():%Y-%m-%d %H:%M} / 라벨 일치 {label_ok:.1%}")
        R.append(f"- 짝의 앞 행(row_id가 작은 쪽 = 원본에 먼저 기록된 행)이 RH인 비율: {rh_first:.1%} (짝 {len(first_cav)}개)")
        R.append(f"- 한쪽만 불량인 짝에서 불량 행이 앞 행인 비율: {cl_first_is_defect:.1%} ({n_obs}/{len(q)}) "
                 f"— 행 순서가 캐비티만 반영한다면 기대값 {exp_first:.1%}, 우연히 이만큼 치우칠 확률 p = {p_val:.3g}")
        if len(q) and cl_first_is_defect - exp_first > 0.15 and p_val < 0.1:
            R.append("- ⚠ **짝 순서에 불량 여부가 섞여 있음(누수)**: 불량 행이 앞에 오는 비율이 캐비티로 설명되는 수준보다 "
                     "훨씬 높음 → 짝 순서는 캐비티가 아니라 기록 순서이며 라벨과 얽혀 있음. "
                     "`features.pair_position`은 모델에 쓰지 말 것 (Ablation 비교용으로만)")
        elif rh_first >= 0.95 or rh_first <= 0.05:
            R.append("- 짝 순서 ≈ 캐비티 → 캐비티 변수로 해석 가능 (대회 허용 범위 확인 후)")
        R.append("")
        bad = df[df.y == 1]
        R.append("**불량 행: 유형 × 캐비티**\n")
        R.append(md(pd.crosstab(bad.shot_type, bad.cavity, margins=True)))
        R.append("\n**캐비티별 불량률 (행 단위)**\n")
        R.append(md(df.groupby("cavity").y.agg(rows="size", defects="sum", defect_rate="mean")))
        R.append("\n**날짜별 행 수·불량 수**\n")
        R.append(md(df.groupby("date").agg(rows=("y", "size"), defects=("y", "sum"),
                                           first_time=("TimeStamp", "min"), last_time=("TimeStamp", "max"))))
        if len(bad):
            R.append("\n**불량 발생 시점 (그날 첫 기록 이후 경과 분)**\n")
            R.append(md(bad.groupby(["date", "shot_type"]).minutes_from_day_start.agg(["size", "min", "median", "max"])))
            if bad.Reason.notna().any():
                R.append("\n**불량 사유(Reason)**\n")
                R.append(md(pd.crosstab(bad.Reason.fillna("(없음)"), bad.shot_type)))
        # 날짜 안에서의 식별력: 모델이 '불량이 난 날의 운전 상태'만 맞히는지, 그날 안에서도 불량을 가려내는지
        pred_path = out_root / "predictions_labeled_cv.csv"
        if pred_path.exists():
            pr = read_csv_any(pred_path)
            pr = pr[pr["product"] == p][["row_id", "cv_risk_score"]]
            ev = df.merge(pr, on="row_id", how="inner")
            if len(ev) and ev.y.sum():
                top10 = ev.cv_risk_score >= ev.cv_risk_score.quantile(0.9)
                rows_ = []
                for d, g in ev.groupby("date"):
                    r = {"date": d, "rows": len(g), "defects": int(g.y.sum()), "defect_rate": float(g.y.mean()),
                         "share_of_overall_top10%": float(top10[g.index].sum() / max(top10.sum(), 1))}
                    if 0 < g.y.sum() < len(g):
                        ap_ = float(average_precision_score(g.y, g.cv_risk_score))
                        r.update({"AP_within_date": ap_, "lift_within_date": ap_ / r["defect_rate"],
                                  "recall@10%_within_date": expected_topk(g.y, g.cv_risk_score,
                                                                         int(np.ceil(len(g) * 0.1))) / g.y.sum()})
                    rows_.append(r)
                R.append("\n**날짜 안에서의 식별력 (run_all의 교차검증 예측 기준)**\n")
                R.append("그날 행끼리만 순위를 매겨도 불량이 위에 오는가? `lift_within_date`≈1이면 그날 안에서는 무작위와 같고, "
                         "모델의 성능은 '불량이 난 날의 운전 상태'를 알아본 데서 나온 것. "
                         "`share_of_overall_top10%`는 전체 위험 상위 10% 중 그날 행의 비율.\n")
                R.append(md(pd.DataFrame(rows_).set_index("date")))

        imp_path = out_root / "tables" / f"importance_{p}.csv"
        top = DEFAULT_TOP
        if imp_path.exists():
            imp = pd.read_csv(imp_path, index_col=0)
            top = [c for c in imp.index if c in feats][:6] or DEFAULT_TOP
        top = [c for c in top if c in feats]
        grp = df.assign(group=np.where(df.y == 0, "normal", df.shot_type))
        R.append("\n**주요 변수 원 단위 중앙값 (정상 vs 불량 유형)**\n")
        R.append(md(grp.groupby("group")[[f"raw_{c}" for c in top]].median().T))
        eb = out_root / "tables" / f"error_bins_{p}.csv"
        if eb.exists():
            bins = pd.read_csv(eb)

            def to_raw(row):
                a, b = map(float, re.findall(r"-?\d+\.?\d*", row["range"])[:2])
                s = scale[p].get(row.feature)
                return f"[{a*s['std']+s['mean']:.2f}, {b*s['std']+s['mean']:.2f}]" if s else ""
            bins["range_raw"] = bins.apply(to_raw, axis=1)
            R.append("\n**오류 구간표의 원 단위 환산 (run_all 결과 error_bins)**\n")
            R.append(md(bins[["feature", "bin", "range", "range_raw", "rows", "defects", "TP", "FN", "FP"]].set_index("feature")))

        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.scatter(t, np.zeros(len(t)), marker="|", s=80, color="#cfd6df", label="production")
        colors = {"process_wide": "#c0392b", "cavity_local": "#e08e0b", "single_row": "#7d3c98"}
        for (ty, cav), g in bad.groupby(["shot_type", "cavity"]):
            ax.scatter(pd.to_datetime(g.TimeStamp), np.where(g.cavity.eq("RH"), 0.5, -0.5), s=40,
                       color=colors.get(ty, "k"), marker="o" if cav == "RH" else "^", label=f"{ty} {cav} ({len(g)})")
        ax.set(yticks=[-0.5, 0, 0.5], yticklabels=["LH defect", "", "RH defect"], ylim=(-1, 1),
               title=f"{p.upper()} defects over time by cavity")
        ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        savefig(fig, out / f"link_timeline_{p}.png")
        R.append(f"\n![](link_timeline_{p}.png)\n")

    if linked:
        pd.concat(linked, ignore_index=True).to_csv(out / "linked_labeled.csv", index=False, encoding="utf-8-sig")
    (out / "scale_params.json").write_text(json.dumps(scale, ensure_ascii=False, indent=1), encoding="utf-8")
    R.insert(2, "\n".join(f"- {x}" for x in lines) + "\n")
    (out / "link_report.md").write_text("\n".join(R), encoding="utf-8")
    log(f"완료: {out / 'link_report.md'}")


if __name__ == "__main__":
    main()
