"""대회 파일 ↔ KAMP 공개 원본 연결 검증 (읽기 전용, 해석·검증 전용).

    python scripts/verify_public_link.py --enable            # config.external.public_path 사용
    python scripts/verify_public_link.py --enable --public 경로(zip 또는 폴더)

기본 꺼짐: config.json external.enabled=false 이면 --enable 없이는 실행하지 않는다.
공개 원본은 읽기만 하고, 결과는 outputs/verification/ 에만 저장한다 (git 제외). 모델 입력으로 쓰지 않는다.

확인하는 것
  1) 연결 확인: 대회 labeled 의 row_id(Unnamed: 0) = 공개 moldset_labeled.csv 의 Unnamed: 0,
     대회 unlabeled 의 row_id = 공개 unlabeled_data.csv 의 Unnamed: 0 인지.
     → 같은 row_id 의 원 단위 값을 파일(제품)별로 표준화(ddof=0)했을 때 대회 z값과 일치하면 같은 행.
  2) 라벨 의미 (사실 1): PassOrFail=1 ⇔ 원본 Reason(불량 사유)이 있음.
  3) 짝 안의 순서 (사실 6): 짝(같은 shot 2행)의 앞 행(row_id 작은 쪽)이 RH 인 비율,
     한쪽만 불량인 짝에서 불량 행이 앞 행인 비율 → 순서가 캐비티가 아니라 기록 순서이며 불량과 얽혀 있는지.
  4) unlabeled 안의 labeled 행 (사실 10): labeled 행과 (TimeStamp, PART_NAME, 원 단위 공정값)이 같은
     행이 대회 unlabeled 에 몇 개 있는지.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _public  # scripts/_public.py (같은 폴더)

ROOT = Path(__file__).resolve().parents[1]
META = ["Unnamed: 0", "_id", "TimeStamp", "PART_NAME"]
Z_TOL = 1e-4  # 표준화 재현 허용 오차 (원본이 float32 로 저장돼 생기는 반올림 차이)


def cavity(part_name: pd.Series) -> pd.Series:
    return part_name.astype(str).str.extract(r"\b(LH|RH)\b", expand=False)


def shot_ids(df: pd.DataFrame, feats: list[str]) -> pd.Series:
    """src/data.assign_shots 와 같은 기준: 공정값 전체가 같은 행 = 같은 shot (전역)."""
    key = pd.util.hash_pandas_object(df[feats].round(10), index=False)
    return pd.Series(pd.factorize(key)[0], index=df.index)


def link_by_row_id(comp: pd.DataFrame, pub: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, dict]:
    """comp(대회, z값)의 row_id 로 pub(원 단위)를 붙이고, pub 을 같은 행 집합에서 표준화해 z값과 비교."""
    m = comp[["row_id"]].merge(pub, left_on="row_id", right_on="Unnamed: 0", how="left", validate="1:1")
    found = m["Unnamed: 0"].notna()
    raw = m.loc[found, feats].astype(float)
    z = (raw - raw.mean()) / raw.std(ddof=0).replace(0, 1.0)
    const = raw.std(ddof=0).eq(0)
    zc = comp.loc[found.to_numpy(), feats].to_numpy(float)
    diff = np.abs(z.to_numpy() - zc)
    diff[:, const.to_numpy()] = 0.0  # 상수열은 대회 파일에서도 0
    per_col = pd.Series(diff.max(axis=0), index=feats)
    info = {"rows": int(len(comp)), "found_in_public": int(found.sum()),
            "max_abs_z_diff": float(per_col.max()) if len(per_col) else None,
            "rows_all_cols_within_tol": int((diff.max(axis=1) <= Z_TOL).sum()),
            "worst_col": str(per_col.idxmax()) if len(per_col) else None}
    return m, info


def raw_key(df: pd.DataFrame, feats: list[str]) -> pd.Series:
    r = df[feats].astype(float).round(3).astype(str).agg("|".join, axis=1)
    return df["TimeStamp"].astype(str) + "|" + df["PART_NAME"].astype(str) + "|" + r


def md_table(d: pd.DataFrame) -> str:
    cols = list(d.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="대회 파일 ↔ KAMP 공개 원본 연결 검증 (읽기 전용)")
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--public", default=None, help="공개 원본 zip 또는 폴더 (기본 config.external.public_path)")
    ap.add_argument("--enable", action="store_true", help="external.enabled=false 여도 이번 한 번 읽음")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본 <output_dir>/verification)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not _public.allowed(cfg, args.enable):
        return 0
    src = _public.public_path(cfg, args.public)
    for name in ("moldset_labeled.csv", "unlabeled_data.csv"):
        if not src.exists() or not _public.has(src, name):
            print(f"공개 원본에 {name} 이 없습니다: {src}")
            return 1
    out = Path(args.out) if args.out else ROOT / cfg["output_dir"] / "verification"
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / cfg["data_dir"]
    id_col, label_col, pos = cfg["id_col"], cfg["label_col"], cfg["positive_label"]

    # 대회 파일 (z값)
    comp_lab, comp_unl = {}, {}
    for p, files in cfg["products"].items():
        comp_lab[p] = pd.read_csv(data_dir / files["labeled"]).rename(columns={id_col: "row_id"})
        comp_unl[p] = pd.read_csv(data_dir / files["unlabeled"]).rename(columns={id_col: "row_id"})
    feats = [c for c in comp_lab[next(iter(comp_lab))].columns if c not in ("row_id", label_col)]

    print(f"공개 원본 읽기(읽기 전용): {src}")
    pub_lab = _public.read_csv(src, "moldset_labeled.csv", usecols=META + ["PassOrFail", "Reason"] + feats)
    pub_unl = _public.read_csv(src, "unlabeled_data.csv", usecols=META + feats)
    print(f"  moldset_labeled.csv {len(pub_lab):,}행, unlabeled_data.csv {len(pub_unl):,}행")

    summary = {"public_source": str(src), "note": "해석·검증 전용. 모델 입력 금지.", "products": {}}
    R = ["# 공개 원본 연결 검증 (자동 생성)", "",
         "> ⚠ 해석·검증 전용입니다. 공개 원본은 대회 규정 확인 전까지 모델 입력으로 쓰지 않습니다.", "",
         f"- 원본: `{src}` (읽기 전용)", f"- 표준화 재현 허용 오차: |Δz| ≤ {Z_TOL:g}", ""]
    link_rows, overlap_rows = [], []
    for p in comp_lab:
        s = {}
        # 1) 연결 확인
        ml, s["labeled_link"] = link_by_row_id(comp_lab[p], pub_lab, feats)
        mu, s["unlabeled_link"] = link_by_row_id(comp_unl[p], pub_unl, feats)

        # 2) 라벨 의미
        lab = comp_lab[p].assign(y=(comp_lab[p][label_col] == pos).astype(int),
                                 cavity=cavity(ml["PART_NAME"]).to_numpy(),
                                 has_reason=ml["Reason"].notna().to_numpy(),
                                 pub_label=ml["PassOrFail"].to_numpy(),
                                 Reason=ml["Reason"].to_numpy(), TimeStamp=ml["TimeStamp"].to_numpy(),
                                 PART_NAME=ml["PART_NAME"].to_numpy())
        s["label_vs_public_passorfail"] = float((lab.y == lab.pub_label).mean())
        s["label_vs_reason_present"] = float((lab.y == lab.has_reason.astype(int)).mean())
        s["reason_counts_defects"] = lab.loc[lab.y == 1, "Reason"].value_counts().to_dict()

        # 3) 짝 안의 순서 (사실 6)
        lab["shot"] = shot_ids(lab, feats)
        lab = lab.sort_values("row_id")
        size = lab.groupby("shot")["row_id"].transform("size")
        pairs = lab[size == 2].copy()
        pairs["first"] = pairs.groupby("shot").cumcount().eq(0)
        first = pairs[pairs["first"]]
        cav_pair = pairs.groupby("shot")["cavity"].agg(lambda c: "".join(sorted(c.dropna())))
        npos = pairs.groupby("shot")["y"].transform("sum")
        one = pairs[npos == 1]
        one_def = one[one.y == 1]
        s["pairs"] = int(pairs["shot"].nunique())
        s["pairs_one_LH_one_RH"] = int(cav_pair.eq("LHRH").sum())
        s["first_row_is_RH_share"] = float(first["cavity"].eq("RH").mean())
        s["one_sided_pairs"] = int(one["shot"].nunique())
        s["one_sided_defect_row_first"] = int(one_def["first"].sum())
        s["one_sided_defect_row_first_share"] = float(one_def["first"].mean()) if len(one_def) else None
        s["one_sided_defect_cavity"] = one_def["cavity"].value_counts().to_dict()
        # 순서가 캐비티를 나타낸다면 기대되는 '불량 행 = 앞 행' 비율: 불량 캐비티가 앞에 올 확률의 평균
        pr = first["cavity"].eq("RH").mean()
        exp = np.where(one_def["cavity"].eq("RH"), pr, 1 - pr)
        s["expected_first_share_if_order_were_cavity"] = float(exp.mean()) if len(exp) else None
        s["all_defects_by_cavity"] = lab.loc[lab.y == 1, "cavity"].value_counts().to_dict()

        # 4) unlabeled 안의 labeled 행 (사실 10)
        mu_ok = mu[mu["Unnamed: 0"].notna()]
        ku = pd.Series(raw_key(mu_ok, feats).to_numpy(), index=mu_ok["row_id"].to_numpy())
        kl = pd.Series(raw_key(ml[ml["Unnamed: 0"].notna()], feats).to_numpy(),
                       index=ml.loc[ml["Unnamed: 0"].notna(), "row_id"].to_numpy())
        ts_u = set(zip(mu_ok["TimeStamp"].astype(str), mu_ok["PART_NAME"].astype(str)))
        in_unl_full = kl.isin(set(ku.values))
        in_unl_ts = pd.Series([(str(t), str(n)) in ts_u for t, n in zip(ml["TimeStamp"], ml["PART_NAME"])],
                              index=ml["row_id"].to_numpy())
        s["labeled_rows_in_unlabeled_exact"] = int(in_unl_full.sum())
        s["labeled_rows_in_unlabeled_by_time_part"] = int(in_unl_ts.sum())
        s["labeled_defects_in_unlabeled_exact"] = int(in_unl_full[lab.set_index("row_id").loc[kl.index, "y"]
                                                                  .to_numpy() == 1].sum())
        inv = ku.reset_index().groupby(0)["index"].apply(list)
        for rid, key in kl[in_unl_full].items():
            overlap_rows.append({"product": p, "labeled_row_id": int(rid),
                                 "unlabeled_row_ids": " ".join(map(str, inv.get(key, [])))})
        summary["products"][p] = s
        link_rows.append(lab[["row_id", "y", "shot", "cavity", "Reason", "TimeStamp"]].assign(product=p))

        L = s["labeled_link"]; U = s["unlabeled_link"]
        R += [f"## {p.upper()}", "",
              "### 1. 연결 확인 (row_id = 원본 Unnamed: 0, 같은 행 집합으로 표준화 재현)",
              md_table(pd.DataFrame([
                  {"대회 파일": "labeled", "행": L["rows"], "원본에서 찾음": L["found_in_public"],
                   "모든 열 |Δz|≤tol 행": L["rows_all_cols_within_tol"], "최대 |Δz|": f"{L['max_abs_z_diff']:.2e}"},
                  {"대회 파일": "unlabeled", "행": U["rows"], "원본에서 찾음": U["found_in_public"],
                   "모든 열 |Δz|≤tol 행": U["rows_all_cols_within_tol"], "최대 |Δz|": f"{U['max_abs_z_diff']:.2e}"}])),
              "", "### 2. 라벨 의미 (사실 1)",
              f"- PassOrFail(대회) = 원본 PassOrFail 일치율: {s['label_vs_public_passorfail']:.1%}",
              f"- PassOrFail=1 ⇔ 원본 Reason 있음 일치율: {s['label_vs_reason_present']:.1%}",
              f"- 불량 사유: {s['reason_counts_defects']}", "",
              "### 3. 짝 안의 순서 (사실 6)",
              f"- 짝 {s['pairs']}개 중 LH 1행 + RH 1행으로 구성된 짝: {s['pairs_one_LH_one_RH']}",
              f"- 앞 행(row_id 작은 쪽)이 RH 인 비율: {s['first_row_is_RH_share']:.1%}",
              f"- 한쪽만 불량인 짝 {s['one_sided_pairs']}개 중 불량 행이 앞 행: {s['one_sided_defect_row_first']}"
              f" ({s['one_sided_defect_row_first_share']:.1%})" if s["one_sided_defect_row_first_share"] is not None
              else f"- 한쪽만 불량인 짝: {s['one_sided_pairs']}",
              f"- 순서가 캐비티를 뜻한다면 기대 비율: {s['expected_first_share_if_order_were_cavity']:.1%}"
              if s["expected_first_share_if_order_were_cavity"] is not None else "",
              f"- 불량 행 캐비티 (한쪽만 불량): {s['one_sided_defect_cavity']} / 전체 불량: {s['all_defects_by_cavity']}", "",
              "### 4. unlabeled 안의 labeled 행 (사실 10)",
              f"- (TimeStamp, PART_NAME, 원 단위 공정값) 완전 일치: **{s['labeled_rows_in_unlabeled_exact']} / {L['rows']}**"
              f" (그중 불량 {s['labeled_defects_in_unlabeled_exact']})",
              f"- (TimeStamp, PART_NAME) 만 일치: {s['labeled_rows_in_unlabeled_by_time_part']} / {L['rows']}", ""]

    (out / "public_link_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                 encoding="utf-8")
    (out / "public_link_report.md").write_text("\n".join(R), encoding="utf-8")
    pd.concat(link_rows).to_csv(out / "labeled_cavity_link.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overlap_rows, columns=["product", "labeled_row_id", "unlabeled_row_ids"]).to_csv(
        out / "labeled_rows_in_unlabeled.csv", index=False, encoding="utf-8-sig")
    print("\n".join(R))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
