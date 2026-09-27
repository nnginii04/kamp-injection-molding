"""대회 파일 ↔ KAMP 공개 원본 연결 검증 (읽기 전용, 해석·검증 전용).

    python scripts/verify_public_link.py --enable            # config.external.public_path 사용
    python scripts/verify_public_link.py --enable --public 경로(zip 또는 폴더)

기본 꺼짐: config.json external.enabled=false 이면 --enable 없이는 실행하지 않는다.
공개 원본은 읽기만 하고, 결과는 outputs/verification/ 에만 저장한다 (git 제외). 모델 입력으로 쓰지 않는다.

확인하는 것
  1) 연결 확인 (대회 파일 ↔ 자기 원본 파일): 대회 labeled 의 Unnamed: 0 = 공개 moldset_labeled.csv 의 Unnamed: 0,
     대회 unlabeled 의 Unnamed: 0 = 공개 unlabeled_data.csv 의 Unnamed: 0 인지.
     → 같은 번호의 원 단위 값을 파일(제품)별로 표준화(ddof=0)했을 때 대회 z값과 |Δz| ≤ Z_TOL 이면 같은 행.
     (labeled 와 unlabeled 의 Unnamed: 0 은 서로 다른 번호 체계라 둘 사이를 잇는 데는 쓰지 않는다)
  2) 라벨 의미 (사실 1): PassOrFail=1 ⇔ 원본 Reason(불량 사유)이 있음.
  3) 짝 안의 순서 (사실 6): 짝(같은 shot 2행)의 앞 행(row_id 작은 쪽)이 RH 인 비율,
     한쪽만 불량인 짝에서 불량 행이 앞 행인 비율 → 순서가 캐비티가 아니라 기록 순서이며 불량과 얽혀 있는지.
  4) unlabeled 안의 labeled 행 (사실 10), 판정 기준 = src/known_facts.OVERLAP_CRITERION
     - 행 단위: (TimeStamp, PART_NAME) 키 1:1 매칭 + 대회 공정변수 24개를 소수 3자리 반올림한 값이 전부 같음
       (원본 labeled 는 float32 라 24.799999 vs 24.8 같은 차이 → 반올림 후 정확히 비교, 별도 허용 오차 없음)
     - shot 단위: (TimeStamp, 공정변수 24개 반올림 값)이 labeled 의 어떤 행과 같음 (PART_NAME 무시 → LH/RH 짝 포함)
     → outputs/verification/unlabeled_overlap_flags.csv (run_all.py 가 external.enabled=true 일 때만 읽음)
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
sys.path.insert(0, str(ROOT))
from src.known_facts import OVERLAP_CRITERION, OVERLAP_ROUND  # noqa: E402
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


def overlap(lab_raw: pd.DataFrame, unl_raw: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, dict]:
    """사실 10 판정 (OVERLAP_CRITERION). lab_raw/unl_raw: row_id, TimeStamp, PART_NAME, 공정변수(원 단위).
    반환: unlabeled 행별 플래그(행 단위·shot 단위, 매칭된 labeled row_id)와 요약."""
    key = ["TimeStamp", "PART_NAME"]
    L = lab_raw[["row_id", *key, *feats]].copy()
    U = unl_raw[["row_id", *key, *feats]].copy()
    for d in (L, U):
        d["TimeStamp"], d["PART_NAME"] = d["TimeStamp"].astype(str), d["PART_NAME"].astype(str)
        d["vals"] = d[feats].astype(float).round(OVERLAP_ROUND).astype(str).agg("|".join, axis=1)
    # 1) 키 매칭: 키가 한쪽 파일 안에서 중복이면 1:1 이 아니므로 제외하고 따로 센다
    dupL, dupU = L.duplicated(key, keep=False), U.duplicated(key, keep=False)
    m = L[~dupL].merge(U[~dupU], on=key, how="inner", suffixes=("_l", "_u"), validate="1:1")
    same = m["vals_l"].eq(m["vals_u"])
    raw_diff = max((float(np.abs(m.loc[same, f + "_l"].astype(float) - m.loc[same, f + "_u"].astype(float)).max())
                    for f in feats), default=np.nan) if same.any() else np.nan
    row_hit = m.loc[same, ["row_id_u", "row_id_l"]]
    # 2) shot 단위: PART_NAME 무시, (TimeStamp, 반올림 값)이 labeled 의 어떤 행과 같음
    shot_keys = set(zip(L["TimeStamp"], L["vals"]))
    U["shot_hit"] = [(t, v) in shot_keys for t, v in zip(U["TimeStamp"], U["vals"])]
    flags = U[["row_id"]].rename(columns={"row_id": "unlabeled_row_id"}).assign(
        in_labeled_train_row=U["row_id"].isin(set(row_hit.row_id_u)).to_numpy(),
        in_labeled_train_shot=U["shot_hit"].to_numpy())
    flags = flags.merge(row_hit.rename(columns={"row_id_u": "unlabeled_row_id", "row_id_l": "labeled_row_id"}),
                        on="unlabeled_row_id", how="left")
    info = {"labeled_rows": int(len(L)), "unlabeled_rows": int(len(U)),
            "labeled_dup_keys": int(dupL.sum()), "unlabeled_dup_keys": int(dupU.sum()),
            "key_matched_1to1": int(len(m)), "key_matched_values_equal": int(same.sum()),
            "key_matched_values_differ": int((~same).sum()),
            "max_raw_diff_before_rounding": raw_diff,
            "unlabeled_rows_in_labeled_row": int(flags.in_labeled_train_row.sum()),
            "unlabeled_rows_in_labeled_shot": int(flags.in_labeled_train_shot.sum())}
    return flags, info


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

    print(f"공개 원본 읽기(읽기 전용): {src.name}")
    pub_lab = _public.read_csv(src, "moldset_labeled.csv", usecols=META + ["PassOrFail", "Reason"] + feats)
    pub_unl = _public.read_csv(src, "unlabeled_data.csv", usecols=META + feats)
    print(f"  moldset_labeled.csv {len(pub_lab):,}행, unlabeled_data.csv {len(pub_unl):,}행")

    summary = {"public_source": src.name, "note": "해석·검증 전용. 모델 입력 금지.", "products": {}}
    R = ["# 공개 원본 연결 검증 (자동 생성)", "",
         "> ⚠ 해석·검증 전용입니다. 공개 원본은 대회 규정 확인 전까지 모델 입력으로 쓰지 않습니다.", "",
         f"- 원본: `{src.name}` (읽기 전용, 경로는 config.external.public_path)",
         f"- 허용 오차: (1) 대회 파일 ↔ 자기 원본 연결은 표준화 재현 |Δz| ≤ {Z_TOL:g}. "
         f"(2) labeled ↔ unlabeled 겹침은 허용 오차 없이 소수 {OVERLAP_ROUND}자리 반올림 값의 완전 일치", ""]
    link_rows, flag_rows = [], []
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

        # 4) unlabeled 안의 labeled 행 (사실 10) — 판정 기준: OVERLAP_CRITERION
        flags, ov = overlap(ml[ml["Unnamed: 0"].notna()], mu[mu["Unnamed: 0"].notna()], feats)
        y_by_row = lab.set_index("row_id")["y"]
        ov["labeled_defects_in_unlabeled"] = int(y_by_row.reindex(flags.labeled_row_id.dropna().astype(int)).sum())
        s["overlap"] = ov
        flag_rows.append(flags[flags.in_labeled_train_row | flags.in_labeled_train_shot].assign(product=p))
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
              f"- 기준: {OVERLAP_CRITERION}",
              f"- (TimeStamp, PART_NAME) 1:1 키 매칭: {ov['key_matched_1to1']} / labeled {ov['labeled_rows']} "
              f"(파일 안 키 중복으로 제외: labeled {ov['labeled_dup_keys']}, unlabeled {ov['unlabeled_dup_keys']}행)",
              f"- 그중 공정변수 24개(소수 {OVERLAP_ROUND}자리)가 전부 같음: **{ov['key_matched_values_equal']} / {L['rows']}**"
              f" (불량 {ov['labeled_defects_in_unlabeled']}), 값이 다른 키 매칭 {ov['key_matched_values_differ']}행, "
              f"반올림 전 원 단위 최대 차이 {ov['max_raw_diff_before_rounding']:.1e}",
              f"- unlabeled 쪽 플래그: 행 단위 {ov['unlabeled_rows_in_labeled_row']}행, "
              f"shot 단위(짝 포함) {ov['unlabeled_rows_in_labeled_shot']}행", ""]

    (out / "public_link_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                 encoding="utf-8")
    (out / "public_link_report.md").write_text("\n".join(R), encoding="utf-8")
    pd.concat(link_rows).to_csv(out / "labeled_cavity_link.csv", index=False, encoding="utf-8-sig")
    cols = ["product", "unlabeled_row_id", "in_labeled_train_row", "in_labeled_train_shot", "labeled_row_id"]
    pd.concat(flag_rows, ignore_index=True)[cols].to_csv(out / "unlabeled_overlap_flags.csv", index=False,
                                                          encoding="utf-8-sig")
    old = out / "labeled_rows_in_unlabeled.csv"  # 이전 형식(판정 기준이 암묵적) — 혼동 방지를 위해 지움
    if old.exists():
        old.unlink()
    print("\n".join(R))
    print(f"\n저장: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
