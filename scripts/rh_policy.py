"""검사 정책 비교: '모델 점수만' vs 'RH 캐비티 우선 규칙' vs '규칙 + 모델' (실험 R, 해석 전용).

왜 하는가
- 공개 원본의 PART_NAME 으로 보면 불량 42건 중 38건이 RH(오른쪽 캐비티) 제품이다 (RG3 25/25, CN7 13/17).
  현장에서 LH/RH 는 부품 번호로 이미 구분되므로, 심사위원이 가장 먼저 물을 질문은
  "그냥 RH 만 검사하면 되지 않나?"이다. 우리 검사 우선순위가 이 단순 규칙보다 나은지, 둘을 합치면 어떤지를 같은 예산에서 비교한다.
- side(LH/RH)는 **모델 입력으로 쓰지 않는다** (external.enabled=false). 모델 점수는 대회 파일만으로 학습한 교차검증
  예측(OOF)이고, side 는 검사 순서를 정하는 규칙에만 쓴다 — 현장에서는 부품 라벨로 바로 알 수 있는 정보.

정책 (모두 행=제품 단위, 예산 = 전체 제품 중 검사 비율)
  model        : 모델 점수 높은 순
  rh_random    : RH 제품 먼저(RH 안에서는 무작위), 남으면 LH 무작위 → 기대값으로 계산
  rh_then_model: RH 제품 먼저 모델 점수 순, 남으면 LH 모델 점수 순
  model_x_side : 모델 점수 × 학습 fold 에서 추정한 side 별 불량률 (fold 안에서만 추정 → 누수 없음)
  random       : 무작위 (= 예산)

실행: python scripts/rh_policy.py --enable [--public 경로] [--fast]
결과: outputs/verification/exp_rh_policy.csv (공개 원본 유래라 제출 zip 제외 폴더)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import _public

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import features  # noqa: E402
from src.cv import run_cv  # noqa: E402
from src.data import assign_shots, load_all  # noqa: E402
from src.evaluate import expected_topk  # noqa: E402

warnings.filterwarnings("ignore")
BUDGETS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def ranked_recall(y, order_key_major, order_key_minor, budget, rng_ties=True):
    """major 내림차순 → minor 내림차순으로 검사. minor 동점은 기대값 처리(expected_topk)."""
    y = np.asarray(y, float)
    k = int(np.ceil(len(y) * budget))
    # major 단계별로 채워 나감
    got, left = 0.0, k
    for v in sorted(np.unique(order_key_major))[::-1]:
        m = order_key_major == v
        n = int(m.sum())
        if left <= 0:
            break
        take = min(left, n)
        got += expected_topk(y[m], np.asarray(order_key_minor)[m], take)
        left -= take
    return got / y.sum() if y.sum() else np.nan


def side_rate_score(oof: pd.DataFrame) -> np.ndarray:
    """fold 별로 학습 쪽(= 그 fold 가 아닌 행)의 side 불량률을 추정해 점수에 곱한다."""
    out = np.zeros(len(oof))
    for k in oof["fold"].unique():
        va = oof["fold"].eq(k).to_numpy()
        tr = ~va
        rate = oof.loc[tr].groupby("side")["y"].mean()
        prior = oof.loc[va, "side"].map(rate).fillna(oof.loc[tr, "y"].mean()).to_numpy()
        out[va] = oof.loc[va, "p"].to_numpy() * (prior + 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enable", action="store_true")
    ap.add_argument("--public", default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--models", default="logreg,guidebook_ae")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if not _public.allowed(cfg, args.enable):
        return
    if args.fast:
        cfg["cv"]["n_repeats"] = 2
    src = _public.public_path(cfg, args.public)
    pub = _public.read_csv(src, "moldset_labeled.csv")[["Unnamed: 0", "PART_NAME"]]
    pub["side"] = pub["PART_NAME"].astype(str).str.extract(r"\b(LH|RH)\b", expand=False)

    data = load_all(cfg, ROOT / cfg["data_dir"])
    raw = data["feature_cols"]
    for p in data["labeled"]:
        data["labeled"][p] = assign_shots(data["labeled"][p], raw, p)
    base, _ = features.select_base_features(data, raw, True)

    from guidebook_audit import patch_make_model  # 같은 폴더
    models = args.models.split(",")
    rows = []
    for p, lab in data["labeled"].items():
        lab = lab.reset_index(drop=True).merge(pub[["Unnamed: 0", "side"]], left_on="row_id",
                                               right_on="Unnamed: 0", how="left", validate="1:1")
        assert lab["side"].notna().all(), "side 연결 실패"
        print(f"{p}: 불량 {int(lab.y.sum())} = RH {int(lab.y[lab.side.eq('RH')].sum())} / "
              f"LH {int(lab.y[lab.side.eq('LH')].sum())}, RH 비율 {lab.side.eq('RH').mean():.3f}", flush=True)
        patch_make_model(p)
        for mode in ("shot", "block"):
            oof, _, _ = run_cv(lab, base, models, cfg, p, mode, log=lambda *_: None)
            oof = oof.merge(lab[["row_id", "side"]], on="row_id", how="left")
            for (name, rep), g in oof.groupby(["model", "repeat"]):
                g = g.reset_index(drop=True)
                y = g["y"].to_numpy()
                is_rh = g["side"].eq("RH").to_numpy().astype(int)
                s = g["p"].to_numpy()
                sx = side_rate_score(g)
                for b in BUDGETS:
                    k = int(np.ceil(len(y) * b))
                    rows += [
                        dict(product=p, cv_mode=mode, model=name, repeat=rep, budget=b, policy="model",
                             recall=expected_topk(y, s, k) / y.sum()),
                        dict(product=p, cv_mode=mode, model=name, repeat=rep, budget=b, policy="rh_random",
                             recall=ranked_recall(y, is_rh, np.zeros(len(y)), b)),
                        dict(product=p, cv_mode=mode, model=name, repeat=rep, budget=b, policy="rh_then_model",
                             recall=ranked_recall(y, is_rh, s, b)),
                        dict(product=p, cv_mode=mode, model=name, repeat=rep, budget=b, policy="model_x_side",
                             recall=expected_topk(y, sx, k) / y.sum()),
                        dict(product=p, cv_mode=mode, model=name, repeat=rep, budget=b, policy="random",
                             recall=min(1.0, k / len(y))),
                    ]
            print(f"  {p} {mode} 완료", flush=True)
    res = pd.DataFrame(rows)
    out = ROOT / cfg["output_dir"] / "verification"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "exp_rh_policy_raw.csv", index=False, encoding="utf-8-sig")
    tab = (res.groupby(["product", "cv_mode", "model", "policy", "budget"])["recall"].mean().round(3)
           .unstack("budget").reset_index())
    tab.to_csv(out / "exp_rh_policy.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
