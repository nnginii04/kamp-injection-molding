"""사출성형 품질불량 사전예측 파이프라인 (팀 최종 베이스라인, v2 기반) — 한 번에 실행.

    python run_all.py                 # data/raw/ 의 대회 데이터로 전체 실행 → outputs/
    python run_all.py --fast          # 반복 수를 줄여 빠르게 확인
    python run_all.py --synthetic     # 가짜 데이터로 동작만 확인 → outputs/synthetic/ (성능 숫자 무의미)

외부(KAMP 공개 원본) 데이터는 읽지 않는다 (config.external 은 scripts/ 전용).

순서: 데이터 진단 → 피처 → (선택)홀드아웃 분리 → 반복 그룹 CV(샷/블록) 모델 비교 → 모델 선택
      → 불량유형별 탐지율 → 오류분석 → 최종 학습·확률 보정 → 예측 파일 3종 → report.md
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import audit as audit_mod  # noqa: E402
from src import errors, evaluate, experiments, features, final, key_figures, known_facts, report  # noqa: E402
from src.cv import fold_defect_check, fold_table, make_groups, run_cv  # noqa: E402
from src.data import assign_shots, load_all  # noqa: E402
from src.evaluate import shot_types  # noqa: E402
from src.models import COMPLEXITY  # noqa: E402
from src.plots import plt, savefig  # noqa: E402


def select_model(table, p, sel, n_repeats):
    metric, mode = sel["metric"], sel["cv_mode"]
    cand = table[(table["product"] == p) & (table.cv_mode == mode) & (table.model != "dummy")].copy()
    best = cand.sort_values(metric, ascending=False).iloc[0]
    chosen = best
    if sel.get("rule") == "one_se" and f"{metric}_sd" in cand.columns:
        se = float(best[f"{metric}_sd"]) / np.sqrt(max(n_repeats, 1))
        ok = cand[cand[metric] >= best[metric] - se].copy()
        ok["cx"] = ok.model.map(lambda m: COMPLEXITY.index(m) if m in COMPLEXITY else 99)
        chosen = ok.sort_values(["cx", metric], ascending=[True, False]).iloc[0]
    dummy = float(table[(table["product"] == p) & (table.cv_mode == mode) & (table.model == "dummy")][metric].mean())
    lift = float(chosen[metric] / dummy) if dummy > 0 else np.nan
    return {"scenario": chosen.scenario, "dataset": chosen.dataset, "model": chosen.model, "cv_mode": mode,
            "score": float(chosen[metric]), "best_model": best.model, "best_score": float(best[metric]),
            "dummy": dummy, "lift_vs_dummy": lift}


def load_labeled_overlap(cfg, out, log) -> dict:
    """unlabeled 중 labeled 와 겹치는 행(사실 10). 판정 기준은 known_facts.OVERLAP_CRITERION.
    대회 파일만으로는 판정할 수 없어(파일별 표준화) external.enabled=true 이고
    scripts/verify_public_link.py 결과(unlabeled_overlap_flags.csv)가 있을 때만 읽는다.
    반환: 제품별 {"row": 행 단위 겹침 row_id 집합, "shot": shot 단위 겹침 row_id 집합} 또는 None."""
    products = list(cfg["products"])
    na_msg = "in_labeled_train_row/_shot = NaN"
    if not cfg.get("external", {}).get("enabled", False):
        log(f"  ⚠ unlabeled 안의 labeled 행(사실 10)은 대회 파일만으로 판정 불가 → {na_msg} (external.enabled=false)")
        return {p: None for p in products}
    f = out / "verification" / "unlabeled_overlap_flags.csv"
    if not f.exists():
        log(f"  ⚠ external.enabled=true 이지만 {rel(f)} 가 없음 → python scripts/verify_public_link.py --enable 을 먼저 실행. "
            f"{na_msg}")
        return {p: None for p in products}
    t = pd.read_csv(f)
    res = {}
    for p in products:
        tp = t[t["product"] == p]
        res[p] = {"row": set(tp.loc[tp.in_labeled_train_row, "unlabeled_row_id"].astype(int)),
                  "shot": set(tp.loc[tp.in_labeled_train_shot, "unlabeled_row_id"].astype(int))}
    log(f"  in_labeled_train: 공개 원본 검증 결과 사용 ({f.name}) "
        + ", ".join(f"{p} 행 {len(v['row'])}·shot {len(v['shot'])}" for p, v in res.items()))
    return res


def rel(p) -> str:
    """저장소 루트 기준 상대경로 (산출물에 로컬 절대경로가 남지 않게)."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return p.name


def run_command(args) -> str:
    opts = [f"--{k.replace('_', '-')}" + ("" if v is True else f" {rel(v) if k in ('config', 'data_dir') else v}")
            for k, v in vars(args).items() if v not in (None, False) and not (k == "config" and
                                                                              Path(v).resolve() == ROOT / "config.json")]
    return " ".join(["python run_all.py", *opts])


def git_commit() -> dict:
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True)
        return {"commit": head.stdout.strip(), "uncommitted_changes": bool(dirty.stdout.strip())}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def library_versions() -> dict:
    import matplotlib
    import scipy
    v = {"python": platform.python_version(), "platform": platform.platform(terse=True), "numpy": np.__version__,
         "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
         "matplotlib": matplotlib.__version__}
    try:
        import lightgbm
        v["lightgbm"] = lightgbm.__version__
    except ImportError:
        v["lightgbm"] = None  # HistGradientBoosting 으로 대체됨
    return v


PRED_COLS = ["mold", "product", "model", "cv_mode", "repeat", "fold", "row_id", "shot_id", "y_true", "y_score", "y_pred"]


def write_pipeline_predictions(oof_all: pd.DataFrame, models: list[str], out: Path, log) -> list[Path]:
    """교차검증 예측을 팀 공통 형식으로 outputs/predictions/pipeline_<모델>.csv 에 저장 (run_evaluation.py 가 읽음).
    같은 폴더의 팀원 파일은 건드리지 않고, 이전 실행의 pipeline_*.csv 만 지우고 새로 쓴다."""
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for old in pred_dir.glob("pipeline_*.csv"):
        old.unlink()
    written = []
    for m in models:
        sub = oof_all[oof_all.model.eq(m)]
        if sub.empty:
            continue
        name = np.where(sub.dataset.eq("pooled"), m + "@pooled", m)
        t = pd.DataFrame({"mold": sub["product"], "product": sub["product"], "model": name, "cv_mode": sub.cv_mode,
                          "repeat": sub.repeat, "fold": sub.fold, "row_id": sub.row_id, "shot_id": sub.shot_id,
                          "y_true": sub.y, "y_score": sub.p, "y_pred": sub.pred})[PRED_COLS]
        f = pred_dir / f"pipeline_{m}.csv"
        t.to_csv(f, index=False, encoding="utf-8-sig")
        written.append(f)
    log(f"  교차검증 예측(공통 형식) {len(written)}개 → {rel(pred_dir)}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.fast:
        cfg["cv"]["n_repeats"] = 2
        cfg["bootstrap"]["n"] = 200
    np.random.seed(cfg["seed"])

    data_dir = Path(args.data_dir) if args.data_dir else ROOT / cfg["data_dir"]
    out = ROOT / cfg["output_dir"]
    if args.synthetic:
        from src.synthetic import generate
        data_dir = generate(ROOT / "data" / "synthetic")
        out = ROOT / cfg["output_dir"] / "synthetic"
        cfg["run_name"] += "-SYNTHETIC(성능 숫자 무의미)"
    fig_dir, tab_dir = out / "figures", out / "tables"
    for d in (fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    # 1. 로드 + 샷 그룹 + 불량 유형 -------------------------------------------
    log("[1/9] 데이터 로드")
    data = load_all(cfg, data_dir)
    raw_feats = data["feature_cols"]
    for p in data["labeled"]:
        data["labeled"][p] = assign_shots(data["labeled"][p], raw_feats, p)
        data["unlabeled"][p] = assign_shots(data["unlabeled"][p], raw_feats, f"{p}u")
    types_by_product = {p: shot_types(lab) for p, lab in data["labeled"].items()}
    products = list(data["labeled"])

    # 2. 진단 -----------------------------------------------------------------
    log("[2/9] 데이터 진단")
    audit, audit_tables = audit_mod.run_audit(data, raw_feats, cfg["seed"], fig_dir)
    for k, t in audit_tables.items():
        t.to_csv(tab_dir / f"{k}.csv", encoding="utf-8-sig")
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for p, a in audit.items():
        log(f"  {p}: 불량 샷 {a['defect_shots']} = 양쪽 {a['process_wide_shots(양쪽 불량)']} + 한쪽 "
            f"{a['cavity_local_shots(한쪽만 불량)']} / 눈금 점검: {a['scale_check']['verdict']}")

    # 3. 피처 -----------------------------------------------------------------
    log("[3/9] 피처 구성")
    fcfg = cfg["features"]
    features.validate_feature_config(fcfg)
    base, dropped = features.select_base_features(data, raw_feats, fcfg["drop_constant"])
    use_if = bool(fcfg.get("unlabeled_iforest_score", False))
    if use_if:
        features.add_iforest_scores(data, base, fcfg["iforest"], cfg["seed"])
    seq_cols = None
    if fcfg.get("sequence_features"):
        w = int(fcfg.get("sequence_window", 5))
        for key in ("labeled", "unlabeled"):
            for p in data[key]:
                data[key][p] = features.add_sequence_features(data[key][p], base, w)
        seq_cols = features.sequence_columns(base, w)
    models = [m for m in cfg["models"] if use_if or m != "iforest_unsup"]
    log(f"  기본 변수 {len(base)}개, 순서 피처 {len(seq_cols or [])}개, 짝 안 위치 변수 미사용(누수), "
        f"unlabeled IF 점수 {'사용' if use_if else '미사용'}")

    # 4. (선택) 홀드아웃 --------------------------------------------------------
    holdout_info, test_frames = None, {}
    hcfg = cfg.get("holdout", {})
    if hcfg.get("enabled"):
        log("[4/9] 홀드아웃 분리 (연속 블록 단위)")
        counts = {}
        for p in products:
            lab = data["labeled"][p].reset_index(drop=True)
            groups = make_groups(lab, "block", cfg["cv"].get("block_shots", 10))
            n_splits = max(2, int(round(1 / hcfg["frac"])))
            tr, te = next(StratifiedGroupKFold(n_splits, shuffle=True, random_state=cfg["seed"])
                          .split(np.zeros(len(lab)), lab.y, groups))
            test_frames[p] = lab.iloc[te].reset_index(drop=True)
            data["labeled"][p] = lab.iloc[tr].reset_index(drop=True)
            counts[p] = {"dev_rows": len(tr), "dev_defects": int(lab.y.iloc[tr].sum()),
                         "test_rows": len(te), "test_defects": int(lab.y.iloc[te].sum())}
        holdout_info = {"frac": hcfg["frac"], "counts": counts}
        log(f"  {counts}")
    else:
        log("[4/9] 홀드아웃 미사용 (labeled 전 행 교차검증 예측을 테스트 예측으로 사용)")

    # 5. CV -------------------------------------------------------------------
    log("[5/9] 반복 그룹 교차검증: " + ", ".join(cfg["cv"]["modes"]))
    oofs, mets, fitted, frames, feats_by = [], [], {}, {}, {}
    for scenario in cfg["scenarios"]:
        feats_by[scenario] = features.check_feature_list(features.model_features(base, scenario, use_if, seq_cols))
        for name, (lab, unl) in features.scenario_frames(data, scenario).items():
            lab = lab.reset_index(drop=True)
            frames[name] = (lab, unl)
            for mode in cfg["cv"]["modes"]:
                o, m, f = run_cv(lab, feats_by[scenario], models, cfg, name, mode, log=log)
                m["scenario"] = scenario
                m.loc[m["product"] == "all", "product"] = name if scenario == "separate" else "all"
                oofs.append(o), mets.append(m)
                fitted.setdefault(name, {}).update(f)
    oof_all, metrics = pd.concat(oofs, ignore_index=True), pd.concat(mets, ignore_index=True)
    metrics.to_csv(tab_dir / "cv_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    num = [c for c in metrics.columns if c not in ("cv_mode", "dataset", "product", "model", "repeat", "scenario")]
    g = metrics.groupby(["cv_mode", "scenario", "dataset", "product", "model"])[num]
    table = g.mean().join(g.std()[["shot_ap", "row_f1", "row_ap"]].add_suffix("_sd")).reset_index()
    table.to_csv(tab_dir / "cv_summary.csv", index=False, encoding="utf-8-sig")

    # 팀 공유용 분할표 + fold 수 점검 (run_cv 와 같은 cv_splits 사용)
    sep_frames = {p: frames[p][0] for p in products if p in frames}
    fold_id = None
    if sep_frames:
        fold_id = pd.concat([fold_table(lab, cfg) for lab in sep_frames.values()], ignore_index=True)
        if holdout_info:
            ho = pd.concat([tf.assign(mold=p)[["mold", "row_id", "shot_id"]] for p, tf in test_frames.items()])
            fold_id = pd.concat([fold_id.assign(is_holdout=False), ho.assign(is_holdout=True)], ignore_index=True)
        fold_id.to_csv(out / "fold_id.csv", index=False, encoding="utf-8-sig")
        fold_check = pd.concat([fold_defect_check(lab, cfg).assign(mold=p) for p, lab in sep_frames.items()],
                               ignore_index=True)
        fold_check.to_csv(tab_dir / "fold_defect_check.csv", index=False, encoding="utf-8-sig")
        log(f"  분할표 저장: {rel(out / 'fold_id.csv')} ({len(fold_id)}행)")
    else:
        fold_check = pd.DataFrame()

    # Ablation: 누수 변수(pair_first)를 켰을 때 vs 껐을 때 — 같은 분할, 모델 선택·예측 파일에는 미사용
    ablation = None
    acfg = cfg.get("ablation", {}).get("pair_position", {})
    if acfg.get("enabled") and "separate" in cfg["scenarios"]:
        log("[5b/9] Ablation: 짝 안 위치 변수(pair_first) — 누수 변수, 비교용")
        a_models = [m for m in acfg.get("models", []) if m in models]
        a_rows = []
        for p in products:
            lab_a = features.add_pair_position(frames[p][0])
            feats_a = features.check_feature_list(feats_by["separate"] + ["pair_first"],
                                                  allow=features.LEAKY_ABLATION_FEATURES)
            for mode in cfg["cv"]["modes"]:
                _, m_on, _ = run_cv(lab_a, feats_a, a_models, cfg, p, mode, log=lambda *_: None)
                a_rows.append(m_on.assign(product=p))
        met_on = pd.concat(a_rows, ignore_index=True)
        keys = ["cv_mode", "product", "model"]
        vals = ["shot_ap", "row_ap", "row_roc_auc", "row_f1@2%", "row_recall@10%"]
        on = met_on.groupby(keys)[vals].mean()
        off = metrics[metrics.scenario.eq("separate") & metrics.model.isin(a_models)].groupby(keys)[vals].mean()
        ablation = off.add_suffix("[OFF]").join(on.add_suffix("[ON 누수]"), how="inner").reset_index()
        for v in vals:
            ablation[f"{v} Δ"] = ablation[f"{v}[ON 누수]"] - ablation[f"{v}[OFF]"]
        ablation.insert(0, "note", "누수 변수 — 비교용")
        ablation.to_csv(tab_dir / "ablation_pair_position.csv", index=False, encoding="utf-8-sig")

    # 추가 실험 (Phase 3): 같은 분할, 모델 선택에는 미사용
    exp_tables, exp_extra, exp_oof = {}, {}, {}
    run_exp = cfg.get("experiments", {}).get("enabled") and "separate" in cfg["scenarios"]
    if run_exp:
        log("[5c/9] 추가 실험 (제품별 vs 통합, class_weight, 기준선) — 모델 선택에는 미사용")
        r = experiments.run_cv_experiments(frames, products, feats_by["separate"], metrics, oof_all, cfg, audit,
                                           log=log)
        exp_tables, exp_extra, exp_oof = r["tables"], r["extra_metrics"], r["extra_oof"]

    # 6. 선택 + 신뢰구간 ----------------------------------------------------------
    log("[6/9] 모델 선택 및 신뢰구간")
    sel = cfg["selection"]
    choices, ci = {}, {}
    for p in products:
        c = select_model(table, p, sel, cfg["cv"]["n_repeats"])
        c["beats_chance"] = bool(c["lift_vs_dummy"] >= cfg.get("min_lift_vs_dummy", 1.5))
        choices[p] = c
        o = oof_all[(oof_all.dataset == c["dataset"]) & (oof_all.model == c["model"]) & (oof_all.repeat == 0)
                    & (oof_all["product"] == p) & (oof_all.cv_mode == c["cv_mode"])]
        ci[p] = evaluate.bootstrap_ci(o, cfg["bootstrap"]["n"], cfg["seed"])
        log(f"  {p}: {c['model']} ({c['scenario']}, {c['cv_mode']} CV) {sel['metric']}={c['score']:.3f}, "
            f"기준선 대비 {c['lift_vs_dummy']:.2f}배")
        if not c["beats_chance"]:
            log(f"  ⚠ {p}: 최고 모델도 무작위 기준선의 {c['lift_vs_dummy']:.2f}배 → 공정변수만으로는 불량 예측이 거의 불가능")

    if run_exp:
        log("[6b/9] 추가 실험 (임계값·검사비율, 이전 베이스라인 비교, 성능 범위, holdout 설계, 가상 선택)")
        exp_tables.update(experiments.run_post_selection(
            frames, products, feats_by["separate"], metrics, oof_all, table, choices, cfg, audit, exp_extra,
            fold_check, ROOT, args.synthetic, log=log, extra_oof=exp_oof, types_by_product=types_by_product))
        for k, t in exp_tables.items():
            if isinstance(t, pd.DataFrame):
                t.to_csv(tab_dir / f"{k}.csv", index=False, encoding="utf-8-sig")
        for _, h in exp_tables.get("hypothetical_selection", pd.DataFrame()).iterrows():
            if h.changes_final_model:
                log(f"  ⚠ {h['product']}: 실험 모델까지 후보에 넣으면 같은 규칙으로 '{h.hypothetical_choice}' 가 선택됨 "
                    f"(현재 '{h.actual_choice}'). 적용하지 않고 보고만 함")

    # 비교 그림: 모델별 샷/블록 CV 나란히
    modes = cfg["cv"]["modes"]
    colors = {"shot": "#9db4d8", "block": "#2f5d9e"}
    fig, axes = plt.subplots(1, len(products), figsize=(6.2 * len(products), 0.42 * len(models) * 2 + 1.6),
                             squeeze=False)
    for ax, p in zip(axes[0], products):
        t = table[table["product"] == p].copy()
        t["label"] = t.model + " (" + t.scenario.str[:4] + ")"
        order = (t[t.cv_mode == sel["cv_mode"]].sort_values("shot_ap").label.tolist())
        ypos = {lab_: i for i, lab_ in enumerate(order)}
        h = 0.8 / len(modes)
        for j, mode in enumerate(modes):
            tm = t[t.cv_mode == mode].set_index("label").reindex(order)
            ax.barh(np.arange(len(order)) + (j - (len(modes) - 1) / 2) * h, tm.shot_ap, height=h,
                    xerr=tm.shot_ap_sd, color=colors.get(mode, "#888"), label=f"{mode} CV", error_kw={"lw": 0.8})
        ax.set_yticks(range(len(order)), order, fontsize=7)
        ax.set(title=f"{p.upper()} shot-level AP (mean ± sd)", xlabel="AP")
        ax.legend(fontsize=8, loc="lower right")
    savefig(fig, fig_dir / "model_comparison.png")

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    budgets = np.linspace(0.005, 0.3, 60)
    for i, p in enumerate(products):
        c = choices[p]
        for mode in modes:
            o = oof_all[(oof_all.dataset == c["dataset"]) & (oof_all.model == c["model"])
                        & (oof_all["product"] == p) & (oof_all.cv_mode == mode)]
            om = o.groupby("row").agg(y=("y", "first"), p=("p", "mean"))
            ax.plot(budgets * 100, [evaluate.recall_at_budget(om.y, om.p, b) for b in budgets],
                    color=f"C{i}", ls="-" if mode == sel["cv_mode"] else ":",
                    label=f"{p.upper()} {c['model']} ({mode} CV)")
    ax.plot(budgets * 100, budgets, "--", color="gray", label="random inspection")
    ax.set(xlabel="inspected share of production (%)", ylabel="defects caught (recall)",
           title="Inspection priority curve (cross-validated)")
    ax.legend(fontsize=7)
    savefig(fig, fig_dir / "inspection_curve.png")

    # 7. 불량 유형별 탐지율 -------------------------------------------------------
    log("[7/9] 불량 유형별 탐지율")
    type_tab = errors.type_table(oof_all, types_by_product, [0.05, 0.10])
    type_tab.to_csv(tab_dir / "defect_type_recall.csv", index=False, encoding="utf-8-sig")

    # 8. 오류분석 ---------------------------------------------------------------
    log("[8/9] 오류분석")
    err = {}
    for p, c in choices.items():
        lab, _ = frames[c["dataset"]]
        e = errors.analyze(p, c["dataset"], c["model"], c["cv_mode"], lab, feats_by[c["scenario"]], oof_all,
                           fitted[c["dataset"]], types_by_product[p], cfg, fig_dir)
        err[p] = e
        e[1].to_csv(tab_dir / f"importance_{p}.csv", encoding="utf-8-sig")
        e[2].to_csv(tab_dir / f"error_bins_{p}.csv", index=False, encoding="utf-8-sig")
        e[3].to_csv(tab_dir / f"fn_cases_{p}.csv", index=False, encoding="utf-8-sig")
        e[4].to_csv(tab_dir / f"interaction_{p}.csv", index=False, encoding="utf-8-sig")

    # 9. 최종 학습 + 예측 파일 -----------------------------------------------------
    log("[9/9] 최종 학습·확률 보정·예측 파일")
    pu, pl, pt, final_summ = [], [], [], []
    overlap = load_labeled_overlap(cfg, out, log)
    for p, c in choices.items():
        tf = test_frames.get(p)
        if tf is not None and c["scenario"] == "pooled":
            tf = tf.assign(is_rg3=int(p == "rg3"))
        u, lpred, tpred, s, _ = final.train_and_predict(p, c, frames, feats_by, oof_all, metrics, cfg, fig_dir,
                                                        types_by_product[p], tf, labeled_overlap=overlap.get(p),
                                                        log=log)
        pu.append(u), pl.append(lpred), final_summ.append(s)
        if tpred is not None:
            pt.append(tpred)
    # unlabeled 예측은 정답이 없으므로 평가 폴더(predictions/)와 분리
    unl_dir = out / "predictions_unlabeled"
    unl_dir.mkdir(parents=True, exist_ok=True)
    for legacy in (out / "predictions_unlabeled.csv", out / "predictions_unlabeled_meta.json"):
        if legacy.exists():  # 이전 버전 위치의 파일 (혼동 방지)
            legacy.unlink()
    pd.concat(pu, ignore_index=True).to_csv(unl_dir / "predictions_unlabeled.csv", index=False, encoding="utf-8-sig")
    unl_meta = {
        "file": "predictions_unlabeled.csv",
        "warnings": [
            "점수는 순위 용도: labeled 와 unlabeled 는 파일별로 따로 표준화돼(같은 z값 ≠ 같은 물리값) 점수·확률의 절대값은 "
            "학습 때와 눈금이 다를 수 있음. 제품(mold) 안의 순위(rank, risk_tier)로만 쓸 것.",
            "labeled 행 포함 가능성: 대회 unlabeled 안에 labeled 행이 들어 있음 (공개 원본 검증: " + known_facts.overlap_text()
            + "). unlabeled 예측을 성능 평가에 쓰거나 unlabeled 로 학습(이상탐지·pseudo-label)하지 말 것.",
            "in_labeled_train_row / in_labeled_train_shot=True 인 행은 그 행 자체 또는 같은 shot 의 짝이 최종 모델·fold 모델의 "
            "학습에 쓰였음 → uncertainty 가 과소평가될 수 있음.",
        ],
        "columns": {
            "mold": "제품 (cn7 / rg3)",
            "row_id": "대회 unlabeled 파일의 Unnamed: 0",
            "score": "최종 모델의 불량 의심 점수 (클수록 의심, 순위 용도)",
            "rank": "제품 안에서의 점수 순위 (1 = 가장 의심)",
            "risk_tier": "제품 안 순위 기준: " + " / ".join(f"상위 {k:.0%}" for k in sorted(cfg["risk_tiers"])) + " / 그 외",
            "uncertainty": "선택된 CV 방식의 fold 모델 전체 예측의 표준편차 (점수 단위, ddof=1)",
            "in_labeled_range": "모든 공정변수가 labeled(학습) 데이터의 최소~최대 범위 안이면 True (False = 학습 범위 밖, 신뢰도 낮음)",
            "in_labeled_train_row": "이 행 자체가 labeled(학습)에 있음 (공개 원본 검증, external 꺼짐이면 NaN)",
            "in_labeled_train_shot": "이 행 또는 같은 shot 의 짝이 labeled(학습)에 있음 (공개 원본 검증, external 꺼짐이면 NaN)",
        },
        "in_labeled_train_criterion": known_facts.OVERLAP_CRITERION,
        "in_labeled_train": {p: ("공개 원본 검증 결과(outputs/verification/unlabeled_overlap_flags.csv)로 채움"
                                 if overlap.get(p) is not None
                                 else "NaN — 대회 파일만으로 판정 불가 (external.enabled=false 또는 검증 결과 없음)")
                             for p in products},
        "uncertainty_n_models": {s_["product"]: s_["uncertainty_fold_models"] for s_ in final_summ},
        "models": {s_["product"]: {"model": s_["model"], "cv_mode": s_["cv_mode"],
                                   "share_above_cv_decision_threshold": s_["unlabeled_share_above_cv_threshold"],
                                   "in_labeled_range_share": s_["unlabeled_in_labeled_range_share"]}
                   for s_ in final_summ},
        "source": known_facts.SOURCE,
    }
    (unl_dir / "predictions_unlabeled_meta.json").write_text(json.dumps(unl_meta, ensure_ascii=False, indent=1,
                                                                    default=str), encoding="utf-8")
    pd.concat(pl, ignore_index=True).to_csv(out / "predictions_labeled_cv.csv", index=False, encoding="utf-8-sig")
    if pt:
        pd.concat(pt, ignore_index=True).to_csv(out / "predictions_holdout.csv", index=False, encoding="utf-8-sig")
    oof_all.to_csv(tab_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    write_pipeline_predictions(oof_all, models, out, log)

    manifest = {
        "run_name": cfg["run_name"], "command": run_command(args),
        "git_commit": git_commit(),
        "versions": library_versions(),
        "seed": cfg["seed"], "input_sha256": data["hashes"],
        "cv": cfg["cv"], "selection": sel, "holdout": hcfg,
        "config": cfg,  # 실행에 쓴 설정 전체 (--fast 등으로 바뀐 값 포함)
        "runtime_sec": round(time.time() - t_start, 1),
    }
    if sep_frames:
        figs = key_figures.make_all(fig_dir, {p: frames[p] for p in sep_frames}, types_by_product, fold_id,
                                    oof_all, choices, exp_tables, ablation)
        log(f"  핵심 그림 {len(figs)}개 → {rel(fig_dir)}")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    report.write(out / "report.md", cfg, audit, dropped, table, ci, choices, final_summ, err, type_tab,
                 manifest, holdout_info, fold_check=fold_check, ablation=ablation, exp=exp_tables)
    log(f"완료: {rel(out / 'report.md')}  ({manifest['runtime_sec']}s)")


if __name__ == "__main__":
    main()
