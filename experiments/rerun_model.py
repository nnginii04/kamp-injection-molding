"""E/D 실험·Budget Curve·캐비티 규칙을 지정한 모델로 다시 돌리고 비교표를 만든다.
사용 (레포 루트에서):
  python experiments/rerun_model.py "<공개 원본 moldset_labeled.csv 경로>"            # logreg, logreg_spc 둘 다
  python experiments/rerun_model.py "<경로>" logreg_spc                                # 하나만
결과: outputs/experiments/<모델>/ (csv·png·run_log.txt), outputs/experiments/compare_logreg_vs_logreg_spc.md
제외(모델과 무관하거나 logreg 전용): e2_rolling_spc(관리도), e3_nested_tuning(logreg C 튜닝),
      e1_null_test(순열 200회, logreg 기준 결론 유지), direction_table(logreg 계수 해석)."""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
assert len(sys.argv) > 1 and os.path.exists(sys.argv[1]), "첫 인자로 공개 원본 moldset_labeled.csv 경로를 주세요"
PUB = os.path.abspath(sys.argv[1])
MODELS = sys.argv[2:] or ["logreg", "logreg_spc"]
SCRIPTS = ["check_pipeline_match.py", "e1_sequence.py", "e4_shot_level.py", "d1_d3_domain.py", "d_e1c_dayholdout.py",
           "budget_curve.py", "budget_plot.py", "cavity_rules.py", "cavity_plot.py"]
fails = []
for m in MODELS:
    out = os.path.join(ROOT, "outputs", "experiments", m); os.makedirs(out, exist_ok=True)
    env = {**os.environ, "KAMP_MODEL": m, "KAMP_PUBLIC_CSV": PUB, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    with open(os.path.join(out, "run_log.txt"), "w", encoding="utf-8") as log:
        for s in SCRIPTS:
            if s == "check_pipeline_match.py" and m == "logreg":
                continue
            t = time.time(); print(f"[{m}] {s} ...", end=" ", flush=True)
            log.write(f"\n===== {s} =====\n"); log.flush()
            r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            ok = r.returncode == 0
            print(("완료" if ok else "실패") + f" ({time.time() - t:.0f}초)", flush=True)
            if not ok:
                fails.append(f"{m}/{s}")
if len(MODELS) == 2:
    subprocess.run([sys.executable, os.path.join(HERE, "compare_models.py"), *MODELS], cwd=ROOT)
print("실패:", fails if fails else "없음", "| 로그: outputs/experiments/<모델>/run_log.txt")
