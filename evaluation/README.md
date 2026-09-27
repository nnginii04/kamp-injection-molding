# 팀 공통 평가 모듈 (`evaluation/`)

모델마다 평가 방식이 다르면 결과를 비교할 수 없습니다. 그래서 **모든 모델의 예측을 같은 형식의 CSV로 저장하고, 같은 함수로 채점합니다.**

```
outputs/predictions/*.csv  ──▶  python run_evaluation.py  ──▶  outputs/evaluation/experiment_summary.csv (+ .md)
```

## 1. 예측 파일 형식 (팀 공통)

한 줄 = 한 행(제품 하나)의 예측 하나. 교차검증을 여러 번 반복했다면 반복마다 줄을 따로 씁니다.

| 열 | 필수 | 뜻 |
|---|---|---|
| `y_true` | ✅ | 정답. 1 = 불량, 0 = 양품. **이 열이 없는 파일은 건너뜁니다** (정답이 없는 unlabeled 예측은 평가 대상이 아님) |
| `y_score` | ✅ | 불량 의심 점수. 클수록 불량 가능성이 높음 (확률이 아니어도 됨) |
| `product` | 권장 | `cn7` / `rg3` (`mold` 라고 써도 같게 인식) |
| `model` | 권장 | 모델 이름 (없으면 파일 이름을 씀) |
| `cv_mode` | 권장 | `shot` 또는 `block`. **없으면 경고와 함께 별도 표로 분리**됩니다 (검증 방식이 다른 수치는 비교하지 않기 위해) |
| `repeat` | 권장 | 교차검증 반복 번호. 있으면 **반복별 평균±편차** |
| `fold` | 권장 | fold 번호 (`outputs/fold_id.csv` 의 번호를 쓰면 run_all.py 와 같은 분할) |
| `row_id` | 권장 | 대회 파일의 `Unnamed: 0`. 이것으로 **전역 shot_id** 를 붙여 샷 단위도 평가 |
| `y_pred` | 선택 | 모델이 내린 판정 (0/1) |
| `shot_id` | 무시 | 넣어도 쓰지 않습니다. `row_id` 로 run_all.py 의 전역 shot_id(공정값이 완전히 같은 행 = 한 shot)를 항상 덮어씁니다. 파일의 묶음이 다르면(예: 연속 행만 묶기) 경고가 뜹니다 |

run_all.py 는 모든 모델의 교차검증 예측을 이 형식으로 `outputs/predictions/pipeline_<모델>.csv` 에 저장합니다
(열 순서: `mold, product, model, cv_mode, repeat, fold, row_id, shot_id, y_true, y_score, y_pred`).
팀원 파일은 같은 폴더에 **`pipeline_` 로 시작하지 않는 이름**으로 넣으세요 (run_all.py 가 `pipeline_*.csv` 만 새로 씁니다).

예시: [`example_predictions.csv`](example_predictions.csv)

```csv
row_id,product,model,repeat,fold,shot_id,y_true,y_score,y_pred
1,cn7,baseline_logreg,0,1,cn7_1,0,0.0026,0
2,cn7,baseline_logreg,0,1,cn7_1,0,0.0026,0     ← 1·2번 행은 LH/RH 짝: 점수가 같고 같은 fold
```

열 이름이 조금 달라도(`y`, `label`, `p`, `prob`, `pred` …) 자동으로 맞춥니다.

### 예측을 만들 때 꼭 지킬 것
1. **교차검증 예측(OOF)으로 저장.** 각 행의 점수는 그 행을 학습에 쓰지 않은 모델이 낸 값이어야 합니다. 학습 데이터에 그대로 예측하면 점수가 부풀려집니다.
2. **LH/RH 짝은 같은 fold에.** 가장 쉬운 방법은 `outputs/fold_id.csv` 의 fold 번호를 그대로 쓰는 것입니다. 갈라진 경우 평가표의 '확인 필요'에 경고가 뜹니다.
3. 가능하면 **반복 교차검증**(예: 3-fold × 5회)으로 `repeat` 열을 넣어 주세요. CN7은 fold 하나에 불량이 몇 개뿐이라 fold별 값은 크게 흔들립니다.
4. **unlabeled 예측은 여기에 넣지 마세요.** 정답이 없고, 대회 unlabeled 안에는 labeled 행이 섞여 있어 평가에 쓰면 안 됩니다 (`outputs/predictions_unlabeled/` 에 따로 둠).

## 2. 실행

VS Code에서 `run_evaluation.py`를 열고 실행 버튼. 또는:

```bash
python run_evaluation.py                   # outputs/predictions/*.csv (run_all.py 결과 + 팀 예측)
python run_evaluation.py --budget 0.02     # F1·FN·FP를 '상위 2% 검사' 기준으로
python run_evaluation.py --threshold 0.3   # F1·FN·FP를 '점수 ≥ 0.3' 기준으로
python run_evaluation.py --no-pipeline     # 팀 예측 파일만 (pipeline_*.csv 제외)
```

- **판정 기준(F1·FN·FP 계산용)**: `--threshold` 와 `--budget` 은 **둘 중 하나만** 쓸 수 있습니다.
  **둘 다 없으면 기본값 = 파일의 `y_pred`(모델이 낸 판정)**, `y_pred` 도 없으면 점수 ≥ 0.5. 실행할 때 어떤 기준을 썼는지 맨 위에 출력합니다.
  (run_all.py 의 `y_pred` 는 학습 fold 내부에서 정한 F1 최적 임계값 기준이라, 기본값으로 돌리면 report.md 의 F1 과 같은 기준)
- 경로 옵션(`--outputs`, `--pred-dir`, `--out-dir`)의 상대경로는 **저장소 루트 기준**입니다 (어느 폴더에서 실행해도 같음).
- 결과표는 **`cv_mode` 별로 나눠서** 출력·저장합니다. shot CV(낙관적)와 block CV(보수적) 수치를 한 줄에서 비교하지 않기 위해서입니다.
- **검산**: `outputs/tables/cv_summary.csv`(report.md 수치의 원본)가 있으면 `pipeline_*.csv` 평가 결과와 비교해
  `outputs/evaluation/crosscheck.csv` 에 저장하고 최대 차이를 출력합니다. 차이가 있으면 종료코드 1.

## 3. 결과 파일 (`outputs/evaluation/`)

| 파일 | 내용 |
|---|---|
| `experiment_summary.csv` | 파일 × 모델 × 제품 × (cv_mode) × 단위(행/샷)마다 한 줄. PR-AUC·ROC-AUC·F1·Precision·Recall 평균±편차, FN·FP, Recall@5/10/20/30%, 무작위 대비 배수 |
| `experiment_summary.md` | 보고서에 붙일 요약표 + 확인 필요 사항 |
| `recall_at_k_detail.csv` | K별 검사 개수, 잡은/놓친 불량 수, 무작위 기준선, (샷 단위) 불량 유형별 재현율 |
| `threshold_detail.csv` | 임계값 0.05~0.95, 상위 1/2/5/10% 기준별 F1·Recall·Precision·FN·FP·찍은 비율 |
| `evaluation_manifest.json` | 입력 파일 SHA-256·행 수, 설정, 건너뛴 파일, 라이브러리 버전 (재현성) |
| `crosscheck.csv` | run_all.py 결과(`pipeline_*.csv`) 평가값 vs report.md 수치(`outputs/tables/cv_summary.csv`) 비교 |

## 4. 함수로 직접 쓰기

```python
from evaluation import evaluate_model, calculate_recall_at_k, evaluate_thresholds, save_experiment_result

evaluate_model("outputs/predictions/baseline_logreg.csv")                 # dict: pr_auc_mean, pr_auc_std, f1_mean, fn_mean ...
evaluate_model(y_true, y_score, budget=0.02)                               # 배열도 가능
calculate_recall_at_k(df, ks=(0.05, 0.10, 0.20, 0.30), level="shot")       # 표
evaluate_thresholds(df)                                                    # 표 (rule = threshold / budget)
save_experiment_result("outputs/predictions", "outputs/evaluation")        # 전체 → 파일
```

## 5. 읽는 법 (보고서 쓸 때 주의)

- **무작위 기준선**: PR-AUC ≈ 불량률(CN7 약 1.4%, RG3 약 2.1%). Recall@K ≈ K. 표의 '배수'가 1에 가까우면 무작위와 같습니다. 배수를 계산할 수 없으면(불량 0 등) `-`.
- **PR-AUC**는 average precision으로 계산합니다. run_all.py의 `ap`와 같은 값입니다.
- **행 vs 샷**: 짝 두 행은 점수가 같아서, 한쪽만 불량이면 행 단위 F1에는 넘을 수 없는 상한이 생깁니다. 샷 단위를 같이 보세요.
- **임계값 표**: 점수의 범위는 모델마다 다릅니다 (가중치 없는 모델은 대부분 0.05 아래, class_weight='balanced' 모델은 훨씬 큼). 그래서 같은 고정 임계값이라도 모델마다 `flag_rate` 가 크게 다릅니다. 모델끼리 비교할 때는 '상위 k%' 줄을 보세요.
- **best_f1**은 채점한 데이터로 고른 임계값이라 낙관적입니다(참고용). 최종 판정 규칙은 교차검증 안에서 정합니다(run_all.py: 교차검증 점수 상위 2%).
- **동점 처리**: Recall@K는 동점을 무작위로 끊었을 때의 기댓값입니다. 파일 순서로 자르면 불량이 앞쪽에 몰린 CN7에서 부풀려집니다.
- **FN·FP**: repeat 단위면 전체 1회분의 평균, fold 단위면 fold 합계입니다.
