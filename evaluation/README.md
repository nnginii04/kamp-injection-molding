# 팀 공통 평가 모듈 (`evaluation/`)

모델마다 평가 방식이 다르면 결과를 비교할 수 없습니다. 그래서 **모든 모델의 예측을 같은 형식의 CSV로 저장하고, 같은 함수로 채점합니다.**

```
outputs/predictions/*.csv  ──▶  python run_evaluation.py  ──▶  outputs/evaluation/experiment_summary.csv (+ .md)
```

## 1. 예측 파일 형식 (A·B에게 공유)

| 열 | 필수 | 뜻 |
|---|---|---|
| `y_true` | ✅ | 정답. 1 = 불량, 0 = 양품 |
| `y_score` | ✅ | 불량 의심 점수. 클수록 불량 가능성이 높음 (확률이 아니어도 됨) |
| `y_pred` | | 모델이 내린 판정 (0/1). 없으면 점수 ≥ 0.5로 판정 |
| `row_id` | 권장 | 대회 파일의 `Unnamed: 0`. 있으면 LH/RH 짝(shot_id)을 자동으로 붙여 **샷 단위**도 평가 |
| `product` | 권장 | `cn7` / `rg3` |
| `model` | | 모델 이름 (없으면 파일 이름을 씀) |
| `repeat` | | 교차검증 반복 번호. 있으면 **반복별 평균±편차** |
| `fold` | | fold 번호. `repeat`가 없으면 fold별 평균±편차 |
| `shot_id` | | LH/RH 짝 번호 (없어도 `row_id`로 자동 연결) |

예시: [`example_predictions.csv`](example_predictions.csv)

```csv
row_id,product,model,repeat,fold,shot_id,y_true,y_score,y_pred
1,cn7,baseline_logreg,0,1,cn7_1,0,0.0026,0
2,cn7,baseline_logreg,0,1,cn7_1,0,0.0026,0     ← 1·2번 행은 LH/RH 짝: 점수가 같고 같은 fold
```

열 이름이 조금 달라도(`y`, `label`, `p`, `prob`, `pred` …) 자동으로 맞춥니다.

### 예측을 만들 때 꼭 지킬 것
1. **교차검증 예측(OOF)으로 저장.** 각 행의 점수는 그 행을 학습에 쓰지 않은 모델이 낸 값이어야 합니다. 학습 데이터에 그대로 예측하면 점수가 부풀려집니다.
2. **LH/RH 짝은 같은 fold에.** 두 행은 공정값이 똑같아서, 짝이 학습과 검증으로 갈라지면 정답을 미리 본 셈이 됩니다. `StratifiedGroupKFold(groups=shot_id)`를 쓰세요. 갈라진 경우 평가표의 '확인 필요'에 경고가 뜹니다.
3. 가능하면 **반복 교차검증**(예: 3-fold × 5회)으로 `repeat` 열을 넣어 주세요. CN7은 fold 하나에 불량이 5~6개뿐이라 fold별 값은 크게 흔들립니다.

## 2. 실행

VS Code에서 `run_evaluation.py`를 열고 실행 버튼. 또는:

```bash
python run_evaluation.py                   # outputs/predictions/*.csv + run_all.py 결과(있으면)
python run_evaluation.py --budget 0.02     # F1·FN·FP를 '상위 2% 검사' 기준으로
python run_evaluation.py --threshold 0.3   # F1·FN·FP를 '점수 ≥ 0.3' 기준으로
python run_evaluation.py --no-pipeline     # 팀 예측 파일만
```

`outputs/tables/oof_predictions.csv`(run_all.py 결과)가 있으면 같이 평가합니다. 이 숫자가 `outputs/report.md`와 같게 나오면 평가 함수가 맞다는 검산이 됩니다.

## 3. 결과 파일 (`outputs/evaluation/`)

| 파일 | 내용 |
|---|---|
| `experiment_summary.csv` | 파일 × 모델 × 제품 × (cv_mode) × 단위(행/샷)마다 한 줄. PR-AUC·ROC-AUC·F1·Precision·Recall 평균±편차, FN·FP, Recall@5/10/20/30%, 무작위 대비 배수 |
| `experiment_summary.md` | 보고서에 붙일 요약표 + 확인 필요 사항 |
| `recall_at_k_detail.csv` | K별 검사 개수, 잡은/놓친 불량 수, 무작위 기준선, (샷 단위) 불량 유형별 재현율 |
| `threshold_detail.csv` | 임계값 0.05~0.95, 상위 1/2/5/10% 기준별 F1·Recall·Precision·FN·FP·찍은 비율 |
| `evaluation_manifest.json` | 입력 파일 SHA-256·행 수, 설정, 라이브러리 버전 (재현성) |

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

- **무작위 기준선**: PR-AUC ≈ 불량률(CN7 약 1.4%, RG3 약 2.1%). Recall@K ≈ K. 표의 '배수'가 1에 가까우면 무작위와 같습니다.
- **PR-AUC**는 average precision으로 계산합니다. run_all.py의 `ap`와 같은 값입니다.
- **행 vs 샷**: 짝 두 행은 점수가 같아서, 한쪽만 불량이면 행 단위 F1에는 넘을 수 없는 상한이 생깁니다. 샷 단위를 같이 보세요.
- **임계값 표**: 불량이 드물어 확률 점수는 대부분 0.05 아래입니다. 그래서 고정 임계값 대부분에서 `flag_rate`가 0이고 Recall이 0입니다. 이건 정상입니다. '상위 k%' 줄을 같이 보세요.
- **best_f1**은 채점한 데이터로 고른 임계값이라 낙관적입니다(참고용). 최종 판정 규칙은 교차검증 안에서 정합니다(run_all.py: 교차검증 점수 상위 2%).
- **동점 처리**: Recall@K는 동점을 무작위로 끊었을 때의 기댓값입니다. 파일 순서로 자르면 불량이 앞쪽에 몰린 CN7에서 부풀려집니다.
- **FN·FP**: repeat 단위면 전체 1회분의 평균, fold 단위면 fold 합계입니다.
