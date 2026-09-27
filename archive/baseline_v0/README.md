# 사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정 — Baseline

2026 제6회 K-인공지능 제조데이터 분석 경진대회 · 사출성형기 AI 데이터셋 (moldset CN7 / RG3)

이 저장소는 **"신뢰할 수 있고 재현 가능한 베이스라인"** 입니다. 최고 성능을 겨루는 단계가 아니라,
데이터의 구조를 확인하고, 데이터 누수 없이 여러 모델을 같은 조건에서 비교하고,
후속 분석(오류 분석·SHAP·검사 우선순위)에 쓸 파일을 남기는 것이 목적입니다.

---

## 1. 폴더 구조

```
project/
├── data/                         # 원본 CSV 4개 (대회 제공, 저장소에는 포함하지 않음 → 직접 넣기)
│   ├── moldset_labeled_cn7.csv     (1211, 26)
│   ├── moldset_labeled_rg3.csv     (1182, 26)
│   ├── moldset_unlabeled_cn7.csv   (35239, 25)
│   └── moldset_unlabeled_rg3.csv   (35941, 25)
├── outputs/                      # 실행 결과 (모두 자동 생성)
│   ├── eda/                        데이터 진단·분포 분석 표(csv)와 그래프(png)
│   ├── models/                     최종 모델(joblib) + 메타(json) + feature importance
│   ├── errors/                     validation / OOF 오류 분석용 데이터 (FN/FP 조건 분석에 사용)
│   ├── thresholds/                 threshold 변화에 따른 trade-off 표·그래프
│   ├── predictions/                unlabeled 예측 결과 (prediction_cn7.csv, prediction_rg3.csv)
│   ├── model_comparison.csv        실험 A/B × 모델 × class_weight 비교표 (random holdout, 명세 기본)
│   ├── model_comparison_all_validations.csv   검증 방식별(random/group/ordered/CV) 전체 결과
│   ├── cv_group5fold_comparison.csv           group-aware 5-fold CV(OOF) 결과 (최종 선정 근거)
│   └── run_log.txt                 python baseline.py 콘솔 출력 전체
├── baseline.ipynb                # 분석·시각화·설명 중심 (baseline.py 의 함수를 순서대로 호출)
├── baseline.py                   # 처음부터 끝까지 자동 실행되는 버전 (함수 단위 구성)
├── tools/make_notebook.py        # 노트북 생성 스크립트 (개발용)
├── requirements.txt
└── README.md
```

## 2. 실행 방법

```bash
pip install -r requirements.txt
# data/ 폴더에 CSV 4개를 넣은 뒤
python baseline.py            # 전체 파이프라인 자동 실행 (약 2~3분), 결과는 outputs/ 에 저장
jupyter notebook baseline.ipynb   # 설명과 그래프를 보며 단계별로 실행
```

- 모든 random seed 는 `42` 로 고정되어 있어 **스크립트와 노트북의 결과가 완전히 동일**합니다 (md5 로 확인).
- 외부 부스팅 패키지(XGBoost/LightGBM)가 없어도 실행됩니다 (sklearn `HistGradientBoostingClassifier` 사용).
- 설정값(경로, seed, threshold, 검증 방식, 최종 모델 수동 지정 등)은 `baseline.py` 상단 `0. 설정값` 에 모여 있습니다.

## 3. 데이터에서 확인한 사실 (가장 중요한 부분)

| # | 확인 사실 | 근거 (outputs/eda/) | 영향 |
|---|---|---|---|
| 1 | **4개 파일이 각각 따로 z-score 표준화**되어 있음 (모든 공정변수 평균 0, 표준편차 1) | `scaling_check.csv` | CN7 vs RG3 평균 차이는 0 으로 "만들어져" 있어 모양(KS)만 비교 가능. labeled/unlabeled 도 따로 표준화 → unlabeled 예측은 **순위 용도로만** 해석 |
| 2 | labeled 의 **50% 가 공정변수 완전 중복**이며, 거의 전부 "인접 2행 쌍" 구조. 불량 42개 중 **36개는 같은 공정값의 정상 행과 쌍** (CN7 11/17, RG3 25/25) | `duplicate_structure.csv` | 같은 입력 → 같은 예측이므로 성능 상한 존재: FP 0 으로 잡을 수 있는 최대 Recall = CN7 0.353 / RG3 0.000. 무작위 split 은 쌍을 양쪽에 나눠 넣어 누수(validation 행의 73~84% 가 train 에 복사본) |
| 3 | `Unnamed: 0` 은 연속 행 번호지만 인접 행 자기상관 0.7~0.8(섞으면 0) → **기록(시간) 순서 보존**. **CN7 불량 17개는 모두 ID 47~120 구간**에 집중 | `id_column_analysis.csv`, `positives_along_index.png` | feature 로 쓰지 않고 ID 로 보존. CN7 은 시간순 holdout 에 불량이 0개 → 시간순 검증 불가 |
| 4 | CN7 vs RG3: KS 최대 0.60, RF 도메인 분류기 AUC **1.000** | `domain_comparison_cn7_vs_rg3.csv`, `domain_classifier_auc.csv` | 같은 분포로 볼 수 없음 → 실험 A(제품별 독립)가 기본, 병합(B)은 이득이 분명할 때만 |
| 5 | `Clamp_Open_Position` 은 labeled 에서 상수(표준편차 0) | `inspection_summary.csv` | feature 에서 제외 (23개 사용) |
| 6 | 정상/불량 비율: CN7 1194/17 (1.40%), RG3 1157/25 (2.12%) | `target_distribution.csv` | 심한 불균형 → Accuracy 로 모델 선택 금지, holdout 양성 3~5개라 보조로 5-fold CV |
| 7 | RG3 는 정상/불량 간 어떤 변수도 유의한 차이 없음 (KS p > 0.3) | `class_conditional_RG3.csv` | 현재 feature 로는 RG3 불량 신호 없음 (모델 결과와 일치) |

**클래스 의미**: `PassOrFail` 0/1 중 **1 을 불량으로 가정** (`POSITIVE_CLASS = 1`). 근거는 1 이 소수 클래스(1.4%, 2.1%)로 일반적인 불량률 수준과 부합한다는 것뿐이며, **공식 데이터 설명 확인이 필요**합니다(TODO). 가설: 1 shot 에서 부품 2개(LH/RH)가 나오고 부품별로 검사 결과가 붙은 구조 → 원본 메타데이터(PART_NAME 등)로 확인 필요.

## 4. 실험 설계

- **실험 A**: CN7, RG3 각각 독립적으로 학습/평가
- **실험 B**: CN7+RG3 병합 + `source` 컬럼(0/1 인코딩). 전체 성능과 함께 제품별 성능(`CN7+RG3[CN7]`, `CN7+RG3[RG3]`)을 따로 기록해 A 와 직접 비교
  - `source` 누수 점검: 제품별 불량률 차이 0.7%p(약한 사전확률), 트리 모델은 source 없이도 제품을 AUC 1.0 으로 구분 → 새로운 누수를 만들지 않음
- **모델** (모두 `sklearn Pipeline`, 전처리는 train 조각에만 fit): Dummy(most_frequent, stratified) / LogisticRegression(StandardScaler, max_iter=5000) / RandomForest(n_estimators=300) / HistGradientBoosting — 각각 기본 vs `class_weight='balanced'`(RF 는 `balanced_subsample`). 튜닝·SMOTE 없음.
- **검증 방식** (`split_data`):

| 방식 | 내용 | 한계 |
|---|---|---|
| `random_holdout` (기본, 명세) | 층화 무작위 20%, random_state=42 | 중복 쌍 누수(73~84%), 양성 3~5개 |
| `group_holdout` | 공정값이 같은 쌍은 같은 쪽에 (StratifiedGroupKFold 첫 fold) | 양성 4~8개 |
| `ordered_holdout` | 제품별 ID 뒤 20% (시간순 근사) | CN7 validation 양성 0개 → 해석 불가 |
| `cv_group5fold` (보조) | 쌍 단위 층화 5-fold, 모든 행의 OOF 확률 | 최종 선정 근거로 사용, 튜닝에는 미사용 |

- **지표**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, TN/FP/FN/TP (threshold 0.5). Recall 과 FN 을 별도 강조.

## 5. 결과 요약 (group-aware 5-fold OOF 기준)

**CN7** (양성 17개)

| 실험 | 모델 | class_weight | Precision | Recall | F1 | ROC-AUC | PR-AUC | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| A | LogisticRegression | None | 0.750 | 0.353 | 0.480 | 0.937 | **0.476** | 11 | 2 |
| B | LogisticRegression | None | 1.000 | 0.353 | 0.522 | 0.935 | 0.472 | 11 | 0 |
| A | HistGradientBoosting | None | 0.667 | 0.235 | 0.348 | 0.962 | 0.459 | 13 | 2 |
| A | RandomForest | None | 1.000 | 0.353 | 0.522 | 0.938 | 0.423 | 11 | 0 |
| A | LogisticRegression | balanced | 0.145 | 0.529 | 0.228 | 0.932 | 0.441 | 8 | 53 |
| A | Dummy | most_frequent | 0 | 0 | 0 | 0.500 | 0.014 | 17 | 0 |

**RG3** (양성 25개): 모든 모델이 ROC-AUC 0.39~0.51, PR-AUC 0.017~0.025 (불량률 0.021) → **Dummy 수준, 신호 없음**.

- Recall 0.353 = 6/17 은 데이터 구조상 "FP 0 으로 잡을 수 있는 최대 Recall" 과 정확히 같습니다. 모델은 구분 가능한 불량은 모두 잡았고, 정상 쌍둥이가 있는 11개는 어떤 모델도 잡을 수 없습니다 (`outputs/errors/A_CN7_LogisticRegression_None_cv_group5fold_oof_errors.csv` 에서 FN 행과 같은 `dup_group` 의 TN 행 확인).
- `class_weight='balanced'` 는 Recall 을 올리는 대신 정상 쌍둥이까지 잡아 FP 가 크게 늘어납니다. 검사 여력이 충분하면 의미가 있으나 baseline 기본값으로는 채택하지 않았습니다.
- random holdout 표(`model_comparison.csv`)는 명세대로 기본 표로 남기되, 양성 3~5개와 중복 쌍 누수 때문에 **모델 우열 판단에는 사용하지 않습니다**.

## 6. 최종 baseline 선정

선정 규칙(`select_final_models`): group 5-fold OOF 에서 실험 A 후보 중 **PR-AUC 최상위군(0.02 이내)** → F1 → Recall → 단순한 모델 순. 실험 B 는 PR-AUC 를 0.02 넘게 개선할 때만 채택. 모든 모델이 Dummy 수준이면 "신호 없음" 표시 + 가장 단순한 비가중 모델을 자리표시자로 사용.

| 제품 | 선정 | 이유 |
|---|---|---|
| CN7 | **실험 A / LogisticRegression / class_weight=None** | PR-AUC 최고(0.476), ROC-AUC 0.94, 가장 단순. RF/HGB 도 동급이며 B 는 이득 없음 |
| RG3 | 실험 A / LogisticRegression / None (자리표시자) | 신호 없음. 예측 파일은 형식상 생성, 해석 금지 |

**A vs B 결론**: 두 제품은 분포가 완전히 다르고(도메인 분류기 AUC 1.0) 병합해도 제품별 성능이 나아지지 않으므로, **제품별 독립 모델(실험 A)이 타당한 baseline 구성**입니다.

## 7. Threshold trade-off (CN7 선정 모델, OOF)

| threshold | Precision | Recall | F1 | FN | FP | 검사 대상 수 |
|---|---|---|---|---|---|---|
| 0.1 | 0.267 | 0.471 | 0.340 | 9 | 22 | 30 |
| 0.3 | 0.750 | 0.353 | 0.480 | 11 | 2 | 8 |
| 0.5 | 0.750 | 0.353 | 0.480 | 11 | 2 | 8 |
| 0.7 | 1.000 | 0.353 | 0.522 | 11 | 0 | 6 |

threshold 를 낮추면 FN 이 줄고 FP 가 늘어납니다. 이 표는 trade-off 이해용이며, 운영 threshold 는 별도 검증(nested CV)으로 정해야 합니다. (`outputs/thresholds/`)

## 8. unlabeled 예측 파일

`outputs/predictions/prediction_cn7.csv`, `prediction_rg3.csv`: `original_index, source, predicted_label(threshold 0.5), defect_probability, inspection_rank`
(`prediction_metadata.json` 에 사용 모델·threshold 기록)

- `inspection_rank` 는 불량 확률 내림차순 1부터의 **baseline ranking** 일 뿐, 최종 검사 우선순위 알고리즘이 아닙니다.
- **주의**: labeled/unlabeled 가 따로 표준화되어 있어 CN7 의 unlabeled 예측 불량률(4.8%)이 labeled 불량률(1.4%)보다 크게 높고, unlabeled 행의 61% 가 학습 범위 밖 값을 포함합니다(`unlabeled_prediction_sanity.csv`). 원본 단위 확인 전까지 확률의 절대값은 신뢰하지 말고 순위만 참고하세요.

## 9. 누수 방지 체크리스트 (명세의 금지사항)

- [x] 전처리(imputer/scaler)는 Pipeline 안에서 train 조각에만 fit — test/unlabeled 로 기준 학습 안 함
- [x] 전체 데이터 scaling 후 split 안 함 (원본이 이미 표준화되어 있다는 점은 별도로 명시)
- [x] validation 으로 hyperparameter 반복 튜닝 안 함 / threshold 최적화 결과를 최종 성능으로 보고 안 함
- [x] `Unnamed: 0` 은 의미 분석 후 ID 로만 보존, feature 미사용
- [x] CN7/RG3 를 이유 없이 병합하지 않음 (A 기본, B 는 비교 실험)
- [x] Accuracy 로 모델 선정 안 함 (PR-AUC/F1/Recall/FN/복잡도 종합)
- [x] 클래스 의미 단정 안 함 (`POSITIVE_CLASS = 1` 가정 + TODO 주석)
- [x] feature importance 를 인과관계로 표현하지 않음

## 10. 한계와 다음 단계

1. **문제 정의**: 행 단위 라벨은 같은 feature 에 상반된 라벨이 붙어 성능 상한이 있음 → shot(쌍) 단위 any-defect 라벨로 재정의 검토
2. **스케일**: 원본 단위 데이터 또는 스케일링 파라미터 확보 → labeled/unlabeled 불일치 해소
3. **클래스 의미**: `PassOrFail` 0/1 공식 확인
4. **오류 분석**: `outputs/errors/*_oof_errors.csv` 로 FN/FP 가 집중되는 공정조건 분석 + SHAP (`outputs/models/final_model_*.joblib` 사용)
5. **threshold**: nested CV 로 결정, 검사 여력 대비 Recall 우선 운영점 설계
6. **CN7 검증**: 불량이 한 구간에 집중되어 시간순 검증 불가 → 추가 기간 데이터 필요
7. **불균형/튜닝**: SMOTE·cost-sensitive·간단한 튜닝은 group-aware CV 안에서만 수행
