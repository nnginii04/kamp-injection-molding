# kamp-injection-molding

제6회 K-인공지능 제조데이터 분석 경진대회 — 사출성형 불량 예측 (팀 공정한 사람들)

## 0. 처음 한 번 (전원)

```bash
git clone <repo>
cd kamp-injection-molding
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

KAMP에서 받은 CSV 4개를 `data/raw/`에 넣는다 (git에는 올리지 않는다).

```
data/raw/moldset_labeled_cn7.csv
data/raw/moldset_labeled_rg3.csv
data/raw/moldset_unlabeled_cn7.csv
data/raw/moldset_unlabeled_rg3.csv
```

```bash
python run_all.py          # 전처리 → 분할 → 베이스라인 학습 → 점검 지표 (약 1~2분)
```

끝나면 `data/processed/`, `outputs/predictions/`, `outputs/model/`, `outputs/evaluation/`가 채워진다.

## 1. 폴더 구조

```
configs/config.yaml         시드·경로·분할 설정·모델 하이퍼파라미터 (숫자는 여기서만 바꾼다)
run_all.py                  전체 파이프라인 (재현성 제출용)
src/
  data/
    loading.py              CSV 4개 적재, mold 열 부여, row_id 부여
    preprocess.py           상수열 제거, shot_id·cavity_pos 부여, mold_cn7, is_idle(unlabeled)
    split.py                홀드아웃 20% + StratifiedGroupKFold 5×5 → data/processed/fold_id.csv
  models/
    logistic.py forest.py   모델 정의 (하나 추가할 때: 파일 하나 + registry.py에 한 줄)
    registry.py             이름 → 모델 객체
    train.py                fold별 학습, 예측 저장, artifact 저장
    artifacts.py            저장된 모델·입력 열 불러오기 (SHAP용)
  evaluation/metrics.py     예측 파일 점검용 지표 (정식 평가는 C 담당 모듈)
  analysis/                 B: 오류분석·SHAP·FN 프로파일링
  inspection/               C: 캘리브레이션·임계값·Inspection Score·예산 곡선
data/raw/                   원본 CSV (git 제외)
data/processed/             labeled.csv, unlabeled.csv, fold_id.csv
outputs/predictions/        모델별 예측 CSV (공통 스키마)
outputs/model/              모델 pkl + 입력 열 json
outputs/evaluation/         지표 요약
```

## 2. 데이터 처리 규칙 (전원 공통, 바꾸려면 팀 합의)

| 항목 | 규칙 | 근거 |
|---|---|---|
| 타깃 | `PassOrFail` 1 = 불량 | 가이드북 실습 코드 기준 |
| 제거 열 | `Clamp_Open_Position` (전부 0) | 상수열 |
| 입력 | 공정 변수 23개 (+ `mold_cn7`, 선택적으로 `cavity_pos`) | |
| shot | 같은 금형에서 공정값 24개가 완전히 같은 연속 2행 = 한 shot (2캐비티 LH/RH) | 라벨 1,249 shot = 1,144쌍 + 105홑 |
| cavity_pos | shot 안 순서 1/2. 홑 행은 1 | 불량이 첫 행에 몰림 (CN7 10/11, RG3 23/25) |
| 홀드아웃 | shot 단위 20%, (금형 × 불량 shot) 층화. **모델 선택·임계값 결정에 쓰지 않는다** | 최종 보고 1회용 |
| 교차검증 | 개발셋 80%를 StratifiedGroupKFold 5등분 × 5반복 (시드 42~46), shot 단위 | 같은 shot의 두 행이 train/test로 갈라지면 누출 |
| 임계값·캘리브레이션 | 각 학습 fold 안에서만 결정 | 누출 방지 |
| unlabeled | `is_idle`=1 (6개 센서가 동시에 열 최솟값) 31,305행(44%)은 미기록 구간으로 취급 | 라벨 데이터에는 0행 |

현재 수치: 라벨 2,393행 / 1,249 shot / 불량 42행(39 shot). 홀드아웃 479행(불량 9행), 개발셋 1,914행(불량 33행). fold별 불량 6~8행.

## 3. 공통 파일 스키마

### data/processed/fold_id.csv (분할은 이 파일 하나만 쓴다)

| 열 | 의미 |
|---|---|
| row_id | labeled.csv의 행 번호 (0~2392, cn7 다음 rg3) |
| shot_id | shot 번호 (0~1248) |
| mold | cn7 / rg3 |
| cavity_pos | 1 / 2 |
| holdout | 1이면 홀드아웃 |
| fold_rep0 … fold_rep4 | 반복 r에서의 fold 번호 0~4. 홀드아웃 행은 -1 |

```python
from src.data.split import load_folds, iter_folds
folds = load_folds(cfg)
for k, train_ids, test_ids in iter_folds(folds, rep=0):   # row_id 배열
    ...
```

### outputs/predictions/{model}__{feature_set}__oof.csv / __holdout.csv

| 열 | 의미 |
|---|---|
| row_id, shot_id, mold, cavity_pos | fold_id.csv와 같음 |
| fold_rep, fold | 어느 반복·fold의 test였는지. 홀드아웃 파일은 둘 다 -1 |
| y_true | 실제 라벨 |
| y_prob | 불량 확률 |
| model | logistic / random_forest / … |
| feature_set | process / process+cavity |

oof 파일은 개발셋 1,914행 × 5반복 = 9,570행. 한 (row_id, fold_rep)마다 정확히 한 번 예측된다.
행별 **Uncertainty**(5반복 표준편차)는 `groupby(["row_id"])["y_prob"].std()`로 바로 나온다.

### outputs/model/

| 파일 | 내용 |
|---|---|
| `{model}__{feature_set}__rep0_fold{k}.pkl` | rep0의 fold 모델 5개 (`config: models.save_all_folds: true`면 25개 전부) |
| `{model}__{feature_set}__final.pkl` | 개발셋 전체로 학습한 모델. 홀드아웃 예측에 쓴 것 |
| `features__{feature_set}.json` | 입력 열 이름·순서 |

```python
from src.models.artifacts import load_model, get_X
m = load_model("random_forest", "process+cavity")                  # final
X = get_X(lab, "process+cavity")                                    # lab = labeled.csv
import shap; shap.TreeExplainer(m).shap_values(X)
```

fold 모델로 SHAP을 낼 때는 그 fold의 test 행에만 적용한다 (학습에 본 행이 섞이면 안 된다).

## 4. 베이스라인 결과 (OOF 25 fold 평균, 참고용)

| 모델 | 입력 | PR-AUC | ROC-AUC | Recall@0.5 | FN@0.5 | Recall@10% |
|---|---|---|---|---|---|---|
| Logistic | process | 0.119 | 0.721 | 0.50 | 3.3 | 0.35 |
| Logistic | process+cavity | 0.146 | 0.793 | 0.55 | 3.0 | 0.40 |
| RandomForest | process | 0.106 | 0.788 | 0.00 | 6.6 | 0.36 |
| RandomForest | process+cavity | 0.116 | 0.821 | 0.00 | 6.6 | 0.43 |

- 불량 비율 1.75%. PR-AUC 0.12~0.15는 무작위(0.0175)의 7~8배지만 절댓값은 낮다.
- RF는 임계값 0.5에서 불량을 하나도 못 잡는다 (전부 정상 예측 → 정확도 98.2%). 정확도가 왜 무의미한지 보여주는 1장 그림 소재.
- `cavity_pos`를 넣으면 두 모델 모두 좋아진다. 불량이 쌍의 첫 행에 몰린 구조 때문이다. 이 변수를 쓸지, 쓴다면 어떻게 해석할지는 B와 상의.
- 홀드아웃은 불량이 9행뿐이어서 수치가 크게 흔들린다. 모델 비교는 OOF로, 홀드아웃은 최종 보고 1회만.

## 5. 새 모델 추가하는 법 (A)

1. `src/models/lightgbm_.py`에 `build(cfg, seed)` 함수 하나 (`cfg["models"]["lightgbm"]` 읽어서 객체 반환)
2. `src/models/registry.py`의 `_BUILDERS`에 `"lightgbm": lightgbm_.build` 한 줄
3. `configs/config.yaml`의 `models.baseline` 목록에 이름 추가, 하이퍼파라미터 블록 추가
4. `python -m src.models.train` → 예측·artifact가 같은 규칙으로 저장된다

## 6. 하지 말 것

- `fold_id.csv`를 다시 만들거나 각자 다른 분할을 쓰는 것 (시드가 같아도 코드가 다르면 다른 분할이 나온다)
- 홀드아웃으로 모델·임계값 고르기
- 행 단위(`train_test_split`) 분할
- unlabeled의 idle 행을 정상 shot으로 취급하는 것
