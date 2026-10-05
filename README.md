# 사출성형 품질불량 사전예측 — 팀 최종 베이스라인

제6회 K-인공지능 제조데이터 분석 경진대회 · 사출성형기 AI 데이터셋 (CN7 / RG3)

사출 공정값(온도·압력·시간 등, 표준화된 값)만으로 **불량을 미리 찾아 검사 우선순위를 정하는** 파이프라인입니다.
결과 보고서는 `outputs/report.md`, 쉬운 설명은 [`notebooks/baseline_explained.ipynb`](notebooks/baseline_explained.ipynb) 에 있습니다.

## 실행

```bash
pip install -r requirements.txt     # Python 3.13, 버전 고정
python run_all.py                   # data/raw/ 의 대회 CSV 4개 → outputs/ (후보 모델 14개 · 14코어 맥북 약 15분, 2코어 환경 약 1~2시간 · --fast 는 반복 2회로 더 빠름)
python run_evaluation.py            # 팀 공통 평가 + report.md 수치 검산 → outputs/evaluation/
python scripts/inspection_priority.py --models cn7=logreg rg3=lgbm --ambiguity outputs/analysis/ambiguity_scores_for_ablation.csv
                                    # 검사 우선순위: 3구역 · Ablation · Budget Curve → outputs/inspection/
```

> ⚠ `--fast` 는 반복을 2회로 줄인 **동작 확인용**입니다. 반복이 적으면 1-SE 규칙의 표준오차가 달라져 **선택 모델이 바뀔 수 있습니다**
> (실제로 `--fast` 에서는 CN7 `ensemble`, RG3 `rf` 가 선택됨). 제출·보고 수치는 반복 5회의 전체 실행(`python run_all.py`) 기준입니다.

데이터 없이 동작만 확인: `python run_all.py --synthetic --fast` (가짜 데이터, 숫자 무의미) · 자동 검사(71개): `python -m unittest discover -s tests`

## 폴더 구조

```
config.json          모든 설정 (seed 42, CV, 모델, 판정 기준, 금지 옵션과 그 이유, external.enabled=false)
run_all.py           전체 파이프라인: 진단 → 교차검증 → 추가 실험 → 오류분석 → 예측 파일 → report.md
run_evaluation.py    팀 공통 평가 (cv_mode 별 표, report.md 수치 자동 검산)
src/                 data · audit · features · cv · models · evaluate · experiments · errors · final
                     · key_figures · report · known_facts · synthetic
evaluation/          팀 공통 평가 모듈 (예측 파일 형식: evaluation/README.md)
scripts/             inspection_priority.py (검사 우선순위: 불확실성 · 3구역 · Ablation · Budget Curve)
                     e1_policy_comparison.py · e2_delay_replay.py · e3_robustness.py (검사 방법 비교 · 검사 지연 재현 · 강건성, 공개 원본 필요)
                     e5_floor_ceiling.py — 바닥(학습 없는 규칙)–모델–천장 비교 + 블록 순열검정 (사전 등록 docs/protocol_E5.md, 결과 outputs/e5/, 공개 원본 필요)
                     guidebook_audit.py · guidebook_split_check.py (가이드북 검증 방식 재현)
                     verify_public_link.py · link_public.py (공개 원본, 읽기 전용·기본 꺼짐)
                     make_notebook.py (설명 노트북 생성·실행) · make_submission.py (제출 zip + 블라인드 검사)
notebooks/           baseline_explained.ipynb (쉬운 한국어 설명, 실행 결과 포함)
tests/               누수·형식 자동 검사 (unittest)
archive/baseline_v0/ 이전 베이스라인 보관 (참고용, 실행 대상 아님)
data/raw/            대회 CSV 4개 (git 제외, 직접 넣기)
outputs/             실행 결과 (git 제외)
  report.md            보고서 근거 (심사 배점 순서, 맨 앞 '한 장 요약')
  figures/key*.png     보고서·발표용 핵심 그림 6개
  fold_id.csv          팀 공유 분할표
  predictions/         교차검증 예측, 팀 공통 형식 (run_all.py 는 pipeline_<모델>.csv)
  predictions_unlabeled/  unlabeled 예측 + 메타데이터 (정답 없음 → 평가 대상 아님)
  evaluation/          run_evaluation.py 결과
  inspection/          검사 우선순위 결과 (ranking.csv · zone_summary.csv · ablation.csv · budget_curve_<제품>.png)
  manifest.json        입력 데이터 SHA-256 · seed · 전체 설정 · 버전 · git 커밋 · 소요시간
```

## 데이터에서 확인한 사실 (핵심 발견)

| # | 사실 | 확인 방법 | 파이프라인의 대응 |
|---|---|---|---|
| 1 | CN7 1,211행(불량 17), RG3 1,182행(불량 25). `PassOrFail=1` 이 불량 | 대회 파일 + 공개 원본의 불량 사유(Reason)와 100% 일치 | 라벨 1 = 불량 |
| 2 | 4개 파일이 **파일마다 따로** 표준화(ddof=0) → labeled 와 unlabeled 의 같은 z값은 다른 물리값 | 파일별 평균 0·표준편차 1 | unlabeled 점수는 순위 용도로만 |
| 3 | `Clamp_Open_Position` 은 labeled 에서 상수, `Unnamed: 0` 은 행 번호 | 대회 파일 | 둘 다 입력에서 제외 (자동 검사) |
| 4 | 공정값이 완전히 같은 두 행 = 한 shot 의 LH/RH 짝. CN7 606 shot, RG3 591 shot | 전역으로 묶기 (연속 행만 묶으면 떨어진 짝이 갈라짐) | 전역 `shot_id` 로 분할·평가 |
| 5 | 짝 한쪽만 불량이 대부분 → 행 단위 정밀도·F1 상한 (CN7 0.607/0.756, RG3 0.500/0.667) | 대회 파일 | 모든 결과표에 상한 병기, shot 단위 지표 함께 보고 |
| 6 | 짝 안의 행 순서는 캐비티가 아니라 기록 순서이고, 한쪽만 불량인 짝 36개 중 33개에서 불량 행이 먼저 기록됨 (앞 행이 RH 인 비율은 CN7 65.8%·RG3 56.3%뿐) | 공개 원본 연결 검증 | 순서 변수 사용 금지 (켜면 실행 중단), ablation 으로만 비교 |
| 7 | CN7 불량은 전부 가동 초기 한 구간(shot 24~60)에 몰림 | 대회 파일 | block CV(연속 10 shot 묶음)로 모델 선택, 3-fold |
| 8 | RG3 는 어떤 모델도 무작위 기준선 수준 = 신호 없음 | 교차검증 | 정직하게 '신호 없음'으로 보고 |
| 9 | CN7 과 RG3 는 분포가 완전히 다름 (도메인 분류기 트리 AUC 1.0) | 교차검증 | 제품별 모델이 기본, 통합은 비교 실험 |
| 10 | 대회 unlabeled 안에 labeled 행이 들어 있음 (CN7 1210/1211, RG3 774/1182) | 공개 원본 연결 검증 ((TimeStamp, PART_NAME) 1:1 + 공정변수 24개 소수 3자리 일치) | unlabeled 로 학습·평가 금지, 예측 파일에 경고 |

추가 발견: 공정변수는 **shot 전체 이상(짝 둘 다 불량)은 먼저 잡지만 한쪽만 불량인 shot 은 일부만 잡습니다** (CN7 block CV: 둘 다 불량 3 shot 은 모두 탐지, 한쪽만 불량 11 shot 중 9 shot 탐지, 끝까지 놓친 2 shot 은 모두 한쪽만 불량). 한쪽만 불량인 shot 은 두 부품의 공정값이 같아 현재 변수로는 구분할 수 없고, 근거가 한 시간대의 14 shot 뿐이라는 한계가 있습니다.

## 결과 요약 (대표 지표 = 검사비율별 재현율)

block CV = 처음 보는 생산 구간을 가정(비관), shot CV = 같은 구간 안을 가정(낙관). 실제 성능은 둘 사이로 봅니다.

| 지표 | CN7 block CV (비관) | CN7 shot CV (낙관) | RG3 block CV (비관) | RG3 shot CV (낙관) | 무작위 검사 |
|---|---|---|---|---|---|
| 상위 2% 검사 시 재현율 | 0.35 | 0.46 | 0.03 | 0.00 | 0.02 |
| 상위 5% 검사 시 재현율 | 0.35 | 0.56 | 0.07 | 0.04 | 0.05 |
| 상위 10% 검사 시 재현율 | 0.49 | 0.73 | 0.13 | 0.08 | 0.10 |
| shot 단위 AP | 0.19 | 0.35 | 0.05 | 0.04 | ≈ 불량률 |

- CN7 모델 = 로지스틱 회귀(class_weight balanced), 무작위 기준선의 약 9배(block shot AP). RG3 모델 = LightGBM 이지만 **신호 없음**(기준선 수준 — block 이 더 높게 나온 것은 흔들림).
- 모델 선택: block CV · shot 단위 AP · 1-SE 규칙(비슷하면 단순한 모델). Accuracy 는 쓰지 않음.
- 이전 베이스라인의 CN7 PR-AUC 0.476 은 v0 설정을 그대로 재현하면 정확히 같게 나오고, 검증 조건(block CV)과
  모델 설정 변경으로 이번 0.24(block) ~ 0.36(shot)이 됨 (report.md 5-5).
- 현장 제안: 공정 데이터 모델은 **shot 전체 이상 선별**에, 캐비티 국소 불량은 금형·비전 검사 등 **별도 수단**으로.
- 위 표의 재현율은 fold 를 합친 뒤 순위를 매긴 값(pooled)입니다. 보고서 4장은 아래 '검사 우선순위'의 **검증 fold 안 순위** 기준을 씁니다.

## 검사 우선순위 (`scripts/inspection_priority.py` → `outputs/inspection/`)

block CV 예측(반복 5회)으로 shot 점수 순위를 매겨 검사량별 불량 포착률을 계산합니다. 순위는 **각 검증 fold 안에서** 매깁니다
(fold 마다 다른 모델이 낸 점수는 눈금이 달라, 합쳐서 순위를 매기면 CN7 검사 10% 포착률이 0.49 로 낮게 나옵니다. 실제 운영은 한 모델의 점수로 순위를 매깁니다).

| 검사량 (부품 기준) | CN7 로지스틱 | RG3 LightGBM | 무작위 검사 |
|---|---|---|---|
| 2% | 0.38 | 0.05 | 0.02 |
| 5% | 0.51 | 0.09 | 0.05 |
| 10% | 0.61 | 0.14 | 0.10 |
| 20% | 0.78 | 0.21 | 0.20 |
| 곡선 면적 | 0.87 ± 0.01 | 0.49 ± 0.04 | 0.50 |

- **3구역 (CN7)**: High(점수 상위 2%) 구역은 부품의 2.2%에 불량 부품의 38%가 들어 있고 구역 안 불량률이 평균의 17.5배, Review(상위 2~10%)는 3.0배, Low(나머지)는 0.4배입니다. RG3 는 순위가 무작위 수준이라 구역 규칙을 적용하지 않습니다.
- **Ablation**: 모델 점수에 불확실성(반복 5회 예측의 표준편차)·Ambiguity 를 더한 순위는 CN7 곡선 면적이 0.87 → 0.71~0.78 로 낮아져 **모델 점수만** 사용합니다. Ambiguity 는 오류 해석에만 씁니다.
- 옵션: `--rank fold`(기본) / `--rank pooled`, `--ambiguity <csv>`, `--novelty <csv>`. 입력은 `outputs/predictions/pipeline_<모델>.csv` 입니다.

## 제출물 구성 (`python scripts/make_submission.py` → `dist/submission.zip`)

대회 요건(requirements, 학습 데이터, README, 테스트 예측결과)을 스크립트가 점검하고, 블라인드 검사(이름·이메일·로컬 경로)도 함께 합니다.

| 구분 | zip 안의 파일 |
|---|---|
| 실행·설명 | `README.md`, `requirements.txt`, `config.json`, `run_all.py`, `run_evaluation.py`, `src/`, `evaluation/`, `scripts/`, `tests/`, `notebooks/baseline_explained.ipynb` |
| 학습 데이터 | `data/raw/` 의 대회 CSV 4개 (`moldset_labeled_cn7/rg3.csv`, `moldset_unlabeled_cn7/rg3.csv`). KAMP 공개 원본은 `--include-public` 일 때만 `data/public/` 에 (출처·SHA-256 메모 `SOURCE.md` 포함) |
| 테스트 예측결과 | `outputs/predictions_labeled_cv.csv` (labeled 전 행의 교차검증 예측 = 테스트 예측), `outputs/predictions/pipeline_*.csv` (후보 모델별 교차검증 예측, 팀 공통 형식) |
| unlabeled 예측 | `outputs/predictions_unlabeled/predictions_unlabeled.csv` + `predictions_unlabeled_meta.json` (점수는 순위 용도, labeled 행 포함 가능성 경고) |
| 결과·근거 | `outputs/report.md`, `outputs/figures/`, `outputs/tables/`, `outputs/evaluation/`, `outputs/fold_id.csv`, `outputs/manifest.json`, `outputs/audit.json`, `outputs/run_log.txt` |
| 참고 | `archive/baseline_v0/` (이전 베이스라인, 실행 대상 아님) |
| 제외 | `.git`, `_incoming/`, `.venv`, 캐시, `outputs/synthetic/`, `outputs/verification/`(공개 원본 유래 — `--include-public` 또는 `--include-verification` 때만 포함), `dist/`, `.blind_words` |

- 학습 데이터를 빼고 공유할 때: `python scripts/make_submission.py --no-data`
- 공개 원본을 함께 넣을 때(팀 결정 2026-10-01, 운영진 확인 2026-09-30): `python scripts/make_submission.py --include-public` — `config.json` `external.public_path` 의 zip/폴더에서 분석 스크립트가 읽는 `moldset_labeled.csv`, `labeled_data.csv`, `unlabeled_data.csv`(약 230 MB, 압축 후 약 60 MB) 를 `data/public/` 에 넣고 출처·SHA-256 을 `data/public/SOURCE.md` 에 적습니다. 용량이 문제면 `--public-file labeled_data.csv --public-file moldset_labeled.csv` 로 제한하고, `unlabeled_data.csv` 는 README 의 출처만 남깁니다.
- zip 을 풀어서 바로 `pip install -r requirements.txt` → `python run_all.py` 로 재현됩니다. 재현 테스트(2026-10-05, 제출 zip을 새 폴더·새 가상환경 Python 3.13 / Linux 2코어에서 실행): 선택 모델(CN7 `logreg` 0.188, RG3 `lgbm` 0.052)·테스트 예측결과·unlabeled 예측이 freeze 결과와 1e-11 이내로 일치, `run_evaluation.py` 검산 504개 일치. 트리 계열(XGBoost·RandomForest·CatBoost)은 CPU·스레드 환경에 따라 소수 셋째 자리가 달라질 수 있습니다(예: CN7 block XGBoost 0.164 ↔ 0.183). 선택 결과에는 영향이 없습니다.

### 예측 결과 파일 두 가지

대회 unlabeled 파일에는 정답이 없고 labeled 행이 섞여 있어(사실 10), 성격이 다른 두 파일을 함께 제출합니다.

| 파일 | 대상 | 내용 | 용도 |
|---|---|---|---|
| A. `outputs/predictions_labeled_cv.csv` | labeled 전 행 | 교차검증 예측 (해당 행을 학습에 쓰지 않은 모델의 점수) + 정답 | 성능을 확인할 수 있는 예측. 보고서 수치의 근거 |
| B. `outputs/predictions_unlabeled/predictions_unlabeled.csv` | unlabeled 전 행 | 점수 · 판정 · 센서값 0 여부 · labeled 와 겹치는 행 여부 | 정답이 없어 평가 대상이 아님. 점수는 순위 용도로만 |

B 의 점수는 labeled 와 unlabeled 가 파일마다 따로 표준화되어 있어(사실 2) 확률로 읽지 않습니다.

## 누수 체크리스트 (모두 `tests/` 로 자동 검사)

- [x] 같은 shot(LH/RH 짝)이 학습·검증에 동시에 들어가지 않음 — shot·block CV 모두 (`test_leakage`)
- [x] block CV 에서 같은 block 이 학습·검증에 동시에 들어가지 않음, block 은 연속 shot
- [x] `shot_id` 는 전역 기준 (공정값 전체가 같은 행) — 연속 행만 묶는 방식 금지
- [x] 입력 변수에 `Unnamed: 0` / `PassOrFail` / `Clamp_Open_Position` / `shot_id` / 짝 안 순서 변수 없음
- [x] 전처리(스케일러)와 이상탐지 기준은 학습 fold 에만 fit, 판정 임계값은 학습 fold 안에서만 결정
- [x] 확률 보정의 내부 검증도 shot 단위로 나눔
- [x] unlabeled 로 학습하지 않음 (unlabeled 이상점수 기본 꺼짐), unlabeled 예측은 평가 폴더와 분리
- [x] 공개 원본은 모델 입력에 쓰지 않음 (`external.enabled=false`)
- [x] 같은 명령 두 번 실행 시 결과 표·예측 파일이 비트 단위로 같음

## 팀원이 자기 모델을 평가하는 법

1. `outputs/fold_id.csv` 를 읽어 대회 CSV 와 `row_id`(= `Unnamed: 0`)로 합치고, `fold_block_r0 == k` 인 행을 검증, 나머지를 학습에 씁니다
   (k = 0, 1, 2). 모든 `fold_<cv_mode>_r<반복>` 열에 대해 반복하면 run_all.py 와 **같은 분할**이 됩니다.
2. 예측을 `product, model, cv_mode, repeat, fold, row_id, y_true, y_score` 열로 저장해 `outputs/predictions/` 에 넣습니다.
   **파일 이름은 `pipeline_` 로 시작하지 않게** (예: `teamA_rf.csv`) — `pipeline_*.csv` 는 run_all.py 가 매번 새로 씁니다.
3. `python run_evaluation.py` → `outputs/evaluation/experiment_summary.md` 에서 우리 모델과 같은 기준(cv_mode 별 표)으로 비교됩니다.
   `shot_id` 는 넣어도 무시하고 전역 기준으로 다시 붙이며, 판정 기준은 `--threshold` 또는 `--budget` 중 하나 (없으면 파일의 `y_pred`).
   자세한 형식: [`evaluation/README.md`](evaluation/README.md)

## 팀 규칙

- **브랜치**: `main` 에 직접 커밋·push 하지 않음. 작업은 `feat/…`, `fix/…` 브랜치에서 하고 PR 로 합침.
- **커밋 메시지**: `종류: 한국어 요약` — 종류는 `feat`(기능) / `fix`(버그·정확성) / `docs`(문서) / `exp`(실험) / `chore`(정리) / `refactor` / `test`.
- **줄바꿈**: `.gitattributes` 로 LF 통일 (Windows 팀원도 그대로 커밋하면 됨).
- **데이터·결과**: `data/raw/`, `outputs/`, `_incoming/` 은 git 에 올리지 않음.
- **블라인드 평가**: 파일·결과물에 이름·학교·로컬 경로를 넣지 않음. 제출 전 `python scripts/make_submission.py` 로 zip 을 만들고 블라인드 검사·제출 요건 점검이 모두 통과인지 확인
  (검사할 이름은 저장소 밖 로컬 파일 `.blind_words` 에 한 줄씩 — git 에 올리지 않음).

## 공개 원본 데이터 사용 범위

KAMP 공개 원본(사출성형기 AI 데이터셋)은 운영진 확인(2026-09-30: 출처·사유·방법을 밝히면 사용 가능)을 받아 아래 범위에서만 씁니다.

| 항목 | 내용 |
|---|---|
| 출처 | 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 사출성형기 AI 데이터셋, KAIST(울산과학기술원, ㈜이피엠솔루션즈), 2020.12.14., www.kamp-ai.kr |
| 사유 | 대회 파일에 없는 시각 · 캐비티(LH/RH) · 불량 사유 · 원 단위 값과 대회 기간 이후 생산분이 있어, 데이터 구조 확인과 오류 해석, 날짜 단위 확인이 가능함 |
| 방법 | 대회 파일과 행 단위로 연결해 라벨이 100% 일치함을 확인한 뒤 읽기 전용으로 사용 (`scripts/verify_public_link.py`, `scripts/link_public.py`) |
| 쓰는 곳 | ① 데이터 구조·누수 확인(사실 6, 10) ② 오류 해석(불량 사유, 캐비티) ③ 대회 이후 기간의 탐색적 날짜 검증(`scripts/e1~e3`). 독립 최종 테스트가 아님 |
| 쓰지 않는 곳 | **모델 학습과 입력**. 최종 모델은 대회 CSV 로만 학습하고 `config.json` 의 `external.enabled` 는 `false` 입니다. LH/RH 구분도 모델 입력에 넣지 않습니다 |

- 원본 파일은 저장소의 `data/raw/` 에 넣지 않습니다. 제출 zip 에는 `python scripts/make_submission.py --include-public` 으로 `data/public/` 에 넣고 출처·SHA-256 을 `data/public/SOURCE.md` 에 적습니다.
- 보고서에 인용하는 공개 원본 수치는 `src/known_facts.py` 한 곳에 출처와 함께 적혀 있습니다.
