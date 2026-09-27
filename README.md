# 사출성형 품질불량 사전예측 — 팀 최종 베이스라인

제6회 K-인공지능 제조데이터 분석 경진대회 · 사출성형기 AI 데이터셋 (CN7 / RG3)

사출 공정값(온도·압력·시간 등, 표준화된 값)만으로 **불량을 미리 찾아 검사 우선순위를 정하는** 파이프라인입니다.
결과 보고서는 `outputs/report.md`, 쉬운 설명은 [`notebooks/baseline_explained.ipynb`](notebooks/baseline_explained.ipynb) 에 있습니다.

## 실행 (3줄)

```bash
pip install -r requirements.txt     # Python 3.13, 버전 고정
python run_all.py                   # data/raw/ 의 대회 CSV 4개 → outputs/ (이 PC 기준 약 7~8분, --fast 는 반복 2회로 더 빠름)
python run_evaluation.py            # 팀 공통 평가 + report.md 수치 검산 → outputs/evaluation/
```

데이터 없이 동작만 확인: `python run_all.py --synthetic --fast` (가짜 데이터, 숫자 무의미) · 누수 자동 검사: `python -m unittest discover -s tests`

## 폴더 구조

```
config.json          모든 설정 (seed 42, CV, 모델, 판정 기준, 금지 옵션과 그 이유, external.enabled=false)
run_all.py           전체 파이프라인: 진단 → 교차검증 → 추가 실험 → 오류분석 → 예측 파일 → report.md
run_evaluation.py    팀 공통 평가 (cv_mode 별 표, report.md 수치 자동 검산)
src/                 data · audit · features · cv · models · evaluate · experiments · errors · final
                     · key_figures · report · known_facts · synthetic
evaluation/          팀 공통 평가 모듈 (예측 파일 형식: evaluation/README.md)
scripts/             verify_public_link.py · link_public.py (공개 원본, 읽기 전용·기본 꺼짐)
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

추가 발견: 공정변수는 **shot 전체 이상(짝 둘 다 불량)은 잡지만 캐비티 국소 불량(한쪽만 불량)은 못 잡습니다** (CN7 block CV 에서 잡힌 불량 = 둘 다 불량인 3 shot 6행뿐). 근거가 연속된 3 shot 뿐이라는 한계가 있습니다.

## 결과 요약 (대표 지표 = 검사비율별 재현율, block CV(비관) ~ shot CV(낙관))

| 제품 | 모델 | 상위 2% 검사 | 상위 5% | 상위 10% | shot AP | 비고 |
|---|---|---|---|---|---|---|
| CN7 | 로지스틱 회귀 (balanced) | 0.35 ~ 0.46 | 0.35 ~ 0.56 | 0.49 ~ 0.73 | 0.19 ~ 0.35 | 무작위 기준선의 약 9배 (block shot AP) |
| RG3 | LightGBM | 0.03 ~ 0.00 | 0.07 ~ 0.04 | 0.13 ~ 0.08 | 0.05 ~ 0.04 | 신호 없음 (기준선 수준) |
| 무작위 검사 | - | 0.02 | 0.05 | 0.10 | ≈ 불량률 | |

- 모델 선택: block CV · shot 단위 AP · 1-SE 규칙(비슷하면 단순한 모델). Accuracy 는 쓰지 않음.
- 이전 베이스라인의 CN7 PR-AUC 0.476 은 v0 설정을 그대로 재현하면 정확히 같게 나오고, 검증 조건(block CV)과
  모델 설정 변경으로 이번 0.24(block) ~ 0.36(shot)이 됨 (report.md 5-5).
- 현장 제안: 공정 데이터 모델은 **shot 전체 이상 선별**에, 캐비티 국소 불량은 금형·비전 검사 등 **별도 수단**으로.

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
- **블라인드 평가**: 파일·결과물에 이름·학교·로컬 경로를 넣지 않음. 제출 전 `python scripts/make_submission.py` 로 zip 을 만들고 검사 결과가 '통과'인지 확인
  (검사할 이름은 저장소 밖 로컬 파일 `.blind_words` 에 한 줄씩 — git 에 올리지 않음).

## 외부 데이터 주의

KAMP 공개 원본은 대회 규정 확인 전까지 **모델 입력 금지**입니다. `scripts/verify_public_link.py` 와 `scripts/link_public.py` 는
해석·검증용으로 zip 을 **읽기만** 하고, `config.json` 의 `external.enabled` 가 기본 `false` 라 `--enable` 없이는 실행되지 않습니다.
원본 파일을 저장소에 복사하지 마세요. 보고서에 인용하는 공개 원본 수치는 `src/known_facts.py` 한 곳에 출처와 함께 적혀 있습니다.
