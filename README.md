# Injection Molding Defect Prediction

사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정

## 실행 방법

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` 하나로 전처리 → 학습 → 평가 → 추론 → 결과파일 생성까지 순서대로 실행됩니다.
결과는 `outputs/` 에 저장됩니다.

## 데이터

`data/raw/` 에 아래 4개 파일을 넣습니다 (git에는 올리지 않음).

| 파일 | 설명 |
| --- | --- |
| moldset_labeled_cn7.csv | CN7 금형 라벨 데이터 (1,211행) |
| moldset_labeled_rg3.csv | RG3 금형 라벨 데이터 (1,182행) |
| moldset_unlabeled_cn7.csv | CN7 금형 비라벨 데이터 (35,239행) |
| moldset_unlabeled_rg3.csv | RG3 금형 비라벨 데이터 (35,941행) |

## 구조

```
configs/config.yaml   시드, 경로, 하이퍼파라미터
src/preprocess.py     중복 제거, 상수열 제거, 금형 병합
src/train.py          베이스라인 + 후보 모델 학습
src/evaluate.py       Repeated Stratified K-Fold, F1, 확률보정
src/explain.py        SHAP, FN/FP 집중 조건 분석
src/predict.py        테스트 예측결과 파일 생성
notebooks/            EDA, 모델링, 오류분석 노트북
outputs/              예측결과, 지표, 그림
docs/                 보고서·발표자료
```

## 환경

Python 3.10+, CPU만으로 실행 가능. 전체 파이프라인은 수 분 내에 완료됩니다.
