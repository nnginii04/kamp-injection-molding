# 사출성형 품질불량 사전예측 — 팀 최종 베이스라인

제6회 K-인공지능 제조데이터 분석 경진대회 · 사출성형기 AI 데이터셋 (CN7 / RG3)

> 🚧 정리 중인 브랜치(`feat/final-baseline`)입니다. 쉬운 설명·결과표·누수 체크리스트가 담긴 전체 README 는 마무리 단계에서 채웁니다.

## 실행

```bash
pip install -r requirements.txt
python run_all.py --synthetic --fast   # 데이터 없이 동작 확인 (가짜 데이터, 성능 숫자 무의미) → outputs/synthetic/
python run_all.py                      # data/raw/ 의 대회 CSV 4개로 전체 실행 → outputs/
python run_evaluation.py               # 팀 공통 평가 → outputs/evaluation/
```

## 폴더

```
config.json          모든 설정 (seed, CV, 모델, 판정 기준, feature 옵션, external.enabled=false)
run_all.py           전체 파이프라인 한 번에 실행
run_evaluation.py    팀 공통 평가 (evaluation/ 사용)
src/                 data · audit · features · cv · models · evaluate · errors · final · report · synthetic
evaluation/          팀 공통 평가 모듈 (예측 파일 형식: evaluation/README.md)
scripts/             공개 원본 연결·검증 (해석 전용, 기본 꺼짐)
archive/             이전 베이스라인 보관 (참고용, 실행 대상 아님)
data/raw/            대회 CSV 4개 (git 제외)
outputs/             실행 결과 (git 제외)
```

## 외부 데이터 주의

KAMP 공개 원본은 대회 규정 확인 전까지 **모델 입력 금지**입니다. `scripts/` 의 해석·검증 스크립트만 읽기 전용으로 쓰며,
`config.json` 의 `external.enabled` 가 기본 `false` 입니다. 원본 파일을 저장소에 복사하지 마세요.
