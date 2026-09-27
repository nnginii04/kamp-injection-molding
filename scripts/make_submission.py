"""대회 제출용 zip 만들기 + 블라인드 검사(개인정보 문자열이 없는지).

    python scripts/make_submission.py                       # dist/submission.zip (학습 데이터 포함 — 대회 요건)
    python scripts/make_submission.py --out dist/v1.zip
    python scripts/make_submission.py --no-data             # data/raw 대회 CSV 빼기 (공유용)
    python scripts/make_submission.py --word 홍길동 --word 우리대학   # 검사할 단어 추가

대회 요건: requirements, 학습 데이터, README, 테스트 예측결과가 zip 안에 있어야 함 → 끝에 포함 여부를 점검해 출력.
데이터: data/raw 중 config.json 의 products 에 적힌 대회 CSV 4개만 넣는다 (KAMP 공개 원본 등 다른 파일은 절대 넣지 않음).
제외: .git, data/synthetic, _incoming, .venv, __pycache__, outputs/synthetic,
      outputs/verification(공개 원본에서 나온 결과, 기본 제외), dist, 편집기·OS 파일
블라인드 검사 (zip 안의 모든 텍스트 파일과 파일 이름)
  - 이메일 주소, 로컬 절대경로(맥·리눅스·윈도우의 사용자 홈 폴더 경로)
  - git 설정의 user.name / user.email, 저장소 루트의 .blind_words(한 줄에 한 단어, git 제외 파일), --word 로 준 단어
    → 검사할 이름을 저장소 안에 적어 두지 않아도 된다
문제가 있으면 목록을 출력하고 종료코드 1 (zip 은 만들어 둠 — 고친 뒤 다시 실행).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", ".venv", "venv", "_incoming", "__pycache__", ".ipynb_checkpoints", ".idea", ".vscode",
                "dist", ".claude"}
EXCLUDE_PATHS = {"data/synthetic", "outputs/synthetic", "outputs/verification"}
EXCLUDE_FILES = {".DS_Store", ".blind_words"}
EXCLUDE_SUFFIX = {".pyc", ".pyo"}
_SEP, _NAME = r"[\\/]+", r"[^\\/\s\"'`]+"
GENERIC = {  # 패턴을 조각으로 조합 — 이 파일 자체가 검사에 걸리지 않게
    "이메일": re.compile(r"[A-Za-z0-9._%+-]+" + "@" + r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "로컬경로": re.compile("/" + "Users" + "/" + _NAME + "|/" + "home" + "/" + _NAME
                         + "|[A-Za-z]:" + _SEP + "Users" + _SEP + _NAME),
}
ALLOWED = {"noreply@anthropic.com"}  # 커밋 도구 서명 등 개인정보가 아닌 주소
PUBLIC_NAMES = {"labeled_data.csv", "unlabeled_data.csv", "moldset_labeled.csv", "supervised_label_cn7.csv"}
REQUIRED = {  # 대회 제출 요건 → zip 안에 있어야 할 것
    "requirements": ["requirements.txt"],
    "README": ["README.md"],
    "테스트 예측결과": ["outputs/predictions_labeled_cv.csv"],
    "교차검증 예측(공통 형식)": ["outputs/predictions/pipeline_*.csv"],
    "unlabeled 예측": ["outputs/predictions_unlabeled/predictions_unlabeled.csv",
                     "outputs/predictions_unlabeled/predictions_unlabeled_meta.json"],
}


def git_identity() -> list[str]:
    words = []
    for key in ("user.name", "user.email"):
        try:
            v = subprocess.run(["git", "config", key], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            v = ""
        if v:
            words.append(v)
            if "@" in v:
                words.append(v.split("@")[0])  # 메일 아이디만 쓰인 경우도 잡기
    return words


def blind_words(extra) -> list[str]:
    f = ROOT / ".blind_words"
    words = git_identity() + list(extra or [])
    if f.exists():
        words += [w.strip() for w in f.read_text(encoding="utf-8").splitlines() if w.strip() and not w.startswith("#")]
    return sorted({w for w in words if len(w) >= 2}, key=str.lower)


def competition_files() -> set[str]:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    return {f"{cfg['data_dir']}/{f[k]}" for f in cfg["products"].values() for k in ("labeled", "unlabeled")}


def collect(include_data: bool, include_verification: bool) -> list[Path]:
    allowed_data = competition_files() if include_data else set()
    skip_paths = set(EXCLUDE_PATHS)
    if include_verification:
        skip_paths.discard("outputs/verification")
    files = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        parts = rel.parts
        if any(x in EXCLUDE_DIRS for x in parts) or p.is_dir():
            continue
        s = rel.as_posix()
        if any(s == d or s.startswith(d + "/") for d in skip_paths):
            continue
        if s.startswith("data/") and p.name != ".gitkeep" and s not in allowed_data:
            continue  # 대회 CSV 4개 말고는 data/ 아래 어떤 파일도 넣지 않음 (공개 원본 차단)
        if p.name in PUBLIC_NAMES:
            continue
        if p.name in EXCLUDE_FILES or p.suffix in EXCLUDE_SUFFIX:
            continue
        files.append(p)
    return files


def scan(name: str, data: bytes, words: list[str]) -> list[tuple[str, str]]:
    hits = []
    for w in words:
        if w.lower() in name.lower():
            hits.append(("파일 이름", w))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return hits  # 이미지 등 바이너리
    for kind, pat in GENERIC.items():
        for m in set(pat.findall(text)):
            if m not in ALLOWED:
                hits.append((kind, m))
    low = text.lower()
    for w in words:
        if w.lower() in low:
            hits.append(("지정 단어", w))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="제출용 zip + 블라인드 검사")
    ap.add_argument("--out", default="dist/submission.zip")
    ap.add_argument("--no-data", action="store_true", help="data/raw 의 대회 CSV 를 빼기 (기본은 포함 — 대회 요건)")
    ap.add_argument("--include-verification", action="store_true", help="outputs/verification 포함 (공개 원본 유래)")
    ap.add_argument("--word", action="append", help="추가로 검사할 단어 (여러 번 가능)")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    words = blind_words(args.word)
    files = collect(not args.no_data, args.include_verification)
    problems = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arc = f.relative_to(ROOT).as_posix()
            data = f.read_bytes()
            z.writestr(zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
            problems += [(arc, k, v) for k, v in scan(arc, data, words)]
    size = out.stat().st_size / 2 ** 20
    raw = sum(f.stat().st_size for f in files) / 2 ** 20
    names = [f.relative_to(ROOT).as_posix() for f in files]
    print(f"zip: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out.name} — 파일 {len(files)}개, "
          f"압축 {size:.1f} MB (원본 합계 {raw:.1f} MB)")
    data_in = sorted(n for n in names if n.startswith("data/") and not n.endswith(".gitkeep"))
    print(f"  학습 데이터: {len(data_in)}개 {[Path(n).name for n in data_in]}" + ("  (--no-data)" if args.no_data else ""))
    leaked = [n for n in names if Path(n).name in PUBLIC_NAMES]
    if leaked:
        problems += [(n, "공개 원본", Path(n).name) for n in leaked]
    print("  제출 요건 점검:")
    for label, pats in REQUIRED.items():
        found = [n for pat in pats for n in names if Path(n).match(pat)]
        ok = all(any(Path(n).match(pat) for n in names) for pat in pats)
        print(f"    [{'O' if ok else 'X'}] {label}: {len(found)}개" + ("" if ok else f" — 없음: {pats}"))
        if not ok:
            problems.append((label, "제출 요건 누락", ", ".join(pats)))
    if not args.no_data and len(data_in) != len(competition_files()):
        problems.append(("data/raw", "제출 요건 누락", "대회 CSV 4개가 모두 있어야 함 (data/raw 에 넣은 뒤 다시 실행)"))
    print(f"  블라인드 검사 단어 {len(words)}개 (git 설정·.blind_words·--word) + 이메일·로컬경로 패턴")
    if problems:
        print(f"\n[블라인드 검사] 문제 {len(problems)}건 — 고친 뒤 다시 실행하세요")
        shown = {}
        for arc, kind, v in problems:
            shown.setdefault(arc, set()).add(f"{kind}: {v}")
        for arc, v in sorted(shown.items()):
            print(f"  - {arc}: " + ", ".join(sorted(v)))
        return 1
    print("\n[블라인드 검사] 통과 — 개인정보로 보이는 문자열 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
