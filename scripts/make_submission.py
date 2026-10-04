"""대회 제출용 zip 만들기 + 블라인드 검사(개인정보 문자열이 없는지).

    python scripts/make_submission.py                       # dist/submission.zip (학습 데이터 포함 — 대회 요건)
    python scripts/make_submission.py --out dist/v1.zip
    python scripts/make_submission.py --no-data             # data/raw 대회 CSV 빼기 (공유용)
    python scripts/make_submission.py --include-public      # KAMP 공개 원본도 data/public/ 에 넣기 (팀 결정 2026-10-01)
    python scripts/make_submission.py --word 홍길동 --word 우리대학   # 검사할 단어 추가

대회 요건: requirements, 학습 데이터, README, 테스트 예측결과가 zip 안에 있어야 함 → 끝에 포함 여부를 점검해 출력.
데이터: data/raw 중 config.json 의 products 에 적힌 대회 CSV 4개만 넣는다. 저장소 안에 있는 다른 데이터 파일은 넣지 않는다.
공개 원본: --include-public 일 때만, config.json external.public_path(zip 또는 폴더)에서 분석 스크립트가 실제로 읽는
      파일(PUBLIC_USED)을 zip 안 data/public/ 에 넣고 출처·SHA-256 메모(data/public/SOURCE.md)를 함께 만든다.
      이때 outputs/verification(공개 원본에서 나온 검증 결과)도 같이 넣는다. 원본 내용은 바꾸지 않는다.
제외: .git, data/synthetic, _incoming, .venv, __pycache__, outputs/synthetic,
      outputs/verification(공개 원본에서 나온 결과, 기본 제외), dist, 편집기·OS 파일
블라인드 검사 (zip 안의 모든 텍스트 파일과 파일 이름)
  - 이메일 주소, 로컬 절대경로(맥·리눅스·윈도우의 사용자 홈 폴더 경로)
  - git 설정의 user.name / user.email, 저장소 루트의 .blind_words(한 줄에 한 단어, git 제외 파일), --word 로 준 단어
    → 검사할 이름을 저장소 안에 적어 두지 않아도 된다
  - 공개 원본 CSV 는 우리가 만든 파일이 아니므로 파일 이름만 검사한다 (내용은 SOURCE.md 의 SHA-256 으로 원본과 대조).
문제가 있으면 목록을 출력하고 종료코드 1 (zip 은 만들어 둠 — 고친 뒤 다시 실행).
"""
from __future__ import annotations

import argparse
import hashlib
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
# --include-public 때 넣는 파일 = 분석 스크립트가 실제로 읽는 것만
# (scripts/external_experiments.py, e1_policy_comparison.py, oneside_matched.py, link_public.py, verify_public_link.py)
PUBLIC_USED = ("moldset_labeled.csv", "labeled_data.csv", "unlabeled_data.csv")
PUBLIC_ARC = "data/public"  # zip 안에서 공개 원본이 놓이는 폴더
PUBLIC_SOURCE = "KAMP 「사출성형기 AI 데이터셋」 원본 (04. Dataset_Molding.zip)"
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
            continue  # 대회 CSV 4개 말고는 data/ 아래 어떤 파일도 넣지 않음 (저장소 안 공개 원본 복사본 차단)
        if p.name in PUBLIC_NAMES:
            continue
        if p.name in EXCLUDE_FILES or p.suffix in EXCLUDE_SUFFIX:
            continue
        files.append(p)
    return files


def public_source(override: str | None) -> Path:
    """공개 원본 위치: --public 인자, 없으면 config.json external.public_path (zip 또는 폴더, ~ 허용)."""
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    raw = override or cfg.get("external", {}).get("public_path")
    if not raw:
        raise SystemExit("--include-public: config.json 의 external.public_path 가 비어 있습니다 (--public 경로 로 지정 가능).")
    src = Path(raw).expanduser()
    if not src.exists():
        raise SystemExit(f"--include-public: 공개 원본이 없습니다: {src.name} (config.json external.public_path 확인)")
    return src


def public_files(src: Path, only=None) -> list[tuple[str, bytes]]:
    """공개 원본(zip/폴더)에서 PUBLIC_USED 를 읽어 (zip 안 경로, 내용) 목록으로. 없는 파일은 알리고 건너뛴다."""
    names = [n for n in PUBLIC_USED if not only or n in set(only)]
    out = []
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            members = z.namelist()
            for n in names:
                hit = [m for m in members if m == n or m.endswith("/" + n)]
                if hit:
                    out.append((f"{PUBLIC_ARC}/{n}", z.read(hit[0])))
                else:
                    print(f"  ⚠ 공개 원본 zip 안에 {n} 이 없어 건너뜀")
    else:
        for n in names:
            p = src / n if src.is_dir() else src
            if p.is_file() and p.name == n:
                out.append((f"{PUBLIC_ARC}/{n}", p.read_bytes()))
            else:
                print(f"  ⚠ 공개 원본 폴더에 {n} 이 없어 건너뜀")
    return out


def public_source_note(entries: list[tuple[str, bytes]]) -> str:
    """data/public/SOURCE.md — 출처, 용도, 파일별 크기·SHA-256. 로컬 경로는 적지 않는다."""
    L = ["# 공개 원본 (data/public/)", "",
         f"- 출처: {PUBLIC_SOURCE}. KAMP 에서 내려받은 파일을 수정 없이 그대로 넣음 (아래 SHA-256 으로 대조 가능).",
         "- 용도: 해석·검증·외부 실험 전용 — `scripts/external_experiments.py`, `scripts/e1_policy_comparison.py`, "
         "`scripts/oneside_matched.py`, `scripts/link_public.py`, `scripts/verify_public_link.py` 가 읽는다. "
         "모델 학습·선택(`run_all.py`)은 이 폴더를 읽지 않는다.",
         "- 근거: 운영진 확인(2026-09-30) — 출처·사용 이유·방법을 보고서에 밝히면 사용 가능. 보고서 본문에 그렇게 적었다.",
         "- 대회 파일과의 연결: 대회 labeled 의 `Unnamed: 0` = 원본 `moldset_labeled.csv` 의 `Unnamed: 0` "
         "(`scripts/verify_public_link.py` 로 검증, 결과는 `outputs/verification/`).",
         "", "| 파일 | 크기 | SHA-256 |", "|---|---|---|"]
    for arc, data in entries:
        L.append(f"| `{Path(arc).name}` | {len(data) / 2 ** 20:.1f} MB | `{hashlib.sha256(data).hexdigest()}` |")
    L += ["", "실행 예: `python scripts/external_experiments.py --public data/public` "
              "(zip 을 푼 폴더에서, `--enable` 이 필요한 스크립트는 붙여서)"]
    return "\n".join(L) + "\n"


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
        found = set()
        for m in pat.finditer(text):
            s, nxt = m.group(), text[m.end():m.end() + 1]
            if kind == "이메일" and (s.startswith("+") or nxt in ("_", "(", "[")):
                continue  # 파이썬 행렬곱 `x@m.coef_` 같은 코드 — 이메일 아님
            found.add(s)
        hits += [(kind, m) for m in sorted(found) if m not in ALLOWED]
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
    ap.add_argument("--include-public", action="store_true",
                    help="KAMP 공개 원본(moldset_labeled/labeled_data/unlabeled_data.csv)을 data/public/ 에 넣고 "
                         "출처 메모 SOURCE.md 추가. outputs/verification 도 함께 포함")
    ap.add_argument("--public", default=None, help="--include-public 의 원본 위치 (zip 또는 폴더, 기본 config external.public_path)")
    ap.add_argument("--public-file", action="append",
                    help="--include-public 에서 넣을 파일을 제한 (여러 번 가능, 예: --public-file labeled_data.csv)")
    ap.add_argument("--word", action="append", help="추가로 검사할 단어 (여러 번 가능)")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    words = blind_words(args.word)
    files = collect(not args.no_data, args.include_verification or args.include_public)
    pub: list[tuple[str, bytes]] = []
    if args.include_public:
        pub = public_files(public_source(args.public), args.public_file)
        if pub:
            pub.append((f"{PUBLIC_ARC}/SOURCE.md", public_source_note(pub).encode("utf-8")))
    problems = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arc = f.relative_to(ROOT).as_posix()
            data = f.read_bytes()
            z.writestr(zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
            problems += [(arc, k, v) for k, v in scan(arc, data, words)]
        for arc, data in pub:
            z.writestr(zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
            # 공개 원본 CSV 는 우리가 만든 파일이 아니므로 이름만 검사 (SOURCE.md 는 내용까지)
            problems += [(arc, k, v) for k, v in scan(arc, b"" if Path(arc).name in PUBLIC_NAMES else data, words)]
    size = out.stat().st_size / 2 ** 20
    raw = (sum(f.stat().st_size for f in files) + sum(len(d) for _, d in pub)) / 2 ** 20
    names = [f.relative_to(ROOT).as_posix() for f in files]
    print(f"zip: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out.name} — 파일 {len(files) + len(pub)}개, "
          f"압축 {size:.1f} MB (원본 합계 {raw:.1f} MB)")
    data_in = sorted(n for n in names if n.startswith("data/") and not n.endswith(".gitkeep"))
    print(f"  학습 데이터: {len(data_in)}개 {[Path(n).name for n in data_in]}" + ("  (--no-data)" if args.no_data else ""))
    if args.include_public:
        pub_csv = [Path(a).name for a, _ in pub if Path(a).name != "SOURCE.md"]
        print(f"  공개 원본: {len(pub_csv)}개 {pub_csv} → {PUBLIC_ARC}/ (+ SOURCE.md 출처·SHA-256), outputs/verification 포함")
    else:
        print("  공개 원본: 넣지 않음 (--include-public 으로 포함)")
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
