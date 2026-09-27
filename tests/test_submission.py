"""제출 zip 스크립트(scripts/make_submission.py): 제외 규칙과 블라인드 검사 함수."""
from __future__ import annotations

import sys
import unittest

from _common import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import make_submission as ms  # noqa: E402


class TestSubmission(unittest.TestCase):
    def test_excluded_paths(self):
        rels = {p.relative_to(ROOT).as_posix() for p in ms.collect(include_data=False, include_verification=False)}
        for bad in (".git/", "data/raw/moldset", "_incoming/", ".venv/", "__pycache__/", "outputs/synthetic/",
                    "outputs/verification/", "dist/", "data/synthetic/"):
            self.assertFalse(any(r.startswith(bad) or f"/{bad}" in f"/{r}" for r in rels), bad)
        self.assertIn("run_all.py", rels)
        self.assertIn("config.json", rels)

    def test_data_only_competition_csvs(self):
        rels = [p.relative_to(ROOT).as_posix() for p in ms.collect(include_data=True, include_verification=False)]
        data = sorted(r for r in rels if r.startswith("data/") and not r.endswith(".gitkeep"))
        self.assertTrue(set(data) <= ms.competition_files(), data)
        self.assertFalse([r for r in rels if r.split("/")[-1] in ms.PUBLIC_NAMES])

    def test_scan_detects_personal_strings(self):
        home = "/" + "Users" + "/" + "someone/x.csv"
        win = "C:" + "\\" + "Users" + "\\" + "someone" + "\\" + "a"
        mail = "abc" + "@" + "example.com"
        hits = ms.scan("a.txt", f"{home} {win} {mail} 홍길동".encode(), ["홍길동"])
        kinds = {k for k, _ in hits}
        self.assertEqual(kinds, {"로컬경로", "이메일", "지정 단어"})
        self.assertEqual(ms.scan("b.txt", ("co-author " + "noreply" + "@" + "anthropic.com").encode(), []), [])
        self.assertEqual(ms.scan("홍길동.txt", b"", ["홍길동"]), [("파일 이름", "홍길동")])
        self.assertEqual(ms.scan("img.png", bytes([0xFF, 0xD8, 0xFF]), ["x"]), [])

    def test_script_itself_is_clean(self):
        src = (ROOT / "scripts" / "make_submission.py").read_bytes()
        self.assertEqual([k for k, _ in ms.scan("make_submission.py", src, [])], [])


if __name__ == "__main__":
    unittest.main()
