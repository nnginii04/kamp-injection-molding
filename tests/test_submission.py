"""제출 zip 스크립트(scripts/make_submission.py): 제외 규칙과 블라인드 검사 함수."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

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
        code = "z = (c" + "@" + "m.components_) + a" + "@" + "lr.coef_[0] + x" + "@" + "w.T(1)"
        self.assertEqual(ms.scan("c.py", code.encode(), []), [])  # 행렬곱 @ 는 이메일이 아님
        self.assertEqual(ms.scan("홍길동.txt", b"", ["홍길동"]), [("파일 이름", "홍길동")])
        self.assertEqual(ms.scan("img.png", bytes([0xFF, 0xD8, 0xFF]), ["x"]), [])

    def test_public_files_folder_zip_and_note(self):
        body = b"a,b\n1,2\n"
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for n in ms.PUBLIC_USED:
                (d / n).write_bytes(body)
            (d / "supervised_label_cn7.csv").write_bytes(b"x")  # 쓰지 않는 공개 파일은 넣지 않음
            got = ms.public_files(d)
            self.assertEqual([a for a, _ in got], [f"{ms.PUBLIC_ARC}/{n}" for n in ms.PUBLIC_USED])
            self.assertEqual([a for a, _ in ms.public_files(d, ["labeled_data.csv"])],
                             [f"{ms.PUBLIC_ARC}/labeled_data.csv"])
            zp = d / "pub.zip"
            with zipfile.ZipFile(zp, "w") as z:
                for n in ms.PUBLIC_USED:
                    z.writestr(f"04. Dataset_Molding/{n}", body)
            self.assertEqual([(a, b) for a, b in ms.public_files(zp)], got)
            (d / "unlabeled_data.csv").unlink()
            self.assertEqual(len(ms.public_files(d)), 2)  # 없는 파일은 건너뜀
            note = ms.public_source_note(got)
            self.assertIn(hashlib.sha256(body).hexdigest(), note)
            self.assertIn("moldset_labeled.csv", note)
            self.assertEqual(ms.scan("SOURCE.md", note.encode(), []), [])  # 메모에 경로·메일 없음
        self.assertTrue(set(ms.PUBLIC_USED) <= ms.PUBLIC_NAMES)

    def test_script_itself_is_clean(self):
        src = (ROOT / "scripts" / "make_submission.py").read_bytes()
        self.assertEqual([k for k, _ in ms.scan("make_submission.py", src, [])], [])


if __name__ == "__main__":
    unittest.main()
