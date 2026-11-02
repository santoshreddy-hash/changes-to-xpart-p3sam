import tempfile
import unittest
from pathlib import Path

from tests import helpers  # noqa: F401
from partseg.cli import build_parser, main


class CliTests(unittest.TestCase):
    """Argument and input errors must exit with code 2 before any GPU work starts."""

    def test_missing_input_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["segment", str(Path(tmp) / "missing.glb"), "-o", tmp, "-q"]), 2)

    def test_empty_folder_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["segment", tmp, "-o", tmp, "-q"]), 2)

    def test_bad_override_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / "a.glb"
            mesh.write_bytes(b"x")
            self.assertEqual(main(["segment", str(mesh), "-o", tmp, "--set", "polish.nope=1", "-q"]), 2)

    def test_bad_config_file_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "c.json"
            cfg.write_text("[]", encoding="utf-8")
            mesh = Path(tmp) / "a.glb"
            mesh.write_bytes(b"x")
            self.assertEqual(main(["segment", str(mesh), "-o", tmp, "--config", str(cfg), "-q"]), 2)

    def test_parser_defaults(self):
        args = build_parser().parse_args(["segment", "a.glb", "-o", "out"])
        self.assertEqual(args.preset, "balanced")
        self.assertFalse(args.no_preview)
        self.assertEqual(args.overrides, [])


if __name__ == "__main__":
    unittest.main()
