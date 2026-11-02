"""End-to-end test on a real mesh. Needs an NVIDIA GPU and the P3-SAM weights; skipped otherwise."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers

MESH = Path(helpers.ROOT) / "uploads" / "bedside_wood_table_vintage_furniture_7aa2f793.glb"


def _gpu_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@unittest.skipUnless(_gpu_available() and MESH.exists(), "needs a CUDA GPU and the sample mesh")
class PipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from partseg import SegmentConfig, load_model, segment_file

        cls.model = load_model()
        cls.cfg = SegmentConfig.preset("balanced")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.first = segment_file(MESH, Path(cls.tmp.name) / "a", cls.cfg, model=cls.model)
        cls.second = segment_file(MESH, Path(cls.tmp.name) / "b", cls.cfg, model=cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_labels_are_complete_and_contiguous(self):
        ids = self.first.face_ids
        self.assertEqual(len(ids), len(self.first.mesh.faces))
        self.assertEqual(sorted(np.unique(ids)), list(range(int(ids.max()) + 1)))
        self.assertGreaterEqual(len(self.first.parts), 4)

    def test_part_zero_is_the_largest(self):
        areas = [p["area"] for p in self.first.parts]
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_parts_are_single_pieces(self):
        self.assertEqual(self.first.metrics["disconnected_parts"], 0)
        self.assertEqual(self.first.metrics["tiny_fragments"], 0)

    def test_two_runs_give_identical_labels(self):
        np.testing.assert_array_equal(self.first.face_ids, self.second.face_ids)

    def test_outputs_are_written_and_consistent(self):
        for kind in ("parts_glb", "segmented_glb", "face_ids", "parts_json", "report_json", "preview"):
            self.assertTrue(Path(self.first.outputs[kind]).is_file(), kind)
        doc = json.loads(Path(self.first.outputs["parts_json"]).read_text(encoding="utf-8"))
        self.assertEqual(doc["num_parts"], len(self.first.parts))
        report = json.loads(Path(self.first.outputs["report_json"]).read_text(encoding="utf-8"))
        self.assertEqual(report["warnings"], [])
        self.assertIn("p3sam", report["stage_seconds"])

    def test_a_missing_file_raises_a_clear_error(self):
        from partseg import InvalidInputError, segment_file

        with self.assertRaises(InvalidInputError):
            segment_file(MESH.with_name("does_not_exist.glb"), self.tmp.name, self.cfg, model=self.model)


if __name__ == "__main__":
    unittest.main()
