import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from tests import helpers
from partseg.config import ExportConfig
from partseg.export import build_parts_table, part_name, segmented_mesh, write_outputs
from partseg.polish import MeshGraph


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.mesh = helpers.subdivided_box(levels=3)
        self.graph = MeshGraph(self.mesh)
        self.ids = helpers.split_labels(self.mesh)
        self.parts = build_parts_table(self.mesh, self.ids, self.graph)

    def test_parts_table(self):
        self.assertEqual(len(self.parts), 2)
        self.assertAlmostEqual(sum(p["area_fraction"] for p in self.parts), 1.0, places=6)
        self.assertEqual(sum(p["faces"] for p in self.parts), len(self.mesh.faces))
        for p in self.parts:
            self.assertEqual(p["name"], part_name(p["id"]))
            self.assertEqual(p["connected_pieces"], 1)
            self.assertRegex(p["color"], r"^#[0-9a-f]{6}$")
        # the two halves share a mesh edge boundary
        self.assertEqual([n["id"] for n in self.parts[0]["neighbors"]], [self.parts[1]["id"]])
        self.assertEqual(self.parts[0]["neighbors"][0]["contact"], "edge")
        self.assertGreater(self.parts[0]["neighbors"][0]["shared_boundary"], 0)

    def test_touching_separate_bodies_are_linked_by_proximity(self):
        mesh = helpers.two_cracked_slabs()
        graph = MeshGraph(mesh)
        ids = (mesh.triangles_center[:, 0] > 0.5).astype(np.int64)
        parts = build_parts_table(mesh, ids, graph)
        self.assertEqual(parts[0]["neighbors"][0]["contact"], "proximity")

    def test_segmented_mesh_has_one_flat_colour_per_part(self):
        seg = segmented_mesh(self.mesh, self.ids)
        self.assertEqual(len(seg.faces), len(self.mesh.faces))
        colours = np.unique(np.asarray(seg.visual.face_colors)[:, :3], axis=0)
        self.assertEqual(len(colours), 2)

    def test_write_outputs(self):
        report = {"input": {"name": "box.glb"}, "num_parts": 2}
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_outputs(Path(tmp) / "nested" / "out", "box", self.mesh, self.ids, self.graph, self.parts,
                                  report, ExportConfig(), np.zeros((8, 8, 3), dtype=np.uint8))
            for kind in ("parts_glb", "segmented_glb", "face_ids", "parts_json", "report_json", "preview"):
                self.assertTrue(Path(paths[kind]).is_file(), kind)
            scene = trimesh.load(paths["parts_glb"])
            self.assertEqual(sorted(scene.geometry.keys()), ["part_000", "part_001"])
            self.assertEqual(sum(len(g.faces) for g in scene.geometry.values()), len(self.mesh.faces))
            doc = json.loads(Path(paths["parts_json"]).read_text(encoding="utf-8"))
            self.assertEqual(doc["schema_version"], 1)
            self.assertEqual(doc["num_parts"], 2)
            self.assertEqual({"id", "name", "color", "faces", "area", "area_fraction", "bbox_min", "bbox_max",
                              "centroid", "connected_pieces", "neighbors"}, set(doc["parts"][0]))
            ids = np.load(paths["face_ids"])
            self.assertEqual(ids.dtype, np.int32)
            np.testing.assert_array_equal(ids, self.ids)

    def test_keep_materials_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_outputs(Path(tmp), "box", self.mesh, self.ids, self.graph, self.parts,
                                  {"input": {}}, ExportConfig(keep_materials=True, preview=False), None)
            self.assertNotIn("preview", paths)
            self.assertEqual(len(trimesh.load(paths["parts_glb"]).geometry), 2)


if __name__ == "__main__":
    unittest.main()
