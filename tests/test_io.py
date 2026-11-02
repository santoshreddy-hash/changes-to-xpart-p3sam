import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from tests import helpers
from partseg.errors import InvalidInputError
from partseg.io import load_mesh, validate_mesh


class LoadMeshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file(self):
        with self.assertRaises(InvalidInputError) as ctx:
            load_mesh(self.dir / "nope.glb")
        self.assertIn("not found", str(ctx.exception))

    def test_unsupported_extension(self):
        path = self.dir / "model.fbx"
        path.write_bytes(b"x")
        with self.assertRaises(InvalidInputError) as ctx:
            load_mesh(path)
        self.assertIn("Unsupported", str(ctx.exception))

    def test_corrupt_file(self):
        path = self.dir / "broken.glb"
        path.write_bytes(b"this is not a glb")
        with self.assertRaises(InvalidInputError):
            load_mesh(path)

    def test_loads_obj_and_reports_info(self):
        path = self.dir / "box.obj"
        trimesh.creation.box().export(str(path))
        mesh, info = load_mesh(path)
        self.assertEqual(info["faces"], len(mesh.faces))
        self.assertEqual(info["format"], "obj")
        self.assertTrue(info["watertight"])

    def test_scene_nodes_are_merged_with_their_transforms(self):
        path = self.dir / "scene.glb"
        helpers.write_scene_glb(path)
        mesh, info = load_mesh(path)
        self.assertGreater(len(mesh.faces), 12)
        self.assertGreater(mesh.bounds[1][0] - mesh.bounds[0][0], 3.0)  # sphere moved 3 units along x

    def test_face_limit(self):
        with self.assertRaises(InvalidInputError) as ctx:
            validate_mesh(trimesh.creation.box(), max_faces=5)
        self.assertIn("above the limit", str(ctx.exception))

    def test_nan_vertices(self):
        mesh = trimesh.creation.box()
        mesh.vertices[0, 0] = np.nan
        with self.assertRaises(InvalidInputError) as ctx:
            validate_mesh(mesh)
        self.assertIn("NaN", str(ctx.exception))

    def test_too_few_faces(self):
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
        with self.assertRaises(InvalidInputError):
            validate_mesh(mesh)

    def test_point_cloud_is_rejected(self):
        with self.assertRaises(InvalidInputError):
            validate_mesh(trimesh.PointCloud(np.random.rand(10, 3)))


if __name__ == "__main__":
    unittest.main()
