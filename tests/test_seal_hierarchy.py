import unittest

import numpy as np

from tests import helpers
from partseg.errors import InvalidInputError
from partseg.export import build_parts_table
from partseg.hierarchy import merge_to_target, part_feature_means
from partseg.polish import MeshGraph
from partseg.seal import seal_parts


class SealTests(unittest.TestCase):
    def test_halves_of_a_box_become_watertight_when_sealed(self):
        mesh = helpers.subdivided_box(levels=3)
        ids = helpers.split_labels(mesh)
        parts = build_parts_table(mesh, ids, MeshGraph(mesh))
        sealed, stats = seal_parts(mesh, ids, parts)
        for p in parts:
            self.assertFalse(stats[p["id"]]["watertight_before"])   # a cut half is open
            self.assertTrue(stats[p["id"]]["watertight_sealed"])
            self.assertGreater(stats[p["id"]]["cap_faces"], 0)
            self.assertTrue(sealed[p["id"]].is_watertight)

    def test_sealing_does_not_modify_the_input_mesh(self):
        mesh = helpers.subdivided_box(levels=2)
        faces_before = len(mesh.faces)
        ids = helpers.split_labels(mesh)
        seal_parts(mesh, ids, build_parts_table(mesh, ids, MeshGraph(mesh)))
        self.assertEqual(len(mesh.faces), faces_before)


class MergeToTargetTests(unittest.TestCase):
    def setUp(self):
        # four parts in a chain 0-1-2-3; parts 1 and 2 have the most similar features
        self.parts = [
            {"id": 0, "neighbors": [{"id": 1}]},
            {"id": 1, "neighbors": [{"id": 0}, {"id": 2}]},
            {"id": 2, "neighbors": [{"id": 1}, {"id": 3}]},
            {"id": 3, "neighbors": [{"id": 2}]},
        ]
        self.face_ids = np.repeat(np.arange(4), 5)
        self.feats = np.array([[1.0, 0.0], [0.0, 1.0], [0.05, 1.0], [1.0, 0.2]])

    def test_merges_the_most_similar_neighbours_first(self):
        out = merge_to_target(self.face_ids, self.parts, self.feats, 3)
        self.assertEqual(len(np.unique(out)), 3)
        self.assertEqual(out[5], out[10])           # parts 1 and 2 merged
        self.assertNotEqual(out[0], out[5])

    def test_target_not_below_current_is_a_no_op(self):
        np.testing.assert_array_equal(merge_to_target(self.face_ids, self.parts, self.feats, 9), self.face_ids)

    def test_invalid_target(self):
        with self.assertRaises(InvalidInputError):
            merge_to_target(self.face_ids, self.parts, self.feats, 0)

    def test_feature_means(self):
        feats = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
        means = part_feature_means(feats, np.array([0, 0, 1]), 2)
        np.testing.assert_allclose(means, [[2.0, 0.0], [0.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
