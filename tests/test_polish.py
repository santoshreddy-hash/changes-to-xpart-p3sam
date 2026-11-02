import unittest

import numpy as np
import trimesh

from tests import helpers
from partseg.config import PolishConfig
from partseg.polish import (
    GCO_AVAILABLE,
    MeshGraph,
    boundary_stats,
    count_disconnected,
    crease_graphcut_polish,
    label_components,
    merge_small_regions,
    planar_fraction,
    polish_labels,
    split_disconnected,
)


class ComponentTests(unittest.TestCase):
    def test_a_crack_does_not_split_a_part(self):
        mesh = helpers.two_cracked_slabs()
        graph = MeshGraph(mesh)
        ids = np.zeros(len(mesh.faces), dtype=np.int64)
        strict = label_components(ids, graph, close_cracks=False)
        tolerant = label_components(ids, graph, close_cracks=True)
        self.assertGreater(len(np.unique(strict)), 1)    # mesh edges alone see fragments (non-manifold contact)
        self.assertEqual(len(np.unique(tolerant)), 1)    # but the bodies touch, so it is one part

    def test_distant_bodies_stay_separate(self):
        a = trimesh.creation.box()
        b = trimesh.creation.box()
        b.apply_translation([6, 0, 0])
        mesh = trimesh.util.concatenate([a, b])
        graph = MeshGraph(mesh)
        out = split_disconnected(np.zeros(len(mesh.faces), dtype=np.int64), graph)
        self.assertEqual(len(np.unique(out)), 2)

    def test_split_orders_parts_by_area(self):
        mesh = helpers.subdivided_box(levels=3)
        graph = MeshGraph(mesh)
        ids = (mesh.triangles_center[:, 0] > 0.6).astype(np.int64)  # small cap vs big remainder
        out = split_disconnected(ids, graph)
        areas = [graph.areas[out == l].sum() for l in np.unique(out)]
        self.assertEqual(areas, sorted(areas, reverse=True))


class FragmentTests(unittest.TestCase):
    def test_tiny_fragment_is_absorbed(self):
        mesh = helpers.subdivided_box(levels=4)
        graph = MeshGraph(mesh)
        ids = np.zeros(len(mesh.faces), dtype=np.int64)
        ids[helpers.split_labels(mesh) == 1] = 1
        stray = np.where(ids == 0)[0][:3]
        ids[stray] = 2                                    # three stray faces carrying their own label
        out = merge_small_regions(ids, graph, min_area_frac=0.003, min_faces=12)
        self.assertEqual(len(np.unique(out)), 2)

    def test_large_parts_survive(self):
        mesh = helpers.subdivided_box(levels=4)
        graph = MeshGraph(mesh)
        out = merge_small_regions(helpers.split_labels(mesh), graph, min_area_frac=0.003, min_faces=12)
        self.assertEqual(len(np.unique(out)), 2)


@unittest.skipUnless(GCO_AVAILABLE, "gco is not installed")
class GraphCutTests(unittest.TestCase):
    def test_graph_cut_straightens_a_jagged_boundary(self):
        # Regression guard: gco silently returned its input for int32 arrays, which made this a no-op.
        mesh = helpers.subdivided_box(levels=4)
        graph = MeshGraph(mesh)
        ids = helpers.jagged_labels(mesh)
        before = boundary_stats(ids, graph)["boundary_len_pct"]
        out = crease_graphcut_polish(ids, graph, band=6, stay_cost=0.06)
        after = boundary_stats(out, graph)["boundary_len_pct"]
        self.assertGreater((out != ids).sum(), 0)
        self.assertLess(after, before * 0.85)


class PlanarityTests(unittest.TestCase):
    def test_slab_is_planar_and_sphere_is_not(self):
        slab = trimesh.creation.box(extents=(4, 4, 0.1))
        sphere = trimesh.creation.icosphere(subdivisions=3)
        g_slab, g_sphere = MeshGraph(slab), MeshGraph(sphere)
        self.assertGreater(planar_fraction(np.zeros(len(slab.faces), int), 0, np.asarray(slab.face_normals), g_slab), 0.85)
        self.assertLess(planar_fraction(np.zeros(len(sphere.faces), int), 0, np.asarray(sphere.face_normals), g_sphere), 0.5)


class PolishLabelsTests(unittest.TestCase):
    def setUp(self):
        self.mesh = helpers.subdivided_box(levels=4)
        self.graph = MeshGraph(self.mesh)
        self.noisy = helpers.jagged_labels(self.mesh)

    def test_result_is_clean_and_connected(self):
        out = polish_labels(self.noisy, self.graph, PolishConfig())
        self.assertEqual(len(np.unique(out)), 2)
        self.assertEqual(count_disconnected(out, self.graph), 0)
        self.assertEqual(out.min(), 0)
        areas = [self.graph.areas[out == l].sum() for l in np.unique(out)]
        self.assertEqual(areas, sorted(areas, reverse=True))   # label 0 is the largest part

    @unittest.skipUnless(GCO_AVAILABLE, "gco is not installed")
    def test_boundary_gets_shorter(self):
        before = boundary_stats(self.noisy, self.graph)["boundary_len_pct"]
        out = polish_labels(self.noisy, self.graph, PolishConfig())
        self.assertLess(boundary_stats(out, self.graph)["boundary_len_pct"], before * 0.7)

    def test_is_deterministic(self):
        a = polish_labels(self.noisy, self.graph, PolishConfig())
        b = polish_labels(self.noisy, self.graph, PolishConfig())
        np.testing.assert_array_equal(a, b)

    def test_every_face_keeps_a_label(self):
        out = polish_labels(self.noisy, self.graph, PolishConfig())
        self.assertEqual(out.shape, (len(self.mesh.faces),))
        self.assertTrue((out >= 0).all())


if __name__ == "__main__":
    unittest.main()
