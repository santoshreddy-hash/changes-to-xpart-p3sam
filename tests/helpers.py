"""Synthetic meshes shared by the tests (no GPU or model weights needed)."""

import os
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def subdivided_box(extents=(2.0, 1.0, 1.0), levels=4):
    mesh = trimesh.creation.box(extents=extents)
    for _ in range(levels):
        mesh = mesh.subdivide()
    return mesh


def split_labels(mesh, x_cut=0.0):
    """Two clean halves along x."""
    return (mesh.triangles_center[:, 0] > x_cut).astype(np.int64)


def jagged_labels(mesh, x_cut=0.3, noise=0.07, seed=0):
    """Two halves with a saw-toothed boundary and stray single-face islands."""
    rng = np.random.default_rng(seed)
    c = mesh.triangles_center
    return (c[:, 0] > x_cut + noise * rng.standard_normal(len(c))).astype(np.int64)


def two_cracked_slabs():
    """Two touching boxes whose vertices are NOT shared (a crack in the mesh), used for one label."""
    a = trimesh.creation.box(extents=(1, 1, 0.2))
    b = trimesh.creation.box(extents=(1, 1, 0.2))
    b.apply_translation([1.0, 0, 0])
    return trimesh.util.concatenate([a, b])


def write_scene_glb(path):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=(1, 1, 1)), node_name="a")
    moved = trimesh.creation.icosphere(subdivisions=2)
    scene.add_geometry(moved, node_name="b", transform=trimesh.transformations.translation_matrix([3, 0, 0]))
    scene.export(str(path))
