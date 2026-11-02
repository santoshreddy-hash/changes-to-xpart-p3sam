"""Procedural assemblies with exact ground-truth part labels.

Each model is built from primitives; every triangle remembers which primitive it came from, so a
segmentation can be scored objectively. Synthetic parts are cleaner than real assets, so treat the
scores as a regression/ablation signal, not as a claim about real-world accuracy.
"""

from __future__ import annotations

import numpy as np
import trimesh


def _box(size, at, subdiv=3):
    m = trimesh.creation.box(extents=size)
    for _ in range(subdiv):
        m = m.subdivide()
    m.apply_translation(at)
    return m


def _cyl(radius, height, at, axis="z", sections=40):
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    if axis == "x":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(at)
    return m.subdivide().subdivide() if len(m.faces) < 1500 else m


def _sphere(radius, at, sub=3):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=radius)
    m.apply_translation(at)
    return m


def _capsule(radius, height, at, axis="z"):
    m = trimesh.creation.capsule(radius=radius, height=height, count=[24, 24])
    if axis == "x":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(at)
    return m


def _cone(radius, height, at, sections=40):
    m = trimesh.creation.cone(radius=radius, height=height, sections=sections)
    m.apply_translation(at)
    return m.subdivide().subdivide()


def assemble(parts: list[trimesh.Trimesh]):
    labels = np.concatenate([np.full(len(p.faces), i, dtype=np.int64) for i, p in enumerate(parts)])
    mesh = trimesh.util.concatenate(parts)
    return mesh, labels


def table():
    return assemble([_box((2.0, 1.2, 0.12), (0, 0, 1.0)),
                     *[_cyl(0.07, 1.0, (sx * 0.85, sy * 0.45, 0.5)) for sx in (-1, 1) for sy in (-1, 1)]])


def chair():
    return assemble([_box((1.0, 1.0, 0.1), (0, 0, 0.8)), _box((1.0, 0.1, 1.0), (0, -0.45, 1.35)),
                     *[_cyl(0.06, 0.8, (sx * 0.4, sy * 0.4, 0.4)) for sx in (-1, 1) for sy in (-1, 1)]])


def cabinet():
    body = _box((1.6, 0.7, 1.4), (0, 0, 0.9))
    doors = [_box((0.72, 0.05, 1.2), (sx * 0.39, 0.37, 0.9)) for sx in (-1, 1)]
    handles = [_cyl(0.035, 0.25, (sx * 0.1, 0.43, 0.9), axis="z") for sx in (-1, 1)]
    feet = [_cyl(0.07, 0.2, (sx * 0.65, sy * 0.25, 0.1)) for sx in (-1, 1) for sy in (-1, 1)]
    return assemble([body, *doors, *handles, *feet])


def lamp():
    return assemble([_cyl(0.45, 0.08, (0, 0, 0.04), sections=64), _cyl(0.04, 1.6, (0, 0, 0.88)),
                     _cone(0.5, 0.55, (0, 0, 1.95))])


def robot():
    return assemble([_box((0.9, 0.5, 1.1), (0, 0, 1.2)), _sphere(0.33, (0, 0, 2.0)),
                     _capsule(0.12, 0.8, (-0.65, 0, 1.2)), _capsule(0.12, 0.8, (0.65, 0, 1.2)),
                     _capsule(0.14, 0.8, (-0.25, 0, 0.45)), _capsule(0.14, 0.8, (0.25, 0, 0.45))])


def rocket():
    return assemble([_capsule(0.4, 1.8, (0, 0, 1.2)), _cone(0.4, 0.8, (0, 0, 2.5)),
                     *[_box((0.05, 0.5, 0.7), (np.cos(a) * 0.55, np.sin(a) * 0.55, 0.45)) for a in (0, 2.09, 4.19)]])


def dumbbell():
    return assemble([_sphere(0.4, (-0.9, 0, 0.4)), _sphere(0.4, (0.9, 0, 0.4)), _cyl(0.09, 1.4, (0, 0, 0.4), axis="x")])


def stool():
    return assemble([_cyl(0.5, 0.1, (0, 0, 0.75), sections=64),
                     *[_cyl(0.05, 0.75, (np.cos(a) * 0.32, np.sin(a) * 0.32, 0.37)) for a in (0.5, 2.6, 4.7)],
                     _cyl(0.3, 0.04, (0, 0, 0.3), sections=64)])


BUILDERS = {"table": table, "chair": chair, "cabinet": cabinet, "lamp": lamp, "robot": robot,
            "rocket": rocket, "dumbbell": dumbbell, "stool": stool}


def build_dataset(names=None):
    out = []
    for name, build in BUILDERS.items():
        if names and name not in names:
            continue
        mesh, labels = build()
        out.append({"name": name, "mesh": mesh, "labels": labels, "gt_parts": int(labels.max()) + 1})
    return out
