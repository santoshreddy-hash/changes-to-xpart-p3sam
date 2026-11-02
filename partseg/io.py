"""Mesh loading and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .errors import InvalidInputError

SUPPORTED_EXTENSIONS = (".glb", ".gltf", ".obj", ".ply", ".stl", ".off")


def load_mesh(path: str | Path, max_faces: int = 2_000_000) -> tuple[trimesh.Trimesh, dict]:
    """
    Load any supported file as one triangle mesh in world coordinates.

    Multi-node scenes are merged with their node transforms applied. Raises
    InvalidInputError with an actionable message for anything unusable.
    """
    path = Path(path)
    if not path.exists():
        raise InvalidInputError(f"Input file not found: {path}")
    if not path.is_file():
        raise InvalidInputError(f"Input path is not a file: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise InvalidInputError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    try:
        mesh = trimesh.load(str(path), force="mesh")
    except Exception as exc:  # trimesh raises many unrelated types for corrupt files
        raise InvalidInputError(f"Could not read '{path.name}': {exc}") from exc
    return validate_mesh(mesh, name=path.name, max_faces=max_faces, source=path)


def validate_mesh(mesh, name: str = "mesh", max_faces: int = 2_000_000, source=None) -> tuple[trimesh.Trimesh, dict]:
    if not isinstance(mesh, trimesh.Trimesh):
        raise InvalidInputError(f"'{name}' did not contain a triangle mesh")
    n_faces = len(mesh.faces)
    if n_faces == 0 or len(mesh.vertices) == 0:
        raise InvalidInputError(f"'{name}' has no faces")
    if n_faces < 4:
        raise InvalidInputError(f"'{name}' has only {n_faces} faces; at least 4 are needed")
    if n_faces > max_faces:
        raise InvalidInputError(
            f"'{name}' has {n_faces:,} faces, above the limit of {max_faces:,}. "
            "Decimate it first or raise max_faces in the config."
        )
    if not np.isfinite(mesh.vertices).all():
        raise InvalidInputError(f"'{name}' contains NaN or infinite vertex coordinates")
    extent = mesh.bounds[1] - mesh.bounds[0]
    if not np.all(extent > 0) and np.count_nonzero(extent > 1e-12) < 2:
        raise InvalidInputError(f"'{name}' is degenerate (zero size along two or more axes)")
    info = {
        "name": name,
        "path": str(source) if source is not None else None,
        "format": Path(name).suffix.lower().lstrip("."),
        "faces": int(n_faces),
        "vertices": int(len(mesh.vertices)),
        "bounds": [[float(v) for v in mesh.bounds[0]], [float(v) for v in mesh.bounds[1]]],
        "watertight": bool(mesh.is_watertight),
        "has_texture": bool(getattr(mesh.visual, "kind", None) == "texture"),
    }
    return mesh, info
