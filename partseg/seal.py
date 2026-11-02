"""Optional: close the open cut surfaces of each part so parts become watertight solids.

Every open boundary loop (the rim where a part was cut from its neighbours) is closed with a
triangle fan around the loop's centre, wound to match the faces next to it. This always closes the
mesh topologically; for the usual nearly flat cut loops the cap is also geometrically sensible.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def cap_open_loops(mesh: trimesh.Trimesh) -> int:
    """Close all boundary loops of ``mesh`` in place. Returns the number of triangles added."""
    faces = np.asarray(mesh.faces)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse.reshape(-1)] == 1]  # edges used by exactly one face, in face direction
    if len(boundary) == 0:
        return 0
    used, local = np.unique(boundary, return_inverse=True)
    local = local.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(local)), (local[:, 0], local[:, 1])), shape=(len(used),) * 2)
    n_loops, loop_of_vertex = connected_components(graph, directed=False)
    vertices = np.asarray(mesh.vertices)
    centres = np.stack([vertices[used[loop_of_vertex == k]].mean(axis=0) for k in range(n_loops)])
    centre_index = len(vertices) + np.arange(n_loops)
    loop_of_edge = loop_of_vertex[local[:, 0]]
    # the cap triangle must traverse each boundary edge in the opposite direction to the existing face
    caps = np.stack([centre_index[loop_of_edge], boundary[:, 1], boundary[:, 0]], axis=1)
    mesh.vertices = np.vstack([vertices, centres])
    mesh.faces = np.vstack([faces, caps])
    return int(len(caps))


def seal_part(sub: trimesh.Trimesh) -> dict:
    """Cap ``sub`` in place. Returns watertight status before/after and the triangles added."""
    was_closed = bool(sub.is_watertight)
    added = 0
    if not was_closed:
        try:
            added = cap_open_loops(sub)
        except Exception:  # an unusual part must not abort the export
            added = 0
    return {
        "watertight_before": was_closed,
        "watertight_sealed": bool(sub.is_watertight),
        "cap_faces": int(added),
    }


def seal_parts(mesh: trimesh.Trimesh, face_ids: np.ndarray, parts: list[dict]):
    """Return ({part id: sealed sub-mesh}, {part id: stats}). The input mesh is not modified."""
    meshes, stats = {}, {}
    for p in parts:
        idx = np.where(face_ids == p["id"])[0]
        sub = mesh.submesh([idx], append=True)
        stats[p["id"]] = seal_part(sub)
        meshes[p["id"]] = sub
    return meshes, stats
