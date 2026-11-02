"""Write the structured output of a segmentation run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from .config import ExportConfig
from .polish import MeshGraph
from .render import part_palette

SCHEMA_VERSION = 1


def part_name(index: int) -> str:
    return f"part_{index:03d}"


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(v) for v in rgb])


def build_parts_table(mesh: trimesh.Trimesh, face_ids: np.ndarray, graph: MeshGraph) -> list[dict]:
    """One record per part: geometry stats, colour, neighbours and shared boundary length."""
    labels = np.unique(face_ids)
    palette = (part_palette(len(labels)) * 255).astype(np.uint8)
    a, b = graph.adj[:, 0], graph.adj[:, 1]
    la, lb = face_ids[a], face_ids[b]
    cross = la != lb
    contacts: dict[int, dict[int, list]] = {int(l): {} for l in labels}

    def _link(x, y, kind, length):
        for u, v in ((int(x), int(y)), (int(y), int(x))):
            entry = contacts[u].setdefault(v, ["proximity", 0.0])
            if kind == "edge":
                entry[0] = "edge"
                entry[1] += length

    for x, y, length in zip(la[cross], lb[cross], graph.edge_len[cross]):
        _link(x, y, "edge", float(length))
    # separate bodies that touch (common in assemblies) share no mesh edge: link them by vertex proximity
    from scipy.spatial import cKDTree

    vert_label = np.full(len(graph.vw), -1, dtype=np.int64)
    vert_label[graph.fw.ravel()] = np.repeat(face_ids, 3)
    used = np.unique(graph.fw.ravel())
    pts, vlab = graph.vw[used], vert_label[used]
    tol = float(np.median(graph.vert_scale[used])) * 1.5
    k = min(12, len(used))
    dist, nbr = cKDTree(pts).query(pts, k=k, distance_upper_bound=tol)
    valid = np.isfinite(dist)
    other = vlab[np.where(valid, nbr, 0)]
    differs = valid & (other != vlab[:, None])
    if differs.any():
        pairs = np.unique(np.stack([np.broadcast_to(vlab[:, None], other.shape)[differs], other[differs]], axis=1), axis=0)
        for x, y in pairs:
            _link(x, y, "proximity", 0.0)

    from .polish import label_components

    comp = label_components(face_ids, graph)
    parts = []
    triangles = mesh.triangles
    for i, label in enumerate(labels):
        sel = face_ids == label
        pts = triangles[sel].reshape(-1, 3)
        area = float(graph.areas[sel].sum())
        parts.append({
            "id": int(label),
            "name": part_name(int(label)),
            "color": _hex(palette[i % len(palette)]),
            "faces": int(sel.sum()),
            "area": area,
            "area_fraction": area / graph.total_area,
            "bbox_min": [float(v) for v in pts.min(0)],
            "bbox_max": [float(v) for v in pts.max(0)],
            "centroid": [float(v) for v in graph.centroids[sel].mean(0)],
            "connected_pieces": int(len(np.unique(comp[sel]))),
            "neighbors": [
                {"id": n, "contact": kind, "shared_boundary": round(length, 6)}
                for n, (kind, length) in sorted(contacts[int(label)].items())
            ],
        })
    return parts


def segmented_mesh(mesh: trimesh.Trimesh, face_ids: np.ndarray) -> trimesh.Trimesh:
    """Single mesh with one flat colour per part and unshared vertices, so colours never blend at boundaries."""
    labels = np.unique(face_ids[face_ids >= 0])
    palette = (part_palette(len(labels)) * 255).astype(np.uint8)
    rgba = np.full((len(face_ids), 4), 255, dtype=np.uint8)
    rgba[:, :3] = 120
    for i, label in enumerate(labels):
        rgba[face_ids == label, :3] = palette[i % len(palette)]
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)].reshape(-1, 3)
    out = trimesh.Trimesh(vertices=tri, faces=np.arange(len(tri)).reshape(-1, 3), process=False)
    out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=np.repeat(rgba, 3, axis=0))
    return out


def _flat_color_mesh(mesh: trimesh.Trimesh, idx: np.ndarray, rgb) -> trimesh.Trimesh:
    sub = mesh.submesh([idx], append=True)
    sub.visual = trimesh.visual.ColorVisuals(
        mesh=sub, face_colors=np.tile(np.array([*rgb, 255], dtype=np.uint8), (len(sub.faces), 1))
    )
    return sub


def write_outputs(out_dir: Path, stem: str, mesh: trimesh.Trimesh, face_ids: np.ndarray, graph: MeshGraph,
                  parts: list[dict], report: dict, cfg: ExportConfig, preview_png: np.ndarray | None) -> dict[str, str]:
    """Write every artefact and return {kind: path}. Files are written atomically-ish (temp then rename)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "parts_glb": out_dir / f"{stem}_parts.glb",
        "segmented_glb": out_dir / f"{stem}_segmented.glb",
        "face_ids": out_dir / f"{stem}_face_ids.npy",
        "parts_json": out_dir / f"{stem}_parts.json",
        "report_json": out_dir / f"{stem}_report.json",
    }
    colors = {p["id"]: tuple(int(p["color"][i:i + 2], 16) for i in (1, 3, 5)) for p in parts}

    # one named node per part (flat colour, or the source material when keep_materials is set)
    scene = trimesh.Scene()
    for p in parts:
        idx = np.where(face_ids == p["id"])[0]
        sub = mesh.submesh([idx], append=True) if cfg.keep_materials else _flat_color_mesh(mesh, idx, colors[p["id"]])
        scene.add_geometry(sub, node_name=p["name"], geom_name=p["name"])
    scene.metadata.update({"generator": "partseg", "schema_version": SCHEMA_VERSION, "num_parts": len(parts)})
    scene.export(str(paths["parts_glb"]))

    segmented_mesh(mesh, face_ids).export(str(paths["segmented_glb"]))

    if cfg.seal:  # opt-in: cap the open cut surfaces so each part is a closed solid where possible
        from .seal import seal_parts

        sealed, seal_stats = seal_parts(mesh, face_ids, parts)
        sealed_scene = trimesh.Scene()
        for p in parts:
            sub = sealed[p["id"]]
            sub.visual = trimesh.visual.ColorVisuals(
                mesh=sub, face_colors=np.tile(np.array([*colors[p["id"]], 255], dtype=np.uint8), (len(sub.faces), 1)))
            sealed_scene.add_geometry(sub, node_name=p["name"], geom_name=p["name"])
            p.update(seal_stats[p["id"]])
        paths["parts_sealed_glb"] = out_dir / f"{stem}_parts_sealed.glb"
        sealed_scene.export(str(paths["parts_sealed_glb"]))

    np.save(paths["face_ids"], face_ids.astype(np.int32))
    paths["parts_json"].write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "source": report.get("input", {}),
        "num_parts": len(parts),
        "parts": parts,
    }, indent=2), encoding="utf-8")
    paths["report_json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    if preview_png is not None:
        import cv2

        paths["preview"] = out_dir / f"{stem}_preview.png"
        cv2.imwrite(str(paths["preview"]), preview_png[:, :, ::-1])
    return {k: str(v) for k, v in paths.items()}
