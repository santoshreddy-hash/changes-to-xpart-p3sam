"""Run baseline vs enhanced vs final (re-prompt + clean-boundary polish) on the 8-mesh suite."""

import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
import trimesh

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import final_pipeline as fp  # noqa: E402
import part_polish as pp  # noqa: E402
import render_raster as rr  # noqa: E402
import hierarchical_clustering as hc  # noqa: E402
from multimesh_suite_base import TEST_MESHES  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "final_pipeline_suite")
CACHE = os.path.join(OUT, "cache")
SUB_KW = dict(stab_thresh=0.75, min_iou=0.15, contain=0.28, n_prompts=64, max_new=8)
VIEWS = [(35, 18), (215, 18)]
COLS = ["1. Baseline P3-SAM", "2. Enhanced (before polish)", "3. FINAL (re-prompt + clean boundaries)"]


def label_img(img, text, color=(235, 235, 235)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (18, 19, 22), -1)
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)
    return out


def colored_mesh(mesh, face_ids):
    """Flat-colored per-face mesh (vertices unshared so part colors never blend at boundaries)."""
    labels = np.unique(face_ids[face_ids >= 0])
    pal = (rr.part_palette(len(labels)) * 255).astype(np.uint8)
    fc = np.full((len(mesh.faces), 4), 255, dtype=np.uint8)
    fc[:, :3] = 120
    for i, l in enumerate(labels):
        fc[face_ids == l, :3] = pal[i % len(pal)]
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)].reshape(-1, 3)
    out = trimesh.Trimesh(vertices=tri, faces=np.arange(len(tri)).reshape(-1, 3), process=False)
    out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=np.repeat(fc, 3, axis=0))
    return out


def get_stage_outputs(item, auto_mask, mesh, n_pts, n_prompts, recompute):
    cache_file = os.path.join(CACHE, item["id"] + ".pkl")
    if os.path.exists(cache_file) and not recompute:
        d = pickle.load(open(cache_file, "rb"))
        d["feats"] = torch.from_numpy(d["feats"]).cuda()
        return d
    base, _, tb = fp.run_baseline(auto_mask, mesh, n_pts, n_prompts)
    enh, res, te = fp.run_enhanced(auto_mask, mesh, n_pts, n_prompts)
    md = res.metadata
    d = {
        "base": base, "enh": enh, "t_base": tb, "t_enh": te,
        "feats_np": md["point_feats"].float().cpu().numpy(),
        "pts": np.asarray(md["sampled_points"]), "fi": np.asarray(md["face_idx"]),
    }
    pickle.dump({**{k: v for k, v in d.items()}, "feats": d["feats_np"]}, open(cache_file, "wb"))
    d["feats"] = md["point_feats"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", nargs="*", default=None)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--prompt-scale", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    global OUT, CACHE
    if args.tag:
        OUT = os.path.join(ROOT, "outputs", "final_pipeline_suite_" + args.tag)
        CACHE = os.path.join(OUT, "cache")
    os.makedirs(CACHE, exist_ok=True)
    items = [m for m in TEST_MESHES if not args.meshes or m["id"] in args.meshes]
    auto_mask = fp.mod.AutoMask(ckpt_path=None)
    summary_path = os.path.join(OUT, "final_summary.json")
    records = {}
    if os.path.exists(summary_path):
        records = {r["id"]: r for r in json.load(open(summary_path))}

    for item in items:
        mid = item["id"]
        print("=" * 80, f"\n{mid}", flush=True)
        mesh = trimesh.load(item["path"], force="mesh")
        F = len(mesh.faces)
        n_pts, n_prompts = fp.sampling_params(F, args.prompt_scale)
        graph = pp.MeshGraph(mesh)
        d = get_stage_outputs(item, auto_mask, mesh, n_pts, n_prompts, args.recompute)
        t0 = time.perf_counter()
        sub, rep = fp.subdivide_large_parts(
            auto_mask.model, d["feats"], d["pts"], d["fi"], d["enh"], graph,
            face_normals=np.asarray(mesh.face_normals), **SUB_KW
        )
        final = fp.polish(sub, graph, n_pts)
        t_post = time.perf_counter() - t0

        base = pp.renumber_by_area(d["base"], graph.areas)
        enh = pp.renumber_by_area(d["enh"], graph.areas)
        final = pp.renumber_by_area(final, graph.areas)
        stats = {n: pp.boundary_stats(i, graph) for n, i in (("base", base), ("enh", enh), ("final", final))}
        disc = {n: pp.count_disconnected(i, graph) for n, i in (("base", base), ("enh", enh), ("final", final))}

        # hierarchical L2 on the final parts
        part_ids, part_feats, _ = hc.extract_part_features(
            mesh, final, point_feats=d["feats"], face_idx=d["fi"], points=d["pts"]
        )
        adj_m, _, _ = hc.build_part_adjacency_graph(mesh, final)
        k2 = max(2, min(6, len(part_ids) // 2))
        hier = hc.run_connectivity_constrained_hierarchical_clustering(part_feats, part_ids, adj_m, [k2])
        p2c = hier[k2]["part_to_cluster"]
        l2 = np.array([p2c.get(f, -1) if f >= 0 else -1 for f in final], dtype=np.int64)

        mdir = os.path.join(OUT, mid)
        os.makedirs(mdir, exist_ok=True)
        np.save(os.path.join(mdir, "final_face_ids.npy"), final)
        colored_mesh(mesh, base).export(os.path.join(mdir, "baseline.glb"))
        colored_mesh(mesh, enh).export(os.path.join(mdir, "enhanced.glb"))
        colored_mesh(mesh, final).export(os.path.join(mdir, "final_parts.glb"))
        colored_mesh(mesh, l2).export(os.path.join(mdir, "final_assemblies_L2.glb"))

        rows = []
        for azim, elev in VIEWS:
            row = [rr.render_parts(mesh, ids, size=640, azim=azim, elev=elev, y_up=item["y_up"])
                   for ids in (base, enh, final)]
            rows.append(row)
        for r, row in enumerate(rows):
            for c in range(3):
                n = len(np.unique(((base, enh, final)[c])))
                row[c] = label_img(row[c], f"{COLS[c]} - {n} parts" if r == 0 else f"{COLS[c].split('. ')[1]} (back view)")
        sheet = np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)
        cv2.imwrite(os.path.join(OUT, f"{mid}_comparison.png"), sheet[:, :, ::-1])
        l2img = rr.render_parts(mesh, l2, size=640, azim=35, elev=18, y_up=item["y_up"])
        cv2.imwrite(os.path.join(OUT, f"{mid}_L2.png"), label_img(l2img, f"Functional assemblies L2 - {len(np.unique(l2))} groups")[:, :, ::-1])

        rec = {
            "id": mid, "name": item["name"], "category": item["category"], "faces": F,
            "subdivided_parts": {str(k): v for k, v in rep.items()},
            "parts": {k: stats[k]["parts"] for k in stats},
            "tiny_fragments": {k: stats[k]["tiny_fragments"] for k in stats},
            "disconnected_parts": disc,
            "boundary_len_pct": {k: round(stats[k]["boundary_len_pct"], 3) for k in stats},
            "boundary_mean_crease_deg": {k: round(stats[k]["boundary_mean_crease_deg"], 2) for k in stats},
            "boundary_on_crease_pct": {k: round(stats[k]["boundary_on_crease_pct"], 2) for k in stats},
            "l2_groups": int(len(np.unique(l2))),
            "time_base_s": round(d["t_base"], 2), "time_enh_s": round(d["t_enh"], 2),
            "time_post_s": round(t_post, 2),
        }
        records[mid] = rec
        print(json.dumps(rec, indent=1), flush=True)
        json.dump(list(records.values()), open(summary_path, "w"), indent=2)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
