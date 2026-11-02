"""Benchmark: baseline P3-SAM vs enhanced P3-SAM vs the full partseg pipeline on the 8-mesh suite.

    python benchmarks/run_suite.py [--meshes id ...] [--preset precise|detailed|fast] [--out DIR]

Writes per-mesh comparison renders, GLBs and final_summary.json (read by build_grid.py).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from partseg import SegmentConfig, load_model, segment_mesh  # noqa: E402
from partseg.export import segmented_mesh  # noqa: E402
from partseg.io import load_mesh  # noqa: E402
from partseg.p3sam import p3sam_module, sampling_params  # noqa: E402
from partseg.polish import boundary_stats, count_disconnected, renumber_by_area  # noqa: E402
from partseg.render import render_parts  # noqa: E402
from multimesh_suite_base import TEST_MESHES  # noqa: E402

COLS = ["1. Baseline P3-SAM", "2. Enhanced P3-SAM (before clean-up)", "3. FINAL partseg"]
VIEWS = [(35, 18), (215, 18)]


def run_baseline(model, mesh, n_pts, n_prompts, seed=42):
    """Plain P3-SAM: single mask, greedy NMS, SciPy clean-up (the reference being improved on)."""
    mod = p3sam_module()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, ids, _ = mod.mesh_sam(
            [model.model, model.model], mesh.copy(), save_path=None, point_num=n_pts, prompt_num=n_prompts,
            threshold=0.95, post_process=False, show_info=False, save_mid_res=False, clean_mesh_flag=False,
            seed=seed, prompt_bs=8, loop_mode="opt3", nms_mode="fast", cleanup_mode="fast",
            multimask_output=False, stability_filter=False, nms_thresh=0.9, refine_mode="scipy",
            preserve_raw_faces=True, second_pass_enabled=False,
        )
    return np.asarray(ids).astype(np.int64), time.perf_counter() - t0


def label_img(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (18, 19, 22), -1)
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", nargs="*")
    ap.add_argument("--preset", default="precise", choices=["precise", "detailed", "fast"])
    ap.add_argument("--out", default=os.path.join("outputs", "final_pipeline_suite"))
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = ap.parse_args()
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    cfg = SegmentConfig.preset(args.preset).override(args.overrides)
    model = load_model()
    summary_path = os.path.join(out_root, "final_summary.json")
    records = {r["id"]: r for r in json.load(open(summary_path))} if os.path.exists(summary_path) else {}

    for item in TEST_MESHES:
        if args.meshes and item["id"] not in args.meshes:
            continue
        mid = item["id"]
        print("=" * 70, f"\n{mid}", flush=True)
        mesh, info = load_mesh(item["path"])
        n_pts, n_prompts = sampling_params(len(mesh.faces), cfg)
        base, t_base = run_baseline(model, mesh, n_pts, n_prompts)
        result = segment_mesh(mesh, cfg, model=model, info=info)
        graph = result.graph
        enh, final = result.stage_labels["p3sam"], result.face_ids
        base = renumber_by_area(base, graph.areas)
        enh = renumber_by_area(enh, graph.areas)
        sets = {"base": base, "enh": enh, "final": final}
        stats = {k: boundary_stats(v, graph) for k, v in sets.items()}

        mdir = os.path.join(out_root, mid)
        os.makedirs(mdir, exist_ok=True)
        np.save(os.path.join(mdir, "final_face_ids.npy"), final)
        for name, ids in (("baseline", base), ("enhanced", enh), ("final_parts", final)):
            segmented_mesh(mesh, ids).export(os.path.join(mdir, f"{name}.glb"))
        rows = []
        for azim, elev in VIEWS:
            row = [render_parts(mesh, ids, size=640, azim=azim, elev=elev, y_up=item["y_up"]) for ids in sets.values()]
            for c, ids in enumerate(sets.values()):
                label = f"{COLS[c]} - {len(np.unique(ids))} parts" if azim == VIEWS[0][0] else COLS[c].split(". ")[1] + " (back view)"
                row[c] = label_img(row[c], label)
            rows.append(np.concatenate(row, axis=1))
        cv2.imwrite(os.path.join(out_root, f"{mid}_comparison.png"), np.concatenate(rows, axis=0)[:, :, ::-1])

        t = result.stage_seconds
        records[mid] = {
            "id": mid, "name": item["name"], "category": item["category"], "faces": len(mesh.faces),
            "parts": {k: stats[k]["parts"] for k in stats},
            "tiny_fragments": {k: stats[k]["tiny_fragments"] for k in stats},
            "disconnected_parts": {k: int(count_disconnected(v, graph)) for k, v in sets.items()},
            "boundary_len_pct": {k: round(stats[k]["boundary_len_pct"], 3) for k in stats},
            "boundary_mean_crease_deg": {k: round(stats[k]["boundary_mean_crease_deg"], 2) for k in stats},
            "boundary_on_crease_pct": {k: round(stats[k]["boundary_on_crease_pct"], 2) for k in stats},
            "parts_after_stage": result.report["parts_after_stage"],
            "time_base_s": round(t_base, 2), "time_enh_s": round(t["p3sam"], 2),
            "time_post_s": round(t["total"] - t["p3sam"] - t.get("model_load", 0.0), 2),
            "time_total_s": round(t["total"] - t.get("model_load", 0.0), 2),
            "warnings": result.warnings,
        }
        print(json.dumps(records[mid]["parts"]), "time", records[mid]["time_total_s"], "s", flush=True)
        json.dump(list(records.values()), open(summary_path, "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
