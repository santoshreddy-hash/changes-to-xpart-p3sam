"""Run partseg on the 8 benchmark meshes and write the full structured output of each into its own sub-folder.

    python benchmarks/export_results.py [--out outputs/partseg_results] [--preset precise|detailed|fast]

Result layout:
    <out>/summary.json
    <out>/<mesh id>/<name>_parts.glb          one named node per part
    <out>/<mesh id>/<name>_segmented.glb      single mesh, flat colour per part
    <out>/<mesh id>/<name>_face_ids.npy       part id per face
    <out>/<mesh id>/<name>_parts.json         per-part table
    <out>/<mesh id>/<name>_report.json        config, timings, metrics, warnings
    <out>/<mesh id>/<name>_preview.png        front and back view
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from partseg import SegmentConfig, load_model, segment_file  # noqa: E402
from multimesh_suite_base import TEST_MESHES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("outputs", "partseg_results"))
    ap.add_argument("--preset", default="precise", choices=["precise", "detailed", "fast"])
    ap.add_argument("--meshes", nargs="*")
    ap.add_argument("--seal", action="store_true", help="also write the sealed parts GLB")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    model = load_model()
    summary = []
    for item in TEST_MESHES:
        if args.meshes and item["id"] not in args.meshes:
            continue
        cfg = SegmentConfig.preset(args.preset)
        cfg.export.up_axis = "y" if item["y_up"] else "z"  # only affects the preview render
        cfg.export.seal = args.seal
        t0 = time.perf_counter()
        result = segment_file(item["path"], os.path.join(out_root, item["id"]), cfg, model=model)
        summary.append({
            "id": item["id"], "name": item["name"], "category": item["category"],
            "faces": len(result.mesh.faces), "parts": len(result.parts),
            "seconds": round(result.stage_seconds["total"] - result.stage_seconds.get("model_load", 0.0), 2),
            "disconnected_parts": result.metrics["disconnected_parts"],
            "warnings": result.warnings,
            "folder": item["id"],
        })
        print(f"{item['id'].ljust(20)} {len(result.parts):>3} parts  {summary[-1]['seconds']:>6.1f}s", flush=True)
    json.dump({"preset": args.preset, "meshes": summary}, open(os.path.join(out_root, "summary.json"), "w"), indent=2)
    print("wrote", out_root)


if __name__ == "__main__":
    main()
