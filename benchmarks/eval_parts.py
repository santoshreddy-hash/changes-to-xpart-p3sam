"""Score segmentations against ground-truth face labels.

    python benchmarks/eval_parts.py [--meshes table chair ...] [--out outputs/eval_synthetic.json]

Metrics (all area-weighted, computed per mesh and averaged):
  ari            adjusted Rand index between prediction and ground truth (1 = identical partition)
  recall_iou     for every ground-truth part, IoU with its best predicted part  (misses and under-splitting hurt)
  precision_iou  for every predicted part, IoU with its best ground-truth part (extra fragments and over-splitting hurt)
  exact          ground-truth parts recovered with IoU >= 0.8
  pred_parts     number of predicted parts, next to the true number
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def overlap_matrix(gt: np.ndarray, pred: np.ndarray, areas: np.ndarray) -> np.ndarray:
    g, p = int(gt.max()) + 1, int(pred.max()) + 1
    m = np.zeros((g, p))
    np.add.at(m, (gt, pred), areas)
    return m


def score(gt: np.ndarray, pred: np.ndarray, areas: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score

    m = overlap_matrix(gt, pred, areas)
    gt_area, pred_area = m.sum(1), m.sum(0)
    union = gt_area[:, None] + pred_area[None, :] - m
    iou = m / np.maximum(union, 1e-12)
    return {
        "ari": float(adjusted_rand_score(gt, pred)),
        "recall_iou": float(iou.max(1)[gt_area > 0].mean()),
        "precision_iou": float(iou.max(0)[pred_area > 0].mean()),  # only labels that actually occur
        "exact": int((iou.max(1) >= 0.8).sum()),
        "gt_parts": int(len(gt_area)),
        "pred_parts": int((pred_area > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", nargs="*")
    ap.add_argument("--out", default=os.path.join("outputs", "eval_synthetic.json"))
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = ap.parse_args()

    from partseg import SegmentConfig, load_model, segment_mesh
    from partseg.io import validate_mesh
    from partseg.p3sam import sampling_params
    from run_suite import run_baseline
    from synthetic import build_dataset

    cfg = SegmentConfig().override(args.overrides)
    model = load_model()
    rows = []
    for item in build_dataset(args.meshes):
        mesh, info = validate_mesh(item["mesh"], name=item["name"])
        n_pts, n_prompts = sampling_params(len(mesh.faces), cfg)
        base, _ = run_baseline(model, mesh, n_pts, n_prompts)
        result = segment_mesh(mesh, cfg, model=model, info=info)
        areas = np.asarray(mesh.area_faces)
        sets = {"baseline": base, "enhanced": result.stage_labels["p3sam"], "final": result.face_ids}
        row = {"mesh": item["name"], "gt_parts": item["gt_parts"], "faces": len(mesh.faces)}
        for name, ids in sets.items():
            row[name] = score(item["labels"], np.asarray(ids), areas)
        rows.append(row)
        print(f"{item['name'].ljust(9)} true {item['gt_parts']:>2} | " + " | ".join(
            f"{k} parts {row[k]['pred_parts']:>2} ARI {row[k]['ari']:.2f} R {row[k]['recall_iou']:.2f} P {row[k]['precision_iou']:.2f} exact {row[k]['exact']}"
            for k in sets), flush=True)

    print("\nMEAN over", len(rows), "meshes")
    for k in ("baseline", "enhanced", "final"):
        print(f"  {k.ljust(9)} ARI {np.mean([r[k]['ari'] for r in rows]):.3f}  recall IoU {np.mean([r[k]['recall_iou'] for r in rows]):.3f}"
              f"  precision IoU {np.mean([r[k]['precision_iou'] for r in rows]):.3f}"
              f"  exact {sum(r[k]['exact'] for r in rows)}/{sum(r['gt_parts'] for r in rows)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
