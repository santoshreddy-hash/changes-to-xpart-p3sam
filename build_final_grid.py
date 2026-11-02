"""Stack the front-view row of every per-mesh comparison sheet into one master image."""

import json
import os

import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "final_pipeline_suite")
records = json.load(open(os.path.join(OUT, "final_summary.json")))
order = ["wooden_sideboard", "vintage_table", "jade_sword", "deer_organic",
         "fall_guy_character", "ceramic_vase", "man_creature", "dining_set"]
by_id = {r["id"]: r for r in records}
rows = []
for mid in order:
    sheet = cv2.imread(os.path.join(OUT, f"{mid}_comparison.png"))
    top = sheet[:640].copy()
    r = by_id[mid]
    p, bl, cr = r["parts"], r["boundary_len_pct"], r["boundary_on_crease_pct"]
    caption = (f"{r['name']}  |  {r['faces']:,} faces  |  parts {p['base']} -> {p['enh']} -> {p['final']}"
               f"  |  boundary length {bl['enh']:.2f}% -> {bl['final']:.2f}%  |  on creases {cr['enh']:.0f}% -> {cr['final']:.0f}%")
    cv2.rectangle(top, (0, 604), (top.shape[1], 640), (18, 19, 22), -1)
    cv2.putText(top, caption, (12, 628), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 220, 255), 1, cv2.LINE_AA)
    rows.append(top)
grid = np.concatenate(rows, axis=0)
tp = sum(r["parts"]["base"] for r in records), sum(r["parts"]["enh"] for r in records), sum(r["parts"]["final"] for r in records)
header = np.full((70, grid.shape[1], 3), (18, 19, 22), dtype=np.uint8)
cv2.putText(header, "P3-SAM optimized pipeline: baseline vs enhanced vs FINAL (re-prompt + clean boundaries)",
            (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(header, f"Total parts over 8 meshes: baseline {tp[0]} -> enhanced {tp[1]} -> final {tp[2]}   (every part is one connected piece; tiny specks merged)",
            (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
grid = np.concatenate([header, grid], axis=0)
path = os.path.join(OUT, "FINAL_master_grid.png")
cv2.imwrite(path, grid)
print(path, grid.shape)
