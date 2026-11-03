import trimesh
import numpy as np
from partseg.adaptive_fast import run_clean_parts_fast_enhanced
from partseg.polish import MeshGraph
from partseg.export import build_parts_table

mesh = trimesh.load('assets/furn16_3k.glb', force='mesh')
face_ids, quality = run_clean_parts_fast_enhanced(mesh)
graph = MeshGraph(mesh)
parts = build_parts_table(mesh, face_ids, graph)

print(f"Total parts found: {len(parts)}")
for p in sorted(parts, key=lambda x: x['id']):
    c = [round(float(x), 2) for x in p['centroid']]
    ymin, ymax = round(p['bbox_min'][1], 2), round(p['bbox_max'][1], 2)
    xmin, xmax = round(p['bbox_min'][0], 2), round(p['bbox_max'][0], 2)
    zmin, zmax = round(p['bbox_min'][2], 2), round(p['bbox_max'][2], 2)
    print(f"Part {p['id']:02d} | faces: {p['faces']:4d} | Y: [{ymin:+0.2f}, {ymax:+0.2f}] | X: [{xmin:+0.2f}, {xmax:+0.2f}] | Z: [{zmin:+0.2f}, {zmax:+0.2f}] | centroid: {c}")

