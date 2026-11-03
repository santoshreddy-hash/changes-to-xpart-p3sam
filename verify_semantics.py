import trimesh
import numpy as np
from partseg.adaptive_fast import run_clean_parts_fast_enhanced
from partseg.polish import MeshGraph
from partseg.export import build_parts_table

mesh = trimesh.load('assets/furn16_3k.glb', force='mesh')
face_ids, quality = run_clean_parts_fast_enhanced(mesh)
graph = MeshGraph(mesh)
parts = build_parts_table(mesh, face_ids, graph)

print("Detailed part inspection:")
for p in parts:
    f_idx = np.where(face_ids == p['id'])[0]
    normals = mesh.face_normals[f_idx]
    mean_n = np.mean(normals, axis=0)
    mean_n /= (np.linalg.norm(mean_n) + 1e-6)
    area = p['area']
    c = np.round(p['centroid'], 2)
    bmin = np.round(p['bbox_min'], 2)
    bmax = np.round(p['bbox_max'], 2)
    print(f"ID {p['id']:02d}: faces={len(f_idx):4d}, area={area:.4f}, mean_norm={np.round(mean_n, 2)}, C={c}, X=[{bmin[0]}, {bmax[0]}], Y=[{bmin[1]}, {bmax[1]}], Z=[{bmin[2]}, {bmax[2]}]")

