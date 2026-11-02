import os
import sys
import time
import json
import psutil
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add P3-SAM/demo to sys.path
p3sam_demo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "P3-SAM", "demo"))
if p3sam_demo_dir not in sys.path:
    sys.path.insert(0, p3sam_demo_dir)

import auto_mask_fast as mod
import hierarchical_clustering as hc

OUT_DIR = "outputs/multimesh_benchmark_suite"
os.makedirs(OUT_DIR, exist_ok=True)

# 8 diverse test meshes spanning different domains
TEST_MESHES = [
    {
        "id": "wooden_sideboard",
        "name": "Wooden Sideboard",
        "category": "Furniture / Storage",
        "path": "uploads/furn16_woodenrack_raw_854k_d95a8fe0.glb",
        "y_up": True,
        "precomputed": True, # Already computed today!
    },
    {
        "id": "vintage_table",
        "name": "Vintage Bedside Table",
        "category": "Furniture / Table",
        "path": "uploads/bedside_wood_table_vintage_furniture_7aa2f793.glb",
        "y_up": True,
        "precomputed": False,
    },
    {
        "id": "jade_sword",
        "name": "Jade Sword & Scabbard",
        "category": "Hard-Surface Prop / Weapon",
        "path": "uploads/jade_sword_650f414d.glb",
        "y_up": False,
        "precomputed": False,
    },
    {
        "id": "deer_organic",
        "name": "Deer with Antlers",
        "category": "Organic Animal",
        "path": "uploads/deer_100k_1ae4cfca.glb",
        "y_up": True,
        "precomputed": False,
    },
    {
        "id": "fall_guy_character",
        "name": "Stylized Dino Character",
        "category": "Character / Creature",
        "path": "uploads/fall_guys_trice_skin_07d92877.glb",
        "y_up": True,
        "precomputed": False,
    },
    {
        "id": "ceramic_vase",
        "name": "Ceramic Vase with Handles",
        "category": "Decorative Prop / Vessel",
        "path": "uploads/vase_8437fc2d.glb",
        "y_up": True,
        "precomputed": False,
    },
    {
        "id": "man_creature",
        "name": "Demon Monster Creature",
        "category": "Organic Character / Monster",
        "path": "All_Input_GLBs (1)/All_Input_GLBs/man_creature_input.glb",
        "y_up": True,
        "precomputed": False,
    },
    {
        "id": "dining_set",
        "name": "Dining Table & 4 Chairs",
        "category": "Multi-Object Assembly",
        "path": "uploads/vihals-vihals-table-and-4-chairs-white-white-tibbleby-beige-grey__1388772_pe964945_s5_10k_2k_59ec9ff8.glb",
        "y_up": True,
        "precomputed": False,
    }
]

def render_mesh_universal(mesh, width=600, height=600, elev=18, azim=35, y_up=True, point_density=400000):
    """
    Renders 3D mesh with consistent lighting and viewport padding.
    """
    points, face_indices = trimesh.sample.sample_surface(mesh, point_density)
    normals = mesh.face_normals[face_indices]
    
    if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None and len(mesh.visual.face_colors) > 0:
        raw_colors = mesh.visual.face_colors[face_indices, :3]
    else:
        raw_colors = np.ones((len(points), 3), dtype=np.uint8) * 200
        
    center = points.mean(axis=0)
    pts = points - center
    scale = np.max(np.abs(pts))
    pts /= (scale + 1e-8)
    
    rad_elev = np.radians(elev)
    rad_azim = np.radians(azim)
    
    if y_up:
        cos_a, sin_a = np.cos(rad_azim), np.sin(rad_azim)
        R_rot = np.array([
            [ cos_a, 0, sin_a],
            [     0, 1,     0],
            [-sin_a, 0, cos_a]
        ], dtype=np.float32)
        
        cos_e, sin_e = np.cos(rad_elev), np.sin(rad_elev)
        R_pitch = np.array([
            [1,     0,      0],
            [0, cos_e, -sin_e],
            [0, sin_e,  cos_e]
        ], dtype=np.float32)
        R = R_pitch @ R_rot
    else:
        cos_a, sin_a = np.cos(rad_azim), np.sin(rad_azim)
        R_z = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=np.float32)
        cos_e, sin_e = np.cos(rad_elev), np.sin(rad_elev)
        R_x = np.array([
            [1, 0,      0],
            [0, cos_e, -sin_e],
            [0, sin_e,  cos_e]
        ], dtype=np.float32)
        R = R_x @ R_z
        
    pts_cam = pts @ R.T
    normals_cam = normals @ R.T
    
    light_dir = np.array([0.35, 0.55, 0.75], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    diffuse = np.maximum(0.0, np.sum(normals_cam * light_dir, axis=-1))
    ambient = 0.42
    intensity = np.clip(ambient + 0.58 * diffuse, 0.0, 1.0)[:, None]
    
    colors_lit = np.clip(raw_colors.astype(np.float32) * intensity, 0, 255).astype(np.uint8)
    
    x_cam = pts_cam[:, 0]
    y_cam = pts_cam[:, 1]
    z_cam = pts_cam[:, 2]
    
    pad = 0.12
    u = ((x_cam + (1.0 + pad)) / (2.0 * (1.0 + pad)) * (width - 1)).astype(np.int32)
    v = (((1.0 + pad) - y_cam) / (2.0 * (1.0 + pad)) * (height - 1)).astype(np.int32)
    
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid]
    v = v[valid]
    z_cam = z_cam[valid]
    colors_lit = colors_lit[valid]
    
    sort_order = np.argsort(z_cam)
    u = u[sort_order]
    v = v[sort_order]
    colors_lit = colors_lit[sort_order]
    
    img = np.full((height, width, 3), 20, dtype=np.uint8)
    for du in [0, 1, -1]:
        for dv in [0, 1, -1]:
            uu = np.clip(u + du, 0, width - 1)
            vv = np.clip(v + dv, 0, height - 1)
            img[vv, uu] = colors_lit
            
    return Image.fromarray(img)

def assign_distinct_colors(mesh, face_ids, seed=42):
    """
    Colorizes mesh faces with high-contrast, visually pleasing HSL colors.
    """
    m = mesh.copy()
    u_parts = np.unique(face_ids[face_ids >= 0])
    np.random.seed(seed)
    
    # Generate golden-ratio distributed distinct hues
    colors = {}
    for i, p in enumerate(u_parts):
        hue = (i * 0.618033988749895) % 1.0
        # Convert HSV to RGB
        c = plt.cm.tab20(i % 20)[:3] if len(u_parts) <= 20 else plt.cm.hsv(hue)[:3]
        colors[p] = (np.array(c) * 255).astype(np.uint8)
        
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    face_colors[:, 3] = 255
    
    for p in u_parts:
        face_colors[face_ids == p, :3] = colors[p]
        
    # Unlabeled faces gray
    face_colors[face_ids < 0, :3] = [120, 120, 120]
    m.visual.face_colors = face_colors
    return m

def compute_ber(mesh, face_ids):
    """Computes boundary edge ratio."""
    face_adj = mesh.face_adjacency
    l1 = face_ids[face_adj[:, 0]]
    l2 = face_ids[face_adj[:, 1]]
    boundary_mask = (l1 != l2) & (l1 >= 0) & (l2 >= 0)
    ber = np.sum(boundary_mask) / len(face_adj) if len(face_adj) > 0 else 0.0
    return float(ber)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_connected_components

def compute_disconnected_components(mesh, face_ids):
    """Counts number of multi-component clusters using memory-efficient sparse graphs."""
    u_parts = np.unique(face_ids[face_ids >= 0])
    face_adj = mesh.face_adjacency
    f1 = face_adj[:, 0]
    f2 = face_adj[:, 1]
    
    disconnected_parts = 0
    for p in u_parts:
        p_faces = np.where(face_ids == p)[0]
        if len(p_faces) <= 1:
            continue
        p_face_set = set(p_faces)
        edges = [(a, b) for a, b in zip(f1, f2) if a in p_face_set and b in p_face_set]
        if len(edges) == 0:
            disconnected_parts += 1
            continue
        local_map = {f: i for i, f in enumerate(p_faces)}
        row = [local_map[a] for a, b in edges] + [local_map[b] for a, b in edges]
        col = [local_map[b] for a, b in edges] + [local_map[a] for a, b in edges]
        data = np.ones(len(row), dtype=bool)
        adj_sparse = csr_matrix((data, (row, col)), shape=(len(p_faces), len(p_faces)))
        n_comp, _ = sparse_connected_components(adj_sparse, directed=False)
        if n_comp > 1:
            disconnected_parts += 1
    return disconnected_parts

