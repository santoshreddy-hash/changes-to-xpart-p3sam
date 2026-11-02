"""
Comprehensive Benchmark Script for Graph Cut Refinement vs. SciPy Cleanup in P3-SAM.

Evaluates on the 854,832-face wooden rack mesh:
1. Baseline (SciPy connected components + flood fill majority vote)
2. Graph Cut with topological mesh adjacency and dihedral/length weights
3. Graph Cut with spatial kNN adjacency over face centroids

Measures:
- Hungarian optimal matching mIoU
- Boundary Precision, Recall, and F1 Score (exact and 1-ring relaxed)
- Boundary Edge Ratio (BER)
- Part count accuracy (K_pred vs K_gt)
- Component separation / accidental merge detection
- Runtime (inference, refinement, total)
- Peak GPU VRAM and CPU RAM
"""

import os
import sys
import time
import json
import psutil
import torch
import trimesh
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

# Add P3-SAM/demo to sys.path
p3sam_demo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "P3-SAM", "demo"))
if p3sam_demo_dir not in sys.path:
    sys.path.insert(0, p3sam_demo_dir)

import auto_mask_fast as mod
from render_utils import render_mesh_to_image


def compute_hungarian_miou(pred_face_ids, gt_face_ids):
    """
    Computes Hungarian optimal bipartite matching mean IoU.
    """
    u_pred = np.unique(pred_face_ids[pred_face_ids >= 0])
    u_gt = np.unique(gt_face_ids[gt_face_ids >= 0])
    
    if len(u_pred) == 0 or len(u_gt) == 0:
        return 0.0
    
    n_p = len(u_pred)
    n_g = len(u_gt)
    iou_matrix = np.zeros((n_g, n_p), dtype=np.float64)
    
    for i, g in enumerate(u_gt):
        g_mask = (gt_face_ids == g)
        g_sum = np.sum(g_mask)
        for j, p in enumerate(u_pred):
            p_mask = (pred_face_ids == p)
            inter = np.sum(g_mask & p_mask)
            union = g_sum + np.sum(p_mask) - inter
            iou_matrix[i, j] = inter / union if union > 0 else 0.0
            
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]
    miou = np.sum(matched_ious) / max(n_g, n_p)
    return float(miou)


def compute_boundary_fscore(mesh, pred_face_ids, gt_face_ids):
    """
    Computes Boundary Precision, Recall, F1 Score, and Boundary Edge Ratio (BER).
    
    Evaluates both:
    1. Exact edge match
    2. 1-ring relaxed match (within 1-hop face neighborhood on mesh graph)
    """
    face_adj = mesh.face_adjacency
    f1 = face_adj[:, 0]
    f2 = face_adj[:, 1]
    
    pred_boundary = (pred_face_ids[f1] != pred_face_ids[f2])
    gt_boundary = (gt_face_ids[f1] != gt_face_ids[f2])
    
    n_pred_b = int(np.sum(pred_boundary))
    n_gt_b = int(np.sum(gt_boundary))
    total_edges = len(face_adj)
    
    ber_pred = n_pred_b / total_edges if total_edges > 0 else 0.0
    ber_gt = n_gt_b / total_edges if total_edges > 0 else 0.0
    
    # Exact match
    exact_tp = int(np.sum(pred_boundary & gt_boundary))
    prec_exact = exact_tp / n_pred_b if n_pred_b > 0 else 0.0
    rec_exact = exact_tp / n_gt_b if n_gt_b > 0 else 0.0
    f1_exact = (2.0 * prec_exact * rec_exact / (prec_exact + rec_exact)) if (prec_exact + rec_exact) > 0 else 0.0
    
    # 1-ring relaxed match:
    # Build incident faces map for boundary edges
    n_faces = len(mesh.faces)
    gt_boundary_faces = np.zeros(n_faces, dtype=bool)
    gt_boundary_faces[f1[gt_boundary]] = True
    gt_boundary_faces[f2[gt_boundary]] = True
    
    pred_boundary_faces = np.zeros(n_faces, dtype=bool)
    pred_boundary_faces[f1[pred_boundary]] = True
    pred_boundary_faces[f2[pred_boundary]] = True
    
    # A pred boundary edge is TP if either of its faces touches a GT boundary face
    tp_pred = pred_boundary & (gt_boundary_faces[f1] | gt_boundary_faces[f2])
    prec_relaxed = float(np.sum(tp_pred)) / n_pred_b if n_pred_b > 0 else 0.0
    
    # A GT boundary edge is recalled if either of its faces touches a pred boundary face
    tp_gt = gt_boundary & (pred_boundary_faces[f1] | pred_boundary_faces[f2])
    rec_relaxed = float(np.sum(tp_gt)) / n_gt_b if n_gt_b > 0 else 0.0
    
    f1_relaxed = (2.0 * prec_relaxed * rec_relaxed / (prec_relaxed + rec_relaxed)) if (prec_relaxed + rec_relaxed) > 0 else 0.0
    
    return {
        "n_pred_boundary_edges": n_pred_b,
        "n_gt_boundary_edges": n_gt_b,
        "ber_pred": float(ber_pred),
        "ber_gt": float(ber_gt),
        "precision_exact": float(prec_exact),
        "recall_exact": float(rec_exact),
        "f1_exact": float(f1_exact),
        "precision_relaxed": float(prec_relaxed),
        "recall_relaxed": float(rec_relaxed),
        "f1_relaxed": float(f1_relaxed),
    }


def check_component_separation(mesh, face_ids):
    """
    Verifies whether physically disconnected mesh components are accidentally merged.
    
    Splits mesh into connected components and checks how parts are distributed across them.
    Returns:
        num_components: number of disconnected geometric components in mesh
        cross_component_parts: list of part IDs that bridge across distinct components
        is_clean: bool, True if distinct components have independent or non-contaminating parts
    """
    components = mesh.split(only_watertight=False)
    n_comp = len(components)
    
    # Fast check using csgraph connected components on face adjacency
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    adj = mesh.face_adjacency
    n_faces = len(mesh.faces)
    data = np.ones(len(adj), dtype=np.int8)
    graph = csr_matrix((data, (adj[:, 0], adj[:, 1])), shape=(n_faces, n_faces))
    n_geom_comp, geom_labels = connected_components(graph, directed=False)
    
    unique_parts = np.unique(face_ids[face_ids >= 0])
    bridged_parts = []
    
    for p in unique_parts:
        part_mask = (face_ids == p)
        comp_hits = np.unique(geom_labels[part_mask])
        if len(comp_hits) > 1:
            # Check significant presence in multiple components (not just 1 stray face)
            counts = [np.sum(part_mask & (geom_labels == c)) for c in comp_hits]
            major_hits = [c for c, cnt in zip(comp_hits, counts) if cnt > 50]
            if len(major_hits) > 1:
                bridged_parts.append(int(p))
                
    return {
        "n_geometric_components": int(n_geom_comp),
        "bridged_parts_count": len(bridged_parts),
        "bridged_part_ids": bridged_parts,
        "is_clean_separation": (len(bridged_parts) == 0),
    }


def save_colorized_mesh(mesh, face_ids, save_path):
    """
    Saves a reproducible colorized mesh with distinct part colors.
    """
    unique_ids = np.unique(face_ids[face_ids >= 0])
    rng = np.random.RandomState(42)
    import colorsys
    
    color_map = {}
    for uid in sorted(unique_ids):
        hue = rng.uniform(0, 1)
        sat = rng.uniform(0.70, 0.95)
        val = rng.uniform(0.75, 0.95)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        color_map[uid] = np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
        
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    for i, fid in enumerate(face_ids):
        if fid in color_map:
            face_colors[i] = color_map[fid]
        else:
            face_colors[i] = [120, 120, 120, 255]
            
    m_copy = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    m_copy.visual.face_colors = face_colors
    m_copy.export(save_path)
    return m_copy


def run_benchmark():
    mesh_path = "uploads/furn16_woodenrack_raw_854k_d95a8fe0.glb"
    ref_gt_path = "outputs/web_furn16_woodenrack_raw_854k_d95a8fe0_66650142/p3sam/furn16_woodenrack_raw_854k_d95a8fe0_face_ids.npy"
    out_dir = "outputs/benchmark_graphcut"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("3D MESH PART SEGMENTATION BENCHMARK: GRAPH CUT BOUNDARY REFINEMENT")
    print("=" * 80)
    print(f"Target Mesh: {mesh_path}")
    print(f"Reference GT Labels: {ref_gt_path}")
    print(f"Output Directory: {out_dir}\n")
    
    # Load input mesh
    raw_mesh = trimesh.load(mesh_path, force="mesh")
    print(f"Loaded Raw Input Mesh: {len(raw_mesh.vertices):,} vertices, {len(raw_mesh.faces):,} faces")
    
    gt_face_ids = np.load(ref_gt_path)
    gt_parts_count = len(np.unique(gt_face_ids[gt_face_ids >= 0]))
    print(f"Loaded Reference GT: {len(gt_face_ids):,} faces, {gt_parts_count} parts\n")
    
    # Initialize P3-SAM model once (frozen weights)
    print("Loading frozen P3-SAM checkpoint onto CUDA...")
    am = mod.AutoMask(ckpt_path=None)
    print("P3-SAM model ready.\n")
    
    configs = [
        {
            "id": "baseline_scipy",
            "name": "Existing Cleanup (SciPy Baseline)",
            "refine_mode": "scipy",
            "gc_smoothness": 1.0,
            "gc_sigma": 0.35,
            "gc_knn_k": 6,
        },
        {
            "id": "graphcut_mesh",
            "name": "Graph Cut (Mesh Topological Adjacency)",
            "refine_mode": "graphcut",
            "gc_smoothness": 1.0,
            "gc_sigma": 0.35,
            "gc_knn_k": 6,
        },
        {
            "id": "graphcut_knn",
            "name": "Graph Cut (Spatial kNN Adjacency)",
            "refine_mode": "graphcut_knn",
            "gc_smoothness": 1.0,
            "gc_sigma": 0.35,
            "gc_knn_k": 6,
        },
    ]
    
    results = []
    rendered_images = {}
    
    proc = psutil.Process()
    
    for idx, cfg in enumerate(configs, 1):
        print("-" * 80)
        print(f"[{idx}/3] RUNNING: {cfg['name']}")
        print(f"Mode: {cfg['refine_mode']} | Smoothness: {cfg['gc_smoothness']} | Sigma: {cfg['gc_sigma']}")
        print("-" * 80)
        
        # Reset memory tracking
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rss_before = proc.memory_info().rss
        
        t0 = time.time()
        mod.set_seed(42)
        
        run_out_dir = os.path.join(out_dir, cfg["id"])
        os.makedirs(run_out_dir, exist_ok=True)
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            aabb, face_ids, cleaned_mesh = am.predict_aabb(
                raw_mesh.copy(),
                save_path=run_out_dir,
                point_num=100000,
                prompt_num=400,
                threshold=0.95,
                post_process=False,
                save_mid_res=False,
                show_info=True,
                seed=42,
                is_parallel=True,
                clean_mesh_flag=True,
                prompt_bs=8,
                loop_mode="opt3",
                nms_mode="fast",
                cleanup_mode="fast",
                multimask_output=False,
                refine_mode=cfg["refine_mode"],
                gc_smoothness=cfg["gc_smoothness"],
                gc_sigma=cfg["gc_sigma"],
                gc_knn_k=cfg["gc_knn_k"],
                gc_n_iter=5,
                preserve_raw_faces=True,
            )
            
        t1 = time.time()
        total_time = t1 - t0
        
        # Memory stats
        peak_vram_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_vram_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
        rss_after = proc.memory_info().rss
        peak_cpu_ram = rss_after / (1024 ** 2)
        
        # Evaluation Metrics
        pred_parts_count = len(np.unique(face_ids[face_ids >= 0]))
        part_count_error = abs(pred_parts_count - gt_parts_count)
        
        miou = compute_hungarian_miou(face_ids, gt_face_ids)
        boundary_metrics = compute_boundary_fscore(cleaned_mesh, face_ids, gt_face_ids)
        component_metrics = check_component_separation(cleaned_mesh, face_ids)
        
        # Check original face IDs preservation
        raw_face_ids = cleaned_mesh.metadata.get("raw_face_ids")
        has_raw_preserved = (raw_face_ids is not None and len(raw_face_ids) == len(raw_mesh.faces))
        
        # Save colorized GLB
        glb_save_path = os.path.join(out_dir, f"woodenrack_{cfg['id']}.glb")
        colorized_mesh = save_colorized_mesh(cleaned_mesh, face_ids, glb_save_path)
        
        # Render high-quality visualization
        img_front = render_mesh_to_image(colorized_mesh, width=800, height=800, elev=20, azim=45)
        img_top = render_mesh_to_image(colorized_mesh, width=800, height=800, elev=60, azim=15)
        
        img_front_path = os.path.join(out_dir, f"render_{cfg['id']}_front.png")
        img_front.save(img_front_path)
        rendered_images[cfg['id']] = img_front
        
        res = {
            "id": cfg["id"],
            "name": cfg["name"],
            "refine_mode": cfg["refine_mode"],
            "gc_smoothness": cfg["gc_smoothness"],
            "gc_sigma": cfg["gc_sigma"],
            "gc_knn_k": cfg["gc_knn_k"],
            "runtime_s": round(total_time, 3),
            "peak_vram_alloc_mb": round(peak_vram_alloc, 1),
            "peak_vram_reserved_mb": round(peak_vram_res, 1),
            "peak_cpu_ram_mb": round(peak_cpu_ram, 1),
            "pred_parts_count": pred_parts_count,
            "gt_parts_count": gt_parts_count,
            "part_count_diff": part_count_error,
            "miou": round(miou, 5),
            "precision_exact": round(boundary_metrics["precision_exact"], 5),
            "recall_exact": round(boundary_metrics["recall_exact"], 5),
            "boundary_f1_exact": round(boundary_metrics["f1_exact"], 5),
            "precision_relaxed": round(boundary_metrics["precision_relaxed"], 5),
            "recall_relaxed": round(boundary_metrics["recall_relaxed"], 5),
            "boundary_f1_relaxed": round(boundary_metrics["f1_relaxed"], 5),
            "ber_pred": round(boundary_metrics["ber_pred"], 5),
            "ber_gt": round(boundary_metrics["ber_gt"], 5),
            "n_boundary_edges": boundary_metrics["n_pred_boundary_edges"],
            "bridged_parts_count": component_metrics["bridged_parts_count"],
            "is_clean_separation": component_metrics["is_clean_separation"],
            "raw_faces_preserved": has_raw_preserved,
            "raw_faces_count": len(raw_face_ids) if has_raw_preserved else 0,
        }
        results.append(res)
        
        print(f"\n[Result {cfg['id']}]:")
        print(f"  mIoU: {res['miou'] * 100:.2f}% | Exact Boundary F1: {res['boundary_f1_exact'] * 100:.2f}% | Relaxed F1: {res['boundary_f1_relaxed'] * 100:.2f}%")
        print(f"  Parts: {res['pred_parts_count']} (GT: {gt_parts_count}) | Bridged Parts: {res['bridged_parts_count']}")
        print(f"  Runtime: {res['runtime_s']}s | Peak VRAM: {res['peak_vram_alloc_mb']} MB | Peak RAM: {res['peak_cpu_ram_mb']} MB\n")
        
    # Save JSON summary
    json_path = os.path.join(out_dir, "benchmark_graphcut_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark results JSON to {json_path}")
    
    # Generate Side-by-Side Comparison Canvas
    print("\nGenerating Side-by-Side Comparison Figure...")
    w, h = 800, 800
    canvas_w = w * 3 + 40
    canvas_h = h + 120
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    
    titles = [
        "Baseline: SciPy Cleanup\n(mIoU: {:.2f}%, F1: {:.2f}%)".format(results[0]["miou"] * 100, results[0]["boundary_f1_relaxed"] * 100),
        "Graph Cut: Mesh Adjacency\n(mIoU: {:.2f}%, F1: {:.2f}%)".format(results[1]["miou"] * 100, results[1]["boundary_f1_relaxed"] * 100),
        "Graph Cut: Spatial kNN\n(mIoU: {:.2f}%, F1: {:.2f}%)".format(results[2]["miou"] * 100, results[2]["boundary_f1_relaxed"] * 100),
    ]
    
    for i, cfg in enumerate(configs):
        img = rendered_images[cfg["id"]]
        x_offset = 10 + i * (w + 10)
        canvas.paste(img, (x_offset, 100))
        
        # Draw header text
        lines = titles[i].split("\n")
        draw.text((x_offset + w // 2 - 140, 25), lines[0], fill=(240, 240, 240))
        if len(lines) > 1:
            draw.text((x_offset + w // 2 - 110, 55), lines[1], fill=(120, 220, 160))
            
    fig_path = os.path.join(out_dir, "p3sam_graphcut_before_after.png")
    canvas.save(fig_path)
    print(f"Saved side-by-side comparison figure to {fig_path}")
    
    # Print Markdown Summary Table
    print("\n" + "=" * 90)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 90)
    header = "| Configuration | mIoU | Boundary F1 (Exact) | Boundary F1 (Relaxed) | BER | Parts (K) | Runtime | Peak VRAM | Accidental Merges |"
    sep    = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    print(header)
    print(sep)
    for r in results:
        merge_status = "None (Clean)" if r["is_clean_separation"] else f"{r['bridged_parts_count']} bridged"
        row = f"| **{r['name']}** | {r['miou']*100:.2f}% | {r['boundary_f1_exact']*100:.2f}% | {r['boundary_f1_relaxed']*100:.2f}% | {r['ber_pred']*100:.3f}% | {r['pred_parts_count']} | {r['runtime_s']:.1f}s | {r['peak_vram_alloc_mb']:.0f} MB | {merge_status} |"
        print(row)
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()
