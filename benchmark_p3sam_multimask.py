import os
import sys
import time
import json
import torch
import trimesh
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path

# Add P3-SAM/demo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "P3-SAM", "demo")))
import auto_mask_fast as mod

def compute_hungarian_miou(pred_face_ids, gt_face_ids):
    """
    Computes Hungarian optimal bipartite matching mean IoU between predicted and ground truth face IDs.
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
    # Penalize unmatched parts by dividing by max(n_g, n_p)
    miou = np.sum(matched_ious) / max(n_g, n_p)
    return float(miou)

def compute_boundary_quality(mesh, face_ids):
    """
    Computes boundary edge ratio on the mesh adjacency graph.
    BER = (Number of boundary edges where adjacent faces have different labels) / (Total adjacency edges)
    """
    face_adj = mesh.face_adjacency
    f1 = face_adj[:, 0]
    f2 = face_adj[:, 1]
    
    l1 = face_ids[f1]
    l2 = face_ids[f2]
    
    boundary_mask = (l1 != l2)
    boundary_edge_count = int(np.sum(boundary_mask))
    total_edges = len(face_adj)
    ber = boundary_edge_count / total_edges if total_edges > 0 else 0.0
    return ber, boundary_edge_count, total_edges

def save_colorized_mesh(mesh, face_ids, save_glb_path):
    """
    Saves a clean colorized GLB mesh with distinct, reproducible colors per part.
    """
    unique_ids = np.unique(face_ids[face_ids >= 0])
    rng = np.random.RandomState(42)
    color_map = {}
    for uid in unique_ids:
        # Generate vibrant pastel / distinct colors in HSV/RGB
        hue = rng.uniform(0, 1)
        sat = rng.uniform(0.65, 0.95)
        val = rng.uniform(0.75, 0.98)
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        color_map[uid] = np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
        
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    for i, fid in enumerate(face_ids):
        if fid in color_map:
            face_colors[i] = color_map[fid]
        else:
            face_colors[i] = [128, 128, 128, 255]
            
    mesh_copy = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    mesh_copy.visual.face_colors = face_colors
    mesh_copy.export(save_glb_path)

def run_benchmark():
    mesh_path = "uploads/furn16_woodenrack_raw_854k_d95a8fe0.glb"
    ref_gt_path = "outputs/web_furn16_woodenrack_raw_854k_d95a8fe0_66650142/p3sam/furn16_woodenrack_raw_854k_d95a8fe0_face_ids.npy"
    out_root = "outputs/benchmark_multimask"
    os.makedirs(out_root, exist_ok=True)
    
    print("=" * 80)
    print("P3-SAM AUTOMATIC SEGMENTATION BENCHMARK: MULTI-MASK PIPELINE UPGRADE")
    print("=" * 80)
    print(f"Target Mesh: {mesh_path}")
    print(f"Reference Baseline Labels: {ref_gt_path}")
    
    mesh = trimesh.load(mesh_path, force="mesh")
    print(f"Loaded Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    gt_face_ids = np.load(ref_gt_path)
    print(f"Loaded Reference GT: {len(gt_face_ids):,} faces, {len(np.unique(gt_face_ids[gt_face_ids >= 0]))} parts")
    
    print("\nLoading frozen P3-SAM checkpoint...")
    am = mod.AutoMask(ckpt_path=None)
    print("P3-SAM loaded successfully onto CUDA.\n")
    
    # 12 Experimental Configurations
    nms_thresholds = [0.3, 0.5, 0.7, 0.9]
    modes = [
        {"name": "Original (1-Mask)", "multimask": False, "stability": False},
        {"name": "Improved (3-Masks)", "multimask": True, "stability": False},
        {"name": "Improved + Stability", "multimask": True, "stability": True},
    ]
    
    results = []
    run_idx = 1
    total_runs = len(modes) * len(nms_thresholds)
    
    for mode_cfg in modes:
        mode_name = mode_cfg["name"]
        multimask = mode_cfg["multimask"]
        stability = mode_cfg["stability"]
        
        for nms_t in nms_thresholds:
            cfg_tag = f"{mode_name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')}_NMS_{nms_t:.1f}"
            cfg_dir = os.path.join(out_root, cfg_tag)
            os.makedirs(cfg_dir, exist_ok=True)
            
            print("-" * 80)
            print(f"[{run_idx}/{total_runs}] Running: {mode_name} | NMS Threshold = {nms_t:.1f}")
            print(f"Configuration Tag: {cfg_tag}")
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                aabb, face_ids, mesh_cleaned = mod.mesh_sam(
                    (am.model, am.model),
                    mesh.copy(),
                    save_path=cfg_dir,
                    point_num=100000,
                    prompt_num=400,
                    save_mid_res=False,
                    show_info=True,
                    clean_mesh_flag=True,
                    seed=42,
                    prompt_bs=8,
                    loop_mode="opt3",
                    nms_mode="fast",
                    cleanup_mode="fast",
                    multimask_output=multimask,
                    nms_thresh=nms_t,
                    min_mask_iou=0.0,
                    min_mask_points=10,
                    stability_filter=stability,
                    stability_thresh=0.85,
                    stability_offset=0.05,
                )
                
            torch.cuda.synchronize()
            total_time = time.perf_counter() - t0
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            unique_parts = np.unique(face_ids[face_ids >= 0])
            num_parts = len(unique_parts)
            
            # Compute Hungarian mIoU against reference GT
            miou = compute_hungarian_miou(face_ids, gt_face_ids)
            
            # Compute Boundary Quality
            ber, boundary_edges, total_edges = compute_boundary_quality(mesh_cleaned, face_ids)
            
            # Save results
            np.save(os.path.join(cfg_dir, "face_ids.npy"), face_ids)
            np.save(os.path.join(cfg_dir, "aabb.npy"), aabb)
            save_colorized_mesh(mesh_cleaned, face_ids, os.path.join(cfg_dir, "segmented_mesh.glb"))
            
            cand_count = 1200 if multimask else 400
            
            record = {
                "run_idx": run_idx,
                "config_tag": cfg_tag,
                "mode": mode_name,
                "nms_thresh": nms_t,
                "multimask": multimask,
                "stability_filter": stability,
                "final_parts": num_parts,
                "miou_vs_ref": round(miou, 4),
                "boundary_edge_ratio_pct": round(ber * 100.0, 3),
                "boundary_edges": boundary_edges,
                "total_edges": total_edges,
                "total_time_s": round(total_time, 2),
                "peak_vram_mb": round(peak_vram_mb, 1),
            }
            results.append(record)
            
            print(f"--> Done in {total_time:.2f}s | Parts: {num_parts} | mIoU: {miou:.4f} | BER: {ber*100.0:.2f}% | VRAM: {peak_vram_mb:.1f} MB")
            run_idx += 1
            
    # Save full results JSON
    json_path = os.path.join(out_root, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {json_path}")
    
    # Print formatted markdown table
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    print(f"| Mode | NMS Thresh | Parts | Reference mIoU | Boundary Edge Ratio (%) | Runtime (s) | Peak VRAM (MB) |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r['mode']} | {r['nms_thresh']:.1f} | {r['final_parts']} | {r['miou_vs_ref']:.4f} | {r['boundary_edge_ratio_pct']:.2f}% | {r['total_time_s']:.2f}s | {r['peak_vram_mb']:.1f} MB |")
    print("=" * 90)

if __name__ == "__main__":
    run_benchmark()
