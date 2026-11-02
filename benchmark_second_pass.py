"""
Benchmark Script for Second-Pass Prompting in P3-SAM.
Compares:
1. Baseline: 400 Prompts (Original Automatic Pipeline)
2. Second Pass: 400 + 50 Prompts
3. Second Pass: 400 + 100 Prompts
4. Second Pass: 400 + 200 Prompts

Measures:
- Hungarian mIoU & Matched mIoU
- Small-part recall (across all 4 GT area groups: <0.5%, 0.5-1%, 1-5%, >5%)
- Part-count accuracy
- Component separation / merging check
- Runtime breakdown (Pass 1 inference, Pass 2 generation & inference, NMS, cleanup, total)
- Additional prompt counts
- Peak GPU VRAM & CPU RAM
- Produces high-resolution 3D visual comparisons and prompt distributions
"""

import os
import sys
import time
import json
import psutil
import torch
import trimesh
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

# Add P3-SAM/demo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "P3-SAM", "demo")))
import auto_mask_fast as mod

def compute_hungarian_metrics(pred_face_ids, gt_face_ids, gt_areas, total_mesh_area):
    """
    Computes:
    - Hungarian mIoU (penalized by max(n_gt, n_pred))
    - Matched mIoU (mean across GT parts)
    - Recall @ IoU >= 0.50 & Recall @ IoU >= 0.75 overall and per area bin:
        * <0.5%
        * 0.5 - 1.0%
        * 1.0 - 5.0%
        * >5.0%
    - Component separation / merger check: count of predicted parts that bridge multiple GT parts
    """
    u_pred = np.unique(pred_face_ids[pred_face_ids >= 0])
    u_gt = np.unique(gt_face_ids[gt_face_ids >= 0])
    
    n_p = len(u_pred)
    n_g = len(u_gt)
    
    if n_p == 0 or n_g == 0:
        return {
            "hungarian_miou": 0.0,
            "matched_miou": 0.0,
            "recall_50_overall": 0.0,
            "recall_75_overall": 0.0,
            "group_metrics": {},
            "n_pred_parts": n_p,
            "n_gt_parts": n_g,
            "part_count_error": abs(n_p - n_g),
            "merged_components_count": 0,
        }

    # Build GT area percentages & classify into bins
    gt_part_bins = {}
    gt_masks = {}
    for g in u_gt:
        m = (gt_face_ids == g)
        gt_masks[g] = m
        p_area = np.sum(gt_areas[m])
        pct = (p_area / total_mesh_area) * 100.0
        if pct < 0.5:
            b = "<0.5%"
        elif pct < 1.0:
            b = "0.5-1.0%"
        elif pct < 5.0:
            b = "1.0-5.0%"
        else:
            b = ">5.0%"
        gt_part_bins[g] = {"bin": b, "area_pct": pct, "faces": int(np.sum(m))}

    # IoU Matrix [n_g, n_p]
    iou_matrix = np.zeros((n_g, n_p), dtype=np.float64)
    pred_masks = {}
    for j, p in enumerate(u_pred):
        pred_masks[p] = (pred_face_ids == p)

    for i, g in enumerate(u_gt):
        g_mask = gt_masks[g]
        g_sum = np.sum(g_mask)
        for j, p in enumerate(u_pred):
            p_mask = pred_masks[p]
            inter = np.sum(g_mask & p_mask)
            union = g_sum + np.sum(p_mask) - inter
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]

    # Overall metrics
    hungarian_miou = float(np.sum(matched_ious) / max(n_g, n_p))
    matched_miou = float(np.mean(matched_ious))
    recall_50_overall = float(np.sum(matched_ious >= 0.50) / n_g)
    recall_75_overall = float(np.sum(matched_ious >= 0.75) / n_g)

    # Per-group breakdown
    group_defs = ["<0.5%", "0.5-1.0%", "1.0-5.0%", ">5.0%"]
    group_metrics = {}
    for b in group_defs:
        group_metrics[b] = {
            "n_gt": 0,
            "recalled_50": 0,
            "recalled_75": 0,
            "recall_50": 0.0,
            "recall_75": 0.0,
            "avg_iou": 0.0,
            "part_details": [],
        }

    for idx, (r, c) in enumerate(zip(row_ind, col_ind)):
        g_id = u_gt[r]
        p_id = u_pred[c]
        cur_iou = matched_ious[idx]
        b = gt_part_bins[g_id]["bin"]
        
        group_metrics[b]["n_gt"] += 1
        if cur_iou >= 0.50:
            group_metrics[b]["recalled_50"] += 1
        if cur_iou >= 0.75:
            group_metrics[b]["recalled_75"] += 1
        group_metrics[b]["part_details"].append({
            "gt_id": int(g_id),
            "pred_id": int(p_id),
            "area_pct": gt_part_bins[g_id]["area_pct"],
            "iou": float(cur_iou),
        })

    for b in group_defs:
        n_b = group_metrics[b]["n_gt"]
        if n_b > 0:
            group_metrics[b]["recall_50"] = group_metrics[b]["recalled_50"] / n_b
            group_metrics[b]["recall_75"] = group_metrics[b]["recalled_75"] / n_b
            group_metrics[b]["avg_iou"] = float(np.mean([d["iou"] for d in group_metrics[b]["part_details"]]))
        else:
            group_metrics[b]["recall_50"] = 0.0
            group_metrics[b]["recall_75"] = 0.0
            group_metrics[b]["avg_iou"] = 0.0

    # Component separation / accidental merger check:
    # A predicted part p is considered an "accidental merger" if it substantially covers (>=25% of GT part faces)
    # more than one distinct GT part.
    merged_components_count = 0
    merger_details = []
    for j, p in enumerate(u_pred):
        p_mask = pred_masks[p]
        p_size = np.sum(p_mask)
        if p_size == 0:
            continue
        covered_gt_parts = []
        for g in u_gt:
            g_mask = gt_masks[g]
            inter = np.sum(p_mask & g_mask)
            g_size = np.sum(g_mask)
            if inter >= 0.25 * g_size and inter >= 0.10 * p_size:
                covered_gt_parts.append((int(g), float(inter / g_size)))
        if len(covered_gt_parts) >= 2:
            merged_components_count += 1
            merger_details.append({"pred_id": int(p), "merged_gt_parts": covered_gt_parts})

    return {
        "hungarian_miou": hungarian_miou,
        "matched_miou": matched_miou,
        "recall_50_overall": recall_50_overall,
        "recall_75_overall": recall_75_overall,
        "n_pred_parts": n_p,
        "n_gt_parts": n_g,
        "part_count_error": abs(n_p - n_g),
        "merged_components_count": merged_components_count,
        "merger_details": merger_details,
        "group_metrics": group_metrics,
        "matched_ious": matched_ious.tolist(),
    }


def save_colorized_mesh(mesh, face_ids, save_glb_path):
    """
    Saves a colorized GLB mesh with visually distinctive colors.
    """
    unique_ids = np.unique(face_ids[face_ids >= 0])
    rng = np.random.RandomState(42)
    color_map = {}
    import colorsys
    for uid in unique_ids:
        hue = rng.uniform(0, 1)
        sat = rng.uniform(0.70, 0.95)
        val = rng.uniform(0.75, 0.98)
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


def render_mesh_views(mesh, face_ids, width=900, height=900, elev=22, azim=40):
    """
    Renders shaded 3D mesh with part colors.
    """
    unique_ids = np.unique(face_ids[face_ids >= 0])
    rng = np.random.RandomState(42)
    import colorsys
    color_map = {}
    for uid in unique_ids:
        hue = rng.uniform(0, 1)
        sat = rng.uniform(0.75, 0.95)
        val = rng.uniform(0.80, 0.98)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        color_map[uid] = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
        
    point_density = 400000
    points, face_indices = trimesh.sample.sample_surface(mesh, point_density)
    normals = mesh.face_normals[face_indices]
    
    # Assign part colors
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for i, f_idx in enumerate(face_indices):
        fid = face_ids[f_idx]
        if fid in color_map:
            colors[i] = color_map[fid]
        else:
            colors[i] = [140, 140, 140]

    # Center & normalize
    center = points.mean(axis=0)
    pts = points - center
    scale = np.max(np.abs(pts))
    pts /= (scale + 1e-8)
    
    rad_elev = np.radians(elev)
    rad_azim = np.radians(azim)
    cos_a, sin_a = np.cos(rad_azim), np.sin(rad_azim)
    R_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    cos_e, sin_e = np.cos(rad_elev), np.sin(rad_elev)
    R_x = np.array([[1, 0, 0], [0, cos_e, -sin_e], [0, sin_e, cos_e]])
    R = R_x @ R_z
    
    pts_cam = pts @ R.T
    normals_cam = normals @ R.T
    
    # Lighting
    light_dir = np.array([0.35, 0.45, 0.82])
    light_dir /= np.linalg.norm(light_dir)
    diffuse = np.maximum(0.0, np.sum(normals_cam * light_dir, axis=-1))
    ambient = 0.45
    intensity = np.clip(ambient + 0.55 * diffuse, 0.0, 1.0)[:, None]
    colors_lit = np.clip(colors.astype(np.float32) * intensity, 0, 255).astype(np.uint8)
    
    # Project
    pad = 0.08
    x_cam = pts_cam[:, 0]
    y_cam = pts_cam[:, 1]
    z_cam = pts_cam[:, 2]
    
    u = ((x_cam + (1.0 + pad)) / (2.0 * (1.0 + pad)) * (width - 1)).astype(np.int32)
    v = (((1.0 + pad) - y_cam) / (2.0 * (1.0 + pad)) * (height - 1)).astype(np.int32)
    
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid]
    v = v[valid]
    z_cam = z_cam[valid]
    colors_lit = colors_lit[valid]
    
    # Sort back to front
    sort_order = np.argsort(z_cam)
    u = u[sort_order]
    v = v[sort_order]
    colors_lit = colors_lit[sort_order]
    
    # Dark modern canvas
    img = np.full((height, width, 3), 24, dtype=np.uint8)
    for du in [0, 1, -1]:
        for dv in [0, 1, -1]:
            uu = np.clip(u + du, 0, width - 1)
            vv = np.clip(v + dv, 0, height - 1)
            img[vv, uu] = colors_lit
            
    return Image.fromarray(img)


def render_prompt_distribution(
    sampled_points,
    first_pass_prompts,
    uncertain_mask,
    second_pass_prompts,
    width=900,
    height=900,
    elev=22,
    azim=40,
):
    """
    Renders 3D prompt selection visualization showing:
    - Base surface point cloud: Dim neutral gray
    - Uncertain regions: Vivid orange/amber
    - Pass 1 FPS prompts: Vibrant blue spheres (radius 5)
    - Pass 2 Targeted prompts: Bright neon green spheres (radius 7)
    """
    center = sampled_points.mean(axis=0)
    pts = sampled_points - center
    scale = np.max(np.abs(pts))
    pts /= (scale + 1e-8)
    
    # Transform coordinates
    rad_elev = np.radians(elev)
    rad_azim = np.radians(azim)
    cos_a, sin_a = np.cos(rad_azim), np.sin(rad_azim)
    R_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    cos_e, sin_e = np.cos(rad_elev), np.sin(rad_elev)
    R_x = np.array([[1, 0, 0], [0, cos_e, -sin_e], [0, sin_e, cos_e]])
    R = R_x @ R_z
    
    pts_cam = pts @ R.T
    
    pad = 0.08
    u = ((pts_cam[:, 0] + (1.0 + pad)) / (2.0 * (1.0 + pad)) * (width - 1)).astype(np.int32)
    v = (((1.0 + pad) - pts_cam[:, 1]) / (2.0 * (1.0 + pad)) * (height - 1)).astype(np.int32)
    z = pts_cam[:, 2]
    
    # Assign point colors
    # Base: (70, 75, 85)
    # Uncertain: (255, 130, 20)
    colors = np.full((len(pts), 3), [70, 75, 85], dtype=np.uint8)
    if uncertain_mask is not None and len(uncertain_mask) == len(pts):
        colors[uncertain_mask] = [255, 140, 20]  # Amber/orange
        
    sort_order = np.argsort(z)
    u_sorted = u[sort_order]
    v_sorted = v[sort_order]
    colors_sorted = colors[sort_order]
    
    img = np.full((height, width, 3), 20, dtype=np.uint8)
    for du in [0, 1]:
        for dv in [0, 1]:
            uu = np.clip(u_sorted + du, 0, width - 1)
            vv = np.clip(v_sorted + dv, 0, height - 1)
            img[vv, uu] = colors_sorted
            
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Render Pass 1 prompts (Blue circles)
    p1_pts = (first_pass_prompts - center) / (scale + 1e-8)
    p1_cam = p1_pts @ R.T
    u1 = ((p1_cam[:, 0] + (1.0 + pad)) / (2.0 * (1.0 + pad)) * (width - 1)).astype(np.int32)
    v1 = (((1.0 + pad) - p1_cam[:, 1]) / (2.0 * (1.0 + pad)) * (height - 1)).astype(np.int32)
    
    r1 = 4
    for px, py in zip(u1, v1):
        if 0 <= px < width and 0 <= py < height:
            draw.ellipse([px - r1, py - r1, px + r1, py + r1], fill=(40, 150, 255), outline=(200, 230, 255), width=1)
            
    # Render Pass 2 prompts (Bright Neon Green circles)
    if second_pass_prompts is not None and len(second_pass_prompts) > 0:
        p2_pts = (second_pass_prompts - center) / (scale + 1e-8)
        p2_cam = p2_pts @ R.T
        u2 = ((p2_cam[:, 0] + (1.0 + pad)) / (2.0 * (1.0 + pad)) * (width - 1)).astype(np.int32)
        v2 = (((1.0 + pad) - p2_cam[:, 1]) / (2.0 * (1.0 + pad)) * (height - 1)).astype(np.int32)
        
        r2 = 6
        for px, py in zip(u2, v2):
            if 0 <= px < width and 0 <= py < height:
                draw.ellipse([px - r2, py - r2, px + r2, py + r2], fill=(20, 255, 90), outline=(255, 255, 255), width=2)

    return pil_img


def run_benchmark():
    mesh_path = "uploads/furn16_woodenrack_raw_854k_d95a8fe0.glb"
    ref_gt_path = "outputs/web_furn16_woodenrack_raw_854k_d95a8fe0_66650142/p3sam/furn16_woodenrack_raw_854k_d95a8fe0_face_ids.npy"
    out_root = "outputs/benchmark_secondpass"
    os.makedirs(out_root, exist_ok=True)

    print("=" * 84)
    print("P3-SAM BENCHMARK: SECOND-PASS PROMPTING MECHANISM FOR MISSED & THIN COMPONENTS")
    print("=" * 84)
    print(f"Target Mesh: {mesh_path}")
    print(f"Reference GT Face Labels: {ref_gt_path}\n")

    mesh = trimesh.load(mesh_path, force="mesh")
    gt_face_ids = np.load(ref_gt_path)

    # Clean mesh to align faces
    mesh_cleaned = mod.clean_mesh(mesh.copy())
    total_mesh_area = float(np.sum(mesh_cleaned.area_faces))
    gt_areas = mesh_cleaned.area_faces

    print(f"Mesh Statistics: {len(mesh.faces):,} raw faces -> {len(mesh_cleaned.faces):,} cleaned faces")
    print(f"Total Mesh Surface Area: {total_mesh_area:.4f}")
    print(f"Ground Truth Parts: {len(np.unique(gt_face_ids[gt_face_ids >= 0]))} parts\n")

    # Initialize frozen P3-SAM model once
    print("Initializing frozen P3-SAM model...")
    auto_mask = mod.AutoMask(ckpt_path=None)
    print("Model initialized on CUDA. Ready for multi-configuration evaluation.\n")

    configs = [
        {
            "name": "Baseline_400",
            "label": "Original 400 Prompts (Single Pass)",
            "second_pass_enabled": False,
            "max_additional_prompts": 0,
            "conf_threshold": 0.60,
            "curvature_threshold": 0.25,
            "min_region_size": 15,
        },
        {
            "name": "SecondPass_Plus50",
            "label": "Second Pass (+50 Targeted Prompts)",
            "second_pass_enabled": True,
            "max_additional_prompts": 50,
            "conf_threshold": 0.60,
            "curvature_threshold": 0.25,
            "min_region_size": 15,
        },
        {
            "name": "SecondPass_Plus100",
            "label": "Second Pass (+100 Targeted Prompts)",
            "second_pass_enabled": True,
            "max_additional_prompts": 100,
            "conf_threshold": 0.60,
            "curvature_threshold": 0.25,
            "min_region_size": 15,
        },
        {
            "name": "SecondPass_Plus200",
            "label": "Second Pass (+200 Targeted Prompts)",
            "second_pass_enabled": True,
            "max_additional_prompts": 200,
            "conf_threshold": 0.60,
            "curvature_threshold": 0.25,
            "min_region_size": 15,
        },
    ]

    benchmark_records = []
    metadata_store = {}

    for idx, cfg in enumerate(configs):
        cfg_name = cfg["name"]
        cfg_dir = os.path.join(out_root, cfg_name)
        os.makedirs(cfg_dir, exist_ok=True)

        print("-" * 84)
        print(f"[{idx + 1}/{len(configs)}] RUNNING CONFIGURATION: {cfg['label']}")
        print(f"Parameters: second_pass={cfg['second_pass_enabled']}, max_additional_prompts={cfg['max_additional_prompts']}")
        print(f"Saving to: {cfg_dir}")

        # Reset GPU peak memory tracking & GC
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        proc = psutil.Process()
        ram_before = proc.memory_info().rss / (1024 * 1024)

        t_start = time.perf_counter()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            aabb, face_ids, res_mesh = mod.mesh_sam(
                [auto_mask.model, auto_mask.model],
                mesh.copy(),
                save_path=cfg_dir,
                point_num=100000,
                prompt_num=400,
                threshold=0.95,
                post_process=False,
                show_info=True,
                save_mid_res=False,
                clean_mesh_flag=True,
                seed=42,
                prompt_bs=8,
                loop_mode="opt3",
                nms_mode="fast",
                cleanup_mode="fast",
                multimask_output=False,
                nms_thresh=0.9,
                refine_mode="scipy",
                preserve_raw_faces=True,
                second_pass_enabled=cfg["second_pass_enabled"],
                max_additional_prompts=cfg["max_additional_prompts"],
                conf_threshold=cfg["conf_threshold"],
                curvature_threshold=cfg["curvature_threshold"],
                min_region_size=cfg["min_region_size"],
                max_second_pass_iterations=1,
            )

        torch.cuda.synchronize()
        t_total = time.perf_counter() - t_start
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        ram_peak = proc.memory_info().rss / (1024 * 1024)

        # Retrieve metadata
        p1_prompts = res_mesh.metadata.get("first_pass_prompts", None)
        sampled_pts = res_mesh.metadata.get("sampled_points", None)
        p2_prompts_list = res_mesh.metadata.get("second_pass_prompts", [])
        uncertain_mask = res_mesh.metadata.get("uncertain_mask", None)

        p2_prompts = np.concatenate(p2_prompts_list, axis=0) if len(p2_prompts_list) > 0 else np.zeros((0, 3), dtype=np.float32)
        actual_add_prompts = len(p2_prompts)

        # Store metadata for visualization
        metadata_store[cfg_name] = {
            "p1_prompts": p1_prompts,
            "p2_prompts": p2_prompts,
            "sampled_pts": sampled_pts,
            "uncertain_mask": uncertain_mask,
            "face_ids": face_ids,
            "res_mesh": res_mesh,
        }

        # Save face IDs & colorized GLB
        np.save(os.path.join(cfg_dir, "face_ids.npy"), face_ids)
        save_colorized_mesh(res_mesh, face_ids, os.path.join(cfg_dir, "colorized_result.glb"))

        # Evaluate Hungarian metrics and group recall
        eval_res = compute_hungarian_metrics(face_ids, gt_face_ids, gt_areas, total_mesh_area)

        record = {
            "name": cfg_name,
            "label": cfg["label"],
            "second_pass_enabled": cfg["second_pass_enabled"],
            "max_additional_prompts": cfg["max_additional_prompts"],
            "actual_additional_prompts": actual_add_prompts,
            "total_prompts": 400 + actual_add_prompts,
            "total_runtime_s": round(t_total, 3),
            "peak_vram_mb": round(peak_vram, 1),
            "peak_ram_mb": round(ram_peak, 1),
            "n_pred_parts": eval_res["n_pred_parts"],
            "n_gt_parts": eval_res["n_gt_parts"],
            "part_count_error": eval_res["part_count_error"],
            "hungarian_miou": round(eval_res["hungarian_miou"] * 100.0, 2),
            "matched_miou": round(eval_res["matched_miou"] * 100.0, 2),
            "recall_50_overall": round(eval_res["recall_50_overall"] * 100.0, 2),
            "recall_75_overall": round(eval_res["recall_75_overall"] * 100.0, 2),
            "merged_components_count": eval_res["merged_components_count"],
            "group_metrics": eval_res["group_metrics"],
        }
        benchmark_records.append(record)

        print(f"Results for {cfg_name}:")
        print(f"  Additional Prompts Sampled: {actual_add_prompts}")
        print(f"  Predicted Parts: {record['n_pred_parts']} (GT: 16, error: {record['part_count_error']})")
        print(f"  Hungarian mIoU: {record['hungarian_miou']}%")
        print(f"  Matched mIoU: {record['matched_miou']}%")
        print(f"  Recall @ IoU>=0.50 (Overall): {record['recall_50_overall']}%")
        print(f"  Thin/Small Parts Recall (1-5% area): {round(eval_res['group_metrics']['1.0-5.0%']['recall_50'] * 100.0, 2)}% (avg IoU: {round(eval_res['group_metrics']['1.0-5.0%']['avg_iou'] * 100.0, 2)}%)")
        print(f"  Accidental Merged Components: {record['merged_components_count']}")
        print(f"  Total Runtime: {record['total_runtime_s']}s | Peak VRAM: {record['peak_vram_mb']}MB\n")

    # Save benchmark JSON summary
    with open(os.path.join(out_root, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)

    # ==================== VISUALIZATION GENERATION ====================
    print("=" * 84)
    print("GENERATING VISUALIZATION ARTIFACTS...")
    print("=" * 84)

    # 1. Prompt Selection Visualizations for Config C (SecondPass_Plus100)
    meta_100 = metadata_store["SecondPass_Plus100"]
    if meta_100["sampled_pts"] is not None and meta_100["p1_prompts"] is not None:
        print("Rendering Prompt Selection 3D Distribution (Front & Angled views)...")
        img_prompts_front = render_prompt_distribution(
            sampled_points=meta_100["sampled_pts"],
            first_pass_prompts=meta_100["p1_prompts"],
            uncertain_mask=meta_100["uncertain_mask"],
            second_pass_prompts=meta_100["p2_prompts"],
            width=900,
            height=900,
            elev=20,
            azim=40,
        )
        img_prompts_side = render_prompt_distribution(
            sampled_points=meta_100["sampled_pts"],
            first_pass_prompts=meta_100["p1_prompts"],
            uncertain_mask=meta_100["uncertain_mask"],
            second_pass_prompts=meta_100["p2_prompts"],
            width=900,
            height=900,
            elev=15,
            azim=110,
        )

        # Composite prompt figure
        prompt_composite = Image.new("RGB", (1820, 980), (15, 17, 23))
        prompt_composite.paste(img_prompts_front, (10, 70))
        prompt_composite.paste(img_prompts_side, (910, 70))
        draw_pc = ImageDraw.Draw(prompt_composite)
        
        # Legend & Title
        draw_pc.text((25, 20), "P3-SAM Second-Pass Prompting: Multi-Factor Uncertainty Map & Targeted Prompts", fill=(255, 255, 255))
        # Legend items
        draw_pc.rectangle([950, 20, 965, 35], fill=(70, 75, 85))
        draw_pc.text((975, 20), "100k Surface Points", fill=(200, 205, 215))
        draw_pc.rectangle([1140, 20, 1155, 35], fill=(255, 140, 20))
        draw_pc.text((1165, 20), "Uncertain Regions (Low-Conf/High-Curv)", fill=(255, 160, 40))
        draw_pc.ellipse([1450, 20, 1465, 35], fill=(40, 150, 255))
        draw_pc.text((1475, 20), "Pass 1 FPS (400 pts)", fill=(100, 190, 255))
        draw_pc.ellipse([1640, 20, 1655, 35], fill=(20, 255, 90))
        draw_pc.text((1665, 20), "Pass 2 Targeted (+100 pts)", fill=(60, 255, 120))

        prompt_vis_path = os.path.join(out_root, "prompt_selection_visualization.png")
        prompt_composite.save(prompt_vis_path)
        print(f"Saved prompt selection visualization to {prompt_vis_path}")

    # 2. Before-and-After Segmentation Renders
    print("Rendering Before-and-After Segmentation Comparison (Baseline vs Second-Pass)...")
    res_base = metadata_store["Baseline_400"]
    res_p100 = metadata_store["SecondPass_Plus100"]

    img_base_render = render_mesh_views(res_base["res_mesh"], res_base["face_ids"], width=900, height=900, elev=20, azim=40)
    img_p100_render = render_mesh_views(res_p100["res_mesh"], res_p100["face_ids"], width=900, height=900, elev=20, azim=40)

    # Side-by-side composite
    comp_img = Image.new("RGB", (1820, 980), (15, 17, 23))
    comp_img.paste(img_base_render, (10, 70))
    comp_img.paste(img_p100_render, (910, 70))

    draw_c = ImageDraw.Draw(comp_img)
    draw_c.text((25, 22), "BEFORE: Original 400 Prompts (Baseline Pipeline)", fill=(240, 240, 250))
    draw_c.text((925, 22), "AFTER: Second-Pass Prompting (400 + 100 Targeted Prompts)", fill=(70, 255, 140))

    comp_vis_path = os.path.join(out_root, "before_after_segmentation_comparison.png")
    comp_img.save(comp_vis_path)
    print(f"Saved segmentation comparison to {comp_vis_path}\n")

    print("=" * 84)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 84)


if __name__ == "__main__":
    run_benchmark()
