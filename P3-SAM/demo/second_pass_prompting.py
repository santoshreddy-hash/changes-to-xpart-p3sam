"""
Second-Pass Prompting Module for P3-SAM 3D Point-Cloud / Mesh Segmentation.

Identifies uncertain, missed, low-confidence, and high-curvature regions from
a first-pass segmentation and generates targeted prompt points to recover small
and thin components without re-running the costly Sonata backbone feature extraction.

Core Workflow:
1. Analyze first-pass coverage, confidence, curvature, and region sizes.
2. Form an uncertainty priority map and filter out isolated single-point noise.
3. Sample additional prompts strictly inside uncertain clusters using Farthest Point Sampling.
4. Reuse pre-extracted Sonata point features (zero backbone re-run).
5. Predict masks only for the new prompts and merge useful candidates into the pool.
6. Apply unified GPU NMS to eliminate redundancy.
"""

import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import fpsample


def compute_point_curvature(points, normals, k=16):
    """
    Computes local geometric surface curvature / normal variation for 3D points.
    
    Parameters:
        points: np.ndarray [N, 3] float
        normals: np.ndarray [N, 3] float
        k: int, nearest neighbor count
        
    Returns:
        curvature: np.ndarray [N] float in [0, 1]
    """
    N = len(points)
    if N <= k:
        return np.zeros(N, dtype=np.float32)
        
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=k)
    
    # Exclude self (index 0)
    nb_normals = normals[idxs[:, 1:]]  # [N, k-1, 3]
    dots = np.clip(np.sum(normals[:, None, :] * nb_normals, axis=-1), -1.0, 1.0)
    # 0 for flat surface (dot=1), up to 1+ for sharp creases/edges
    curvature = 1.0 - np.mean(dots, axis=-1).astype(np.float32)
    return np.clip(curvature, 0.0, 2.0)


def identify_uncertain_regions(
    points,
    normals,
    cand_masks_np,
    cand_ious_np,
    cand_probs_np=None,
    conf_threshold=0.60,
    curvature_threshold=0.25,
    min_region_size=15,
):
    """
    Analyzes first-pass candidate predictions and geometry to identify missed,
    low-confidence, and high-curvature regions.
    
    Parameters:
        points: np.ndarray [N, 3] float
        normals: np.ndarray [N, 3] float
        cand_masks_np: np.ndarray [P, N] bool
        cand_ious_np: np.ndarray [P] float
        cand_probs_np: np.ndarray [P, N] float or None
        conf_threshold: float, confidence below which points are deemed uncertain
        curvature_threshold: float, curvature above which points indicate edges/thin features
        min_region_size: int, minimum connected point count to filter out stray noise
        
    Returns:
        uncertain_mask: np.ndarray [N] bool, points that belong to validated uncertain clusters
        uncertain_priority: np.ndarray [N] float, priority score for sampling
        diagnostics: dict with breakdown counts
    """
    N = len(points)
    P = len(cand_masks_np)
    
    if P == 0:
        uncertain_mask = np.ones(N, dtype=bool)
        return uncertain_mask, np.ones(N, dtype=np.float32), {"n_unlabeled": N, "n_low_conf": N}

    # 1. Hit Count & Unlabeled points
    hit_count = np.sum(cand_masks_np, axis=0)  # [N]
    unlabeled_mask = (hit_count == 0)
    n_unlabeled = int(np.sum(unlabeled_mask))

    # 2. Maximum Prediction Confidence
    if cand_probs_np is not None and len(cand_probs_np) > 0:
        max_conf = np.max(cand_probs_np, axis=0)  # [N]
    else:
        # Approximate confidence via max IoU of covering masks
        # [P, N] * [P, 1] -> [P, N]
        weighted = cand_masks_np.astype(np.float32) * np.array(cand_ious_np)[:, None]
        max_conf = np.max(weighted, axis=0) if P > 0 else np.zeros(N, dtype=np.float32)
        
    low_conf_mask = (max_conf < conf_threshold)
    n_low_conf = int(np.sum(low_conf_mask))

    # 3. Geometric Curvature
    curv = compute_point_curvature(points, normals, k=16)
    high_curv_mask = (curv > curvature_threshold)
    n_high_curv = int(np.sum(high_curv_mask))

    # 4. Conflicting Neighbor / Boundary Margin Ambiguity
    # If a point has competing masks with close probabilities:
    conflict_mask = np.zeros(N, dtype=bool)
    if cand_probs_np is not None and P >= 2:
        # Top-2 probability margin
        part_probs_sorted = np.sort(cand_probs_np, axis=0)  # ascending
        top1 = part_probs_sorted[-1]
        top2 = part_probs_sorted[-2]
        margin = top1 - top2
        conflict_mask = (top1 > 0.40) & (margin < 0.15)
    n_conflict = int(np.sum(conflict_mask))

    # 5. Composite Raw Uncertainty Mask
    # Points are uncertain if they are:
    # - completely unlabeled (hit_count == 0)
    # - low-confidence predictions (max_conf < conf_threshold) with non-saturated coverage (hit_count <= 4)
    # - geometric crease/curvature features without strong confident coverage (curv > curvature_threshold and max_conf < 0.75)
    # - conflicting boundary margin ambiguity between top-2 masks
    raw_uncertain = (
        unlabeled_mask | 
        (low_conf_mask & (hit_count <= 4)) | 
        (high_curv_mask & (max_conf < 0.75)) | 
        conflict_mask
    )
    
    # Priority score for weighting
    priority = (
        2.5 * unlabeled_mask.astype(np.float32) +
        1.8 * (1.0 - np.clip(max_conf, 0.0, 1.0)) +
        1.2 * np.clip(curv, 0.0, 1.0) +
        0.8 * conflict_mask.astype(np.float32)
    )

    # 6. Spatial Connected-Component Filtering (Eliminate Isolated Noise Points)
    uncertain_indices = np.where(raw_uncertain)[0]
    n_raw_uncertain = len(uncertain_indices)
    
    if n_raw_uncertain == 0:
        return np.zeros(N, dtype=bool), priority, {
            "n_unlabeled": n_unlabeled,
            "n_low_conf": n_low_conf,
            "n_high_curv": n_high_curv,
            "n_conflict": n_conflict,
            "n_raw_uncertain": 0,
            "n_filtered_uncertain": 0,
            "n_clusters": 0,
        }

    # Build kNN graph over uncertain points to find connected components
    pts_u = points[uncertain_indices]
    tree_u = cKDTree(pts_u)
    
    # Estimate adaptive neighborhood radius: 3.0x median nearest neighbor distance (min 0.02)
    dists_u, _ = tree_u.query(pts_u, k=2)
    median_d = np.median(dists_u[:, 1]) if len(dists_u) > 0 else 0.05
    radius = max(median_d * 3.0, 0.02)
    
    pairs = tree_u.query_pairs(radius, output_type='ndarray')
    if len(pairs) > 0:
        data = np.ones(len(pairs), dtype=np.int8)
        u_graph = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n_raw_uncertain, n_raw_uncertain))
        n_comp, comp_labels = connected_components(u_graph, directed=False)
        
        # Count points per component
        comp_counts = np.bincount(comp_labels, minlength=n_comp)
        valid_comp = (comp_counts >= min_region_size)
        valid_u_mask = valid_comp[comp_labels]
        
        filtered_indices = uncertain_indices[valid_u_mask]
        n_clusters = int(np.sum(valid_comp))
    else:
        # No pairs within radius -> all singletons
        if n_raw_uncertain >= min_region_size:
            filtered_indices = uncertain_indices
            n_clusters = 1
        else:
            filtered_indices = np.array([], dtype=np.int64)
            n_clusters = 0

    # Fallback: If strict min_region_size removed all points despite significant raw uncertainty,
    # keep the highest priority uncertain points so targeted sampling can still rescue the regions
    if len(filtered_indices) == 0 and n_raw_uncertain > 0:
        top_k = min(n_raw_uncertain, max(min_region_size * 2, 100))
        sub_prio = priority[uncertain_indices]
        top_sort = np.argsort(-sub_prio)[:top_k]
        filtered_indices = uncertain_indices[top_sort]
        n_clusters = 1

    final_uncertain_mask = np.zeros(N, dtype=bool)
    final_uncertain_mask[filtered_indices] = True

    diagnostics = {
        "n_unlabeled": n_unlabeled,
        "n_low_conf": n_low_conf,
        "n_high_curv": n_high_curv,
        "n_conflict": n_conflict,
        "n_raw_uncertain": n_raw_uncertain,
        "n_filtered_uncertain": int(len(filtered_indices)),
        "n_clusters": n_clusters,
    }
    return final_uncertain_mask, priority, diagnostics


def sample_second_pass_prompts(
    points,
    uncertain_mask,
    uncertain_priority,
    max_additional_prompts=100,
    seed=42,
):
    """
    Selects targeted additional prompt coordinates inside validated uncertain regions
    using Farthest Point Sampling (FPS) to ensure maximum coverage across separate components.
    
    Parameters:
        points: np.ndarray [N, 3] float
        uncertain_mask: np.ndarray [N] bool
        uncertain_priority: np.ndarray [N] float
        max_additional_prompts: int
        seed: int
        
    Returns:
        additional_prompts: np.ndarray [M, 3] float
        selected_point_indices: np.ndarray [M] int
    """
    cand_indices = np.where(uncertain_mask)[0]
    n_cand = len(cand_indices)
    
    if n_cand == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64)

    target_count = min(max_additional_prompts, n_cand)
    
    if n_cand <= target_count:
        return points[cand_indices], cand_indices

    cand_pts = points[cand_indices]
    
    # Run FPS within the candidate uncertain points to disperse prompts
    fps_sub_idx = fpsample.fps_sampling(cand_pts, target_count)
    selected_indices = cand_indices[fps_sub_idx]
    additional_prompts = points[selected_indices]
    
    return additional_prompts, selected_indices


def run_second_pass_inference(
    model_parallel,
    feats_gpu,
    points_gpu,
    second_pass_prompts,
    prompt_bs=8,
    multimask_output=False,
    min_mask_iou=0.0,
    min_mask_points=10,
    stability_filter=False,
    stability_thresh=0.85,
    stability_offset=0.05,
    return_probs=False,
    get_mask_fn=None,
    show_info=False,
):
    """
    Executes prompt inference strictly for second-pass prompts while reusing
    cached GPU point features (zero backbone re-run).
    
    Parameters:
        model_parallel: PyTorch module
        feats_gpu: torch.Tensor [N, 512] on GPU
        points_gpu: torch.Tensor [N, 3] on GPU
        second_pass_prompts: np.ndarray [M, 3]
        prompt_bs: int
        multimask_output: bool
        min_mask_iou: float
        min_mask_points: int
        stability_filter: bool
        stability_thresh: float
        stability_offset: float
        return_probs: bool
        get_mask_fn: callable (get_mask_fast)
        show_info: bool
        
    Returns:
        pass2_masks_gpu: torch.Tensor [P2, N] bool CUDA tensor
        pass2_ious_gpu: torch.Tensor [P2] float CUDA tensor
        pass2_probs_gpu: torch.Tensor [P2, N] float CUDA tensor or None
    """
    M = len(second_pass_prompts)
    point_num = points_gpu.shape[0]
    if M == 0:
        empty_mask = torch.zeros((0, point_num), dtype=torch.bool, device="cuda")
        empty_iou = torch.zeros(0, dtype=torch.float32, device="cuda")
        empty_prob = torch.zeros((0, point_num), dtype=torch.float32, device="cuda") if return_probs else None
        return empty_mask, empty_iou, empty_prob

    step_num = (M + prompt_bs - 1) // prompt_bs
    mask_list = []
    iou_list = []
    prob_list = [] if return_probs else None

    for i in range(step_num):
        cur_prompt = second_pass_prompts[prompt_bs * i : prompt_bs * (i + 1)]
        K_cur = len(cur_prompt)
        if K_cur == 0:
            continue
            
        if multimask_output:
            if stability_filter:
                res = get_mask_fn(
                    model_parallel, feats_gpu, points_gpu, cur_prompt,
                    calc_stability=True, stability_offset=stability_offset, return_probs=return_probs
                )
                if return_probs:
                    m1, m2, m3, pred_iou, stab_scores, (p1, p2, p3) = res
                else:
                    m1, m2, m3, pred_iou, stab_scores = res
            else:
                res = get_mask_fn(
                    model_parallel, feats_gpu, points_gpu, cur_prompt,
                    calc_stability=False, return_probs=return_probs
                )
                if return_probs:
                    m1, m2, m3, pred_iou, (p1, p2, p3) = res
                else:
                    m1, m2, m3, pred_iou = res

            cand_masks = torch.stack([m1, m2, m3], dim=-1).permute(1, 2, 0).reshape(K_cur * 3, point_num)
            cand_ious = pred_iou.reshape(K_cur * 3)
            if return_probs:
                cand_probs = torch.stack([p1, p2, p3], dim=-1).permute(1, 2, 0).reshape(K_cur * 3, point_num)

            valid = torch.ones(K_cur * 3, dtype=torch.bool, device=cand_masks.device)
            if min_mask_iou > 0.0:
                valid = valid & (cand_ious >= min_mask_iou)
            if min_mask_points > 0:
                cand_sizes = cand_masks.sum(dim=1)
                valid = valid & (cand_sizes >= min_mask_points)
            if stability_filter:
                cand_stabs = stab_scores.reshape(K_cur * 3)
                valid = valid & (cand_stabs >= stability_thresh)

            if valid.any():
                mask_list.append(cand_masks[valid])
                iou_list.append(cand_ious[valid])
                if return_probs:
                    prob_list.append(cand_probs[valid])
        else:
            if return_probs:
                m1, m2, m3, pred_iou, (p1, p2, p3) = get_mask_fn(
                    model_parallel, feats_gpu, points_gpu, cur_prompt,
                    calc_stability=False, return_probs=True
                )
            else:
                m1, m2, m3, pred_iou = get_mask_fn(
                    model_parallel, feats_gpu, points_gpu, cur_prompt,
                    calc_stability=False, return_probs=False
                )

            pred_mask = torch.stack([m1, m2, m3], dim=-1)
            max_idx = torch.argmax(pred_iou, dim=-1)
            ar = torch.arange(K_cur, device=pred_mask.device)
            selected = pred_mask[:, ar, max_idx]  # [N, K_cur]
            selected_iou = pred_iou[ar, max_idx]  # [K_cur]
            if return_probs:
                p_stack = torch.stack([p1, p2, p3], dim=-1)
                selected_prob = p_stack[:, ar, max_idx]  # [N, K_cur]

            valid = torch.ones(K_cur, dtype=torch.bool, device=selected.device)
            if min_mask_iou > 0.0:
                valid = valid & (selected_iou >= min_mask_iou)
            if min_mask_points > 0:
                cand_sizes = selected.sum(dim=0)
                valid = valid & (cand_sizes >= min_mask_points)

            if valid.any():
                mask_list.append(selected[:, valid].transpose(0, 1))
                iou_list.append(selected_iou[valid])
                if return_probs:
                    prob_list.append(selected_prob[:, valid].transpose(0, 1))

    if len(mask_list) > 0:
        pass2_masks_gpu = torch.cat(mask_list, dim=0)
        pass2_ious_gpu = torch.cat(iou_list, dim=0)
        pass2_probs_gpu = torch.cat(prob_list, dim=0) if return_probs and len(prob_list) > 0 else None
    else:
        pass2_masks_gpu = torch.zeros((0, point_num), dtype=torch.bool, device="cuda")
        pass2_ious_gpu = torch.zeros(0, dtype=torch.float32, device="cuda")
        pass2_probs_gpu = torch.zeros((0, point_num), dtype=torch.float32, device="cuda") if return_probs else None

    return pass2_masks_gpu, pass2_ious_gpu, pass2_probs_gpu


def merge_candidate_pools(
    pass1_masks,
    pass1_ious,
    pass1_probs,
    pass2_masks,
    pass2_ious,
    pass2_probs,
):
    """
    Merges Pass 1 and Pass 2 candidate masks into a unified candidate pool.
    
    Parameters:
        pass1_*: PyTorch CUDA tensors or None
        pass2_*: PyTorch CUDA tensors or None
        
    Returns:
        merged_masks: torch.Tensor [P_total, N] bool
        merged_ious: torch.Tensor [P_total] float
        merged_probs: torch.Tensor [P_total, N] float or None
    """
    if pass2_masks is None or len(pass2_masks) == 0:
        return pass1_masks, pass1_ious, pass1_probs

    if pass1_masks is None or len(pass1_masks) == 0:
        return pass2_masks, pass2_ious, pass2_probs

    merged_masks = torch.cat([pass1_masks, pass2_masks], dim=0)
    merged_ious = torch.cat([pass1_ious, pass2_ious], dim=0)
    
    if pass1_probs is not None and pass2_probs is not None:
        merged_probs = torch.cat([pass1_probs, pass2_probs], dim=0)
    else:
        merged_probs = None

    return merged_masks, merged_ious, merged_probs
