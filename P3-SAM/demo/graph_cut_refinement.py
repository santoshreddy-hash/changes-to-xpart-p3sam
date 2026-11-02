"""
Graph Cut Refinement Module for 3D Mesh Part Segmentation (P3-SAM).

Provides multi-label Graph Cut boundary refinement using alpha-expansion:
1. Topological mesh adjacency graph with dihedral angle and edge length weighting:
   w(f, g) = (edge_length / mean_length) * exp(-theta^2 / sigma^2)
2. Spatial kNN graph over face centroids for meshes with non-manifold or poor connectivity:
   w_knn(f, g) = exp(-d^2 / (2 * sigma_d^2)) * exp(-theta^2 / sigma_theta^2)
3. Soft point-to-face probability transfer and negative log-likelihood unary potentials:
   U(f, l) = -log(P(f, l) + epsilon)
4. Multi-label energy minimization via alpha-expansion (gco / pygco).
"""

import time
import numpy as np
import trimesh
from scipy.spatial import cKDTree

try:
    import gco
    GCO_AVAILABLE = True
except ImportError:
    GCO_AVAILABLE = False


def build_mesh_adjacency_graph(mesh, sigma=0.35, length_weight=True):
    """
    Constructs a topological face-adjacency graph from the 3D mesh.
    
    Edges connect adjacent faces that share an edge in the mesh manifold.
    Pairwise weight:
        w(f, g) = edge_length * exp(-theta^2 / sigma^2)
    where theta is the dihedral angle between face normals of f and g.
    
    Parameters:
        mesh: trimesh.Trimesh
        sigma: float, dihedral angle sensitivity parameter in radians (default: 0.35 rad ~ 20 deg)
        length_weight: bool, whether to scale weights by normalized shared edge length
        
    Returns:
        edges: np.ndarray [E, 2] int32, sorted such that edges[:, 0] < edges[:, 1]
        edge_weights: np.ndarray [E] float64
    """
    adj = mesh.face_adjacency
    if len(adj) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float64)

    angles = mesh.face_adjacency_angles  # [E] dihedral angle in radians
    
    # Orientation: ensure u < v for gco
    e0 = np.minimum(adj[:, 0], adj[:, 1]).astype(np.int32)
    e1 = np.maximum(adj[:, 0], adj[:, 1]).astype(np.int32)
    edges = np.stack([e0, e1], axis=1)

    # Angle term: exp(-theta^2 / sigma^2)
    # When theta is small (flat surface), exp(...) ~ 1.0 (high penalty for cutting)
    # When theta is large (e.g. 90-degree corner, theta ~ 1.57), exp(...) -> 0 (cut penalty vanishes)
    angle_weights = np.exp(-(angles ** 2) / (sigma ** 2))

    if length_weight and len(mesh.face_adjacency_edges) > 0:
        edges_v = mesh.face_adjacency_edges
        edge_coords = mesh.vertices[edges_v]
        lengths = np.linalg.norm(edge_coords[:, 0] - edge_coords[:, 1], axis=-1)
        mean_l = np.mean(lengths)
        if mean_l > 1e-8:
            norm_lengths = lengths / mean_l
        else:
            norm_lengths = np.ones_like(lengths)
        edge_weights = (norm_lengths * angle_weights).astype(np.float64)
    else:
        edge_weights = angle_weights.astype(np.float64)

    return edges, edge_weights


def build_spatial_knn_graph(mesh, k=6, sigma_dist=None, sigma_angle=0.35):
    """
    Constructs a spatial kNN graph over face centroids for meshes without reliable
    topological connectivity (e.g. non-manifold geometry, soups of triangles, or open boundaries).
    
    Edges connect the k nearest face centroids in 3D Euclidean space.
    Pairwise weight:
        w_knn(f, g) = exp(-d^2 / (2 * sigma_d^2)) * exp(-theta^2 / sigma_theta^2)
        
    Parameters:
        mesh: trimesh.Trimesh
        k: int, number of nearest neighbors per face
        sigma_dist: float or None, spatial distance scale (defaults to mean neighbor distance)
        sigma_angle: float, normal angular difference scale in radians
        
    Returns:
        edges: np.ndarray [E, 2] int32, sorted such that edges[:, 0] < edges[:, 1]
        edge_weights: np.ndarray [E] float64
    """
    n_faces = len(mesh.faces)
    if n_faces <= 1:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float64)

    centroids = mesh.triangles.mean(axis=1)  # [F, 3]
    normals = mesh.face_normals             # [F, 3]
    
    tree = cKDTree(centroids)
    k_query = min(k + 1, n_faces)
    dists, indices = tree.query(centroids, k=k_query)

    # Exclude self-neighbors (index 0)
    src = np.repeat(np.arange(n_faces, dtype=np.int32), k_query - 1)
    dst = indices[:, 1:].reshape(-1).astype(np.int32)
    dist_flat = dists[:, 1:].reshape(-1)

    # Dot product between face normals
    src_normals = normals[src]
    dst_normals = normals[dst]
    dots = np.clip(np.sum(src_normals * dst_normals, axis=1), -1.0, 1.0)
    theta = np.arccos(dots)

    # Determine sigma_dist
    if sigma_dist is None:
        mean_d = np.mean(dist_flat)
        sigma_dist = mean_d if mean_d > 1e-8 else 1.0

    weights_raw = np.exp(-(dist_flat ** 2) / (2.0 * (sigma_dist ** 2))) * np.exp(-(theta ** 2) / (sigma_angle ** 2))

    # Form undirected edges u < v
    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    valid = (u != v)
    u, v, weights_raw = u[valid], v[valid], weights_raw[valid]

    # Combine duplicate edges (e.g. if A is neighbor of B and B is neighbor of A)
    edge_key = u.astype(np.int64) * n_faces + v.astype(np.int64)
    order = np.argsort(edge_key)
    u_sorted = u[order]
    v_sorted = v[order]
    w_sorted = weights_raw[order]
    keys_sorted = edge_key[order]

    unique_keys, first_idx = np.unique(keys_sorted, return_index=True)
    counts = np.diff(np.append(first_idx, len(keys_sorted)))
    
    w_final = w_sorted[first_idx].copy()
    two_mask = (counts >= 2)
    if np.any(two_mask):
        w_final[two_mask] = (w_sorted[first_idx[two_mask]] + w_sorted[first_idx[two_mask] + 1]) * 0.5

    edges = np.stack([u_sorted[first_idx], v_sorted[first_idx]], axis=1)
    return edges, w_final.astype(np.float64)


def transfer_point_probs_to_faces(face_idx, point_probs, n_faces, initial_face_ids=None, prior_weight=0.75):
    """
    Transfers continuous point-level probabilities onto mesh faces.
    
    For faces with sampled points:
        P(f, l) = (1 / |S_f|) * sum_{p in S_f} P_point(p, l)
        
    For faces without direct sampled points:
        If initial_face_ids is provided, uses a softened prior centered on the initial label:
            P(f, l) = prior_weight if l == initial_face_ids[f] else (1 - prior_weight) / (L - 1)
        Otherwise uses a uniform prior:
            P(f, l) = 1.0 / L
            
    Parameters:
        face_idx: np.ndarray [N] int, face index corresponding to each sampled point
        point_probs: np.ndarray [N, L] float, probability of each point belonging to label l
        n_faces: int, total number of faces in mesh
        initial_face_ids: np.ndarray [F] int or None, initial coarse face labels (0..L-1)
        prior_weight: float in (0, 1), confidence given to initial label on unsampled faces
        
    Returns:
        face_probs: np.ndarray [F, L] float32 in [0, 1], normalized per face
    """
    N, L = point_probs.shape
    face_prob_sum = np.zeros((n_faces, L), dtype=np.float32)
    face_point_cnt = np.zeros((n_faces, 1), dtype=np.float32)

    # Vectorized accumulation
    np.add.at(face_prob_sum, face_idx, point_probs.astype(np.float32))
    np.add.at(face_point_cnt, face_idx, 1.0)

    has_points = (face_point_cnt[:, 0] > 0)
    face_probs = np.zeros((n_faces, L), dtype=np.float32)
    face_probs[has_points] = face_prob_sum[has_points] / face_point_cnt[has_points]

    # For unsampled faces:
    unsampled = ~has_points
    if np.any(unsampled):
        if initial_face_ids is not None and len(initial_face_ids) == n_faces:
            # Construct softened prior based on initial flood fill labels
            p_other = (1.0 - prior_weight) / max(1, L - 1)
            prior = np.full((np.sum(unsampled), L), p_other, dtype=np.float32)
            init_unsampled = initial_face_ids[unsampled]
            valid_init = (init_unsampled >= 0) & (init_unsampled < L)
            if np.any(valid_init):
                rows = np.where(valid_init)[0]
                cols = init_unsampled[valid_init]
                prior[rows, cols] = prior_weight
            face_probs[unsampled] = prior
        else:
            # Uniform prior: neutral data cost
            face_probs[unsampled] = 1.0 / L

    # Normalize rows to sum to 1.0
    row_sums = np.sum(face_probs, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    face_probs = face_probs / row_sums

    return face_probs


def compute_unary_costs(face_probs, epsilon=1e-6, scale=1000.0):
    """
    Calculates negative log probability unary costs:
        U(f, l) = -log(P(f, l) + epsilon)
        
    Scaled by `scale` to preserve precision during integer discretization in graph cut.
    
    Parameters:
        face_probs: np.ndarray [F, L] float
        epsilon: float, small numerical stabilizer
        scale: float, energy scaling factor (default: 1000.0)
        
    Returns:
        unary_cost: np.ndarray [F, L] float64 (or scaled int32)
    """
    neg_log = -np.log(np.clip(face_probs, epsilon, 1.0))
    return (neg_log * scale).astype(np.float64)


def compute_pairwise_potentials(n_labels, scale=1000.0):
    """
    Calculates standard Potts model pairwise compatibility matrix:
        V(l1, l2) = 0 if l1 == l2 else scale
        
    Parameters:
        n_labels: int, number of discrete segmentation parts
        scale: float, base smoothness scale
        
    Returns:
        pairwise_cost: np.ndarray [L, L] float64
    """
    return ((1.0 - np.eye(n_labels, dtype=np.float64)) * scale)


def run_multilabel_graphcut(
    edges,
    edge_weights,
    face_probs,
    smoothness_weight=1.0,
    algorithm='expansion',
    init_labels=None,
    n_iter=5,
    cost_scale=1000.0,
):
    """
    Solves multi-label Graph Cut optimization using Boykov-Kolmogorov alpha-expansion.
    
    Energy formulation:
        E(L) = sum_{f} U(f, L_f) + smoothness_weight * sum_{(f, g) in E} w(f, g) * [L_f != L_g]
        
    Parameters:
        edges: np.ndarray [E, 2] int32
        edge_weights: np.ndarray [E] float64
        face_probs: np.ndarray [F, L] float
        smoothness_weight: float, multiplier on edge smoothness terms
        algorithm: 'expansion' or 'swap'
        init_labels: np.ndarray [F] int32 or None, initial label configuration
        n_iter: int, max alpha-expansion iterations (-1 for convergence)
        cost_scale: float, scaling factor for energy potentials
        
    Returns:
        refined_labels: np.ndarray [F] int32 in range [0, L-1]
        diagnostics: dict with runtime and energy statistics
    """
    if not GCO_AVAILABLE:
        raise RuntimeError("gco package (gco-wrapper / pygco) is not installed in the environment.")

    t0 = time.time()
    n_faces, n_labels = face_probs.shape
    
    unary_cost = compute_unary_costs(face_probs, epsilon=1e-6, scale=cost_scale)
    pairwise_cost = compute_pairwise_potentials(n_labels, scale=smoothness_weight)
    
    # Scale edge weights by cost_scale
    scaled_edge_weights = (edge_weights * cost_scale).astype(np.float64)

    # Ensure init_labels is valid int32 [F]
    if init_labels is not None:
        init_labels = np.ascontiguousarray(np.clip(init_labels, 0, n_labels - 1), dtype=np.int32)

    t1 = time.time()
    refined_labels = gco.cut_general_graph(
        edges,
        scaled_edge_weights,
        unary_cost,
        pairwise_cost,
        n_iter=n_iter,
        algorithm=algorithm,
        init_labels=init_labels,
    )
    t2 = time.time()

    diagnostics = {
        "n_faces": n_faces,
        "n_labels": n_labels,
        "n_edges": len(edges),
        "setup_time_s": t1 - t0,
        "solver_time_s": t2 - t1,
        "total_time_s": t2 - t0,
    }
    return refined_labels, diagnostics


def refine_mesh_segmentation(
    mesh,
    face_idx,
    point_probs,
    initial_face_ids,
    unique_labels,
    refine_mode="graphcut",
    smoothness_weight=1.0,
    sigma=0.35,
    knn_k=6,
    n_iter=5,
):
    """
    High-level orchestrator for Graph Cut refinement.
    
    Parameters:
        mesh: trimesh.Trimesh
        face_idx: np.ndarray [N], face index for each point sample
        point_probs: np.ndarray [N, L], continuous probabilities
        initial_face_ids: np.ndarray [F], initial face labels
        unique_labels: np.ndarray [L], original part label IDs corresponding to 0..L-1
        refine_mode: 'graphcut' (topological) or 'graphcut_knn' (spatial kNN)
        smoothness_weight: float, pairwise energy multiplier
        sigma: float, dihedral / angle sensitivity
        knn_k: int, nearest neighbors for spatial graph
        n_iter: int, max iterations
        
    Returns:
        refined_face_ids: np.ndarray [F], refined face labels mapped back to original label IDs
        diagnostics: dict
    """
    n_faces = len(mesh.faces)
    L = len(unique_labels)
    label_to_dense = {orig: i for i, orig in enumerate(unique_labels)}
    dense_to_label = {i: orig for i, orig in enumerate(unique_labels)}

    # Map initial face IDs to 0..L-1
    dense_init = np.zeros(n_faces, dtype=np.int32)
    for orig, i in label_to_dense.items():
        dense_init[initial_face_ids == orig] = i

    # Transfer point probabilities to faces
    t_start = time.time()
    face_probs = transfer_point_probs_to_faces(
        face_idx=face_idx,
        point_probs=point_probs,
        n_faces=n_faces,
        initial_face_ids=dense_init,
        prior_weight=0.75,
    )

    # Build Graph
    t_g0 = time.time()
    if refine_mode == "graphcut_knn":
        edges, weights = build_spatial_knn_graph(mesh, k=knn_k, sigma_angle=sigma)
    else:
        edges, weights = build_mesh_adjacency_graph(mesh, sigma=sigma, length_weight=True)
    t_g1 = time.time()

    # Run Solver
    refined_dense, solver_diag = run_multilabel_graphcut(
        edges=edges,
        edge_weights=weights,
        face_probs=face_probs,
        smoothness_weight=smoothness_weight,
        algorithm='expansion',
        init_labels=dense_init,
        n_iter=n_iter,
    )
    t_end = time.time()

    # Map back to original label IDs
    refined_face_ids = np.zeros(n_faces, dtype=initial_face_ids.dtype)
    for i, orig in dense_to_label.items():
        refined_face_ids[refined_dense == i] = orig

    diag = {
        "mode": refine_mode,
        "n_faces": n_faces,
        "n_parts": L,
        "graph_build_time_s": t_g1 - t_g0,
        "solver_time_s": solver_diag["solver_time_s"],
        "total_refine_time_s": t_end - t_start,
        "changed_faces": int(np.sum(refined_face_ids != initial_face_ids)),
        "changed_percent": float(np.mean(refined_face_ids != initial_face_ids) * 100.0),
    }
    return refined_face_ids, diag
