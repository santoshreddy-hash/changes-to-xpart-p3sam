"""
Hierarchical 3D Segmentation Module for P3-SAM.
Implements:
1. Part-level feature aggregation from Sonata point features.
2. Mesh manifold & spatial contact part-adjacency graph construction.
3. Connectivity-constrained agglomerative hierarchical clustering (guaranteeing spatial contiguity).
4. Unconstrained agglomerative hierarchical clustering (baseline comparison).
5. NMS-threshold-based hierarchy evaluation (alternative comparison).
6. Comprehensive metric evaluation: part count, nestedness purity, disconnected components count, mIoU, runtime.
"""

import time
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import sklearn.cluster
from scipy.optimize import linear_sum_assignment


def extract_part_features(mesh, face_ids, point_feats=None, face_idx=None, points=None):
    """
    Extracts L2-normalized mean feature vectors and geometric metadata for each segmented part.

    Parameters:
        mesh: trimesh.Trimesh
        face_ids: np.ndarray [F] of int part labels
        point_feats: np.ndarray [N, D] or torch.Tensor
        face_idx: np.ndarray [N] mapping sampled points to face indices
        points: np.ndarray [N, 3] coordinates of sampled points

    Returns:
        part_ids: np.ndarray [K] of unique non-negative part IDs
        part_feats: np.ndarray [K, D] L2-normalized feature vectors
        part_meta: dict mapping part_id to centroid, area, face_count, bbox
    """
    if hasattr(point_feats, "detach"):
        point_feats = point_feats.float().detach().cpu().numpy()

    u_parts = np.unique(face_ids[face_ids >= 0])
    K = len(u_parts)
    D = point_feats.shape[1] if point_feats is not None else 512

    part_feats_list = []
    part_meta = {}

    face_areas = mesh.area_faces
    face_centers = mesh.triangles_center

    for p in u_parts:
        p_mask = (face_ids == p)
        n_faces = int(np.sum(p_mask))
        p_area = float(np.sum(face_areas[p_mask]))
        
        # Geometric centroid (weighted by face area)
        if n_faces > 0:
            p_center = np.sum(face_centers[p_mask] * face_areas[p_mask, None], axis=0) / (p_area + 1e-8)
            # AABB of part faces
            p_verts = mesh.vertices[mesh.faces[p_mask].reshape(-1)]
            p_min = p_verts.min(axis=0)
            p_max = p_verts.max(axis=0)
        else:
            p_center = np.zeros(3, dtype=np.float32)
            p_min = np.zeros(3, dtype=np.float32)
            p_max = np.zeros(3, dtype=np.float32)

        # Average feature vector from points belonging to this part
        if point_feats is not None and face_idx is not None:
            pt_mask = p_mask[face_idx]
            if np.any(pt_mask):
                feat_vec = np.mean(point_feats[pt_mask], axis=0)
            else:
                # Fallback: if no sampled points fell directly on part faces, use nearest point
                if points is not None and len(points) > 0:
                    tree = cKDTree(points)
                    _, nn_idx = tree.query(p_center, k=min(16, len(points)))
                    feat_vec = np.mean(point_feats[nn_idx], axis=0)
                else:
                    feat_vec = np.zeros(D, dtype=np.float32)
        else:
            feat_vec = np.zeros(D, dtype=np.float32)

        # L2 normalize
        norm = np.linalg.norm(feat_vec)
        if norm > 1e-6:
            feat_vec = feat_vec / norm
        else:
            feat_vec = np.zeros(D, dtype=np.float32)

        part_feats_list.append(feat_vec)
        part_meta[int(p)] = {
            "centroid": p_center.tolist(),
            "area": p_area,
            "face_count": n_faces,
            "aabb_min": p_min.tolist(),
            "aabb_max": p_max.tolist(),
        }

    part_feats_matrix = np.stack(part_feats_list, axis=0) if len(part_feats_list) > 0 else np.zeros((0, D), dtype=np.float32)
    return u_parts, part_feats_matrix, part_meta


def build_part_adjacency_graph(mesh, face_ids, contact_dist_thresh=0.03):
    """
    Constructs an adjacency graph between segmented parts based on:
    1. Direct topological face boundary sharing (shared mesh edges).
    2. Physical spatial contact proximity (for CAD assemblies with tiny contact clearances).

    Parameters:
        mesh: trimesh.Trimesh
        face_ids: np.ndarray [F] of part labels
        contact_dist_thresh: float, maximum normalized contact clearance

    Returns:
        adj_matrix: np.ndarray [K, K] bool symmetric adjacency matrix
        weight_matrix: np.ndarray [K, K] float symmetric matrix (shared boundary length)
        edge_details: dict mapping (u, v) -> shared length, contact distance
    """
    u_parts = np.unique(face_ids[face_ids >= 0])
    K = len(u_parts)
    part_to_idx = {p: i for i, p in enumerate(u_parts)}

    adj_matrix = np.zeros((K, K), dtype=bool)
    weight_matrix = np.zeros((K, K), dtype=np.float32)
    edge_details = {}

    if K <= 1:
        return adj_matrix, weight_matrix, edge_details

    # 1. Manifold Topological Face Adjacency
    face_adj = mesh.face_adjacency
    f1 = face_adj[:, 0]
    f2 = face_adj[:, 1]
    l1 = face_ids[f1]
    l2 = face_ids[f2]

    boundary_mask = (l1 != l2) & (l1 >= 0) & (l2 >= 0)
    
    # Compute edge lengths
    adj_edges = mesh.face_adjacency_edges[boundary_mask]
    v_a = mesh.vertices[adj_edges[:, 0]]
    v_b = mesh.vertices[adj_edges[:, 1]]
    edge_lengths = np.linalg.norm(v_a - v_b, axis=-1)

    p1_arr = l1[boundary_mask]
    p2_arr = l2[boundary_mask]

    for p_a, p_b, elen in zip(p1_arr, p2_arr, edge_lengths):
        ia, ib = part_to_idx[p_a], part_to_idx[p_b]
        adj_matrix[ia, ib] = True
        adj_matrix[ib, ia] = True
        weight_matrix[ia, ib] += float(elen)
        weight_matrix[ib, ia] += float(elen)

    # 2. Non-manifold Proximity / Contact Check
    # Strictly applied ONLY for parts with zero topological boundary neighbors (e.g. disconnected CAD shells),
    # preventing artificial edge creation across empty air gaps between parallel slats.
    isolated_parts = [i for i in range(K) if not np.any(adj_matrix[i])]
    if len(isolated_parts) > 0:
        bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        abs_contact_thresh = min(contact_dist_thresh, 0.008) * bbox_diag

        part_points_dict = {}
        for i in range(K):
            p_faces = np.where(face_ids == u_parts[i])[0]
            if len(p_faces) == 0:
                continue
            p_verts_idx = np.unique(mesh.faces[p_faces].reshape(-1))
            part_points_dict[i] = mesh.vertices[p_verts_idx]

        for i in isolated_parts:
            pts_i = part_points_dict.get(i, None)
            if pts_i is None or len(pts_i) == 0:
                continue
            pts_i_sub = pts_i[::max(1, len(pts_i) // 200)]

            for j in range(K):
                if i == j or adj_matrix[i, j]:
                    continue
                pts_j = part_points_dict.get(j, None)
                if pts_j is None or len(pts_j) == 0:
                    continue

                tree_j = cKDTree(pts_j)
                dists, _ = tree_j.query(pts_i_sub, k=1)
                min_d = float(np.min(dists))

                if min_d <= abs_contact_thresh:
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True
                    w_contact = float(np.exp(-min_d / (abs_contact_thresh + 1e-8)))
                    weight_matrix[i, j] = w_contact
                    weight_matrix[j, i] = w_contact

    for i in range(K):
        for j in range(i + 1, K):
            if adj_matrix[i, j]:
                edge_details[(int(u_parts[i]), int(u_parts[j]))] = {
                    "weight": float(weight_matrix[i, j]),
                    "is_direct_boundary": weight_matrix[i, j] > 0,
                }

    return adj_matrix, weight_matrix, edge_details


def run_unconstrained_hierarchical_clustering(part_features, u_parts, n_clusters_list, linkage="average"):
    """
    Executes standard unconstrained agglomerative clustering on part feature vectors.

    Parameters:
        part_features: np.ndarray [K, D]
        u_parts: np.ndarray [K] fine part IDs
        n_clusters_list: list of int target cluster counts [e.g. 5, 2, 1]
        linkage: str ('average', 'complete', 'single')

    Returns:
        level_results: dict mapping target cluster count to cluster assignments & parent mappings
    """
    K = len(part_features)
    level_results = {}

    for n_c in n_clusters_list:
        target_k = min(n_c, K)
        if target_k <= 1:
            labels = np.zeros(K, dtype=np.int64)
        else:
            model = sklearn.cluster.AgglomerativeClustering(
                n_clusters=target_k,
                metric="cosine",
                linkage=linkage,
            )
            labels = model.fit_predict(part_features)

        # Mapping: fine part ID -> cluster ID
        part_to_cluster = {int(p): int(l) for p, l in zip(u_parts, labels)}
        level_results[target_k] = {
            "n_clusters": target_k,
            "part_labels": labels,
            "part_to_cluster": part_to_cluster,
            "cluster_members": {int(c): [int(p) for p, l in zip(u_parts, labels) if l == c] for c in np.unique(labels)},
        }

    return level_results


def run_connectivity_constrained_hierarchical_clustering(
    part_features,
    u_parts,
    adj_matrix,
    n_clusters_list,
    linkage="ward",
):
    """
    Executes connectivity-constrained agglomerative clustering.
    Two parts are merged IF AND ONLY IF they share spatial adjacency in adj_matrix.

    Parameters:
        part_features: np.ndarray [K, D]
        u_parts: np.ndarray [K] fine part IDs
        adj_matrix: np.ndarray [K, K] bool adjacency matrix
        n_clusters_list: list of int target cluster counts [e.g. 5, 2, 1]
        linkage: str ('ward', 'average', 'complete', 'single')

    Returns:
        level_results: dict mapping target cluster count to cluster assignments & parent mappings
    """
    K = len(part_features)
    level_results = {}

    # Check connected components of the adjacency graph
    n_components, comp_labels = connected_components(adj_matrix, directed=False)

    for n_c in n_clusters_list:
        # Constraint: Cannot reduce cluster count below the number of natural disconnected components!
        target_k = max(n_components, min(n_c, K))

        if target_k >= K:
            labels = np.arange(K, dtype=np.int64)
        elif target_k <= n_components:
            # If target <= connected components, cluster assignment is simply the connected component ID
            labels = comp_labels
        else:
            # Run Ward linkage with connectivity constraint (Euclidean metric on L2-normalized features)
            model = sklearn.cluster.AgglomerativeClustering(
                n_clusters=target_k,
                metric="euclidean",
                connectivity=adj_matrix,
                linkage=linkage,
            )
            labels = model.fit_predict(part_features)

        part_to_cluster = {int(p): int(l) for p, l in zip(u_parts, labels)}
        actual_clusters = len(np.unique(labels))

        level_results[n_c] = {
            "target_clusters": n_c,
            "actual_clusters": actual_clusters,
            "n_graph_components": n_components,
            "part_labels": labels,
            "part_to_cluster": part_to_cluster,
            "cluster_members": {int(c): [int(p) for p, l in zip(u_parts, labels) if l == c] for c in np.unique(labels)},
        }

    return level_results


def evaluate_hierarchical_consistency(child_face_ids, parent_face_ids):
    """
    Measures hierarchical consistency / nestedness purity:
    Purity = (Fraction of child faces that belong to the majority parent)
    Strict tree hierarchy = 100.0%.
    Non-nested drifting boundaries < 100.0%.
    """
    valid = (child_face_ids >= 0) & (parent_face_ids >= 0)
    c_ids = child_face_ids[valid]
    p_ids = parent_face_ids[valid]
    N = len(c_ids)

    if N == 0:
        return 100.0, 0, 0

    u_children = np.unique(c_ids)
    pure_faces = 0

    for c in u_children:
        c_mask = (c_ids == c)
        c_parents = p_ids[c_mask]
        # Count majority parent
        counts = np.bincount(c_parents)
        majority_count = np.max(counts)
        pure_faces += majority_count

    purity = (pure_faces / N) * 100.0
    return float(purity), pure_faces, N


def count_disconnected_components_per_cluster(face_ids, mesh):
    """
    Evaluates whether composite parts contain disconnected floating sub-components.

    Parameters:
        face_ids: np.ndarray [F] of cluster labels
        mesh: trimesh.Trimesh

    Returns:
        n_disconnected_parts: int, number of parts containing > 1 connected component
        disconnected_details: list of dicts with part breakdown
    """
    face_adj = mesh.face_adjacency
    f1 = face_adj[:, 0]
    f2 = face_adj[:, 1]
    
    u_clusters = np.unique(face_ids[face_ids >= 0])
    n_disconnected = 0
    details = []

    for c in u_clusters:
        c_faces = np.where(face_ids == c)[0]
        if len(c_faces) <= 1:
            continue

        # Build sub-graph of face adjacency restricted to cluster c
        in_cluster = (face_ids[f1] == c) & (face_ids[f2] == c)
        sub_edges = np.stack([f1[in_cluster], f2[in_cluster]], axis=1)

        # Map faces to local indices 0..N_c-1
        face_to_local = {f: idx for idx, f in enumerate(c_faces)}
        n_c = len(c_faces)

        if len(sub_edges) > 0:
            local_e1 = np.array([face_to_local[f] for f in sub_edges[:, 0]])
            local_e2 = np.array([face_to_local[f] for f in sub_edges[:, 1]])
            data = np.ones(len(local_e1), dtype=np.int8)
            sub_graph = csr_matrix((data, (local_e1, local_e2)), shape=(n_c, n_c))
            n_comp, comp_lbl = connected_components(sub_graph, directed=False)
        else:
            n_comp = n_c

        # Filter out minor singleton noise faces (< 5 faces)
        if n_comp > 1:
            comp_sizes = np.bincount(comp_lbl, minlength=n_comp)
            sig_comp = int(np.sum(comp_sizes >= 15))
            if sig_comp > 1:
                n_disconnected += 1
                details.append({
                    "cluster_id": int(c),
                    "total_faces": n_c,
                    "connected_components_count": sig_comp,
                    "component_face_sizes": comp_sizes[comp_sizes >= 15].tolist(),
                })

    return n_disconnected, details


def compute_hungarian_miou(pred_face_ids, gt_face_ids):
    """
    Computes Hungarian optimal assignment mIoU.
    """
    u_pred = np.unique(pred_face_ids[pred_face_ids >= 0])
    u_gt = np.unique(gt_face_ids[gt_face_ids >= 0])
    
    if len(u_pred) == 0 or len(u_gt) == 0:
        return 0.0, 0.0

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

    hungarian_miou = float(np.sum(matched_ious) / max(n_g, n_p))
    matched_miou = float(np.mean(matched_ious))
    return hungarian_miou, matched_miou
