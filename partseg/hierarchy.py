"""Optional: merge the final parts down to a requested number of parts."""

from __future__ import annotations

import numpy as np

from .errors import InvalidInputError


def part_feature_means(feats: np.ndarray, point_labels: np.ndarray, n_parts: int) -> np.ndarray:
    """Mean P3-SAM feature of every part (rows follow part id)."""
    dim = feats.shape[1]
    sums = np.zeros((n_parts, dim))
    counts = np.bincount(point_labels, minlength=n_parts).astype(np.float64)
    for j in range(dim):
        sums[:, j] = np.bincount(point_labels, weights=feats[:, j], minlength=n_parts)
    counts[counts == 0] = 1.0
    return sums / counts[:, None]


def merge_to_target(face_ids: np.ndarray, parts: list[dict], features: np.ndarray, target: int) -> np.ndarray:
    """
    Agglomerate neighbouring parts (Ward linkage on mean features, connectivity from the part contacts)
    until ``target`` parts remain. Returns relabelled face ids (0 = largest part is NOT guaranteed here;
    the caller renumbers). Parts are only merged with parts they touch, so no part spans a gap.
    """
    from scipy.sparse import coo_matrix
    from sklearn.cluster import AgglomerativeClustering

    n = len(parts)
    if target < 1:
        raise InvalidInputError("target part count must be at least 1")
    if target >= n:
        return face_ids
    index = {p["id"]: i for i, p in enumerate(parts)}
    rows, cols = [], []
    for p in parts:
        for nb in p["neighbors"]:
            rows.append(index[p["id"]])
            cols.append(index[nb["id"]])
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.int8)
    feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
    labels = AgglomerativeClustering(n_clusters=target, connectivity=adj, linkage="ward").fit_predict(feats)
    lut = np.empty(int(face_ids.max()) + 1, dtype=np.int64)
    for p in parts:
        lut[p["id"]] = labels[index[p["id"]]]
    return lut[face_ids]
