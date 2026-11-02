"""Split large parts that P3-SAM left as one blob.

Two complementary, guarded strategies:

* ``reprompt_large_parts``: prompt the mask decoder at many points inside a big part and
  accept only sub-masks that several independent prompts agree on.
* ``split_by_features``: cluster P3-SAM's own point features inside a big part, and keep a
  split only if the mask decoder, prompted inside each child, returns that child.

Both skip flat panels (a table top is one part) and never touch the part boundary labels
of other parts. All sampling is seeded (fixed FPS start points, fixed clustering seeds).
"""

from __future__ import annotations

import logging

import fpsample
import numpy as np
import torch
from scipy.spatial import cKDTree

from .config import FeatureSplitConfig, RepromptConfig
from .p3sam import p3sam_module
from .polish import planar_fraction

log = logging.getLogger("partseg")


def _assign_points_to_faces(face_ids, new_ids, parents, point_parent, point_labels, face_idx, graph, width):
    """Turn per-point labels inside split parents into per-face labels (majority vote + nearest fill)."""
    for parent in parents:
        faces_p = np.where(face_ids == parent)[0]
        sel = point_parent == parent
        f_pts = face_idx[sel].astype(np.int64)
        uk, cnt = np.unique(f_pts * width + point_labels[sel], return_counts=True)
        uf, ulab = uk // width, uk % width
        order = np.lexsort((-cnt, uf))
        uf_s, ulab_s = uf[order], ulab[order]
        first = np.r_[True, uf_s[1:] != uf_s[:-1]]
        faces_with, lab_with = uf_s[first], ulab_s[first]
        new_ids[faces_with] = lab_with
        rest = np.setdiff1d(faces_p, faces_with)
        if len(rest) and len(faces_with):
            _, nn = cKDTree(graph.centroids[faces_with]).query(graph.centroids[rest])
            new_ids[rest] = lab_with[nn]


def reprompt_large_parts(model, feats, pts, face_idx, face_ids, graph, face_normals, cfg: RepromptConfig):
    """One round of consensus re-prompting. Returns (new face ids, {parent label: sub-parts added})."""
    mod = p3sam_module()
    n_pts = len(pts)
    pts_gpu = torch.from_numpy(pts).float().cuda()
    point_parent = face_ids[face_idx]
    new_ids = face_ids.copy()
    next_label = int(face_ids.max()) + 1
    point_labels = point_parent.copy()
    parents: list[int] = []
    report: dict[int, int] = {}

    for label in np.unique(face_ids[face_ids >= 0]):
        if graph.areas[face_ids == label].sum() / graph.total_area < cfg.big_frac:
            continue
        sel = point_parent == label
        idx = np.where(sel)[0]
        n_in = len(idx)
        if n_in < 600:
            continue
        if planar_fraction(face_ids, label, face_normals, graph) >= cfg.planar_guard:
            continue
        k = int(min(cfg.prompts, n_in // 30))
        sub = pts[idx]
        picked = np.unique(np.concatenate([
            fpsample.fps_sampling(sub, k, start_idx=int(s * n_in // cfg.starts)) for s in range(cfg.starts)
        ]))
        prompts = sub[picked]
        sel_gpu = torch.from_numpy(sel).cuda()

        cands = []  # (score, mask over this part's points, prompt id)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b0 in range(0, len(prompts), cfg.batch):
                m1, m2, m3, iou, stab = mod.get_mask_fast(
                    model, feats, pts_gpu, prompts[b0:b0 + cfg.batch], calc_stability=True
                )
                masks = torch.stack([m1, m2, m3], dim=-1)
                total = masks.sum(0).float()
                inside = (masks & sel_gpu[:, None, None]).sum(0).float()
                frac = inside / n_in
                contained = inside / total.clamp(min=1)
                ok = ((frac >= cfg.min_frac) & (frac <= cfg.max_frac) & (contained >= cfg.contain)
                      & (iou >= cfg.min_iou) & (stab >= cfg.stability) & (total / n_pts <= cfg.max_global))
                for j, m in torch.nonzero(ok).tolist():
                    cands.append((float(iou[j, m] * stab[j, m]), masks[:, j, m][sel_gpu].cpu().numpy(), b0 + j))

        if len(cands) > 1:
            stack = torch.from_numpy(np.stack([c[1] for c in cands]).astype(np.float32)).cuda()
            inter = stack @ stack.T
            size = stack.sum(1)
            ious = (inter / (size[:, None] + size[None, :] - inter).clamp(min=1)).cpu().numpy()
            prompt_id = np.array([c[2] for c in cands])
            support = [1 + int(((ious[i] >= cfg.support_iou) & (prompt_id != prompt_id[i])).sum())
                       for i in range(len(cands))]
        else:
            support = [1] * len(cands)
        cands = [(c[0] * s, c[1]) for c, s in zip(cands, support) if s >= cfg.min_support]
        cands.sort(key=lambda c: -c[0])

        accepted, covered = [], np.zeros(n_in, dtype=bool)
        for _score, mask in cands:
            size = mask.sum()
            if size == 0 or (mask & covered).sum() / size > cfg.nms_overlap:
                continue
            accepted.append(mask)
            covered |= mask
            if len(accepted) >= cfg.max_new:
                break
        if not accepted:
            continue

        sub_labels = np.full(n_in, label, dtype=np.int64)
        taken = np.zeros(n_in, dtype=bool)
        for mask in accepted:
            fresh = mask & ~taken
            sub_labels[fresh] = next_label
            taken |= fresh
            next_label += 1
        point_labels[idx] = sub_labels
        parents.append(int(label))
        report[int(label)] = len(accepted)

    if parents:
        _assign_points_to_faces(face_ids, new_ids, parents, point_parent, point_labels, face_idx, graph,
                                width=next_label + 1)
    return new_ids, report


def _knn_majority(labels, pts, k=16, rounds=2):
    """Spatially smooth point labels so cluster edges lose their comb noise."""
    _, nbr = cKDTree(pts).query(pts, k=k)
    lab = labels.copy()
    n_lab = int(lab.max()) + 1
    for _ in range(rounds):
        votes = np.zeros((len(lab), n_lab), dtype=np.float32)
        for j in range(k):
            np.add.at(votes, (np.arange(len(lab)), lab[nbr[:, j]]), 1.0)
        lab = votes.argmax(1)
    return lab


def _global_features(feats: np.ndarray, pca_dim: int = 16, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import PCA

    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(fn), size=min(len(fn), 20000), replace=False)
    z = PCA(n_components=pca_dim, random_state=seed).fit(fn[sample]).transform(fn)
    return z / (np.linalg.norm(z, axis=1).mean() + 1e-9)


def feature_clusters(pts, z, tau, k_max, min_frac, n_super, validate=None, pos_weight=0.5, seed=0):
    """
    Connectivity-constrained Ward clustering of per-point features z inside one part.

    Splits are accepted top-down where the merge height clearly exceeds the heights inside
    both children (ratio >= tau), both children hold >= min_frac of the points, and the
    optional ``validate(points_a, points_b)`` callback agrees. Returns point labels.
    """
    from scipy.sparse import coo_matrix
    from sklearn.cluster import AgglomerativeClustering, KMeans

    n = len(pts)
    xp = (pts - pts.mean(0)) / (pts.std(0).mean() + 1e-9) * pos_weight * 0.35
    m = int(np.clip(n // 150, 20, n_super))
    sp = KMeans(n_clusters=m, n_init=1, random_state=seed).fit(np.concatenate([z, xp], axis=1)).labels_
    cnt = np.bincount(sp, minlength=m).astype(np.float64)
    sp_feat = np.stack([np.bincount(sp, weights=z[:, j], minlength=m) for j in range(z.shape[1])], 1) / cnt[:, None]
    _, nbr = cKDTree(pts).query(pts, k=8)
    a = np.repeat(sp, 7)
    b = sp[nbr[:, 1:]].ravel()
    keep = a != b
    adj = coo_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(m, m)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.int8)
    ward = AgglomerativeClustering(n_clusters=None, distance_threshold=0, connectivity=adj,
                                   linkage="ward", compute_distances=True).fit(sp_feat)
    children, dist = ward.children_, ward.distances_
    height = np.zeros(2 * m - 1)
    size = np.zeros(2 * m - 1)
    size[:m] = cnt
    members = {i: [i] for i in range(m)}
    for i, (c1, c2) in enumerate(children):
        node = m + i
        height[node] = dist[i]
        size[node] = size[c1] + size[c2]
        members[node] = members[c1] + members[c2]
    min_pts = min_frac * n

    def split_ratio(node):
        if node < m:
            return 0.0
        c1, c2 = children[node - m]
        if size[c1] < min_pts or size[c2] < min_pts:
            return 0.0
        return height[node] / max(height[c1], height[c2], 1e-9)

    clusters, blocked = [2 * m - 2], set()
    while len(clusters) < k_max:
        candidates = [c for c in clusters if c not in blocked]
        if not candidates:
            break
        best = max(candidates, key=split_ratio)
        if split_ratio(best) < tau:
            break
        c1, c2 = children[best - m].tolist()
        if validate is not None:
            ok = validate(np.where(np.isin(sp, members[c1]))[0], np.where(np.isin(sp, members[c2]))[0])
            if not ok:
                blocked.add(best)
                continue
        clusters.remove(best)
        clusters.extend([c1, c2])
    sp_label = np.zeros(m, dtype=np.int64)
    for ci, node in enumerate(clusters):
        sp_label[members[node]] = ci
    return sp_label[sp]


def _make_validator(model, feats_gpu, pts, idx, iou_min):
    """A split is real only if the mask decoder, prompted inside each child, returns that child."""
    mod = p3sam_module()
    pts_gpu = torch.from_numpy(pts).float().cuda()
    n_in = len(idx)
    idx_t = torch.from_numpy(idx).cuda()

    def validate(p1, p2):
        prompts = []
        for p in (p1, p2):
            part_pts = pts[idx[p]]
            prompts.append(part_pts[np.argmin(np.linalg.norm(part_pts - part_pts.mean(0), axis=1))])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            m1, m2, m3, _ = mod.get_mask_fast(model, feats_gpu, pts_gpu, np.stack(prompts))
            masks = torch.stack([m1, m2, m3], dim=-1)[idx_t]
        for j, p in enumerate((p1, p2)):
            child = torch.zeros(n_in, dtype=torch.bool, device=masks.device)
            child[torch.from_numpy(p).cuda()] = True
            inter = (masks[:, j, :] & child[:, None]).sum(0).float()
            union = (masks[:, j, :] | child[:, None]).sum(0).float()
            if float((inter / union.clamp(min=1)).max()) < iou_min:
                return False
        return True

    return validate


def split_by_features(model, feats_gpu, feats_np, pts, face_idx, face_ids, graph, face_normals,
                      cfg: FeatureSplitConfig):
    """Returns (new face ids, {parent label: number of clusters})."""
    new_ids = face_ids.copy()
    point_parent = face_ids[face_idx]
    next_label = int(face_ids.max()) + 1
    report: dict[int, int] = {}
    z_all = _global_features(feats_np)

    for label in np.unique(face_ids[face_ids >= 0]):
        area_frac = graph.areas[face_ids == label].sum() / graph.total_area
        if area_frac < cfg.big_frac:
            continue
        if planar_fraction(face_ids, label, face_normals, graph) >= cfg.planar_guard:
            continue
        idx = np.where(point_parent == label)[0]
        if len(idx) < 800:
            continue
        validate = _make_validator(model, feats_gpu, pts, idx, cfg.iou_min)
        lab = feature_clusters(pts[idx], z_all[idx], tau=cfg.tau_big if area_frac >= cfg.huge_area else cfg.tau,
                               k_max=cfg.k_max, min_frac=cfg.min_frac, n_super=cfg.n_super, validate=validate)
        if lab.max() == 0:
            continue
        lab = _knn_majority(lab, pts[idx])
        k = int(lab.max()) + 1
        faces_p = np.where(face_ids == label)[0]
        uk, cnt = np.unique(face_idx[idx].astype(np.int64) * k + lab, return_counts=True)
        uf, ul = uk // k, uk % k
        order = np.lexsort((-cnt, uf))
        uf_s, ul_s = uf[order], ul[order]
        first = np.r_[True, uf_s[1:] != uf_s[:-1]]
        faces_with, lab_with = uf_s[first], ul_s[first]
        rest = np.setdiff1d(faces_p, faces_with)
        local = np.full(len(face_ids), -1, dtype=np.int64)
        local[faces_with] = lab_with
        if len(rest) and len(faces_with):
            _, nn = cKDTree(graph.centroids[faces_with]).query(graph.centroids[rest])
            local[rest] = lab_with[nn]
        for c in range(k):
            new_ids[local == c] = label if c == 0 else next_label
            if c > 0:
                next_label += 1
        report[int(label)] = k
    return new_ids, report
