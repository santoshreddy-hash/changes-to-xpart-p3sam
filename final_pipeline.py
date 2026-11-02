"""Final part-segmentation pipeline: optimized P3-SAM -> large-part re-prompting -> clean-boundary polish."""

import os
import sys
import time

import fpsample
import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "P3-SAM", "demo")
if DEMO not in sys.path:
    sys.path.insert(0, DEMO)

import auto_mask_fast as mod  # noqa: E402
import part_polish as pp  # noqa: E402


def sampling_params(n_faces, prompt_scale=1.0):
    n_pts = min(100000, max(15000, n_faces * 2))
    n_prompts = min(400, int(min(300, max(60, int(np.sqrt(n_faces) * 2.5))) * prompt_scale))
    return n_pts, n_prompts


def run_baseline(auto_mask, mesh, n_pts, n_prompts, seed=42):
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, fids, res = mod.mesh_sam(
            [auto_mask.model, auto_mask.model], mesh.copy(), save_path=None,
            point_num=n_pts, prompt_num=n_prompts, threshold=0.95, post_process=False,
            show_info=False, save_mid_res=False, clean_mesh_flag=False, seed=seed,
            prompt_bs=8, loop_mode="opt3", nms_mode="fast", cleanup_mode="fast",
            multimask_output=False, stability_filter=False, nms_thresh=0.9,
            refine_mode="scipy", preserve_raw_faces=True, second_pass_enabled=False,
        )
    return np.asarray(fids).copy(), res, time.perf_counter() - t0


def run_enhanced(auto_mask, mesh, n_pts, n_prompts, seed=42, extra_prompts=None):
    t0 = time.perf_counter()
    extra = extra_prompts if extra_prompts is not None else min(60, n_prompts // 4)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, fids, res = mod.mesh_sam(
            [auto_mask.model, auto_mask.model], mesh.copy(), save_path=None,
            point_num=n_pts, prompt_num=n_prompts, threshold=0.95, post_process=False,
            show_info=False, save_mid_res=False, clean_mesh_flag=False, seed=seed,
            prompt_bs=8, loop_mode="opt3", nms_mode="fast", cleanup_mode="fast",
            multimask_output=True, stability_filter=True, nms_thresh=0.7,
            refine_mode="graphcut", preserve_raw_faces=True, second_pass_enabled=True,
            max_additional_prompts=extra, conf_threshold=0.60, curvature_threshold=0.25,
            min_region_size=10,
        )
    return np.asarray(fids).copy(), res, time.perf_counter() - t0


def subdivide_large_parts(model_parallel, feats, pts, face_idx, face_ids, graph,
                          big_frac=0.08, n_prompts=48, bs=8, min_frac=0.04, max_frac=0.70,
                          min_iou=0.20, stab_thresh=0.80, contain=0.30, nms_overlap=0.45,
                          max_new=8, max_global=0.55, verbose=False, planar_guard=0.85, face_normals=None,
                          min_support=2, support_iou=0.6, n_starts=3):
    """Re-prompt inside every large part and split it along confident sub-masks."""
    n_pts = len(pts)
    pts_gpu = torch.from_numpy(pts).float().cuda()
    if face_normals is None:
        raise ValueError("face_normals required for the planar guard")
    pl = face_ids[face_idx]
    new_ids = face_ids.copy()
    next_label = int(face_ids.max()) + 1
    pl_new = pl.copy()
    parents = []
    report = {}
    for l in np.unique(face_ids[face_ids >= 0]):
        if graph.areas[face_ids == l].sum() / graph.total_area < big_frac:
            continue
        sel = pl == l
        idx = np.where(sel)[0]
        n_in = len(idx)
        if n_in < 600:
            continue
        if pp.planar_fraction(face_ids, l, face_normals, graph) >= planar_guard:
            if verbose:
                print(f'  part {l}: planar panel, skipped', flush=True)
            continue
        k = int(min(n_prompts, n_in // 30))
        sub = pts[idx]
        picked = np.unique(np.concatenate([
            fpsample.fps_sampling(sub, k, start_idx=int(s * n_in // n_starts)) for s in range(n_starts)
        ]))
        prompts = sub[picked]
        k = len(prompts)
        sel_gpu = torch.from_numpy(sel).cuda()
        cands = []
        dbg_counts = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b0 in range(0, k, bs):
                m1, m2, m3, iou, stab = mod.get_mask_fast(
                    model_parallel, feats, pts_gpu, prompts[b0:b0 + bs], calc_stability=True
                )
                masks = torch.stack([m1, m2, m3], dim=-1)
                tot = masks.sum(0).float()
                inside = (masks & sel_gpu[:, None, None]).sum(0).float()
                frac = inside / n_in
                cont = inside / tot.clamp(min=1)
                ok = ((frac >= min_frac) & (frac <= max_frac) & (cont >= contain)
                      & (iou >= min_iou) & (stab >= stab_thresh)
                      & (tot / n_pts <= max_global))
                dbg_counts.append(int(ok.sum()))
                for j, m in torch.nonzero(ok).tolist():
                    cands.append((float(iou[j, m] * stab[j, m]), masks[:, j, m][sel_gpu].cpu().numpy(), b0 + j))
        if verbose:
            print(f'  part {l}: area {graph.areas[face_ids == l].sum() / graph.total_area:.2f}, pts {n_in}, ok per batch {dbg_counts}, cands {len(cands)}', flush=True)
        if len(cands) > 1:
            M = torch.from_numpy(np.stack([c[1] for c in cands]).astype(np.float32)).cuda()
            inter = M @ M.T
            sz = M.sum(1)
            ious = (inter / (sz[:, None] + sz[None, :] - inter).clamp(min=1)).cpu().numpy()
            pid = np.array([c[2] for c in cands])
            support = [1 + int(((ious[i] >= support_iou) & (pid != pid[i])).sum()) for i in range(len(cands))]
        else:
            support = [1] * len(cands)
        cands = [(c[0] * s, c[1]) for c, s in zip(cands, support) if s >= min_support]
        if verbose:
            print(f'    supported candidates: {len(cands)}', flush=True)
        cands.sort(key=lambda c: -c[0])
        accepted = []
        covered = np.zeros(n_in, dtype=bool)
        for score, m in cands:
            size = m.sum()
            if size == 0 or (m & covered).sum() / size > nms_overlap:
                continue
            accepted.append(m)
            covered |= m
            if len(accepted) >= max_new:
                break
        if not accepted:
            continue
        if not accepted:
            continue
        sub_lab = np.full(n_in, l, dtype=np.int64)
        taken = np.zeros(n_in, dtype=bool)
        for m in accepted:
            fresh = m & ~taken
            sub_lab[fresh] = next_label
            taken |= fresh
            next_label += 1
        pl_new[idx] = sub_lab
        parents.append(l)
        report[int(l)] = len(accepted)
    if not parents:
        return new_ids, report

    width = next_label + 1
    for l in parents:
        faces_p = np.where(face_ids == l)[0]
        pts_sel = (pl == l)
        f_pts = face_idx[pts_sel].astype(np.int64)
        lab_pts = pl_new[pts_sel]
        uk, cnt = np.unique(f_pts * width + lab_pts, return_counts=True)
        uf, ulab = uk // width, uk % width
        order = np.lexsort((-cnt, uf))
        uf_s, ulab_s = uf[order], ulab[order]
        first = np.r_[True, uf_s[1:] != uf_s[:-1]]
        faces_with, lab_with = uf_s[first], ulab_s[first]
        new_ids[faces_with] = lab_with
        rest = np.setdiff1d(faces_p, faces_with, assume_unique=False)
        if len(rest) and len(faces_with):
            tree = cKDTree(graph.centroids[faces_with])
            _, nn = tree.query(graph.centroids[rest])
            new_ids[rest] = lab_with[nn]
    return new_ids, report


def _knn_majority(labels, pts, k=16, rounds=2):
    """Spatially smooth point labels so cluster edges lose their comb noise."""
    tree = cKDTree(pts)
    _, nbr = tree.query(pts, k=k)
    lab = labels.copy()
    n_lab = int(lab.max()) + 1
    for _ in range(rounds):
        votes = np.zeros((len(lab), n_lab), dtype=np.float32)
        for j in range(k):
            np.add.at(votes, (np.arange(len(lab)), lab[nbr[:, j]]), 1.0)
        lab = votes.argmax(1)
    return lab


def feature_clusters(feats, pts, n_super=300, k_max=8, tau=1.6, min_frac=0.03, pos_weight=0.5,
                     pca_dim=16, seed=0, z=None, sep_min=0.0, validate=None):
    """
    Connectivity-constrained Ward clustering of P3-SAM point features inside one part.
    Splits are accepted top-down only where the merge height clearly exceeds the heights
    inside both children (ratio >= tau) and both children hold >= min_frac of the points.
    Returns (labels per point, list of accepted split ratios).
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from scipy.sparse import coo_matrix

    n = len(pts)
    if z is None:
        f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        rng = np.random.default_rng(seed)
        sub = rng.choice(n, size=min(n, 20000), replace=False)
        pca = PCA(n_components=pca_dim, random_state=seed).fit(f[sub])
        xf = pca.transform(f)
        xf /= np.linalg.norm(xf, axis=1).mean() + 1e-9
    else:
        xf = z
    xp = (pts - pts.mean(0)) / (pts.std(0).mean() + 1e-9) * pos_weight * 0.35
    m = int(np.clip(n // 150, 20, n_super))
    km = KMeans(n_clusters=m, n_init=1, random_state=seed).fit(np.concatenate([xf, xp], axis=1))
    sp = km.labels_
    cnt = np.bincount(sp, minlength=m).astype(np.float64)
    sp_feat = np.stack([np.bincount(sp, weights=xf[:, j], minlength=m) for j in range(xf.shape[1])], 1) / cnt[:, None]
    tree = cKDTree(pts)
    _, nbr = tree.query(pts, k=8)
    a = np.repeat(sp, 7)
    b = sp[nbr[:, 1:]].ravel()
    keep = a != b
    adj = coo_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(m, m)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.int8)
    ag = AgglomerativeClustering(n_clusters=None, distance_threshold=0, connectivity=adj,
                                 linkage="ward", compute_distances=True).fit(sp_feat)
    children, dist = ag.children_, ag.distances_
    height = np.zeros(2 * m - 1)
    size = np.zeros(2 * m - 1)
    size[:m] = cnt
    members = {i: [i] for i in range(m)}
    fsum = np.zeros((2 * m - 1, xf.shape[1]))
    fsum[:m] = sp_feat * cnt[:, None]
    for i, (c1, c2) in enumerate(children):
        fsum[m + i] = fsum[c1] + fsum[c2]
    for i, (c1, c2) in enumerate(children):
        node = m + i
        height[node] = dist[i]
        size[node] = size[c1] + size[c2]
        members[node] = members[c1] + members[c2]
    root = 2 * m - 2
    clusters = [root]
    ratios = []
    min_pts = min_frac * n

    def child_sep(node):
        c1, c2 = children[node - m]
        return float(np.linalg.norm(fsum[c1] / size[c1] - fsum[c2] / size[c2]))

    def split_ratio(node):
        if node < m:
            return 0.0
        c1, c2 = children[node - m]
        if size[c1] < min_pts or size[c2] < min_pts:
            return 0.0
        if child_sep(node) < sep_min:
            return 0.0
        return height[node] / max(height[c1], height[c2], 1e-9)

    blocked = set()
    while len(clusters) < k_max:
        cand = [c for c in clusters if c not in blocked]
        if not cand:
            break
        best = max(cand, key=split_ratio)
        r = split_ratio(best)
        if r < tau:
            break
        c1, c2 = children[best - m].tolist()
        info = None
        if validate is not None:
            p1 = np.where(np.isin(sp, members[c1]))[0]
            p2 = np.where(np.isin(sp, members[c2]))[0]
            ok, info = validate(p1, p2)
            if not ok:
                blocked.add(best)
                ratios.append(("rejected", round(float(r), 2), info))
                continue
        clusters.remove(best)
        clusters.extend([c1, c2])
        ratios.append((round(float(r), 2), round(child_sep(best), 3)) if info is None else (round(float(r), 2), info))
    sp_label = np.zeros(m, dtype=np.int64)
    for ci, node in enumerate(clusters):
        sp_label[members[node]] = ci
    return sp_label[sp], ratios


def _make_validator(model, feats_gpu, pts, idx, iou_min):
    """A split is real only if the mask decoder, prompted inside each child, returns that child."""
    pts_gpu = torch.from_numpy(pts).float().cuda()
    n_in = len(idx)
    idx_t = torch.from_numpy(idx).cuda()

    def validate(p1, p2):
        prompts = []
        for p in (p1, p2):
            c = pts[idx[p]].mean(0)
            prompts.append(pts[idx[p]][np.argmin(np.linalg.norm(pts[idx[p]] - c, axis=1))])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            m1, m2, m3, _ = mod.get_mask_fast(model, feats_gpu, pts_gpu, np.stack(prompts))
            masks = torch.stack([m1, m2, m3], dim=-1)[idx_t]  # [n_in, 2, 3]
        out = []
        for j, p in enumerate((p1, p2)):
            child = torch.zeros(n_in, dtype=torch.bool, device=masks.device)
            child[torch.from_numpy(p).cuda()] = True
            inter = (masks[:, j, :] & child[:, None]).sum(0).float()
            union = (masks[:, j, :] | child[:, None]).sum(0).float()
            out.append(float((inter / union.clamp(min=1)).max()))
        return min(out) >= iou_min, tuple(round(v, 2) for v in out)

    return validate


def subdivide_by_features(feats, pts, face_idx, face_ids, graph, face_normals, big_frac=0.06,
                          planar_guard=0.85, verbose=False, tau=1.3, tau_big=1.15, huge_area=0.25, sep_rel=0.0,
                          model=None, feats_gpu=None, iou_min=0.30, **kw):
    """Split large non-planar parts along clusters of P3-SAM's own point features."""
    from sklearn.decomposition import PCA

    new_ids = face_ids.copy()
    pl = face_ids[face_idx]
    next_label = int(face_ids.max()) + 1
    report = {}
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(0)
    gsub = rng.choice(len(fn), size=min(len(fn), 20000), replace=False)
    z_all = PCA(n_components=16, random_state=0).fit(fn[gsub]).transform(fn)
    z_all /= np.linalg.norm(z_all, axis=1).mean() + 1e-9
    labs = np.unique(pl[pl >= 0])
    cents = np.stack([z_all[pl == l].mean(0) for l in labs])
    dd = np.linalg.norm(cents[:, None] - cents[None], axis=2)[np.triu_indices(len(labs), 1)]
    d_ref = float(np.percentile(dd, 25)) if len(dd) else 0.0
    for l in np.unique(face_ids[face_ids >= 0]):
        area_frac = graph.areas[face_ids == l].sum() / graph.total_area
        if area_frac < big_frac:
            continue
        if pp.planar_fraction(face_ids, l, face_normals, graph) >= planar_guard:
            continue
        idx = np.where(pl == l)[0]
        if len(idx) < 800:
            continue
        validate = None
        if model is not None:
            validate = _make_validator(model, feats_gpu, pts, idx, iou_min)
        lab, ratios = feature_clusters(feats[idx], pts[idx], tau=tau_big if area_frac >= huge_area else tau,
                                       z=z_all[idx], sep_min=sep_rel * d_ref, validate=validate, **kw)
        if lab.max() == 0:
            if verbose:
                print(f"  part {l}: no clear structure", flush=True)
            continue
        lab = _knn_majority(lab, pts[idx])
        k = int(lab.max()) + 1
        faces_p = np.where(face_ids == l)[0]
        f_pts = face_idx[idx].astype(np.int64)
        uk, cnt = np.unique(f_pts * k + lab, return_counts=True)
        uf, ul = uk // k, uk % k
        order = np.lexsort((-cnt, uf))
        uf_s, ul_s = uf[order], ul[order]
        first = np.r_[True, uf_s[1:] != uf_s[:-1]]
        faces_with, lab_with = uf_s[first], ul_s[first]
        rest = np.setdiff1d(faces_p, faces_with)
        out = np.full(len(face_ids), -1, dtype=np.int64)
        out[faces_with] = lab_with
        if len(rest) and len(faces_with):
            _, nn = cKDTree(graph.centroids[faces_with]).query(graph.centroids[rest])
            out[rest] = lab_with[nn]
        for c in range(k):
            sel = out == c
            new_ids[sel] = l if c == 0 else next_label
            if c > 0:
                next_label += 1
        report[int(l)] = (k, ratios)
        if verbose:
            print(f"  part {l} (area {area_frac:.2f}): {k} clusters, (ratio, sep) {ratios}, d_ref {d_ref:.3f}", flush=True)
    return new_ids, report


def smoothing_scale(graph):
    """Boundary-smoothing radius in face rings, scaled to the mesh's physical size."""
    sigma = float(np.clip(0.02 * np.sqrt(graph.total_area) / graph.median_len, 1.0, 12.0))
    hops = int(np.clip(1.5 * sigma ** 2, 3, 60))
    band = int(np.clip(round(2.5 * sigma) + 2, 4, 14))
    return sigma, hops, band


def polish(face_ids, graph, n_pts, min_area_frac=0.003, stay_cost=0.06, sigma=0.55,
           panel_area=0.04, panel_planar=0.9):
    min_faces = max(12, int(graph.n_faces * 0.0002))
    _, hops, band = smoothing_scale(graph)
    ids = pp.split_disconnected(face_ids, graph)
    ids = pp.merge_small_regions(ids, graph, min_area_frac=min_area_frac, min_faces=min_faces)
    ids = pp.absorb_spiky_islands(ids, graph)

    # large flat panels (table tops, back boards) may gain faces but never lose them
    protected = np.zeros(graph.n_faces, dtype=bool)
    for l in np.unique(ids[ids >= 0]):
        m = ids == l
        if graph.areas[m].sum() >= panel_area * graph.total_area and                 pp.planar_fraction(ids, l, graph.normals, graph) >= panel_planar:
            protected |= m
    anchor = ids.copy()

    def keep_panels(x):
        x = x.copy()
        lost = protected & (x != anchor)
        x[lost] = anchor[lost]
        return x

    ids = keep_panels(pp.snap_to_patches(ids, graph))
    ids = keep_panels(pp.snap_spill_to_creases(ids, graph))
    ids = keep_panels(pp.crease_graphcut_polish(ids, graph, band=band, sigma=sigma, stay_cost=stay_cost))
    ids = keep_panels(pp.smooth_boundaries(ids, graph, band=band, hops=hops, sigma=sigma))
    ids = pp.split_disconnected(ids, graph)
    ids = pp.merge_small_regions(ids, graph, min_area_frac=min_area_frac, min_faces=min_faces)
    ids = pp.absorb_spiky_islands(ids, graph)
    return pp.renumber_by_area(ids, graph.areas)
