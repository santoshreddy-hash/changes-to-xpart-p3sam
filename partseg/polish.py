"""
Geometry post-processing for P3-SAM part labels.

Turns noisy per-face labels into clean parts: crack-tolerant component splitting,
fragment merging, crease snapping, a crease-aware graph cut and label diffusion.
All graph operations use a vertex-welded face adjacency, so UV / normal seams in
GLB files do not break connectivity. Face indices are never changed.
"""

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

try:
    import gco

    GCO_AVAILABLE = True
except ImportError:
    GCO_AVAILABLE = False



class MeshGraph:
    """Welded face-adjacency graph with edge length, dihedral angle and face areas."""

    def __init__(self, mesh):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        diag = float(np.linalg.norm(verts.max(0) - verts.min(0))) + 1e-12
        q = np.round(verts / (diag * 1e-7)).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        welded = trimesh.Trimesh(vertices=verts[first], faces=inv[faces], process=False)
        self.n_faces = len(faces)
        self.vw = np.asarray(welded.vertices, dtype=np.float64)
        self.fw = np.asarray(welded.faces, dtype=np.int64)
        tri = self.vw[self.fw]
        e = np.stack([np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1),
                      np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
                      np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)], axis=1)
        self.face_scale = e.max(axis=1)
        self.vert_scale = np.zeros(len(self.vw))
        np.maximum.at(self.vert_scale, self.fw.ravel(), np.repeat(self.face_scale, 3))
        self.areas = np.asarray(mesh.area_faces, dtype=np.float64)
        self.centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
        adj = welded.face_adjacency
        self.adj = adj.astype(np.int64)
        self.angle = np.nan_to_num(np.asarray(welded.face_adjacency_angles, dtype=np.float64))
        ev = welded.face_adjacency_edges
        self.edge_len = np.linalg.norm(
            welded.vertices[ev[:, 0]] - welded.vertices[ev[:, 1]], axis=1
        )
        self.total_area = float(self.areas.sum())
        self.median_len = float(np.median(self.edge_len)) if len(self.edge_len) else 1.0
        self.median_area = float(np.median(self.areas)) + 1e-18
        self.normals = np.asarray(mesh.face_normals, dtype=np.float64)
        self.scale = float(np.sqrt(self.total_area))

    def geodesic_sparse(self, mask=None):
        a, b = self.adj[:, 0], self.adj[:, 1]
        w = self.edge_len + 1e-12
        if mask is not None:
            a, b, w = a[mask], b[mask], w[mask]
        return coo_matrix((w, (a, b)), shape=(self.n_faces, self.n_faces)).tocsr()

    def sparse(self, mask=None):
        a, b = self.adj[:, 0], self.adj[:, 1]
        if mask is not None:
            a, b = a[mask], b[mask]
        return coo_matrix(
            (np.ones(len(a), dtype=np.float32), (a, b)), shape=(self.n_faces, self.n_faces)
        ).tocsr()


def renumber_by_area(face_ids, areas):
    out = np.full_like(face_ids, -1)
    labels = np.unique(face_ids[face_ids >= 0])
    sums = np.array([areas[face_ids == l].sum() for l in labels])
    for new, old in enumerate(labels[np.argsort(-sums)]):
        out[face_ids == old] = new
    return out


def label_components(face_ids, graph, close_cracks=True):
    """Connected patches per label; patches that touch across a crack (within one triangle size) are joined."""
    a, b = graph.adj[:, 0], graph.adj[:, 1]
    same = (face_ids[a] == face_ids[b]) & (face_ids[a] >= 0)
    g = coo_matrix(
        (np.ones(int(same.sum()), dtype=np.float32), (a[same], b[same])),
        shape=(graph.n_faces, graph.n_faces),
    )
    _, comp = connected_components(g, directed=False)
    comp = comp.astype(np.int64)
    comp[face_ids < 0] = -1
    if not close_cracks:
        return comp
    n_comp = int(comp.max()) + 1
    parent = np.arange(n_comp)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(c1, c2):
        r1, r2 = find(c1), find(c2)
        if r1 != r2:
            parent[r1] = r2

    for l in np.unique(face_ids[face_ids >= 0]):
        faces_l = np.where(face_ids == l)[0]
        comps_l = np.unique(comp[faces_l])
        if len(comps_l) < 2:
            continue
        # 1) fragments that meet at a shared vertex (non-manifold edges, pinches) are one part
        corners = np.stack([graph.fw[faces_l].ravel(), np.repeat(comp[faces_l], 3)], axis=1)
        pairs = np.unique(corners, axis=0)  # sorted by vertex, then component
        same_vertex = pairs[1:, 0] == pairs[:-1, 0]
        for c1, c2 in zip(pairs[:-1, 1][same_vertex], pairs[1:, 1][same_vertex]):
            union(c1, c2)
        # 2) fragments separated by a hairline crack: vertices closer than one triangle size
        vids, first = np.unique(pairs[:, 0], return_index=True)
        vcomp = np.zeros(len(graph.vw), dtype=np.int64)
        vcomp[vids] = pairs[first, 1]
        pts = graph.vw[vids]
        tree = cKDTree(pts)
        k = min(8, len(vids))
        dist, nbr = tree.query(pts, k=k)
        sv = graph.vert_scale[vids]
        for j in range(1, k):
            w = nbr[:, j]
            ok = (vcomp[vids] != vcomp[vids[w]]) & (dist[:, j] <= np.maximum(sv, sv[w]))
            for c1, c2 in zip(vcomp[vids[ok]], vcomp[vids[w[ok]]]):
                union(c1, c2)
    roots = np.array([find(c) for c in range(n_comp)])
    out = np.where(comp >= 0, roots[np.clip(comp, 0, n_comp - 1)], -1)
    return out


def split_disconnected(face_ids, graph):
    """Every connected patch of a label becomes its own part (crack-tolerant)."""
    return renumber_by_area(label_components(face_ids, graph), graph.areas)


def merge_small_regions(face_ids, graph, min_area_frac=0.0015, min_faces=12, max_rounds=12, speck_frac=0.0001):
    """Merge fragments into the neighbour sharing the longest boundary."""
    ids = face_ids.copy()
    a, b = graph.adj[:, 0], graph.adj[:, 1]
    for _ in range(max_rounds):
        labels, counts = np.unique(ids, return_counts=True)
        area = np.array([graph.areas[ids == l].sum() for l in labels])
        small = [
            (ar, l)
            for l, c, ar in zip(labels, counts, area)
            if l >= 0 and (ar < min_area_frac * graph.total_area or c < min_faces)
        ]
        if not small or len(labels) <= 1:
            break
        small.sort()
        la, lb = ids[a], ids[b]
        diff = la != lb
        changed = False
        for ar, l in small:
            sel = diff & ((la == l) | (lb == l))
            if not sel.any():
                # isolated body: keep unless it is a speck
                if ar < speck_frac * graph.total_area:
                    others = np.where((ids != l) & (ids >= 0))[0]
                    cen = graph.centroids[ids == l].mean(0)
                    ids[ids == l] = ids[others[np.argmin(np.linalg.norm(graph.centroids[others] - cen, axis=1))]]
                    la, lb = ids[a], ids[b]
                    diff = la != lb
                    changed = True
                continue
            other = np.where(la[sel] == l, lb[sel], la[sel])
            lens = graph.edge_len[sel]
            ok = other >= 0
            if not ok.any():
                continue
            cand = np.unique(other[ok])
            best = cand[np.argmax([lens[ok][other[ok] == c].sum() for c in cand])]
            ids[ids == l] = best
            la, lb = ids[a], ids[b]
            diff = la != lb
            changed = True
        if not changed:
            break
    return renumber_by_area(ids, graph.areas)


def absorb_spiky_islands(face_ids, graph, max_area_frac=0.02, enclosure=0.8, min_compactness=1.7):
    """Absorb small regions that sit inside one neighbour and have a ragged, spiky outline."""
    ids = face_ids.copy()
    a, b = graph.adj[:, 0], graph.adj[:, 1]
    for _ in range(4):
        la, lb = ids[a], ids[b]
        diff = (la != lb) & (la >= 0) & (lb >= 0)
        changed = False
        for l in np.unique(ids[ids >= 0]):
            m = ids == l
            area = graph.areas[m].sum()
            if area > max_area_frac * graph.total_area:
                continue
            sel = diff & ((la == l) | (lb == l))
            if not sel.any():
                continue
            other = np.where(la[sel] == l, lb[sel], la[sel])
            lens = graph.edge_len[sel]
            cand = np.unique(other)
            per = np.array([lens[other == c].sum() for c in cand])
            top = cand[per.argmax()]
            perim = per.sum()
            if per.max() / perim < enclosure:
                continue
            if perim / (2.0 * np.sqrt(np.pi * area) + 1e-12) < min_compactness:
                continue
            ids[m] = top
            la, lb = ids[a], ids[b]
            diff = (la != lb) & (la >= 0) & (lb >= 0)
            changed = True
        if not changed:
            break
    return renumber_by_area(ids, graph.areas)


def boundary_stats(face_ids, graph):
    a, b = graph.adj[:, 0], graph.adj[:, 1]
    diff = (face_ids[a] != face_ids[b]) & (face_ids[a] >= 0) & (face_ids[b] >= 0)
    total_len = float(graph.edge_len.sum()) + 1e-12
    blen = float(graph.edge_len[diff].sum())
    if diff.any():
        ang = np.degrees(graph.angle[diff])
        w = graph.edge_len[diff]
        mean_crease = float((ang * w).sum() / w.sum())
        on_crease = float(w[ang > 20.0].sum() / w.sum())
    else:
        mean_crease, on_crease = 0.0, 0.0
    labels, counts = np.unique(face_ids[face_ids >= 0], return_counts=True)
    area = np.array([graph.areas[face_ids == l].sum() for l in labels])
    tiny = int(np.sum(area < 0.001 * graph.total_area))
    return {
        "parts": int(len(labels)),
        "tiny_fragments": tiny,
        "boundary_len_pct": 100.0 * blen / total_len,
        "boundary_mean_crease_deg": mean_crease,
        "boundary_on_crease_pct": 100.0 * on_crease,
    }


def count_disconnected(face_ids, graph):
    comp = label_components(face_ids, graph)
    n = 0
    for l in np.unique(face_ids[face_ids >= 0]):
        if len(np.unique(comp[face_ids == l])) > 1:
            n += 1
    return n


def crease_graphcut_polish(
    face_ids,
    graph,
    radius=0.015,
    band=5,
    sigma=0.55,
    crease_floor=0.05,
    stay_cost=0.12,
    length_weight=1.0,
    n_iter=4,
):
    """
    Re-solve labels in a band around every boundary so cuts snap to creases and
    shorten. Faces outside the band are fixed; faces inside pay a small cost per
    ring of distance from the original boundary for changing label.
    """
    if not GCO_AVAILABLE:
        raise RuntimeError("gco is required for crease_graphcut_polish")
    ids = face_ids.copy()
    a, b = graph.adj[:, 0], graph.adj[:, 1]
    diff = (ids[a] != ids[b]) & (ids[a] >= 0) & (ids[b] >= 0)
    if not diff.any():
        return ids
    seeds = np.unique(np.concatenate([a[diff], b[diff]]))
    rad = max(radius * graph.scale, band * graph.median_len)
    dist = dijkstra(graph.geodesic_sparse(), directed=False, indices=seeds, min_only=True, limit=rad)
    in_band = np.isfinite(dist) & (ids >= 0)
    band_faces = np.where(in_band)[0]
    local = -np.ones(graph.n_faces, dtype=np.int64)
    local[band_faces] = np.arange(len(band_faces))

    labels = np.unique(ids[ids >= 0])
    dense = -np.ones(graph.n_faces, dtype=np.int64)
    for i, l in enumerate(labels):
        dense[ids == l] = i
    L = len(labels)
    nb = len(band_faces)

    w_edge = (graph.edge_len / graph.median_len) * (
        crease_floor + (1.0 - crease_floor) * np.exp(-(graph.angle ** 2) / (sigma ** 2))
    )
    w_edge = w_edge * length_weight

    unary = np.zeros((nb, L), dtype=np.float64)
    d_norm = dist[band_faces] / rad
    area_rel = graph.areas[band_faces] / graph.median_area
    change_cost = stay_cost * (0.2 + 8.0 * d_norm) * area_rel
    unary += change_cost[:, None]
    unary[np.arange(nb), dense[band_faces]] = 0.0

    la, lb = local[a], local[b]
    both = (la >= 0) & (lb >= 0)
    only_a = (la >= 0) & (lb < 0)
    only_b = (lb >= 0) & (la < 0)
    for sel, inner, outer in ((only_a, la, b), (only_b, lb, a)):
        if not sel.any():
            continue
        rows = inner[sel]
        out_lab = dense[outer[sel]]
        w = w_edge[sel]
        unary += np.bincount(rows, weights=w, minlength=nb)[:, None]
        np.add.at(unary, (rows, out_lab), -w)

    edges = np.stack([la[both], lb[both]], axis=1).astype(np.int32)
    # gco silently returns the initial labels for int32 inputs; float64 is required
    ew = (w_edge[both] * 1000.0).astype(np.float64)
    data = (unary * 1000.0).astype(np.float64)
    pairwise = (1.0 - np.eye(L)).astype(np.float64)
    init = np.ascontiguousarray(dense[band_faces], dtype=np.int32)
    res = gco.cut_general_graph(
        edges, ew, data, pairwise, n_iter=n_iter, algorithm="expansion", init_labels=init
    )
    res = np.asarray(res, dtype=np.int64)
    new_ids = ids.copy()
    new_ids[band_faces] = labels[res]
    return new_ids


def planar_fraction(face_ids, label, mesh_normals, graph, tol_deg=12.0):
    """Area fraction of a part whose normals lie within tol_deg of its principal normal axis (sign-free)."""
    m = face_ids == label
    n = mesh_normals[m]
    a = graph.areas[m]
    if a.sum() <= 0:
        return 0.0
    cov = (n * a[:, None]).T @ n
    w, v = np.linalg.eigh(cov)
    axis = v[:, -1]
    cosang = np.abs(n @ axis)
    return float(a[cosang >= np.cos(np.radians(tol_deg))].sum() / a.sum())


def crease_patches(graph, crease_deg=30.0):
    """Connected patches of faces joined by non-crease edges (smooth geometric superfaces)."""
    smooth = graph.angle < np.radians(crease_deg)
    a, b = graph.adj[smooth, 0], graph.adj[smooth, 1]
    g = coo_matrix((np.ones(len(a), dtype=np.float32), (a, b)), shape=(graph.n_faces,) * 2)
    _, comp = connected_components(g, directed=False)
    return comp


def snap_to_patches(face_ids, graph, crease_deg=30.0, max_area_frac=0.08, min_vote=0.70,
                    planar_deg=8.0, planar_min=0.9, planar_vote=0.55):
    """
    Relabel each small smooth patch to its dominant label when that label clearly
    dominates, so part boundaries land on real creases instead of sampling noise.
    Large smooth patches (organic bodies) are left to the smoothing stage.
    """
    ids = face_ids.copy()
    comp = crease_patches(graph, crease_deg)
    n_comp = int(comp.max()) + 1
    valid = ids >= 0
    patch_area = np.bincount(comp, weights=graph.areas, minlength=n_comp)
    labels = np.unique(ids[valid])
    dense = -np.ones(graph.n_faces, dtype=np.int64)
    for i, l in enumerate(labels):
        dense[ids == l] = i
    L = len(labels)
    votes = np.zeros((n_comp, L), dtype=np.float64)
    np.add.at(votes, (comp[valid], dense[valid]), graph.areas[valid])
    top = votes.argmax(axis=1)
    share = votes[np.arange(n_comp), top] / np.maximum(votes.sum(axis=1), 1e-12)
    dom = np.zeros((n_comp, 3))
    np.add.at(dom, comp, graph.normals * graph.areas[:, None])
    dom /= np.maximum(np.linalg.norm(dom, axis=1, keepdims=True), 1e-12)
    flat = (np.einsum("ij,ij->i", graph.normals, dom[comp]) >= np.cos(np.radians(planar_deg))) * graph.areas
    planar_frac = np.bincount(comp, weights=flat, minlength=n_comp) / np.maximum(patch_area, 1e-18)
    small = (patch_area <= max_area_frac * graph.total_area) & (share >= min_vote)
    flat_panel = (planar_frac >= planar_min) & (share >= planar_vote)
    eligible = small | flat_panel
    flip = eligible[comp] & valid & (dense != top[comp])
    ids[flip] = labels[top[comp[flip]]]
    return ids


def snap_spill_to_creases(face_ids, graph, crease_deg=30.0, radius=0.015, min_vote=0.6):
    """
    Labels often spill a little across a crease because of sampling noise. For each
    smooth patch with a clear majority label, faces near its border that carry the
    label of a neighbouring patch are returned to the patch's own label.
    """
    ids = face_ids.copy()
    comp = crease_patches(graph, crease_deg)
    n_comp = int(comp.max()) + 1
    valid = ids >= 0
    labels = np.unique(ids[valid])
    dense = -np.ones(graph.n_faces, dtype=np.int64)
    for i, l in enumerate(labels):
        dense[ids == l] = i
    L = len(labels)
    votes = np.zeros((n_comp, L), dtype=np.float64)
    np.add.at(votes, (comp[valid], dense[valid]), graph.areas[valid])
    top = votes.argmax(axis=1)
    share = votes[np.arange(n_comp), top] / np.maximum(votes.sum(axis=1), 1e-12)
    a, b = graph.adj[:, 0], graph.adj[:, 1]
    cross = comp[a] != comp[b]
    if not cross.any():
        return ids
    allowed = set()
    pa, pb = comp[a[cross]], comp[b[cross]]
    keys = np.unique(np.concatenate([pa * L + top[pb], pb * L + top[pa]]))
    seeds = np.unique(np.concatenate([a[cross], b[cross]]))
    smooth = ~cross
    dist = dijkstra(graph.geodesic_sparse(smooth), directed=False, indices=seeds, min_only=True,
                    limit=radius * graph.scale)
    near = np.isfinite(dist) & valid
    own = top[comp]
    cand = near & (dense != own) & (share[comp] >= min_vote)
    fkey = comp * L + np.clip(dense, 0, L - 1)
    flip = cand & np.isin(fkey, keys)
    ids[flip] = labels[own[flip]]
    return ids


def smooth_boundaries(face_ids, graph, band=5, hops=4, rounds=2, sigma=0.55, floor=0.02,
                      margin=0.04, thin_frac=0.6):
    """
    Crease-aware label diffusion: one-hot labels are diffused over the mesh with
    weights that vanish across sharp creases, then boundary faces adopt the
    dominant label. Smooths organic boundaries while keeping hard edges; thin
    parts (mostly within the boundary band) are frozen so they cannot erode.
    """
    ids = face_ids.copy()
    a, b = graph.adj[:, 0], graph.adj[:, 1]
    w = floor + (1.0 - floor) * np.exp(-(graph.angle ** 2) / (sigma ** 2))
    W = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(graph.n_faces,) * 2).tocsr()
    eye = coo_matrix((np.ones(graph.n_faces), (np.arange(graph.n_faces),) * 2)).tocsr()
    S = (eye + W).tocsr()
    S = coo_matrix(S.multiply(1.0 / np.asarray(S.sum(axis=1)).reshape(-1, 1))).tocsr().astype(np.float32)
    for _ in range(rounds):
        diff = (ids[a] != ids[b]) & (ids[a] >= 0) & (ids[b] >= 0)
        if not diff.any():
            break
        seeds = np.unique(np.concatenate([a[diff], b[diff]]))
        dist = dijkstra(graph.sparse(), directed=False, indices=seeds, min_only=True, limit=band + 0.5)
        in_band = np.isfinite(dist) & (ids >= 0)
        labels = np.unique(ids[ids >= 0])
        dense = -np.ones(graph.n_faces, dtype=np.int64)
        for i, l in enumerate(labels):
            dense[ids == l] = i
        L = len(labels)
        thin = np.zeros(L, dtype=bool)
        for i in range(L):
            m = dense == i
            thin[i] = in_band[m].mean() >= thin_frac
        X = np.zeros((graph.n_faces, L), dtype=np.float32)
        valid = dense >= 0
        X[np.where(valid)[0], dense[valid]] = 1.0
        for _h in range(hops):
            X = S @ X
        X[:, thin] = -1.0
        cand = X.argmax(axis=1)
        own = X[np.arange(graph.n_faces), np.clip(dense, 0, L - 1)]
        best = X[np.arange(graph.n_faces), cand]
        switch = in_band & valid & (cand != dense) & (~thin[np.clip(dense, 0, L - 1)]) & (best > own + margin)
        if not switch.any():
            break
        ids[switch] = labels[cand[switch]]
    return ids


def smoothing_scale(graph, radius=0.02):
    """Boundary-smoothing radius in face rings, scaled to the mesh's physical size."""
    sigma = float(np.clip(radius * np.sqrt(graph.total_area) / graph.median_len, 1.0, 12.0))
    hops = int(np.clip(1.5 * sigma ** 2, 3, 60))
    band = int(np.clip(round(2.5 * sigma) + 2, 4, 14))
    return sigma, hops, band


def polish_labels(face_ids, graph, cfg, log=None):
    """
    Full clean-up chain applied to raw part labels (cfg is a PolishConfig).
    Returns labels renumbered so that label 0 is the largest part.
    """
    min_faces = max(12, int(graph.n_faces * 0.0002))
    _, hops, band = smoothing_scale(graph, cfg.smooth_radius)
    ids = split_disconnected(face_ids, graph)
    ids = merge_small_regions(ids, graph, min_area_frac=cfg.min_area_frac, min_faces=min_faces,
                              speck_frac=cfg.speck_frac)
    ids = absorb_spiky_islands(ids, graph)

    # large flat panels (table tops, back boards) may gain faces in their interior never lose them;
    # their boundary band stays free so edges can still straighten
    protected = np.zeros(graph.n_faces, dtype=bool)
    for label in np.unique(ids[ids >= 0]):
        m = ids == label
        if (graph.areas[m].sum() >= cfg.panel_area * graph.total_area
                and planar_fraction(ids, label, graph.normals, graph) >= cfg.panel_planar):
            protected |= m
    a_, b_ = graph.adj[:, 0], graph.adj[:, 1]
    edge_diff = (ids[a_] != ids[b_]) & (ids[a_] >= 0) & (ids[b_] >= 0)
    if protected.any() and edge_diff.any():
        seeds = np.unique(np.concatenate([a_[edge_diff], b_[edge_diff]]))
        dist = dijkstra(graph.geodesic_sparse(), directed=False, indices=seeds, min_only=True,
                        limit=cfg.panel_margin * graph.scale)
        protected &= ~np.isfinite(dist)
    anchor = ids.copy()

    def keep_panels(x):
        x = x.copy()
        lost = protected & (x != anchor)
        x[lost] = anchor[lost]
        return x

    ids = keep_panels(snap_to_patches(ids, graph))
    ids = keep_panels(snap_spill_to_creases(ids, graph))
    if GCO_AVAILABLE:
        ids = keep_panels(crease_graphcut_polish(
            ids, graph, band=band, sigma=cfg.crease_sigma, crease_floor=cfg.crease_floor,
            stay_cost=cfg.stay_cost, length_weight=cfg.length_weight))
    elif log is not None:
        log.warning("gco is not installed: skipping the crease-aware graph cut (boundaries will be rougher)")
    ids = keep_panels(smooth_boundaries(ids, graph, band=band, hops=hops, sigma=cfg.crease_sigma))
    ids = split_disconnected(ids, graph)
    ids = merge_small_regions(ids, graph, min_area_frac=cfg.min_area_frac, min_faces=min_faces,
                              speck_frac=cfg.speck_frac)
    ids = absorb_spiky_islands(ids, graph)
    return renumber_by_area(ids, graph.areas)
