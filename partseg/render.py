"""CPU triangle rasterizer (painter's algorithm, flat shading, 2x supersampling) with part outlines."""

import colorsys

import cv2
import numpy as np


def part_palette(n):
    base = [
        (0.89, 0.29, 0.25), (0.20, 0.55, 0.85), (0.30, 0.72, 0.38), (0.96, 0.69, 0.18),
        (0.60, 0.40, 0.80), (0.95, 0.50, 0.15), (0.20, 0.75, 0.75), (0.85, 0.35, 0.65),
        (0.55, 0.65, 0.25), (0.45, 0.45, 0.85), (0.80, 0.55, 0.40), (0.35, 0.60, 0.55),
    ]
    out = list(base[:n])
    i = 0
    while len(out) < n:
        h = (i * 0.61803398875) % 1.0
        s = 0.55 + 0.35 * ((i * 7) % 5) / 4.0
        v = 0.70 + 0.28 * ((i * 3) % 4) / 3.0
        out.append(colorsys.hsv_to_rgb(h, s, v))
        i += 1
    return np.array(out, dtype=np.float32)


def _view_matrix(azim, elev, y_up):
    ra, re = np.radians(azim), np.radians(elev)
    ca, sa, ce, se = np.cos(ra), np.sin(ra), np.cos(re), np.sin(re)
    if y_up:
        r_az = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)
    else:
        r_az = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
    r_pitch = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=np.float64)
    return r_pitch @ r_az


def render_parts(mesh, face_ids, size=900, azim=35, elev=18, y_up=True, outline=True,
                 palette=None, bg=(18, 19, 22), ss=2):
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    center = (verts.max(0) + verts.min(0)) / 2.0
    R = _view_matrix(azim, elev, y_up)
    pc = (verts - center) @ R.T
    scale = np.abs(pc[:, :2]).max() + 1e-9
    pad = 0.10
    W = size * ss
    sx = (pc[:, 0] / scale / (1 + pad) * 0.5 + 0.5) * (W - 1)
    sy = (0.5 - pc[:, 1] / scale / (1 + pad) * 0.5) * (W - 1)
    fx = np.stack([sx[faces], sy[faces]], axis=-1)
    depth = pc[faces, 2].mean(axis=1)

    normals = np.asarray(mesh.face_normals, dtype=np.float64) @ R.T
    key = np.array([0.35, 0.55, 0.75])
    key /= np.linalg.norm(key)
    fill = np.array([-0.6, -0.2, 0.5])
    fill /= np.linalg.norm(fill)
    lam = np.clip(np.abs(normals @ key), 0, 1) * 0.62 + np.clip(np.abs(normals @ fill), 0, 1) * 0.18
    shade = np.clip(0.34 + lam, 0.0, 1.05)

    ids = np.asarray(face_ids)
    labels = np.unique(ids[ids >= 0])
    pal = palette if palette is not None else part_palette(len(labels))
    lut = {l: pal[i % len(pal)] for i, l in enumerate(labels)}
    base = np.full((len(faces), 3), 0.5, dtype=np.float32)
    pid = np.zeros(len(faces), dtype=np.int64)
    for i, l in enumerate(labels):
        m = ids == l
        base[m] = lut[l]
        pid[m] = i + 1
    rgb = np.clip(base * shade[:, None], 0, 1) * 255.0
    rgb = rgb[:, ::-1].astype(np.uint8)  # cv2 uses BGR

    order = np.argsort(depth)
    img = np.empty((W, W, 3), dtype=np.uint8)
    img[:] = np.array(bg[::-1], dtype=np.uint8)
    idimg = np.zeros((W, W, 3), dtype=np.uint8)
    tri = np.round(fx * 16.0).astype(np.int32)
    idc = np.stack([pid & 255, (pid >> 8) & 255, np.zeros_like(pid)], axis=1).astype(np.uint8)
    cols = [tuple(int(v) for v in c) for c in rgb]
    idcols = [tuple(int(v) for v in c) for c in idc]
    fill_poly = cv2.fillConvexPoly
    for i in order:
        t = tri[i]
        fill_poly(img, t, cols[i], lineType=cv2.LINE_8, shift=4)
        if outline:
            fill_poly(idimg, t, idcols[i], lineType=cv2.LINE_8, shift=4)

    if outline:
        code = idimg[:, :, 0].astype(np.int32) | (idimg[:, :, 1].astype(np.int32) << 8)
        edge = np.zeros(code.shape, dtype=bool)
        dx = (code[:, 1:] != code[:, :-1]) & (code[:, 1:] > 0) & (code[:, :-1] > 0)
        dy = (code[1:, :] != code[:-1, :]) & (code[1:, :] > 0) & (code[:-1, :] > 0)
        edge[:, 1:] |= dx
        edge[1:, :] |= dy
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
        img[edge] = (img[edge] * 0.18).astype(np.uint8)

    if ss > 1:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img[:, :, ::-1].copy()
