import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from skimage import measure
from typing import Callable, Tuple, List, Union
from torch import nn
from tqdm import tqdm
from einops import repeat
import traceback
import pymeshlab
import tempfile
import logging

logger = logging.getLogger("mesh_utils")


def random_sample_pointcloud(mesh: trimesh.Trimesh, num=30000, seed=42):
    # points, face_idx = mesh.sample(num, return_index=True)
    points, face_idx = trimesh.sample.sample_surface(mesh, num, seed=seed)
    normals = mesh.face_normals[face_idx]
    rng = np.random.default_rng(seed=seed)
    index = rng.choice(num, num, replace=False)
    return points[index], normals[index]


def sharp_sample_pointcloud(mesh: trimesh.Trimesh, num=16384):
    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:, i]], np.sum(VN[F[:, i]] * N, axis=-1)), axis=-1)
        VN2[F[:, i]] = np.min(dot, axis=-1)

    sharp_mask = VN2 < 0.985
    # collect edge
    edge_a = np.concatenate((F[:, 0], F[:, 1], F[:, 2]))
    edge_b = np.concatenate((F[:, 1], F[:, 2], F[:, 0]))
    sharp_edge = sharp_mask[edge_a] * sharp_mask[edge_b]
    edge_a = edge_a[sharp_edge > 0]
    edge_b = edge_b[sharp_edge > 0]

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = np.random.rand(num)
    w = np.random.rand(num, 1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    return samples, normals


def SampleMesh(V, F, origin_num, seed=42):
    """Sample a mesh to get random points and normals.
    Args:
        V (np.ndarray): Vertices of the mesh.
        F (np.ndarray): Faces of the mesh.
        origin_num (int): Number of original faces to sample from.
    Returns:
        surface_data (dict): Dictionary containing sampled points and normals.
    The dictionary contains:
            - "random_surface": Sampled points and normals from the mesh.
            - "random_surface_fill": Boolean array indicating whether the points are from the fill region.
            - "sharp_surface": Sampled points and normals from the sharp edges of the mesh.
    """

    mesh = trimesh.Trimesh(vertices=V, faces=F[:origin_num])
    mesh_fill = trimesh.Trimesh(vertices=V, faces=F[origin_num:])

    area = mesh.area
    area_fill = mesh_fill.area
    sample_num = 499712 // 2
    num_fill = int(sample_num * (area_fill / (area + area_fill)))
    num = sample_num - num_fill
    # if not mesh.is_watertight:
    #     raise ValueError
    random_surface, random_normal = random_sample_pointcloud(mesh, num=num, seed=seed)
    if num_fill == 0:
        random_surface_fill, random_normal_fill = np.zeros((0, 3)), np.zeros((0, 3))
    else:
        random_surface_fill, random_normal_fill = random_sample_pointcloud(
            mesh_fill, num=num_fill, seed=seed
        )
    random_sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)

    # save_surface
    surface = np.concatenate((random_surface, random_normal), axis=1).astype(np.float16)
    surface_fill = np.concatenate(
        (random_surface_fill, random_normal_fill), axis=1
    ).astype(np.float16)
    sharp_surface = np.concatenate((random_sharp_surface, sharp_normal), axis=1).astype(
        np.float16
    )

    a, b = np.ones(num), np.zeros(num_fill)

    surface_data = {
        "random_surface": np.concatenate((surface, surface_fill), axis=0),
        "random_surface_fill": np.concatenate((a, b)).astype(bool),
        "sharp_surface": sharp_surface,
    }

    return surface_data


def load_surface_points(
    rng,
    random_surface,
    sharpedge_surface,
    pc_size,
    pc_sharpedge_size,
    return_sharpedge_label=True,
    return_normal=True,
):
    """
    sample surface points based on pc_size and pc_sharpedge_size
    Args:
        rng: Random number generator
        random_surface: Array of random surface points
        sharpedge_surface: Array of sharp edge surface points
    Returns:
        surface: Array of surface points and normals
        geo_points: Array of geo points
    """

    surface_normal = []
    if pc_size > 0:
        ind = rng.choice(random_surface.shape[0], pc_size, replace=False)
        random_surface = random_surface[ind]
        if return_sharpedge_label:
            sharpedge_label = np.zeros((pc_size, 1))
            random_surface = np.concatenate((random_surface, sharpedge_label), axis=1)
        surface_normal.append(random_surface)

    if pc_sharpedge_size > 0:
        ind_sharpedge = rng.choice(
            sharpedge_surface.shape[0], pc_sharpedge_size, replace=False
        )
        sharpedge_surface = sharpedge_surface[ind_sharpedge]
        if return_sharpedge_label:
            sharpedge_label = np.ones((pc_sharpedge_size, 1))
            sharpedge_surface = np.concatenate(
                (sharpedge_surface, sharpedge_label), axis=1
            )
        surface_normal.append(sharpedge_surface)

    surface_normal = np.concatenate(surface_normal, axis=0)
    surface_normal = torch.FloatTensor(surface_normal)
    surface = surface_normal[:, 0:3]
    normal = surface_normal[:, 3:6]
    assert surface.shape[0] == pc_size + pc_sharpedge_size

    geo_points = 0.0
    normal = torch.nn.functional.normalize(normal, p=2, dim=1)
    if return_normal:
        surface = torch.cat([surface, normal], dim=-1)
    if return_sharpedge_label:
        surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)
    return surface, geo_points


def sample_bbox_points_from_trimesh(mesh, aabb, num_points, seed=42):
    _faces = mesh.faces
    _vertices = mesh.vertices
    _faces = np.reshape(_faces, (-1))
    num_parts = aabb.shape[0]
    _points = _points = torch.from_numpy(_vertices[_faces])
    _part_mask = torch.all(
        (_points[None, :, :3] >= aabb[:, :1]) & (_points[None, :, :3] <= aabb[:, 1:]),
        dim=-1,
    )
    _part_mask = torch.any(torch.reshape(_part_mask, (num_parts, -1, 3)), dim=-1)
    faces_idx_in_bbox = [torch.nonzero(x).squeeze(-1).numpy() for x in _part_mask]
    # in case some parts are empty(inside surface)
    valid_parts_mask = torch.tensor(
        [len(x) > 0 for x in faces_idx_in_bbox], dtype=torch.bool, device=_points.device
    )
    aabb = aabb[valid_parts_mask]
    # print(len(faces_idx_in_bbox), len(aabb))
    faces_idx_in_bbox = [x for x in faces_idx_in_bbox if len(x) > 0]
    num_valid_parts = len(faces_idx_in_bbox)
    # process valid parts
    mesh_in_bbox = mesh.submesh(faces_idx_in_bbox, append=False)
    points, normals = [], []
    for part in mesh_in_bbox:
        # part_points, face_idx = part.sample(num_points, return_index=True)
        part_points, face_idx = trimesh.sample.sample_surface(
            part, num_points, seed=seed
        )
        part_normals = part.face_normals[face_idx]
        points.append(torch.from_numpy(part_points))
        normals.append(torch.from_numpy(part_normals))
    out = torch.concat(
        [torch.stack(points, dim=0), torch.stack(normals, dim=0)], dim=-1
    )
    out = torch.concat(
        [
            out,
            torch.zeros(
                [num_valid_parts, num_points, 1], dtype=out.dtype, device=out.device
            ),
        ],
        dim=-1,
    )  # add sharp edge label
    return out, valid_parts_mask


def sample_surface_inbbox(
    rng,
    object_surface_raw,
    aabb,
    pc_size_bbox,
    return_normal=True,
    return_sharpedge_label=True,
):
    """
    Sample surface points within the bounding box defined by aabb.
    Args:
        object_surface_raw: Raw surface points from the object
        aabb: [K,2,3] Axis-aligned bounding box defined by min and max corners
        pc_size_bbox: Number of points to sample within the bounding box
    Returns:
        part_surface_inbbox: Sampled surface points within the bounding box
    """
    num_parts = aabb.shape[0]
    object_all_surface = torch.from_numpy(
        np.concatenate(
            [
                object_surface_raw["random_surface"],
                object_surface_raw["sharp_surface"],
            ],
            axis=0,
        )
    )  # [N,6]
    sharpedge_labels = torch.concat(
        [
            torch.zeros(object_surface_raw["random_surface"].shape[0], 1),
            torch.ones(object_surface_raw["sharp_surface"].shape[0], 1),
        ],
        dim=0,
    )
    sampled_masks = torch.all(
        (object_all_surface[None, :, :3] >= aabb[:, :1])
        & (object_all_surface[None, :, :3] <= aabb[:, 1:]),
        dim=-1,
    )
    surfaces = []
    valid_index = []
    for idx, sampled_mask in enumerate(sampled_masks):
        part_surface_inbbox = object_all_surface[sampled_mask]
        sharpedge_label = sharpedge_labels[sampled_mask]
        # TODO: drop inside parts
        if part_surface_inbbox.shape[0] == 0:
            continue
        try:
            ind = rng.choice(part_surface_inbbox.shape[0], pc_size_bbox, replace=False)
        except ValueError:
            ind = np.arange(part_surface_inbbox.shape[0])
            ind = np.concatenate([
                ind,
                rng.choice(
                    part_surface_inbbox.shape[0],
                    pc_size_bbox - part_surface_inbbox.shape[0],
                    replace=True,
                ),
            ])
        part_surface_inbbox = part_surface_inbbox[ind]
        sharpedge_label = sharpedge_label[ind]
        # point feat
        surface = part_surface_inbbox[:, 0:3]
        normal = part_surface_inbbox[:, 3:6]
        # TODO: check normal
        # normal = torch.nn.functional.normalize(normal, p=2, dim=1)
        if return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if return_sharpedge_label:
            surface = torch.cat(
                [surface, sharpedge_label],
                dim=-1,
            )
        surfaces.append(surface)
        valid_index.append(idx)
    surface = torch.stack(surfaces, dim=0)
    return surface, torch.tensor(valid_index)


TRIPO_PALETTE = np.array([
    [29, 57, 166, 255],   # Royal Blue
    [20, 221, 228, 255],  # Cyan / Turquoise
    [245, 166, 35, 255],  # Amber / Gold
    [228, 64, 95, 255],   # Magenta / Coral
    [32, 165, 48, 255],   # Emerald Green
    [155, 81, 224, 255],  # Deep Purple
    [242, 120, 75, 255],  # Warm Orange
    [64, 158, 255, 255],  # Sky Blue
    [255, 99, 132, 255],  # Pastel Pink
    [75, 192, 192, 255],  # Teal
    [255, 205, 86, 255],  # Sun Yellow
    [153, 102, 255, 255], # Violet
    [54, 162, 235, 255],  # Cobalt
    [255, 159, 64, 255],  # Tangerine
    [46, 204, 113, 255],  # Mint Green
    [231, 76, 60, 255],   # Ruby Red
    [52, 152, 219, 255],  # Cerulean
    [241, 196, 15, 255],  # Bright Gold
    [155, 89, 182, 255],  # Amethyst
    [26, 188, 156, 255],  # Aqua
    [230, 126, 34, 255],  # Carrot Orange
    [149, 165, 166, 255], # Silver
    [52, 73, 94, 255],    # Navy Slate
    [192, 57, 43, 255],   # Pomegranate
    [39, 174, 96, 255],   # Jade
    [142, 68, 173, 255],  # Plum
    [41, 128, 185, 255],  # Ocean Blue
    [211, 84, 0, 255],    # Rust
    [127, 140, 141, 255], # Cool Gray
    [22, 160, 133, 255],  # Dark Cyan
    [243, 156, 18, 255],  # Orange Yellow
    [189, 195, 199, 255], # Light Platinum
], dtype=np.uint8)


def seal_and_cap_mesh(sub: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Watertight QuickCap / Manifold Loop Capping:
    Detects all open 3D boundary loops on sliced or extracted part submeshes,
    and caps them with outward-consistent triangulation fans so that exploded
    parts have solid, watertight mating faces with zero holes.
    """
    if sub is None or not hasattr(sub, "vertices") or len(sub.vertices) == 0:
        return sub
    m = sub.copy()
    try:
        import networkx as nx
        face_edges = m.edges
        edge_set = set(map(tuple, face_edges))
        boundary_directed = [tuple(e) for e in face_edges if (e[1], e[0]) not in edge_set]
        if not boundary_directed:
            trimesh.repair.fill_holes(m)
            trimesh.repair.fix_normals(m)
            return m
        
        G = nx.DiGraph()
        for u, v in boundary_directed:
            G.add_edge(u, v)
            
        visited_edges = set()
        new_v = list(m.vertices)
        new_f = list(m.faces)
        
        for u_start, v_next in boundary_directed:
            if (u_start, v_next) in visited_edges:
                continue
            cycle = [u_start]
            curr = v_next
            visited_edges.add((u_start, v_next))
            while curr != u_start and curr in G:
                cycle.append(curr)
                succs = [s for s in G.successors(curr) if (curr, s) not in visited_edges]
                if not succs:
                    break
                next_node = succs[0]
                visited_edges.add((curr, next_node))
                curr = next_node
                if len(cycle) > len(boundary_directed) + 10:
                    break
                    
            if len(cycle) >= 3 and curr == u_start:
                center = np.mean(m.vertices[cycle], axis=0)
                c_idx = len(new_v)
                new_v.append(center)
                for i in range(len(cycle)):
                    u_node = cycle[i]
                    v_node = cycle[(i + 1) % len(cycle)]
                    new_f.append([v_node, u_node, c_idx])
                    
        capped = trimesh.Trimesh(vertices=new_v, faces=new_f, process=True)
        trimesh.repair.fill_holes(capped)
        trimesh.repair.fix_normals(capped)
        return capped
    except Exception as e:
        logger.warning(f"seal_and_cap_mesh fallback: {e}")
        try:
            trimesh.repair.fill_holes(m)
            trimesh.repair.fix_normals(m)
        except Exception:
            pass
        return m


def clean_and_merge_parts(parts, iou_thresh=0.80, containment_thresh=0.85):
    """
    Merges duplicate or heavily co-extensive parts (e.g. duplicate whole-body boxes).
    Prevents over-merging distinct adjacent components like shelves and posts.
    """
    valid = [p for p in parts if p is not None and hasattr(p, "vertices") and len(p.vertices) > 0 and hasattr(p, "faces") and len(p.faces) > 0]
    if len(valid) <= 1:
        return valid

    merged = list(valid)
    changed = True
    while changed:
        changed = False
        n = len(merged)
        for i in range(n):
            if changed:
                break
            box_i = merged[i].bounds
            vol_i = np.prod(box_i[1] - box_i[0])
            if vol_i < 1e-8:
                continue
            for j in range(i + 1, n):
                box_j = merged[j].bounds
                vol_j = np.prod(box_j[1] - box_j[0])
                if vol_j < 1e-8:
                    continue
                inter_min = np.maximum(box_i[0], box_j[0])
                inter_max = np.minimum(box_i[1], box_j[1])
                inter_dim = np.maximum(0, inter_max - inter_min)
                inter_vol = np.prod(inter_dim)
                if inter_vol <= 0:
                    continue
                union_vol = vol_i + vol_j - inter_vol
                iou = inter_vol / union_vol if union_vol > 0 else 0
                contain_i = inter_vol / vol_i
                contain_j = inter_vol / vol_j

                # Only merge true duplicate parts that heavily overlap in both directions
                if iou > iou_thresh or (contain_i > containment_thresh and contain_j > containment_thresh):
                    logger.info(f"[MERGE] Merging duplicate part {i} (vol={vol_i:.4f}) and part {j} (vol={vol_j:.4f}) with IoU={iou:.3f}")
                    combined = trimesh.util.concatenate([merged[i], merged[j]])
                    combined.update_faces(combined.unique_faces())
                    combined.remove_unreferenced_vertices()
                    del merged[j]
                    merged[i] = combined
                    changed = True
                    break
    return merged


def create_tripo_assembly_scene(part_meshes, names=None, colors=None, auto_clean=True, auto_seal=False):
    """
    Constructs a Tripo3D-compatible multi-node trimesh.Scene.
    Every part submesh is normalized so its local vertex coordinates are centered
    around [0, 0, 0], and its world position is encoded as a Node translation
    transform matrix T_i = C_i.
    Includes automated part deduplication (clean_and_merge_parts) and watertight
    boundary loop capping (seal_and_cap_mesh) to prevent holes and color overlap.
    """
    assembly = trimesh.Scene()
    valid_parts = [p for p in part_meshes if p is not None and hasattr(p, "vertices") and len(p.vertices) > 0 and hasattr(p, "faces") and len(p.faces) > 0]
    
    if auto_clean and len(valid_parts) > 1:
        valid_parts = clean_and_merge_parts(valid_parts)
        
    for i, p in enumerate(valid_parts):
        node_name = names[i] if (names is not None and i < len(names)) else f"tripo_part_{i}"
        color = colors[i] if (colors is not None and i < len(colors)) else TRIPO_PALETTE[i % len(TRIPO_PALETTE)]
        
        # Seal and cap open boundary holes
        p_proc = seal_and_cap_mesh(p) if auto_seal else p
        
        # Compute geometric centroid (bbox center) in world space
        center = (p_proc.bounds[0] + p_proc.bounds[1]) / 2.0
        
        # Local-center the vertices
        p_local = p_proc.copy()
        p_local.vertices = p_local.vertices - center
        
        # Assign distinct vibrant face colors
        p_local.visual.face_colors = color
        
        # Set node transformation matrix with translation = center
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = center
        
        assembly.add_geometry(p_local, geom_name=node_name, node_name=node_name, transform=transform)
    return assembly


def convert_gltf_matrix_to_translation(glb_bytes: bytes) -> bytes:
    """
    Post-processes exported GLB binary to convert node matrix transforms into
    clean explicit GLTF translation: [x, y, z] vectors, matching Tripo3D.
    """
    import struct, json
    if len(glb_bytes) < 20:
        return glb_bytes
    magic, version, length = struct.unpack('<4sII', glb_bytes[:12])
    if magic != b'glTF':
        return glb_bytes
    chunk0_len, chunk0_type = struct.unpack('<II', glb_bytes[12:20])
    chunk0_data = glb_bytes[20:20 + chunk0_len]
    chunk1_data = glb_bytes[20 + chunk0_len:]
    
    try:
        gltf = json.loads(chunk0_data.decode('utf-8'))
        for node in gltf.get('nodes', []):
            if 'matrix' in node and node['matrix'] is not None:
                m = node['matrix']
                # Check for pure translation matrix (identity rotation and scale)
                is_pure_trans = (
                    abs(m[0] - 1.0) < 1e-5 and abs(m[5] - 1.0) < 1e-5 and 
                    abs(m[10] - 1.0) < 1e-5 and abs(m[15] - 1.0) < 1e-5 and
                    abs(m[1]) < 1e-5 and abs(m[2]) < 1e-5 and abs(m[3]) < 1e-5 and
                    abs(m[4]) < 1e-5 and abs(m[6]) < 1e-5 and abs(m[7]) < 1e-5 and
                    abs(m[8]) < 1e-5 and abs(m[9]) < 1e-5 and abs(m[11]) < 1e-5
                )
                if is_pure_trans:
                    node['translation'] = [round(float(m[12]), 7), round(float(m[13]), 7), round(float(m[14]), 7)]
                    del node['matrix']
        
        new_json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
        pad = (4 - (len(new_json_bytes) % 4)) % 4
        new_json_bytes += b' ' * pad
        new_total_len = 12 + 8 + len(new_json_bytes) + len(chunk1_data)
        
        header = struct.pack('<4sII', magic, version, new_total_len)
        chunk0_header = struct.pack('<II', len(new_json_bytes), chunk0_type)
        return header + chunk0_header + new_json_bytes + chunk1_data
    except Exception:
        return glb_bytes


def export_tripo_assembly_glb(scene_or_parts, filepath, names=None, colors=None):
    """
    Exports a segmented model as a Tripo3D-compatible multi-node GLB assembly.
    """
    from pathlib import Path
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(scene_or_parts, trimesh.Scene):
        assembly_scene = scene_or_parts
    elif isinstance(scene_or_parts, (list, tuple)):
        assembly_scene = create_tripo_assembly_scene(scene_or_parts, names=names, colors=colors)
    else:
        assembly_scene = trimesh.Scene(scene_or_parts)
    
    glb_data = assembly_scene.export(file_type="glb")
    cleaned_glb = convert_gltf_matrix_to_translation(glb_data)
    with open(filepath, "wb") as f:
        f.write(cleaned_glb)
    print(f"[TRIPO_EXPORT] Saved multi-node assembly GLB to {filepath}")
    return filepath


def explode_mesh_tripo(mesh, explosion_scale=0.4, progress=1.0, easing=True):
    """
    Applies Tripo3D's exact explosion mechanics (reverse-engineered from CRvlez9U.js).
    - Union bounding box center C_global and maximum distance E
    - Normalized distance weight: w_i = ||C_i - C_global|| / E
    - Staggered local timing wave: tau_i = clamp((progress - (1 - w_i) * 0.10) / 0.60, 0, 1)
    - Overshoot cubic polynomial: x(tau_i) = 1 + 2.2*(tau_i - 1)^3 + 1.2*(tau_i - 1)^2
    - Distance-scaled displacement: Delta_i = x(tau_i) * E * (0.35 + 0.65 * min(1, w_i)) * explosion_scale * d_i
    """
    if isinstance(mesh, (list, tuple)):
        scene = create_tripo_assembly_scene(mesh)
    elif isinstance(mesh, trimesh.Scene):
        scene = mesh.copy()
    elif isinstance(mesh, trimesh.Trimesh):
        print("Warning: Single mesh provided, can't create exploded view")
        return trimesh.Scene(mesh)
    else:
        scene = trimesh.Scene(mesh)

    if len(scene.geometry) <= 1:
        return scene

    part_nodes = []
    part_centers = []
    
    # Collect world centroids for all geometries
    for geom_name in scene.geometry.keys():
        transform = scene.graph[geom_name][0]
        geom = scene.geometry[geom_name]
        if hasattr(geom, "vertices") and len(geom.vertices) > 0:
            world_verts = trimesh.transformations.transform_points(geom.vertices, transform)
            center = (world_verts.min(axis=0) + world_verts.max(axis=0)) / 2.0
            part_nodes.append(geom_name)
            part_centers.append(center)
            
    if not part_centers:
        return scene

    part_centers = np.array(part_centers)
    global_min = np.min(part_centers, axis=0)
    global_max = np.max(part_centers, axis=0)
    global_center = (global_min + global_max) / 2.0

    # Max distance E from global center
    distances = np.linalg.norm(part_centers - global_center, axis=1)
    max_distance = np.max(distances)
    if max_distance < 1e-6:
        max_distance = 1.0

    exploded_scene = trimesh.Scene()
    
    for i, geom_name in enumerate(part_nodes):
        geom = scene.geometry[geom_name]
        c = part_centers[i]
        l_vec = c - global_center
        u = np.linalg.norm(l_vec)
        
        # Outward unit direction vector (with jitter if at center)
        if u < 1e-4:
            d_vec = np.array([1.0, (np.random.rand() - 0.5) * 0.01, (np.random.rand() - 0.5) * 0.01])
            d_vec = d_vec / np.linalg.norm(d_vec)
        else:
            d_vec = l_vec / u
            
        w_i = u / max_distance
        
        # Staggered local timing: F(w_i, progress)
        t_local = np.clip((progress - (1.0 - w_i) * 0.10) / 0.60, 0.0, 1.0)
        
        # Overshoot cubic polynomial: x(tau) = 1 + 2.2*(tau - 1)^3 + 1.2*(tau - 1)^2
        if easing:
            eased_progress = 1.0 + 2.2 * ((t_local - 1.0) ** 3) + 1.2 * ((t_local - 1.0) ** 2)
        else:
            eased_progress = t_local
            
        # Distance scaling: R(w_i) = E * (0.35 + 0.65 * min(1, w_i)) * explosion_scale
        r_dist = max_distance * (0.35 + 0.65 * min(1.0, w_i)) * explosion_scale
        offset = d_vec * (eased_progress * r_dist)
        
        # Update node transformation
        orig_transform = scene.graph[geom_name][0].copy()
        new_transform = orig_transform.copy()
        new_transform[:3, 3] = new_transform[:3, 3] + offset
        
        exploded_scene.add_geometry(geom, geom_name=geom_name, node_name=geom_name, transform=new_transform)

    print(f"[TRIPO_EXPLODE] Exploded {len(part_nodes)} parts using Tripo3D exact formulas (scale={explosion_scale}, progress={progress})")
    return exploded_scene


# Backward-compatible alias
explode_mesh = explode_mesh_tripo


def generate_tripo_viewer_html(glb_filename: str, output_html_path: str, model_title: str = "3D Segmented & Exploded Model"):
    """
    Generates a standalone, production-ready interactive 3D WebGL viewer HTML file
    implementing Tripo3D's exact explosion state machine and rotation tumbling mechanics.
    """
    from pathlib import Path
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{model_title} - Tripo3D Exploded View</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
  <style>
    :root {{
      --bg-gradient: radial-gradient(circle at 50% 30%, #1a1e29 0%, #0d0f14 100%);
      --surface-glass: rgba(22, 27, 34, 0.75);
      --surface-border: rgba(255, 255, 255, 0.12);
      --accent-color: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --accent-hover: #4f46e5;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
    body {{
      font-family: 'Outfit', sans-serif;
      background: var(--bg-gradient);
      color: var(--text-main);
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}
    #canvas-container {{
      width: 100vw;
      height: 100vh;
      position: absolute;
      top: 0;
      left: 0;
      z-index: 1;
    }}
    /* Top Bar */
    header {{
      position: absolute;
      top: 20px;
      left: 24px;
      right: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 10;
      pointer-events: none;
    }}
    .badge-container {{
      display: flex;
      align-items: center;
      gap: 12px;
      pointer-events: auto;
      background: var(--surface-glass);
      backdrop-filter: blur(16px);
      padding: 10px 18px;
      border-radius: 14px;
      border: 1px solid var(--surface-border);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
    }}
    .logo-badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 30px;
      height: 30px;
      background: linear-gradient(135deg, #6366f1, #a855f7);
      border-radius: 8px;
      font-weight: 700;
      font-size: 14px;
      color: #fff;
    }}
    .title-group h1 {{
      font-size: 16px;
      font-weight: 600;
      letter-spacing: -0.01em;
    }}
    .title-group p {{
      font-size: 12px;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }}

    /* Control Panel Bottom */
    .controls-panel {{
      position: absolute;
      bottom: 28px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      align-items: center;
      gap: 18px;
      background: var(--surface-glass);
      backdrop-filter: blur(20px);
      padding: 12px 24px;
      border-radius: 20px;
      border: 1px solid var(--surface-border);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
      z-index: 10;
    }}
    .explode-btn {{
      display: flex;
      align-items: center;
      gap: 10px;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: #fff;
      border: none;
      padding: 10px 22px;
      font-size: 14px;
      font-weight: 600;
      font-family: 'Outfit', sans-serif;
      border-radius: 12px;
      cursor: pointer;
      box-shadow: 0 4px 18px var(--accent-glow);
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .explode-btn:hover {{
      transform: translateY(-2px);
      box-shadow: 0 6px 24px rgba(99, 102, 241, 0.55);
    }}
    .explode-btn.active {{
      background: linear-gradient(135deg, #ec4899, #f43f5e);
      box-shadow: 0 4px 18px rgba(244, 63, 94, 0.4);
    }}
    .slider-group {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .slider-group label {{
      font-size: 13px;
      color: var(--text-muted);
      font-weight: 500;
    }}
    .slider-group input[type="range"] {{
      -webkit-appearance: none;
      width: 140px;
      height: 6px;
      background: rgba(255, 255, 255, 0.15);
      border-radius: 4px;
      outline: none;
      cursor: pointer;
    }}
    .slider-group input[type="range"]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #818cf8;
      cursor: pointer;
      box-shadow: 0 0 10px rgba(129, 140, 248, 0.8);
      transition: transform 0.15s;
    }}
    .slider-group input[type="range"]::-webkit-slider-thumb:hover {{
      transform: scale(1.2);
    }}

    /* Parts Drawer Right */
    .parts-drawer {{
      position: absolute;
      top: 80px;
      right: 24px;
      width: 250px;
      max-height: calc(100vh - 180px);
      background: var(--surface-glass);
      backdrop-filter: blur(18px);
      border-radius: 16px;
      border: 1px solid var(--surface-border);
      padding: 16px;
      display: flex;
      flex-direction: column;
      z-index: 10;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}
    .parts-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .parts-header h3 {{
      font-size: 14px;
      font-weight: 600;
    }}
    .parts-count {{
      font-size: 11px;
      color: #a855f7;
      background: rgba(168, 85, 247, 0.15);
      padding: 2px 8px;
      border-radius: 20px;
      font-weight: 600;
    }}
    .parts-list {{
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding-right: 4px;
    }}
    .parts-list::-webkit-scrollbar {{
      width: 4px;
    }}
    .parts-list::-webkit-scrollbar-thumb {{
      background: rgba(255, 255, 255, 0.2);
      border-radius: 4px;
    }}
    .part-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 10px;
      background: rgba(255, 255, 255, 0.03);
      border-radius: 8px;
      font-size: 12px;
      cursor: pointer;
      transition: background 0.15s;
    }}
    .part-item:hover {{
      background: rgba(255, 255, 255, 0.08);
    }}
    .part-badge {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .part-color-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }}
    .loading-overlay {{
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: var(--bg-gradient);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 100;
      transition: opacity 0.4s ease;
    }}
    .spinner {{
      width: 44px;
      height: 44px;
      border: 3px solid rgba(99, 102, 241, 0.2);
      border-top-color: #6366f1;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-bottom: 16px;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>
  <div id="loading" class="loading-overlay">
    <div class="spinner"></div>
    <p style="font-size: 15px; color: var(--text-muted);">Loading 3D Assembly Mesh...</p>
  </div>

  <header>
    <div class="badge-container">
      <div class="logo-badge">3D</div>
      <div class="title-group">
        <h1>{model_title}</h1>
        <p>Tripo3D Matching Explosion Engine</p>
      </div>
    </div>
  </header>

  <div class="parts-drawer">
    <div class="parts-header">
      <h3>Parts Hierarchy</h3>
      <span id="part-count" class="parts-count">0</span>
    </div>
    <div id="parts-container" class="parts-list"></div>
  </div>

  <div class="controls-panel">
    <button id="explode-toggle" class="explode-btn">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
      </svg>
      <span id="explode-text">Explode</span>
    </button>
    <div class="slider-group">
      <label>Displacement:</label>
      <input type="range" id="explode-slider" min="0" max="1" step="0.005" value="0">
    </div>
  </div>

  <div id="canvas-container"></div>

  <script>
    // --- THREE.JS SCENE SETUP ---
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0d0f14');

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 100);
    camera.position.set(2.5, 1.8, 3.2);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false, powerPreference: "high-performance" }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Studio Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight1.position.set(5, 10, 7);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x818cf8, 0.6);
    dirLight2.position.set(-5, -2, -5);
    scene.add(dirLight2);

    // Grid Floor
    const gridHelper = new THREE.GridHelper(10, 20, 0x334155, 0x1e293b);
    gridHelper.position.y = -1.0;
    scene.add(gridHelper);

    // --- REVERSE-ENGINEERED TRIPO3D EXPLOSION ENGINE (CRvlez9U.js) ---
    const States = {{
      Idle: 'idle',
      Expanding: 'expanding',
      Rotating: 'rotating',
      Collapsing: 'collapsing'
    }};

    let currentState = States.Idle;
    let animProgress = 0;
    let animStartTime = 0;
    let startProgress = 0;
    const ANIM_DURATION = 800; // ms

    let partsData = new Map();
    let globalCenter = new THREE.Vector3();
    let maxDistance = 1.0;
    let isExploded = false;

    // Tripo3D exact easing polynomials
    const tripoOvershootEasing = (e) => 1 + 2.2 * Math.pow(e - 1, 3) + 1.2 * Math.pow(e - 1, 2);
    const tripoCollapseEasing = (e) => e < 0.5 ? 4 * e * e * e : 1 - Math.pow(-2 * e + 2, 3) / 2;

    function initExplosionData(model) {{
      const bBoxUnion = new THREE.Box3();
      const meshes = [];

      model.traverse((child) => {{
        if (child.isMesh) {{
          meshes.push(child);
          if (child.material) {{
            child.material.side = THREE.DoubleSide;
            child.material.depthWrite = true;
            child.material.polygonOffset = true;
            child.material.polygonOffsetFactor = 1.0;
            child.material.polygonOffsetUnits = 1.0;
          }}
          child.geometry.computeBoundingBox();
          const partBox = child.geometry.boundingBox.clone();
          partBox.applyMatrix4(child.matrixWorld);
          bBoxUnion.union(partBox);
        }}
      }});

      bBoxUnion.getCenter(globalCenter);
      const bBoxSize = new THREE.Vector3();
      bBoxUnion.getSize(bBoxSize);
      gridHelper.position.y = bBoxUnion.min.y - 0.05;

      // Adjust camera to fit
      const maxDim = Math.max(bBoxSize.x, bBoxSize.y, bBoxSize.z);
      camera.position.set(maxDim * 1.6, maxDim * 1.2, maxDim * 2.2);
      controls.target.copy(globalCenter);
      controls.update();

      // Compute max distance
      maxDistance = 0;
      meshes.forEach((mesh) => {{
        const pos = new THREE.Vector3();
        mesh.getWorldPosition(pos);
        const dist = pos.distanceTo(globalCenter);
        if (dist > maxDistance) maxDistance = dist;
      }});
      if (maxDistance < 1e-4) maxDistance = 1.0;

      // Register parts data
      partsData.clear();
      const partsContainer = document.getElementById('parts-container');
      partsContainer.innerHTML = '';
      document.getElementById('part-count').innerText = meshes.length;

      meshes.forEach((mesh, idx) => {{
        const worldPos = new THREE.Vector3();
        mesh.getWorldPosition(worldPos);

        const lVec = worldPos.clone().sub(globalCenter);
        let dist = lVec.length();
        if (dist < 1e-4) {{
          lVec.set((Math.random() - 0.5) * 0.01, (Math.random() - 0.5) * 0.01, (Math.random() - 0.5) * 0.01);
        }}
        lVec.normalize();

        const geomCenter = new THREE.Vector3();
        if (mesh.geometry.boundingBox) {{
          mesh.geometry.boundingBox.getCenter(geomCenter);
        }}

        // Tripo tumbling rotation vector and speed
        const rotAxis = new THREE.Vector3(1, (Math.random() - 0.5) * 0.6, (Math.random() - 0.5) * 0.6).normalize();
        const rotSpeedFactor = (0.5 + Math.random()) * (Math.random() > 0.5 ? 1 : -1);

        const partInfo = {{
          mesh: mesh,
          originalLocalPosition: mesh.position.clone(),
          originalRotation: mesh.quaternion.clone(),
          explodeDirection: lVec.clone(),
          distanceWeight: dist / maxDistance,
          geometryCenterOffset: geomCenter.clone(),
          accumulatedRotation: 0,
          rotationAxis: rotAxis,
          rotationSpeedFactor: rotSpeedFactor,
          snapshotRotation: 0
        }};
        partsData.set(mesh, partInfo);

        // UI Item
        const item = document.createElement('div');
        item.className = 'part-item';
        const colorHex = '#' + (mesh.material?.color?.getHexString() || '818cf8');
        item.innerHTML = `
          <div class="part-badge">
            <span class="part-color-dot" style="background: ${{colorHex}}"></span>
            <span>${{mesh.name || 'Part ' + idx}}</span>
          </div>
          <span style="opacity: 0.6; font-size: 10px;">👁</span>
        `;
        item.onclick = () => {{
          mesh.visible = !mesh.visible;
          item.style.opacity = mesh.visible ? '1' : '0.35';
        }};
        partsContainer.appendChild(item);
      }});
    }}

    // Apply Tripo3D displacement and tumbling
    function applyExplosion(tValue, deltaSec = 0.016) {{
      const tempVec = new THREE.Vector3();
      const rotQuat = new THREE.Quaternion();
      const isCollapsing = (currentState === States.Collapsing);
      const collapseProgress = isCollapsing ? Math.min((performance.now() - animStartTime) / ANIM_DURATION, 1) : 0;
      const collapseEase = tripoCollapseEasing(collapseProgress);

      partsData.forEach((info, mesh) => {{
        if (isCollapsing) {{
          // Unwind tumbling smoothly to zero
          info.accumulatedRotation = info.snapshotRotation * (1 - collapseEase);
          rotQuat.setFromAxisAngle(info.rotationAxis, info.accumulatedRotation);
          rotQuat.premultiply(info.originalRotation);

          // Pivot compensation around geometric center
          const rOrig = info.geometryCenterOffset.clone().multiply(mesh.scale).applyQuaternion(info.originalRotation);
          const rCur = info.geometryCenterOffset.clone().multiply(mesh.scale).applyQuaternion(rotQuat);
          const pivotComp = rOrig.sub(rCur);

          // Rewind displacement: (1 - collapseEase)
          const rDist = maxDistance * (0.35 + 0.65 * Math.min(1, info.distanceWeight));
          const disp = info.explodeDirection.clone().multiplyScalar((1 - collapseEase) * rDist);

          mesh.position.copy(info.originalLocalPosition).add(pivotComp).add(disp);
          mesh.quaternion.copy(rotQuat);
          return;
        }}

        // Staggered local timing: F(distanceWeight, progress)
        const tLocal = Math.max(0, Math.min(1, (tValue - (1 - info.distanceWeight) * 0.10) / 0.60));
        const easedProgress = tripoOvershootEasing(tLocal);

        // Tumbling rotation once exploded: I(tLocal)
        if (tLocal >= 1.0) {{
          info.accumulatedRotation += 0.02 * info.rotationSpeedFactor * (deltaSec * 60);
          rotQuat.setFromAxisAngle(info.rotationAxis, info.accumulatedRotation);
          rotQuat.premultiply(info.originalRotation);

          // True geometric center pivot compensation
          const rOrig = info.geometryCenterOffset.clone().multiply(mesh.scale).applyQuaternion(info.originalRotation);
          const rCur = info.geometryCenterOffset.clone().multiply(mesh.scale).applyQuaternion(rotQuat);
          const pivotComp = rOrig.sub(rCur);

          const rDist = maxDistance * (0.35 + 0.65 * Math.min(1, info.distanceWeight));
          const disp = info.explodeDirection.clone().multiplyScalar(rDist);

          mesh.position.copy(info.originalLocalPosition).add(pivotComp).add(disp);
          mesh.quaternion.copy(rotQuat);
        }} else {{
          // Pure translation outward
          const rDist = maxDistance * (0.35 + 0.65 * Math.min(1, info.distanceWeight));
          const disp = info.explodeDirection.clone().multiplyScalar(easedProgress * rDist);
          mesh.position.copy(info.originalLocalPosition).add(disp);
          mesh.quaternion.copy(info.originalRotation);
        }}
      }});
    }}

    // State machine updates
    function updateAnimation(deltaSec) {{
      if (currentState === States.Idle) return;

      const now = performance.now();
      const elapsed = now - animStartTime;
      const step = Math.min(elapsed / ANIM_DURATION, 1.0);

      if (currentState === States.Expanding) {{
        animProgress = startProgress + (1.0 - startProgress) * step;
        document.getElementById('explode-slider').value = animProgress;
        applyExplosion(animProgress, deltaSec);
        if (step >= 1.0) {{
          currentState = States.Rotating;
        }}
      }} else if (currentState === States.Rotating) {{
        applyExplosion(1.0, deltaSec);
      }} else if (currentState === States.Collapsing) {{
        animProgress = startProgress * (1.0 - step);
        document.getElementById('explode-slider').value = animProgress;
        applyExplosion(animProgress, deltaSec);
        if (step >= 1.0) {{
          currentState = States.Idle;
          // Reset cleanly to exact rest positions
          partsData.forEach((info, mesh) => {{
            mesh.position.copy(info.originalLocalPosition);
            mesh.quaternion.copy(info.originalRotation);
            info.accumulatedRotation = 0;
          }});
        }}
      }}
    }}

    // Controls
    const explodeBtn = document.getElementById('explode-toggle');
    const explodeSlider = document.getElementById('explode-slider');
    const explodeText = document.getElementById('explode-text');

    explodeBtn.onclick = () => {{
      isExploded = !isExploded;
      explodeBtn.classList.toggle('active', isExploded);
      explodeText.innerText = isExploded ? 'Collapse' : 'Explode';
      animStartTime = performance.now();
      startProgress = parseFloat(explodeSlider.value);

      if (isExploded) {{
        currentState = States.Expanding;
      }} else {{
        // Snapshot current rotation for smooth unwinding
        partsData.forEach((info) => {{
          info.snapshotRotation = info.accumulatedRotation;
        }});
        currentState = States.Collapsing;
      }}
    }};

    explodeSlider.oninput = (e) => {{
      const val = parseFloat(e.target.value);
      currentState = States.Idle;
      isExploded = (val > 0.5);
      explodeBtn.classList.toggle('active', isExploded);
      explodeText.innerText = isExploded ? 'Collapse' : 'Explode';
      applyExplosion(val, 0);
    }};

    // Load Model
    const loader = new THREE.GLTFLoader();
    loader.load('{glb_filename}', (gltf) => {{
      const model = gltf.scene;
      scene.add(model);
      initExplosionData(model);
      document.getElementById('loading').style.opacity = '0';
      setTimeout(() => document.getElementById('loading').style.display = 'none', 400);
    }}, undefined, (err) => {{
      console.warn('Could not auto-load {glb_filename}. Drag and drop a GLB file to view.', err);
      document.getElementById('loading').innerHTML = `
        <p style="color: #ef4444; font-size: 16px; font-weight: 600;">Could not automatically load {glb_filename}</p>
        <p style="color: var(--text-muted); font-size: 13px; margin-top: 8px;">Run a local web server (e.g. <code>python -m http.server</code>) or drag-and-drop your GLB file here.</p>
      `;
    }});

    // Drag-and-drop GLB support
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => {{
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith('.glb')) {{
        const reader = new FileReader();
        reader.onload = (ev) => {{
          loader.parse(ev.target.result, '', (gltf) => {{
            // Remove old meshes
            while(scene.children.length > 0) {{ 
              const obj = scene.children[0];
              scene.remove(obj); 
            }}
            scene.add(ambientLight);
            scene.add(dirLight1);
            scene.add(dirLight2);
            scene.add(gridHelper);
            scene.add(gltf.scene);
            initExplosionData(gltf.scene);
            document.getElementById('loading').style.display = 'none';
          }});
        }};
        reader.readAsArrayBuffer(file);
      }}
    }});

    // Window Resize
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    // Animation Loop
    let clock = new THREE.Clock();
    function animate() {{
      requestAnimationFrame(animate);
      const delta = clock.getDelta();
      controls.update();
      updateAnimation(delta);
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[TRIPO_VIEWER] Saved interactive 3D explosion viewer to {output_html_path}")
    return output_html_path



def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_depth: int = 7,
    indexing: str = "ij",
    octree_resolution: int = None,
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution
    if octree_resolution is None:
        length = bbox_max - bbox_min
        num_cells = np.exp2(octree_depth)

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    """
    修复维度问题的PyTorch实现
    Args:
        input_tensor: shape [D, D, D], torch.float16
        alpha: 标量偏移值
    Returns:
        mask: shape [D, D, D], torch.int32 表面掩码
    """
    device = input_tensor.device
    D = input_tensor.shape[0]
    signed_val = 0.0

    # 添加偏移并处理无效值
    val = input_tensor + alpha
    valid_mask = val > -9000  # 假设-9000是无效值

    # 改进的邻居获取函数（保持维度一致）
    def get_neighbor(t, shift, axis):
        """根据指定轴进行位移并保持维度一致"""
        if shift == 0:
            return t.clone()

        # 确定填充轴（输入为[D, D, D]对应z,y,x轴）
        pad_dims = [0, 0, 0, 0, 0, 0]  # 格式：[x前，x后，y前，y后，z前，z后]

        # 根据轴类型设置填充
        if axis == 0:  # x轴（最后一个维度）
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:  # y轴（中间维度）
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:  # z轴（第一个维度）
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        # 执行填充（添加batch和channel维度适配F.pad）
        padded = F.pad(
            t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode="replicate"
        )  # 反转顺序适配F.pad

        # 构建动态切片索引
        slice_dims = [slice(None)] * 3  # 初始化为全切片
        if axis == 0:  # x轴（dim=2）
            if shift > 0:
                slice_dims[0] = slice(shift, None)
            else:
                slice_dims[0] = slice(None, shift)
        elif axis == 1:  # y轴（dim=1）
            if shift > 0:
                slice_dims[1] = slice(shift, None)
            else:
                slice_dims[1] = slice(None, shift)
        elif axis == 2:  # z轴（dim=0）
            if shift > 0:
                slice_dims[2] = slice(shift, None)
            else:
                slice_dims[2] = slice(None, shift)

        # 应用切片并恢复维度
        padded = padded.squeeze(0).squeeze(0)
        sliced = padded[slice_dims]
        return sliced

    # 获取各方向邻居（确保维度一致）
    left = get_neighbor(val, 1, axis=0)  # x方向
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)  # y方向
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)  # z方向
    up = get_neighbor(val, -1, axis=2)

    # 处理边界无效值（使用where保持维度一致）
    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left = safe_where(left)
    right = safe_where(right)
    back = safe_where(back)
    front = safe_where(front)
    down = safe_where(down)
    up = safe_where(up)

    # 计算符号一致性（转换为float32确保精度）
    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack(
        [
            torch.sign(left.to(torch.float32)),
            torch.sign(right.to(torch.float32)),
            torch.sign(back.to(torch.float32)),
            torch.sign(front.to(torch.float32)),
            torch.sign(down.to(torch.float32)),
            torch.sign(up.to(torch.float32)),
        ],
        dim=0,
    )

    # 检查所有符号是否一致
    same_sign = torch.all(neighbors_sign == sign, dim=0)

    # 生成最终掩码
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)


@torch.no_grad()
def extract_geometry_fast(
    geometric_func: Callable,
    device: torch.device,
    batch_size: int = 1,
    bounds: Union[Tuple[float], List[float], float] = (
        -1.25,
        -1.25,
        -1.25,
        1.25,
        1.25,
        1.25,
    ),
    octree_depth: int = 7,
    num_chunks: int = 10000,
    disable: bool = True,
    mc_level: float = -1 / 512,
    octree_resolution: int = None,
    diffdmc=None,
    rotation_matrix=None,
    mc_mode="mc",
    dtype=torch.float16,
    min_resolution: int = 95,
):
    """

    Args:
        geometric_func:
        device:
        bounds:
        octree_depth:
        batch_size:
        num_chunks:
        disable:

    Returns:

    """

    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
    if mc_mode == "dmc" and diffdmc is None:
        from diso import DiffDMC
        diffdmc = DiffDMC(dtype=torch.float32).to(device)
    if octree_resolution is None:
        octree_resolution = 2**octree_depth

    assert (
        octree_resolution >= 256
    ), "octree resolution must be at least 256 for fast inference"

    resolutions = []
    if octree_resolution < min_resolution:
        resolutions.append(octree_resolution)
    while octree_resolution >= min_resolution:
        resolutions.append(octree_resolution)
        octree_resolution = octree_resolution // 2
    resolutions.reverse()
    bbox_min = np.array(bounds[0:3])
    bbox_max = np.array(bounds[3:6])
    bbox_size = bbox_max - bbox_min

    dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
    dilate.weight = torch.nn.Parameter(
        torch.ones(dilate.weight.shape, dtype=dtype, device=device)
    )

    xyz_samples, grid_size, length = generate_dense_grid_points(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_resolution=resolutions[0],
        indexing="ij",
    )

    grid_size = np.array(grid_size)
    xyz_samples = torch.FloatTensor(xyz_samples).to(device).half()

    if mc_level == -1:
        print(
            f"Training with soft labels, inference with sigmoid and marching cubes"
            f" level 0."
        )
    elif mc_level == 0:
        print(f"VAE Trained with TSDF, inference with marching cubes level 0.")
    else:
        print(
            "VAE Trained with Occupancy, inference with marching cubes level"
            f" {mc_level}."
        )
    batch_logits = []
    for start in tqdm(
        range(0, xyz_samples.shape[0], num_chunks),
        desc=f"MC Level {mc_level} Implicit Function:",
        disable=disable,
        leave=False,
    ):
        queries = xyz_samples[start : start + num_chunks, :]
        batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
        logits = geometric_func(batch_queries)
        if mc_level == -1:
            mc_level = 0
            print(
                f"Training with soft labels, inference with sigmoid and marching cubes"
                f" level 0."
            )
            logits = torch.sigmoid(logits) * 2 - 1
        batch_logits.append(logits)

    grid_logits = (
        torch.cat(batch_logits, dim=1)
        .view((batch_size, grid_size[0], grid_size[1], grid_size[2]))
        .half()
    )

    for octree_depth_now in resolutions[1:]:
        grid_size = np.array([octree_depth_now + 1] * 3)
        resolution = bbox_size / octree_depth_now
        next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
        if octree_depth_now == resolutions[-1]:
            next_logits = torch.full(
                next_index.shape, float("nan"), dtype=dtype, device=device
            )
        else:
            next_logits = torch.full(
                next_index.shape, -10000.0, dtype=dtype, device=device
            )

        FN = extract_near_surface_volume_fn
        curr_points = FN(grid_logits.squeeze(0), mc_level)
        curr_points += grid_logits.squeeze(0).abs() < min(
            0.95, 0.95 * 128 * 4 / octree_depth_now
        )
        if octree_depth_now > 510:
            expand_num = 0
        else:
            expand_num = 1
        for i in range(expand_num):
            curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
        (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)
        next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
        for i in range(1):
            next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
        nidx = torch.where(next_index > 0)
        next_points = torch.stack(nidx, dim=1)
        next_points = next_points * torch.tensor(
            resolution, device=device
        ) + torch.tensor(bbox_min, device=device)
        batch_logits = []
        for start in tqdm(
            range(0, next_points.shape[0], num_chunks),
            desc=f"MC Level {octree_depth_now + 1} Implicit Function:",
            disable=disable,
            leave=False,
        ):
            queries = next_points[start : start + num_chunks, :]
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            logits = geometric_func(batch_queries)
            if mc_level == -1:
                mc_level = 0
                print(
                    f"Training with soft labels, inference with sigmoid and marching"
                    f" cubes level 0."
                )
                logits = torch.sigmoid(logits) * 2 - 1
            batch_logits.append(logits)
        grid_logits = torch.cat(batch_logits, dim=1).half()
        next_logits[nidx] = grid_logits[0]
        grid_logits = next_logits.unsqueeze(0)
    # s_mc = time.time()
    mesh_v_f = []
    has_surface = np.zeros((batch_size,), dtype=np.bool_)
    for i in range(batch_size):
        try:
            if mc_mode == "mc":
                if len(resolutions) > 1:
                    mask = (next_index > 0).cpu().numpy()
                    grid_logits = grid_logits.cpu().numpy()
                    vertices, faces, normals, _ = measure.marching_cubes(
                        grid_logits[i], mc_level, method="lewiner", mask=mask
                    )
                else:
                    vertices, faces, normals, _ = measure.marching_cubes(
                        grid_logits[i].cpu().numpy(), mc_level, method="lewiner"
                    )
                vertices = vertices / (grid_size - 1) * bbox_size + bbox_min
                # vertices[:, [0, 1]] = vertices[:, [1, 0]]
            elif mc_mode == "dmc":
                part_logits = -grid_logits[i].to(torch.float32).contiguous()
                verts, faces = diffdmc(
                    part_logits, deform=None, return_quads=False, normalize=False
                )
                if len(verts) == 0:
                    mesh_v_f.append((None, None))
                    has_surface[i] = False
                    continue
                res_t = torch.tensor(resolution, device=device, dtype=verts.dtype)
                bmin_t = torch.tensor(bbox_min, device=device, dtype=verts.dtype)
                verts = verts * res_t + bmin_t
                vertices = verts.detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()[:, ::-1]
            elif mc_mode == "odc":
                # https://github.com/KAIST-Visual-AI-Group/ODC
                from .occupancy_dual_contouring import occupancy_dual_contouring
                import torch.nn.functional as F

                odc = occupancy_dual_contouring("cuda")

                size = grid_logits.shape[-1]
                grid_logits = grid_logits.reshape(1, 1, size, size, size)

                def implicit_function(xyz):
                    xyz = xyz.reshape(1, -1, 1, 1, 3).float()
                    # print(grid_logits.dtype, xyz.dtype)
                    outputs = F.grid_sample(grid_logits.float(), xyz)
                    outputs = -outputs.reshape(-1)
                    return outputs

                num_cells = (
                    octree_resolution
                    if octree_resolution is not None
                    else np.exp2(octree_depth)
                )
                vertices, triangles = odc.extract_mesh(
                    implicit_function,
                    isolevel=mc_level,
                    min_coord=bbox_min,
                    max_coord=bbox_max,
                    num_grid=1024,
                )
                vertices = vertices.detach().cpu().numpy()
                faces = triangles.detach().cpu().numpy()[:, ::-1]
            else:
                raise ValueError(f"Unknown marching cubes mode: {mc_mode}")
            mesh_v_f.append((vertices.astype(np.float32), np.ascontiguousarray(faces)))
            has_surface[i] = True

        except ValueError:
            traceback.print_exc()
            mesh_v_f.append((None, None))
            has_surface[i] = False

        except RuntimeError:
            traceback.print_exc()
            mesh_v_f.append((None, None))
            has_surface[i] = False
    return mesh_v_f, has_surface


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = trimesh.load(temp_file.name)
    # 检查加载的对象类型
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        # 如果是Scene，遍历所有的geometry并合并
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        mesh.export(temp_file.name)
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(temp_file.name)
    return mesh


def remove_overlength_edge(mesh: pymeshlab.MeshSet, max_length: float):
    mesh.apply_filter("compute_selection_by_edge_length", threshold=max_length)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter(
        "compute_selection_by_small_disconnected_components_per_face", nbfaceratio=0.005
    )
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def fix_mesh(mesh: trimesh.Trimesh):
    """
    Non-destructive watertight mesh repair:
    Removes duplicated faces/unreferenced vertices and seals all open boundary loops.
    Never deletes large planar triangles (which caused holes in parts).
    """
    if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        return mesh
    try:
        m = mesh.copy()
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        m = seal_and_cap_mesh(m)
        trimesh.repair.fix_normals(m)
        return m
    except Exception as e:
        logger.warning(f"fix_mesh fallback: {e}")
        return mesh

