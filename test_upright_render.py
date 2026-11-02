import trimesh
import numpy as np
from PIL import Image

def render_upright_mesh(mesh, width=800, height=800, elev=15, azim=35, point_density=500000):
    points, face_indices = trimesh.sample.sample_surface(mesh, point_density)
    normals = mesh.face_normals[face_indices]
    
    if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None and len(mesh.visual.face_colors) > 0:
        raw_colors = mesh.visual.face_colors[face_indices, :3]
    else:
        raw_colors = np.ones((len(points), 3), dtype=np.uint8) * 200
        
    # Rotate mesh so longest axis (X, extent 1.87) is pointing UP along camera's Y axis
    # X -> Z, Y -> X, Z -> Y or rotation around Y by -90 deg
    # Let's rotate around Y by 90 degrees: [x, y, z] -> [z, y, -x] or swap
    # If X is height, in upright view: Up is X. So we want X to map to Z_world or Y_screen.
    # Let's rotate around Z or Y:
    # Rotate around Y axis by 90 degrees:
    # new_x = z, new_y = y, new_z = -x
    R_up = np.array([
        [0,  0, 1],
        [0,  1, 0],
        [-1, 0, 0]
    ], dtype=np.float32)
    
    # Or test: if X was along horizontal, rotate in XY:
    # let's test R_up that puts X to Z:
    # [0, 0, 1]
    # [0, 1, 0]
    # [-1, 0, 0]
    
    pts_rot = points @ R_up.T
    norms_rot = normals @ R_up.T
    
    # Center and normalize
    center = pts_rot.mean(axis=0)
    pts = pts_rot - center
    scale = np.max(np.abs(pts))
    pts /= (scale + 1e-8)
    
    rad_elev = np.radians(elev)
    rad_azim = np.radians(azim)
    
    cos_a, sin_a = np.cos(rad_azim), np.sin(rad_azim)
    R_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    
    cos_e, sin_e = np.cos(rad_elev), np.sin(rad_elev)
    R_x = np.array([
        [1, 0,      0],
        [0, cos_e, -sin_e],
        [0, sin_e,  cos_e]
    ])
    
    R = R_x @ R_z
    pts_cam = pts @ R.T
    normals_cam = norms_rot @ R.T
    
    light_dir = np.array([0.3, 0.4, 0.86])
    light_dir /= np.linalg.norm(light_dir)
    diffuse = np.maximum(0.0, np.sum(normals_cam * light_dir, axis=-1))
    ambient = 0.45
    intensity = np.clip(ambient + 0.55 * diffuse, 0.0, 1.0)[:, None]
    
    colors_lit = np.clip(raw_colors.astype(np.float32) * intensity, 0, 255).astype(np.uint8)
    
    x_cam = pts_cam[:, 0]
    y_cam = pts_cam[:, 1]
    z_cam = pts_cam[:, 2]
    
    pad = 0.1
    u = ((x_cam + (1.0 + pad)) / (2.0 * (1.0 + pad)) * (width - 1)).astype(np.int32)
    v = (((1.0 + pad) - y_cam) / (2.0 * (1.0 + pad)) * (height - 1)).astype(np.int32)
    
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid]
    v = v[valid]
    z_cam = z_cam[valid]
    colors_lit = colors_lit[valid]
    
    sort_order = np.argsort(z_cam)
    u = u[sort_order]
    v = v[sort_order]
    colors_lit = colors_lit[sort_order]
    
    img = np.full((height, width, 3), 24, dtype=np.uint8)
    for du in [0, 1, -1]:
        for dv in [0, 1, -1]:
            uu = np.clip(u + du, 0, width - 1)
            vv = np.clip(v + dv, 0, height - 1)
            img[vv, uu] = colors_lit
            
    return Image.fromarray(img)

if __name__ == "__main__":
    m = trimesh.load("outputs/complete_pipeline_showcase/Step4_SecondPass_TargetedPrompting.glb", force="mesh")
    img = render_upright_mesh(m)
    img.save("outputs/complete_pipeline_showcase/test_upright.png")
    print("Saved test upright render!")
