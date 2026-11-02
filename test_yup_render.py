import trimesh
import numpy as np
from PIL import Image

def render_mesh_yup(mesh, width=900, height=900, elev=18, azim=35, point_density=600000):
    points, face_indices = trimesh.sample.sample_surface(mesh, point_density)
    normals = mesh.face_normals[face_indices]
    
    if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None and len(mesh.visual.face_colors) > 0:
        raw_colors = mesh.visual.face_colors[face_indices, :3]
    else:
        raw_colors = np.ones((len(points), 3), dtype=np.uint8) * 200
        
    # In this mesh, Y is Up (+Y top, -Y legs), X is Width, Z is Depth
    center = points.mean(axis=0)
    pts = points - center
    scale = np.max(np.abs(pts))
    pts /= (scale + 1e-8)
    
    rad_elev = np.radians(elev)
    rad_azim = np.radians(azim)
    
    # Orbit around Y (vertical axis)
    cos_a, sin_a = np.cos(rad_azim), np.sin(rad_azim)
    R_y = np.array([
        [ cos_a, 0, sin_a],
        [     0, 1,     0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)
    
    # Pitch elevation looking slightly down from above
    cos_e, sin_e = np.cos(rad_elev), np.sin(rad_elev)
    R_x = np.array([
        [1,     0,      0],
        [0, cos_e, -sin_e],
        [0, sin_e,  cos_e]
    ], dtype=np.float32)
    
    R = R_x @ R_y
    pts_cam = pts @ R.T
    normals_cam = normals @ R.T
    
    # Lighting from front-top-right in camera space
    light_dir = np.array([0.35, 0.55, 0.75], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    diffuse = np.maximum(0.0, np.sum(normals_cam * light_dir, axis=-1))
    ambient = 0.42
    intensity = np.clip(ambient + 0.58 * diffuse, 0.0, 1.0)[:, None]
    
    colors_lit = np.clip(raw_colors.astype(np.float32) * intensity, 0, 255).astype(np.uint8)
    
    x_cam = pts_cam[:, 0]
    y_cam = pts_cam[:, 1]
    z_cam = pts_cam[:, 2]
    
    pad = 0.12
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
    
    img = np.full((height, width, 3), 18, dtype=np.uint8)
    for du in [0, 1, -1]:
        for dv in [0, 1, -1]:
            uu = np.clip(u + du, 0, width - 1)
            vv = np.clip(v + dv, 0, height - 1)
            img[vv, uu] = colors_lit
            
    return Image.fromarray(img)

if __name__ == "__main__":
    m = trimesh.load("outputs/complete_pipeline_showcase/Step4_SecondPass_TargetedPrompting.glb", force="mesh")
    img = render_mesh_yup(m, elev=15, azim=30)
    img.save("outputs/complete_pipeline_showcase/test_perfect_upright.png")
    print("Saved perfect upright render!")
