import os
import sys
import numpy as np
import trimesh
from PIL import Image

def render_mesh_to_image(mesh, width=800, height=800, elev=20, azim=45, point_density=500000):
    """
    Renders a colorized 3D mesh into a high-quality shaded 2D PNG using vectorized camera projection and z-buffering.
    """
    points, face_indices = trimesh.sample.sample_surface(mesh, point_density)
    normals = mesh.face_normals[face_indices]
    
    # Extract face colors
    if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None and len(mesh.visual.face_colors) > 0:
        raw_colors = mesh.visual.face_colors[face_indices, :3]
    else:
        raw_colors = np.ones((len(points), 3), dtype=np.uint8) * 200
        
    # Center and normalize coordinates
    center = points.mean(axis=0)
    pts = points - center
    scale = np.max(np.abs(pts))
    pts /= (scale + 1e-8)
    
    # Camera rotation
    rad_elev = np.radians(elev)
    rad_azim = np.radians(azim)
    
    # Azimuth rotation around Z
    cos_a, sin_a = np.cos(rad_azim), np.sin(rad_azim)
    R_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    
    # Elevation rotation around X
    cos_e, sin_e = np.cos(rad_elev), np.sin(rad_elev)
    R_x = np.array([
        [1, 0,      0],
        [0, cos_e, -sin_e],
        [0, sin_e,  cos_e]
    ])
    
    R = R_x @ R_z
    pts_cam = pts @ R.T
    normals_cam = normals @ R.T
    
    # Directional light from front-top-right: [0.3, 0.4, 0.86]
    light_dir = np.array([0.3, 0.4, 0.86])
    light_dir /= np.linalg.norm(light_dir)
    diffuse = np.maximum(0.0, np.sum(normals_cam * light_dir, axis=-1))
    ambient = 0.45
    intensity = np.clip(ambient + 0.55 * diffuse, 0.0, 1.0)[:, None]
    
    colors_lit = np.clip(raw_colors.astype(np.float32) * intensity, 0, 255).astype(np.uint8)
    
    # Project to screen: orthographic projection
    # X -> u, Y -> v, Z -> depth
    x_cam = pts_cam[:, 0]
    y_cam = pts_cam[:, 1]
    z_cam = pts_cam[:, 2]  # depth
    
    # Viewport bounds with padding
    pad = 0.1
    u = ((x_cam + (1.0 + pad)) / (2.0 * (1.0 + pad)) * (width - 1)).astype(np.int32)
    v = (((1.0 + pad) - y_cam) / (2.0 * (1.0 + pad)) * (height - 1)).astype(np.int32)
    
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid]
    v = v[valid]
    z_cam = z_cam[valid]
    colors_lit = colors_lit[valid]
    
    # Sort points from back to front
    sort_order = np.argsort(z_cam)
    u = u[sort_order]
    v = v[sort_order]
    colors_lit = colors_lit[sort_order]
    
    # Background: clean modern dark gradient or studio dark gray
    img = np.full((height, width, 3), 28, dtype=np.uint8)
    
    # Render points with 2x2 splatting for smooth coverage
    for du in [0, 1, -1]:
        for dv in [0, 1, -1]:
            uu = np.clip(u + du, 0, width - 1)
            vv = np.clip(v + dv, 0, height - 1)
            img[vv, uu] = colors_lit
            
    return Image.fromarray(img)

if __name__ == "__main__":
    print("Renderer module defined!")
