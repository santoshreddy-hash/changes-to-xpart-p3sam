import os
import shutil
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = "outputs/complete_pipeline_showcase"
os.makedirs(OUT_DIR, exist_ok=True)

def render_mesh_yup(mesh, width=900, height=900, elev=18, azim=30, point_density=600000):
    points, face_indices = trimesh.sample.sample_surface(mesh, point_density)
    normals = mesh.face_normals[face_indices]
    
    if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None and len(mesh.visual.face_colors) > 0:
        raw_colors = mesh.visual.face_colors[face_indices, :3]
    else:
        raw_colors = np.ones((len(points), 3), dtype=np.uint8) * 200
        
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

# Step definition with source GLB paths
steps = [
    {
        "step_num": "Step 1",
        "title": "Baseline P3-SAM",
        "subtitle": "Single-Mask Greedy, FPS 400",
        "desc": "Selects highest IoU mask per prompt. Misses thin horizontal crossbar; jagged point projection boundaries.",
        "src_glb": "outputs/benchmark_secondpass/Baseline_400/colorized_result.glb",
        "out_glb": os.path.join(OUT_DIR, "Step1_Baseline_P3SAM.glb"),
        "preview_png": os.path.join(OUT_DIR, "step1_baseline_preview.png"),
        "parts_count": 19,
        "miou": "73.04%"
    },
    {
        "step_num": "Step 2",
        "title": "Multi-Mask + Stability",
        "subtitle": "All 3 Scales + Stability Filter",
        "desc": "Retains all 3 mask scales per prompt. Filters unstable masks (tau +/- 0.05). Full-pool FP32 GPU NMS.",
        "src_glb": "outputs/benchmark_multimask/Improved_plus_Stability_NMS_0.7/segmented_mesh.glb",
        "out_glb": os.path.join(OUT_DIR, "Step2_MultiMask_Stability.glb"),
        "preview_png": os.path.join(OUT_DIR, "step2_multimask_preview.png"),
        "parts_count": 22,
        "miou": "68.77%"
    },
    {
        "step_num": "Step 3",
        "title": "Graph Cut Refinement",
        "subtitle": "MRF Alpha-Expansion on Dual Graph",
        "desc": "Minimizes unary negative log-probs and pairwise dihedral angle normal difference. Snaps boundaries to crease edges.",
        "src_glb": "outputs/benchmark_graphcut/woodenrack_graphcut_mesh.glb",
        "out_glb": os.path.join(OUT_DIR, "Step3_GraphCut_Refinement.glb"),
        "preview_png": os.path.join(OUT_DIR, "step3_graphcut_preview.png"),
        "parts_count": 21,
        "miou": "73.55%"
    },
    {
        "step_num": "Step 4",
        "title": "Second-Pass Prompting",
        "subtitle": "Targeted Sampling for Thin Parts",
        "desc": "Samples 100 targeted prompts in uncertainty regions; reuses cached Sonata features. Recovers thin crossbar (Part 376, 90.0% IoU).",
        "src_glb": "outputs/benchmark_secondpass/SecondPass_Plus100/colorized_result.glb",
        "out_glb": os.path.join(OUT_DIR, "Step4_SecondPass_TargetedPrompting.glb"),
        "preview_png": os.path.join(OUT_DIR, "step4_secondpass_preview.png"),
        "parts_count": 23,
        "miou": "73.04% (100% recall)"
    },
    {
        "step_num": "Step 5a",
        "title": "Hierarchical Level 3",
        "subtitle": "Fine Structural Parts (K=22)",
        "desc": "Base structural decomposition of all 22 individual parts (shelves, upright legs, panels, divider slats).",
        "src_glb": "outputs/benchmark_hierarchical/level3_fine_mesh.glb",
        "out_glb": os.path.join(OUT_DIR, "Step5a_Hierarchical_Level3_Fine.glb"),
        "preview_png": os.path.join(OUT_DIR, "step5a_hierarchical_fine_preview.png"),
        "parts_count": 22,
        "miou": "Base Level"
    },
    {
        "step_num": "Step 5b",
        "title": "Hierarchical Level 2",
        "subtitle": "Medium Functional Assemblies (K=5)",
        "desc": "Connectivity-constrained clustering merges adjacent slats into 5 shelf and frame functional units with 0 disconnected parts.",
        "src_glb": "outputs/benchmark_hierarchical/constrained/level2_med_mesh.glb",
        "out_glb": os.path.join(OUT_DIR, "Step5b_Hierarchical_Level2_Medium.glb"),
        "preview_png": os.path.join(OUT_DIR, "step5b_hierarchical_medium_preview.png"),
        "parts_count": 5,
        "miou": "100% Purity"
    },
    {
        "step_num": "Step 5c",
        "title": "Hierarchical Level 1",
        "subtitle": "Coarse Macro-Assembly (K=2)",
        "desc": "Top-level macro segmentation: Outer Frame / Shell vs Inner Shelving core. 100% strict tree nestedness.",
        "src_glb": "outputs/benchmark_hierarchical/constrained/level1_coarse_mesh.glb",
        "out_glb": os.path.join(OUT_DIR, "Step5c_Hierarchical_Level1_Coarse.glb"),
        "preview_png": os.path.join(OUT_DIR, "step5c_hierarchical_coarse_preview.png"),
        "parts_count": 2,
        "miou": "100% Purity"
    }
]

rendered_images = []
for item in steps:
    print(f"Processing {item['step_num']}: {item['title']}...")
    shutil.copyfile(item["src_glb"], item["out_glb"])
    
    mesh = trimesh.load(item["out_glb"], force="mesh")
    img = render_mesh_yup(mesh, width=900, height=900, elev=18, azim=30, point_density=600000)
    img.save(item["preview_png"])
    rendered_images.append((item, img))
    print(f"  -> Saved GLB: {item['out_glb']} ({os.path.getsize(item['out_glb'])/1024/1024:.2f} MB)")
    print(f"  -> Saved Upright Preview: {item['preview_png']}")

# Generate unified composite showcase image
print("Generating composite upright progression banner...")
fig, axes = plt.subplots(1, 7, figsize=(32, 5.8), facecolor='#121316')
for idx, (item, img) in enumerate(rendered_images):
    ax = axes[idx]
    ax.imshow(img)
    title_text = f"{item['step_num']}\n{item['title']}\n{item['subtitle']}\n[{item['parts_count']} Parts]"
    ax.set_title(title_text, fontsize=9.5, color='white', fontweight='bold', pad=8)
    ax.axis('off')

plt.tight_layout()
composite_path = os.path.join(OUT_DIR, "complete_methods_progression_showcase.png")
plt.savefig(composite_path, dpi=200, bbox_inches='tight', facecolor='#121316')
plt.close()
print(f"Composite showcase saved to: {composite_path}")
