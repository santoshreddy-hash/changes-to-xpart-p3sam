import os
import trimesh
import numpy as np

files = {
    'Step 1 (Baseline 1-Mask)': 'outputs/benchmark_secondpass/Baseline_400/colorized_result.glb',
    'Step 2 (Multi-Mask + Stability)': 'outputs/benchmark_multimask/Improved_plus_Stability_NMS_0.7/segmented_mesh.glb',
    'Step 3 (Graph Cut Refinement)': 'outputs/benchmark_graphcut/woodenrack_graphcut_mesh.glb',
    'Step 4 (Second-Pass Prompting)': 'outputs/benchmark_secondpass/SecondPass_Plus100/colorized_result.glb',
    'Step 5a (Hierarchical Fine L3)': 'outputs/benchmark_hierarchical/level3_fine_mesh.glb',
    'Step 5b (Hierarchical Medium L2)': 'outputs/benchmark_hierarchical/constrained/level2_med_mesh.glb',
    'Step 5c (Hierarchical Coarse L1)': 'outputs/benchmark_hierarchical/constrained/level1_coarse_mesh.glb',
}

for name, rel_path in files.items():
    if not os.path.exists(rel_path):
        print(f"Missing: {rel_path}")
        continue
    m = trimesh.load(rel_path, force='mesh')
    size_mb = os.path.getsize(rel_path) / (1024 * 1024)
    if hasattr(m.visual, 'face_colors') and m.visual.face_colors is not None:
        colors = np.unique(m.visual.face_colors[:, :3], axis=0)
        n_colors = len(colors)
    else:
        n_colors = 0
    print(f"{name:32s} | Faces: {len(m.faces):7d} | Verts: {len(m.vertices):7d} | Colors (parts): {n_colors:2d} | Size: {size_mb:5.2f} MB")
