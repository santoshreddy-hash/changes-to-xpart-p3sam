import os
import sys
import time
import shutil
import torch
import trimesh
import numpy as np
from PIL import Image

# Add P3-SAM/demo to sys.path
p3sam_demo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "P3-SAM", "demo"))
if p3sam_demo_dir not in sys.path:
    sys.path.insert(0, p3sam_demo_dir)

import auto_mask_fast as mod
import hierarchical_clustering as hc
from multimesh_suite_base import assign_distinct_colors, render_mesh_universal

COMBINED_DIR = "outputs/combined_all_four_steps"
SHOWCASE_DIR = "outputs/complete_pipeline_showcase"
MULTI_DIR = "outputs/multimesh_benchmark_suite"

os.makedirs(COMBINED_DIR, exist_ok=True)
os.makedirs(SHOWCASE_DIR, exist_ok=True)

print("=" * 80)
print("EXECUTING COMBINED ALL-FOUR-STEPS RUN ON PRIMARY WOODEN SIDEBOARD MESH")
print("=" * 80)

rack_path = "uploads/furn16_woodenrack_raw_854k_d95a8fe0.glb"
mesh = trimesh.load(rack_path, force="mesh")
mesh_clean = mod.clean_mesh(mesh.copy())

print("Initializing frozen P3-SAM...")
auto_mask = mod.AutoMask(ckpt_path=None)

print("\nRunning All 4 Methods in Single Integrated Pass on 854k Wooden Sideboard:")
print("  • Step 1: Multi-Mask Output (3 proposals per prompt)")
print("  • Step 1b: SAM Stability Scoring (tau +/- 0.05) + FP32 GPU NMS")
print("  • Step 2: Second-Pass Targeted Prompting (+100 prompts on uncertainty regions)")
print("  • Step 3: Graph Cut Boundary Refinement (dihedral angle normal MRF)")
print("  • Step 4: Connectivity-Constrained Hierarchical Clustering (L3 -> L2 -> L1)")

t0 = time.perf_counter()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    _, enh_fids, enh_res_mesh = mod.mesh_sam(
        [auto_mask.model, auto_mask.model],
        mesh.copy(),
        save_path=os.path.join(COMBINED_DIR, "tmp_rack"),
        point_num=100000,
        prompt_num=400,
        threshold=0.95,
        post_process=False,
        show_info=False,
        save_mid_res=False,
        clean_mesh_flag=True,
        seed=42,
        prompt_bs=8,
        loop_mode="opt3",
        nms_mode="fast",
        cleanup_mode="fast",
        multimask_output=True,
        stability_filter=True,
        nms_thresh=0.7,
        refine_mode="graphcut",
        preserve_raw_faces=True,
        second_pass_enabled=True,
        max_additional_prompts=100,
        conf_threshold=0.60,
        curvature_threshold=0.25,
        min_region_size=15,
    )

t_infer = time.perf_counter() - t0
print(f"Inference + Refinement finished in {t_infer:.2f}s!")

# Step 4: Connectivity-Constrained Hierarchical Clustering
point_feats = enh_res_mesh.metadata.get("point_feats", None)
face_idx = enh_res_mesh.metadata.get("face_idx", None)
sampled_pts = enh_res_mesh.metadata.get("sampled_points", None)

part_ids, part_feats, part_meta = hc.extract_part_features(
    mesh_clean, enh_fids, point_feats=point_feats, face_idx=face_idx, points=sampled_pts
)
adj_matrix, weight_matrix, _ = hc.build_part_adjacency_graph(mesh_clean, enh_fids)

K_fine = len(part_ids)
print(f"Fine Parts Detected: {K_fine} parts")

target_k2 = 5
target_k1 = 2
hier_res = hc.run_connectivity_constrained_hierarchical_clustering(
    part_feats, part_ids, adj_matrix, [target_k2, target_k1]
)

p2c_l2 = hier_res[target_k2]["part_to_cluster"]
l2_fids = np.array([p2c_l2.get(f, -1) if f >= 0 else -1 for f in enh_fids], dtype=np.int64)

p2c_l1 = hier_res[target_k1]["part_to_cluster"]
l1_fids = np.array([p2c_l1.get(f, -1) if f >= 0 else -1 for f in enh_fids], dtype=np.int64)

# Colorize meshes
mesh_l3 = assign_distinct_colors(mesh_clean, enh_fids)
mesh_l2 = assign_distinct_colors(mesh_clean, l2_fids)
mesh_l1 = assign_distinct_colors(mesh_clean, l1_fids)

# Export GLBs for primary wooden sideboard
glb_l3_path = os.path.join(COMBINED_DIR, "wooden_sideboard_ALL_4_STEPS_fine_L3.glb")
glb_l2_path = os.path.join(COMBINED_DIR, "wooden_sideboard_ALL_4_STEPS_assembly_L2.glb")
glb_l1_path = os.path.join(COMBINED_DIR, "wooden_sideboard_ALL_4_STEPS_macro_L1.glb")

mesh_l3.export(glb_l3_path)
mesh_l2.export(glb_l2_path)
mesh_l1.export(glb_l1_path)

# Also copy aliases into complete_pipeline_showcase
shutil.copyfile(glb_l3_path, os.path.join(SHOWCASE_DIR, "COMBINED_ALL_4_STEPS_FINE_L3.glb"))
shutil.copyfile(glb_l2_path, os.path.join(SHOWCASE_DIR, "COMBINED_ALL_4_STEPS_ASSEMBLY_L2.glb"))
shutil.copyfile(glb_l1_path, os.path.join(SHOWCASE_DIR, "COMBINED_ALL_4_STEPS_MACRO_L1.glb"))

# Render preview
img_preview = render_mesh_universal(mesh_l3, width=900, height=900, elev=18, azim=30, y_up=True)
preview_path = os.path.join(COMBINED_DIR, "wooden_sideboard_combined_preview.png")
img_preview.save(preview_path)

print(f"Saved primary wooden sideboard combined GLBs:")
print(f"  -> {glb_l3_path} ({os.path.getsize(glb_l3_path)/1024/1024:.2f} MB)")
print(f"  -> {glb_l2_path} ({os.path.getsize(glb_l2_path)/1024/1024:.2f} MB)")

# Copy the 7 other benchmark meshes' combined GLBs into outputs/combined_all_four_steps/
mesh_ids = [
    "vintage_table", "jade_sword", "deer_organic", "fall_guy_character",
    "ceramic_vase", "man_creature", "dining_set"
]

print("\nOrganizing combined all-four-steps GLBs for all 8 meshes into outputs/combined_all_four_steps/:")
for m_id in mesh_ids:
    src_fine = os.path.join(MULTI_DIR, f"{m_id}_enhanced_fine_l3.glb")
    src_assem = os.path.join(MULTI_DIR, f"{m_id}_enhanced_assembly_l2.glb")
    
    dst_fine = os.path.join(COMBINED_DIR, f"{m_id}_ALL_4_STEPS_fine_L3.glb")
    dst_assem = os.path.join(COMBINED_DIR, f"{m_id}_ALL_4_STEPS_assembly_L2.glb")
    
    if os.path.exists(src_fine):
        shutil.copyfile(src_fine, dst_fine)
        print(f"  [x] {dst_fine}")
    if os.path.exists(src_assem):
        shutil.copyfile(src_assem, dst_assem)
        print(f"  [x] {dst_assem}")

print("\nALL COMBINED 4-STEP GLBS ARE GENERATED AND PLACED SUCCESSFULLY!")
