import os, sys, time, io, torch, trimesh, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_mask_fast as mod

MESH_PATH = r"C:\Users\ctruh\Hunyuan3D-Part\XPart\data\000.glb"
OUT_AABB = r"C:\Users\ctruh\Hunyuan3D-Part\XPart\results\precomputed_aabb_000.npy"
OUT_DIR = r"C:\Users\ctruh\Hunyuan3D-Part\P3-SAM\demo\results\for_xpart"
os.makedirs(OUT_DIR, exist_ok=True)

am = mod.AutoMask(ckpt_path=None)
mesh = trimesh.load(MESH_PATH, force="mesh")
print(f"mesh: {len(mesh.faces):,} faces")

torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.autocast("cuda", dtype=torch.bfloat16):
    aabb, face_ids, mesh_out = mod.mesh_sam(
        (am.model, am.model_parallel), mesh.copy(), OUT_DIR,
        point_num=100000, prompt_num=400,
        save_mid_res=False, show_info=True,
        post_process=True, clean_mesh_flag=True, seed=42,
        prompt_bs=4,
        loop_mode="opt3", nms_mode="fast", cleanup_mode="fast",
    )
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
n_parts = len(np.unique(face_ids))
print(f"TOTAL TIME: {elapsed:.2f}s, parts: {n_parts}, aabb shape: {aabb.shape}")

np.save(OUT_AABB, aabb)
print(f"Saved aabb to {OUT_AABB}")
