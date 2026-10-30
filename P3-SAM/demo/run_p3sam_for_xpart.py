import os, sys, time, io, torch, trimesh, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_mask_fast as mod

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
default_mesh = os.path.join(ROOT_DIR, "assets", "orange_astronaut_turbo.glb")
mesh_path = sys.argv[1] if len(sys.argv) > 1 else default_mesh

basename = os.path.splitext(os.path.basename(mesh_path))[0]
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT_DIR, "outputs", f"p3sam_{basename}")
os.makedirs(out_dir, exist_ok=True)
out_aabb = os.path.join(out_dir, f"{basename}_aabb.npy")

print(f"Loading mesh: {mesh_path}")
mesh = trimesh.load(mesh_path, force="mesh")
print(f"Mesh loaded: {len(mesh.faces):,} faces")

am = mod.AutoMask(ckpt_path=None)

prompt_bs = int(os.environ.get("P3SAM_PROMPT_BS", sys.argv[3] if len(sys.argv) > 3 else "4"))
print(f"Running P3-SAM with prompt_bs = {prompt_bs}")

torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.autocast("cuda", dtype=torch.bfloat16):
    aabb, face_ids, mesh_out = mod.mesh_sam(
        (am.model, am.model), mesh.copy(), out_dir,
        point_num=100000, prompt_num=400,
        save_mid_res=True, show_info=True,
        post_process=True, clean_mesh_flag=True, seed=42,
        prompt_bs=prompt_bs,
        loop_mode="opt3", nms_mode="fast", cleanup_mode="fast",
    )
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
valid_parts = len(np.unique(face_ids[face_ids >= 0])) if np.any(face_ids >= 0) else len(np.unique(face_ids))
np.save(out_aabb, aabb)
print(f"Saved aabb to {out_aabb}")
mesh_out_glb = os.path.join(out_dir, "p3sam_cleaned_mesh.glb")
mesh_out.export(mesh_out_glb)
print(f"Saved cleaned mesh to {mesh_out_glb}")
print(f"Results saved in {out_dir}")

