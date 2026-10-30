import os
import sys
import json
import time
import uuid
import asyncio
import threading
import subprocess
import numpy as np
import trimesh
from pathlib import Path
from aiohttp import web

# Global GPU pipeline lock to prevent multiple concurrent PyTorch jobs from thrashing VRAM
pipeline_lock = threading.Lock()

@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=200)
    else:
        try:
            response = await handler(request)
        except web.HTTPException as ex:
            response = ex
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

ROOT_DIR = Path(__file__).parent.parent.resolve()
WEB_DIR = Path(__file__).parent.resolve()
STATIC_DIR = WEB_DIR / "static"
UPLOADS_DIR = ROOT_DIR / "uploads"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ASSETS_DIR = ROOT_DIR / "assets"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory jobs tracking
jobs = {}

# Python executable
PYTHON_EXE = sys.executable

async def get_presets(request):
    """Return empty list - sample models removed as requested."""
    return web.json_response({"presets": []})


async def upload_file(request):
    """Handle 3D model file upload and calculate mesh details (triangles, quads, vertices)."""
    reader = await request.multipart()
    field = await reader.next()
    if not field or not field.filename:
        return web.json_response({"error": "No file uploaded"}, status=400)
    
    ext = Path(field.filename).suffix.lower()
    if ext not in [".glb", ".obj", ".ply"]:
        return web.json_response({"error": f"Unsupported file type {ext}. Please upload .glb, .obj, or .ply"}, status=400)
        
    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{Path(field.filename).stem}_{file_id}{ext}"
    dest_path = UPLOADS_DIR / safe_name
    
    size = 0
    with open(dest_path, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)
            
    # Calculate detailed mesh statistics immediately
    mesh_stats = {
        "triangles": 0,
        "quads": 0,
        "faces": 0,
        "vertices": 0,
        "watertight": False,
        "dimensions": [0.0, 0.0, 0.0]
    }
    try:
        import trimesh
        m = trimesh.load(str(dest_path), force="mesh")
        if hasattr(m, "faces") and len(m.faces) > 0:
            mesh_stats["triangles"] = int(len(m.faces))
            mesh_stats["faces"] = int(len(m.faces))
            mesh_stats["vertices"] = int(len(m.vertices))
            mesh_stats["watertight"] = bool(m.is_watertight)
            dims = (m.bounds[1] - m.bounds[0]).tolist()
            mesh_stats["dimensions"] = [round(float(d), 3) for d in dims]
            
            # Detect quads (coplanar triangle pairs sharing an edge)
            try:
                adj = m.face_adjacency
                dots = np.sum(m.face_normals[adj[:, 0]] * m.face_normals[adj[:, 1]], axis=1)
                mesh_stats["quads"] = int(np.sum(dots > 0.999) // 2)
            except Exception:
                mesh_stats["quads"] = 0
    except Exception as e:
        print(f"Stats calculation error on upload: {e}")
            
    return web.json_response({
        "status": "success",
        "filename": safe_name,
        "original_name": field.filename,
        "path": f"/uploads/{safe_name}",
        "full_path": str(dest_path),
        "size": size,
        "stats": mesh_stats
    })


def run_pipeline_sync(job_id, mesh_path, params):
    """Runs the segmentation pipeline and Tripo conversion in a worker thread."""
    jobs[job_id]["stage"] = "Queued for GPU resources..."
    pipeline_lock.acquire()
    try:
        t_total_start = time.perf_counter()
        jobs[job_id]["status"] = "running"
        jobs[job_id]["stage"] = "Initializing GPU & Models..."
        jobs[job_id]["progress"] = 10
        jobs[job_id]["timings"] = {"step1": 0.0, "step2": 0.0, "step3": 0.0, "total": 0.0}
        
        mesh_p = Path(mesh_path)
        stem = mesh_p.stem
        out_job_dir = OUTPUTS_DIR / f"web_{stem}_{job_id}"
        out_job_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if mesh is already an assembly with multiple parts
        import trimesh
        loaded = trimesh.load(str(mesh_p))
        is_already_segmented = False
        if isinstance(loaded, trimesh.Scene) and len(loaded.geometry) > 1:
            is_already_segmented = True
            
        sys.path.insert(0, str(ROOT_DIR / "XPart"))
        from partgen.utils.mesh_utils import (
            create_tripo_assembly_scene,
            export_tripo_assembly_glb,
            explode_mesh_tripo,
            generate_tripo_viewer_html
        )
        
        step1_time = 0.0
        step2_time = 0.0
        
        if is_already_segmented:
            t2_s = time.perf_counter()
            jobs[job_id]["stage"] = "Converting pre-segmented parts to Tripo3D multi-node assembly..."
            jobs[job_id]["progress"] = 50
            
            part_meshes = []
            for name, geom in loaded.geometry.items():
                if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                    try:
                        xform = loaded.graph[name][0]
                        wg = geom.copy()
                        wg.apply_transform(xform)
                    except Exception:
                        wg = geom.copy()
                    part_meshes.append(wg)
                    
            assembly_scene = create_tripo_assembly_scene(part_meshes, auto_clean=True, auto_seal=True)
            step2_time = time.perf_counter() - t2_s
        else:
            # Step 1: P3-SAM
            t1_s = time.perf_counter()
            jobs[job_id]["stage"] = "Step 1/3: Running P3-SAM 3D feature and bounding-box estimation..."
            jobs[job_id]["progress"] = 25
            
            p3sam_script = ROOT_DIR / "P3-SAM" / "demo" / "run_p3sam_for_xpart.py"
            p3sam_out_dir = out_job_dir / "p3sam"
            p3sam_out_dir.mkdir(parents=True, exist_ok=True)
            
            cmd_p3sam = [
                PYTHON_EXE, str(p3sam_script),
                str(mesh_p), str(p3sam_out_dir), "32"
            ]
            env = os.environ.copy()
            env["TEMP"] = "C:\\Users\\7950x_RTX4090\\AppData\\Local\\Temp"
            env["TMP"] = "C:\\Users\\7950x_RTX4090\\AppData\\Local\\Temp"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["P3SAM_PROMPT_BS"] = "32"
            
            res_p3 = subprocess.run(cmd_p3sam, cwd=str(ROOT_DIR), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            step1_time = time.perf_counter() - t1_s
            jobs[job_id]["timings"]["step1"] = round(step1_time, 2)
            aabb_npy = p3sam_out_dir / f"{stem}_aabb.npy"
            
            # Check for face_ids from P3-SAM
            face_ids_cands = list(p3sam_out_dir.glob("*face_ids.npy"))
            post_cands = [p for p in face_ids_cands if "final_post" in p.name]
            best_face_ids = post_cands[0] if post_cands else (face_ids_cands[0] if face_ids_cands else None)
            
            # Step 2: Partitioning / XPart DiT
            t2_s = time.perf_counter()
            mode = params.get("mode", "tripo_exact")
            if mode == "tripo_exact" and best_face_ids and best_face_ids.exists():
                jobs[job_id]["stage"] = "Step 2/3: Partitioning input mesh into watertight Tripo3D manifold parts..."
                jobs[job_id]["progress"] = 65
                
                face_ids = np.load(best_face_ids)
                
                # Locate the corresponding mesh processed by P3-SAM
                mesh_cands = []
                comp_glb = best_face_ids.with_name(best_face_ids.name.replace("_face_ids.npy", ".glb"))
                if comp_glb.exists():
                    mesh_cands.append(comp_glb)
                mesh_cands.extend([
                    p3sam_out_dir / "p3sam_cleaned_mesh.glb",
                    p3sam_out_dir / "auto_mask_mesh_final_post.glb",
                    p3sam_out_dir / "auto_mask_mesh_final.glb",
                    p3sam_out_dir / "auto_mask_mesh_org.glb",
                    p3sam_out_dir / "auto_mask_mesh.glb",
                    mesh_p
                ])
                
                target_mesh = None
                for cand in mesh_cands:
                    if cand and cand.exists():
                        try:
                            m = trimesh.load(str(cand), force="mesh")
                            if hasattr(m, "faces") and len(m.faces) == len(face_ids):
                                target_mesh = m
                                logger.info(f"Matched segmentation mesh: {cand.name} with {len(m.faces)} faces")
                                break
                        except Exception:
                            continue
                            
                if target_mesh is None:
                    target_mesh = trimesh.load(str(mesh_p), force="mesh")
                
                # Resilient alignment: ensure face_ids dimension strictly matches target_mesh.faces
                num_faces = len(target_mesh.faces)
                if len(face_ids) != num_faces:
                    logger.warning(f"Aligning face_ids length ({len(face_ids)}) to target mesh faces ({num_faces})")
                    if len(face_ids) < num_faces:
                        padded = np.full(num_faces, -1, dtype=face_ids.dtype)
                        padded[:len(face_ids)] = face_ids
                        face_ids = padded
                    else:
                        face_ids = face_ids[:num_faces]
                        
                unique_ids = [uid for uid in np.unique(face_ids) if uid >= 0]
                raw_parts = []
                for uid in unique_ids:
                    mask = (face_ids == uid)
                    if not np.any(mask):
                        continue
                    face_idx = np.nonzero(mask)[0]
                    sub = target_mesh.submesh([face_idx], append=True)
                    if sub is not None and hasattr(sub, "faces") and len(sub.faces) > 0:
                        raw_parts.append(sub)
                        
                if not raw_parts:
                    raw_parts = [target_mesh]
                    
                assembly_scene = create_tripo_assembly_scene(raw_parts, auto_clean=True, auto_seal=True)
            else:
                jobs[job_id]["stage"] = "Step 2/3: Running X-Part DiT diffusion & DiffDMC CUDA GPU marching cubes..."
                jobs[job_id]["progress"] = 55
                
                xpart_script = ROOT_DIR / "XPart" / "demo.py"
                cmd_xpart = [
                    PYTHON_EXE, str(xpart_script),
                    "--config", str(ROOT_DIR / "XPart" / "partgen" / "config" / "infer.yaml"),
                    "--mesh_path", str(mesh_p),
                    "--save_dir", str(out_job_dir),
                    "--mc_algo", params.get("mc_algo", "dmc"),
                    "--num_inference_steps", str(params.get("steps", 50)),
                    "--octree_resolution", str(params.get("resolution", 512))
                ]
                if aabb_npy and aabb_npy.exists():
                    cmd_xpart.extend(["--aabb_path", str(aabb_npy)])
                    
                env["XPART_DIFFUSION_PART_CHUNK"] = "16"
                res_xpart = subprocess.run(cmd_xpart, cwd=str(ROOT_DIR), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
                
                candidates = list(out_job_dir.rglob("train_cfg_*_dit_boxgpt_*.glb"))
                if candidates:
                    generated_glb = candidates[0]
                    loaded_gen = trimesh.load(str(generated_glb))
                    if isinstance(loaded_gen, trimesh.Scene):
                        part_meshes = [g for g in loaded_gen.geometry.values() if hasattr(g, "faces") and len(g.faces) > 0]
                    elif isinstance(loaded_gen, trimesh.Trimesh):
                        part_meshes = [loaded_gen] if hasattr(loaded_gen, "faces") and len(loaded_gen.faces) > 0 else []
                    else:
                        part_meshes = []
                    if not part_meshes:
                        part_meshes = [trimesh.load(str(mesh_p), force="mesh")]
                    assembly_scene = create_tripo_assembly_scene(part_meshes, auto_clean=True, auto_seal=True)
                else:
                    raise RuntimeError(f"XPart failed to generate output. Stderr: {res_xpart.stderr}")
            
            step2_time = time.perf_counter() - t2_s
            jobs[job_id]["timings"]["step2"] = round(step2_time, 2)

        # Step 3: Tripo3D Multi-Node Assembly & Wave Explosion Export
        t3_s = time.perf_counter()
        jobs[job_id]["stage"] = "Step 3/3: Exporting Tripo3D multi-node assembly & exact explosion wave..."
        jobs[job_id]["progress"] = 85
        
        assembled_glb_name = f"{stem}_assembly.glb"
        assembled_glb_path = out_job_dir / assembled_glb_name
        export_tripo_assembly_glb(assembly_scene, assembled_glb_path)

        exploded_glb_name = f"{stem}_exploded.glb"
        exploded_glb_path = out_job_dir / exploded_glb_name
        exploded_scene = explode_mesh_tripo(assembly_scene, explosion_scale=params.get("explosion_scale", 0.35))
        export_tripo_assembly_glb(exploded_scene, exploded_glb_path)

        viewer_html_name = f"{stem}_viewer.html"
        viewer_html_path = out_job_dir / viewer_html_name
        generate_tripo_viewer_html(
            glb_filename=assembled_glb_name,
            output_html_path=viewer_html_path,
            model_title=f"{stem} (Tripo3D Matching Engine)"
        )
        step3_time = time.perf_counter() - t3_s
        total_time = time.perf_counter() - t_total_start
        
        jobs[job_id]["timings"] = {
            "step1": round(step1_time, 2),
            "step2": round(step2_time, 2),
            "step3": round(step3_time, 2),
            "total": round(total_time, 2),
            "formatted": f"{total_time:.2f}s"
        }

        # Calculate rich details for each segmented part
        part_details = []
        total_faces_count = 0
        total_verts_count = 0
        total_quads_count = 0
        for idx, (name, geom) in enumerate(assembly_scene.geometry.items()):
            if hasattr(geom, "faces") and len(geom.faces) > 0:
                total_faces_count += len(geom.faces)
                total_verts_count += len(geom.vertices)
                dims = (geom.bounds[1] - geom.bounds[0]).tolist()
                
                # Check quads
                p_quads = 0
                try:
                    p_adj = geom.face_adjacency
                    p_dots = np.sum(geom.face_normals[p_adj[:, 0]] * geom.face_normals[p_adj[:, 1]], axis=1)
                    p_quads = int(np.sum(p_dots > 0.999) // 2)
                    total_quads_count += p_quads
                except Exception:
                    pass
                
                # Color hex
                color_hex = "#6366f1"
                try:
                    if hasattr(geom.visual, "face_colors") and len(geom.visual.face_colors) > 0:
                        fc = geom.visual.face_colors[0][:3]
                        color_hex = f"#{int(fc[0]):02x}{int(fc[1]):02x}{int(fc[2]):02x}"
                except Exception:
                    pass
                    
                part_details.append({
                    "id": idx,
                    "name": name,
                    "color": color_hex,
                    "faces": int(len(geom.faces)),
                    "triangles": int(len(geom.faces)),
                    "quads": p_quads,
                    "vertices": int(len(geom.vertices)),
                    "watertight": bool(geom.is_watertight),
                    "dimensions": [round(float(d), 3) for d in dims],
                    "volume": round(float(geom.volume), 5) if geom.is_volume else 0.0,
                    "area": round(float(geom.area), 4) if hasattr(geom, "area") else 0.0
                })

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["stage"] = "Pipeline Completed!"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = {
            "assembly_glb": f"/outputs/{out_job_dir.relative_to(OUTPUTS_DIR).as_posix()}/{assembled_glb_name}",
            "exploded_glb": f"/outputs/{out_job_dir.relative_to(OUTPUTS_DIR).as_posix()}/{exploded_glb_name}",
            "viewer_html": f"/outputs/{out_job_dir.relative_to(OUTPUTS_DIR).as_posix()}/{viewer_html_name}",
            "parts_count": len(assembly_scene.geometry),
            "timing": jobs[job_id]["timings"],
            "mesh_details": {
                "triangles": total_faces_count,
                "quads": total_quads_count,
                "faces": total_faces_count,
                "vertices": total_verts_count,
                "parts_count": len(part_details),
                "watertight": all(p["watertight"] for p in part_details)
            },
            "parts": part_details
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        pipeline_lock.release()


async def start_segmentation(request):
    """Trigger the pipeline segmentation job."""
    data = await request.json()
    mesh_path = data.get("mesh_path")
    if not mesh_path:
        return web.json_response({"error": "mesh_path is required"}, status=400)
    
    # Resolve relative paths
    if mesh_path.startswith("/assets/"):
        full_path = ASSETS_DIR / mesh_path.replace("/assets/", "")
    elif mesh_path.startswith("/outputs/"):
        full_path = OUTPUTS_DIR / mesh_path.replace("/outputs/", "")
    elif mesh_path.startswith("/uploads/"):
        full_path = UPLOADS_DIR / mesh_path.replace("/uploads/", "")
    else:
        full_path = Path(mesh_path)
        
    if not full_path.exists():
        return web.json_response({"error": f"File not found: {mesh_path}"}, status=404)

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "stage": "Job queued in pipeline...",
        "progress": 0,
        "mesh_path": str(full_path),
        "created_at": time.time()
    }

    params = {
        "mode": data.get("mode", "xpart_dit"),
        "steps": int(data.get("steps", 50)),
        "resolution": int(data.get("resolution", 512)),
        "mc_algo": data.get("mc_algo", "dmc"),
        "explosion_scale": float(data.get("explosion_scale", 0.35))
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_pipeline_sync, job_id, full_path, params)

    return web.json_response({"job_id": job_id, "status": "queued"})


async def get_job_status(request):
    """Poll job status and output paths."""
    job_id = request.match_info.get("job_id")
    if job_id not in jobs:
        return web.json_response({"error": "Job not found"}, status=404)
    return web.json_response(jobs[job_id])


async def index_handler(request):
    """Serve index.html."""
    return web.FileResponse(STATIC_DIR / "index.html")


def create_app():
    app = web.Application(client_max_size=200 * 1024 * 1024, middlewares=[cors_middleware]) # 200MB upload limit
    
    # Routes
    app.router.add_get("/", index_handler)
    app.router.add_get("/api/presets", get_presets)
    app.router.add_post("/api/upload", upload_file)
    app.router.add_post("/api/segment", start_segmentation)
    app.router.add_get("/api/status/{job_id}", get_job_status)
    
    # Static files
    app.router.add_static("/static/", STATIC_DIR)
    app.router.add_static("/assets/", ASSETS_DIR)
    app.router.add_static("/outputs/", OUTPUTS_DIR)
    app.router.add_static("/uploads/", UPLOADS_DIR)
        
    return app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"==================================================")
    print(f"  Tripo3D Matching Web Studio & Segmentation Server")
    print(f"  Running at: http://localhost:{port}")
    print(f"==================================================")
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)
