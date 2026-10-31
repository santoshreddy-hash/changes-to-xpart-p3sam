import os
import sys
import json
import time
import uuid
import asyncio
import threading
import subprocess
import logging
import numpy as np
import trimesh
from pathlib import Path
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tripo3d_server")

# Global GPU pipeline lock and process tracker
pipeline_lock = threading.Lock()
active_subprocesses = set()
active_subprocesses_lock = threading.Lock()

def register_subprocess(proc):
    with active_subprocesses_lock:
        active_subprocesses.add(proc)

def unregister_subprocess(proc):
    with active_subprocesses_lock:
        active_subprocesses.discard(proc)

def kill_process_tree(pid):
    """Forcefully kill a process and all its children on Windows."""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
    except Exception as e:
        logger.warning(f"Failed to taskkill PID {pid}: {e}")

def free_all_gpu_resources_and_abort():
    """Immediately stops all running pipeline tasks, terminates child subprocesses, and flushes GPU VRAM."""
    logger.info("Freeing GPU resources and terminating conflicting background tasks...")
    # Kill tracked pipeline subprocesses
    with active_subprocesses_lock:
        for proc in list(active_subprocesses):
            try:
                kill_process_tree(proc.pid)
            except Exception as e:
                logger.warning(f"Error terminating child proc {proc.pid}: {e}")
        active_subprocesses.clear()

    # Preempt any queued/running jobs in jobs dictionary
    for jid, job_info in list(jobs.items()):
        if job_info.get("status") in ["running", "queued"]:
            job_info["status"] = "cancelled"
            job_info["stage"] = "Cancelled: Preempted to make GPU 100% available for new pipeline run."

    # Flush PyTorch CUDA cache and run garbage collection
    try:
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        logger.info("PyTorch CUDA cache flushed. RTX 4090 GPU is completely dedicated.")
    except Exception as e:
        logger.warning(f"CUDA cache flush notice: {e}")

def run_command_tracked(cmd, cwd, env, job_id=None):
    """Execute a subprocess while tracking it for instant preemption/cancellation."""
    if job_id and jobs.get(job_id, {}).get("status") == "cancelled":
        raise RuntimeError(f"Job {job_id} was preempted before execution.")
        
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    register_subprocess(proc)
    try:
        stdout, stderr = proc.communicate()
        if job_id and jobs.get(job_id, {}).get("status") == "cancelled":
            raise RuntimeError(f"Job {job_id} was preempted during execution.")
        return proc.returncode, stdout, stderr
    finally:
        unregister_subprocess(proc)

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
    """Runs the segmentation pipeline with P3-SAM first, followed by the selected engine."""
    jobs[job_id]["stage"] = "Acquiring dedicated GPU resources..."
    pipeline_lock.acquire()
    try:
        if jobs.get(job_id, {}).get("status") == "cancelled":
            logger.info(f"Job {job_id} cancelled before GPU execution started.")
            return

        t_total_start = time.perf_counter()
        jobs[job_id]["status"] = "running"
        jobs[job_id]["stage"] = "Initializing GPU & Models for input..."
        jobs[job_id]["progress"] = 10
        jobs[job_id]["timings"] = {"step1": 0.0, "step2": 0.0, "step3": 0.0, "total": 0.0}
        
        mesh_p = Path(mesh_path)
        stem = mesh_p.stem
        out_job_dir = OUTPUTS_DIR / f"web_{stem}_{job_id}"
        out_job_dir.mkdir(parents=True, exist_ok=True)
        
        sys.path.insert(0, str(ROOT_DIR / "XPart"))
        from partgen.utils.mesh_utils import (
            create_tripo_assembly_scene,
            export_tripo_assembly_glb,
            explode_mesh_tripo,
            generate_tripo_viewer_html
        )

        # -------------------------------------------------------------
        # Input Normalization: whatever the input is (single mesh or multi-node scene),
        # unify geometries so P3-SAM receives a coherent, complete 3D object.
        # -------------------------------------------------------------
        import trimesh
        loaded = trimesh.load(str(mesh_p))
        p3sam_input_path = out_job_dir / f"{stem}_p3sam_input.glb"
        
        if isinstance(loaded, trimesh.Scene):
            logger.info(f"Input is a Scene with {len(loaded.geometry)} geometries. Unifying for P3-SAM...")
            geoms_to_merge = []
            for g_name, geom in loaded.geometry.items():
                if hasattr(geom, "faces") and len(geom.faces) > 0:
                    try:
                        xform = loaded.graph[g_name][0]
                        wg = geom.copy()
                        wg.apply_transform(xform)
                    except Exception:
                        wg = geom.copy()
                    geoms_to_merge.append(wg)
            if geoms_to_merge:
                input_mesh = trimesh.util.concatenate(geoms_to_merge)
                input_mesh.export(str(p3sam_input_path))
            else:
                input_mesh = trimesh.load(str(mesh_p), force="mesh")
                input_mesh.export(str(p3sam_input_path))
        else:
            input_mesh = loaded
            input_mesh.export(str(p3sam_input_path))
            
        mode = params.get("mode", "tripo_exact")
        logger.info(f"Starting Job {job_id}: Selected Pipeline Engine = '{mode}' on {mesh_p.name}")

        # -------------------------------------------------------------
        # STEP 1: P3-SAM Optimized 3D Feature & Bounding-Box Estimation
        # (Always executes 1st for all inputs as requested by user)
        # -------------------------------------------------------------
        t1_s = time.perf_counter()
        jobs[job_id]["stage"] = "Step 1/3: Running P3-SAM GPU feature & 3D bounding-box estimation..."
        jobs[job_id]["progress"] = 25
        
        p3sam_script = ROOT_DIR / "P3-SAM" / "demo" / "auto_mask_fast.py"
        p3sam_out_dir = out_job_dir / "p3sam"
        p3sam_out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd_p3sam = [
            PYTHON_EXE, "-u", str(p3sam_script),
            "--mesh_path", str(p3sam_input_path),
            "--output_path", str(p3sam_out_dir),
            "--parallel", "0",
            "--post_process", "1",
            "--prompt_bs", "8",
            "--loop_mode", "opt3",
            "--nms_mode", "fast",
            "--cleanup_mode", "fast",
            "--bf16", "1",
            "--save_only_glbs", "1",
            "--save_face_ids", str(p3sam_out_dir / f"{stem}_face_ids.npy")
        ]
        env = os.environ.copy()
        env["TEMP"] = "C:\\Users\\7950x_RTX4090\\AppData\\Local\\Temp"
        env["TMP"] = "C:\\Users\\7950x_RTX4090\\AppData\\Local\\Temp"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["P3SAM_PROMPT_BS"] = "8"
        
        ret_p3, out_p3, err_p3 = run_command_tracked(cmd_p3sam, cwd=ROOT_DIR, env=env, job_id=job_id)
        if ret_p3 != 0:
            raise RuntimeError(f"P3-SAM execution failed (code {ret_p3}): {err_p3 or out_p3}")
            
        step1_time = time.perf_counter() - t1_s
        jobs[job_id]["timings"]["step1"] = round(step1_time, 2)
        logger.info(f"P3-SAM completed in {step1_time:.2f}s")
        
        # Locate generated AABB boxes and face assignment IDs from P3-SAM
        aabb_npy = None
        aabb_cands = [
            p3sam_out_dir / f"{stem}_p3sam_input_aabb.npy",
            p3sam_out_dir / f"{stem}_aabb.npy",
            p3sam_out_dir / "auto_mask_mesh_final_post_aabb.npy"
        ]
        for ac in aabb_cands:
            if ac.exists():
                aabb_npy = ac
                break
        if not aabb_npy:
            all_a = list(p3sam_out_dir.glob("*aabb*.npy"))
            if all_a:
                aabb_npy = all_a[0]
                
        face_ids_cands = list(p3sam_out_dir.glob("*face_ids.npy"))
        post_cands = [p for p in face_ids_cands if "final_post" in p.name]
        best_face_ids = post_cands[0] if post_cands else (face_ids_cands[0] if face_ids_cands else None)

        # -------------------------------------------------------------
        # STEP 2: Selected Pipeline Execution
        # (Whichever pipeline is selected in the dropdown ONLY runs here)
        # -------------------------------------------------------------
        t2_s = time.perf_counter()
        
        if mode == "tripo_exact" and best_face_ids and best_face_ids.exists():
            # PIPELINE 1: Tripo Exact Partitioning (~10.8s Total)
            jobs[job_id]["stage"] = "Step 2/3: Partitioning input mesh into exact Tripo3D parts (preserving textures)..."
            jobs[job_id]["progress"] = 65
            
            face_ids = np.load(best_face_ids)
            target_mesh = input_mesh
            should_preserve = params.get("preserve_materials", True)
            
            num_faces = len(target_mesh.faces)
            if len(face_ids) != num_faces:
                logger.info(f"Projecting {len(face_ids)} face_ids to {num_faces} faces via cKDTree...")
                try:
                    from scipy.spatial import cKDTree
                    cand_cleaned = p3sam_out_dir / "p3sam_cleaned_mesh.glb"
                    if cand_cleaned.exists():
                        cm = trimesh.load(str(cand_cleaned), force="mesh")
                        src_centers = cm.triangles.mean(axis=1) if len(cm.faces) == len(face_ids) else target_mesh.triangles.mean(axis=1)[:len(face_ids)]
                    else:
                        src_centers = target_mesh.triangles.mean(axis=1)[:len(face_ids)]
                    dst_centers = target_mesh.triangles.mean(axis=1)
                    tree = cKDTree(src_centers)
                    _, nearest = tree.query(dst_centers)
                    face_ids = face_ids[nearest]
                except Exception as kerr:
                    logger.warning(f"KDTree projection fallback: {kerr}")
                    if len(face_ids) < num_faces:
                        padded = np.full(num_faces, -1, dtype=face_ids.dtype)
                        padded[:len(face_ids)] = face_ids
                        face_ids = padded
                    else:
                        face_ids = face_ids[:num_faces]
                        
            # Fill unassigned/boundary faces to eliminate gaps
            unassigned = (face_ids < 0)
            if np.any(unassigned) and np.any(~unassigned):
                try:
                    from scipy.spatial import cKDTree
                    centers = target_mesh.triangles.mean(axis=1)
                    valid_idx = np.where(~unassigned)[0]
                    unassigned_idx = np.where(unassigned)[0]
                    tree = cKDTree(centers[valid_idx])
                    _, nearest = tree.query(centers[unassigned_idx])
                    face_ids[unassigned_idx] = face_ids[valid_idx[nearest]]
                except Exception as err:
                    logger.warning(f"Unassigned face fill error: {err}")
                    
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
                
            assembly_scene = create_tripo_assembly_scene(
                raw_parts,
                auto_clean=False,
                auto_seal=False,
                preserve_materials=should_preserve
            )
        else:
            # PIPELINES 2, 3, 4: Generative X-Part DiT Engine (Fast, Lossless GPU, or Baseline)
            m_desc = "X-Part DiT diffusion & DiffDMC CUDA GPU marching cubes"
            if mode == "xpart_fast":
                m_desc = f"X-Part Fast DiT ({params.get('steps', 20)} steps, {params.get('resolution', 384)} res, GPU DiffDMC)"
            elif mode == "xpart_lossless":
                m_desc = f"X-Part Lossless GPU ({params.get('steps', 50)} steps, {params.get('resolution', 512)} res, GPU DiffDMC)"
            elif mode == "xpart_baseline":
                m_desc = f"Original X-Part Baseline ({params.get('steps', 50)} steps, {params.get('resolution', 512)} res, CPU MC)"
                
            jobs[job_id]["stage"] = f"Step 2/3: Running {m_desc}..."
            jobs[job_id]["progress"] = 55
            
            xpart_script = ROOT_DIR / "XPart" / "demo.py"
            cmd_xpart = [
                PYTHON_EXE, "-u", str(xpart_script),
                "--config", str(ROOT_DIR / "XPart" / "partgen" / "config" / "infer.yaml"),
                "--mesh_path", str(p3sam_input_path),
                "--save_dir", str(out_job_dir),
                "--mc_algo", str(params.get("mc_algo", "dmc")),
                "--num_inference_steps", str(params.get("steps", 50)),
                "--octree_resolution", str(params.get("resolution", 512))
            ]
            if aabb_npy and aabb_npy.exists():
                cmd_xpart.extend(["--aabb_path", str(aabb_npy)])
                logger.info(f"Passing precomputed P3-SAM bounding boxes from: {aabb_npy.name}")
                
            env["XPART_DIFFUSION_PART_CHUNK"] = str(params.get("chunk_size", 16))
            ret_x, out_x, err_x = run_command_tracked(cmd_xpart, cwd=ROOT_DIR, env=env, job_id=job_id)
            if ret_x != 0:
                raise RuntimeError(f"X-Part execution failed (code {ret_x}): {err_x or out_x}")
                
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
                    part_meshes = [trimesh.load(str(p3sam_input_path), force="mesh")]
                assembly_scene = create_tripo_assembly_scene(part_meshes, auto_clean=True, auto_seal=False, preserve_materials=False)
            else:
                raise RuntimeError(f"XPart failed to generate output mesh in {out_job_dir}. Logs: {err_x or out_x}")
        
        step2_time = time.perf_counter() - t2_s
        jobs[job_id]["timings"]["step2"] = round(step2_time, 2)
        logger.info(f"Step 2 ({mode}) completed in {step2_time:.2f}s")


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
    try:
        data = await request.json()
    except Exception as ex:
        try:
            raw = await request.text()
            data = json.loads(raw)
        except Exception:
            return web.json_response({"error": f"Invalid JSON payload: {ex}"}, status=400)
            
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

    # Immediately abort conflicting jobs, terminate subprocesses, and flush GPU VRAM
    free_all_gpu_resources_and_abort()

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "stage": "Job queued in pipeline...",
        "progress": 0,
        "mesh_path": str(full_path),
        "created_at": time.time()
    }

    mode = data.get("mode", "tripo_exact")
    if mode == "xpart_fast":
        default_steps = 20
        default_res = 384
        default_mc = "dmc"
        default_chunk = 16
        default_preserve = False
    elif mode == "xpart_lossless":
        default_steps = 50
        default_res = 512
        default_mc = "dmc"
        default_chunk = 16
        default_preserve = False
    elif mode == "xpart_baseline":
        default_steps = 50
        default_res = 512
        default_mc = "mc"
        default_chunk = 4
        default_preserve = False
    else:
        mode = "tripo_exact"
        default_steps = 50
        default_res = 512
        default_mc = "dmc"
        default_chunk = 16
        default_preserve = True

    params = {
        "mode": mode,
        "steps": int(data.get("steps", default_steps)),
        "resolution": int(data.get("resolution", default_res)),
        "mc_algo": str(data.get("mc_algo", default_mc)),
        "chunk_size": int(data.get("chunk_size", default_chunk)),
        "explosion_scale": float(data.get("explosion_scale", 0.35)),
        "preserve_materials": bool(data.get("preserve_materials", default_preserve))
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)), help="Port to listen on (default: 8080 or PORT env var)")
    args, _ = parser.parse_known_args()
    port = args.port
    print(f"==================================================")
    print(f"  Tripo3D Matching Web Studio & Segmentation Server")
    print(f"  Running at: http://localhost:{port}")
    print(f"==================================================")
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)
