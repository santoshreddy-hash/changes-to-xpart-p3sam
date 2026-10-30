import os
import sys
import json
import time
import uuid
import asyncio
import subprocess
from pathlib import Path
from aiohttp import web

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
    """Return available pre-existing 3D models and previously segmented outputs."""
    presets = []
    
    # 1. Existing assets
    if ASSETS_DIR.exists():
        for glb in ASSETS_DIR.glob("*.glb"):
            presets.append({
                "id": glb.stem,
                "name": glb.name,
                "path": f"/assets/{glb.name}",
                "type": "raw_asset"
            })
            
    # 2. Existing Tripo-converted assemblies in outputs
    for assembly_glb in OUTPUTS_DIR.glob("**/*_assembly.glb"):
        rel_path = assembly_glb.relative_to(OUTPUTS_DIR).as_posix()
        presets.append({
            "id": assembly_glb.stem,
            "name": f"Segmented: {assembly_glb.stem.replace('_assembly', '')}",
            "path": f"/outputs/{rel_path}",
            "type": "segmented_assembly"
        })
        
    return web.json_response({"presets": presets})


async def upload_file(request):
    """Handle 3D model file upload."""
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
            
    return web.json_response({
        "status": "success",
        "filename": safe_name,
        "path": f"/uploads/{safe_name}",
        "full_path": str(dest_path),
        "size": size
    })


def run_pipeline_sync(job_id, mesh_path, params):
    """Runs the segmentation pipeline and Tripo conversion in a worker thread."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["stage"] = "Initializing GPU & Models..."
        jobs[job_id]["progress"] = 10
        
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
        
        if is_already_segmented:
            jobs[job_id]["stage"] = "Converting pre-segmented parts to Tripo3D multi-node assembly..."
            jobs[job_id]["progress"] = 50
            time.sleep(0.5)
            
            part_meshes = []
            for name, geom in loaded.geometry.items():
                if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                    xform = loaded.graph[name][0]
                    wg = geom.copy()
                    wg.apply_transform(xform)
                    part_meshes.append(wg)
                    
            assembly_scene = create_tripo_assembly_scene(part_meshes)
        else:
            # Run segmentation pipeline
            jobs[job_id]["stage"] = "Step 1/3: Running P3-SAM fast 3D part bounding-box estimation..."
            jobs[job_id]["progress"] = 25
            
            p3sam_script = ROOT_DIR / "P3-SAM" / "demo" / "run_p3sam_for_xpart.py"
            p3sam_out_dir = out_job_dir / "p3sam"
            p3sam_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Run P3-SAM
            cmd_p3sam = [
                PYTHON_EXE, str(p3sam_script),
                str(mesh_p), str(p3sam_out_dir), "4"
            ]
            env = os.environ.copy()
            env["TEMP"] = "C:\\Users\\7950x_RTX4090\\AppData\\Local\\Temp"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            
            res_p3 = subprocess.run(cmd_p3sam, cwd=str(ROOT_DIR), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            aabb_npy = p3sam_out_dir / f"{stem}_aabb.npy"
            
            # Check for face_ids from P3-SAM
            face_ids_cands = list(p3sam_out_dir.glob("*face_ids.npy"))
            post_cands = [p for p in face_ids_cands if "final_post" in p.name]
            best_face_ids = post_cands[0] if post_cands else (face_ids_cands[0] if face_ids_cands else None)
            
            mode = params.get("mode", "tripo_exact")
            if mode == "tripo_exact" and best_face_ids and best_face_ids.exists():
                jobs[job_id]["stage"] = "Step 2/3: Partitioning input mesh into watertight Tripo3D manifold parts..."
                jobs[job_id]["progress"] = 60
                
                face_ids = np.load(best_face_ids)
                input_mesh = trimesh.load(str(mesh_p), force="mesh")
                unique_ids = [uid for uid in np.unique(face_ids) if uid >= 0]
                raw_parts = [input_mesh.submesh([face_ids == uid], append=True) for uid in unique_ids]
                
                # Automatically deduplicate overlapping bounding boxes, merge redundant fragments, and cap open cut holes
                assembly_scene = create_tripo_assembly_scene(raw_parts, auto_clean=True, auto_seal=True)
            else:
                jobs[job_id]["stage"] = "Step 2/3: Running X-Part DiT diffusion & DiffDMC CUDA GPU marching cubes..."
                jobs[job_id]["progress"] = 50
                
                # Run XPart demo.py
                xpart_script = ROOT_DIR / "XPart" / "demo.py"
                cmd_xpart = [
                    PYTHON_EXE, str(xpart_script),
                    "--mesh_path", str(mesh_p),
                    "--save_dir", str(out_job_dir),
                    "--mc_algo", params.get("mc_algo", "dmc"),
                    "--num_inference_steps", str(params.get("steps", 50)),
                    "--octree_resolution", str(params.get("resolution", 512))
                ]
                if aabb_npy and aabb_npy.exists():
                    cmd_xpart.extend(["--aabb_path", str(aabb_npy)])
                    
                res_xpart = subprocess.run(cmd_xpart, cwd=str(ROOT_DIR), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
                print("XPart output:\n", res_xpart.stdout[-1000:])
                
                # Find generated glb
                candidates = list(out_job_dir.glob("train_cfg_*_dit_boxgpt_*.glb"))
                if candidates:
                    generated_glb = candidates[0]
                    loaded_gen = trimesh.load(str(generated_glb))
                    part_meshes = [g for g in loaded_gen.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
                    assembly_scene = create_tripo_assembly_scene(part_meshes, auto_clean=True, auto_seal=True)
                else:
                    raise RuntimeError(f"XPart failed to generate output. Stderr: {res_xpart.stderr}")

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

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["stage"] = "Pipeline Completed!"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = {
            "assembly_glb": f"/outputs/{out_job_dir.relative_to(OUTPUTS_DIR).as_posix()}/{assembled_glb_name}",
            "exploded_glb": f"/outputs/{out_job_dir.relative_to(OUTPUTS_DIR).as_posix()}/{exploded_glb_name}",
            "viewer_html": f"/outputs/{out_job_dir.relative_to(OUTPUTS_DIR).as_posix()}/{viewer_html_name}",
            "parts_count": len(assembly_scene.geometry)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


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
