r"""
Script to package clean, self-contained code zip files for all 4 pipelines:
1. pipeline_1_clean_parts_fast.zip
2. pipeline_2_clean_parts_precise.zip
3. pipeline_3_benchmark_suite.zip
4. pipeline_4_tripo_exact.zip
5. all_4_pipelines_code.zip

Saved directly in the workspace root: C:\Users\7950x_RTX4090\Desktop\part-segmentation
"""

import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Define helper to add files to a zip
def add_file(zf, file_path, arcname=None):
    fp = Path(file_path)
    if not fp.exists():
        print(f"Warning: File not found: {fp}")
        return
    if arcname is None:
        arcname = fp.relative_to(ROOT_DIR).as_posix()
    zf.write(str(fp), arcname)

def add_dir(zf, dir_path, arc_prefix=None, filter_exts=None, exclude_dirs=None):
    dp = Path(dir_path)
    if not dp.exists():
        print(f"Warning: Dir not found: {dp}")
        return
    if exclude_dirs is None:
        exclude_dirs = {"__pycache__", ".git", ".pytest_cache", ".idea", ".vscode"}
    for root, dirs, files in os.walk(str(dp)):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if filter_exts and not any(f.endswith(ext) for ext in filter_exts):
                continue
            full_path = Path(root) / f
            rel = full_path.relative_to(ROOT_DIR).as_posix()
            if arc_prefix:
                rel = f"{arc_prefix}/{full_path.relative_to(dp).as_posix()}"
            zf.write(str(full_path), rel)

# Readmes for each pipeline
README_P1 = """# Pipeline 1: Clean Parts Fast (In-Memory GPU)

## Overview
High-speed production segmentation pipeline designed for real-time asset processing.
- **Latency**: ~1-3s Total on NVIDIA RTX 4090.
- **Architecture**: In-memory P3-SAM with 1-pass Test-Time Augmentation (TTA), prompt batching (BS=8), bf16 mixed precision.
- **Post-Processing**: Connected component island pruning, area-weighted renumbering, instant submesh extraction.

## Key Code Files
- `partseg/config.py`: Preset `fast` configuration (1-pass TTA, high-speed filtering).
- `partseg/p3sam.py`: Warm in-memory P3-SAM model interface and inference wrapper.
- `partseg/pipeline.py`: Main `segment_mesh()` orchestration.
- `partseg/polish.py`: Boundary cleanup and connected component clustering.
- `P3-SAM/demo/auto_mask_fast.py`: Optimized vectorized P3-SAM engine.
- `web_app/server.py` & `web_app/static/`: Interactive Web Studio integration.

## Usage
### CLI:
```bash
partseg input_mesh.glb --preset fast --out-dir outputs/fast_run
```
### Python API:
```python
import trimesh
from partseg import SegmentConfig, segment_mesh

mesh = trimesh.load("input_mesh.glb", force="mesh")
cfg = SegmentConfig.preset("fast")
result = segment_mesh(mesh, cfg)
result.export_parts("outputs/fast_run")
```
### Web UI:
Select `⚡ Clean Parts Fast (~1-3s)` and click **Start Segmentation**.
"""

README_P2 = """# Pipeline 2: Clean Parts Precise (Production Quality / final_glbs)

## Overview
High-precision segmentation pipeline that produced the high-fidelity results in `outputs/final_glbs`.
- **Latency**: ~5-8s Total on NVIDIA RTX 4090.
- **Architecture**: 16-pass Test-Time Augmentation (TTA), multi-prompt voting across rotational perturbations.
- **Post-Processing**: α-expansion graph-cut boundary energy minimization, dihedral crease snapping, sub-element watertight planar cap sealing.

## Key Code Files
- `partseg/config.py`: Preset `precise` configuration (16-pass TTA, graph cuts enabled).
- `partseg/polish.py`: Graph-cut boundary smoothing (`refine_boundaries_graph_cut`) and crease alignment (`snap_to_creases`).
- `partseg/seal.py`: Watertight planar cap boundary sealing for sliced submeshes.
- `run_combined_all_steps.py`: Standalone script reproducing the multi-step quality pipeline.
- `benchmark_graph_cut.py` & `benchmark_second_pass.py`: Validation suites.

## Usage
### CLI:
```bash
partseg input_mesh.glb --preset precise --out-dir outputs/precise_run
```
### Standalone Runner:
```bash
python run_combined_all_steps.py
```
### Web UI:
Select `🎯 Clean Parts Precise (~5-8s)` and click **Start Segmentation**.
"""

README_P3 = """# Pipeline 3: Benchmark 3-Way Suite (Base vs Enh vs Final / final_pipeline_suite)

## Overview
Quantitative and visual comparative benchmark suite that generated `outputs/final_pipeline_suite`.
Runs 3 engines side-by-side on the exact same input mesh:
1. **Baseline Vanilla P3-SAM**: Raw original P3-SAM with default multi-masking.
2. **Enhanced P3-SAM**: Fast vectorized inference + 16-pass voting.
3. **Final Regularized PartSeg**: Full pipeline with graph-cut smoothing, crease snapping, and hole sealing.

Computes:
- Dihedral boundary crease alignment percentage.
- Disconnected component / island count.
- Boundary perimeter length ratio.
- Latency benchmarks.
- Side-by-side rendered comparison cards and multi-mesh grid.

## Key Code Files
- `run_multimesh_suite.py` & `multimesh_suite_base.py`: Multi-mesh automated benchmark runner.
- `benchmarks/run_suite.py`: Core benchmark harness.
- `benchmarks/eval_parts.py`: Quantitative boundary metrics computation.
- `build_complete_showcase.py` & `build_master_multimesh_grid.py`: HTML & PNG visual grid generators.
- `scripts/generate_pdf_report.py`: Comprehensive analytical PDF report generator.
- `web_app/server.py` (`mode == "benchmark_suite"`): Interactive in-web 3-way evaluation.

## Usage
### CLI:
```bash
python run_multimesh_suite.py
```
### Web UI:
Select `📊 Benchmark 3-Way Suite (Base/Enh/Final)` and click **Start Segmentation**.
Outputs `baseline.glb`, `enhanced.glb`, `final_parts.glb`, and live metrics.
"""

README_P4 = """# Pipeline 4: Tripo Exact Partitioning (In-Memory GPU, Original Textures)

## Overview
High-speed exact mesh partitioning pipeline that preserves 100% of authentic PBR paint, vertex colors, UV maps, and texture materials.
- **Latency**: ~2-4s Total on NVIDIA RTX 4090.
- **Architecture**: In-memory warm P3-SAM GPU model (`get_warm_model()`).
- **Tripo3D Matching**: Converts segmented components into a hierarchical multi-node Tripo assembly scene with outward explosion wave kinematics.

## Key Code Files
- `XPart/partgen/utils/mesh_utils.py`:
  - `create_tripo_assembly_scene()`: Assembles submeshes while preserving original PBR textures and materials.
  - `export_tripo_assembly_glb()`: Exports valid multi-node hierarchical GLB.
  - `explode_mesh_tripo()`: Computes directional explosion wave vectors from bounding-box centroids.
  - `generate_tripo_viewer_html()`: Three.js interactive viewer with explosion slider.
- `export_tripo_assembly.py`: Standalone CLI export tool.
- `web_app/server.py` (`mode == "tripo_exact"`): High-speed in-memory web handler.

## Usage
### Standalone CLI:
```bash
python export_tripo_assembly.py --input mesh.glb --output_dir outputs/tripo_exact
```
### Web UI:
Select `🧩 Tripo Exact Partitioning (In-Memory, ~2-4s)` and click **Start Segmentation**.
"""

README_ALL = """# Complete Part Segmentation Pipeline Code Suite

This master archive contains the complete source code, utilities, configurations, and benchmarks for all 4 pipelines:

1. **Pipeline 1: Clean Parts Fast (`clean_parts_fast`)**
   - High-speed in-memory GPU P3-SAM (~1-3s total).
2. **Pipeline 2: Clean Parts Precise (`clean_parts_precise`)**
   - Quality-optimized PartSeg pipeline with 16-pass TTA, graph cuts, crease snapping, and cap sealing.
3. **Pipeline 3: Benchmark 3-Way Suite (`benchmark_suite`)**
   - Side-by-side comparative suite (Baseline vs Enhanced vs Final PartSeg) with boundary metrics and grid generators.
4. **Pipeline 4: Tripo Exact Partitioning (`tripo_exact`)**
   - Authentic PBR texture-preserving exact partitioning and Tripo3D hierarchical assembly.

## Directory Structure
- `partseg/`: Core PartSeg segmentation library (CLI, config, pipeline, polish, seal, export).
- `P3-SAM/`: P3-SAM neural network model architecture and demo runners.
- `XPart/partgen/utils/`: Tripo3D multi-node assembly, explosion, and PBR material preservation utilities.
- `web_app/`: Web Studio server (`server.py`) and UI (`static/index.html`, `app.js`, `style.css`).
- `benchmarks/`: Benchmark suites, quantitative evaluation metrics, and result exporters.
- `scripts/`: PDF report generator and automation utilities.
- `tests/`: Comprehensive unit and regression test suite.
"""

def create_zips():
    code_exts = {".py", ".html", ".js", ".css", ".json", ".txt", ".md", ".yaml", ".yml", ".sh", ".bat"}
    
    # -------------------------------------------------------------
    # 1. Pipeline 1: Clean Parts Fast
    # -------------------------------------------------------------
    p1_path = ROOT_DIR / "pipeline_1_clean_parts_fast.zip"
    print(f"Creating {p1_path.name}...")
    with zipfile.ZipFile(p1_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", README_P1)
        add_dir(zf, ROOT_DIR / "partseg", filter_exts=code_exts)
        add_file(zf, ROOT_DIR / "P3-SAM" / "model.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "auto_mask_fast.py")
        add_file(zf, ROOT_DIR / "web_app" / "server.py")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "index.html")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "app.js")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "style.css")
        add_file(zf, ROOT_DIR / "requirements-partseg.txt")
    print(f"  -> Created {p1_path.name} ({p1_path.stat().st_size / 1024:.1f} KB)")

    # -------------------------------------------------------------
    # 2. Pipeline 2: Clean Parts Precise
    # -------------------------------------------------------------
    p2_path = ROOT_DIR / "pipeline_2_clean_parts_precise.zip"
    print(f"Creating {p2_path.name}...")
    with zipfile.ZipFile(p2_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", README_P2)
        add_dir(zf, ROOT_DIR / "partseg", filter_exts=code_exts)
        add_file(zf, ROOT_DIR / "P3-SAM" / "model.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "auto_mask_fast.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "graph_cut_refinement.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "second_pass_prompting.py")
        add_file(zf, ROOT_DIR / "benchmark_graph_cut.py")
        add_file(zf, ROOT_DIR / "benchmark_second_pass.py")
        add_file(zf, ROOT_DIR / "benchmark_p3sam_multimask.py")
        add_file(zf, ROOT_DIR / "run_combined_all_steps.py")
        add_file(zf, ROOT_DIR / "web_app" / "server.py")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "index.html")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "app.js")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "style.css")
        add_file(zf, ROOT_DIR / "requirements-partseg.txt")
    print(f"  -> Created {p2_path.name} ({p2_path.stat().st_size / 1024:.1f} KB)")

    # -------------------------------------------------------------
    # 3. Pipeline 3: Benchmark 3-Way Suite
    # -------------------------------------------------------------
    p3_path = ROOT_DIR / "pipeline_3_benchmark_suite.zip"
    print(f"Creating {p3_path.name}...")
    with zipfile.ZipFile(p3_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", README_P3)
        add_dir(zf, ROOT_DIR / "benchmarks", filter_exts=code_exts)
        add_dir(zf, ROOT_DIR / "scripts", filter_exts=code_exts)
        add_dir(zf, ROOT_DIR / "partseg", filter_exts=code_exts)
        add_file(zf, ROOT_DIR / "run_multimesh_suite.py")
        add_file(zf, ROOT_DIR / "multimesh_suite_base.py")
        add_file(zf, ROOT_DIR / "build_complete_showcase.py")
        add_file(zf, ROOT_DIR / "build_master_multimesh_grid.py")
        add_file(zf, ROOT_DIR / "generate_comparison_figures.py")
        add_file(zf, ROOT_DIR / "benchmark_hierarchical.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "model.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "auto_mask.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "auto_mask_fast.py")
        add_file(zf, ROOT_DIR / "web_app" / "server.py")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "index.html")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "app.js")
        add_file(zf, ROOT_DIR / "requirements-partseg.txt")
    print(f"  -> Created {p3_path.name} ({p3_path.stat().st_size / 1024:.1f} KB)")

    # -------------------------------------------------------------
    # 4. Pipeline 4: Tripo Exact Partitioning
    # -------------------------------------------------------------
    p4_path = ROOT_DIR / "pipeline_4_tripo_exact.zip"
    print(f"Creating {p4_path.name}...")
    with zipfile.ZipFile(p4_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", README_P4)
        add_file(zf, ROOT_DIR / "XPart" / "partgen" / "utils" / "mesh_utils.py")
        add_file(zf, ROOT_DIR / "XPart" / "partgen" / "utils" / "misc.py")
        add_file(zf, ROOT_DIR / "export_tripo_assembly.py")
        add_dir(zf, ROOT_DIR / "partseg", filter_exts=code_exts)
        add_file(zf, ROOT_DIR / "P3-SAM" / "model.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "auto_mask_fast.py")
        add_file(zf, ROOT_DIR / "web_app" / "server.py")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "index.html")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "app.js")
        add_file(zf, ROOT_DIR / "web_app" / "static" / "style.css")
        add_file(zf, ROOT_DIR / "requirements-partseg.txt")
    print(f"  -> Created {p4_path.name} ({p4_path.stat().st_size / 1024:.1f} KB)")

    # -------------------------------------------------------------
    # 5. Master Suite: all_4_pipelines_code.zip
    # -------------------------------------------------------------
    p_all_path = ROOT_DIR / "all_4_pipelines_code.zip"
    print(f"Creating {p_all_path.name}...")
    with zipfile.ZipFile(p_all_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README_PIPELINES_OVERVIEW.md", README_ALL)
        add_dir(zf, ROOT_DIR / "partseg", filter_exts=code_exts)
        add_dir(zf, ROOT_DIR / "web_app", filter_exts=code_exts)
        add_dir(zf, ROOT_DIR / "benchmarks", filter_exts=code_exts)
        add_dir(zf, ROOT_DIR / "scripts", filter_exts=code_exts)
        add_dir(zf, ROOT_DIR / "tests", filter_exts=code_exts)
        add_dir(zf, ROOT_DIR / "docs", filter_exts=code_exts)
        add_file(zf, ROOT_DIR / "P3-SAM" / "model.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "auto_mask.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "auto_mask_fast.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "graph_cut_refinement.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "second_pass_prompting.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "hierarchical_clustering.py")
        add_file(zf, ROOT_DIR / "P3-SAM" / "demo" / "run_p3sam_for_xpart.py")
        add_file(zf, ROOT_DIR / "XPart" / "partgen" / "utils" / "mesh_utils.py")
        add_file(zf, ROOT_DIR / "XPart" / "partgen" / "utils" / "misc.py")
        add_file(zf, ROOT_DIR / "run_combined_all_steps.py")
        add_file(zf, ROOT_DIR / "run_multimesh_suite.py")
        add_file(zf, ROOT_DIR / "multimesh_suite_base.py")
        add_file(zf, ROOT_DIR / "benchmark_graph_cut.py")
        add_file(zf, ROOT_DIR / "benchmark_second_pass.py")
        add_file(zf, ROOT_DIR / "benchmark_p3sam_multimask.py")
        add_file(zf, ROOT_DIR / "benchmark_hierarchical.py")
        add_file(zf, ROOT_DIR / "build_complete_showcase.py")
        add_file(zf, ROOT_DIR / "build_master_multimesh_grid.py")
        add_file(zf, ROOT_DIR / "generate_comparison_figures.py")
        add_file(zf, ROOT_DIR / "export_tripo_assembly.py")
        add_file(zf, ROOT_DIR / "requirements-partseg.txt")
        add_file(zf, ROOT_DIR / "README.md")
        add_file(zf, ROOT_DIR / "Dockerfile")
        add_file(zf, ROOT_DIR / ".dockerignore")
    print(f"  -> Created {p_all_path.name} ({p_all_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    create_zips()
    print("\nAll pipeline zip files created successfully in workspace root.")
