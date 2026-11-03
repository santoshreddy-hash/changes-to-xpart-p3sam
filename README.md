# 3D Part Segmentation & Assembly Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Precision bfloat16](https://img.shields.io/badge/Precision-bfloat16-purple.svg)](https://pytorch.org/docs/stable/amp.html)
[![License: Tencent Hunyuan](https://img.shields.io/badge/License-Hunyuan%20Non--Commercial-lightgrey.svg)](LICENSE)

High-performance 3D mesh part segmentation, crease-aligned boundary regularization, semantic part intelligence, and animated exploded assembly generation. Powered by in-memory GPU-accelerated **P3-SAM** and **X-Part** (Hunyuan3D-Part).

---

## Table of Contents

- [Architectural Overview](#architectural-overview)
- [The 4 Core Pipelines](#the-4-core-pipelines)
  - [Pipeline 1: Clean Parts Fast (v1 & Fast v2 Enhanced)](#pipeline-1-clean-parts-fast-v1--fast-v2-enhanced)
  - [Pipeline 2: Clean Parts Precise](#pipeline-2-clean-parts-precise)
  - [Pipeline 3: Benchmark & Evaluation Suite](#pipeline-3-benchmark--evaluation-suite)
  - [Pipeline 4: Tripo Exact Partitioning](#pipeline-4-tripo-exact-partitioning)
- [Semantic Part Intelligence (Method 1 vs Method 2)](#semantic-part-intelligence-method-1-vs-method-2)
- [Interactive Web Studio](#interactive-web-studio)
- [CLI & Python API Usage](#cli--python-api-usage)
- [Benchmarking & Quality Metrics](#benchmarking--quality-metrics)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)
- [License & Acknowledgments](#license--acknowledgments)

---

## Architectural Overview

```
                                      [Input 3D Mesh (.glb / .obj / .ply)]
                                                       │
                                                       ▼
                                     ┌───────────────────────────────────┐
                                     │  Warm GPU P3-SAM Residency (VRAM)  │
                                     │   (bfloat16, Batched Inference)   │
                                     └─────────────────┬─────────────────┘
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     │                                 │                                 │
                     ▼                                 ▼                                 ▼
         ┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
         │      PIPELINE 1       │         │      PIPELINE 2       │         │      PIPELINE 4       │
         │   Clean Parts Fast    │         │  Clean Parts Precise  │         │      Tripo Exact      │
         ├───────────────────────┤         ├───────────────────────┤         ├───────────────────────┤
         │ • Curvature Sampling  │         │ • 3-Pass Feature TTA  │         │ • In-Memory Slice     │
         │ • Smart Contrast FPS  │         │ • Dual-Tier Graph Cut │         │ • 100% PBR Textures   │
         │ • Topological Anchor  │         │ • Dihedral Crease Snap│         │ • Vertex Color Audit  │
         │ • Narrow-Band Polish  │         │ • Satellite Clean     │         │ • Exploded Transforms │
         │ • ~1.5s - 2.5s Total  │         │ • Watertight Seal Rims│         │ • ~2.0s - 3.5s Total  │
         └───────────┬───────────┘         └───────────┬───────────┘         └───────────┬───────────┘
                     │                                 │                                 │
                     └─────────────────────────────────┼─────────────────────────────────┘
                                                       │
                                                       ▼
                                     ┌───────────────────────────────────┐
                                     │    Agglomeration & Intelligence   │
                                     │ • Hierarchical Ward Clustering    │
                                     │ • Semantic Ontology & Kinematics  │
                                     │ • Solid Watertight Capping        │
                                     └─────────────────┬─────────────────┘
                                                       │
                                                       ▼
                                    [Multi-Node Exploded Assembly (.glb)]
                                    [Telemetry HUD & Structured Reports ]
```

---

## The 4 Core Pipelines

### Pipeline 1: Clean Parts Fast (v1 & Fast v2 Enhanced)
*Engineered for real-time applications, interactive web viewers, and high-throughput asset ingestion.*
* **Latency:** ~1.5s – 2.8s total on an NVIDIA RTX 4090.
* **Top-5 Prioritized Upgrades in Fast v2:**
  1. **Adaptive Curvature Sampling:** Concentrates 2–3× higher point density across thin components (table legs, brackets, handles, sharp rims) weighted by face curvature and normal variance, without increasing the overall point budget.
  2. **Smart Feature-Contrast Prompts:** Weighted Farthest Point Sampling (FPS) places prompt seeds preferentially along surface discontinuities, preventing prompts from crowding flat coplanar regions.
  3. **Direct Topological Face-Anchoring:** Bypasses 3D Euclidean KNN point-cloud voting to eliminate spatial part dilution and component swallowing.
  4. **Fast Narrow-Band Boundary Refinement:** Snaps boundary interfaces to sharp dihedral creases in <25 ms.
  5. **Fast-to-Precise Health Check Guard:** Continuously computes boundary metrics; triggers a precise pass only if assignment confidence drops below safe thresholds.

### Pipeline 2: Clean Parts Precise
*Production-grade CAD-quality segmentation maximizing physical seam precision.*
* **Latency:** ~5.0s – 7.5s on NVIDIA RTX 4090.
* **Key Features:**
  * **3-Pass Feature Test-Time Augmentation (TTA):** Canonical, mirror-reflected, and perturbed passes merged with confidence voting.
  * **Dual-Tier Graph Cut Regularization:** Dihedral angle energy minimization aligning **97.35%** of part boundaries to physical 90° mechanical joints and creases.
  * **Satellite Tooth Absorption:** Strict edge-connectivity pruning to eliminate jagged boundary teeth.
  * **Watertight Solid Cut-Rim Sealing (`partseg.seal`):** Automatically caps open severed loops with planar triangle fans, converting hollow shells into closed, manifold solids ready for 3D printing (STL slicers) and physics collision engines.

### Pipeline 3: Benchmark & Evaluation Suite
*Empirical evaluation harness comparing baseline, enhanced, and final boundary segmentation.*
* **Ground-Truth Metrics:** Computes Precision, Recall, F1 Score, Over-segmentation index, Under-segmentation index, and Dihedral Crease Error.
* **Multi-Mesh Runner:** Runs automated suites across complex models (`wooden_rack`, `furn16_3k`, `stanford_dragon`, `man_creature`, etc.).
* **Automated Visual Reports:** Generates side-by-side zoomed seam crops, exploded assembly grids, HTML documentation, and multi-page PDF executive summaries.

### Pipeline 4: Tripo Exact Partitioning
*1:1 geometry and texture preservation pipeline designed for textured game assets and animated assemblies.*
* **Authentic Material Retention:** Slices submeshes cleanly while preserving 100% of authentic PBR material channels, UV coordinate maps, and vertex colors.
* **Animated Exploded Wave:** Computes outwards centroid displacement trajectories along part bounding normals for interactive WebGL inspection.

---

## Semantic Part Intelligence (Method 1 vs Method 2)

The system supports two complementary segmentation paradigms:

```
METHOD 1: Pure Geometric Fast v2               METHOD 2: Semantic Part Intelligence
┌───────────────────────────────────────┐      ┌───────────────────────────────────────┐
│ • Class-Agnostic Geometric Separation │      │ • Domain Ontology & Functional Roles  │
│ • Anonymous Nodes: part_000, part_001 │      │ • True Names: Main_Countertop, Leg_FL │
│ • Sliced on dihedral angle thresholds │      │ • Kinematics: Drawers slide along +Z  │
│ • No concept of what parts do/are     │      │ • Explains WHY the cut exists         │
└───────────────────────────────────────┘      └───────────────────────────────────────┘
```

### Quantitative Scorecard on `assets/furn16_3k.glb` (Antique Bureau Credenza)

| Feature | Method 1: Pure Geometric | Method 2: Semantic Intelligence |
| :--- | :--- | :--- |
| **Object Class Understanding** | None (Unsupervised) | **Antique Bureau Credenza / Desk** |
| **Part Identification** | Anonymous (`part_000` to `part_013`) | **True Semantic Names** (e.g., `Top_Left_Drawer_Face`) |
| **Boundary Snapping** | **86.6%** crease alignment | **86.6%** crease alignment (identical sharp seams) |
| **Assembly Hierarchy** | Flat disjoint list | **Carcass Frame $\to$ Drawer Bank $\to$ Trim Rails** |
| **Kinematic Degrees of Freedom** | Unknown / Rigid | **6 Sliding Drawers: Prismatic ($+Z$ translation)** |
| **Mechanical Rationale** | Dihedral angle $>25^\circ$ | **Clearance reveals & mechanical slide tracks** |
| **Processing Time** | **2.55s** | **2.63s** ($+80\text{ ms}$ ontology mapping) |

*Interactive 3D Viewer:* Launch `generate_method_comparison.py` to open [`comparison_viewer.html`](file:///outputs/method_comparison/comparison_viewer.html) with the live **Semantic Part Inspector**.

---

## Interactive Web Studio

A full-featured WebGL studio is included, allowing interactive testing, segmentation, and live telemetry.

```bash
# Start the web studio daemon
python web_app/server.py --port 8085
```

Navigate to: **`http://localhost:8085`**

### Features:
* **Persistent In-Memory GPU Residency:** The P3-SAM model remains resident in GPU VRAM across requests, dropping execution latency from ~20s down to ~1–3s.
* **Live Boundary Quality Telemetry HUD:**
  * **Crease Snapping:** Real-time percentage of boundary edges resting on sharp surface creases.
  * **Mean Confidence:** Average neural assignment probability.
  * **Seam Edge Efficiency:** Boundary perimeter edge count (lower values denote straighter, cleaner separation).
* **Target Parts Control:** Dropdown (`Auto`, `4`, `6`, `8`, `12`, `16`) allowing instant agglomerative Ward clustering to merge touching subcomponents into macro-assemblies in <120 ms.
* **Seal Rims Toggle:** Optional automated planar cap synthesis to produce solid watertight CAD parts.
* **Exploded View Slider:** Real-time continuous part disassembly with coordinate animations.

---

## CLI & Python API Usage

### Command-Line Interface (`partseg`)

```bash
# Verify environment and CUDA device readiness
python -m partseg doctor --load-model

# Fast segmentation (Pipeline 1)
python -m partseg segment assets/furn16_3k.glb -o outputs/test_fast --preset fast

# Precise CAD segmentation with crease snapping and rim sealing (Pipeline 2)
python -m partseg segment assets/furn16_3k.glb -o outputs/test_precise --preset precise --seal

# Agglomerate into exactly 6 macro-components
python -m partseg segment assets/furn16_3k.glb -o outputs/test_merged --target-parts 6
```

### Python API

```python
import trimesh
from partseg import SegmentConfig, segment_mesh, load_model

# 1. Load mesh
mesh = trimesh.load("assets/furn16_3k.glb", force="mesh")

# 2. Configure preset (fast, precise, or detailed)
cfg = SegmentConfig.preset("fast")
cfg.polish.min_area_frac = 0.001
cfg.export.seal = True

# 3. Execute in-memory segmentation
model = load_model()
result = segment_mesh(mesh, cfg, model=model)

# 4. Access outputs
print(f"Discovered {result.num_parts} parts in {result.timings['total']:.2f}s")
print(f"Crease alignment: {result.report['quality']['crease_snapping_pct']}%")

# 5. Export structured assembly
result.export("outputs/my_segmented_asset")
```

---

## Benchmarking & Quality Metrics

Measured across representative benchmark models on an **NVIDIA RTX 4090 (24GB)**:

| Asset | Faces | Method | Parts | Crease Alignment | Seam Perimeter | Total Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **furn16_3k** (Bureau Credenza) | 3,000 | Baseline | 8 | 41.2% | 480 edges | 22.4s |
| | 3,000 | **Fast v2** | **14** | **86.6%** | **239 edges** | **2.55s** |
| | 3,000 | **Precise** | **14** | **97.3%** | **212 edges** | **5.42s** |
| **wooden_rack** (Display Shelf) | 854,832 | Baseline | 12 | 34.8% | 3,120 edges | 52.2s |
| | 854,832 | **Fast v2** | **15** | **88.4%** | **1,410 edges** | **6.10s** |
| | 854,832 | **Precise** | **15** | **96.8%** | **1,180 edges** | **14.3s** |
| **stanford_dragon** | 871,306 | Baseline | 18 | 29.1% | 4,210 edges | 51.0s |
| | 871,306 | **Precise** | **22** | **91.5%** | **1,840 edges** | **12.8s** |

---

## Repository Structure

```
part-segmentation/
├── partseg/                          # Production segmentation package
│   ├── adaptive_fast.py              # Pipeline 1 Fast v2: Curvature sampling, Smart FPS, Topological anchoring
│   ├── pipeline.py                   # Unified segmentation orchestration (fast/precise/detailed)
│   ├── polish.py                     # Dual-tier graph-cut, patch snapping & boundary clean-up
│   ├── p3sam.py                      # Warm GPU model interface & sampling parameters
│   ├── hierarchy.py                  # Hierarchical Ward clustering agglomeration
│   ├── seal.py                       # Watertight solid cut-rim capping
│   ├── export.py                     # Structured multi-node GLB assembly & JSON reports
│   ├── config.py                     # Dataclass configuration schemas
│   └── cli.py                        # Command-line entry points
├── P3-SAM/                           # Low-level P3-SAM model & demo runners
│   ├── demo/auto_mask_fast.py        # Vectorized GPU inference engine
│   ├── demo/graph_cut_refinement.py  # Standalone graph cut refiner
│   └── model.py                      # P3-SAM backbone network definition
├── XPart/                            # Generative Part Decomposition (Hunyuan3D-Part)
│   ├── partgen/partformer_pipeline.py# Diffusion Transformer (DiT) pipeline
│   ├── partgen/models/partformer_dit.py # DiT model architecture
│   └── demo.py                       # X-Part inference demo
├── web_app/                          # Interactive Web Studio
│   ├── server.py                     # Async server with persistent GPU residency
│   ├── static/index.html             # Web Studio frontend UI
│   └── static/app.js                 # Three.js viewport, telemetry HUD, exploded controls
├── benchmarks/                       # Empirical evaluation & testing
│   ├── run_suite.py                  # Automated multi-mesh suite runner
│   ├── eval_parts.py                 # Ground-truth metric computation
│   └── build_grid.py                 # Visual summary table generator
├── generate_method_comparison.py     # Comparison generator for Method 1 vs Method 2
├── package_pipeline_zips.py          # Standalone zip packager for pipelines 1-4
├── requirements-partseg.txt          # Python dependencies
└── Dockerfile                        # Containerized production runtime
```

---

## Installation & Quickstart

### 1. Prerequisites
* Linux or Windows with an NVIDIA GPU (RTX 3080/4080/4090 or A100 recommended).
* CUDA Toolkit 12.1+ and Python 3.10+.

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/santoshreddy-hash/changes-to-xpart-p3sam.git
cd changes-to-xpart-p3sam

# Create virtual environment
conda create -n partseg python=3.10 -y
conda activate partseg

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install package requirements
pip install -r requirements-partseg.txt
```

### 3. Verify Installation & Model Weights

```bash
# Preload P3-SAM checkpoint and verify GPU acceleration
python -m partseg doctor --load-model

# Run self-tests
python -m unittest discover -s tests -t .
```

---

## License & Acknowledgments

* **P3-SAM and X-Part** are derivatives based on [Tencent's Hunyuan3D-Part](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part), released under the Tencent Hunyuan Non-Commercial License Agreement.
* See `LICENSE` and `NOTICE` for full terms.
