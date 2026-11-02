# partseg - clean part segmentation

`partseg` turns a 3D mesh into clean, named parts. It runs P3-SAM and cleans every boundary so parts are
single connected pieces with smooth edges. The default `precise` preset keeps P3-SAM's splits; the optional
`detailed` preset also splits parts that P3-SAM left as one blob.

## Requirements

- NVIDIA GPU with CUDA (validated on an RTX 4090, 24 GB), Python 3.10.
- `pip install -r requirements-partseg.txt` (install the CUDA build of PyTorch first; see the file).
- The P3-SAM weights download on first use; copy the cache folder to run offline.
- Check the machine: `python -m partseg doctor --load-model`

## Command line

```bash
python -m partseg segment chair.glb -o out/chair
python -m partseg segment models/ -o out            # a folder: one sub-folder per mesh, model loaded once
python -m partseg segment chair.glb -o out --preset detailed   # also split large parts (see Presets)
python -m partseg segment chair.glb -o out --preset fast
python -m partseg segment chair.glb -o out --set polish.stay_cost=0.04 --keep-materials
python -m partseg segment chair.glb -o out --parts 6     # merge neighbouring parts down to at most 6
python -m partseg segment chair.glb -o out --seal        # also write a GLB with the cut surfaces capped
```

Exit codes: `0` success, `1` some files in a batch failed, `2` bad input or config,
`3` environment problem (no CUDA, missing weights), `4` pipeline failure.

## Python

```python
from partseg import SegmentConfig, load_model, segment_file

model = load_model()                                   # once per process
cfg = SegmentConfig()                                   # the default preset, "precise"
result = segment_file("chair.glb", "out/chair", cfg, model=model)
print(len(result.parts), result.stage_seconds["total"], result.warnings)
```

`segment_mesh(trimesh_mesh, cfg, model)` returns labels without writing files.
`result.face_ids[i]` is the part of face `i`; part `0` is always the largest.

## Output (one folder per mesh)

| File | Content |
|---|---|
| `<name>_parts.glb` | one named node per part (`part_000`, ...), flat colour or original material (`--keep-materials`) |
| `<name>_segmented.glb` | one mesh, one flat colour per part, colours never blend at boundaries |
| `<name>_face_ids.npy` | `int32` part id per face, same face order as the loaded mesh |
| `<name>_parts.json` | per part: id, name, colour, faces, area, area fraction, bbox, centroid, `connected_pieces`, neighbours (`contact`: `edge` or `proximity`) |
| `<name>_report.json` | config used, timings per stage, parts after each stage, boundary metrics, warnings, environment |
| `<name>_preview.png` | front and back render (skipped with `--no-preview`) |

Both JSON files carry `schema_version`.

### Optional outputs (off by default, results unchanged unless you ask)

- `--parts N` merges neighbouring parts (Ward linkage on mean P3-SAM features, only across parts that touch)
  until at most N remain. It can only reduce the count.
- `--seal` writes `<name>_parts_sealed.glb`: every open cut loop is closed with a triangle fan so parts become
  closed solids where possible. `parts.json` then records `watertight_before`, `watertight_sealed` and `cap_faces`.
  Caps are topological closures, so they are sensible for flat cuts and rough for strongly curved ones.
- To keep smaller parts (knobs, screws) lower `--set polish.min_area_frac=0.001`; the default 0.003 merges
  attached fragments below 0.3% of the surface into a neighbour.

## What the pipeline does

1. **P3-SAM, multi-mask pass**: stability filtering, second-pass prompting, graph cut. Backbone features are
   averaged over 16 voxel-sampling seeds (`p3sam.feature_tta`), which cancels most of P3-SAM's seed noise.
2. **Boundary clean-up**: crack-tolerant component splitting, fragment merging, crease snapping, a
   crease-aware graph cut and label diffusion. Flat panels (a table top) are never split.
3. *`detailed` only* **re-prompting**: inside large parts, a sub-part is accepted only when several independent
   prompts agree.
4. *`detailed` only* **feature splitting**: large organic parts are split along clusters of P3-SAM's own
   features, and each split must be confirmed by the mask decoder prompted inside both children.

## Presets

| Preset | What it does | When to use it |
|---|---|---|
| `precise` (default) | steps 1 and 2 | clean, hard-surface or CAD-like assets; the only preset with a measured accuracy advantage |
| `detailed` | steps 1 to 4 | organic characters or furniture where P3-SAM merges obvious parts (a table top with its body); review the result |
| `fast` | `precise` with 4-pass feature averaging | shorter runs; matched `precise` on the ground-truth benchmark, but seed-to-seed stability is lower |

Every tunable is in `partseg/config.py`; override with `--set section.key=value` or `--config file.json`.

## Measured behaviour (8-mesh benchmark, RTX 4090)

Run `python benchmarks/run_suite.py [--preset detailed]` then `python benchmarks/build_grid.py <folder>`.
Results land in `outputs/final_pipeline_suite/` (comparison images, `final_summary.json`, `FINAL_master_grid.png`).

| Preset | Parts over the 8 meshes | Average time per mesh |
|---|---|---|
| baseline P3-SAM | 105 | 3.7 s |
| `precise` (default) | 137 | 10.4 s |
| `detailed` | 179 | 18.5 s |

- Time per mesh with `precise`: 2.7 to 30.8 s once the model is loaded (the 854k-face sideboard takes about 31 s).
- Every final part is one connected piece, apart from a few specks under 0.01% of the surface that are
  merged into the nearest part.
- On real meshes `detailed` finds splits `precise` misses (bedside table 12 parts vs 7, deer 16 vs 8, dining set
  28 vs 19), but there is no ground truth for real meshes, so which is more correct is not measured.

## Ground-truth benchmark

`python benchmarks/eval_parts.py` builds 8 procedural assemblies with exact part labels
(`benchmarks/synthetic.py`) and scores each configuration against them. Synthetic parts are far cleaner than
real assets, so read the numbers as a regression and ablation signal, not as real-world accuracy.

| Configuration | ARI | Part recall (IoU) | Part precision (IoU) | True parts recovered |
|---|---|---|---|---|
| Baseline P3-SAM | 0.961 | 0.942 | 0.994 | 38 / 42 |
| Enhanced P3-SAM | 0.988 | 0.968 | 0.956 | 39 / 42 |
| `precise` (default) | 0.986 | 0.965 | 0.966 | 39 / 42 |
| `fast` | 0.986 | 0.966 | 0.966 | 39 / 42 |
| `detailed` | 0.755 | 0.707 | 0.457 | 25 / 42 |

**What this means.** On clean geometry where P3-SAM is already right, the `detailed` re-prompting and feature
splitting stages add false splits (a 3-part dumbbell becomes 16 parts, a 3-part lamp becomes 10). They were
tuned by eye on 8 real meshes where P3-SAM under-splits (a deer, characters), where they do add real parts, but
that gain has not been measured against ground truth. That is why `precise` is the default.

`benchmarks/run_suite.py` remains the visual benchmark on the 8 real meshes.

## Deployment files (not verified here)

`Dockerfile` and `.github/workflows/ci.yml` were written on a Windows machine without Docker or Actions access,
so neither has been run. The CI job runs only the tests that need no GPU.

## Web app

`python web_app/server.py --port 8080`. The **Clean Parts** pipeline in the dropdown runs this package as a
subprocess (default `precise` preset) and partitions the mesh by the resulting labels, keeping the original textures.
`GET /api/health` reports the GPU and active jobs. Each job starts a new process, so it pays about
15 s of model loading; use the CLI or Python API for batches.

## Tests

```bash
python -m unittest discover -s tests -t .
```

39 tests run without a GPU. `tests/test_pipeline_gpu.py` (6 tests) runs the full pipeline on a real mesh and is
skipped when no GPU or sample mesh is present.

## Known limits

- **The default (`precise`) can under-split.** It keeps P3-SAM's splits, so a table top merged with its body or a
  deer's torso and neck stay merged. `--preset detailed` splits more but can add false splits (see the ground-truth table).
- **Not bit-reproducible across separate runs.** Seeds are fixed and a run is repeatable inside one process,
  but GPU kernel selection can differ between processes, so a few boundaries may move or the part count may
  change by one or two. Treat the output as stable, not as byte-identical. Investigation so far: the backbone
  features themselves differ between processes (max difference about 0.2, in discrete jumps, in roughly a third
  of processes), independent of every later stage. Forcing spconv's MaskImplicitGemm kernel or fp32 accumulation did
  not remove it on all meshes (`MaskSplitImplicitGemm` is unsupported for this layer, `Native` is worse).
  Likely cause: timing-based kernel auto-tuning; not yet fixed.
- P3-SAM's decoder does not recognise some anatomical boundaries (a deer's torso, neck and head are one part);
  the pipeline keeps that rather than invent splits.
- Seams that follow irregular geometry (a roughly cut board edge) stay irregular because they lie on real creases.
- Limit of 2,000,000 faces per mesh (`max_faces`). Decimate larger meshes first.
- The web server is single-GPU: one job at a time, a new job pre-empts running ones.
- Linux and Docker are untested; the validated environment is Windows 11.
