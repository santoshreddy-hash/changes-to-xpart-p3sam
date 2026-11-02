# partseg - clean part segmentation

`partseg` turns a 3D mesh into clean, named parts. It runs P3-SAM, splits parts that P3-SAM left
as one blob, and cleans every boundary so parts are single connected pieces with smooth edges.

## Requirements

- NVIDIA GPU with CUDA (validated on an RTX 4090, 24 GB), Python 3.10.
- `pip install -r requirements-partseg.txt` (install the CUDA build of PyTorch first; see the file).
- The P3-SAM weights download on first use; copy the cache folder to run offline.
- Check the machine: `python -m partseg doctor --load-model`

## Command line

```bash
python -m partseg segment chair.glb -o out/chair
python -m partseg segment models/ -o out            # a folder: one sub-folder per mesh, model loaded once
python -m partseg segment chair.glb -o out --preset fast
python -m partseg segment chair.glb -o out --set polish.stay_cost=0.04 --keep-materials
```

Exit codes: `0` success, `1` some files in a batch failed, `2` bad input or config,
`3` environment problem (no CUDA, missing weights), `4` pipeline failure.

## Python

```python
from partseg import SegmentConfig, load_model, segment_file

model = load_model()                                   # once per process
cfg = SegmentConfig.preset("balanced")
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

## What the pipeline does

1. **P3-SAM, multi-mask pass**: stability filtering, second-pass prompting, graph cut. Backbone features are
   averaged over 16 voxel-sampling seeds (`p3sam.feature_tta`), which cancels most of P3-SAM's seed noise.
2. **Re-prompting**: inside large parts, a sub-part is accepted only when several independent prompts agree.
3. **Feature splitting**: large organic parts are split along clusters of P3-SAM's own features, and each
   split must be confirmed by the mask decoder prompted inside both children.
4. **Boundary clean-up**: crack-tolerant component splitting, fragment merging, crease snapping, a
   crease-aware graph cut and label diffusion. Flat panels (a table top) are never split.

Presets: `balanced` (default) and `fast` (one re-prompt round, no feature splitting, 4-pass averaging).
Every tunable is in `partseg/config.py`; override with `--set section.key=value` or `--config file.json`.

## Measured behaviour (8-mesh benchmark, RTX 4090)

Run `python benchmarks/run_suite.py` then `python benchmarks/build_grid.py`. Results land in
`outputs/final_pipeline_suite/` (comparison images, `final_summary.json`, `FINAL_master_grid.png`).

- Parts over the 8 meshes: baseline P3-SAM 105, enhanced P3-SAM 144, final partseg 179.
- Time per mesh: 3 to 33 s once the model is loaded, 18.5 s on average (the 854k-face sideboard takes about 31 s).
- Every final part is one connected piece, apart from a few specks under 0.01% of the surface that are
  merged into the nearest part.

## Web app

`python web_app/server.py --port 8080`. The **Clean Parts** pipeline in the dropdown runs this package as a
subprocess and partitions the mesh by the resulting labels, keeping the original textures.
`GET /api/health` reports the GPU and active jobs. Each job starts a new process, so it pays about
15 s of model loading; use the CLI or Python API for batches.

## Tests

```bash
python -m unittest discover -s tests -t .
```

39 tests run without a GPU. `tests/test_pipeline_gpu.py` (6 tests) runs the full pipeline on a real mesh and is
skipped when no GPU or sample mesh is present.

## Known limits

- **Not bit-reproducible across separate runs.** Seeds are fixed and a run is repeatable inside one process,
  but GPU kernel selection can differ between processes, so a few boundaries may move or the part count may
  change by one or two. Treat the output as stable, not as byte-identical.
- P3-SAM's decoder does not recognise some anatomical boundaries (a deer's torso, neck and head are one part);
  the pipeline keeps that rather than invent splits.
- Seams that follow irregular geometry (a roughly cut board edge) stay irregular because they lie on real creases.
- Limit of 2,000,000 faces per mesh (`max_faces`). Decimate larger meshes first.
- The web server is single-GPU: one job at a time, a new job pre-empts running ones.
- Linux and Docker are untested; the validated environment is Windows 11.
