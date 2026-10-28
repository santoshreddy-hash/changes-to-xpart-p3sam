# Part Segmentation

Optimized P3-SAM segmentation pipeline + the X-Part generative part-decomposition
pipeline wired to use it, based on [Tencent's Hunyuan3D-Part](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part).
See `LICENSE` / `NOTICE` for the original project's license terms (TENCENT HUNYUAN
NON-COMMERCIAL LICENSE AGREEMENT) -- this repo is a derivative work under those terms.

## What's in here

- **`P3-SAM/`** -- the optimized P3-SAM segmentation pipeline.
  - `demo/auto_mask_fast.py` -- optimized pipeline (see "P3-SAM optimizations" below).
  - `demo/auto_mask.py` -- original baseline, kept for reference/comparison.
  - `demo/run_p3sam_for_xpart.py` -- standalone script that runs the optimized
    pipeline on a mesh and saves the predicted part bounding boxes as a `.npy`,
    for use by X-Part (see below).
  - `model.py` -- the P3-SAM model definition (unmodified).
- **`XPart/`** -- X-Part's generative part-decomposition pipeline, patched to:
  1. use the optimized P3-SAM pipeline above for its internal bbox-prediction
     step (`partgen/bbox_estimator/auto_mask_api.py`) instead of its own
     bundled, unoptimized, buggy copy,
  2. accept a precomputed bbox via `--aabb_path` (`demo.py`) so P3-SAM can be
     run standalone first, with the full GPU to itself, instead of competing
     for memory with X-Part's own much larger models,
  3. process parts in small batches instead of all at once in several places
     that otherwise pin GPU memory at its ceiling on multi-part meshes:
     the diffusion denoising loop (`partformer_pipeline.py`), the per-part
     conditioning encoder (`partgen/models/autoencoders/model.py`), and the
     nearest-point matching step (`partgen/models/conditioner/condioner_release.py`).
  4. keep the sonata/spconv segmentation-feature backbone in fp32 while the
     rest of the pipeline runs in bf16 (spconv has no bf16 dtype mapping for
     some of its internal tensors).
- **`outputs/`** -- the two generated results referenced by this repo:
  - `generated_parts_assembled_50step.glb` -- full quality, 50 diffusion steps,
    produced before the part-chunking fix (X-Part's own internal bbox predictor
    replaced with the optimized P3-SAM one, but diffusion still unchunked).
  - `generated_parts_assembled_chunked20step.glb` -- 20 diffusion steps, with
    the part-chunked diffusion loop active (smoke-tested fix validating the
    ~50-70x per-step speedup from chunking).

## P3-SAM optimizations (`auto_mask_fast.py` vs `auto_mask.py`)

1. **Fixed a hard-coded settings bug**: the original `mesh_sam()` silently
   overwrote the caller's `point_num`/`prompt_num` arguments with hard-coded
   values (`100000`/`400`), so changing those settings had no effect.
2. **bf16 precision** for the main network inference pass (the single biggest
   speed win, ~25-35% of total runtime), while keeping the NMS overlap-check
   matmul explicitly in fp32 (`torch.autocast(..., enabled=False)`) since it
   needs full precision to stay accurate.
3. **Vectorized NMS**: the overlap-check between candidate parts is computed
   as one GPU matrix operation instead of a Python loop over every pair.
4. **Fast cleanup**: grouping connected faces into final parts uses SciPy's
   compiled `connected_components` instead of a hand-written Python BFS --
   verified bit-identical output, pure speedup, scales well with mesh size.

Combined, measured across 4 meshes (40k-871k faces): 1.8x-2.4x total speedup
over the original pipeline, with part counts staying within normal bf16
run-to-run variance (no quality regression).

## X-Part part-chunking fix

X-Part's diffusion denoising loop processes all detected parts through its
21-layer transformer in a single batch, every one of (by default) 50 steps.
On meshes that decompose into ~20+ parts this pins GPU memory at its ceiling,
which on Windows can silently trigger NVIDIA's "CUDA sysmem fallback" (spilling
to system RAM over PCIe instead of erroring) -- the process doesn't crash, it
just runs 50-70x slower with no warning. Processing parts in small batches
(default 6, configurable via `XPART_DIFFUSION_PART_CHUNK` env var) instead of
all at once keeps peak memory bounded regardless of part count. Measured
per-step diffusion time: ~250-350s/step unchunked -> ~5s/step chunked, on the
same RTX 4080 Laptop GPU (12GB) and the same 23-part test mesh.

The DiT has no cross-part attention mixing (self/cross-attention run per-part
along the batch axis), so chunking the batch dimension doesn't change what the
model computes for each part.

## Reproducing the two output files

```bash
# 1. Run P3-SAM standalone (fast, full GPU) to get the part bounding boxes
cd P3-SAM/demo
python run_p3sam_for_xpart.py
# -> writes XPart/results/precomputed_aabb_000.npy

# 2a. generated_parts_assembled_50step.glb -- full quality, 50 steps,
#     default chunk size (6)
cd ../../XPart
python demo.py --config partgen/config/infer.yaml \
    --mesh_path ./data/000.glb --save_dir ./results/ \
    --aabb_path ./results/precomputed_aabb_000.npy \
    --num_inference_steps 50

# 2b. generated_parts_assembled_chunked20step.glb -- fast smoke test that
#     validated the chunking fix, 20 steps, chunk size 4
XPART_DIFFUSION_PART_CHUNK=4 python demo.py --config partgen/config/infer.yaml \
    --mesh_path ./data/000.glb --save_dir ./results_smoketest/ \
    --aabb_path ./results/precomputed_aabb_000.npy \
    --num_inference_steps 20
```

Both runs used the same `data/000.glb` input mesh and the same precomputed
`aabb` from step 1 -- they differ only in step count and chunk size, both now
exposed as CLI flags (`--num_inference_steps`, `XPART_DIFFUSION_PART_CHUNK`
env var) instead of being hardcoded.

Requires the pretrained weights, auto-downloaded from HuggingFace
(`tencent/Hunyuan3D-Part`) on first run. See each subproject's own
`requirements.txt` for dependencies.
