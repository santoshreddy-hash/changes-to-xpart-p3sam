# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from partgen.partformer_pipeline import PartFormerPipeline

import pytorch_lightning as pl
from partgen.utils.misc import get_config_from_file
import argparse 
import numpy as np
import torch


def main(ckpt_path, config, mesh_path, save_dir, ignore_keys=(), aabb_path=None, num_inference_steps=50, octree_resolution=512, mc_algo="dmc"):
    import time
    start_total = time.perf_counter()
    # Enable TF32 and cuDNN benchmark for faster lossless GPU execution
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Set random seed
    pl.seed_everything(2026, workers=True)
    # pipeline = PartFormerPipeline.from_single_file(
    #     ckpt_path=ckpt_path,
    #     config=config,
    #     verbose=True,
    #     ignore_keys=ignore_keys,
    # )
    pipeline = PartFormerPipeline.from_pretrained(
        model_path="tencent/Hunyuan3D-Part",
        verbose=True,
    )
    pipeline.to(device="cuda", dtype=torch.bfloat16)
    # The sonata/spconv backbone used for segmentation features only supports
    # fp32 (or fp16 via its own internal autocast) stored weights -- its CUDA
    # bridge has no bf16 dtype mapping for e.g. conv bias tensors. The rest of
    # the pipeline (VAE, conditioner's attention encoders, DiT) is fine in
    # bf16; the conditioner's code already casts this submodule's *inputs* to
    # float32, so keep its *weights* float32 too for consistency.
    pipeline.conditioner.seg_feat_encoder.to(dtype=torch.float32)
    cfg = -1.0
    # for mesh paths
    uid = Path(mesh_path).stem
    additional_params = {"output_type": "trimesh"}
    if aabb_path is not None:
        # Skip X-Part's own (slow, GPU-memory-hungry-alongside-the-rest-of-
        # the-pipeline) bbox predictor and use boxes already computed by a
        # standalone, full-GPU P3-SAM run instead. aabb must be in the SAME
        # coordinate space as the raw (unnormalized) input mesh -- check_inputs
        # normalizes it internally using the same center/scale it computes
        # for mesh_path.
        aabb = np.load(aabb_path)
        print(f"Using precomputed aabb from {aabb_path}: {aabb.shape}")
        # Not needed this run -- free its GPU memory before the heavy
        # generation stage, which is already pinned near the VRAM ceiling.
        if getattr(pipeline, "bbox_predictor", None) is not None:
            pipeline.bbox_predictor.model.cpu()
            pipeline.bbox_predictor.model_parallel.cpu()
            pipeline.bbox_predictor = None
        pipeline_start = time.perf_counter()
        obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = pipeline(
            mesh_path=mesh_path,
            aabb=aabb,
            octree_resolution=octree_resolution,
            num_inference_steps=num_inference_steps,
            mc_algo=mc_algo,
            **additional_params,
        )
        pipeline_time = time.perf_counter() - pipeline_start
    else:
        pipeline_start = time.perf_counter()
        obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = pipeline(
            mesh_path=mesh_path,
            octree_resolution=octree_resolution,
            num_inference_steps=num_inference_steps,
            mc_algo=mc_algo,
            **additional_params,
        )
        pipeline_time = time.perf_counter() - pipeline_start
    from partgen.utils.mesh_utils import export_tripo_assembly_glb, generate_tripo_viewer_html

    assembled_glb_name = f"train_cfg_{cfg:04f}_dit_boxgpt_{uid}.glb"
    exploded_glb_name = f"train_cfg_{cfg:04f}_explode_boxgpt_{uid}.glb"
    viewer_html_name = f"train_cfg_{cfg:04f}_viewer_{uid}.html"

    export_tripo_assembly_glb(obj_mesh, save_dir / assembled_glb_name)
    out_bbox.export(save_dir / f"train_cfg_{cfg:04f}_dit_boxgpt_{uid}_bbox.glb")
    mesh_gt_bbox.export(save_dir / f"train_cfg_{cfg:04f}_input_boxgpt_{uid}.glb")
    export_tripo_assembly_glb(explode_object, save_dir / exploded_glb_name)
    
    # Generate standalone Tripo3D interactive viewer
    generate_tripo_viewer_html(
        glb_filename=assembled_glb_name,
        output_html_path=save_dir / viewer_html_name,
        model_title=f"{uid} - Segmented Assembly"
    )
    print(f"Successfully exported all meshes and interactive viewer to {save_dir}")
    total_time = time.perf_counter() - start_total
    print(f"\n================ TIMING REPORT ================")
    print(f"  Pipeline Inference & Reconstruction: {pipeline_time:.2f} s")
    print(f"  Total Run Time (incl. model loading): {total_time:.2f} s")
    print(f"================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
    )
    # input_dir
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="./data/test.glb",
    )
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument(
        "--aabb_path",
        type=str,
        default=None,
        help="Path to a precomputed aabb .npy (from a standalone P3-SAM run), "
             "skips X-Part's own internal bbox predictor.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Diffusion sampling steps. Use 20 to reproduce "
             "generated_parts_assembled_chunked20step.glb, 50 (default) to "
             "reproduce generated_parts_assembled_50step.glb.",
    )
    parser.add_argument(
        "--octree_resolution",
        type=int,
        default=512,
        help="Marching cubes grid resolution (512 default, 256 or 384 for 3-4x faster extraction).",
    )
    parser.add_argument(
        "--mc_algo",
        type=str,
        default="dmc",
        choices=["mc", "dmc"],
        help="Marching cubes algorithm: 'dmc' for GPU-accelerated DiffDMC (~70x faster), 'mc' for scikit-image on CPU.",
    )
    args = parser.parse_args()
    config = get_config_from_file(
        args.config
        if args.config
        else str(Path(__file__).parent / "partgen/config" / "infer.yaml")
    )
    assert hasattr(config, "ckpt") or hasattr(
        config, "ckpt_path"
    ), "ckpt or ckpt_path must be specified in config"
    ckpt_path = Path(args.ckpt if args.ckpt else config.ckpt_path)
    ignore_keys = config.get("ignore_keys", [])
    save_dir = (
        Path(args.save_dir)
        / ckpt_path.parent.parent.relative_to(
            ckpt_path.parent.parent.parent.parent.parent
        )
        # / ckpt_path.stem
    )
    print(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    main(
        ckpt_path=ckpt_path,
        config=config,
        mesh_path=args.mesh_path,
        save_dir=save_dir,
        ignore_keys=ignore_keys,
        aabb_path=args.aabb_path,
        num_inference_steps=args.num_inference_steps,
        octree_resolution=args.octree_resolution,
        mc_algo=args.mc_algo,
    )
