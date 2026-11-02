"""Adapter around the P3-SAM demo code: model loading and the multi-mask enhanced pass."""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .config import SegmentConfig
from .errors import EnvironmentNotReadyError, PipelineError

log = logging.getLogger("partseg")

REPO_ROOT = Path(__file__).resolve().parent.parent
P3SAM_DEMO = REPO_ROOT / "P3-SAM" / "demo"


def p3sam_module():
    """Import (once) the P3-SAM demo module that implements mesh_sam()."""
    if not P3SAM_DEMO.is_dir():
        raise EnvironmentNotReadyError(f"P3-SAM sources not found at {P3SAM_DEMO}")
    if str(P3SAM_DEMO) not in sys.path:
        sys.path.insert(0, str(P3SAM_DEMO))
    try:
        import auto_mask_fast  # noqa: WPS433
    except Exception as exc:
        raise EnvironmentNotReadyError(
            f"Could not import the P3-SAM runtime ({exc}). "
            "Install the packages from requirements-partseg.txt in a CUDA-enabled environment."
        ) from exc
    return auto_mask_fast


def check_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise EnvironmentNotReadyError(
            "CUDA is not available. P3-SAM needs an NVIDIA GPU with a CUDA-enabled PyTorch build."
        )


def load_model(ckpt_path: str | None = None):
    """Load the P3-SAM checkpoint onto the GPU (downloads it on first use unless cached)."""
    check_cuda()
    mod = p3sam_module()
    t0 = time.perf_counter()
    sink = io.StringIO()  # the loader prints download banners even when the weights are cached
    try:
        with contextlib.redirect_stdout(sink):
            model = mod.AutoMask(ckpt_path=ckpt_path)
        log.debug("model loader output: %s", sink.getvalue().strip())
    except Exception as exc:
        raise EnvironmentNotReadyError(
            f"Could not load the P3-SAM weights ({exc}). If the machine is offline, "
            "download the checkpoint once on a connected machine and copy the cache folder."
        ) from exc
    log.info("P3-SAM model ready in %.1fs", time.perf_counter() - t0)
    return model


def sampling_params(n_faces: int, cfg: SegmentConfig) -> tuple[int, int]:
    """Number of surface points and first-pass prompts for a mesh of n_faces."""
    s = cfg.sampling
    n_pts = int(min(s.max_points, max(s.min_points, n_faces * s.points_per_face)))
    base = min(s.base_prompt_cap, max(60, int(np.sqrt(n_faces) * 2.5)))
    n_prompts = int(min(s.max_prompts, int(base * s.prompt_scale)))
    return n_pts, n_prompts


def run_enhanced(model, mesh, cfg: SegmentConfig, n_pts: int, n_prompts: int, prompt_batch: int | None = None,
                 feature_seed: int | None = None):
    """
    Run multi-mask P3-SAM (stability filter, second pass, graph cut) and return
    (face_ids, metadata). metadata carries the backbone features, the sampled
    points and the face each point lies on, which later stages reuse.
    """
    import torch

    mod = p3sam_module()
    s, p = cfg.sampling, cfg.p3sam
    extra = min(p.extra_prompts_cap, n_prompts // 4)
    batch = prompt_batch or s.prompt_batch
    sink = io.StringIO()  # the demo code prints progress timers; keep them out of the user's console
    try:
        with contextlib.redirect_stdout(sink), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, fids, res = mod.mesh_sam(
                [model.model, model.model], mesh.copy(), save_path=None,
                point_num=n_pts, prompt_num=n_prompts, threshold=0.95, post_process=False,
                show_info=False, save_mid_res=False, clean_mesh_flag=False, seed=s.seed,
                prompt_bs=batch, loop_mode="opt3", nms_mode="fast", cleanup_mode="fast",
                multimask_output=True, stability_filter=True, nms_thresh=p.nms_thresh,
                refine_mode="graphcut", preserve_raw_faces=True, second_pass_enabled=True,
                max_additional_prompts=extra, conf_threshold=p.conf_threshold,
                curvature_threshold=p.curvature_threshold, min_region_size=p.min_region_size,
                feature_tta=p.feature_tta, feature_seed=feature_seed,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if batch > 2 and n_pts > s.min_points:
            log.warning("GPU out of memory: retrying with a smaller batch and fewer points")
            return run_enhanced(model, mesh, cfg, max(s.min_points, int(n_pts * 0.6)),
                                n_prompts, prompt_batch=max(2, batch // 2), feature_seed=feature_seed)
        raise PipelineError("The GPU ran out of memory even with reduced settings") from None
    except Exception as exc:
        raise PipelineError(f"P3-SAM failed: {exc}") from exc
    finally:
        log.debug("P3-SAM stdout:\n%s", sink.getvalue())
    face_ids = np.asarray(fids).astype(np.int64).copy()
    meta = {
        "feats": res.metadata["point_feats"],
        "points": np.asarray(res.metadata["sampled_points"]),
        "face_idx": np.asarray(res.metadata["face_idx"]),
    }
    return face_ids, meta
