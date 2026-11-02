"""The production entry points: segment_mesh() and segment_file()."""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .config import SegmentConfig
from .errors import PartsegError, PipelineError
from .io import load_mesh, validate_mesh
from .polish import MeshGraph, boundary_stats, count_disconnected, polish_labels

log = logging.getLogger("partseg")

Progress = Callable[[str, float], None]


@dataclass
class SegmentResult:
    face_ids: np.ndarray
    parts: list
    stage_seconds: dict
    warnings: list
    metrics: dict
    report: dict
    outputs: dict = field(default_factory=dict)
    mesh: object = None
    graph: object = None
    stage_labels: dict = field(default_factory=dict)


def _stage(progress: Progress | None, name: str, fraction: float) -> None:
    log.info("stage: %s", name)
    if progress is not None:
        progress(name, fraction)


def _ensure_labeled(face_ids: np.ndarray, graph: MeshGraph) -> np.ndarray:
    """Every face must carry a label: give stragglers the label of the nearest labelled face."""
    missing = face_ids < 0
    if not missing.any():
        return face_ids
    from scipy.spatial import cKDTree

    have = np.where(~missing)[0]
    if len(have) == 0:
        return np.zeros_like(face_ids)
    out = face_ids.copy()
    _, nn = cKDTree(graph.centroids[have]).query(graph.centroids[missing])
    out[missing] = face_ids[have[nn]]
    return out


def segment_mesh(mesh, config: SegmentConfig | None = None, model=None,
                 progress: Progress | None = None, info: dict | None = None) -> SegmentResult:
    """
    Segment a trimesh.Trimesh into clean parts.

    Pass a loaded ``model`` (partseg.p3sam.load_model()) to reuse it across many meshes.
    All seeds are fixed, so a run is repeatable within a process. Across separate runs the GPU kernels
    can differ slightly, which may move a few part boundaries or change the part count by one or two.
    Optional stages that fail are skipped with a warning; only the base P3-SAM pass is fatal.
    """
    from . import p3sam, subdivide

    cfg = config or SegmentConfig()
    cfg.validate()
    if info is None:
        mesh, info = validate_mesh(mesh, name="mesh", max_faces=cfg.max_faces)
    t_total = time.perf_counter()
    times: dict[str, float] = {}
    warnings: list[str] = []

    if model is None:
        _stage(progress, "loading model", 0.02)
        t0 = time.perf_counter()
        model = p3sam.load_model()
        times["model_load"] = time.perf_counter() - t0
    else:
        p3sam.check_cuda()

    graph = MeshGraph(mesh)
    normals = np.asarray(mesh.face_normals)
    n_pts, n_prompts = p3sam.sampling_params(len(mesh.faces), cfg)

    _stage(progress, "P3-SAM multi-mask pass", 0.10)
    t0 = time.perf_counter()
    face_ids, meta = p3sam.run_enhanced(model, mesh, cfg, n_pts, n_prompts)
    times["p3sam"] = time.perf_counter() - t0
    base_parts = int(len(np.unique(face_ids)))
    feats_gpu, pts, face_idx = meta["feats"], meta["points"], meta["face_idx"]
    stage_counts = {"p3sam": base_parts}
    stage_labels = {"p3sam": face_ids.copy()}

    if cfg.reprompt.enabled and cfg.reprompt.rounds > 0:
        _stage(progress, "re-prompting large parts", 0.55)
        t0 = time.perf_counter()
        try:
            for _ in range(cfg.reprompt.rounds):
                face_ids, added = subdivide.reprompt_large_parts(
                    model.model, feats_gpu, pts, face_idx, face_ids, graph, normals, cfg.reprompt)
                if not added:
                    break
            stage_counts["reprompt"] = int(len(np.unique(face_ids)))
            stage_labels["reprompt"] = face_ids.copy()
        except Exception as exc:  # optional refinement: keep the previous labels
            warnings.append(f"re-prompting skipped: {exc}")
            log.warning("re-prompting skipped: %s", exc)
        times["reprompt"] = time.perf_counter() - t0

    if cfg.features.enabled:
        _stage(progress, "feature-guided splitting", 0.70)
        t0 = time.perf_counter()
        try:
            feats_np = feats_gpu.float().cpu().numpy().astype(np.float32)
            face_ids, _ = subdivide.split_by_features(
                model.model, feats_gpu, feats_np, pts, face_idx, face_ids, graph, normals, cfg.features)
            stage_counts["features"] = int(len(np.unique(face_ids)))
            stage_labels["features"] = face_ids.copy()
        except Exception as exc:
            warnings.append(f"feature splitting skipped: {exc}")
            log.warning("feature splitting skipped: %s", exc)
        times["features"] = time.perf_counter() - t0

    _stage(progress, "boundary clean-up", 0.85)
    t0 = time.perf_counter()
    if cfg.polish.enabled:
        try:
            face_ids = polish_labels(face_ids, graph, cfg.polish, log=log)
        except Exception as exc:
            warnings.append(f"boundary clean-up failed, using raw labels: {exc}")
            log.warning("boundary clean-up failed: %s", exc)
    face_ids = _ensure_labeled(face_ids, graph)
    # contiguous ids ordered by area so part_000 is always the largest
    from .polish import renumber_by_area

    face_ids = renumber_by_area(face_ids, graph.areas)
    times["polish"] = time.perf_counter() - t0
    stage_counts["final"] = int(len(np.unique(face_ids)))
    times["total"] = time.perf_counter() - t_total

    from .export import build_parts_table

    parts = build_parts_table(mesh, face_ids, graph)
    metrics = boundary_stats(face_ids, graph)
    metrics["disconnected_parts"] = int(count_disconnected(face_ids, graph))
    report = {
        "schema_version": 1,
        "partseg_version": __version__,
        "input": info,
        "config": cfg.to_dict(),
        "sampling": {"points": n_pts, "prompts": n_prompts},
        "parts_after_stage": stage_counts,
        "num_parts": len(parts),
        "metrics": metrics,
        "stage_seconds": {k: round(v, 3) for k, v in times.items()},
        "warnings": warnings,
        "environment": _environment(),
    }
    return SegmentResult(face_ids=face_ids, parts=parts, stage_seconds=times, warnings=warnings,
                         metrics=metrics, report=report, mesh=mesh, graph=graph,
                         stage_labels={**stage_labels, "final": face_ids.copy()})


def segment_file(path: str | Path, out_dir: str | Path, config: SegmentConfig | None = None, model=None,
                 progress: Progress | None = None) -> SegmentResult:
    """Load ``path``, segment it and write the structured outputs into ``out_dir``."""
    cfg = config or SegmentConfig()
    cfg.validate()
    path = Path(path)
    mesh, info = load_mesh(path, max_faces=cfg.max_faces)
    result = segment_mesh(mesh, cfg, model=model, progress=progress, info=info)
    _stage(progress, "writing outputs", 0.95)
    from .export import write_outputs

    preview = _render_preview(mesh, result.face_ids, cfg, path)
    try:
        result.outputs = write_outputs(Path(out_dir), path.stem, mesh, result.face_ids,
                                       result.graph, result.parts, result.report, cfg.export, preview)
    except PartsegError:
        raise
    except Exception as exc:
        raise PipelineError(f"Could not write outputs to '{out_dir}': {exc}") from exc
    result.report["outputs"] = result.outputs
    _stage(progress, "done", 1.0)
    return result


def _render_preview(mesh, face_ids, cfg: SegmentConfig, path: Path):
    if not cfg.export.preview:
        return None
    try:
        import numpy as np
        from .render import render_parts

        axis = cfg.export.up_axis
        y_up = axis == "y" or (axis == "auto" and path.suffix.lower() in (".glb", ".gltf"))
        size = cfg.export.preview_size
        views = [render_parts(mesh, face_ids, size=size, azim=a, elev=18, y_up=y_up) for a in (35, 215)]
        return np.concatenate(views, axis=1)
    except Exception as exc:  # a preview is a convenience, never a reason to fail
        log.warning("preview skipped: %s", exc)
        return None


def _environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch always present when we got this far
        pass
    return env
