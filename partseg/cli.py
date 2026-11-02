"""Command line interface: ``python -m partseg segment|doctor``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import SegmentConfig
from .errors import InvalidInputError, PartsegError
from .io import SUPPORTED_EXTENSIONS

log = logging.getLogger("partseg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="partseg", description="Clean part segmentation of 3D meshes (P3-SAM based)."
    )
    parser.add_argument("--version", action="version", version=f"partseg {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    seg = sub.add_parser("segment", help="segment one or more meshes",
                         description="Segment meshes and write parts GLB, labels and reports.")
    seg.add_argument("inputs", nargs="+", help=f"mesh files or folders ({', '.join(SUPPORTED_EXTENSIONS)})")
    seg.add_argument("-o", "--output", required=True, help="output folder (one sub-folder per input when batching)")
    seg.add_argument("--preset", default="precise", choices=["precise", "detailed", "fast"],
                     help="precise (default): clean-up only; detailed: also split large parts; fast: shorter averaging")
    seg.add_argument("--config", help="JSON file with config overrides")
    seg.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                     help="override one config value, e.g. polish.stay_cost=0.05 (repeatable)")
    seg.add_argument("--seed", type=int, help="random seed (default 42)")
    seg.add_argument("--keep-materials", action="store_true", help="keep source textures/colours in the parts GLB")
    seg.add_argument("--no-preview", action="store_true", help="skip the preview image")
    seg.add_argument("--parts", type=int, metavar="N", help="merge neighbouring parts down to at most N parts")
    seg.add_argument("--seal", action="store_true", help="also write <name>_parts_sealed.glb with cut surfaces capped")
    seg.add_argument("--up-axis", choices=["auto", "y", "z"], help="up axis for the preview render only")
    seg.add_argument("--json", action="store_true", help="print a machine-readable summary on stdout")
    verbosity = seg.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true")
    verbosity.add_argument("-q", "--quiet", action="store_true")

    doc = sub.add_parser("doctor", help="check that this machine can run the pipeline")
    doc.add_argument("--load-model", action="store_true", help="also load the P3-SAM weights (slower)")
    return parser


def _collect_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
            if not found:
                raise InvalidInputError(f"No supported mesh files found in folder '{p}'")
            files.extend(found)
        elif p.exists():
            files.append(p)
        else:
            raise InvalidInputError(f"Input not found: {p}")
    return files


def _build_config(args) -> SegmentConfig:
    cfg = SegmentConfig.preset(args.preset)
    if args.config:
        cfg = SegmentConfig.from_json(args.config, base=cfg)
    if args.overrides:
        cfg = cfg.override(args.overrides)
    if args.seed is not None:
        cfg.sampling.seed = args.seed
    if args.keep_materials:
        cfg.export.keep_materials = True
    if args.no_preview:
        cfg.export.preview = False
    if args.parts is not None:
        cfg.target_parts = args.parts
    if args.seal:
        cfg.export.seal = True
    if args.up_axis:
        cfg.export.up_axis = args.up_axis
    cfg.validate()
    return cfg


def _cmd_segment(args) -> int:
    from .p3sam import load_model
    from .pipeline import segment_file

    cfg = _build_config(args)
    files = _collect_inputs(args.inputs)
    out_root = Path(args.output)
    single = len(files) == 1 and Path(args.inputs[0]).is_file()
    log.info("segmenting %d file(s) with preset '%s'", len(files), args.preset)
    model = load_model()
    summary, failures = [], 0
    for path in files:
        out_dir = out_root if single else out_root / path.stem
        try:
            result = segment_file(path, out_dir, cfg, model=model)
        except PartsegError as exc:
            failures += 1
            log.error("%s: %s", path.name, exc)
            summary.append({"input": str(path), "status": "error", "error": str(exc)})
            if single:
                return exc.exit_code
            continue
        for warning in result.warnings:
            log.warning("%s: %s", path.name, warning)
        log.info("%s: %d parts in %.1fs -> %s", path.name, len(result.parts),
                 result.stage_seconds["total"], out_dir)
        summary.append({"input": str(path), "status": "ok", "parts": len(result.parts),
                        "seconds": round(result.stage_seconds["total"], 2), "outputs": result.outputs,
                        "warnings": result.warnings})
    if args.json:
        print(json.dumps({"results": summary}, indent=2))
    return 1 if failures else 0


def _cmd_doctor(args) -> int:
    import importlib

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    add("python", sys.version_info >= (3, 9), sys.version.split()[0])
    for module, required in (("numpy", True), ("scipy", True), ("trimesh", True), ("sklearn", True),
                             ("cv2", True), ("fpsample", True), ("gco", True), ("torch", True),
                             ("spconv", True), ("torch_scatter", True)):
        try:
            mod = importlib.import_module(module)
            add(module, True, getattr(mod, "__version__", ""))
        except Exception as exc:
            add(module, not required, f"missing ({type(exc).__name__})")
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            add("cuda", True, f"{props.name}, {props.total_memory / 1e9:.0f} GB")
        else:
            add("cuda", False, "no CUDA device visible to PyTorch")
    except Exception as exc:
        add("cuda", False, str(exc))
    from .p3sam import P3SAM_DEMO

    add("P3-SAM sources", P3SAM_DEMO.is_dir(), str(P3SAM_DEMO))
    if args.load_model:
        try:
            from .p3sam import load_model

            load_model()
            add("P3-SAM weights", True, "loaded")
        except PartsegError as exc:
            add("P3-SAM weights", False, str(exc))
    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'OK  ' if ok else 'FAIL'} {name.ljust(width)}  {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 3


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.INFO
    if getattr(args, "verbose", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        return _cmd_segment(args) if args.command == "segment" else _cmd_doctor(args)
    except PartsegError as exc:
        log.error("%s", exc)
        return exc.exit_code
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
