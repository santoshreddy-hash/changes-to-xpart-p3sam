"""Typed configuration for the partseg pipeline.

Every tunable lives here with the value that was validated on the 8-mesh benchmark.
Configs can be built from presets, JSON files or ``section.key=value`` overrides.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import InvalidInputError


@dataclass
class SamplingConfig:
    """Surface sampling and prompt budget for the P3-SAM step."""

    min_points: int = 15000
    max_points: int = 100000
    points_per_face: float = 2.0
    base_prompt_cap: int = 300
    max_prompts: int = 400
    prompt_scale: float = 1.0
    prompt_batch: int = 8
    seed: int = 42


@dataclass
class P3SamConfig:
    """Multi-mask P3-SAM with stability filtering, second pass and graph cut."""

    nms_thresh: float = 0.7
    extra_prompts_cap: int = 60
    conf_threshold: float = 0.60
    curvature_threshold: float = 0.25
    min_region_size: int = 10
    feature_tta: int = 16  # backbone passes averaged over voxel-sampling seeds; cancels P3-SAM's seed noise


@dataclass
class RepromptConfig:
    """Re-prompt inside large parts and split them along multi-prompt consensus masks."""

    enabled: bool = True
    rounds: int = 3
    big_frac: float = 0.08
    prompts: int = 64
    batch: int = 8
    min_frac: float = 0.04
    max_frac: float = 0.70
    min_iou: float = 0.15
    stability: float = 0.75
    contain: float = 0.28
    nms_overlap: float = 0.45
    max_new: int = 8
    max_global: float = 0.55
    planar_guard: float = 0.85
    min_support: int = 2
    support_iou: float = 0.6
    starts: int = 3


@dataclass
class FeatureSplitConfig:
    """Split large organic parts along P3-SAM feature clusters validated by the mask decoder."""

    enabled: bool = True
    big_frac: float = 0.06
    planar_guard: float = 0.85
    tau: float = 1.3
    tau_big: float = 1.2
    huge_area: float = 0.25
    k_max: int = 6
    min_frac: float = 0.10
    iou_min: float = 0.4
    n_super: int = 300


@dataclass
class PolishConfig:
    """Boundary clean-up: fragments, creases, graph cut and smoothing."""

    enabled: bool = True
    min_area_frac: float = 0.003
    speck_frac: float = 0.0001
    stay_cost: float = 0.06
    crease_sigma: float = 0.55
    crease_floor: float = 0.05
    length_weight: float = 1.0
    smooth_radius: float = 0.02
    panel_area: float = 0.04
    panel_planar: float = 0.9
    panel_margin: float = 0.03


@dataclass
class ExportConfig:
    """Which files segment_file() writes."""

    keep_materials: bool = False
    preview: bool = True
    preview_size: int = 640
    up_axis: str = "auto"  # auto | y | z (only affects the preview render)


@dataclass
class SegmentConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    p3sam: P3SamConfig = field(default_factory=P3SamConfig)
    reprompt: RepromptConfig = field(default_factory=RepromptConfig)
    features: FeatureSplitConfig = field(default_factory=FeatureSplitConfig)
    polish: PolishConfig = field(default_factory=PolishConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    max_faces: int = 2_000_000

    # ------------------------------------------------------------------ presets
    @classmethod
    def preset(cls, name: str = "balanced") -> "SegmentConfig":
        cfg = cls()
        if name == "balanced":
            return cfg
        if name == "fast":
            cfg.reprompt.rounds = 1
            cfg.reprompt.starts = 1
            cfg.features.enabled = False
            cfg.p3sam.feature_tta = 4
            return cfg
        raise InvalidInputError(f"Unknown preset '{name}'. Available presets: balanced, fast")

    # ---------------------------------------------------------------- (de)serialize
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict, base: "SegmentConfig | None" = None) -> "SegmentConfig":
        cfg = copy.deepcopy(base) if base is not None else cls()
        _apply(cfg, data, path="")
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path, base: "SegmentConfig | None" = None) -> "SegmentConfig":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidInputError(f"Cannot read config file '{path}': {exc}") from exc
        if not isinstance(data, dict):
            raise InvalidInputError(f"Config file '{path}' must contain a JSON object")
        return cls.from_dict(data, base=base)

    def override(self, assignments: list[str]) -> "SegmentConfig":
        """Apply ``section.key=value`` strings (e.g. ``polish.stay_cost=0.05``)."""
        cfg = copy.deepcopy(self)
        for item in assignments:
            if "=" not in item:
                raise InvalidInputError(f"Override '{item}' must look like section.key=value")
            key, raw = item.split("=", 1)
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            node: dict[str, Any] = {}
            cursor = node
            parts = key.strip().split(".")
            for part in parts[:-1]:
                cursor[part] = {}
                cursor = cursor[part]
            cursor[parts[-1]] = value
            _apply(cfg, node, path="")
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        s, r, f, p = self.sampling, self.reprompt, self.features, self.polish
        problems = []
        if not 1000 <= s.min_points <= s.max_points:
            problems.append("sampling.min_points must be >= 1000 and <= sampling.max_points")
        if s.points_per_face <= 0 or s.prompt_scale <= 0:
            problems.append("sampling.points_per_face and sampling.prompt_scale must be positive")
        if s.prompt_batch < 1 or s.max_prompts < 1:
            problems.append("sampling.prompt_batch and sampling.max_prompts must be >= 1")
        if r.rounds < 0 or r.starts < 1 or r.min_support < 1:
            problems.append("reprompt.rounds >= 0, reprompt.starts >= 1 and reprompt.min_support >= 1 are required")
        if not 0 < f.min_frac < 0.5 or f.k_max < 2:
            problems.append("features.min_frac must be in (0, 0.5) and features.k_max >= 2")
        if not 0 <= p.min_area_frac < 0.2 or p.stay_cost < 0:
            problems.append("polish.min_area_frac must be in [0, 0.2) and polish.stay_cost >= 0")
        if self.export.up_axis not in ("auto", "y", "z"):
            problems.append("export.up_axis must be auto, y or z")
        if self.max_faces < 1:
            problems.append("max_faces must be positive")
        if problems:
            raise InvalidInputError("Invalid configuration: " + "; ".join(problems))


def _apply(node: Any, data: dict, path: str) -> None:
    """Recursively copy ``data`` onto the dataclass ``node``, rejecting unknown keys and wrong types."""
    valid = {f.name: f for f in dataclasses.fields(node)}
    for key, value in data.items():
        where = f"{path}{key}"
        if key not in valid:
            raise InvalidInputError(f"Unknown config key '{where}'. Valid keys here: {', '.join(sorted(valid))}")
        current = getattr(node, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise InvalidInputError(f"Config section '{where}' must be an object")
            _apply(current, value, path=where + ".")
            continue
        setattr(node, key, _coerce(where, current, value))


def _coerce(where: str, current: Any, value: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        raise InvalidInputError(f"Config value '{where}' must be true or false, got {value!r}")
    if isinstance(current, int):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise InvalidInputError(f"Config value '{where}' must be an integer, got {value!r}")
    if isinstance(current, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise InvalidInputError(f"Config value '{where}' must be a number, got {value!r}")
    if isinstance(current, str):
        if isinstance(value, str):
            return value
        raise InvalidInputError(f"Config value '{where}' must be a string, got {value!r}")
    return value
