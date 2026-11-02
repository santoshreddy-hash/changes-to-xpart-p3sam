"""partseg - clean, reproducible part segmentation of 3D meshes on top of P3-SAM.

    from partseg import segment_file, SegmentConfig
    result = segment_file("chair.glb", "out/chair", SegmentConfig.preset("balanced"))

The heavy imports (torch, the P3-SAM runtime) happen lazily inside the functions.
"""

__version__ = "1.0.0"

from .config import SegmentConfig  # noqa: E402
from .errors import (  # noqa: E402
    EnvironmentNotReadyError,
    InvalidInputError,
    PartsegError,
    PipelineError,
)

__all__ = [
    "SegmentConfig",
    "segment_mesh",
    "segment_file",
    "load_model",
    "PartsegError",
    "InvalidInputError",
    "EnvironmentNotReadyError",
    "PipelineError",
    "__version__",
]


def segment_mesh(*args, **kwargs):
    """See partseg.pipeline.segment_mesh."""
    from .pipeline import segment_mesh as _segment_mesh

    return _segment_mesh(*args, **kwargs)


def segment_file(*args, **kwargs):
    """See partseg.pipeline.segment_file."""
    from .pipeline import segment_file as _segment_file

    return _segment_file(*args, **kwargs)


def load_model(*args, **kwargs):
    """Load the P3-SAM model once and pass it to segment_mesh()/segment_file() for batches."""
    from .p3sam import load_model as _load_model

    return _load_model(*args, **kwargs)
