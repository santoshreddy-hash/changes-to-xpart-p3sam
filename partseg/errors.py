"""Exceptions raised by partseg. Each carries the process exit code the CLI uses."""


class PartsegError(Exception):
    """Base class for every error partseg raises on purpose."""

    exit_code = 4


class InvalidInputError(PartsegError):
    """The input file is missing, unsupported, empty or corrupt."""

    exit_code = 2


class EnvironmentNotReadyError(PartsegError):
    """CUDA, model weights or a required library is unavailable."""

    exit_code = 3


class PipelineError(PartsegError):
    """The segmentation pipeline failed after the input was accepted."""

    exit_code = 4
