"""Errors raised by the Phase 0 notebook and helpers."""


class Phase0Error(Exception):
    """Base error. The message is meant to be shown in the notebook."""


class GpuUnavailableError(Phase0Error):
    """CUDA is not available, so the verification run cannot start."""


class TransformersVersionError(Phase0Error):
    """The installed transformers build cannot load Qwen3-VL."""


class PillowExtensionError(Phase0Error):
    """Pillow's Python files and compiled extension are not the same build."""


class ModelLoadError(Phase0Error):
    """The selected model failed to load. This does not select a fallback model."""


class ImageValidationError(Phase0Error):
    """An image path or download could not be used as model input."""


class InferenceFailedError(Phase0Error):
    """One generation call failed. The original exception is chained."""


class MeasurementError(Phase0Error):
    """A report tried to record a result that was not measured."""
