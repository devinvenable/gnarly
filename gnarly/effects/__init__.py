"""Effects module for image/video effects."""

from .base import Effect
from .ca_effect import CAEffect
from .pipeline import EffectPipeline
from .zoom_effect import ZoomEffect

# Lazy imports for GPU-dependent effects
def __getattr__(name):
    if name == "DeepDreamEffect":
        from .deep_dream_effect import DeepDreamEffect
        return DeepDreamEffect
    if name == "MorphEffect":
        from .morph_effect import MorphEffect
        return MorphEffect
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Effect",
    "CAEffect",
    "DeepDreamEffect",
    "MorphEffect",
    "EffectPipeline",
    "ZoomEffect",
]
