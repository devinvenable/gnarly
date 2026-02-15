"""Effects module for image/video effects."""

from .base import Effect
from .ca_effect import CAEffect
from .deep_dream_effect import DeepDreamEffect
from .pipeline import EffectPipeline
from .zoom_effect import ZoomEffect

__all__ = ["Effect", "CAEffect", "DeepDreamEffect", "EffectPipeline", "ZoomEffect"]
