"""Effect composition pipeline."""

import numpy as np

from .base import Effect


class EffectPipeline:
    """Chain multiple effects and apply them in sequence."""

    def __init__(self, effects: list[Effect]):
        """Initialize pipeline with ordered effects.

        Args:
            effects: Effects to apply in order.
        """
        self.effects = list(effects)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all effects to a frame in sequence.

        Args:
            frame: Input frame.

        Returns:
            The processed frame after all effects.
        """
        output = frame
        for effect in self.effects:
            output = effect.apply(output)
        return output

    def reset(self) -> None:
        """Reset all effect states in the pipeline."""
        for effect in self.effects:
            effect.reset()
