"""Base effect protocol."""

from typing import Protocol

import numpy as np


class Effect(Protocol):
    """Protocol for image effects."""

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply effect to a frame.

        Args:
            frame: Input frame (RGB).

        Returns:
            Processed frame (RGB).
        """
        ...

    def reset(self) -> None:
        """Reset effect state."""
        ...
