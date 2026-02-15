"""Continuous zoom effect."""

import numpy as np

from ..core.utils import zoom_image


class ZoomEffect:
    """Continuous zoom effect that zooms in/out over time.

    Applies a smooth zoom effect that oscillates between min and max zoom levels.
    """

    def __init__(
        self,
        speed: float = 1.02,
        min_zoom: float = 1.0,
        max_zoom: float = 2.0,
    ):
        """Initialize the zoom effect.

        Args:
            speed: Zoom factor multiplier per frame (1.02 = 2% zoom per frame).
            min_zoom: Minimum zoom level (1.0 = no zoom).
            max_zoom: Maximum zoom level before reversing.
        """
        self.speed = speed
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self._current_zoom = min_zoom
        self._direction = 1  # 1 = zooming in, -1 = zooming out

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply zoom effect to frame.

        Args:
            frame: Input frame (RGB).

        Returns:
            Zoomed frame.
        """
        # Apply zoom at current level
        result = zoom_image(frame, self._current_zoom)

        # Update zoom for next frame
        if self._direction == 1:
            self._current_zoom *= self.speed
            if self._current_zoom >= self.max_zoom:
                self._current_zoom = self.max_zoom
                self._direction = -1
        else:
            self._current_zoom /= self.speed
            if self._current_zoom <= self.min_zoom:
                self._current_zoom = self.min_zoom
                self._direction = 1

        return result

    def reset(self) -> None:
        """Reset effect state."""
        self._current_zoom = self.min_zoom
        self._direction = 1
