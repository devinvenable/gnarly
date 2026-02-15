"""Cellular automata overlay effect."""

import cv2
import numpy as np

from ..core.ca import CAEngine


class CAEffect:
    """Cellular automata overlay effect.

    Applies a CA grid as a mask to blend between original
    and accumulated frames.
    """

    def __init__(
        self,
        width: int,
        height: int,
        grid_scale: int = 4,
        rule: str = "Conway",
        divisor: int = 2,
        blend_alpha: float = 0.3,
    ):
        """Initialize the CA effect.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            grid_scale: Size of each CA cell in pixels.
            rule: CA rule name.
            divisor: Divisor for DivisorRule.
            blend_alpha: Blend factor for live cells.
        """
        self.width = width
        self.height = height
        self.grid_scale = grid_scale
        self.blend_alpha = blend_alpha

        # CA grid dimensions
        grid_width = width // grid_scale
        grid_height = height // grid_scale

        self.engine = CAEngine(grid_width, grid_height, rule, divisor)
        self._accumulated = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply CA overlay effect to frame.

        Args:
            frame: Input frame (RGB).

        Returns:
            Processed frame with CA overlay.
        """
        # Initialize accumulated image on first frame
        if self._accumulated is None:
            self._accumulated = frame.copy()

        # Step the CA
        grid = self.engine.step()

        # Reinitialize if grid dies
        if self.engine.is_dead():
            self.engine.reset()
            grid = self.engine.grid

        # Upscale grid to frame size
        upscaled = cv2.resize(
            grid.astype(np.float32),
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        )

        # Create masks
        alive_mask = upscaled[:, :, np.newaxis]
        dead_mask = 1 - alive_mask

        # Blend: live cells show blend of frame and accumulated,
        # dead cells fade toward black (creating visible CA pattern)
        fade_factor = 0.95  # Dead cells fade slightly each frame

        composite = (
            frame.astype(np.float32) * self.blend_alpha
            + self._accumulated.astype(np.float32) * (1 - self.blend_alpha)
        ) * alive_mask + (self._accumulated.astype(np.float32) * fade_factor) * dead_mask

        result = np.clip(composite, 0, 255).astype(np.uint8)

        # Update accumulated
        self._accumulated = result.copy()

        return result

    def reset(self) -> None:
        """Reset effect state."""
        self.engine.reset()
        self._accumulated = None
