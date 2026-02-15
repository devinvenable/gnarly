"""Base detection protocol and data structures."""

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np


@dataclass
class Detection:
    """Standardized detection result.

    Attributes:
        box: Bounding box in [x1, y1, x2, y2] pixel coordinates.
        score: Confidence score (0.0 to 1.0).
        label: Class label name.
        class_id: Class integer ID.
        mask: Optional segmentation mask.
    """

    box: np.ndarray
    score: float
    label: str
    class_id: int
    mask: Optional[np.ndarray] = None


class Detector(Protocol):
    """Protocol for object detectors."""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a frame.

        Args:
            frame: Input frame (RGB).

        Returns:
            List of Detection objects.
        """
        ...
