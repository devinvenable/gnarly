"""Image utility functions."""

import numpy as np
import cv2


def blend_images(
    img1: np.ndarray, img2: np.ndarray, alpha: np.ndarray | float
) -> np.ndarray:
    """Blend two images using alpha mask or scalar.

    Args:
        img1: First image (background).
        img2: Second image (foreground).
        alpha: Blend factor (0-1). Can be scalar or 2D/3D array.

    Returns:
        Blended image as uint8 array.

    Raises:
        ValueError: If image shapes don't match.
    """
    if img2 is None:
        return img1.astype(np.uint8)

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    if isinstance(alpha, np.ndarray) and alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    blended = img1.astype(np.float32) * (1 - alpha) + img2.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def zoom_image(image: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Zoom into the center of an image.

    Args:
        image: Input image.
        zoom_factor: Zoom level (1.0 = no zoom, 2.0 = 2x zoom).

    Returns:
        Zoomed image at original dimensions.
    """
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    zoom_width = int(width / zoom_factor)
    zoom_height = int(height / zoom_factor)

    x1 = max(0, center_x - zoom_width // 2)
    y1 = max(0, center_y - zoom_height // 2)
    x2 = min(width, x1 + zoom_width)
    y2 = min(height, y1 + zoom_height)

    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
