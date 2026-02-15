"""Video and image I/O utilities."""

from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_video_file(filename: str) -> bool:
    """Check if filename has a video extension.

    Args:
        filename: Path to file.

    Returns:
        True if file has a video extension.
    """
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS


class VideoReader:
    """Read frames from a video file or image."""

    def __init__(self, path: str):
        """Initialize the reader.

        Args:
            path: Path to video or image file.
        """
        self.path = path
        self.is_video = is_video_file(path)
        self._cap = None
        self._image = None
        self._fps = 15
        self._frame_count = 1

        if self.is_video:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise IOError(f"Cannot open video file: {path}")
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self._image = np.array(Image.open(path).convert("RGB"))

    @property
    def fps(self) -> float:
        """Get frames per second."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Get total frame count (1 for images)."""
        return self._frame_count

    def read_frame(self) -> Tuple[bool, np.ndarray | None]:
        """Read the next frame.

        Returns:
            Tuple of (success, frame). Frame is RGB numpy array.
        """
        if self.is_video:
            ret, frame = self._cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return ret, frame
        else:
            if self._image is not None:
                img = self._image
                self._image = None  # Only return once
                return True, img
            return False, None

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames."""
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            yield frame

    def close(self):
        """Release resources."""
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoWriter:
    """Write frames to a video file."""

    def __init__(self, path: str, width: int, height: int, fps: float = 15.0):
        """Initialize the writer.

        Args:
            path: Output video path.
            width: Frame width.
            height: Frame height.
            fps: Frames per second.
        """
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise IOError(f"Cannot create video writer: {path}")

    def write_frame(self, frame: np.ndarray):
        """Write a frame to the video.

        Args:
            frame: RGB numpy array.
        """
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr_frame)

    def close(self):
        """Release resources."""
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
