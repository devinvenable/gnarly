"""Face detection and landmark extraction using dlib."""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .base import Detection


@dataclass
class FaceDetection(Detection):
    """Face detection with landmarks.

    Attributes:
        box: Bounding box in [x1, y1, x2, y2] pixel coordinates.
        score: Confidence score (1.0 for dlib detections).
        label: Always "face".
        class_id: Always 0.
        landmarks: 68-point facial landmarks as (68, 2) array.
    """

    landmarks: np.ndarray = None


def get_shape_predictor_path() -> str:
    """Get path to dlib shape predictor, downloading if needed.

    Downloads the 68-point facial landmark model from HuggingFace Hub
    if not present locally.

    Returns:
        Path to shape_predictor_68_face_landmarks.dat file.

    Raises:
        SystemExit: If download fails.
    """
    predictor_filename = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_filename):
        print(f"'{predictor_filename}' not found. Downloading from Hugging Face Hub...")
        try:
            from huggingface_hub import hf_hub_download

            predictor_path = hf_hub_download(
                repo_id="public-data/dlib_face_landmark_model",
                filename=predictor_filename,
            )
            print(f"Downloaded predictor to {predictor_path}")
        except ImportError:
            print("Please install huggingface_hub with 'pip install huggingface_hub'")
            sys.exit(1)
        except Exception as e:
            print(f"Error downloading predictor: {e}")
            sys.exit(1)
    else:
        predictor_path = predictor_filename
    return predictor_path


def detect_faces_and_landmarks(
    image_array: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Detect faces and extract 68-point landmarks.

    Args:
        image_array: Input image as RGB numpy array.

    Returns:
        Tuple of (boxes, landmarks_list) where:
            - boxes: (N, 4) array of [x1, y1, x2, y2] bounding boxes
            - landmarks_list: List of N (68, 2) landmark arrays
    """
    import dlib

    detector = dlib.get_frontal_face_detector()
    predictor_path = get_shape_predictor_path()
    predictor = dlib.shape_predictor(predictor_path)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)

    boxes = []
    landmarks_list = []

    for rect in rects:
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        boxes.append([x1, y1, x2, y2])

        shape = predictor(gray, rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        landmarks_list.append(landmarks)

    print(f"Detected {len(landmarks_list)} face(s)")
    return np.array(boxes), landmarks_list


class FaceDetector:
    """Face detector using dlib frontal face detector with landmarks."""

    def __init__(self):
        """Initialize detector (lazy loads dlib models on first use)."""
        self._detector = None
        self._predictor = None

    def _ensure_initialized(self):
        """Lazy initialization of dlib models."""
        if self._detector is None:
            import dlib

            self._detector = dlib.get_frontal_face_detector()
            predictor_path = get_shape_predictor_path()
            self._predictor = dlib.shape_predictor(predictor_path)

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a frame.

        Args:
            frame: Input frame (RGB).

        Returns:
            List of FaceDetection objects with landmarks.
        """
        self._ensure_initialized()

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        rects = self._detector(gray, 1)

        detections = []
        for rect in rects:
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            box = np.array([x1, y1, x2, y2])

            shape = self._predictor(gray, rect)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            detections.append(
                FaceDetection(
                    box=box,
                    score=1.0,
                    label="face",
                    class_id=0,
                    landmarks=landmarks,
                )
            )

        return detections
