"""Detection module for object and face detection."""

from .base import Detection, Detector
from .face import FaceDetection, FaceDetector, detect_faces_and_landmarks, get_shape_predictor_path

__all__ = [
    "Detection",
    "Detector",
    "FaceDetection",
    "FaceDetector",
    "detect_faces_and_landmarks",
    "get_shape_predictor_path",
]
