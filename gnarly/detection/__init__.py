"""Detection module for object and face detection."""

from .base import Detection, Detector

# Lazy imports for GPU-dependent detectors
def __getattr__(name):
    if name == "YOLODetector":
        from .yolo import YOLODetector
        return YOLODetector
    if name == "EfficientDetDetector":
        from .efficientdet import EfficientDetDetector
        return EfficientDetDetector
    if name in ("FaceDetection", "FaceDetector", "detect_faces_and_landmarks", "get_shape_predictor_path"):
        from . import face
        return getattr(face, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Detection",
    "Detector",
    "FaceDetection",
    "FaceDetector",
    "detect_faces_and_landmarks",
    "get_shape_predictor_path",
    "YOLODetector",
    "EfficientDetDetector",
]
