import numpy as np
from gnarly.detection.base import Detection, Detector
from typing import List

def test_detection_dataclass():
    box = np.array([0, 0, 100, 100])
    d = Detection(box=box, score=0.9, label="person", class_id=1)
    assert np.array_equal(d.box, box)
    assert d.score == 0.9
    assert d.label == "person"
    assert d.class_id == 1
    assert d.mask is None

class MockDetector:
    def detect(self, frame: np.ndarray) -> List[Detection]:
        return [
            Detection(box=np.array([0, 0, 10, 10]), score=0.8, label="cat", class_id=2)
        ]

def test_detector_protocol():
    detector: Detector = MockDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert len(detections) == 1
    assert detections[0].label == "cat"

if __name__ == "__main__":
    test_detection_dataclass()
    test_detector_protocol()
    print("All tests passed!")
