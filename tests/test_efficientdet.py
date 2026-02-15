import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from gnarly.detection.efficientdet import EfficientDetDetector

class TestEfficientDetDetector(unittest.TestCase):
    @patch('gnarly.detection.efficientdet.create_model')
    @patch('gnarly.detection.efficientdet.DetBenchPredict')
    def test_init(self, mock_predict, mock_create):
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        mock_predict_instance = MagicMock()
        mock_predict_instance.to.return_value = mock_predict_instance
        mock_predict.return_value = mock_predict_instance
        
        # Test with CPU to avoid CUDA requirements in tests
        detector = EfficientDetDetector(device='cpu')
        
        self.assertEqual(detector.confidence_threshold, 0.5)
        mock_create.assert_called_once_with('tf_efficientdet_d0', pretrained=True)
        mock_predict.assert_called_once_with(mock_model)
        mock_predict_instance.to.assert_called()
        mock_predict_instance.eval.assert_called_once()

    @patch('gnarly.detection.efficientdet.create_model')
    @patch('gnarly.detection.efficientdet.DetBenchPredict')
    def test_detect_empty(self, mock_predict, mock_create):
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        mock_predict_instance = MagicMock()
        mock_predict_instance.to.return_value = mock_predict_instance
        mock_predict.return_value = mock_predict_instance
        
        # Mock model call
        mock_predict_instance.return_value = torch.zeros((1, 10, 6))
        
        detector = EfficientDetDetector(device='cpu')
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        self.assertEqual(len(detections), 0)

    @patch('gnarly.detection.efficientdet.create_model')
    @patch('gnarly.detection.efficientdet.DetBenchPredict')
    def test_detect_objects(self, mock_predict, mock_create):
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        mock_predict_instance = MagicMock()
        mock_predict_instance.to.return_value = mock_predict_instance
        mock_predict.return_value = mock_predict_instance
        
        # Mock model output: tensor of shape [1, 2, 6]
        # [x1, y1, x2, y2, score, label]
        detections = torch.tensor([[
            [10, 20, 30, 40, 0.9, 1.0],
            [50, 60, 70, 80, 0.1, 2.0]  # Should be filtered out by threshold
        ]])
        mock_predict_instance.return_value = detections
        
        detector = EfficientDetDetector(confidence_threshold=0.5, device='cpu')
        # Frame size 512x512 to match target_size, so no scaling needed for simple test
        frame = np.zeros((512, 512, 3), dtype=np.uint8)
        results = detector.detect(frame)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].label, "1")
        self.assertAlmostEqual(results[0].score, 0.9, places=6)
        np.testing.assert_array_equal(results[0].box, np.array([10, 20, 30, 40]))

    @patch('gnarly.detection.efficientdet.create_model')
    @patch('gnarly.detection.efficientdet.DetBenchPredict')
    def test_detect_scaling(self, mock_predict, mock_create):
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        mock_predict_instance = MagicMock()
        mock_predict_instance.to.return_value = mock_predict_instance
        mock_predict.return_value = mock_predict_instance
        
        # Mock model output: [10, 20, 30, 40, 0.9, 1.0] on 512x512 input
        detections = torch.tensor([[[10, 20, 30, 40, 0.9, 1.0]]])
        mock_predict_instance.return_value = detections
        
        detector = EfficientDetDetector(confidence_threshold=0.5, device='cpu')
        # Frame size 1024x1024, so boxes should be scaled by 2x
        frame = np.zeros((1024, 1024, 3), dtype=np.uint8)
        results = detector.detect(frame)
        
        self.assertEqual(len(results), 1)
        # 10 * (1024/512) = 20
        # 20 * (1024/512) = 40
        # 30 * (1024/512) = 60
        # 40 * (1024/512) = 80
        np.testing.assert_array_equal(results[0].box, np.array([20, 40, 60, 80]))

if __name__ == '__main__':
    unittest.main()
