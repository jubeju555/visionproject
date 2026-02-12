"""
Unit tests for Vision Engine implementation.

Tests the MediaPipeVisionEngine class including:
- Initialization
- Camera capture
- MediaPipe integration
- Thread safety
- Data structure output
- Smoothing functionality
- Graceful shutdown
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch
import cv2

from src.vision.vision_engine_impl import MediaPipeVisionEngine, VisionData


class TestVisionData:
    """Test VisionData structure."""

    def test_vision_data_creation(self):
        """Test VisionData can be created with required fields."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        data = VisionData(frame=frame, frame_rgb=frame_rgb)

        assert data.frame is not None
        assert data.frame_rgb is not None
        assert data.landmarks == []
        assert data.timestamp is None

    def test_vision_data_with_landmarks(self):
        """Test VisionData with landmarks."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = [
            {
                "handedness": "Right",
                "landmarks": [{"x": 100, "y": 200, "z": 0.0}],
                "confidence": 0.95,
            }
        ]

        data = VisionData(frame=frame, frame_rgb=frame_rgb, landmarks=landmarks, timestamp=123.456)

        assert len(data.landmarks) == 1
        assert data.landmarks[0]["handedness"] == "Right"
        assert data.timestamp == 123.456


class TestMediaPipeVisionEngine:
    """Test MediaPipeVisionEngine class."""

    def test_initialization_with_defaults(self):
        """Test engine can be initialized with default parameters."""
        engine = MediaPipeVisionEngine()

        assert engine.camera_id == 0
        assert engine.fps == 30
        assert engine.max_num_hands == 2
        assert not engine.is_running()

    def test_initialization_with_custom_params(self):
        """Test engine initialization with custom parameters."""
        engine = MediaPipeVisionEngine(
            camera_id=1, fps=60, max_num_hands=1, enable_smoothing=True, smoothing_factor=0.7
        )

        assert engine.camera_id == 1
        assert engine.fps == 60
        assert engine.max_num_hands == 1
        assert engine.enable_smoothing is True
        assert engine.smoothing_factor == 0.7

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.hands.Hands")
    def test_initialize_success(self, mock_hands, mock_video_capture):
        """Test successful initialization."""
        # Mock camera
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        }.get(prop, 0.0)
        mock_video_capture.return_value = mock_capture

        # Mock MediaPipe
        mock_hands_instance = MagicMock()
        mock_hands.return_value = mock_hands_instance

        engine = MediaPipeVisionEngine()
        result = engine.initialize()

        assert result is True
        assert engine._capture is not None
        assert engine._hands is not None
        mock_video_capture.assert_called_once_with(0)
        mock_capture.set.assert_any_call(cv2.CAP_PROP_FPS, 30)

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.hands.Hands")
    def test_initialize_camera_failure(self, mock_hands, mock_video_capture):
        """Test initialization fails when camera cannot be opened."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = False
        mock_video_capture.return_value = mock_capture

        engine = MediaPipeVisionEngine()
        result = engine.initialize()

        assert result is False

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.hands.Hands")
    def test_start_stop_capture(self, mock_hands, mock_video_capture):
        """Test starting and stopping capture thread."""
        # Setup mocks
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.read.return_value = (False, None)  # Return False to exit loop quickly
        mock_capture.get.side_effect = lambda prop: 30.0 if prop == cv2.CAP_PROP_FPS else 640.0
        mock_video_capture.return_value = mock_capture

        mock_hands_instance = MagicMock()
        mock_hands.return_value = mock_hands_instance

        engine = MediaPipeVisionEngine()
        engine.initialize()

        # Start capture
        engine.start_capture()
        assert engine.is_running()

        # Give thread a moment to start
        time.sleep(0.1)

        # Stop capture
        engine.stop_capture()
        assert not engine.is_running()

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.hands.Hands")
    def test_get_frame_returns_none_when_no_frames(self, mock_hands, mock_video_capture):
        """Test get_frame returns None when no frames available."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture

        mock_hands_instance = MagicMock()
        mock_hands.return_value = mock_hands_instance

        engine = MediaPipeVisionEngine()
        engine.initialize()

        frame = engine.get_frame()
        assert frame is None

    def test_extract_landmarks_no_hands(self):
        """Test landmark extraction when no hands detected."""
        engine = MediaPipeVisionEngine()

        # Mock results with no hands
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = None

        frame_shape = (480, 640, 3)
        landmarks = engine._extract_landmarks(mock_results, frame_shape)

        assert landmarks == []

    def test_extract_landmarks_one_hand(self):
        """Test landmark extraction with one hand detected."""
        engine = MediaPipeVisionEngine()

        # Mock a single hand with 21 landmarks
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0

        mock_hand = MagicMock()
        mock_hand.landmark = [mock_landmark] * 21

        mock_handedness_class = MagicMock()
        mock_handedness_class.label = "Right"
        mock_handedness_class.score = 0.95

        mock_handedness = MagicMock()
        mock_handedness.classification = [mock_handedness_class]

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [mock_hand]
        mock_results.multi_handedness = [mock_handedness]

        frame_shape = (480, 640, 3)
        landmarks = engine._extract_landmarks(mock_results, frame_shape)

        assert len(landmarks) == 1
        assert landmarks[0]["handedness"] == "Right"
        assert landmarks[0]["confidence"] == 0.95
        assert len(landmarks[0]["landmarks"]) == 21
        assert len(landmarks[0]["landmarks_normalized"]) == 21

    def test_extract_landmarks_two_hands(self):
        """Test landmark extraction with two hands detected."""
        engine = MediaPipeVisionEngine()

        # Mock landmarks
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0

        # Mock two hands
        mock_hand1 = MagicMock()
        mock_hand1.landmark = [mock_landmark] * 21

        mock_hand2 = MagicMock()
        mock_hand2.landmark = [mock_landmark] * 21

        # Mock handedness info
        mock_handedness_class1 = MagicMock()
        mock_handedness_class1.label = "Right"
        mock_handedness_class1.score = 0.95

        mock_handedness_class2 = MagicMock()
        mock_handedness_class2.label = "Left"
        mock_handedness_class2.score = 0.92

        mock_handedness1 = MagicMock()
        mock_handedness1.classification = [mock_handedness_class1]

        mock_handedness2 = MagicMock()
        mock_handedness2.classification = [mock_handedness_class2]

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [mock_hand1, mock_hand2]
        mock_results.multi_handedness = [mock_handedness1, mock_handedness2]

        frame_shape = (480, 640, 3)
        landmarks = engine._extract_landmarks(mock_results, frame_shape)

        assert len(landmarks) == 2
        assert landmarks[0]["handedness"] == "Right"
        assert landmarks[1]["handedness"] == "Left"

    def test_smoothing_disabled_by_default(self):
        """Test that smoothing is disabled by default."""
        engine = MediaPipeVisionEngine(enable_smoothing=False)

        # Create mock landmarks
        landmarks = [
            {
                "handedness": "Right",
                "confidence": 0.9,
                "landmarks": [{"x": 100, "y": 200, "z": 0.0}] * 21,
                "landmarks_normalized": [{"x": 0.5, "y": 0.5, "z": 0.0}] * 21,
            }
        ]

        # First call should return landmarks as-is
        result = engine._apply_smoothing(landmarks)
        assert result == landmarks

    def test_smoothing_first_frame(self):
        """Test smoothing behavior on first frame."""
        engine = MediaPipeVisionEngine(enable_smoothing=True, smoothing_factor=0.5)

        landmarks = [
            {
                "handedness": "Right",
                "confidence": 0.9,
                "landmarks": [{"x": 100, "y": 200, "z": 0.0}] * 21,
                "landmarks_normalized": [{"x": 0.5, "y": 0.5, "z": 0.0}] * 21,
            }
        ]

        # First frame should pass through unchanged
        result = engine._apply_smoothing(landmarks)
        assert result == landmarks
        assert engine._prev_landmarks is not None

    def test_smoothing_subsequent_frames(self):
        """Test smoothing on subsequent frames."""
        engine = MediaPipeVisionEngine(enable_smoothing=True, smoothing_factor=0.5)

        # First frame
        landmarks1 = [
            {
                "handedness": "Right",
                "confidence": 0.9,
                "landmarks": [{"x": 100, "y": 200, "z": 0.0}] * 21,
                "landmarks_normalized": [{"x": 0.5, "y": 0.5, "z": 0.0}] * 21,
            }
        ]
        engine._apply_smoothing(landmarks1)

        # Second frame with different values
        landmarks2 = [
            {
                "handedness": "Right",
                "confidence": 0.9,
                "landmarks": [{"x": 200, "y": 300, "z": 0.1}] * 21,
                "landmarks_normalized": [{"x": 0.6, "y": 0.6, "z": 0.1}] * 21,
            }
        ]

        result = engine._apply_smoothing(landmarks2)

        # Check that smoothing was applied (values should be between original and new)
        assert result[0]["landmarks"][0]["x"] != 100  # Not equal to original
        assert result[0]["landmarks"][0]["x"] != 200  # Not equal to new
        # With alpha=0.5, result should be between the two values
        assert 100 < result[0]["landmarks"][0]["x"] < 200

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.hands.Hands")
    def test_cleanup(self, mock_hands, mock_video_capture):
        """Test cleanup releases all resources."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture

        mock_hands_instance = MagicMock()
        mock_hands.return_value = mock_hands_instance

        engine = MediaPipeVisionEngine()
        engine.initialize()
        engine.cleanup()

        mock_hands_instance.close.assert_called_once()
        mock_capture.release.assert_called_once()
        assert engine._hands is None
        assert engine._capture is None

    def test_get_vision_data_timeout(self):
        """Test get_vision_data with timeout."""
        engine = MediaPipeVisionEngine()

        # Queue is empty, should timeout
        data = engine.get_vision_data(timeout=0.1)
        assert data is None

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.hands.Hands")
    def test_thread_safety(self, mock_hands, mock_video_capture):
        """Test thread-safe access to latest frame."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture

        mock_hands_instance = MagicMock()
        mock_hands.return_value = mock_hands_instance

        engine = MediaPipeVisionEngine()
        engine.initialize()

        # Simulate setting a frame (would happen in capture thread)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with engine._lock:
            engine._latest_frame = test_frame

        # Get frame from main thread
        frame = engine.get_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)


class TestVisionEngineIntegration:
    """Integration tests for vision engine (these may be skipped if no camera available)."""

    @pytest.mark.skipif(not cv2.VideoCapture(0).isOpened(), reason="No camera available")
    def test_real_camera_initialization(self):
        """Test initialization with real camera (if available)."""
        engine = MediaPipeVisionEngine(camera_id=0)
        result = engine.initialize()

        if result:
            # If camera available, test basic operations
            assert engine._capture is not None
            assert engine._hands is not None
            engine.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
