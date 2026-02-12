"""
Unit tests for PyQt6 UI implementation.

Tests the PyQt6-based UI framework components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt

from src.ui.pyqt6_ui import (
    PyQt6UI,
    PyQt6MainWindow,
    VisionWorker,
    CameraWidget,
    StatusPanel,
    ControlsPanel,
)
from src.vision.vision_engine_impl import MediaPipeVisionEngine, VisionData


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mock_vision_engine():
    """Create a mock VisionEngine."""
    engine = Mock(spec=MediaPipeVisionEngine)
    engine.initialize.return_value = True
    engine.start_capture.return_value = None
    engine.stop_capture.return_value = None
    engine.cleanup.return_value = None
    engine.is_running.return_value = True
    engine.set_smoothing.return_value = None
    return engine


class TestCameraWidget:
    """Tests for CameraWidget."""

    def test_camera_widget_creation(self, qapp):
        """Test CameraWidget can be created."""
        widget = CameraWidget()
        assert widget is not None
        assert widget.minimumSize().width() == 640
        assert widget.minimumSize().height() == 480

    def test_display_frame(self, qapp):
        """Test displaying a frame."""
        widget = CameraWidget()

        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [100, 150, 200]  # BGR color

        # Should not raise an exception
        widget.display_frame(frame)

        # Verify pixmap was set
        assert widget.pixmap() is not None


class TestStatusPanel:
    """Tests for StatusPanel."""

    def test_status_panel_creation(self, qapp):
        """Test StatusPanel can be created."""
        panel = StatusPanel()
        assert panel is not None
        assert panel.title() == "System Status"

    def test_update_mode(self, qapp):
        """Test updating mode display."""
        panel = StatusPanel()
        panel.update_mode("AUDIO")
        assert panel.mode_value.text() == "AUDIO"

    def test_update_gesture(self, qapp):
        """Test updating gesture display."""
        panel = StatusPanel()
        panel.update_gesture("Thumbs Up", 0.95)
        assert panel.gesture_value.text() == "Thumbs Up"
        assert panel.confidence_value.text() == "0.95"

    def test_update_fps(self, qapp):
        """Test updating FPS display."""
        panel = StatusPanel()
        panel.update_fps(30.5)
        assert panel.fps_value.text() == "30.5"

    def test_update_latency(self, qapp):
        """Test updating latency display."""
        panel = StatusPanel()
        panel.update_latency(25.7)
        assert panel.latency_value.text() == "26 ms"

    def test_update_hands(self, qapp):
        """Test updating hands count."""
        panel = StatusPanel()
        panel.update_hands(2)
        assert panel.hands_value.text() == "2"


class TestControlsPanel:
    """Tests for ControlsPanel."""

    def test_controls_panel_creation(self, qapp):
        """Test ControlsPanel can be created."""
        panel = ControlsPanel()
        assert panel is not None
        assert panel.title() == "Controls"

    def test_debug_toggle(self, qapp):
        """Test debug mode toggle."""
        panel = ControlsPanel()

        # Connect signal to a mock
        mock_handler = Mock()
        panel.debug_toggled.connect(mock_handler)

        # Click the button
        panel.debug_button.click()

        # Verify signal was emitted
        mock_handler.assert_called_once_with(True)

    def test_smoothing_toggle(self, qapp):
        """Test smoothing toggle."""
        panel = ControlsPanel()

        # Connect signal to a mock
        mock_handler = Mock()
        panel.smoothing_toggled.connect(mock_handler)

        # Button starts checked, so first click unchecks
        panel.smoothing_button.click()

        # Verify signal was emitted
        mock_handler.assert_called_once_with(False)

    def test_reset_button(self, qapp):
        """Test reset button."""
        panel = ControlsPanel()

        # Connect signal to a mock
        mock_handler = Mock()
        panel.reset_requested.connect(mock_handler)

        # Click the button
        panel.reset_button.click()

        # Verify signal was emitted
        mock_handler.assert_called_once()


class TestPyQt6MainWindow:
    """Tests for PyQt6MainWindow."""

    def test_main_window_creation(self, qapp, mock_vision_engine):
        """Test main window can be created."""
        window = PyQt6MainWindow(mock_vision_engine)
        assert window is not None
        assert window.windowTitle() == "Gesture Media Interface - PyQt6"

    def test_window_components(self, qapp, mock_vision_engine):
        """Test main window has all required components."""
        window = PyQt6MainWindow(mock_vision_engine)

        # Check components exist
        assert window.camera_widget is not None
        assert window.status_panel is not None
        assert window.controls_panel is not None

    def test_debug_toggle_handler(self, qapp, mock_vision_engine):
        """Test debug mode toggle handler."""
        window = PyQt6MainWindow(mock_vision_engine)

        # Initially false
        assert window.debug_mode is False

        # Toggle debug mode
        window._on_debug_toggled(True)
        assert window.debug_mode is True

        window._on_debug_toggled(False)
        assert window.debug_mode is False

    def test_smoothing_toggle_handler(self, qapp, mock_vision_engine):
        """Test smoothing toggle handler."""
        window = PyQt6MainWindow(mock_vision_engine)

        # Toggle smoothing
        window._on_smoothing_toggled(True)
        mock_vision_engine.set_smoothing.assert_called_with(True)

        window._on_smoothing_toggled(False)
        mock_vision_engine.set_smoothing.assert_called_with(False)

    def test_draw_landmarks_no_hands(self, qapp, mock_vision_engine):
        """Test drawing landmarks with no hands detected."""
        window = PyQt6MainWindow(mock_vision_engine)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = window._draw_landmarks(frame, [])

        # Should return unchanged frame
        assert np.array_equal(result, frame)

    def test_draw_landmarks_with_hand(self, qapp, mock_vision_engine):
        """Test drawing landmarks with detected hand."""
        window = PyQt6MainWindow(mock_vision_engine)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create mock hand landmarks with coordinates within frame
        landmarks = [
            {
                "handedness": "Right",
                "confidence": 0.95,
                "landmarks": [
                    {"x": 100 + i * 5, "y": 100 + i * 5, "z": 0.0} for i in range(21)
                ],
            }
        ]

        # Should not raise an exception
        result = window._draw_landmarks(frame, landmarks)

        # Verify result is a numpy array with correct shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)


class TestVisionWorker:
    """Tests for VisionWorker thread."""

    def test_worker_creation(self, qapp, mock_vision_engine):
        """Test VisionWorker can be created."""
        worker = VisionWorker(mock_vision_engine)
        assert worker is not None
        assert worker.vision_engine == mock_vision_engine

    def test_worker_stop(self, qapp, mock_vision_engine):
        """Test stopping the worker."""
        worker = VisionWorker(mock_vision_engine)

        # Start and stop
        worker._running = True
        worker.stop()

        assert worker._running is False


class TestPyQt6UI:
    """Tests for PyQt6UI class."""

    @patch("src.ui.pyqt6_ui.MediaPipeVisionEngine")
    def test_ui_initialization(self, mock_engine_class, qapp):
        """Test PyQt6UI initialization."""
        # Setup mock
        mock_engine = Mock(spec=MediaPipeVisionEngine)
        mock_engine.initialize.return_value = True
        mock_engine_class.return_value = mock_engine

        # Create UI
        ui = PyQt6UI()

        # Initialize
        result = ui.initialize()

        assert result is True
        assert ui._initialized is True
        assert ui.main_window is not None

    def test_ui_initialization_with_engine(self, qapp, mock_vision_engine):
        """Test PyQt6UI initialization with provided engine."""
        ui = PyQt6UI(vision_engine=mock_vision_engine)

        # Initialize
        result = ui.initialize()

        assert result is True
        assert ui._initialized is True
        assert ui.vision_engine == mock_vision_engine

    def test_display_mode(self, qapp, mock_vision_engine):
        """Test display_mode method."""
        ui = PyQt6UI(vision_engine=mock_vision_engine)
        ui.initialize()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = ui.display_mode(frame, "AUDIO")

        # Should update status panel
        assert ui.main_window.status_panel.mode_value.text() == "AUDIO"

    def test_display_gesture(self, qapp, mock_vision_engine):
        """Test display_gesture method."""
        ui = PyQt6UI(vision_engine=mock_vision_engine)
        ui.initialize()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = ui.display_gesture(frame, "Wave")

        # Should update status panel
        assert ui.main_window.status_panel.gesture_value.text() == "Wave"

    def test_cleanup(self, qapp, mock_vision_engine):
        """Test cleanup method."""
        ui = PyQt6UI(vision_engine=mock_vision_engine)
        ui.initialize()

        # Cleanup
        ui.cleanup()

        # Verify engine cleanup was called
        mock_vision_engine.cleanup.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
