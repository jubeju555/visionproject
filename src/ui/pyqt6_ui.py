"""
PyQt6-based UI implementation for the gesture-media-interface.

This module provides a modern, thread-safe UI using PyQt6 with:
- Left Panel: Live camera feed with hand landmark overlay
- Right Panel: Current mode, gesture, metrics, and status
- Interactive Controls: Debug mode, smoothing, system reset
- Thread-safe signal/slot communication
- Non-blocking capture thread integration
"""

import logging
import os
import time
from typing import Optional, Dict, Any, List
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QFrame,
    QDialog,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

from src.core.app_ui import AppUI
from src.vision.vision_engine_impl import MediaPipeVisionEngine, VisionData
from src.gesture.gesture_recognition_engine import (
    GestureRecognitionEngine,
    GestureEvent,
)
from src.gesture.rectangle_gestures import (
    RectangleGestureDetector,
    RectangleFrame,
    ScreenshotCapture,
)
from src.core.state_manager import StateManager
from src.core.mode_router import ApplicationMode
import queue

logger = logging.getLogger(__name__)


class VisionWorker(QThread):
    """
    Worker thread for VisionEngine to prevent blocking the UI.

    This thread runs the VisionEngine capture loop and emits signals
    when new frames are available, ensuring thread-safe communication.
    """

    # Signal emitted when new vision data is available
    vision_data_ready = pyqtSignal(object)  # VisionData
    error_occurred = pyqtSignal(str)

    def __init__(self, vision_engine: MediaPipeVisionEngine):
        """
        Initialize vision worker.

        Args:
            vision_engine: VisionEngine instance to run
        """
        super().__init__()
        self.vision_engine = vision_engine
        self._running = False

    def run(self):
        """Main worker thread loop."""
        logger.info("Vision worker thread started")
        self._running = True

        try:
            while self._running:
                # Get vision data with timeout
                vision_data = self.vision_engine.get_vision_data(timeout=0.1)

                if vision_data:
                    # Emit signal with vision data
                    self.vision_data_ready.emit(vision_data)

        except Exception as e:
            logger.error(f"Error in vision worker: {e}", exc_info=True)
            self.error_occurred.emit(str(e))
        finally:
            logger.info("Vision worker thread stopped")

    def stop(self):
        """Stop the worker thread."""
        self._running = False


class GestureWorker(QThread):
    """
    Worker thread for GestureRecognitionEngine.

    This thread runs the GestureRecognitionEngine and emits signals
    when new gestures are recognized.
    """

    # Signal emitted when new gesture is recognized
    gesture_recognized = pyqtSignal(object)  # GestureEvent
    error_occurred = pyqtSignal(str)

    def __init__(self, gesture_engine: GestureRecognitionEngine):
        """
        Initialize gesture worker.

        Args:
            gesture_engine: GestureRecognitionEngine instance
        """
        super().__init__()
        self.gesture_engine = gesture_engine
        self._running = False

    def run(self):
        """Main worker thread loop."""
        logger.info("Gesture worker thread started")
        self._running = True

        try:
            while self._running:
                # Get gesture events with timeout
                gesture_event = self.gesture_engine.get_event(timeout=0.1)

                if gesture_event:
                    # Emit signal with gesture event
                    self.gesture_recognized.emit(gesture_event)

        except Exception as e:
            logger.error(f"Error in gesture worker: {e}", exc_info=True)
            self.error_occurred.emit(str(e))
        finally:
            logger.info("Gesture worker thread stopped")

    def stop(self):
        """Stop the worker thread."""
        self._running = False



class CameraWidget(QLabel):
    """
    Widget for displaying camera feed with landmark overlays.

    Displays the live camera feed with hand landmarks drawn on top.
    """

    def __init__(self, parent=None):
        """Initialize camera widget."""
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: black; }")
        self.setMinimumSize(640, 480)
        self.setScaledContents(True)

    def display_frame(self, frame: np.ndarray):
        """
        Display a frame in the widget.

        Args:
            frame: BGR frame from OpenCV
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get frame dimensions
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width

        # Create QImage
        q_image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # Create pixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.setPixmap(pixmap)


class StatusPanel(QGroupBox):
    """
    Widget for displaying system status and metrics.

    Shows current mode, detected gesture, confidence, parameters,
    FPS, and latency information.
    """

    def __init__(self, parent=None):
        """Initialize status panel."""
        super().__init__("System Status", parent)

        # Create layout
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Font for labels
        label_font = QFont()
        label_font.setPointSize(10)

        value_font = QFont()
        value_font.setPointSize(10)
        value_font.setBold(True)

        # Current Mode
        mode_label = QLabel("Current Mode:")
        mode_label.setFont(label_font)
        self.mode_value = QLabel("IDLE")
        self.mode_value.setFont(value_font)
        self.mode_value.setStyleSheet("QLabel { color: #4CAF50; }")
        layout.addWidget(mode_label)
        layout.addWidget(self.mode_value)

        # Separator
        layout.addWidget(self._create_separator())

        # Detected Gesture
        gesture_label = QLabel("Detected Gesture:")
        gesture_label.setFont(label_font)
        self.gesture_value = QLabel("None")
        self.gesture_value.setFont(value_font)
        self.gesture_value.setStyleSheet("QLabel { color: #2196F3; }")
        layout.addWidget(gesture_label)
        layout.addWidget(self.gesture_value)

        # Confidence Score
        confidence_label = QLabel("Confidence:")
        confidence_label.setFont(label_font)
        self.confidence_value = QLabel("0.00")
        self.confidence_value.setFont(value_font)
        layout.addWidget(confidence_label)
        layout.addWidget(self.confidence_value)

        # Separator
        layout.addWidget(self._create_separator())

        # Audio Parameters (stubbed)
        audio_label = QLabel("Audio Parameters:")
        audio_label.setFont(label_font)
        self.audio_volume = QLabel("Volume: --")
        self.audio_tempo = QLabel("Tempo: --")
        self.audio_pitch = QLabel("Pitch: --")
        layout.addWidget(audio_label)
        layout.addWidget(self.audio_volume)
        layout.addWidget(self.audio_tempo)
        layout.addWidget(self.audio_pitch)

        # Separator
        layout.addWidget(self._create_separator())

        # Image Editing Status (stubbed)
        image_label = QLabel("Image Editing Status:")
        image_label.setFont(label_font)
        self.image_status = QLabel("Not Active")
        layout.addWidget(image_label)
        layout.addWidget(self.image_status)

        # Separator
        layout.addWidget(self._create_separator())

        # FPS Display
        fps_label = QLabel("FPS:")
        fps_label.setFont(label_font)
        self.fps_value = QLabel("0.0")
        self.fps_value.setFont(value_font)
        self.fps_value.setStyleSheet("QLabel { color: #FF9800; }")
        layout.addWidget(fps_label)
        layout.addWidget(self.fps_value)

        # Latency Estimate
        latency_label = QLabel("Latency:")
        latency_label.setFont(label_font)
        self.latency_value = QLabel("0 ms")
        self.latency_value.setFont(value_font)
        layout.addWidget(latency_label)
        layout.addWidget(self.latency_value)

        # Hands Detected
        hands_label = QLabel("Hands Detected:")
        hands_label.setFont(label_font)
        self.hands_value = QLabel("0")
        self.hands_value.setFont(value_font)
        layout.addWidget(hands_label)
        layout.addWidget(self.hands_value)
        
        # Separator
        layout.addWidget(self._create_separator())
        
        # Performance Metrics Section
        perf_label = QLabel("Performance Metrics:")
        perf_label.setFont(label_font)
        layout.addWidget(perf_label)
        
        # End-to-End Latency
        self.e2e_latency = QLabel("E2E Latency: -- ms")
        self.e2e_latency.setFont(QFont("Monospace", 9))
        layout.addWidget(self.e2e_latency)
        
        # Dropped Frames
        self.dropped_frames = QLabel("Dropped: 0")
        self.dropped_frames.setFont(QFont("Monospace", 9))
        layout.addWidget(self.dropped_frames)
        
        # Queue Status
        self.queue_status = QLabel("Queue: --")
        self.queue_status.setFont(QFont("Monospace", 9))
        layout.addWidget(self.queue_status)

        # Add stretch to push everything to top
        layout.addStretch()

        self.setLayout(layout)

    def _create_separator(self):
        """Create a horizontal separator line."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        return separator

    def update_mode(self, mode: str):
        """Update current mode display."""
        self.mode_value.setText(mode)

    def update_gesture(self, gesture: str, confidence: float = 0.0):
        """Update detected gesture and confidence."""
        self.gesture_value.setText(gesture)
        self.confidence_value.setText(f"{confidence:.2f}")

    def update_fps(self, fps: float):
        """Update FPS display."""
        self.fps_value.setText(f"{fps:.1f}")

    def update_latency(self, latency_ms: float):
        """Update latency display."""
        self.latency_value.setText(f"{latency_ms:.0f} ms")

    def update_hands(self, count: int):
        """Update hands detected count."""
        self.hands_value.setText(str(count))

    def update_image_status(self, status: str):
        """Update image editing status display."""
        self.image_status.setText(status)
    
    def update_performance_metrics(self, e2e_latency: float, dropped_frames: int, queue_status: str):
        """Update performance metrics display."""
        self.e2e_latency.setText(f"E2E Latency: {e2e_latency:.1f} ms")
        self.dropped_frames.setText(f"Dropped: {dropped_frames}")
        self.queue_status.setText(f"Queue: {queue_status}")


class ControlsPanel(QGroupBox):
    """
    Widget for interactive controls.

    Provides buttons for toggling debug mode, smoothing,
    and resetting system state.
    """

    # Signals for control actions
    debug_toggled = pyqtSignal(bool)
    smoothing_toggled = pyqtSignal(bool)
    reset_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize controls panel."""
        super().__init__("Controls", parent)

        # Create layout
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Debug Mode Toggle
        self.debug_button = QPushButton("Toggle Debug Mode")
        self.debug_button.setCheckable(True)
        self.debug_button.setChecked(False)
        self.debug_button.clicked.connect(self._on_debug_clicked)
        layout.addWidget(self.debug_button)

        # Smoothing Toggle
        self.smoothing_button = QPushButton("Toggle Smoothing")
        self.smoothing_button.setCheckable(True)
        self.smoothing_button.setChecked(True)  # Default enabled
        self.smoothing_button.clicked.connect(self._on_smoothing_clicked)
        layout.addWidget(self.smoothing_button)

        # Reset Button
        self.reset_button = QPushButton("Reset System State")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        layout.addWidget(self.reset_button)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("QLabel { color: #4CAF50; }")
        layout.addWidget(self.status_label)

        # Add stretch
        layout.addStretch()

        self.setLayout(layout)

    def _on_debug_clicked(self, checked: bool):
        """Handle debug button click."""
        self.debug_toggled.emit(checked)
        self.status_label.setText(f"Debug Mode: {'ON' if checked else 'OFF'}")

    def _on_smoothing_clicked(self, checked: bool):
        """Handle smoothing button click."""
        self.smoothing_toggled.emit(checked)
        self.status_label.setText(f"Smoothing: {'ON' if checked else 'OFF'}")

    def _on_reset_clicked(self):
        """Handle reset button click."""
        self.reset_requested.emit()
        self.status_label.setText("System Reset")


class GestureGuideDialog(QDialog):
    """Dialog that lists available gestures and their mappings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gesture Guide")
        self.setMinimumSize(520, 560)

        layout = QVBoxLayout()

        title = QLabel("Gesture Guide")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #2a231c;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Recognized gestures and recommended mappings for each mode."
        )
        subtitle.setStyleSheet("color: #6a5d4f;")
        layout.addWidget(subtitle)

        content = QLabel()
        content.setTextFormat(Qt.TextFormat.RichText)
        content.setWordWrap(True)
        content.setText(
            "<b>System</b><br/>"
            "- Both palms open for 2 seconds: switch mode<br/><br/>"
            "- Rectangle frame visible: ready to capture<br/>"
            "- Snap (pinch) with either hand: capture screenshot<br/>"
            "- Double pinch (both hands within 0.35s): capture screenshot<br/><br/>"
            "<b>Audio Control Mode</b><br/>"
            "- Fist: play/pause<br/>"
            "- Swipe right: next track<br/>"
            "- Swipe left: previous track<br/>"
            "- Pinch distance: volume (continuous)<br/>"
            "- Open palm vertical position: tempo (stubbed)<br/>"
            "- Open palm horizontal position: pitch (stubbed)<br/><br/>"
            "<b>Image Editing Mode</b><br/>"
            "- Pinch + drag: translate<br/>"
            "- Two-hand spread: scale<br/>"
            "- Circular motion: rotate<br/>"
            "- Swipe left: undo<br/>"
            "- Open palm vertical position: brightness<br/><br/>"
            "<b>Recognized Gestures</b><br/>"
            "- open_palm, fist, pinch, two_fingers, three_fingers<br/>"
            "- swipe_left, swipe_right, vertical_motion, circular_motion, two_hand_spread"
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.addWidget(content)
        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self.setLayout(layout)

    def _on_debug_clicked(self, checked: bool):
        """Handle debug button click."""
        self.debug_toggled.emit(checked)
        self.status_label.setText(f"Debug Mode: {'ON' if checked else 'OFF'}")

    def _on_smoothing_clicked(self, checked: bool):
        """Handle smoothing button click."""
        self.smoothing_toggled.emit(checked)
        self.status_label.setText(f"Smoothing: {'ON' if checked else 'OFF'}")

    def _on_reset_clicked(self):
        """Handle reset button click."""
        self.reset_requested.emit()
        self.status_label.setText("System Reset")


class PyQt6MainWindow(QMainWindow):
    """
    Main window for PyQt6 UI.

    Contains left panel (camera feed), right panel (status),
    and controls for interaction.
    """

    def __init__(self, vision_engine: MediaPipeVisionEngine, performance_monitor=None):
        """
        Initialize main window.

        Args:
            vision_engine: VisionEngine instance
            performance_monitor: Optional PerformanceMonitor instance
        """
        super().__init__()
        self.vision_engine = vision_engine
        self.vision_worker: Optional[VisionWorker] = None
        self.performance_monitor = performance_monitor
        
        # Create gesture recognition engine
        self.vision_to_gesture_queue = queue.Queue(maxsize=10)
        self.gesture_engine = GestureRecognitionEngine(
            input_queue=self.vision_to_gesture_queue,
            max_queue_size=10,
            history_size=10,
            performance_monitor=performance_monitor,
        )
        self.gesture_worker: Optional[GestureWorker] = None
        
        # Create mode router
        self.mode_router = StateManager(mode_switch_duration=2.0)
        self.mode_router.initialize()
        
        # Register mode change callback
        self.mode_router.register_mode_change_callback(self._on_mode_changed)

        # FPS tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.last_frame_time = time.time()

        # Debug mode
        self.debug_mode = False
        
        # Latest gesture
        self.latest_gesture: Optional[str] = None
        self.latest_gesture_confidence: float = 0.0

        # Rectangle capture state
        self.rectangle_detector = RectangleGestureDetector()
        self.screenshot_capture = ScreenshotCapture(output_dir="./screenshots")
        self.latest_rectangle: Optional[RectangleFrame] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.last_capture_time = 0.0
        self.capture_cooldown = 1.0
        self.last_pinch_time = {"left": 0.0, "right": 0.0}
        self.double_pinch_window = 0.35
        self.pending_snap = False
        self.pending_snap_hand: Optional[str] = None
        self.pending_snap_time = 0.0
        
        # Performance metrics update timer
        if self.performance_monitor:
            self.perf_timer = QTimer()
            self.perf_timer.timeout.connect(self._update_performance_metrics)
            self.perf_timer.start(1000)  # Update every second

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI components."""
        self.setWindowTitle("Gesture Media Interface - PyQt6")
        self.setGeometry(100, 100, 1200, 700)

        self._setup_menu()

        self.setStyleSheet(
            "QMainWindow { background-color: #0f172a; }"
            "QWidget { color: #e2e8f0; font-family: 'Poppins', 'Segoe UI', 'Trebuchet MS'; }"
            "QGroupBox {"
            "  border: 1px solid #1f2a44;"
            "  border-radius: 14px;"
            "  margin-top: 14px;"
            "  background-color: #111827;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  subcontrol-position: top left;"
            "  padding: 2px 10px;"
            "  color: #94a3b8;"
            "  font-weight: 600;"
            "}"
            "QPushButton {"
            "  background-color: #2563eb;"
            "  color: #ffffff;"
            "  border-radius: 10px;"
            "  padding: 9px 14px;"
            "  font-weight: 600;"
            "}"
            "QPushButton:checked { background-color: #1d4ed8; }"
            "QPushButton:hover { background-color: #3b82f6; }"
            "QLabel#AppTitle { font-size: 20px; font-weight: 700; color: #f8fafc; }"
            "QLabel#AppSubtitle { font-size: 11px; color: #94a3b8; }"
        )

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout (horizontal)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left Panel - Camera Feed
        self.camera_widget = CameraWidget()
        self.camera_widget.setStyleSheet(
            "QLabel { background-color: #0b1020; border: 2px solid #1f2937;"
            " border-radius: 14px; }"
        )
        main_layout.addWidget(self.camera_widget, stretch=2)

        # Right Panel - Status and Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # Header
        header_widget = QWidget()
        header_layout = QVBoxLayout()
        header_layout.setSpacing(2)
        header_layout.setContentsMargins(2, 2, 2, 2)
        title = QLabel("Gesture Media Interface")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Live hand-tracking control surface")
        subtitle.setObjectName("AppSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_widget.setLayout(header_layout)
        right_layout.addWidget(header_widget)

        # Status Panel
        self.status_panel = StatusPanel()
        # Initialize with current mode
        self.status_panel.update_mode(self.mode_router.get_mode().value.upper())
        right_layout.addWidget(self.status_panel)

        # Controls Panel
        self.controls_panel = ControlsPanel()
        right_layout.addWidget(self.controls_panel)

        # Connect control signals
        self.controls_panel.debug_toggled.connect(self._on_debug_toggled)
        self.controls_panel.smoothing_toggled.connect(self._on_smoothing_toggled)
        self.controls_panel.reset_requested.connect(self._on_reset_requested)

        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel, stretch=1)

        central_widget.setLayout(main_layout)

    def _setup_menu(self):
        """Setup the menu bar and actions."""
        help_menu = self.menuBar().addMenu("Help")
        gesture_action = help_menu.addAction("Gesture Guide")
        gesture_action.triggered.connect(self._show_gesture_guide)

    def _show_gesture_guide(self):
        """Show the gesture guide dialog."""
        dialog = GestureGuideDialog(self)
        dialog.exec()

    def start_vision_engine(self):
        """Start the vision engine and worker thread."""
        logger.info("Starting vision engine in UI")

        # Start vision engine capture
        self.vision_engine.start_capture()

        # Create and start vision worker thread
        self.vision_worker = VisionWorker(self.vision_engine)
        self.vision_worker.vision_data_ready.connect(self._on_vision_data)
        self.vision_worker.error_occurred.connect(self._on_error)
        self.vision_worker.start()

        logger.info("Vision worker thread started")
        
        # Start gesture recognition engine
        logger.info("Starting gesture recognition engine")
        self.gesture_engine.start()
        
        # Create and start gesture worker thread
        self.gesture_worker = GestureWorker(self.gesture_engine)
        self.gesture_worker.gesture_recognized.connect(self._on_gesture_recognized)
        self.gesture_worker.error_occurred.connect(self._on_error)
        self.gesture_worker.start()
        
        logger.info("Gesture worker thread started")

    def stop_vision_engine(self):
        """Stop the vision engine and worker thread."""
        logger.info("Stopping vision engine in UI")
        
        # Stop gesture worker thread
        if self.gesture_worker:
            self.gesture_worker.stop()
            self.gesture_worker.wait(2000)  # Wait up to 2 seconds
        
        # Stop gesture recognition engine
        if self.gesture_engine:
            self.gesture_engine.stop()

        # Stop vision worker thread
        if self.vision_worker:
            self.vision_worker.stop()
            self.vision_worker.wait(2000)  # Wait up to 2 seconds

        # Stop vision engine
        self.vision_engine.stop_capture()

        logger.info("Vision engine and gesture engine stopped")

    @pyqtSlot(object)
    def _on_vision_data(self, vision_data: VisionData):
        """
        Handle new vision data from worker thread.

        Args:
            vision_data: VisionData from VisionEngine
        """
        # Feed vision data to gesture engine
        try:
            self.vision_to_gesture_queue.put_nowait(vision_data)
        except queue.Full:
            # Drop oldest data if queue is full
            try:
                self.vision_to_gesture_queue.get_nowait()
                self.vision_to_gesture_queue.put_nowait(vision_data)
            except (queue.Empty, queue.Full):
                pass
        
        # Keep latest raw frame for capture
        self.latest_frame = vision_data.frame.copy()

        # Draw landmarks on frame
        frame = vision_data.frame.copy()
        frame = self._draw_landmarks(frame, vision_data.landmarks)

        # Detect and draw rectangle overlay
        timestamp = vision_data.timestamp or time.time()
        self.latest_rectangle = self.rectangle_detector.detect_rectangle(
            vision_data.landmarks,
            timestamp,
        )
        if self.latest_rectangle:
            frame = self._draw_rectangle_overlay(frame, self.latest_rectangle)

        # Display frame
        self.camera_widget.display_frame(frame)

        # Update status panel
        hand_count = len(vision_data.landmarks)
        self.status_panel.update_hands(hand_count)

        # Update gesture info with latest recognized gesture
        if self.latest_gesture:
            self.status_panel.update_gesture(
                self.latest_gesture, 
                self.latest_gesture_confidence
            )
        elif hand_count > 0:
            self.status_panel.update_gesture("Processing...", 0.0)
        else:
            self.status_panel.update_gesture("None", 0.0)

        # Calculate and update FPS
        self._update_fps()

        # Calculate and update latency
        current_time = time.time()
        latency_ms = (current_time - self.last_frame_time) * 1000
        self.status_panel.update_latency(latency_ms)
        self.last_frame_time = current_time

    def _draw_landmarks(
        self, frame: np.ndarray, landmarks: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw hand landmarks on frame.

        Args:
            frame: BGR frame
            landmarks: List of hand landmark data

        Returns:
            Frame with landmarks drawn
        """
        if not landmarks:
            return frame

        # Draw for each detected hand
        for hand_data in landmarks:
            handedness = hand_data["handedness"]
            hand_landmarks = hand_data["landmarks"]
            confidence = hand_data["confidence"]

            # Draw landmarks as circles
            for i, lm in enumerate(hand_landmarks):
                x, y = lm["x"], lm["y"]

                # Draw landmark point
                color = (0, 255, 0) if not self.debug_mode else (255, 0, 0)
                cv2.circle(frame, (x, y), 5, color, -1)

                # Draw landmark index in debug mode
                if self.debug_mode and i in [0, 4, 8, 12, 16, 20]:
                    cv2.putText(
                        frame,
                        str(i),
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1,
                    )

            # Draw handedness label
            if hand_landmarks:
                wrist = hand_landmarks[0]
                label = f"{handedness} ({confidence:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (wrist["x"], wrist["y"] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            # Draw connections between landmarks
            connections = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),  # Thumb
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),  # Index
                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),  # Middle
                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),  # Ring
                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),  # Pinky
            ]

            for start_idx, end_idx in connections:
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    color = (0, 255, 0) if not self.debug_mode else (255, 0, 0)
                    cv2.line(
                        frame, (start["x"], start["y"]), (end["x"], end["y"]), color, 2
                    )

        return frame

    def _draw_rectangle_overlay(
        self, frame: np.ndarray, rectangle: RectangleFrame
    ) -> np.ndarray:
        """Draw rectangle overlay on the frame."""
        if rectangle is None:
            return frame

        height, width = frame.shape[:2]
        corners = rectangle.get_pixel_coordinates(width, height)
        points = np.array(
            [
                corners["top_left"],
                corners["top_right"],
                corners["bottom_right"],
                corners["bottom_left"],
            ],
            dtype=np.int32,
        )

        color = (0, 200, 255) if not self.debug_mode else (255, 200, 0)
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

        label = f"Rectangle {rectangle.confidence:.2f}"
        label_pos = corners["top_left"]
        cv2.putText(
            frame,
            label,
            (label_pos[0], max(15, label_pos[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        return frame

    def _update_fps(self):
        """Update FPS calculation and display."""
        self.frame_count += 1

        # Update FPS every 30 frames
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = 30 / elapsed
            self.fps_start_time = time.time()
            self.status_panel.update_fps(self.current_fps)
    
    def _update_performance_metrics(self):
        """Update performance metrics display from performance monitor."""
        if not self.performance_monitor:
            return
        
        try:
            # Get metrics summary
            summary = self.performance_monitor.get_metrics_summary()
            
            # Update E2E latency
            e2e_latency = summary['e2e_latency']['avg_ms']
            
            # Calculate total dropped frames
            total_dropped = sum(
                stage['dropped_frames'] 
                for stage in summary['stages'].values()
            )
            
            # Format queue status with better name mapping
            queue_name_map = {
                'vision_output_queue': 'V:Out',
                'vision_input_queue': 'V:In',
                'gesture_input_queue': 'G:In',
                'gesture_output_queue': 'G:Out',
            }
            
            queue_info = []
            for name, metrics in summary['queues'].items():
                short_name = queue_name_map.get(name, name[:6])  # Use mapping or truncate
                queue_info.append(f"{short_name}:{metrics['size']}/{metrics['capacity']}")
            queue_status = ", ".join(queue_info) if queue_info else "N/A"
            
            # Update status panel
            self.status_panel.update_performance_metrics(e2e_latency, total_dropped, queue_status)
            
        except Exception as e:
            logger.debug(f"Error updating performance metrics: {e}")

    @pyqtSlot(object)
    def _on_gesture_recognized(self, gesture_event: GestureEvent):
        """
        Handle gesture recognition from worker thread.
        
        Args:
            gesture_event: GestureEvent from GestureRecognitionEngine
        """
        # Update latest gesture
        self.latest_gesture = gesture_event.gesture_name
        self.latest_gesture_confidence = gesture_event.confidence_score
        
        # Log gesture recognition
        logger.info(
            f"Gesture recognized: {gesture_event.gesture_name} "
            f"({gesture_event.confidence_score:.2f}) "
            f"from {gesture_event.hand_id} hand"
        )
        
        # Route gesture through mode router
        gesture_data = {
            'hand_id': gesture_event.hand_id,
            'confidence': gesture_event.confidence_score,
            'timestamp': gesture_event.timestamp,
            'gesture_type': gesture_event.gesture_type,
        }
        
        # Add additional data if present
        if gesture_event.additional_data:
            gesture_data.update(gesture_event.additional_data)
        
        # Route gesture to mode router (non-blocking)
        self.mode_router.route_gesture(gesture_event.gesture_name, gesture_data)

        # Capture screenshot when pinch is detected inside a rectangle frame
        if gesture_event.gesture_name == "pinch":
            self._handle_pinch_capture(gesture_event)

    def _handle_pinch_capture(self, gesture_event: GestureEvent) -> None:
        """Handle snap or double-pinch capture logic."""
        if not self.latest_rectangle or self.latest_frame is None:
            return

        hand_id = gesture_event.hand_id
        if hand_id not in ("left", "right"):
            return

        now = time.time()
        other_hand = "right" if hand_id == "left" else "left"
        other_pinch_time = self.last_pinch_time.get(other_hand, 0.0)
        self.last_pinch_time[hand_id] = now

        if now - other_pinch_time <= self.double_pinch_window:
            self.pending_snap = False
            self.pending_snap_hand = None
            self._trigger_capture(reason="double pinch")
            return

        self.pending_snap = True
        self.pending_snap_hand = hand_id
        self.pending_snap_time = now

        QTimer.singleShot(int(self.double_pinch_window * 1000), self._finalize_snap_capture)

    def _finalize_snap_capture(self) -> None:
        """Finalize single-hand snap capture after double-pinch window expires."""
        if not self.pending_snap:
            return

        now = time.time()
        if now - self.pending_snap_time >= self.double_pinch_window:
            self._trigger_capture(reason="snap")
        self.pending_snap = False
        self.pending_snap_hand = None

    def _trigger_capture(self, reason: str) -> None:
        """Capture and save a screenshot if cooldown allows."""
        now = time.time()
        if (now - self.last_capture_time) < self.capture_cooldown:
            return

        saved_path = self.screenshot_capture.capture_and_save(
            self.latest_frame,
            self.latest_rectangle,
            name_prefix="rectangle",
        )
        if saved_path:
            self.last_capture_time = now
            filename = os.path.basename(saved_path)
            self.status_panel.update_image_status(f"Captured ({reason}): {filename}")

    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        """Handle error from worker thread."""
        logger.error(f"Error from worker: {error_msg}")

    @pyqtSlot(bool)
    def _on_debug_toggled(self, enabled: bool):
        """Handle debug mode toggle."""
        self.debug_mode = enabled
        logger.info(f"Debug mode: {'enabled' if enabled else 'disabled'}")

    @pyqtSlot(bool)
    def _on_smoothing_toggled(self, enabled: bool):
        """Handle smoothing toggle."""
        self.vision_engine.set_smoothing(enabled)
        logger.info(f"Smoothing: {'enabled' if enabled else 'disabled'}")

    @pyqtSlot()
    def _on_reset_requested(self):
        """Handle reset request."""
        logger.info("System reset requested")
        # Reset to NEUTRAL mode
        self.mode_router.set_mode(ApplicationMode.NEUTRAL)
        self.status_panel.update_mode(ApplicationMode.NEUTRAL.value.upper())
    
    def _on_mode_changed(self, new_mode: ApplicationMode):
        """
        Handle mode change from mode router.
        
        This callback is called from the mode router when mode changes.
        
        Args:
            new_mode: The new application mode
        """
        logger.info(f"Mode changed to: {new_mode.value}")
        # Update UI to show new mode (use QTimer to ensure UI thread safety)
        QTimer.singleShot(0, lambda: self.status_panel.update_mode(new_mode.value.upper()))

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Window closing, cleaning up...")
        self.stop_vision_engine()
        
        # Clean up mode router
        if self.mode_router:
            self.mode_router.cleanup()
        
        event.accept()


class PyQt6UI(AppUI):
    """
    PyQt6 implementation of AppUI interface.

    Provides a modern UI with thread-safe signal/slot communication
    and non-blocking integration with VisionEngine.
    """

    def __init__(self, vision_engine: Optional[MediaPipeVisionEngine] = None, performance_monitor=None):
        """
        Initialize PyQt6 UI.

        Args:
            vision_engine: Optional VisionEngine instance
            performance_monitor: Optional PerformanceMonitor instance
        """
        super().__init__()
        self.app: Optional[QApplication] = None
        self.main_window: Optional[PyQt6MainWindow] = None
        self.vision_engine = vision_engine
        self.performance_monitor = performance_monitor
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize PyQt6 UI system.

        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing PyQt6 UI")

            # Create QApplication if needed
            if not QApplication.instance():
                self.app = QApplication([])
            else:
                self.app = QApplication.instance()

            # Create vision engine if not provided
            if not self.vision_engine:
                logger.info("Creating default VisionEngine")
                self.vision_engine = MediaPipeVisionEngine(
                    camera_id=-1,
                    fps=30,
                    max_queue_size=2,
                    enable_smoothing=True,
                    smoothing_factor=0.3,
                    max_num_hands=2,
                    performance_monitor=self.performance_monitor,
                )

                if not self.vision_engine.initialize():
                    logger.error("Failed to initialize VisionEngine")
                    return False

            # Create main window
            self.main_window = PyQt6MainWindow(self.vision_engine, self.performance_monitor)

            self._initialized = True
            logger.info("PyQt6 UI initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PyQt6 UI: {e}", exc_info=True)
            return False

    def render_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Render frame with UI overlays.

        Note: In PyQt6 implementation, rendering is handled by the UI thread.
        This method is provided for interface compatibility.
        """
        return frame

    def draw_landmarks(self, frame: np.ndarray, landmarks: Any) -> np.ndarray:
        """
        Draw hand landmarks.

        Note: In PyQt6 implementation, landmark drawing is handled by the UI thread.
        This method is provided for interface compatibility.
        """
        return frame

    def display_mode(self, frame: np.ndarray, mode: str) -> np.ndarray:
        """
        Display current mode.

        Note: In PyQt6 implementation, mode display is handled by status panel.
        This method is provided for interface compatibility.
        """
        if self.main_window:
            self.main_window.status_panel.update_mode(mode)
        return frame

    def display_gesture(self, frame: np.ndarray, gesture: str) -> np.ndarray:
        """
        Display detected gesture.

        Note: In PyQt6 implementation, gesture display is handled by status panel.
        This method is provided for interface compatibility.
        """
        if self.main_window:
            self.main_window.status_panel.update_gesture(gesture)
        return frame

    def display_message(self, frame: np.ndarray, message: str) -> np.ndarray:
        """
        Display message.

        Note: In PyQt6 implementation, messages are handled by status panel.
        This method is provided for interface compatibility.
        """
        return frame

    def show_frame(self, frame: np.ndarray) -> None:
        """
        Show frame in window.

        Note: In PyQt6 implementation, frame display is handled automatically.
        This method is provided for interface compatibility.
        """
        pass

    def handle_key_input(self) -> Optional[str]:
        """
        Handle keyboard input.

        Note: In PyQt6 implementation, input is handled by Qt event system.
        This method is provided for interface compatibility.
        """
        return None

    def run(self):
        """Run the UI event loop."""
        if not self._initialized:
            logger.error("UI not initialized. Call initialize() first.")
            return

        logger.info("Starting PyQt6 UI")

        # Show window
        self.main_window.show()

        # Start vision engine
        self.main_window.start_vision_engine()

        # Run event loop
        self.app.exec()

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up PyQt6 UI")

        if self.main_window:
            self.main_window.stop_vision_engine()

        if self.vision_engine:
            self.vision_engine.cleanup()

        logger.info("PyQt6 UI cleanup complete")
