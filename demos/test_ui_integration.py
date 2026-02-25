#!/usr/bin/env python3
"""
Integration test for PyQt6 UI with mock camera.

This script tests the UI with a simulated camera feed.
"""

import sys
import logging
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock
import queue
import time

from src.ui.pyqt6_ui import PyQt6UI
from src.vision.vision_engine_impl import VisionData

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockVisionEngine:
    """Mock VisionEngine for testing."""

    def __init__(self):
        self.initialized = False
        self.running = False
        self.smoothing = True
        self._output_queue = queue.Queue(maxsize=2)
        self._frame_count = 0

    def initialize(self) -> bool:
        """Initialize mock engine."""
        logger.info("Mock VisionEngine: initialize()")
        self.initialized = True
        return True

    def start_capture(self):
        """Start mock capture."""
        logger.info("Mock VisionEngine: start_capture()")
        self.running = True
        # Generate some mock frames
        self._generate_mock_frames()

    def stop_capture(self):
        """Stop mock capture."""
        logger.info("Mock VisionEngine: stop_capture()")
        self.running = False

    def cleanup(self):
        """Cleanup mock engine."""
        logger.info("Mock VisionEngine: cleanup()")
        self.running = False
        self.initialized = False

    def is_running(self) -> bool:
        """Check if running."""
        return self.running

    def set_smoothing(self, enabled: bool):
        """Set smoothing."""
        logger.info(f"Mock VisionEngine: set_smoothing({enabled})")
        self.smoothing = enabled

    def get_vision_data(self, timeout: float = None):
        """Get mock vision data."""
        if not self.running:
            return None

        # Generate a frame with mock hand landmarks
        frame = self._generate_frame()
        landmarks = self._generate_landmarks()

        vision_data = VisionData(
            frame=frame,
            frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            landmarks=landmarks,
            timestamp=time.time(),
        )

        return vision_data

    def _generate_frame(self):
        """Generate a mock camera frame."""
        self._frame_count += 1

        # Create a frame with a gradient
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add gradient
        for i in range(480):
            frame[i, :] = [i // 2, (480 - i) // 2, 100]

        # Add frame counter text
        cv2.putText(
            frame,
            f"Mock Frame {self._frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return frame

    def _generate_landmarks(self):
        """Generate mock hand landmarks."""
        # Simulate hand movement
        offset_x = int(100 * np.sin(self._frame_count * 0.1))
        offset_y = int(50 * np.cos(self._frame_count * 0.1))

        landmarks = [
            {
                "handedness": "Right",
                "confidence": 0.85 + 0.1 * np.sin(self._frame_count * 0.05),
                "landmarks": [
                    {
                        "x": 200 + offset_x + i * 10,
                        "y": 200 + offset_y + i * 10,
                        "z": 0.0,
                    }
                    for i in range(21)
                ],
            }
        ]

        return landmarks

    def _generate_mock_frames(self):
        """Pre-generate some mock frames."""
        pass  # Frames generated on-demand


def main():
    """Main test function."""
    logger.info("Starting PyQt6 UI Integration Test with Mock Camera")

    try:
        # Create mock vision engine
        mock_engine = MockVisionEngine()

        # Create PyQt6 UI with mock engine
        ui = PyQt6UI(vision_engine=mock_engine)

        # Initialize
        if not ui.initialize():
            logger.error("Failed to initialize PyQt6 UI")
            return 1

        logger.info("PyQt6 UI initialized successfully with mock camera")
        logger.info(
            "The UI will show a gradient background with simulated hand landmarks"
        )
        logger.info("Close the window to exit")

        # Run the UI (blocks until window is closed)
        ui.run()

        # Cleanup
        ui.cleanup()

        logger.info("Integration test finished successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error in integration test: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
