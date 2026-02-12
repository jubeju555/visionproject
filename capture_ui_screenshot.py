#!/usr/bin/env python3
"""
Script to capture a screenshot of the PyQt6 UI.
"""

import sys
import time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap
import logging

from test_ui_integration import MockVisionEngine
from src.ui.pyqt6_ui import PyQt6MainWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Capture screenshot of UI."""
    logger.info("Creating UI for screenshot...")
    
    # Create application
    app = QApplication([])
    
    # Create mock engine
    mock_engine = MockVisionEngine()
    mock_engine.initialize()
    mock_engine.start_capture()
    
    # Create window
    window = PyQt6MainWindow(mock_engine)
    window.show()
    
    # Start vision engine
    window.start_vision_engine()
    
    # Wait a bit for frames to be processed
    def take_screenshot():
        logger.info("Taking screenshot...")
        pixmap = window.grab()
        pixmap.save("pyqt6_ui_screenshot.png")
        logger.info("Screenshot saved to pyqt6_ui_screenshot.png")
        
        # Close after a moment
        QTimer.singleShot(500, app.quit)
    
    # Take screenshot after 2 seconds
    QTimer.singleShot(2000, take_screenshot)
    
    # Run app
    app.exec()
    
    # Cleanup
    window.stop_vision_engine()
    mock_engine.cleanup()
    
    logger.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
