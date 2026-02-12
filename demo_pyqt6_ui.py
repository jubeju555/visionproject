#!/usr/bin/env python3
"""
Demo application for PyQt6 UI Framework.

This script demonstrates the PyQt6 UI system with:
- Live camera feed with hand landmark overlay
- Real-time status updates
- Interactive controls
- Thread-safe signal/slot communication

Press Ctrl+C or close the window to exit.
"""

import sys
import logging
from src.ui.pyqt6_ui import PyQt6UI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    logger.info("Starting PyQt6 UI Demo")

    try:
        # Create PyQt6 UI
        ui = PyQt6UI()

        # Initialize
        if not ui.initialize():
            logger.error("Failed to initialize PyQt6 UI")
            return 1

        logger.info("PyQt6 UI initialized successfully")

        # Run the UI (blocks until window is closed)
        ui.run()

        # Cleanup
        ui.cleanup()

        logger.info("Demo finished")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
