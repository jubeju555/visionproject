#!/usr/bin/env python3
"""
Demo application for PyQt6 UI Framework.

This script demonstrates the PyQt6 UI system with:
- Live camera feed with hand landmark overlay
- Real-time status updates
- Interactive controls
- Thread-safe signal/slot communication
- Performance monitoring and metrics

Press Ctrl+C or close the window to exit.
"""

import sys
import logging
from src.ui.pyqt6_ui import PyQt6UI
from src.core.performance_monitor import PerformanceMonitor
from src.core.shutdown_handler import get_shutdown_handler, register_cleanup

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    logger.info("Starting PyQt6 UI Demo with Performance Monitoring")

    try:
        # Create performance monitor
        perf_monitor = PerformanceMonitor()
        logger.info("Performance monitor initialized")
        
        # Get shutdown handler
        shutdown_handler = get_shutdown_handler()
        
        # Create PyQt6 UI with performance monitor
        ui = PyQt6UI(performance_monitor=perf_monitor)

        # Initialize
        if not ui.initialize():
            logger.error("Failed to initialize PyQt6 UI")
            return 1

        logger.info("PyQt6 UI initialized successfully")
        
        # Register cleanup callbacks
        register_cleanup(ui.cleanup, name="PyQt6UI_cleanup")
        register_cleanup(perf_monitor.log_summary, name="PerformanceMonitor_summary")

        # Run the UI (blocks until window is closed)
        ui.run()

        # Cleanup (will also be called by shutdown handler)
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
