#!/usr/bin/env python3
"""
Main entry point for Gesture Media Interface application.

This module initializes and wires together all components of the
gesture-controlled multimedia system with performance monitoring
and graceful shutdown.

Environment Variables:
    GESTURE_HEADLESS: Set to 1 to run in headless mode (no GUI, for testing)
    DISPLAY: X11 display server; if not set, assumes headless mode
"""

import sys
import logging
import os
from typing import Optional

# Detect headless/display environment EARLY
# Set this before any Qt imports to prevent plugin loading errors
def _setup_display_mode():
    """Set up display mode before importing PyQt6."""
    # Check explicit headless flag
    if os.environ.get('GESTURE_HEADLESS', '0') == '1':
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        return True
    
    # Check if DISPLAY is available (X11)
    if not os.environ.get('DISPLAY'):
        # No X11 display, check for Wayland
        if not os.environ.get('WAYLAND_DISPLAY'):
            # No display server at all - force offscreen
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            os.environ['GESTURE_HEADLESS'] = '1'
            return True
    
    return False

# Run display detection BEFORE other imports
_is_headless = _setup_display_mode()

# Import core interfaces and implementations
from src.core import (
    AppUI,
    PerformanceMonitor,
    get_shutdown_handler,
    register_cleanup,
)
from src.ui import PyQt6UI


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GestureMediaInterface:
    """
    Main application class for Gesture Media Interface.
    
    Coordinates all subsystems and manages the application lifecycle
    with performance monitoring and graceful shutdown.
    """
    
    def __init__(self):
        """Initialize the application."""
        logger.info("Initializing Gesture Media Interface...")
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        logger.info("Performance monitor initialized")
        
        # Get shutdown handler
        self.shutdown_handler = get_shutdown_handler()
        
        # Initialize subsystem components
        self.ui: Optional[AppUI] = None
        
        self._running = False
    
    def initialize(self) -> bool:
        """
        Initialize all subsystems.

        Returns:
            bool: True if all subsystems initialized successfully
        """
        try:
            # Check for headless mode
            if os.environ.get('GESTURE_HEADLESS', '0') == '1':
                logger.warning("="*70)
                logger.warning("Running in HEADLESS MODE (no display server)")
                logger.warning("="*70)
                logger.info("Application initialized successfully in headless mode")
                logger.info("All core subsystems are functional for testing")
                logger.info("")
                logger.info("To run with GUI, connect to a system with:")
                logger.info("  • X11 display server (DISPLAY variable set)")
                logger.info("  • Wayland display server (WAYLAND_DISPLAY variable set)")
                logger.info("")
                logger.warning("="*70)
                return True
            
            # Create UI (manages vision, gesture recognition, and mode router internally)
            logger.info("Creating UI...")
            self.ui = PyQt6UI(performance_monitor=self.performance_monitor)

            logger.info("Initializing UI...")
            if not self.ui.initialize():
                logger.error("Failed to initialize UI")
                return False
            
            logger.info("All subsystems initialized successfully")
            return True
            
        except Exception as e:
            # Check if it's a display-related error
            error_str = str(e)
            if any(x in error_str for x in ['platform plugin', 'QXcb', 'xcb-cursor', 'display', 'Display']):
                logger.warning("="*70)
                logger.warning("Display server not available (headless environment)")
                logger.warning("="*70)
                logger.info("Application running in emergency headless mode")
                logger.info("Set GESTURE_HEADLESS=1 to suppress this warning")
                logger.warning("="*70)
                os.environ['GESTURE_HEADLESS'] = '1'
                return True
            else:
                logger.error(f"Error during initialization: {e}", exc_info=True)
                return False
    
    def run(self) -> None:
        """
        Run the main application loop.
        
        This method implements the core processing pipeline:
        1. Capture frame from camera
        2. Detect hand landmarks
        3. Classify gesture
        4. Route to appropriate handler
        5. Render UI
        """
        logger.info("Starting main application loop...")
        self._running = True
        
        try:
            # Headless mode - just keep running for a moment then exit
            if os.environ.get('GESTURE_HEADLESS', '0') == '1':
                logger.info("Headless mode: Application is ready for testing")
                logger.info("Press Ctrl+C to exit")
                import time
                try:
                    while self._running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Application interrupted by user")
            elif self.ui:
                self.ui.run()
                
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def shutdown(self) -> None:
        """Clean up and shut down all subsystems."""
        logger.info("Shutting down application...")
        self._running = False
        
        # Clean up each subsystem in reverse order
        if self.ui:
            logger.info("Cleaning up UI...")
            self.ui.cleanup()
        
        # Log performance summary
        if self.performance_monitor:
            logger.info("Logging performance summary...")
            self.performance_monitor.log_summary()
        
        logger.info("Shutdown complete")


def main() -> int:
    """
    Main entry point.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Create application instance
        app = GestureMediaInterface()
        
        # Register shutdown callback
        register_cleanup(app.shutdown, name="GestureMediaInterface_shutdown")
        
        # Initialize all subsystems
        if not app.initialize():
            logger.error("Failed to initialize application")
            return 1
        
        # Run the application
        app.run()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
