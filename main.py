#!/usr/bin/env python3
"""
Main entry point for Gesture Media Interface application.

This module initializes and wires together all components of the
gesture-controlled multimedia system.
"""

import sys
import logging
from typing import Optional

# Import core interfaces and implementations
from src.core import (
    VisionEngine,
    GestureEngine,
    ModeRouter,
    AudioController,
    ImageEditor,
    AppUI,
    StateManager,
    ApplicationMode,
)
from src.vision import CameraCapture
from src.gesture import HandTracker
from src.audio import AudioPlayer
from src.image import ImageManipulator
from src.ui import UIRenderer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GestureMediaInterface:
    """
    Main application class for Gesture Media Interface.
    
    Coordinates all subsystems and manages the application lifecycle.
    """
    
    def __init__(self):
        """Initialize the application."""
        logger.info("Initializing Gesture Media Interface...")
        
        # Initialize subsystem components
        self.vision_engine: Optional[VisionEngine] = None
        self.gesture_engine: Optional[GestureEngine] = None
        self.mode_router: Optional[ModeRouter] = None
        self.audio_controller: Optional[AudioController] = None
        self.image_editor: Optional[ImageEditor] = None
        self.ui: Optional[AppUI] = None
        
        self._running = False
    
    def initialize(self) -> bool:
        """
        Initialize all subsystems.
        
        Returns:
            bool: True if all subsystems initialized successfully
        """
        try:
            # Create subsystem instances
            logger.info("Creating subsystem instances...")
            self.vision_engine = CameraCapture(camera_id=0)
            self.gesture_engine = HandTracker()
            self.mode_router = StateManager()
            self.audio_controller = AudioPlayer()
            self.image_editor = ImageManipulator()
            self.ui = UIRenderer(window_name="Gesture Media Interface")
            
            # Initialize each subsystem
            logger.info("Initializing vision engine...")
            if not self.vision_engine.initialize():
                logger.error("Failed to initialize vision engine")
                return False
            
            logger.info("Initializing gesture engine...")
            if not self.gesture_engine.initialize():
                logger.error("Failed to initialize gesture engine")
                return False
            
            logger.info("Initializing mode router...")
            if not self.mode_router.initialize():
                logger.error("Failed to initialize mode router")
                return False
            
            logger.info("Initializing audio controller...")
            if not self.audio_controller.initialize():
                logger.error("Failed to initialize audio controller")
                return False
            
            logger.info("Initializing image editor...")
            if not self.image_editor.initialize():
                logger.error("Failed to initialize image editor")
                return False
            
            logger.info("Initializing UI...")
            if not self.ui.initialize():
                logger.error("Failed to initialize UI")
                return False
            
            # Register gesture handlers (placeholder for now)
            self._register_handlers()
            
            logger.info("All subsystems initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            return False
    
    def _register_handlers(self) -> None:
        """
        Register gesture handlers for each mode.
        
        This method sets up the mapping between gestures and actions
        for different application modes.
        """
        # TODO: Implement gesture handler registration
        # This is where we would register handlers like:
        # self.mode_router.register_handler(ApplicationMode.AUDIO, "play", self._handle_play)
        # self.mode_router.register_handler(ApplicationMode.AUDIO, "pause", self._handle_pause)
        # etc.
        logger.info("Gesture handlers registration (placeholder)")
        pass
    
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
            # Start frame capture
            if self.vision_engine:
                self.vision_engine.start_capture()
            
            # Main loop (placeholder - actual implementation will come later)
            while self._running:
                # TODO: Implement main processing loop
                # 1. Get frame from vision engine
                # 2. Process frame with gesture engine
                # 3. Route gesture through mode router
                # 4. Process events
                # 5. Render UI
                # 6. Handle user input
                
                # For now, just a placeholder
                pass
                
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
        
        if self.image_editor:
            logger.info("Cleaning up image editor...")
            self.image_editor.cleanup()
        
        if self.audio_controller:
            logger.info("Cleaning up audio controller...")
            self.audio_controller.cleanup()
        
        if self.mode_router:
            logger.info("Cleaning up mode router...")
            self.mode_router.cleanup()
        
        if self.gesture_engine:
            logger.info("Cleaning up gesture engine...")
            self.gesture_engine.cleanup()
        
        if self.vision_engine:
            logger.info("Cleaning up vision engine...")
            self.vision_engine.cleanup()
        
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
        
        # Initialize all subsystems
        if not app.initialize():
            logger.error("Failed to initialize application")
            return 1
        
        # Run the application
        app.run()
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
