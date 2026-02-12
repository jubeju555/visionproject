#!/usr/bin/env python3
"""
Complete integration example: VisionEngine + GestureEngine + ImageEditor

Demonstrates full gesture-controlled image editing with live camera feed.
This example shows how to:
1. Capture frames from VisionEngine
2. Recognize gestures with GestureRecognitionEngine
3. Apply edits with ImageManipulator
4. Display results with OpenCV

Usage:
- Run with camera connected
- Press 'f' to freeze current frame and enter edit mode
- Use gestures to edit the frozen image:
  * Pinch + drag: Move image
  * Circular motion: Rotate
  * Two-hand stretch: Scale
  * Swipe left: Undo
  * Open palm (vertical position): Brightness
- Press 's' to save edited image
- Press 'r' to reset to original
- Press 'q' to quit
"""

import sys
import logging
import cv2
import numpy as np
import queue
import time
from pathlib import Path

from src.vision.vision_engine_impl import MediaPipeVisionEngine
from src.gesture.gesture_recognition_engine import GestureRecognitionEngine
from src.image.editor import ImageManipulator
from src.image.gesture_integration import GestureImageEditorIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GestureImageEditorApp:
    """
    Complete application integrating vision, gestures, and image editing.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.vision_engine = None
        self.gesture_engine = None
        self.image_editor = None
        self.gesture_integration = None
        
        # Application state
        self.edit_mode = False
        self.output_dir = Path("/tmp")
        self.save_counter = 0
        
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Initializing application components...")
            
            # Initialize VisionEngine
            self.vision_engine = MediaPipeVisionEngine(
                camera_id=0,
                fps=30,
                max_num_hands=2,
                enable_smoothing=True
            )
            
            if not self.vision_engine.initialize():
                logger.error("Failed to initialize VisionEngine")
                return False
            
            # Initialize GestureEngine
            gesture_input_queue = self.vision_engine._output_queue
            self.gesture_engine = GestureRecognitionEngine(
                input_queue=gesture_input_queue,
                max_queue_size=10,
                history_size=10
            )
            
            # Initialize ImageEditor
            self.image_editor = ImageManipulator(max_undo_stack=50)
            if not self.image_editor.initialize():
                logger.error("Failed to initialize ImageEditor")
                return False
            
            # Initialize GestureIntegration
            self.gesture_integration = GestureImageEditorIntegration(
                gesture_engine=self.gesture_engine,
                image_editor=self.image_editor,
                translation_sensitivity=2.0,
                rotation_sensitivity=5.0,
                scale_sensitivity=0.1,
                brightness_sensitivity=0.5
            )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    def start(self) -> None:
        """Start all components."""
        logger.info("Starting components...")
        
        # Start VisionEngine
        self.vision_engine.start_capture()
        
        # Start GestureEngine
        self.gesture_engine.start()
        
        logger.info("Components started")
    
    def stop(self) -> None:
        """Stop all components."""
        logger.info("Stopping components...")
        
        # Stop gesture integration if running
        if self.gesture_integration and self.gesture_integration.is_running():
            self.gesture_integration.stop()
        
        # Stop GestureEngine
        if self.gesture_engine:
            self.gesture_engine.stop()
        
        # Stop VisionEngine
        if self.vision_engine:
            self.vision_engine.stop_capture()
        
        logger.info("Components stopped")
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up...")
        
        if self.gesture_integration:
            self.gesture_integration.stop()
        
        if self.image_editor:
            self.image_editor.cleanup()
        
        if self.gesture_engine:
            self.gesture_engine.stop()
        
        if self.vision_engine:
            self.vision_engine.cleanup()
        
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")
    
    def enter_edit_mode(self, frame: np.ndarray) -> None:
        """
        Enter edit mode with frozen frame.
        
        Args:
            frame: Frame to freeze and edit
        """
        logger.info("Entering edit mode...")
        
        # Freeze frame in editor
        if self.image_editor.freeze_frame(frame):
            self.edit_mode = True
            
            # Start gesture integration
            self.gesture_integration.start()
            
            logger.info("Edit mode activated - use gestures to edit")
        else:
            logger.error("Failed to freeze frame")
    
    def exit_edit_mode(self) -> None:
        """Exit edit mode and return to live view."""
        logger.info("Exiting edit mode...")
        
        self.edit_mode = False
        
        # Stop gesture integration
        if self.gesture_integration.is_running():
            self.gesture_integration.stop()
        
        logger.info("Edit mode deactivated")
    
    def save_current_image(self) -> None:
        """Save the current edited image."""
        if not self.edit_mode:
            logger.warning("Not in edit mode - nothing to save")
            return
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"edited_image_{timestamp}.png"
        
        # Save
        if self.image_editor.save_image(str(filename)):
            logger.info(f"Image saved: {filename}")
            self.save_counter += 1
        else:
            logger.error("Failed to save image")
    
    def draw_ui_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw UI overlay with instructions and status.
        
        Args:
            frame: Frame to draw on
            
        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Mode indicator
        mode_text = "EDIT MODE" if self.edit_mode else "LIVE VIEW"
        mode_color = (0, 255, 0) if self.edit_mode else (255, 255, 255)
        cv2.putText(frame, mode_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, mode_color, 2)
        
        # Instructions
        if self.edit_mode:
            instructions = [
                "Pinch+Drag: Move | Circular: Rotate | 2-Hand: Scale",
                "Swipe Left: Undo | Palm Tilt: Brightness",
                "Press 's' to save | 'r' to reset | 'q' to quit"
            ]
        else:
            instructions = [
                "Press 'f' to freeze frame and enter edit mode",
                "Press 'q' to quit"
            ]
        
        y = 70
        for instruction in instructions:
            cv2.putText(frame, instruction, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25
        
        return frame
    
    def run(self) -> int:
        """
        Main application loop.
        
        Returns:
            int: Exit code
        """
        logger.info("Starting main loop...")
        
        try:
            while True:
                # Get frame
                if self.edit_mode:
                    # In edit mode, show edited image
                    frame = self.image_editor.get_image()
                    if frame is None:
                        logger.warning("No edited image available")
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    # In live mode, show camera feed
                    frame = self.vision_engine.get_frame()
                    if frame is None:
                        # Wait for first frame
                        cv2.waitKey(10)
                        continue
                
                # Draw UI overlay
                display_frame = self.draw_ui_overlay(frame)
                
                # Show frame
                cv2.imshow("Gesture Image Editor", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Quit
                    logger.info("Quit requested")
                    break
                
                elif key == ord('f') and not self.edit_mode:
                    # Freeze frame and enter edit mode
                    self.enter_edit_mode(frame)
                
                elif key == ord('s') and self.edit_mode:
                    # Save current edited image
                    self.save_current_image()
                
                elif key == ord('r') and self.edit_mode:
                    # Reset to original
                    self.image_editor.reset()
                    logger.info("Reset to original image")
                
                elif key == ord('e') and self.edit_mode:
                    # Exit edit mode
                    self.exit_edit_mode()
            
            logger.info("Main loop ended")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return 0
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            return 1


def main():
    """Main entry point."""
    logger.info("=== Gesture Image Editor Application ===")
    
    app = GestureImageEditorApp()
    
    try:
        # Initialize
        if not app.initialize():
            logger.error("Initialization failed")
            return 1
        
        # Start
        app.start()
        
        # Give components time to start
        time.sleep(1.0)
        
        # Run main loop
        exit_code = app.run()
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        app.stop()
        app.cleanup()


if __name__ == "__main__":
    sys.exit(main())
