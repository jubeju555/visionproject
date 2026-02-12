"""
Gesture-to-ImageEditor integration module.

Maps gesture events to ImageEditor operations for gesture-controlled editing.
Implements the gesture mappings specified in the requirements:
- Pinch + drag → move region (translate)
- Two-hand stretch → scale
- Circular motion → rotate
- Swipe left → undo
- Palm tilt → brightness adjustment
"""

import logging
import queue
import threading
import time
from typing import Optional, Dict, Any
import numpy as np

from src.gesture.gesture_recognition_engine import GestureRecognitionEngine, GestureEvent
from src.image.editor import ImageManipulator

logger = logging.getLogger(__name__)


class GestureImageEditorIntegration:
    """
    Integration layer between GestureRecognitionEngine and ImageManipulator.
    
    Translates gesture events into image editing operations with appropriate
    parameter mapping based on continuous control data.
    """
    
    def __init__(
        self,
        gesture_engine: GestureRecognitionEngine,
        image_editor: ImageManipulator,
        translation_sensitivity: float = 2.0,
        rotation_sensitivity: float = 5.0,
        scale_sensitivity: float = 0.1,
        brightness_sensitivity: float = 0.5,
    ):
        """
        Initialize gesture-editor integration.
        
        Args:
            gesture_engine: GestureRecognitionEngine instance
            image_editor: ImageManipulator instance
            translation_sensitivity: Multiplier for translation gestures
            rotation_sensitivity: Degrees per circular motion unit
            scale_sensitivity: Scale factor change per spread gesture
            brightness_sensitivity: Brightness change per tilt unit
        """
        self.gesture_engine = gesture_engine
        self.image_editor = image_editor
        
        # Sensitivity settings
        self.translation_sensitivity = translation_sensitivity
        self.rotation_sensitivity = rotation_sensitivity
        self.scale_sensitivity = scale_sensitivity
        self.brightness_sensitivity = brightness_sensitivity
        
        # State tracking
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Gesture state tracking
        self._last_pinch_position: Optional[tuple] = None
        self._accumulated_rotation: float = 0.0
        self._accumulated_scale: float = 1.0
        self._last_palm_tilt: float = 0.0
        
        # Debouncing for discrete gestures
        self._last_undo_time: float = 0.0
        self._undo_cooldown: float = 1.0  # seconds
        
    def start(self) -> None:
        """Start the integration loop."""
        if self._running:
            logger.warning("Integration already running")
            return
        
        logger.info("Starting Gesture-ImageEditor integration")
        self._running = True
        self._thread = threading.Thread(
            target=self._integration_loop,
            daemon=True,
            name="GestureImageEditorThread"
        )
        self._thread.start()
        logger.info("Gesture-ImageEditor integration started")
    
    def stop(self) -> None:
        """Stop the integration loop."""
        if not self._running:
            logger.warning("Integration not running")
            return
        
        logger.info("Stopping Gesture-ImageEditor integration")
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Integration thread did not stop gracefully")
        
        logger.info("Gesture-ImageEditor integration stopped")
    
    def is_running(self) -> bool:
        """Check if integration is running."""
        return self._running
    
    def _integration_loop(self) -> None:
        """Main integration loop."""
        logger.info("Integration loop started")
        
        try:
            while self._running:
                # Get gesture event with timeout
                event = self.gesture_engine.get_event(timeout=0.1)
                
                if event is None:
                    continue
                
                # Process gesture event
                self._process_gesture_event(event)
                
        except Exception as e:
            logger.error(f"Error in integration loop: {e}", exc_info=True)
        finally:
            logger.info("Integration loop ended")
    
    def _process_gesture_event(self, event: GestureEvent) -> None:
        """
        Process a gesture event and map to image editor operations.
        
        Args:
            event: GestureEvent to process
        """
        gesture_name = event.gesture_name
        additional_data = event.additional_data or {}
        
        # Static gesture mappings
        if event.gesture_type == 'static':
            self._process_static_gesture(event)
        
        # Dynamic gesture mappings
        elif event.gesture_type == 'dynamic':
            self._process_dynamic_gesture(event)
    
    def _process_static_gesture(self, event: GestureEvent) -> None:
        """
        Process static gesture for continuous controls.
        
        Args:
            event: GestureEvent with static gesture
        """
        gesture_name = event.gesture_name
        additional_data = event.additional_data or {}
        
        # Pinch gesture: Translate if dragging
        if gesture_name == 'pinch':
            self._handle_pinch_drag(additional_data)
        
        # Open palm: Can be used for brightness via vertical position
        elif gesture_name == 'open_palm':
            self._handle_palm_tilt(additional_data)
    
    def _process_dynamic_gesture(self, event: GestureEvent) -> None:
        """
        Process dynamic gesture.
        
        Args:
            event: GestureEvent with dynamic gesture
        """
        gesture_name = event.gesture_name
        
        # Swipe left: Undo
        if gesture_name == 'swipe_left':
            self._handle_undo()
        
        # Circular motion: Rotate
        elif gesture_name == 'circular_motion':
            self._handle_rotation()
        
        # Two-hand spread: Scale
        elif gesture_name == 'two_hand_spread':
            self._handle_scale()
    
    def _handle_pinch_drag(self, data: Dict[str, Any]) -> None:
        """
        Handle pinch + drag gesture for translation.
        
        Args:
            data: Additional gesture data with hand position
        """
        # Get hand position from data
        h_pos = data.get('horizontal_position', 0.5)
        v_pos = data.get('vertical_position', 0.5)
        
        with self._lock:
            if self._last_pinch_position is not None:
                # Calculate delta from last position
                last_h, last_v = self._last_pinch_position
                dh = (h_pos - last_h) * 640  # Assuming 640px width
                dv = (v_pos - last_v) * 480  # Assuming 480px height
                
                # Apply translation with sensitivity
                dx = dh * self.translation_sensitivity
                dy = dv * self.translation_sensitivity
                
                if abs(dx) > 1 or abs(dy) > 1:
                    logger.debug(f"Translate: dx={dx:.1f}, dy={dy:.1f}")
                    self.image_editor.translate(dx, dy)
            
            # Update last position
            self._last_pinch_position = (h_pos, v_pos)
    
    def _handle_rotation(self) -> None:
        """Handle circular motion gesture for rotation."""
        with self._lock:
            # Apply incremental rotation
            angle = self.rotation_sensitivity
            self._accumulated_rotation += angle
            
            logger.debug(f"Rotate: angle={angle}°, total={self._accumulated_rotation}°")
            self.image_editor.rotate(angle)
    
    def _handle_scale(self) -> None:
        """Handle two-hand stretch gesture for scaling."""
        with self._lock:
            # Apply incremental scale
            scale_delta = self.scale_sensitivity
            new_scale = 1.0 + scale_delta
            self._accumulated_scale *= new_scale
            
            logger.debug(f"Scale: factor={new_scale}, total={self._accumulated_scale}")
            self.image_editor.scale(new_scale)
    
    def _handle_undo(self) -> None:
        """Handle swipe left gesture for undo with debouncing."""
        current_time = time.time()
        
        with self._lock:
            # Check cooldown
            if current_time - self._last_undo_time < self._undo_cooldown:
                logger.debug("Undo on cooldown, ignoring")
                return
            
            logger.info("Undo triggered by swipe left")
            self.image_editor.undo()
            self._last_undo_time = current_time
    
    def _handle_palm_tilt(self, data: Dict[str, Any]) -> None:
        """
        Handle palm tilt gesture for brightness adjustment.
        
        Uses vertical position of palm: higher = brighter, lower = darker
        
        Args:
            data: Additional gesture data with vertical position
        """
        v_pos = data.get('vertical_position', 0.5)
        
        with self._lock:
            # Map vertical position (0=top, 1=bottom) to brightness (-1 to 1)
            # Center (0.5) = no change (0)
            # Top (0) = brighten (positive)
            # Bottom (1) = darken (negative)
            brightness = (0.5 - v_pos) * 2.0 * self.brightness_sensitivity
            
            # Only apply if significant change
            if abs(brightness - self._last_palm_tilt) > 0.05:
                logger.debug(f"Brightness: {brightness:.2f}")
                self.image_editor.adjust_brightness(brightness)
                self._last_palm_tilt = brightness
    
    def reset_state(self) -> None:
        """Reset gesture state tracking."""
        with self._lock:
            self._last_pinch_position = None
            self._accumulated_rotation = 0.0
            self._accumulated_scale = 1.0
            self._last_palm_tilt = 0.0
            self._last_undo_time = 0.0
            logger.debug("Gesture state reset")
    
    def set_sensitivities(
        self,
        translation: Optional[float] = None,
        rotation: Optional[float] = None,
        scale: Optional[float] = None,
        brightness: Optional[float] = None,
    ) -> None:
        """
        Update sensitivity settings.
        
        Args:
            translation: New translation sensitivity
            rotation: New rotation sensitivity
            scale: New scale sensitivity
            brightness: New brightness sensitivity
        """
        with self._lock:
            if translation is not None:
                self.translation_sensitivity = translation
                logger.info(f"Translation sensitivity: {translation}")
            if rotation is not None:
                self.rotation_sensitivity = rotation
                logger.info(f"Rotation sensitivity: {rotation}")
            if scale is not None:
                self.scale_sensitivity = scale
                logger.info(f"Scale sensitivity: {scale}")
            if brightness is not None:
                self.brightness_sensitivity = brightness
                logger.info(f"Brightness sensitivity: {brightness}")
