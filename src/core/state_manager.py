"""
Mode router implementation.

This module implements state management and command routing.
"""

from typing import Optional, Dict, Any, Callable, List
import queue
import threading
import time
import logging
from src.core.mode_router import ModeRouter, ApplicationMode

logger = logging.getLogger(__name__)


class StateManager(ModeRouter):
    """
    Concrete implementation of ModeRouter.
    
    Manages application state and routes gesture commands.
    Implements mode switching via both palms open gesture.
    """
    
    def __init__(self, mode_switch_duration: float = 2.0):
        """
        Initialize state manager.
        
        Args:
            mode_switch_duration: Duration in seconds both palms must be open to switch modes
        """
        self._current_mode = ApplicationMode.NEUTRAL
        self._event_queue = queue.Queue()
        self._handlers = {}
        self._lock = threading.Lock()
        self._running = False
        
        # Mode switching detection
        self._mode_switch_duration = mode_switch_duration
        self._both_palms_open_start: Optional[float] = None
        self._mode_change_callbacks: List[Callable[[ApplicationMode], None]] = []
    
    def initialize(self) -> bool:
        """
        Initialize mode router.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        logger.info("Initializing StateManager (Mode Router)")
        self._running = True
        self._both_palms_open_start = None
        logger.info(f"StateManager initialized in {self._current_mode.value} mode")
        return True
    
    def set_mode(self, mode: ApplicationMode) -> None:
        """
        Set application mode.
        
        Args:
            mode: Mode to switch to
        """
        with self._lock:
            old_mode = self._current_mode
            self._current_mode = mode
            
        if old_mode != mode:
            logger.info(f"Mode changed: {old_mode.value} -> {mode.value}")
            self._notify_mode_change(mode)
    
    def get_mode(self) -> ApplicationMode:
        """
        Get current mode.
        
        Returns:
            ApplicationMode: Current mode
        """
        with self._lock:
            return self._current_mode
    
    def register_mode_change_callback(self, callback: Callable[[ApplicationMode], None]) -> None:
        """
        Register a callback to be notified of mode changes.
        
        Args:
            callback: Function to call when mode changes, receives new mode as parameter
        """
        with self._lock:
            if callback not in self._mode_change_callbacks:
                self._mode_change_callbacks.append(callback)
                callback_name = getattr(callback, '__name__', repr(callback))
                logger.debug(f"Registered mode change callback: {callback_name}")
    
    def _notify_mode_change(self, new_mode: ApplicationMode) -> None:
        """
        Notify all registered callbacks of mode change.
        
        Args:
            new_mode: The new application mode
        """
        with self._lock:
            callbacks = self._mode_change_callbacks.copy()
        
        for callback in callbacks:
            try:
                callback(new_mode)
            except Exception as e:
                callback_name = getattr(callback, '__name__', repr(callback))
                logger.error(f"Error in mode change callback {callback_name}: {e}", exc_info=True)
    
    def route_gesture(self, gesture: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Route gesture to handler.
        
        First checks for mode switch gesture (both palms open for 2 seconds).
        If not mode switching, routes gesture to handler for current mode.
        
        Args:
            gesture: Gesture type
            data: Additional gesture data
        """
        if not self._running:
            return
        
        # Check for mode switch gesture: both palms open
        if self._check_mode_switch_gesture(gesture, data):
            return  # Mode switch handled, don't route to handlers
        
        # Route gesture to handler for current mode
        current_mode = self.get_mode()
        handler_key = (current_mode, gesture)
        
        with self._lock:
            handler = self._handlers.get(handler_key)
        
        if handler:
            try:
                # Call handler in non-blocking manner
                handler(data)
                logger.debug(f"Routed gesture '{gesture}' to handler in {current_mode.value} mode")
            except Exception as e:
                logger.error(f"Error in gesture handler for {gesture} in {current_mode.value}: {e}", exc_info=True)
        else:
            logger.debug(f"No handler registered for gesture '{gesture}' in {current_mode.value} mode")
    
    def _check_mode_switch_gesture(self, gesture: str, data: Optional[Dict[str, Any]]) -> bool:
        """
        Check if gesture is the mode switch gesture (both palms open).
        
        Mode switch requires both hands showing open palm gesture for configured duration.
        
        Args:
            gesture: Gesture type
            data: Additional gesture data
            
        Returns:
            bool: True if mode switch was triggered, False otherwise
        """
        # Check if this is an open palm gesture
        if gesture != "open_palm":
            self._both_palms_open_start = None
            return False
        
        # Check if we have data about both hands
        if not data:
            self._both_palms_open_start = None
            return False
        
        # Look for both hands in data
        # Data can contain hand_id or we need to track multiple hands
        hand_id = data.get('hand_id', '')
        
        # For simplicity, we'll check if the gesture indicates "both" hands
        # or if we receive the gesture and it's marked as both-hands gesture
        if hand_id == 'both' or data.get('both_hands', False):
            current_time = time.time()
            
            if self._both_palms_open_start is None:
                # Start timing
                self._both_palms_open_start = current_time
                logger.debug("Both palms open detected - starting mode switch timer")
                return False
            
            # Check if duration threshold met
            elapsed = current_time - self._both_palms_open_start
            if elapsed >= self._mode_switch_duration:
                logger.info(f"Mode switch gesture detected (both palms open for {elapsed:.1f}s)")
                self._both_palms_open_start = None
                self._cycle_mode()
                return True
        else:
            # Not both hands, reset timer
            self._both_palms_open_start = None
        
        return False
    
    def _cycle_mode(self) -> None:
        """Cycle to the next mode in sequence: NEUTRAL -> AUDIO_CONTROL -> IMAGE_EDITING -> NEUTRAL."""
        current_mode = self.get_mode()
        
        if current_mode == ApplicationMode.NEUTRAL:
            next_mode = ApplicationMode.AUDIO_CONTROL
        elif current_mode == ApplicationMode.AUDIO_CONTROL:
            next_mode = ApplicationMode.IMAGE_EDITING
        else:  # IMAGE_EDITING
            next_mode = ApplicationMode.NEUTRAL
        
        self.set_mode(next_mode)
    
    def register_handler(self, mode: ApplicationMode, gesture: str, handler: Callable) -> None:
        """
        Register gesture handler.
        
        Args:
            mode: Application mode
            gesture: Gesture type
            handler: Handler function
        """
        key = (mode, gesture)
        with self._lock:
            self._handlers[key] = handler
        logger.debug(f"Registered handler for ({mode.value}, {gesture})")
    
    def dispatch_event(self, event: Dict[str, Any]) -> None:
        """
        Dispatch event to queue for processing.
        
        Args:
            event: Event dictionary
        """
        if not self._running:
            return
        
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            logger.warning("Event queue full, dropping event")
    
    def process_events(self) -> None:
        """
        Process pending events from the event queue.
        
        This method should be called periodically (e.g., in main loop or UI update).
        It processes all pending events without blocking.
        """
        if not self._running:
            return
        
        processed_count = 0
        
        # Process all pending events (non-blocking)
        while True:
            try:
                event = self._event_queue.get_nowait()
                processed_count += 1
                
                # Extract gesture info from event
                gesture = event.get('gesture', event.get('gesture_name', ''))
                data = event.get('data', event)
                
                # Route gesture to appropriate handler
                self.route_gesture(gesture, data)
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)
        
        if processed_count > 0:
            logger.debug(f"Processed {processed_count} events")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up StateManager")
        self._running = False
        
        # Clear event queue
        cleared_count = 0
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        
        if cleared_count > 0:
            logger.debug(f"Cleared {cleared_count} pending events")
        
        # Clear handlers and callbacks
        with self._lock:
            self._handlers.clear()
            self._mode_change_callbacks.clear()
        
        logger.info("StateManager cleanup complete")
