"""
Graceful shutdown handler for the gesture media interface.

This module provides signal handling and coordinated shutdown of all subsystems:
- Catches SIGINT, SIGTERM signals
- Coordinates cleanup of all threads
- Ensures proper resource release
- Logs shutdown progress
"""

import signal
import logging
import sys
import atexit
from typing import List, Callable, Optional
from threading import Event, Lock

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """
    Central shutdown coordinator.
    
    Manages graceful shutdown of all subsystems by:
    - Registering cleanup callbacks
    - Handling system signals (SIGINT, SIGTERM)
    - Coordinating ordered shutdown
    - Preventing duplicate cleanup
    """
    
    def __init__(self):
        """Initialize shutdown handler."""
        self._lock = Lock()
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._shutdown_event = Event()
        self._shutdown_in_progress = False
        self._shutdown_complete = False
        
        # Register signal handlers
        self._register_signal_handlers()
        
        # Register atexit handler as fallback
        atexit.register(self._atexit_handler)
        
        logger.info("Shutdown handler initialized")
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        # Handle Ctrl+C (SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Handle kill command (SIGTERM)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.debug("Signal handlers registered (SIGINT, SIGTERM)")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal: {signal_name}")
        
        # Trigger shutdown
        self.trigger_shutdown()
    
    def _atexit_handler(self) -> None:
        """Fallback handler called at exit."""
        if not self._shutdown_complete:
            logger.debug("Atexit handler triggered")
            self.trigger_shutdown()
    
    def register_cleanup(self, callback: Callable[[], None], name: Optional[str] = None) -> None:
        """
        Register a cleanup callback.
        
        Callbacks are executed in reverse order of registration (LIFO).
        
        Args:
            callback: Function to call during shutdown
            name: Optional name for logging
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)
            callback_name = name or callback.__name__
            logger.debug(f"Registered cleanup callback: {callback_name}")
    
    def trigger_shutdown(self) -> None:
        """
        Trigger graceful shutdown.
        
        This can be called from signal handlers, the main thread,
        or from worker threads.
        """
        with self._lock:
            if self._shutdown_in_progress:
                logger.debug("Shutdown already in progress")
                return
            
            self._shutdown_in_progress = True
        
        logger.info("=" * 60)
        logger.info("INITIATING GRACEFUL SHUTDOWN")
        logger.info("=" * 60)
        
        # Set shutdown event
        self._shutdown_event.set()
        
        # Execute cleanup callbacks in reverse order (LIFO)
        cleanup_count = len(self._cleanup_callbacks)
        logger.info(f"Executing {cleanup_count} cleanup callbacks...")
        
        for idx, callback in enumerate(reversed(self._cleanup_callbacks), 1):
            try:
                callback_name = callback.__name__
                logger.info(f"[{idx}/{cleanup_count}] Cleaning up: {callback_name}")
                callback()
                logger.info(f"[{idx}/{cleanup_count}] Cleanup complete: {callback_name}")
            except Exception as e:
                logger.error(f"Error in cleanup callback {callback_name}: {e}", exc_info=True)
        
        with self._lock:
            self._shutdown_complete = True
        
        logger.info("=" * 60)
        logger.info("SHUTDOWN COMPLETE")
        logger.info("=" * 60)
    
    def is_shutdown_requested(self) -> bool:
        """
        Check if shutdown has been requested.
        
        Worker threads should check this periodically and exit gracefully.
        
        Returns:
            True if shutdown is in progress
        """
        return self._shutdown_event.is_set()
    
    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for shutdown signal.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if shutdown was signaled, False if timeout
        """
        return self._shutdown_event.wait(timeout)
    
    def reset(self) -> None:
        """
        Reset shutdown handler.
        
        WARNING: Only use this for testing purposes.
        """
        with self._lock:
            self._cleanup_callbacks.clear()
            self._shutdown_event.clear()
            self._shutdown_in_progress = False
            self._shutdown_complete = False
        logger.debug("Shutdown handler reset")


# Global singleton instance
_shutdown_handler: Optional[ShutdownHandler] = None


def get_shutdown_handler() -> ShutdownHandler:
    """
    Get the global shutdown handler instance.
    
    Returns:
        ShutdownHandler singleton
    """
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = ShutdownHandler()
    return _shutdown_handler


def register_cleanup(callback: Callable[[], None], name: Optional[str] = None) -> None:
    """
    Register a cleanup callback with the global shutdown handler.
    
    Args:
        callback: Function to call during shutdown
        name: Optional name for logging
    """
    get_shutdown_handler().register_cleanup(callback, name)


def trigger_shutdown() -> None:
    """Trigger graceful shutdown of the application."""
    get_shutdown_handler().trigger_shutdown()


def is_shutdown_requested() -> bool:
    """
    Check if shutdown has been requested.
    
    Returns:
        True if shutdown is in progress
    """
    return get_shutdown_handler().is_shutdown_requested()


def wait_for_shutdown(timeout: Optional[float] = None) -> bool:
    """
    Wait for shutdown signal.
    
    Args:
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if shutdown was signaled, False if timeout
    """
    return get_shutdown_handler().wait_for_shutdown(timeout)
