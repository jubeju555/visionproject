"""
Abstract base class for application mode router and state manager.

This module defines the interface for managing application state and
routing gesture commands to appropriate action modules.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from enum import Enum


class ApplicationMode(Enum):
    """Enumeration of application modes."""
    NEUTRAL = "neutral"
    AUDIO_CONTROL = "audio_control"
    IMAGE_EDITING = "image_editing"


class ModeRouter(ABC):
    """
    Abstract interface for state management and command routing.
    
    The ModeRouter is responsible for:
    - Managing application state/mode
    - Routing gesture commands to appropriate handlers
    - Maintaining thread-safe event queue
    - Coordinating between different action modules
    """
    
    def __init__(self):
        """Initialize the mode router."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the mode router and event queue.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def set_mode(self, mode: ApplicationMode) -> None:
        """
        Set the current application mode.
        
        Args:
            mode: The application mode to switch to
        """
        pass
    
    @abstractmethod
    def get_mode(self) -> ApplicationMode:
        """
        Get the current application mode.
        
        Returns:
            ApplicationMode: Current mode
        """
        pass
    
    @abstractmethod
    def route_gesture(self, gesture: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Route a gesture command to the appropriate handler.
        
        Args:
            gesture: Gesture type identifier
            data: Optional additional data associated with the gesture
        """
        pass
    
    @abstractmethod
    def register_handler(self, mode: ApplicationMode, gesture: str, handler: Callable) -> None:
        """
        Register a gesture handler for a specific mode.
        
        Args:
            mode: Application mode
            gesture: Gesture type identifier
            handler: Callable handler function
        """
        pass
    
    @abstractmethod
    def dispatch_event(self, event: Dict[str, Any]) -> None:
        """
        Dispatch an event to the event queue.
        
        Args:
            event: Event dictionary containing type and data
        """
        pass
    
    @abstractmethod
    def process_events(self) -> None:
        """Process pending events from the event queue."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and clear event queue."""
        pass
