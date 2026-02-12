"""
Abstract base class for application UI rendering.

This module defines the interface for rendering the user interface
and displaying feedback to the user.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class AppUI(ABC):
    """
    Abstract interface for UI rendering layer.
    
    The AppUI is responsible for:
    - Rendering video frames with overlays
    - Displaying gesture feedback
    - Showing mode indicators
    - Rendering controls and menus
    - Managing UI state
    """
    
    def __init__(self):
        """Initialize the UI."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the UI system.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def render_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Render a frame with UI overlays.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            np.ndarray: Frame with UI elements rendered
        """
        pass
    
    @abstractmethod
    def draw_landmarks(self, frame: np.ndarray, landmarks: Any) -> np.ndarray:
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame: Input frame as numpy array
            landmarks: Hand landmarks to draw
            
        Returns:
            np.ndarray: Frame with landmarks drawn
        """
        pass
    
    @abstractmethod
    def display_mode(self, frame: np.ndarray, mode: str) -> np.ndarray:
        """
        Display the current application mode on the frame.
        
        Args:
            frame: Input frame as numpy array
            mode: Current application mode
            
        Returns:
            np.ndarray: Frame with mode indicator
        """
        pass
    
    @abstractmethod
    def display_gesture(self, frame: np.ndarray, gesture: str) -> np.ndarray:
        """
        Display the detected gesture on the frame.
        
        Args:
            frame: Input frame as numpy array
            gesture: Detected gesture type
            
        Returns:
            np.ndarray: Frame with gesture indicator
        """
        pass
    
    @abstractmethod
    def display_message(self, frame: np.ndarray, message: str) -> np.ndarray:
        """
        Display a message on the frame.
        
        Args:
            frame: Input frame as numpy array
            message: Message to display
            
        Returns:
            np.ndarray: Frame with message
        """
        pass
    
    @abstractmethod
    def show_frame(self, frame: np.ndarray) -> None:
        """
        Show the frame in a window.
        
        Args:
            frame: Frame to display
        """
        pass
    
    @abstractmethod
    def handle_key_input(self) -> Optional[str]:
        """
        Handle keyboard input from the user.
        
        Returns:
            Optional[str]: Key pressed, or None if no input
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and close windows."""
        pass
