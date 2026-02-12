"""
Abstract base class for vision/camera input engine.

This module defines the interface for capturing frames from camera
and managing the frame capture thread.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class VisionEngine(ABC):
    """
    Abstract interface for camera input and frame capture.
    
    The VisionEngine is responsible for:
    - Managing camera input device
    - Capturing frames in a separate thread
    - Providing thread-safe frame access
    - Handling camera lifecycle (initialization, cleanup)
    """
    
    def __init__(self):
        """Initialize the vision engine."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the camera and start frame capture.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame.
        
        Returns:
            Optional[np.ndarray]: Latest frame as numpy array, or None if unavailable
        """
        pass
    
    @abstractmethod
    def start_capture(self) -> None:
        """Start the frame capture thread."""
        pass
    
    @abstractmethod
    def stop_capture(self) -> None:
        """Stop the frame capture thread and release resources."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the capture thread is running.
        
        Returns:
            bool: True if running, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and release camera."""
        pass
