"""
Abstract base class for image editing operations.

This module defines the interface for image manipulation operations
controlled by gestures.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from enum import Enum


class ImageOperation(Enum):
    """Enumeration of image editing operations."""
    ROTATE = "rotate"
    SCALE = "scale"
    CROP = "crop"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    FILTER = "filter"


class ImageEditor(ABC):
    """
    Abstract interface for image manipulation operations.
    
    The ImageEditor is responsible for:
    - Loading and saving images
    - Applying image transformations
    - Managing image editing state
    - Providing real-time preview
    """
    
    def __init__(self):
        """Initialize the image editor."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the image editor.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_image(self, filepath: str) -> bool:
        """
        Load an image for editing.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def save_image(self, filepath: str) -> bool:
        """
        Save the current image to file.
        
        Args:
            filepath: Path where to save the image
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_image(self) -> Optional[np.ndarray]:
        """
        Get the current image.
        
        Returns:
            Optional[np.ndarray]: Current image as numpy array, or None if unavailable
        """
        pass
    
    @abstractmethod
    def rotate(self, angle: float) -> None:
        """
        Rotate the image by the specified angle.
        
        Args:
            angle: Rotation angle in degrees
        """
        pass
    
    @abstractmethod
    def scale(self, factor: float) -> None:
        """
        Scale the image by the specified factor.
        
        Args:
            factor: Scaling factor (1.0 = original size)
        """
        pass
    
    @abstractmethod
    def crop(self, x: int, y: int, width: int, height: int) -> None:
        """
        Crop the image to the specified region.
        
        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of crop region
            height: Height of crop region
        """
        pass
    
    @abstractmethod
    def adjust_brightness(self, value: float) -> None:
        """
        Adjust image brightness.
        
        Args:
            value: Brightness adjustment value (-1.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def adjust_contrast(self, value: float) -> None:
        """
        Adjust image contrast.
        
        Args:
            value: Contrast adjustment value (-1.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def apply_filter(self, filter_name: str) -> None:
        """
        Apply a filter to the image.
        
        Args:
            filter_name: Name of the filter to apply
        """
        pass
    
    @abstractmethod
    def undo(self) -> None:
        """Undo the last operation."""
        pass
    
    @abstractmethod
    def redo(self) -> None:
        """Redo the last undone operation."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset image to original state."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
