"""
Abstract base class for gesture recognition engine.

This module defines the interface for hand landmark detection and
gesture classification using MediaPipe.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np


class GestureEngine(ABC):
    """
    Abstract interface for hand landmark detection and gesture classification.
    
    The GestureEngine is responsible for:
    - Processing frames to detect hand landmarks
    - Classifying gestures from landmarks
    - Managing MediaPipe hand tracking
    - Providing gesture events to the application
    """
    
    def __init__(self):
        """Initialize the gesture engine."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize MediaPipe and gesture recognition models.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a frame to detect hand landmarks and classify gestures.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing:
                - 'landmarks': List of hand landmarks
                - 'gesture': Detected gesture type
                - 'confidence': Confidence score
                - Additional gesture-specific data
            Returns None if no hand detected
        """
        pass
    
    @abstractmethod
    def get_landmarks(self) -> Optional[List[Any]]:
        """
        Get the current hand landmarks.
        
        Returns:
            Optional[List[Any]]: List of hand landmarks, or None if unavailable
        """
        pass
    
    @abstractmethod
    def classify_gesture(self, landmarks: List[Any]) -> Optional[str]:
        """
        Classify gesture from hand landmarks.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Optional[str]: Gesture type identifier, or None if no gesture recognized
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and release MediaPipe."""
        pass
