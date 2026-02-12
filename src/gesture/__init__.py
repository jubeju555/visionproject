"""
Gesture recognition module.

This module implements hand tracking and gesture classification
using MediaPipe.
"""

from .hand_tracker import HandTracker
from .gesture_classifier import GestureClassifier
from .gesture_recognition_engine import (
    GestureRecognitionEngine,
    GestureEvent,
    StaticGesture,
    DynamicGesture,
)

__all__ = [
    'HandTracker',
    'GestureClassifier',
    'GestureRecognitionEngine',
    'GestureEvent',
    'StaticGesture',
    'DynamicGesture',
]
