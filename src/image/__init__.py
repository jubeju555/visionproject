"""
Image editing module.

This module implements image manipulation operations.
"""

from .editor import ImageManipulator, TransformType, EditorState
from .gesture_integration import GestureImageEditorIntegration

__all__ = [
    'ImageManipulator',
    'TransformType',
    'EditorState',
    'GestureImageEditorIntegration',
]
