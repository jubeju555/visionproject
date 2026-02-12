"""
Vision module for camera input and frame capture.

This module implements the VisionEngine interface for capturing
frames from camera in a separate thread.
"""

from .camera_capture import CameraCapture
from .vision_engine_impl import MediaPipeVisionEngine, VisionData

__all__ = ['CameraCapture', 'MediaPipeVisionEngine', 'VisionData']
