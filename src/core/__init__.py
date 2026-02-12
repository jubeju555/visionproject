"""
Core module for gesture-media-interface.

This module contains the abstract base classes and interfaces that define
the core architecture of the application.
"""

from .vision_engine import VisionEngine
from .gesture_engine import GestureEngine
from .mode_router import ModeRouter, ApplicationMode
from .audio_controller import AudioController, PlaybackState
from .image_editor import ImageEditor, ImageOperation
from .app_ui import AppUI
from .state_manager import StateManager
from .performance_monitor import PerformanceMonitor
from .shutdown_handler import ShutdownHandler, get_shutdown_handler, register_cleanup, trigger_shutdown, is_shutdown_requested, wait_for_shutdown

__all__ = [
    'VisionEngine',
    'GestureEngine',
    'ModeRouter',
    'ApplicationMode',
    'AudioController',
    'PlaybackState',
    'ImageEditor',
    'ImageOperation',
    'AppUI',
    'StateManager',
    'PerformanceMonitor',
    'ShutdownHandler',
    'get_shutdown_handler',
    'register_cleanup',
    'trigger_shutdown',
    'is_shutdown_requested',
    'wait_for_shutdown',
]
