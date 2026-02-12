"""
Gesture Media Interface - Main package.

A production-grade, modular Python application for real-time gesture-controlled
multimedia and image manipulation.
"""

__version__ = '0.1.0'
__author__ = 'Gesture Media Interface Team'

from . import core
from . import vision
from . import gesture
from . import audio
from . import image

# Import UI module only if PyQt6 is available
try:
    from . import ui
    __all__ = [
        'core',
        'vision',
        'gesture',
        'audio',
        'image',
        'ui',
    ]
except ImportError:
    __all__ = [
        'core',
        'vision',
        'gesture',
        'audio',
        'image',
    ]
