"""
Test cases for core module interfaces.
"""

import pytest
from src.core import (
    VisionEngine,
    GestureEngine,
    ModeRouter,
    AudioController,
    ImageEditor,
    AppUI,
)


def test_imports():
    """Test that all core interfaces can be imported."""
    assert VisionEngine is not None
    assert GestureEngine is not None
    assert ModeRouter is not None
    assert AudioController is not None
    assert ImageEditor is not None
    assert AppUI is not None


# TODO: Add more comprehensive tests once implementations are complete
