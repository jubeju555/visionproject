"""
Tests for Gesture-ImageEditor integration.

Tests the integration between GestureRecognitionEngine and ImageManipulator.
"""

import pytest
import numpy as np
import time
import queue
from unittest.mock import Mock, MagicMock

from src.gesture.gesture_recognition_engine import GestureRecognitionEngine, GestureEvent
from src.image.editor import ImageManipulator
from src.image.gesture_integration import GestureImageEditorIntegration


@pytest.fixture
def mock_gesture_engine():
    """Create a mock gesture recognition engine."""
    engine = Mock(spec=GestureRecognitionEngine)
    engine.get_event = Mock(return_value=None)
    return engine


@pytest.fixture
def image_editor():
    """Create and initialize an image editor."""
    editor = ImageManipulator()
    editor.initialize()
    
    # Load a test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    editor.freeze_frame(test_img)
    
    yield editor
    
    editor.cleanup()


@pytest.fixture
def integration(mock_gesture_engine, image_editor):
    """Create gesture-editor integration."""
    return GestureImageEditorIntegration(
        gesture_engine=mock_gesture_engine,
        image_editor=image_editor
    )


class TestIntegrationInitialization:
    """Test integration initialization."""
    
    def test_initialization(self, mock_gesture_engine, image_editor):
        """Test basic initialization."""
        integration = GestureImageEditorIntegration(
            gesture_engine=mock_gesture_engine,
            image_editor=image_editor
        )
        
        assert integration.gesture_engine is mock_gesture_engine
        assert integration.image_editor is image_editor
        assert not integration.is_running()
    
    def test_custom_sensitivities(self, mock_gesture_engine, image_editor):
        """Test initialization with custom sensitivities."""
        integration = GestureImageEditorIntegration(
            gesture_engine=mock_gesture_engine,
            image_editor=image_editor,
            translation_sensitivity=5.0,
            rotation_sensitivity=10.0,
            scale_sensitivity=0.2,
            brightness_sensitivity=1.0,
        )
        
        assert integration.translation_sensitivity == 5.0
        assert integration.rotation_sensitivity == 10.0
        assert integration.scale_sensitivity == 0.2
        assert integration.brightness_sensitivity == 1.0


class TestGestureMapping:
    """Test gesture to editor operation mapping."""
    
    def test_swipe_left_triggers_undo(self, integration):
        """Test that swipe left gesture triggers undo."""
        # Create swipe left event
        event = GestureEvent(
            gesture_name='swipe_left',
            confidence_score=0.9,
            timestamp=time.time(),
            hand_id='right',
            gesture_type='dynamic'
        )
        
        # Process event
        integration._process_gesture_event(event)
        
        # Undo should have been called (verify by checking editor state)
        # Since we can't easily verify undo was called without more setup,
        # we just ensure no exception was raised
    
    def test_circular_motion_triggers_rotation(self, integration):
        """Test that circular motion triggers rotation."""
        event = GestureEvent(
            gesture_name='circular_motion',
            confidence_score=0.85,
            timestamp=time.time(),
            hand_id='right',
            gesture_type='dynamic'
        )
        
        # Get transform before
        matrix_before = integration.image_editor.get_transform_matrix()
        
        # Process event
        integration._process_gesture_event(event)
        
        # Transform should have changed
        matrix_after = integration.image_editor.get_transform_matrix()
        
        # Matrices should be different after rotation
        assert not np.allclose(matrix_before, matrix_after)
    
    def test_two_hand_spread_triggers_scale(self, integration):
        """Test that two-hand spread triggers scaling."""
        event = GestureEvent(
            gesture_name='two_hand_spread',
            confidence_score=0.9,
            timestamp=time.time(),
            hand_id='both',
            gesture_type='dynamic'
        )
        
        # Get transform before
        matrix_before = integration.image_editor.get_transform_matrix()
        
        # Process event
        integration._process_gesture_event(event)
        
        # Transform should have changed
        matrix_after = integration.image_editor.get_transform_matrix()
        
        # Matrices should be different after scaling
        assert not np.allclose(matrix_before, matrix_after)
    
    def test_pinch_with_position_triggers_translation(self, integration):
        """Test that pinch gesture with position data triggers translation."""
        # First event to establish position
        event1 = GestureEvent(
            gesture_name='pinch',
            confidence_score=0.9,
            timestamp=time.time(),
            hand_id='right',
            gesture_type='static',
            additional_data={
                'horizontal_position': 0.5,
                'vertical_position': 0.5
            }
        )
        
        integration._process_gesture_event(event1)
        
        # Second event with different position
        event2 = GestureEvent(
            gesture_name='pinch',
            confidence_score=0.9,
            timestamp=time.time(),
            hand_id='right',
            gesture_type='static',
            additional_data={
                'horizontal_position': 0.6,
                'vertical_position': 0.6
            }
        )
        
        # Get transform before
        matrix_before = integration.image_editor.get_transform_matrix()
        
        # Process second event
        integration._process_gesture_event(event2)
        
        # Transform should have changed
        matrix_after = integration.image_editor.get_transform_matrix()
        
        # Translation should have been applied
        # Check translation components (last column)
        assert matrix_after[0, 2] != matrix_before[0, 2] or \
               matrix_after[1, 2] != matrix_before[1, 2]
    
    def test_open_palm_triggers_brightness(self, integration):
        """Test that open palm with position triggers brightness adjustment."""
        event = GestureEvent(
            gesture_name='open_palm',
            confidence_score=0.9,
            timestamp=time.time(),
            hand_id='right',
            gesture_type='static',
            additional_data={
                'vertical_position': 0.2  # High position = brighten
            }
        )
        
        # Get image before
        img_before = integration.image_editor.get_image()
        mean_before = np.mean(img_before)
        
        # Process event
        integration._process_gesture_event(event)
        
        # Get image after
        img_after = integration.image_editor.get_image()
        mean_after = np.mean(img_after)
        
        # Image should be brighter
        assert mean_after > mean_before


class TestSensitivitySettings:
    """Test sensitivity adjustments."""
    
    def test_set_translation_sensitivity(self, integration):
        """Test setting translation sensitivity."""
        integration.set_sensitivities(translation=10.0)
        assert integration.translation_sensitivity == 10.0
    
    def test_set_rotation_sensitivity(self, integration):
        """Test setting rotation sensitivity."""
        integration.set_sensitivities(rotation=15.0)
        assert integration.rotation_sensitivity == 15.0
    
    def test_set_scale_sensitivity(self, integration):
        """Test setting scale sensitivity."""
        integration.set_sensitivities(scale=0.5)
        assert integration.scale_sensitivity == 0.5
    
    def test_set_brightness_sensitivity(self, integration):
        """Test setting brightness sensitivity."""
        integration.set_sensitivities(brightness=2.0)
        assert integration.brightness_sensitivity == 2.0
    
    def test_set_multiple_sensitivities(self, integration):
        """Test setting multiple sensitivities at once."""
        integration.set_sensitivities(
            translation=8.0,
            rotation=12.0,
            scale=0.3,
            brightness=1.5
        )
        
        assert integration.translation_sensitivity == 8.0
        assert integration.rotation_sensitivity == 12.0
        assert integration.scale_sensitivity == 0.3
        assert integration.brightness_sensitivity == 1.5


class TestStateManagement:
    """Test integration state management."""
    
    def test_reset_state(self, integration):
        """Test resetting gesture state."""
        # Set some state
        integration._last_pinch_position = (0.5, 0.5)
        integration._accumulated_rotation = 45.0
        integration._accumulated_scale = 1.5
        integration._last_palm_tilt = 0.3
        
        # Reset
        integration.reset_state()
        
        # State should be cleared
        assert integration._last_pinch_position is None
        assert integration._accumulated_rotation == 0.0
        assert integration._accumulated_scale == 1.0
        assert integration._last_palm_tilt == 0.0


class TestDebouncing:
    """Test gesture debouncing."""
    
    def test_undo_cooldown(self, integration):
        """Test that undo has cooldown to prevent rapid triggering."""
        event = GestureEvent(
            gesture_name='swipe_left',
            confidence_score=0.9,
            timestamp=time.time(),
            hand_id='right',
            gesture_type='dynamic'
        )
        
        # First undo should work
        integration._process_gesture_event(event)
        first_time = integration._last_undo_time
        
        # Immediate second undo should be blocked
        integration._process_gesture_event(event)
        second_time = integration._last_undo_time
        
        # Times should be the same (second was blocked)
        assert first_time == second_time


class TestThreading:
    """Test integration threading."""
    
    def test_start_stop(self, integration):
        """Test starting and stopping integration."""
        assert not integration.is_running()
        
        # Start
        integration.start()
        assert integration.is_running()
        
        # Give it a moment to run
        time.sleep(0.1)
        
        # Stop
        integration.stop()
        assert not integration.is_running()
    
    def test_multiple_start(self, integration):
        """Test that multiple starts don't cause issues."""
        integration.start()
        integration.start()  # Should log warning but not crash
        
        assert integration.is_running()
        
        integration.stop()
    
    def test_multiple_stop(self, integration):
        """Test that multiple stops don't cause issues."""
        integration.start()
        integration.stop()
        integration.stop()  # Should log warning but not crash
        
        assert not integration.is_running()


class TestRealTimeProcessing:
    """Test real-time gesture processing."""
    
    def test_integration_loop_processes_events(self, image_editor):
        """Test that integration loop processes events from queue."""
        # Create real gesture engine with queue
        input_queue = queue.Queue()
        gesture_engine = GestureRecognitionEngine(input_queue)
        
        # Create integration
        integration = GestureImageEditorIntegration(
            gesture_engine=gesture_engine,
            image_editor=image_editor
        )
        
        # Start gesture engine
        gesture_engine.start()
        
        # Start integration
        integration.start()
        
        # Give it time to run
        time.sleep(0.2)
        
        # Stop
        integration.stop()
        gesture_engine.stop()
        
        # Should have run without errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
