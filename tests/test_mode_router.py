"""
Test cases for Mode Router implementation.
"""

import pytest
import time
from unittest.mock import Mock, call
from src.core.state_manager import StateManager
from src.core.mode_router import ApplicationMode


class TestStateManager:
    """Test suite for StateManager (Mode Router implementation)."""
    
    def test_initialization(self):
        """Test state manager initialization."""
        router = StateManager()
        assert router.initialize() is True
        assert router.get_mode() == ApplicationMode.NEUTRAL
        router.cleanup()
    
    def test_mode_switching(self):
        """Test basic mode switching."""
        router = StateManager()
        router.initialize()
        
        # Test switching to each mode
        router.set_mode(ApplicationMode.AUDIO_CONTROL)
        assert router.get_mode() == ApplicationMode.AUDIO_CONTROL
        
        router.set_mode(ApplicationMode.IMAGE_EDITING)
        assert router.get_mode() == ApplicationMode.IMAGE_EDITING
        
        router.set_mode(ApplicationMode.NEUTRAL)
        assert router.get_mode() == ApplicationMode.NEUTRAL
        
        router.cleanup()
    
    def test_mode_change_callback(self):
        """Test mode change callback mechanism."""
        router = StateManager()
        router.initialize()
        
        # Create mock callback
        callback = Mock()
        router.register_mode_change_callback(callback)
        
        # Change mode
        router.set_mode(ApplicationMode.AUDIO_CONTROL)
        
        # Verify callback was called with new mode
        callback.assert_called_once_with(ApplicationMode.AUDIO_CONTROL)
        
        router.cleanup()
    
    def test_multiple_callbacks(self):
        """Test multiple mode change callbacks."""
        router = StateManager()
        router.initialize()
        
        # Register multiple callbacks
        callback1 = Mock()
        callback2 = Mock()
        router.register_mode_change_callback(callback1)
        router.register_mode_change_callback(callback2)
        
        # Change mode
        router.set_mode(ApplicationMode.IMAGE_EDITING)
        
        # Verify both callbacks were called
        callback1.assert_called_once_with(ApplicationMode.IMAGE_EDITING)
        callback2.assert_called_once_with(ApplicationMode.IMAGE_EDITING)
        
        router.cleanup()
    
    def test_handler_registration(self):
        """Test gesture handler registration."""
        router = StateManager()
        router.initialize()
        
        # Create mock handler
        handler = Mock()
        
        # Register handler
        router.register_handler(ApplicationMode.NEUTRAL, "open_palm", handler)
        
        # Route gesture
        router.route_gesture("open_palm", {"hand_id": "left"})
        
        # Verify handler was called
        handler.assert_called_once()
        
        router.cleanup()
    
    def test_gesture_routing_by_mode(self):
        """Test that gestures are routed only to handlers for current mode."""
        router = StateManager()
        router.initialize()
        
        # Create handlers for different modes
        neutral_handler = Mock()
        audio_handler = Mock()
        
        router.register_handler(ApplicationMode.NEUTRAL, "fist", neutral_handler)
        router.register_handler(ApplicationMode.AUDIO_CONTROL, "fist", audio_handler)
        
        # In NEUTRAL mode, only neutral handler should be called
        router.set_mode(ApplicationMode.NEUTRAL)
        router.route_gesture("fist", {"hand_id": "right"})
        
        neutral_handler.assert_called_once()
        audio_handler.assert_not_called()
        
        # Switch to AUDIO mode
        neutral_handler.reset_mock()
        audio_handler.reset_mock()
        
        router.set_mode(ApplicationMode.AUDIO_CONTROL)
        router.route_gesture("fist", {"hand_id": "right"})
        
        neutral_handler.assert_not_called()
        audio_handler.assert_called_once()
        
        router.cleanup()
    
    def test_mode_switch_gesture_both_palms(self):
        """Test mode switching via both palms open gesture."""
        router = StateManager(mode_switch_duration=0.1)  # Short duration for testing
        router.initialize()
        
        # Initial mode should be NEUTRAL
        assert router.get_mode() == ApplicationMode.NEUTRAL
        
        # Simulate both palms open gesture (multiple times over threshold duration)
        gesture_data = {"hand_id": "both", "both_hands": True}
        
        # First call starts the timer
        router.route_gesture("open_palm", gesture_data)
        assert router.get_mode() == ApplicationMode.NEUTRAL  # Still in NEUTRAL
        
        # Wait for duration threshold
        time.sleep(0.15)
        
        # Second call should trigger mode switch
        router.route_gesture("open_palm", gesture_data)
        assert router.get_mode() == ApplicationMode.AUDIO_CONTROL  # Switched to AUDIO
        
        router.cleanup()
    
    def test_mode_cycle(self):
        """Test that modes cycle in correct order."""
        router = StateManager(mode_switch_duration=0.05)
        router.initialize()
        
        callback = Mock()
        router.register_mode_change_callback(callback)
        
        # Start in NEUTRAL
        assert router.get_mode() == ApplicationMode.NEUTRAL
        
        # Cycle to AUDIO_CONTROL
        gesture_data = {"hand_id": "both", "both_hands": True}
        router.route_gesture("open_palm", gesture_data)
        time.sleep(0.06)
        router.route_gesture("open_palm", gesture_data)
        assert router.get_mode() == ApplicationMode.AUDIO_CONTROL
        
        # Cycle to IMAGE_EDITING
        time.sleep(0.06)
        router.route_gesture("open_palm", gesture_data)
        time.sleep(0.06)
        router.route_gesture("open_palm", gesture_data)
        assert router.get_mode() == ApplicationMode.IMAGE_EDITING
        
        # Cycle back to NEUTRAL
        time.sleep(0.06)
        router.route_gesture("open_palm", gesture_data)
        time.sleep(0.06)
        router.route_gesture("open_palm", gesture_data)
        assert router.get_mode() == ApplicationMode.NEUTRAL
        
        router.cleanup()
    
    def test_mode_switch_timer_reset(self):
        """Test that mode switch timer resets if gesture is interrupted."""
        router = StateManager(mode_switch_duration=0.2)
        router.initialize()
        
        # Start both palms open
        gesture_data = {"hand_id": "both", "both_hands": True}
        router.route_gesture("open_palm", gesture_data)
        
        # Wait but not long enough
        time.sleep(0.1)
        
        # Interrupt with different gesture
        router.route_gesture("fist", {"hand_id": "left"})
        
        # Wait for original threshold
        time.sleep(0.15)
        
        # Try both palms again - should not trigger because timer was reset
        router.route_gesture("open_palm", gesture_data)
        assert router.get_mode() == ApplicationMode.NEUTRAL  # Still in NEUTRAL
        
        router.cleanup()
    
    def test_event_dispatch_and_processing(self):
        """Test event dispatch and processing."""
        router = StateManager()
        router.initialize()
        
        # Register handler
        handler = Mock()
        router.register_handler(ApplicationMode.NEUTRAL, "pinch", handler)
        
        # Dispatch event
        event = {
            "gesture_name": "pinch",
            "hand_id": "right",
            "confidence": 0.95
        }
        router.dispatch_event(event)
        
        # Process events
        router.process_events()
        
        # Verify handler was called
        handler.assert_called_once()
        
        router.cleanup()
    
    def test_no_handler_for_gesture(self):
        """Test that routing works gracefully when no handler is registered."""
        router = StateManager()
        router.initialize()
        
        # Route gesture without registered handler (should not crash)
        router.route_gesture("swipe_left", {"hand_id": "right"})
        
        router.cleanup()
    
    def test_cleanup(self):
        """Test cleanup clears all state."""
        router = StateManager()
        router.initialize()
        
        # Register handler and callback
        handler = Mock()
        callback = Mock()
        router.register_handler(ApplicationMode.NEUTRAL, "fist", handler)
        router.register_mode_change_callback(callback)
        
        # Dispatch some events
        router.dispatch_event({"gesture": "test"})
        
        # Cleanup
        router.cleanup()
        
        # Verify cleanup worked (no errors on re-cleanup)
        router.cleanup()
    
    def test_thread_safety_mode_get_set(self):
        """Test thread-safe mode get/set operations."""
        import threading
        
        router = StateManager()
        router.initialize()
        
        results = []
        
        def switch_mode(mode):
            for _ in range(10):
                router.set_mode(mode)
                current = router.get_mode()
                results.append(current)
        
        # Create multiple threads
        threads = [
            threading.Thread(target=switch_mode, args=(ApplicationMode.AUDIO_CONTROL,)),
            threading.Thread(target=switch_mode, args=(ApplicationMode.IMAGE_EDITING,)),
        ]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Verify no crashes and all results are valid modes
        assert len(results) == 20
        for result in results:
            assert isinstance(result, ApplicationMode)
        
        router.cleanup()


def test_application_mode_enum():
    """Test ApplicationMode enum values."""
    assert ApplicationMode.NEUTRAL.value == "neutral"
    assert ApplicationMode.AUDIO_CONTROL.value == "audio_control"
    assert ApplicationMode.IMAGE_EDITING.value == "image_editing"
    
    # Test that all modes are unique
    modes = [ApplicationMode.NEUTRAL, ApplicationMode.AUDIO_CONTROL, ApplicationMode.IMAGE_EDITING]
    assert len(modes) == len(set(modes))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
