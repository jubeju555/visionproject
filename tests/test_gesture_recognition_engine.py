"""
Unit tests for Gesture Recognition Engine.

Tests the GestureRecognitionEngine class including:
- Static gesture recognition
- Dynamic gesture recognition
- Event emission
- Threading behavior
- Confidence scoring
"""

import pytest
import time
import queue
from unittest.mock import MagicMock
from dataclasses import asdict

from src.gesture.gesture_recognition_engine import (
    GestureRecognitionEngine,
    GestureEvent,
    StaticGesture,
    DynamicGesture,
)


class MockVisionData:
    """Mock vision data for testing."""
    
    def __init__(self, landmarks, timestamp=None):
        self.landmarks = landmarks
        self.timestamp = timestamp or time.time()


class TestGestureEvent:
    """Test GestureEvent dataclass."""
    
    def test_gesture_event_creation(self):
        """Test GestureEvent can be created."""
        event = GestureEvent(
            gesture_name="open_palm",
            confidence_score=0.9,
            timestamp=123.456,
            hand_id="right",
            gesture_type="static"
        )
        
        assert event.gesture_name == "open_palm"
        assert event.confidence_score == 0.9
        assert event.timestamp == 123.456
        assert event.hand_id == "right"
        assert event.gesture_type == "static"
        

class TestGestureRecognitionEngine:
    """Test GestureRecognitionEngine class."""
    
    def test_initialization(self):
        """Test engine can be initialized."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue)
        
        assert engine.input_queue == input_queue
        assert not engine.is_running()
        
    def test_start_stop(self):
        """Test starting and stopping the engine."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue)
        
        engine.start()
        assert engine.is_running()
        
        time.sleep(0.1)  # Give thread time to start
        
        engine.stop()
        assert not engine.is_running()
        
    def test_open_palm_recognition(self):
        """Test recognition of open palm gesture."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue)
        
        # Create mock landmarks for open palm (all fingers extended)
        landmarks_normalized = []
        
        # Wrist
        landmarks_normalized.append({'x': 0.5, 'y': 0.5, 'z': 0.0})
        
        # Thumb (extended away from palm)
        for i in range(4):
            landmarks_normalized.append({'x': 0.3 + i * 0.05, 'y': 0.5, 'z': 0.0})
            
        # Index finger (extended upward)
        landmarks_normalized.append({'x': 0.45, 'y': 0.5, 'z': 0.0})  # MCP
        landmarks_normalized.append({'x': 0.45, 'y': 0.4, 'z': 0.0})  # PIP
        landmarks_normalized.append({'x': 0.45, 'y': 0.3, 'z': 0.0})  # DIP
        landmarks_normalized.append({'x': 0.45, 'y': 0.2, 'z': 0.0})  # TIP
        
        # Middle finger (extended upward)
        landmarks_normalized.append({'x': 0.5, 'y': 0.5, 'z': 0.0})
        landmarks_normalized.append({'x': 0.5, 'y': 0.4, 'z': 0.0})
        landmarks_normalized.append({'x': 0.5, 'y': 0.3, 'z': 0.0})
        landmarks_normalized.append({'x': 0.5, 'y': 0.15, 'z': 0.0})
        
        # Ring finger (extended upward)
        landmarks_normalized.append({'x': 0.55, 'y': 0.5, 'z': 0.0})
        landmarks_normalized.append({'x': 0.55, 'y': 0.4, 'z': 0.0})
        landmarks_normalized.append({'x': 0.55, 'y': 0.3, 'z': 0.0})
        landmarks_normalized.append({'x': 0.55, 'y': 0.2, 'z': 0.0})
        
        # Pinky (extended upward)
        landmarks_normalized.append({'x': 0.6, 'y': 0.5, 'z': 0.0})
        landmarks_normalized.append({'x': 0.6, 'y': 0.4, 'z': 0.0})
        landmarks_normalized.append({'x': 0.6, 'y': 0.3, 'z': 0.0})
        landmarks_normalized.append({'x': 0.6, 'y': 0.25, 'z': 0.0})
        
        hand_data = {
            'handedness': 'Right',
            'landmarks_normalized': landmarks_normalized,
            'confidence': 0.95
        }
        
        gesture, confidence = engine._classify_static_gesture(hand_data)
        
        assert gesture == StaticGesture.OPEN_PALM
        assert confidence > 0.7
        
    def test_fist_recognition(self):
        """Test recognition of fist gesture."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue)
        
        # Create mock landmarks for fist (all fingers curled close to wrist)
        landmarks_normalized = []
        
        # Wrist
        landmarks_normalized.append({'x': 0.5, 'y': 0.5, 'z': 0.0})
        
        # Thumb (curled but not touching index)
        landmarks_normalized.append({'x': 0.45, 'y': 0.48, 'z': 0.0})
        landmarks_normalized.append({'x': 0.42, 'y': 0.46, 'z': 0.0})
        landmarks_normalized.append({'x': 0.40, 'y': 0.45, 'z': 0.0})
        landmarks_normalized.append({'x': 0.38, 'y': 0.44, 'z': 0.0})  # Thumb tip
        
        # Index finger (curled)
        landmarks_normalized.append({'x': 0.48, 'y': 0.48, 'z': 0.0})
        landmarks_normalized.append({'x': 0.47, 'y': 0.46, 'z': 0.0})
        landmarks_normalized.append({'x': 0.46, 'y': 0.45, 'z': 0.0})
        landmarks_normalized.append({'x': 0.45, 'y': 0.44, 'z': 0.0})  # Index tip (away from thumb)
        
        # Other fingers (all curled close to wrist)
        for i in range(12):
            landmarks_normalized.append({
                'x': 0.5 + (i % 4) * 0.02,
                'y': 0.5 - (i // 4) * 0.02,
                'z': 0.0
            })
            
        hand_data = {
            'handedness': 'Right',
            'landmarks_normalized': landmarks_normalized,
            'confidence': 0.95
        }
        
        gesture, confidence = engine._classify_static_gesture(hand_data)
        
        assert gesture == StaticGesture.FIST
        assert confidence > 0.7
        
    def test_pinch_recognition(self):
        """Test recognition of pinch gesture."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue)
        
        # Create mock landmarks for pinch (thumb and index finger touching)
        landmarks_normalized = []
        
        # Wrist
        landmarks_normalized.append({'x': 0.5, 'y': 0.5, 'z': 0.0})
        
        # Thumb leading to tip near index
        landmarks_normalized.append({'x': 0.45, 'y': 0.5, 'z': 0.0})
        landmarks_normalized.append({'x': 0.4, 'y': 0.45, 'z': 0.0})
        landmarks_normalized.append({'x': 0.35, 'y': 0.4, 'z': 0.0})
        landmarks_normalized.append({'x': 0.3, 'y': 0.35, 'z': 0.0})  # Thumb tip
        
        # Index finger leading to tip near thumb
        landmarks_normalized.append({'x': 0.5, 'y': 0.5, 'z': 0.0})
        landmarks_normalized.append({'x': 0.45, 'y': 0.45, 'z': 0.0})
        landmarks_normalized.append({'x': 0.4, 'y': 0.4, 'z': 0.0})
        landmarks_normalized.append({'x': 0.31, 'y': 0.36, 'z': 0.0})  # Index tip (close to thumb)
        
        # Other fingers (curled)
        for i in range(12):
            landmarks_normalized.append({'x': 0.55, 'y': 0.5, 'z': 0.0})
            
        hand_data = {
            'handedness': 'Right',
            'landmarks_normalized': landmarks_normalized,
            'confidence': 0.95
        }
        
        gesture, confidence = engine._classify_static_gesture(hand_data)
        
        assert gesture == StaticGesture.PINCH
        assert confidence > 0.5
        
    def test_two_fingers_recognition(self):
        """Test recognition of two fingers gesture."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue)
        
        # Create mock landmarks with index and middle fingers extended
        landmarks_normalized = []
        
        # Wrist
        landmarks_normalized.append({'x': 0.5, 'y': 0.5, 'z': 0.0})
        
        # Thumb (curled)
        for i in range(4):
            landmarks_normalized.append({'x': 0.45, 'y': 0.5, 'z': 0.0})
            
        # Index finger (extended)
        landmarks_normalized.append({'x': 0.48, 'y': 0.5, 'z': 0.0})
        landmarks_normalized.append({'x': 0.48, 'y': 0.4, 'z': 0.0})
        landmarks_normalized.append({'x': 0.48, 'y': 0.3, 'z': 0.0})
        landmarks_normalized.append({'x': 0.48, 'y': 0.2, 'z': 0.0})
        
        # Middle finger (extended)
        landmarks_normalized.append({'x': 0.52, 'y': 0.5, 'z': 0.0})
        landmarks_normalized.append({'x': 0.52, 'y': 0.4, 'z': 0.0})
        landmarks_normalized.append({'x': 0.52, 'y': 0.3, 'z': 0.0})
        landmarks_normalized.append({'x': 0.52, 'y': 0.2, 'z': 0.0})
        
        # Ring and pinky (curled)
        for i in range(8):
            landmarks_normalized.append({'x': 0.55, 'y': 0.5, 'z': 0.0})
            
        hand_data = {
            'handedness': 'Right',
            'landmarks_normalized': landmarks_normalized,
            'confidence': 0.95
        }
        
        gesture, confidence = engine._classify_static_gesture(hand_data)
        
        assert gesture == StaticGesture.TWO_FINGERS
        
    def test_calculate_distance(self):
        """Test distance calculation utility."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue)
        
        point1 = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        point2 = {'x': 3.0, 'y': 4.0, 'z': 0.0}
        
        distance = engine._calculate_distance(point1, point2)
        
        assert abs(distance - 5.0) < 0.001
        
    def test_calculate_angle(self):
        """Test angle calculation utility."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue)
        
        # Right angle test
        point1 = {'x': 1.0, 'y': 0.0}
        point2 = {'x': 0.0, 'y': 0.0}
        point3 = {'x': 0.0, 'y': 1.0}
        
        angle = engine._calculate_angle(point1, point2, point3)
        
        assert abs(angle - 90.0) < 0.1
        
    def test_event_emission(self):
        """Test that events are properly emitted."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue, max_queue_size=5)
        
        # Create a test event
        event = GestureEvent(
            gesture_name="test_gesture",
            confidence_score=0.8,
            timestamp=time.time(),
            hand_id="right",
            gesture_type="static"
        )
        
        # Emit event
        engine._emit_event(event)
        
        # Check it's in the queue
        retrieved_event = engine.get_event(timeout=0.1)
        
        assert retrieved_event is not None
        assert retrieved_event.gesture_name == "test_gesture"
        assert retrieved_event.confidence_score == 0.8
        
    def test_latest_gesture_tracking(self):
        """Test that latest gesture is tracked properly."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue)
        
        # Initially no gesture
        assert engine.get_latest_gesture() is None
        
        # Emit an event
        event = GestureEvent(
            gesture_name="test_gesture",
            confidence_score=0.8,
            timestamp=time.time(),
            hand_id="right",
            gesture_type="static"
        )
        
        engine._emit_event(event)
        
        # Check latest gesture is updated
        latest = engine.get_latest_gesture()
        assert latest is not None
        assert latest.gesture_name == "test_gesture"
        
    def test_velocity_calculation(self):
        """Test velocity calculation from history."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue)
        
        # Create history with movement
        hand_data_1 = {
            'handedness': 'Right',
            'landmarks_normalized': [
                {'x': 0.4, 'y': 0.5, 'z': 0.0}  # Wrist
            ] + [{'x': 0.4, 'y': 0.5, 'z': 0.0}] * 20,
            'confidence': 0.95
        }
        
        hand_data_2 = {
            'handedness': 'Right',
            'landmarks_normalized': [
                {'x': 0.5, 'y': 0.5, 'z': 0.0}  # Wrist moved right
            ] + [{'x': 0.5, 'y': 0.5, 'z': 0.0}] * 20,
            'confidence': 0.95
        }
        
        engine._update_history([hand_data_1])
        engine._update_history([hand_data_2])
        
        velocity = engine._calculate_velocity()
        
        assert velocity is not None
        vx, vy = velocity
        assert vx > 0  # Moved right
        assert abs(vy) < 0.01  # Minimal vertical movement
        
    def test_integration_with_vision_data(self):
        """Test integration with VisionEngine data."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue, max_queue_size=5)
        
        # Start engine
        engine.start()
        
        # Create open palm landmarks
        landmarks_normalized = [{'x': 0.5, 'y': 0.5 - i * 0.05, 'z': 0.0} for i in range(21)]
        hand_data = {
            'handedness': 'Right',
            'landmarks_normalized': landmarks_normalized,
            'landmarks': landmarks_normalized,
            'confidence': 0.95
        }
        
        # Put vision data into input queue
        vision_data = MockVisionData([hand_data])
        input_queue.put(vision_data)
        
        # Wait for processing
        time.sleep(0.3)
        
        # Check for events
        event = engine.get_event(timeout=0.5)
        
        engine.stop()
        
        # Should have detected something
        assert event is not None
        assert event.hand_id == "right"
        
    def test_history_size_limit(self):
        """Test that history is limited to specified size."""
        input_queue = queue.Queue()
        engine = GestureRecognitionEngine(input_queue, history_size=5)
        
        # Add more frames than history size
        for i in range(10):
            hand_data = {
                'handedness': 'Right',
                'landmarks_normalized': [{'x': 0.5, 'y': 0.5, 'z': 0.0}] * 21,
                'confidence': 0.95
            }
            engine._update_history([hand_data])
            
        # History should be limited
        assert len(engine._landmark_history) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
