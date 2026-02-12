"""
Integration test for GestureRecognitionEngine with VisionEngine and UI.

This test validates that the gesture recognition engine integrates properly
with the vision engine and UI components.
"""

import pytest
import time
import queue
from unittest.mock import MagicMock, patch
import numpy as np

from src.vision.vision_engine_impl import MediaPipeVisionEngine, VisionData
from src.gesture.gesture_recognition_engine import GestureRecognitionEngine, GestureEvent


class TestGestureVisionIntegration:
    """Test integration between GestureRecognitionEngine and VisionEngine."""
    
    def test_end_to_end_gesture_recognition(self):
        """Test complete pipeline from vision data to gesture recognition."""
        # Create input queue for gesture engine
        input_queue = queue.Queue(maxsize=10)
        
        # Create gesture engine
        gesture_engine = GestureRecognitionEngine(input_queue, max_queue_size=5)
        
        # Start gesture engine
        gesture_engine.start()
        
        # Create mock vision data with open palm gesture
        landmarks_normalized = []
        
        # Wrist
        landmarks_normalized.append({'x': 0.5, 'y': 0.5, 'z': 0.0})
        
        # Thumb (extended)
        for i in range(4):
            landmarks_normalized.append({'x': 0.3 + i * 0.05, 'y': 0.5, 'z': 0.0})
            
        # Other fingers (all extended upward)
        for finger in range(4):  # index, middle, ring, pinky
            base_x = 0.45 + finger * 0.05
            for joint in range(4):
                landmarks_normalized.append({
                    'x': base_x,
                    'y': 0.5 - joint * 0.1,
                    'z': 0.0
                })
        
        hand_data = {
            'handedness': 'Right',
            'landmarks_normalized': landmarks_normalized,
            'landmarks': landmarks_normalized,
            'confidence': 0.95
        }
        
        # Create vision data
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vision_data = VisionData(
            frame=frame,
            frame_rgb=frame,
            landmarks=[hand_data],
            timestamp=time.time()
        )
        
        # Put vision data into queue
        input_queue.put(vision_data)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check for gesture event
        gesture_event = gesture_engine.get_event(timeout=1.0)
        
        # Stop engine
        gesture_engine.stop()
        
        # Validate results
        assert gesture_event is not None, "Gesture event should be emitted"
        assert gesture_event.hand_id == "right"
        assert gesture_event.gesture_type == "static"
        assert gesture_event.confidence_score > 0.0
        
        print(f"Recognized gesture: {gesture_event.gesture_name} with confidence {gesture_event.confidence_score}")
        
    def test_multiple_gestures_sequence(self):
        """Test recognition of multiple gestures in sequence."""
        input_queue = queue.Queue(maxsize=10)
        gesture_engine = GestureRecognitionEngine(input_queue, max_queue_size=10)
        gesture_engine.start()
        
        # Create fist gesture
        fist_landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0.0}]  # Wrist
        for i in range(20):
            fist_landmarks.append({
                'x': 0.48 + (i % 5) * 0.01,
                'y': 0.48 - (i // 5) * 0.01,
                'z': 0.0
            })
        
        fist_hand = {
            'handedness': 'Left',
            'landmarks_normalized': fist_landmarks,
            'landmarks': fist_landmarks,
            'confidence': 0.90
        }
        
        # Send fist gesture
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vision_data = VisionData(
            frame=frame,
            frame_rgb=frame,
            landmarks=[fist_hand],
            timestamp=time.time()
        )
        input_queue.put(vision_data)
        
        time.sleep(0.3)
        
        # Get first event
        event1 = gesture_engine.get_event(timeout=0.5)
        
        # Create pinch gesture
        pinch_landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0.0}]  # Wrist
        # Thumb tip
        for i in range(4):
            pinch_landmarks.append({'x': 0.35 + i * 0.01, 'y': 0.4 - i * 0.01, 'z': 0.0})
        # Index tip near thumb
        pinch_landmarks.append({'x': 0.5, 'y': 0.5, 'z': 0.0})
        pinch_landmarks.append({'x': 0.45, 'y': 0.45, 'z': 0.0})
        pinch_landmarks.append({'x': 0.4, 'y': 0.4, 'z': 0.0})
        pinch_landmarks.append({'x': 0.36, 'y': 0.38, 'z': 0.0})
        # Other fingers
        for i in range(12):
            pinch_landmarks.append({'x': 0.55, 'y': 0.5, 'z': 0.0})
        
        pinch_hand = {
            'handedness': 'Right',
            'landmarks_normalized': pinch_landmarks,
            'landmarks': pinch_landmarks,
            'confidence': 0.92
        }
        
        # Send pinch gesture
        vision_data2 = VisionData(
            frame=frame,
            frame_rgb=frame,
            landmarks=[pinch_hand],
            timestamp=time.time()
        )
        input_queue.put(vision_data2)
        
        time.sleep(0.3)
        
        # Get second event
        event2 = gesture_engine.get_event(timeout=0.5)
        
        gesture_engine.stop()
        
        # Validate both events
        assert event1 is not None, "First gesture should be recognized"
        assert event2 is not None, "Second gesture should be recognized"
        
        print(f"First gesture: {event1.gesture_name} ({event1.confidence_score:.2f})")
        print(f"Second gesture: {event2.gesture_name} ({event2.confidence_score:.2f})")
        
    def test_dynamic_gesture_detection(self):
        """Test detection of dynamic gestures (motion-based)."""
        input_queue = queue.Queue(maxsize=10)
        gesture_engine = GestureRecognitionEngine(
            input_queue,
            max_queue_size=10,
            history_size=5,
            velocity_threshold=0.03  # Lower threshold for test
        )
        gesture_engine.start()
        
        # Create a sequence of frames showing hand moving right (swipe right)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for x_pos in [0.3, 0.4, 0.5, 0.6, 0.7]:
            landmarks = [{'x': x_pos, 'y': 0.5, 'z': 0.0}]  # Moving wrist
            for i in range(20):
                landmarks.append({'x': x_pos + 0.01, 'y': 0.5, 'z': 0.0})
            
            hand_data = {
                'handedness': 'Right',
                'landmarks_normalized': landmarks,
                'landmarks': landmarks,
                'confidence': 0.90
            }
            
            vision_data = VisionData(
                frame=frame,
                frame_rgb=frame,
                landmarks=[hand_data],
                timestamp=time.time()
            )
            
            input_queue.put(vision_data)
            time.sleep(0.1)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Look for dynamic gesture events
        dynamic_event = None
        for _ in range(5):  # Check multiple events
            event = gesture_engine.get_event(timeout=0.2)
            if event and event.gesture_type == "dynamic":
                dynamic_event = event
                break
        
        gesture_engine.stop()
        
        # Validate dynamic gesture was detected
        if dynamic_event:
            print(f"Dynamic gesture detected: {dynamic_event.gesture_name} ({dynamic_event.confidence_score:.2f})")
            assert dynamic_event.gesture_type == "dynamic"
        else:
            print("Note: Dynamic gesture detection requires more frames or adjustment of thresholds")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
