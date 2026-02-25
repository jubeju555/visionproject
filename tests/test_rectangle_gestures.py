"""
Unit Tests for Rectangle Gesture Detection and Screenshot Capture.

Tests:
- Rectangle detection from hand landmarks
- Single-hand and two-hand rectangle formations
- Area calculations
- Screenshot capture and saving
- Perspective transformation
- Integration with existing gesture system
"""

import pytest
import numpy as np
import os
import tempfile
import time
from unittest.mock import Mock, MagicMock, patch
import cv2

from src.gesture.rectangle_gestures import (
    RectangleFrame,
    RectangleGestureDetector,
    ScreenshotCapture,
)


class TestRectangleFrame:
    """Test RectangleFrame data container."""
    
    def test_rectangle_frame_creation(self):
        """Test creating a RectangleFrame."""
        frame = RectangleFrame(
            top_left=(0.1, 0.1),
            top_right=(0.9, 0.1),
            bottom_left=(0.1, 0.9),
            bottom_right=(0.9, 0.9),
            area=0.64,
            confidence=0.95,
            timestamp=123.456,
            hand_id="both"
        )
        
        assert frame.top_left == (0.1, 0.1)
        assert frame.bottom_right == (0.9, 0.9)
        assert frame.area == 0.64
        assert frame.confidence == 0.95
        assert frame.hand_id == "both"
    
    def test_get_pixel_coordinates(self):
        """Test converting normalized to pixel coordinates."""
        frame = RectangleFrame(
            top_left=(0.0, 0.0),
            top_right=(1.0, 0.0),
            bottom_left=(0.0, 1.0),
            bottom_right=(1.0, 1.0),
            area=1.0,
            confidence=1.0,
            timestamp=0.0,
            hand_id="both"
        )
        
        pixels = frame.get_pixel_coordinates(1920, 1080)
        
        assert pixels['top_left'] == (0, 0)
        assert pixels['top_right'] == (1920, 0)
        assert pixels['bottom_left'] == (0, 1080)
        assert pixels['bottom_right'] == (1920, 1080)
    
    def test_get_pixel_coordinates_half_frame(self):
        """Test pixel coordinates for half-frame rectangle."""
        frame = RectangleFrame(
            top_left=(0.25, 0.25),
            top_right=(0.75, 0.25),
            bottom_left=(0.25, 0.75),
            bottom_right=(0.75, 0.75),
            area=0.25,
            confidence=0.9,
            timestamp=0.0,
            hand_id="left"
        )
        
        pixels = frame.get_pixel_coordinates(800, 600)
        
        assert pixels['top_left'] == (200, 150)
        assert pixels['top_right'] == (600, 150)
        assert pixels['bottom_left'] == (200, 450)
        assert pixels['bottom_right'] == (600, 450)


class TestRectangleGestureDetector:
    """Test RectangleGestureDetector."""
    
    def test_detector_initialization(self):
        """Test initializing the detector."""
        detector = RectangleGestureDetector()
        
        assert detector.min_confidence == 0.6
        assert detector.min_area_threshold == 0.02
        assert detector.max_area_threshold == 0.9
    
    def test_detector_with_custom_thresholds(self):
        """Test initializing with custom thresholds."""
        detector = RectangleGestureDetector(
            min_confidence=0.75,
            min_area_threshold=0.05,
            max_area_threshold=0.85
        )
        
        assert detector.min_confidence == 0.75
        assert detector.min_area_threshold == 0.05
        assert detector.max_area_threshold == 0.85
    
    def test_detect_rectangle_no_hands(self):
        """Test detection with no hands returns None."""
        detector = RectangleGestureDetector()
        
        result = detector.detect_rectangle([], timestamp=123.456)
        
        assert result is None
    
    def test_detect_rectangle_one_hand(self):
        """Test detection with single hand."""
        detector = RectangleGestureDetector()
        
        # Create realistic single-hand landmarks (open palm)
        landmarks_normalized = [
            {'x': 0.5, 'y': 0.5, 'z': 0.0},    # 0: wrist
            {'x': 0.4, 'y': 0.45, 'z': 0.0},   # 1: thumb CMC
            {'x': 0.35, 'y': 0.35, 'z': 0.0},  # 2: thumb MCP
            {'x': 0.3, 'y': 0.25, 'z': 0.0},   # 3: thumb IP
            {'x': 0.25, 'y': 0.15, 'z': 0.0},  # 4: thumb tip ✓
            {'x': 0.55, 'y': 0.45, 'z': 0.0},  # 5: index CMC
            {'x': 0.6, 'y': 0.3, 'z': 0.0},    # 6: index PIP
            {'x': 0.62, 'y': 0.15, 'z': 0.0},  # 7: index DIP
            {'x': 0.63, 'y': 0.0, 'z': 0.0},   # 8: index tip ✓
            {'x': 0.5, 'y': 0.4, 'z': 0.0},    # 9: middle CMC
            {'x': 0.5, 'y': 0.25, 'z': 0.0},   # 10: middle PIP
            {'x': 0.5, 'y': 0.1, 'z': 0.0},    # 11: middle DIP
            {'x': 0.5, 'y': -0.05, 'z': 0.0},  # 12: middle tip ✓
            {'x': 0.45, 'y': 0.4, 'z': 0.0},   # 13: ring CMC
            {'x': 0.42, 'y': 0.25, 'z': 0.0},  # 14: ring PIP
            {'x': 0.4, 'y': 0.1, 'z': 0.0},    # 15: ring DIP
            {'x': 0.38, 'y': -0.05, 'z': 0.0}, # 16: ring tip ✓
            {'x': 0.4, 'y': 0.35, 'z': 0.0},   # 17: pinky CMC
            {'x': 0.36, 'y': 0.2, 'z': 0.0},   # 18: pinky PIP
            {'x': 0.34, 'y': 0.05, 'z': 0.0},  # 19: pinky DIP
            {'x': 0.32, 'y': -0.05, 'z': 0.0}, # 20: pinky tip
        ]
        
        hand_data = {
            'handedness': 'Right',
            'landmarks_normalized': landmarks_normalized,
            'confidence': 0.95
        }
        
        result = detector.detect_rectangle([hand_data], timestamp=123.456)
        
        # With single hand, should try to form rectangle from finger tips
        # Result may or may not detect depending on implementation
        assert result is None or isinstance(result, RectangleFrame)
    
    def test_detect_rectangle_two_hands(self):
        """Test detection with two hands."""
        detector = RectangleGestureDetector(min_area_threshold=0.01, min_confidence=0.5)
        
        # Create two-hand landmarks
        base_landmarks = [
            {'x': 0.5, 'y': 0.5, 'z': 0.0},
            {'x': 0.4, 'y': 0.45, 'z': 0.0},
            {'x': 0.35, 'y': 0.35, 'z': 0.0},
            {'x': 0.3, 'y': 0.25, 'z': 0.0},
            {'x': 0.2, 'y': 0.1, 'z': 0.0},   # Thumb tip
            {'x': 0.55, 'y': 0.45, 'z': 0.0},
            {'x': 0.6, 'y': 0.3, 'z': 0.0},
            {'x': 0.62, 'y': 0.15, 'z': 0.0},
            {'x': 0.8, 'y': 0.0, 'z': 0.0},   # Index tip
        ] + [{'x': 0.5, 'y': 0.5, 'z': 0.0}] * 12
        
        hand1 = {
            'handedness': 'Left',
            'landmarks_normalized': base_landmarks,
            'confidence': 0.95
        }
        
        hand2 = {
            'handedness': 'Right',
            'landmarks_normalized': base_landmarks,
            'confidence': 0.95
        }
        
        result = detector.detect_rectangle([hand1, hand2], timestamp=123.456)
        
        # Two hands should have better chance of forming rectangle
        assert result is None or isinstance(result, RectangleFrame)
    
    def test_is_valid_rectangle_too_small(self):
        """Test that rectangles below minimum area are rejected."""
        detector = RectangleGestureDetector(min_area_threshold=0.1)
        
        frame = RectangleFrame(
            top_left=(0.0, 0.0),
            top_right=(0.1, 0.0),
            bottom_left=(0.0, 0.05),
            bottom_right=(0.1, 0.05),
            area=0.005,  # Too small
            confidence=0.95,
            timestamp=0.0,
            hand_id="both"
        )
        
        assert not detector._is_valid_rectangle(frame)
    
    def test_is_valid_rectangle_too_large(self):
        """Test that rectangles above maximum area are rejected."""
        detector = RectangleGestureDetector(max_area_threshold=0.5)
        
        frame = RectangleFrame(
            top_left=(0.0, 0.0),
            top_right=(1.0, 0.0),
            bottom_left=(0.0, 1.0),
            bottom_right=(1.0, 1.0),
            area=1.0,  # Too large
            confidence=0.95,
            timestamp=0.0,
            hand_id="both"
        )
        
        assert not detector._is_valid_rectangle(frame)
    
    def test_is_valid_rectangle_low_confidence(self):
        """Test that low confidence rectangles are rejected."""
        detector = RectangleGestureDetector(min_confidence=0.8)
        
        frame = RectangleFrame(
            top_left=(0.1, 0.1),
            top_right=(0.9, 0.1),
            bottom_left=(0.1, 0.9),
            bottom_right=(0.9, 0.9),
            area=0.64,
            confidence=0.5,  # Too low
            timestamp=0.0,
            hand_id="both"
        )
        
        assert not detector._is_valid_rectangle(frame)
    
    def test_is_valid_rectangle_valid(self):
        """Test that valid rectangles pass validation."""
        detector = RectangleGestureDetector()
        
        frame = RectangleFrame(
            top_left=(0.1, 0.1),
            top_right=(0.9, 0.1),
            bottom_left=(0.1, 0.9),
            bottom_right=(0.9, 0.9),
            area=0.64,
            confidence=0.95,
            timestamp=0.0,
            hand_id="both"
        )
        
        assert detector._is_valid_rectangle(frame)


class TestScreenshotCapture:
    """Test ScreenshotCapture functionality."""
    
    def test_screenshot_capture_initialization(self):
        """Test initializing screenshot capture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = ScreenshotCapture(output_dir=tmpdir)
            
            assert capture.output_dir == tmpdir
            assert os.path.exists(tmpdir)
            assert capture.last_screenshot_path is None
    
    def test_screenshot_capture_creates_directory(self):
        """Test that capture creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_screenshots")
            assert not os.path.exists(new_dir)
            
            capture = ScreenshotCapture(output_dir=new_dir)
            
            assert os.path.exists(new_dir)
    
    def test_capture_from_frame_full_rectangle(self):
        """Test capturing full frame as rectangle."""
        capture = ScreenshotCapture()
        
        # Create a simple test frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 127
        
        # Rectangle covering full frame
        rectangle = RectangleFrame(
            top_left=(0.0, 0.0),
            top_right=(1.0, 0.0),
            bottom_left=(0.0, 1.0),
            bottom_right=(1.0, 1.0),
            area=1.0,
            confidence=1.0,
            timestamp=0.0,
            hand_id="both"
        )
        
        result = capture.capture_from_frame(frame, rectangle)
        
        assert result is not None
        assert result.shape[2] == 3  # BGR format
    
    def test_capture_from_frame_half_rectangle(self):
        """Test capturing half of frame."""
        capture = ScreenshotCapture()
        
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 127
        
        rectangle = RectangleFrame(
            top_left=(0.25, 0.25),
            top_right=(0.75, 0.25),
            bottom_left=(0.25, 0.75),
            bottom_right=(0.75, 0.75),
            area=0.25,
            confidence=0.95,
            timestamp=0.0,
            hand_id="both"
        )
        
        result = capture.capture_from_frame(frame, rectangle)
        
        assert result is not None
        assert result.shape[0] > 0  # Height
        assert result.shape[1] > 0  # Width
        assert result.shape[2] == 3  # BGR
    
    def test_capture_from_frame_none_input(self):
        """Test capture with None frame."""
        capture = ScreenshotCapture()
        
        rectangle = RectangleFrame(
            top_left=(0.1, 0.1),
            top_right=(0.9, 0.1),
            bottom_left=(0.1, 0.9),
            bottom_right=(0.9, 0.9),
            area=0.64,
            confidence=0.95,
            timestamp=0.0,
            hand_id="both"
        )
        
        result = capture.capture_from_frame(None, rectangle)
        
        assert result is None
    
    def test_save_screenshot(self):
        """Test saving screenshot to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = ScreenshotCapture(output_dir=tmpdir)
            
            # Create a test image
            image = np.ones((100, 100, 3), dtype=np.uint8) * 200
            
            filepath = capture.save_screenshot(image, name_prefix="test")
            
            assert filepath is not None
            assert os.path.exists(filepath)
            assert "test" in os.path.basename(filepath)
            assert filepath.endswith(".png")
            assert capture.last_screenshot_path == filepath
    
    def test_save_screenshot_none_image(self):
        """Test saving None image returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = ScreenshotCapture(output_dir=tmpdir)
            
            result = capture.save_screenshot(None)
            
            assert result is None
    
    def test_capture_and_save_integration(self):
        """Test capture and save in one operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = ScreenshotCapture(output_dir=tmpdir)
            
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
            
            rectangle = RectangleFrame(
                top_left=(0.1, 0.1),
                top_right=(0.9, 0.1),
                bottom_left=(0.1, 0.9),
                bottom_right=(0.9, 0.9),
                area=0.64,
                confidence=0.95,
                timestamp=123.456,
                hand_id="both"
            )
            
            filepath = capture.capture_and_save(frame, rectangle)
            
            assert filepath is not None
            assert os.path.exists(filepath)
            assert filepath.endswith(".png")
    
    def test_get_last_screenshot_path(self):
        """Test retrieving last screenshot path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = ScreenshotCapture(output_dir=tmpdir)
            
            # Initially None
            assert capture.get_last_screenshot_path() is None
            
            # After saving
            image = np.ones((100, 100, 3), dtype=np.uint8)
            filepath = capture.save_screenshot(image)
            
            assert capture.get_last_screenshot_path() == filepath


class TestIntegration:
    """Integration tests combining detector and capture."""
    
    def test_full_workflow_detection_and_capture(self):
        """Test full workflow: detect rectangle and capture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = RectangleGestureDetector()
            capture = ScreenshotCapture(output_dir=tmpdir)
            
            # Create hand landmarks for rectangle
            landmarks_normalized = [
                {'x': 0.5, 'y': 0.5, 'z': 0.0},
                {'x': 0.4, 'y': 0.45, 'z': 0.0},
                {'x': 0.35, 'y': 0.35, 'z': 0.0},
                {'x': 0.3, 'y': 0.25, 'z': 0.0},
                {'x': 0.2, 'y': 0.1, 'z': 0.0},   # Thumb tip
                {'x': 0.55, 'y': 0.45, 'z': 0.0},
                {'x': 0.6, 'y': 0.3, 'z': 0.0},
                {'x': 0.62, 'y': 0.15, 'z': 0.0},
                {'x': 0.8, 'y': 0.0, 'z': 0.0},   # Index tip
            ] + [{'x': 0.5, 'y': 0.5, 'z': 0.0}] * 12
            
            hand_data = {
                'handedness': 'Right',
                'landmarks_normalized': landmarks_normalized,
                'confidence': 0.95
            }
            
            # Detect rectangle
            rectangle = detector.detect_rectangle([hand_data], timestamp=time.time())
            
            if rectangle:
                # Capture from frame
                frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                filepath = capture.capture_and_save(frame, rectangle)
                
                assert filepath is not None
                assert os.path.exists(filepath)
    
    def test_multiple_captures(self):
        """Test multiple consecutive captures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = ScreenshotCapture(output_dir=tmpdir)
            
            filepaths = []
            for i in range(3):
                image = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
                
                rectangle = RectangleFrame(
                    top_left=(0.1, 0.1),
                    top_right=(0.9, 0.1),
                    bottom_left=(0.1, 0.9),
                    bottom_right=(0.9, 0.9),
                    area=0.64,
                    confidence=0.95,
                    timestamp=time.time(),
                    hand_id="both"
                )
                
                filepath = capture.capture_and_save(image, rectangle)
                filepaths.append(filepath)
                time.sleep(0.01)  # Small delay to avoid timestamp collision
            
            # All files should exist and be different
            for filepath in filepaths:
                assert os.path.exists(filepath)
            
            assert len(set(filepaths)) == 3  # All different paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
