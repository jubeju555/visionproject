"""
Rectangle Gesture Recognition and Screenshot Capture Module.

Detects when user forms a rectangle/square with thumbs and index fingers,
tracks the area, and captures screenshots when snap gesture is made.

Features:
- Real-time rectangle frame detection from hand landmarks
- Area extraction from camera frame
- Screenshot saving to disk
- Integration with image editor
"""

import logging
import math
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class RectangleFrame:
    """Data container for rectangle frame information."""
    top_left: Tuple[float, float]      # (x, y) normalized coordinates
    top_right: Tuple[float, float]
    bottom_left: Tuple[float, float]
    bottom_right: Tuple[float, float]
    area: float                         # Normalized area (0-1)
    confidence: float                   # Detection confidence (0-1)
    timestamp: float                    # Frame timestamp
    hand_id: str                        # "left", "right", or "both"
    
    def get_pixel_coordinates(self, frame_width: int, frame_height: int) -> Dict[str, Tuple[int, int]]:
        """
        Convert normalized coordinates to pixel coordinates.
        
        Args:
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels
            
        Returns:
            Dictionary with pixel coordinates for each corner
        """
        return {
            'top_left': (int(self.top_left[0] * frame_width), int(self.top_left[1] * frame_height)),
            'top_right': (int(self.top_right[0] * frame_width), int(self.top_right[1] * frame_height)),
            'bottom_left': (int(self.bottom_left[0] * frame_width), int(self.bottom_left[1] * frame_height)),
            'bottom_right': (int(self.bottom_right[0] * frame_width), int(self.bottom_right[1] * frame_height)),
        }


class RectangleGestureDetector:
    """
    Detects rectangle frames formed by hand landmarks.
    
    Algorithm:
    1. Find thumb tip and index finger tip positions for both hands
    2. Check if they form a rectangle-like shape
    3. Calculate homography/perspective to extract the area
    4. Return rectangle frame data for visualization and capture
    """
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        min_area_threshold: float = 0.02,  # Minimum 2% of frame area
        max_area_threshold: float = 0.9,   # Maximum 90% of frame area
    ):
        """
        Initialize rectangle gesture detector.
        
        Args:
            min_confidence: Minimum confidence for rectangle detection
            min_area_threshold: Minimum area threshold (normalized)
            max_area_threshold: Maximum area threshold (normalized)
        """
        self.min_confidence = min_confidence
        self.min_area_threshold = min_area_threshold
        self.max_area_threshold = max_area_threshold
        self._prev_rectangle: Optional[RectangleFrame] = None
    
    def detect_rectangle(
        self,
        landmarks: List[Dict[str, Any]],
        timestamp: float
    ) -> Optional[RectangleFrame]:
        """
        Detect rectangle frame from hand landmarks.
        
        Args:
            landmarks: List of hand landmark dictionaries containing positions
            timestamp: Frame timestamp
            
        Returns:
            RectangleFrame if rectangle detected, None otherwise
        """
        # Need at least 2 hands to form a rectangle
        if len(landmarks) < 2:
            self._prev_rectangle = None
            return None
        
        # Try both single-hand and two-hand rectangle formations
        rectangle = self._detect_two_hand_rectangle(landmarks, timestamp)
        if rectangle:
            self._prev_rectangle = rectangle
            return rectangle
        
        # Single hand rectangle (thumb and index fingers)
        rectangle = self._detect_single_hand_rectangle(landmarks[0], landmarks[0].get('handedness', 'Unknown'), timestamp)
        if rectangle:
            self._prev_rectangle = rectangle
            return rectangle
        
        self._prev_rectangle = None
        return None
    
    def _detect_two_hand_rectangle(
        self,
        landmarks: List[Dict[str, Any]],
        timestamp: float
    ) -> Optional[RectangleFrame]:
        """
        Detect rectangle formed by two hands (each hand at one corner).
        
        Args:
            landmarks: List of hand landmarks
            timestamp: Frame timestamp
            
        Returns:
            RectangleFrame if detected, None otherwise
        """
        if len(landmarks) < 2:
            return None
        
        # Get thumb and index tips from both hands
        hand1 = landmarks[0]
        hand2 = landmarks[1]
        
        # Extract key points: thumb tip (4) and index tip (8)
        h1_landmarks_normalized = hand1.get('landmarks_normalized', [])
        h2_landmarks_normalized = hand2.get('landmarks_normalized', [])
        
        if len(h1_landmarks_normalized) < 9 or len(h2_landmarks_normalized) < 9:
            return None
        
        h1_thumb_tip = h1_landmarks_normalized[4]
        h1_index_tip = h1_landmarks_normalized[8]
        h2_thumb_tip = h2_landmarks_normalized[4]
        h2_index_tip = h2_landmarks_normalized[8]
        
        # Check if points form a rough rectangle
        corners = [
            (h1_thumb_tip['x'], h1_thumb_tip['y']),
            (h1_index_tip['x'], h1_index_tip['y']),
            (h2_thumb_tip['x'], h2_thumb_tip['y']),
            (h2_index_tip['x'], h2_index_tip['y']),
        ]
        
        # Try to fit a rectangle to these points
        rectangle = self._fit_rectangle_to_points(corners, timestamp, "both")
        
        if rectangle and self._is_valid_rectangle(rectangle):
            return rectangle
        
        return None
    
    def _detect_single_hand_rectangle(
        self,
        hand_data: Dict[str, Any],
        hand_id: str,
        timestamp: float
    ) -> Optional[RectangleFrame]:
        """
        Detect rectangle from single hand (extended fingers forming corners).
        
        Args:
            hand_data: Single hand landmark dictionary
            hand_id: Hand identifier ("left" or "right")
            timestamp: Frame timestamp
            
        Returns:
            RectangleFrame if detected, None otherwise
        """
        landmarks_normalized = hand_data.get('landmarks_normalized', [])
        if len(landmarks_normalized) < 21:
            return None
        
        # Use thumb, index, middle, ring fingers for rectangle corners
        # Thumb (4), Index (8), Middle (12), Ring (16)
        try:
            corners = [
                (landmarks_normalized[4]['x'], landmarks_normalized[4]['y']),    # Thumb
                (landmarks_normalized[8]['x'], landmarks_normalized[8]['y']),    # Index
                (landmarks_normalized[12]['x'], landmarks_normalized[12]['y']),  # Middle
                (landmarks_normalized[16]['x'], landmarks_normalized[16]['y']),  # Ring
            ]
            
            rectangle = self._fit_rectangle_to_points(corners, timestamp, hand_id)
            
            if rectangle and self._is_valid_rectangle(rectangle):
                return rectangle
        except (KeyError, IndexError):
            pass
        
        return None
    
    def _fit_rectangle_to_points(
        self,
        points: List[Tuple[float, float]],
        timestamp: float,
        hand_id: str
    ) -> Optional[RectangleFrame]:
        """
        Fit a rectangle to given points using PCA.
        
        Args:
            points: List of (x, y) tuples
            timestamp: Frame timestamp
            hand_id: Hand identifier
            
        Returns:
            RectangleFrame if fitting successful, None otherwise
        """
        if len(points) < 3:
            return None
        
        try:
            points_array = np.array(points, dtype=np.float32)
            
            # Use minimum area rectangle
            center, (width, height), angle = cv2.minAreaRect(points_array)
            
            # Get corners of the rotated rectangle
            box = cv2.boxPoints((center, (width, height), angle))
            
            if len(box) != 4:
                return None
            
            # Sort corners: top_left, top_right, bottom_right, bottom_left
            box = np.int32(box)
            box = sorted(box.tolist(), key=lambda p: (p[1], p[0]))  # Sort by y, then x
            
            top_points = box[:2]
            bottom_points = box[2:]
            
            top_left = (min(top_points, key=lambda p: p[0])[0] / 1.0, min(top_points, key=lambda p: p[1])[1] / 1.0)
            top_right = (max(top_points, key=lambda p: p[0])[0] / 1.0, min(top_points, key=lambda p: p[1])[1] / 1.0)
            bottom_left = (min(bottom_points, key=lambda p: p[0])[0] / 1.0, max(bottom_points, key=lambda p: p[1])[1] / 1.0)
            bottom_right = (max(bottom_points, key=lambda p: p[0])[0] / 1.0, max(bottom_points, key=lambda p: p[1])[1] / 1.0)
            
            # Normalize coordinates to 0-1 range
            top_left = (np.clip(top_left[0], 0, 1), np.clip(top_left[1], 0, 1))
            top_right = (np.clip(top_right[0], 0, 1), np.clip(top_right[1], 0, 1))
            bottom_left = (np.clip(bottom_left[0], 0, 1), np.clip(bottom_left[1], 0, 1))
            bottom_right = (np.clip(bottom_right[0], 0, 1), np.clip(bottom_right[1], 0, 1))
            
            # Calculate area
            w = abs(bottom_right[0] - bottom_left[0])
            h = abs(bottom_right[1] - top_right[1])
            area = w * h
            
            # Confidence based on how well the points fit the rectangle
            confidence = min(0.95, 0.7 + (w * h * 0.2))
            
            return RectangleFrame(
                top_left=top_left,
                top_right=top_right,
                bottom_left=bottom_left,
                bottom_right=bottom_right,
                area=area,
                confidence=confidence,
                timestamp=timestamp,
                hand_id=hand_id
            )
        except Exception as e:
            logger.debug(f"Error fitting rectangle: {e}")
            return None
    
    def _is_valid_rectangle(self, rectangle: RectangleFrame) -> bool:
        """
        Validate if rectangle meets criteria.
        
        Args:
            rectangle: RectangleFrame to validate
            
        Returns:
            True if rectangle is valid, False otherwise
        """
        # Check area thresholds
        if rectangle.area < self.min_area_threshold or rectangle.area > self.max_area_threshold:
            return False
        
        # Check confidence threshold
        if rectangle.confidence < self.min_confidence:
            return False
        
        # Check if corners are reasonably positioned
        corners = [rectangle.top_left, rectangle.top_right, rectangle.bottom_left, rectangle.bottom_right]
        for corner in corners:
            if not (0 <= corner[0] <= 1 and 0 <= corner[1] <= 1):
                return False
        
        return True


class ScreenshotCapture:
    """
    Captures screenshots of areas defined by rectangle gestures.
    
    Features:
    - Extracts region from frame based on rectangle corners
    - Handles perspective correction for rotated rectangles
    - Saves screenshots with timestamps
    - Integrates with image editor
    """
    
    def __init__(self, output_dir: str = "./screenshots"):
        """
        Initialize screenshot capture.
        
        Args:
            output_dir: Directory to save screenshots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.last_screenshot_path: Optional[str] = None
    
    def capture_from_frame(
        self,
        frame: np.ndarray,
        rectangle: RectangleFrame
    ) -> Optional[np.ndarray]:
        """
        Capture image region from frame based on rectangle.
        
        Args:
            frame: Input frame (BGR format)
            rectangle: Rectangle defining capture region
            
        Returns:
            Captured image as numpy array, or None if capture failed
        """
        if frame is None or rectangle is None:
            return None
        
        try:
            height, width = frame.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            corners_pixel = rectangle.get_pixel_coordinates(width, height)
            
            # Source points (rectangle in original frame)
            src_points = np.array([
                corners_pixel['top_left'],
                corners_pixel['top_right'],
                corners_pixel['bottom_right'],
                corners_pixel['bottom_left'],
            ], dtype=np.float32)
            
            # Destination points (straight rectangle)
            dst_width = int(abs(rectangle.bottom_right[0] - rectangle.bottom_left[0]) * width)
            dst_height = int(abs(rectangle.bottom_right[1] - rectangle.top_right[1]) * height)
            
            # Ensure minimum size
            dst_width = max(dst_width, 50)
            dst_height = max(dst_height, 50)
            
            dst_points = np.array([
                [0, 0],
                [dst_width, 0],
                [dst_width, dst_height],
                [0, dst_height],
            ], dtype=np.float32)
            
            # Get perspective transform matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective warp to extract the region
            extracted = cv2.warpPerspective(frame, matrix, (dst_width, dst_height))
            
            return extracted
        except Exception as e:
            logger.error(f"Error capturing from frame: {e}")
            return None
    
    def save_screenshot(
        self,
        image: np.ndarray,
        name_prefix: str = "screenshot"
    ) -> Optional[str]:
        """
        Save captured image to disk.
        
        Args:
            image: Image to save (BGR format)
            name_prefix: Prefix for filename
            
        Returns:
            Path to saved file, or None if save failed
        """
        if image is None:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{name_prefix}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            success = cv2.imwrite(filepath, image)
            
            if success:
                self.last_screenshot_path = filepath
                logger.info(f"Screenshot saved to {filepath}")
                return filepath
            else:
                logger.error(f"Failed to save screenshot to {filepath}")
                return None
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return None
    
    def capture_and_save(
        self,
        frame: np.ndarray,
        rectangle: RectangleFrame,
        name_prefix: str = "screenshot"
    ) -> Optional[str]:
        """
        Capture from frame and save in one operation.
        
        Args:
            frame: Input frame
            rectangle: Rectangle defining capture region
            name_prefix: Prefix for filename
            
        Returns:
            Path to saved file, or None if failed
        """
        extracted = self.capture_from_frame(frame, rectangle)
        if extracted is not None:
            return self.save_screenshot(extracted, name_prefix)
        return None
    
    def get_last_screenshot_path(self) -> Optional[str]:
        """Get path to last saved screenshot."""
        return self.last_screenshot_path
