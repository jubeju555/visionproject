"""
Vision Engine Implementation with MediaPipe Hands integration.

This module provides the complete implementation of VisionEngine with:
- Webcam capture at 30 FPS
- BGR to RGB conversion
- MediaPipe Hands landmark detection
- Dual-hand tracking support (21 landmarks per hand)
- Non-blocking threaded design
- Queue-based structured data output
- Optional exponential smoothing
- Graceful shutdown and camera release
"""

import threading
import queue
import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2
import mediapipe as mp

from src.core.vision_engine import VisionEngine

logger = logging.getLogger(__name__)


class VisionData:
    """Structured data container for vision output."""
    
    def __init__(
        self,
        frame: np.ndarray,
        frame_rgb: np.ndarray,
        landmarks: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[float] = None
    ):
        """
        Initialize vision data.
        
        Args:
            frame: Original BGR frame
            frame_rgb: RGB converted frame
            landmarks: List of hand landmarks (one dict per detected hand)
            timestamp: Frame timestamp
        """
        self.frame = frame
        self.frame_rgb = frame_rgb
        self.landmarks = landmarks or []
        self.timestamp = timestamp


class MediaPipeVisionEngine(VisionEngine):
    """
    Vision Engine implementation with MediaPipe Hands integration.
    
    Captures webcam frames at 30 FPS, detects hand landmarks using MediaPipe,
    and provides structured output via a thread-safe queue.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        fps: int = 30,
        max_queue_size: int = 2,
        enable_smoothing: bool = False,
        smoothing_factor: float = 0.5,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the Vision Engine.
        
        Args:
            camera_id: Camera device ID
            fps: Target frames per second
            max_queue_size: Maximum size of output queue
            enable_smoothing: Enable exponential smoothing for landmarks
            smoothing_factor: Smoothing factor (0-1, higher = more smoothing)
            max_num_hands: Maximum number of hands to detect (1 or 2)
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        super().__init__()
        self.camera_id = camera_id
        self.fps = fps
        self.max_queue_size = max_queue_size
        self.enable_smoothing = enable_smoothing
        self.smoothing_factor = smoothing_factor
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Threading components
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Camera and MediaPipe components
        self._capture: Optional[cv2.VideoCapture] = None
        self._mp_hands = None
        self._hands = None
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        
        # Output queue for structured data
        self._output_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        
        # Latest frame for get_frame() method
        self._latest_frame: Optional[np.ndarray] = None
        
        # Smoothing state
        self._prev_landmarks: Optional[List[Any]] = None
    
    def initialize(self) -> bool:
        """
        Initialize camera and MediaPipe Hands.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info(f"Initializing Vision Engine with camera {self.camera_id}")
            
            # Initialize MediaPipe Hands
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            # Initialize camera
            self._capture = cv2.VideoCapture(self.camera_id)
            if not self._capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual camera properties
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Camera initialized: {width}x{height} @ {actual_fps} FPS")
            logger.info(f"MediaPipe Hands initialized: max_hands={self.max_num_hands}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Vision Engine: {e}", exc_info=True)
            return False
    
    def start_capture(self) -> None:
        """Start the frame capture and processing thread."""
        if self._running:
            logger.warning("Vision Engine is already running")
            return
        
        if not self._capture or not self._capture.isOpened():
            logger.error("Camera not initialized. Call initialize() first.")
            return
        
        logger.info("Starting Vision Engine capture thread")
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="VisionEngineThread"
        )
        self._capture_thread.start()
        logger.info("Vision Engine capture thread started")
    
    def stop_capture(self) -> None:
        """Stop the frame capture thread."""
        if not self._running:
            logger.warning("Vision Engine is not running")
            return
        
        logger.info("Stopping Vision Engine capture thread")
        self._running = False
        
        # Wait for thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not stop gracefully")
        
        logger.info("Vision Engine capture thread stopped")
    
    def _capture_loop(self) -> None:
        """
        Main capture loop running in separate thread.
        
        Continuously captures frames, processes them with MediaPipe,
        and pushes structured data to the output queue.
        """
        logger.info("Vision Engine capture loop started")
        frame_count = 0
        
        try:
            while self._running:
                # Capture frame
                ret, frame = self._capture.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                frame_count += 1
                
                # Convert BGR to RGB (MediaPipe requirement)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe Hands
                results = self._hands.process(frame_rgb)
                
                # Extract landmarks
                landmarks = self._extract_landmarks(results, frame.shape)
                
                # Apply smoothing if enabled
                if self.enable_smoothing and landmarks:
                    landmarks = self._apply_smoothing(landmarks)
                
                # Update latest frame (thread-safe)
                with self._lock:
                    self._latest_frame = frame.copy()
                
                # Create structured data
                vision_data = VisionData(
                    frame=frame,
                    frame_rgb=frame_rgb,
                    landmarks=landmarks,
                    timestamp=cv2.getTickCount() / cv2.getTickFrequency()
                )
                
                # Push to queue (non-blocking)
                try:
                    self._output_queue.put_nowait(vision_data)
                except queue.Full:
                    # Drop oldest frame if queue is full
                    try:
                        self._output_queue.get_nowait()
                        self._output_queue.put_nowait(vision_data)
                    except (queue.Empty, queue.Full):
                        pass
                
                # Rate limiting to achieve target FPS
                if self.fps > 0:
                    cv2.waitKey(max(1, int(1000 / self.fps)))
        
        except Exception as e:
            logger.error(f"Error in capture loop: {e}", exc_info=True)
        finally:
            logger.info(f"Vision Engine capture loop ended (processed {frame_count} frames)")
    
    def _extract_landmarks(
        self,
        results: Any,
        frame_shape: Tuple[int, int, int]
    ) -> List[Dict[str, Any]]:
        """
        Extract hand landmarks from MediaPipe results.
        
        Args:
            results: MediaPipe processing results
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            List of dictionaries, one per detected hand, each containing:
                - 'handedness': 'Left' or 'Right'
                - 'landmarks': List of 21 landmarks as (x, y, z) tuples
                - 'landmarks_normalized': List of 21 normalized landmarks
                - 'confidence': Detection confidence score
        """
        if not results.multi_hand_landmarks:
            return []
        
        height, width, _ = frame_shape
        hands_data = []
        
        # Process each detected hand
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness (left/right)
            handedness = "Unknown"
            confidence = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness_info = results.multi_handedness[idx]
                handedness = handedness_info.classification[0].label
                confidence = handedness_info.classification[0].score
            
            # Extract 21 landmarks
            landmarks = []
            landmarks_normalized = []
            
            for landmark in hand_landmarks.landmark:
                # Normalized coordinates (0-1)
                landmarks_normalized.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
                
                # Pixel coordinates
                landmarks.append({
                    'x': int(landmark.x * width),
                    'y': int(landmark.y * height),
                    'z': landmark.z
                })
            
            hand_data = {
                'handedness': handedness,
                'landmarks': landmarks,
                'landmarks_normalized': landmarks_normalized,
                'confidence': confidence
            }
            hands_data.append(hand_data)
        
        return hands_data
    
    def _apply_smoothing(
        self,
        landmarks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply exponential smoothing to landmarks.
        
        Uses formula: smoothed = alpha * current + (1 - alpha) * previous
        where alpha = 1 - smoothing_factor
        
        Args:
            landmarks: Current landmarks data
            
        Returns:
            Smoothed landmarks data
        """
        if not self._prev_landmarks or len(self._prev_landmarks) != len(landmarks):
            # First frame or hand count changed, no smoothing
            self._prev_landmarks = landmarks
            return landmarks
        
        alpha = 1.0 - self.smoothing_factor
        smoothed_landmarks = []
        
        for curr_hand, prev_hand in zip(landmarks, self._prev_landmarks):
            smoothed_hand = {
                'handedness': curr_hand['handedness'],
                'confidence': curr_hand['confidence'],
                'landmarks': [],
                'landmarks_normalized': []
            }
            
            # Smooth each landmark
            for curr_lm, prev_lm in zip(
                curr_hand['landmarks'],
                prev_hand['landmarks']
            ):
                smoothed_lm = {
                    'x': int(alpha * curr_lm['x'] + (1 - alpha) * prev_lm['x']),
                    'y': int(alpha * curr_lm['y'] + (1 - alpha) * prev_lm['y']),
                    'z': alpha * curr_lm['z'] + (1 - alpha) * prev_lm['z']
                }
                smoothed_hand['landmarks'].append(smoothed_lm)
            
            # Smooth normalized landmarks
            for curr_lm_norm, prev_lm_norm in zip(
                curr_hand['landmarks_normalized'],
                prev_hand['landmarks_normalized']
            ):
                smoothed_lm_norm = {
                    'x': alpha * curr_lm_norm['x'] + (1 - alpha) * prev_lm_norm['x'],
                    'y': alpha * curr_lm_norm['y'] + (1 - alpha) * prev_lm_norm['y'],
                    'z': alpha * curr_lm_norm['z'] + (1 - alpha) * prev_lm_norm['z']
                }
                smoothed_hand['landmarks_normalized'].append(smoothed_lm_norm)
            
            smoothed_landmarks.append(smoothed_hand)
        
        self._prev_landmarks = smoothed_landmarks
        return smoothed_landmarks
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame (thread-safe).
        
        Returns:
            Latest frame as BGR numpy array, or None if unavailable
        """
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None
    
    def get_vision_data(self, timeout: Optional[float] = None) -> Optional[VisionData]:
        """
        Get the next vision data from the output queue.
        
        Args:
            timeout: Maximum time to wait in seconds (None = blocking)
            
        Returns:
            VisionData object or None if timeout/empty
        """
        try:
            if timeout is None:
                return self._output_queue.get(block=True)
            else:
                return self._output_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def is_running(self) -> bool:
        """
        Check if the capture thread is running.
        
        Returns:
            True if running, False otherwise
        """
        return self._running
    
    def cleanup(self) -> None:
        """Clean up resources and release camera."""
        logger.info("Cleaning up Vision Engine")
        
        # Stop capture thread
        self.stop_capture()
        
        # Release MediaPipe resources
        if self._hands:
            self._hands.close()
            self._hands = None
            logger.info("MediaPipe Hands released")
        
        # Release camera
        if self._capture:
            self._capture.release()
            self._capture = None
            logger.info("Camera released")
        
        # Clear queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Vision Engine cleanup complete")
