"""
Gesture Recognition Engine Implementation.

This module provides deterministic gesture classification based on:
- Landmark distance thresholds
- Angle calculations
- Velocity vectors across frames

Runs in its own thread and consumes landmark data from VisionEngine.
"""

import threading
import queue
import logging
import time
import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StaticGesture(Enum):
    """Enumeration of static gesture types."""
    OPEN_PALM = "open_palm"
    FIST = "fist"
    PINCH = "pinch"
    TWO_FINGERS = "two_fingers"
    THREE_FINGERS = "three_fingers"
    UNKNOWN = "unknown"


class DynamicGesture(Enum):
    """Enumeration of dynamic gesture types."""
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    VERTICAL_MOTION = "vertical_motion"
    CIRCULAR_MOTION = "circular_motion"
    TWO_HAND_SPREAD = "two_hand_spread"
    NONE = "none"


@dataclass
class GestureEvent:
    """Structured gesture event output."""
    gesture_name: str
    confidence_score: float
    timestamp: float
    hand_id: str  # "left", "right", or "both"
    gesture_type: str  # "static" or "dynamic"
    additional_data: Optional[Dict[str, Any]] = None


class GestureRecognitionEngine:
    """
    Gesture Recognition Engine that runs in its own thread.
    
    Consumes landmark data from VisionEngine and emits structured gesture events.
    Uses deterministic classification based on geometric analysis.
    """
    
    def __init__(
        self,
        input_queue: queue.Queue,
        max_queue_size: int = 10,
        history_size: int = 10,
        velocity_threshold: float = 0.05,
        circular_threshold: float = 0.3,
    ):
        """
        Initialize Gesture Recognition Engine.
        
        Args:
            input_queue: Queue to receive vision data from VisionEngine
            max_queue_size: Maximum size of output event queue
            history_size: Number of frames to keep for motion analysis
            velocity_threshold: Threshold for detecting significant motion
            circular_threshold: Threshold for detecting circular motion
        """
        self.input_queue = input_queue
        self.max_queue_size = max_queue_size
        self.history_size = history_size
        self.velocity_threshold = velocity_threshold
        self.circular_threshold = circular_threshold
        
        # Output queue for gesture events
        self._event_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        
        # Threading components
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # History tracking for dynamic gestures
        self._landmark_history: List[List[Dict[str, Any]]] = []
        
        # Latest gesture state
        self._latest_gesture: Optional[GestureEvent] = None
        
    def start(self) -> None:
        """Start the gesture recognition thread."""
        if self._running:
            logger.warning("Gesture Recognition Engine already running")
            return
            
        logger.info("Starting Gesture Recognition Engine")
        self._running = True
        self._thread = threading.Thread(
            target=self._recognition_loop,
            daemon=True,
            name="GestureRecognitionThread"
        )
        self._thread.start()
        logger.info("Gesture Recognition Engine started")
        
    def stop(self) -> None:
        """Stop the gesture recognition thread."""
        if not self._running:
            logger.warning("Gesture Recognition Engine not running")
            return
            
        logger.info("Stopping Gesture Recognition Engine")
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Recognition thread did not stop gracefully")
                
        logger.info("Gesture Recognition Engine stopped")
        
    def is_running(self) -> bool:
        """Check if the recognition thread is running."""
        return self._running
        
    def get_event(self, timeout: Optional[float] = None) -> Optional[GestureEvent]:
        """
        Get the next gesture event from the output queue.
        
        Args:
            timeout: Maximum time to wait in seconds (None = blocking)
            
        Returns:
            GestureEvent or None if timeout/empty
        """
        try:
            if timeout is None:
                return self._event_queue.get(block=True)
            else:
                return self._event_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
            
    def get_latest_gesture(self) -> Optional[GestureEvent]:
        """
        Get the latest recognized gesture (thread-safe).
        
        Returns:
            Latest GestureEvent or None if no gesture recognized yet
        """
        with self._lock:
            return self._latest_gesture
            
    def _recognition_loop(self) -> None:
        """Main recognition loop running in separate thread."""
        logger.info("Gesture recognition loop started")
        
        try:
            while self._running:
                # Get vision data from input queue
                try:
                    vision_data = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                if vision_data is None:
                    continue
                    
                # Extract landmarks
                landmarks = getattr(vision_data, 'landmarks', [])
                timestamp = getattr(vision_data, 'timestamp', time.time())
                
                if not landmarks:
                    continue
                    
                # Update history
                self._update_history(landmarks)
                
                # Recognize gestures for each hand
                events = []
                static_gestures_by_hand = {}
                
                for hand_idx, hand_data in enumerate(landmarks):
                    # Classify static gesture
                    static_gesture, static_confidence = self._classify_static_gesture(hand_data)
                    
                    hand_id = hand_data.get('handedness', 'Unknown').lower()
                    static_gestures_by_hand[hand_id] = (static_gesture, static_confidence)
                    
                    # Create event for static gesture
                    if static_gesture != StaticGesture.UNKNOWN:
                        event = GestureEvent(
                            gesture_name=static_gesture.value,
                            confidence_score=static_confidence,
                            timestamp=timestamp,
                            hand_id=hand_id,
                            gesture_type='static'
                        )
                        events.append(event)
                
                # Check for both hands showing open palm (for mode switching)
                if len(landmarks) == 2:
                    both_open = all(
                        gesture == StaticGesture.OPEN_PALM
                        for gesture, _ in static_gestures_by_hand.values()
                    )
                    if both_open:
                        # Calculate average confidence
                        avg_confidence = sum(conf for _, conf in static_gestures_by_hand.values()) / 2
                        # Emit special event for both palms open
                        event = GestureEvent(
                            gesture_name="open_palm",
                            confidence_score=avg_confidence,
                            timestamp=timestamp,
                            hand_id="both",
                            gesture_type='static',
                            additional_data={'both_hands': True}
                        )
                        events.append(event)
                
                # Classify dynamic gestures (requires history)
                if len(self._landmark_history) >= 3:
                    dynamic_gesture, dynamic_confidence, hand_id = self._classify_dynamic_gesture()
                    
                    if dynamic_gesture != DynamicGesture.NONE:
                        event = GestureEvent(
                            gesture_name=dynamic_gesture.value,
                            confidence_score=dynamic_confidence,
                            timestamp=timestamp,
                            hand_id=hand_id,
                            gesture_type='dynamic'
                        )
                        events.append(event)
                
                # Emit events
                for event in events:
                    self._emit_event(event)
                    
        except Exception as e:
            logger.error(f"Error in recognition loop: {e}", exc_info=True)
        finally:
            logger.info("Gesture recognition loop ended")
            
    def _update_history(self, landmarks: List[Dict[str, Any]]) -> None:
        """
        Update landmark history for motion analysis.
        
        Args:
            landmarks: Current frame landmarks
        """
        self._landmark_history.append(landmarks)
        
        # Keep only recent history
        if len(self._landmark_history) > self.history_size:
            self._landmark_history.pop(0)
            
    def _emit_event(self, event: GestureEvent) -> None:
        """
        Emit a gesture event to the output queue.
        
        Args:
            event: GestureEvent to emit
        """
        # Update latest gesture
        with self._lock:
            self._latest_gesture = event
            
        # Try to add to queue
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            # Drop oldest event if queue is full
            try:
                self._event_queue.get_nowait()
                self._event_queue.put_nowait(event)
            except (queue.Empty, queue.Full):
                pass
                
    # ==================== Static Gesture Recognition ====================
    
    def _classify_static_gesture(
        self, hand_data: Dict[str, Any]
    ) -> Tuple[StaticGesture, float]:
        """
        Classify static gesture from hand landmarks.
        
        Args:
            hand_data: Dictionary containing hand landmarks
            
        Returns:
            Tuple of (StaticGesture, confidence_score)
        """
        landmarks = hand_data.get('landmarks_normalized', [])
        if len(landmarks) != 21:
            return StaticGesture.UNKNOWN, 0.0
            
        # Check each gesture type
        # Priority: Pinch > Fist > Two Fingers > Three Fingers > Open Palm
        
        # Check pinch (thumb tip near index finger tip)
        is_pinch, pinch_conf = self._is_pinch(landmarks)
        if is_pinch:
            return StaticGesture.PINCH, pinch_conf
            
        # Check fist (all fingers curled)
        is_fist, fist_conf = self._is_fist(landmarks)
        if is_fist:
            return StaticGesture.FIST, fist_conf
            
        # Count extended fingers
        extended_fingers = self._count_extended_fingers(landmarks)
        
        if extended_fingers == 2:
            return StaticGesture.TWO_FINGERS, 0.85
        elif extended_fingers == 3:
            return StaticGesture.THREE_FINGERS, 0.85
        elif extended_fingers >= 4:
            return StaticGesture.OPEN_PALM, 0.9
            
        return StaticGesture.UNKNOWN, 0.0
        
    def _is_pinch(self, landmarks: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """
        Check if hand is making a pinch gesture.
        
        Args:
            landmarks: Normalized hand landmarks
            
        Returns:
            Tuple of (is_pinch, confidence)
        """
        # Thumb tip (4) and index finger tip (8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance
        distance = self._calculate_distance(thumb_tip, index_tip)
        
        # Pinch threshold (normalized coordinates)
        pinch_threshold = 0.05
        
        if distance < pinch_threshold:
            # Confidence based on how close they are
            confidence = max(0.5, 1.0 - (distance / pinch_threshold))
            return True, confidence
            
        return False, 0.0
        
    def _is_fist(self, landmarks: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """
        Check if hand is making a fist gesture.
        
        Args:
            landmarks: Normalized hand landmarks
            
        Returns:
            Tuple of (is_fist, confidence)
        """
        # Check if all fingertips are curled (close to palm)
        wrist = landmarks[0]
        fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        
        curled_count = 0
        for tip in fingertips:
            # Distance from fingertip to wrist
            dist = self._calculate_distance(tip, wrist)
            
            # If fingertip is close to wrist, it's curled
            if dist < 0.3:  # Threshold for curled finger
                curled_count += 1
                
        # Fist if at least 4 fingers are curled
        if curled_count >= 4:
            confidence = 0.7 + (curled_count / 5.0) * 0.3
            return True, confidence
            
        return False, 0.0
        
    def _count_extended_fingers(self, landmarks: List[Dict[str, Any]]) -> int:
        """
        Count the number of extended fingers.
        
        Args:
            landmarks: Normalized hand landmarks
            
        Returns:
            Number of extended fingers (0-5)
        """
        extended = 0
        
        # Check thumb (special case - check x distance instead)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # Thumb is extended if tip is further from wrist than IP joint
        if self._calculate_distance(thumb_tip, landmarks[0]) > \
           self._calculate_distance(thumb_mcp, landmarks[0]):
            extended += 1
            
        # Check other fingers (index, middle, ring, pinky)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            
            # Finger is extended if tip is above (lower y value) the PIP joint
            if tip['y'] < pip['y'] - 0.05:  # Threshold for extension
                extended += 1
                
        return extended
        
    # ==================== Dynamic Gesture Recognition ====================
    
    def _classify_dynamic_gesture(self) -> Tuple[DynamicGesture, float, str]:
        """
        Classify dynamic gesture from landmark history.
        
        Returns:
            Tuple of (DynamicGesture, confidence_score, hand_id)
        """
        if len(self._landmark_history) < 3:
            return DynamicGesture.NONE, 0.0, "unknown"
            
        # Check two-hand spread first (requires both hands)
        is_spread, spread_conf = self._is_two_hand_spread()
        if is_spread:
            return DynamicGesture.TWO_HAND_SPREAD, spread_conf, "both"
            
        # Analyze single hand motion
        # Use the first hand in the most recent frame
        if not self._landmark_history[-1]:
            return DynamicGesture.NONE, 0.0, "unknown"
            
        hand_data = self._landmark_history[-1][0]
        hand_id = hand_data.get('handedness', 'Unknown').lower()
        
        # Calculate velocity vector
        velocity = self._calculate_velocity()
        
        if velocity is None:
            return DynamicGesture.NONE, 0.0, hand_id
            
        vx, vy = velocity
        speed = math.sqrt(vx * vx + vy * vy)
        
        # Check if motion is significant
        if speed < self.velocity_threshold:
            return DynamicGesture.NONE, 0.0, hand_id
            
        # Check circular motion
        is_circular, circular_conf = self._is_circular_motion()
        if is_circular:
            return DynamicGesture.CIRCULAR_MOTION, circular_conf, hand_id
            
        # Check swipe gestures based on velocity direction
        if abs(vx) > abs(vy) * 1.5:  # Horizontal motion dominant
            if vx > 0:
                confidence = min(0.95, 0.7 + speed * 2.0)
                return DynamicGesture.SWIPE_RIGHT, confidence, hand_id
            else:
                confidence = min(0.95, 0.7 + speed * 2.0)
                return DynamicGesture.SWIPE_LEFT, confidence, hand_id
        elif abs(vy) > abs(vx) * 1.5:  # Vertical motion dominant
            confidence = min(0.95, 0.7 + speed * 2.0)
            return DynamicGesture.VERTICAL_MOTION, confidence, hand_id
            
        return DynamicGesture.NONE, 0.0, hand_id
        
    def _calculate_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Calculate velocity vector from landmark history.
        
        Returns:
            Tuple of (vx, vy) or None if insufficient history
        """
        if len(self._landmark_history) < 2:
            return None
            
        # Use wrist (landmark 0) for overall hand motion
        current_frame = self._landmark_history[-1]
        previous_frame = self._landmark_history[-2]
        
        if not current_frame or not previous_frame:
            return None
            
        # Get wrist position from first hand in each frame
        current_wrist = current_frame[0].get('landmarks_normalized', [None])[0]
        previous_wrist = previous_frame[0].get('landmarks_normalized', [None])[0]
        
        if current_wrist is None or previous_wrist is None:
            return None
            
        vx = current_wrist['x'] - previous_wrist['x']
        vy = current_wrist['y'] - previous_wrist['y']
        
        return (vx, vy)
        
    def _is_circular_motion(self) -> Tuple[bool, float]:
        """
        Check if hand is making circular motion.
        
        Returns:
            Tuple of (is_circular, confidence)
        """
        if len(self._landmark_history) < 5:
            return False, 0.0
            
        # Extract wrist positions from recent history
        positions = []
        for frame in self._landmark_history[-5:]:
            if frame and len(frame) > 0:
                wrist = frame[0].get('landmarks_normalized', [None])[0]
                if wrist:
                    positions.append((wrist['x'], wrist['y']))
                    
        if len(positions) < 5:
            return False, 0.0
            
        # Calculate center of positions
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)
        
        # Calculate distances from center
        distances = [
            math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2)
            for p in positions
        ]
        
        # Check if distances are relatively consistent (circular path)
        avg_distance = sum(distances) / len(distances)
        
        # Guard against division by zero
        if avg_distance == 0:
            return False, 0.0
            
        variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
        
        # Low variance indicates circular motion
        if variance < self.circular_threshold * avg_distance:
            confidence = min(0.9, 0.6 + (1.0 - variance / avg_distance))
            return True, confidence
            
        return False, 0.0
        
    def _is_two_hand_spread(self) -> Tuple[bool, float]:
        """
        Check if both hands are spreading apart.
        
        Returns:
            Tuple of (is_spreading, confidence)
        """
        if len(self._landmark_history) < 2:
            return False, 0.0
            
        current_frame = self._landmark_history[-1]
        previous_frame = self._landmark_history[-2]
        
        # Need both hands in both frames
        if len(current_frame) < 2 or len(previous_frame) < 2:
            return False, 0.0
            
        # Calculate distance between hands in both frames
        current_dist = self._calculate_hand_distance(current_frame)
        previous_dist = self._calculate_hand_distance(previous_frame)
        
        if current_dist is None or previous_dist is None:
            return False, 0.0
            
        # Check if distance is increasing
        distance_change = current_dist - previous_dist
        
        if distance_change > 0.05:  # Threshold for spreading
            confidence = min(0.9, 0.7 + distance_change * 5.0)
            return True, confidence
            
        return False, 0.0
        
    def _calculate_hand_distance(
        self, frame_landmarks: List[Dict[str, Any]]
    ) -> Optional[float]:
        """
        Calculate distance between two hands in a frame.
        
        Args:
            frame_landmarks: Landmarks for all hands in frame
            
        Returns:
            Distance between hands or None if insufficient data
        """
        if len(frame_landmarks) < 2:
            return None
            
        # Get wrist positions
        hand1_landmarks = frame_landmarks[0].get('landmarks_normalized', [])
        hand2_landmarks = frame_landmarks[1].get('landmarks_normalized', [])
        
        if len(hand1_landmarks) < 1 or len(hand2_landmarks) < 1:
            return None
            
        wrist1 = hand1_landmarks[0]
        wrist2 = hand2_landmarks[0]
        
        return self._calculate_distance(wrist1, wrist2)
        
    # ==================== Utility Functions ====================
    
    def _calculate_distance(
        self, point1: Dict[str, float], point2: Dict[str, float]
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point with 'x', 'y', 'z' keys
            point2: Second point with 'x', 'y', 'z' keys
            
        Returns:
            Distance between points
        """
        dx = point1['x'] - point2['x']
        dy = point1['y'] - point2['y']
        dz = point1.get('z', 0.0) - point2.get('z', 0.0)
        
        return math.sqrt(dx * dx + dy * dy + dz * dz)
        
    def _calculate_angle(
        self, point1: Dict[str, float], point2: Dict[str, float], point3: Dict[str, float]
    ) -> float:
        """
        Calculate angle at point2 formed by point1-point2-point3.
        
        Args:
            point1: First point
            point2: Vertex point
            point3: Third point
            
        Returns:
            Angle in degrees
        """
        # Vectors from point2 to point1 and point3
        v1x = point1['x'] - point2['x']
        v1y = point1['y'] - point2['y']
        v2x = point3['x'] - point2['x']
        v2y = point3['y'] - point2['y']
        
        # Calculate dot product and magnitudes
        dot_product = v1x * v2x + v1y * v2y
        mag1 = math.sqrt(v1x * v1x + v1y * v1y)
        mag2 = math.sqrt(v2x * v2x + v2y * v2y)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        # Calculate angle
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
        
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
