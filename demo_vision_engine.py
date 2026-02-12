#!/usr/bin/env python3
"""
Demo script for Vision Engine.

This script demonstrates the MediaPipeVisionEngine functionality:
- Captures frames from webcam
- Detects hand landmarks
- Displays the results with visual feedback
- Prints landmark information

Press 'q' to quit.
"""

import cv2
import time
import logging
from src.vision.vision_engine_impl import MediaPipeVisionEngine, VisionData

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def draw_landmarks_on_frame(frame, landmarks):
    """
    Draw hand landmarks on the frame.

    Args:
        frame: Input frame (BGR)
        landmarks: List of hand landmark data
    """
    if not landmarks:
        return frame

    # Draw for each detected hand
    for hand_data in landmarks:
        handedness = hand_data["handedness"]
        hand_landmarks = hand_data["landmarks"]
        confidence = hand_data["confidence"]

        # Draw landmarks as circles
        for i, lm in enumerate(hand_landmarks):
            x, y = lm["x"], lm["y"]

            # Draw landmark point
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw landmark index for key points
            if i in [0, 4, 8, 12, 16, 20]:  # Wrist and fingertips
                cv2.putText(
                    frame,
                    str(i),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )

        # Draw handedness label
        if hand_landmarks:
            wrist = hand_landmarks[0]
            label = f"{handedness} ({confidence:.2f})"
            cv2.putText(
                frame,
                label,
                (wrist["x"], wrist["y"] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        # Draw connections between landmarks (simplified)
        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),  # Thumb
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),  # Index
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),  # Middle
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),  # Ring
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),  # Pinky
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                cv2.line(frame, (start["x"], start["y"]), (end["x"], end["y"]), (0, 255, 0), 2)

    return frame


def main():
    """Main demo function."""
    logger.info("Starting Vision Engine Demo")
    logger.info("Press 'q' to quit, 's' to toggle smoothing")

    # Create vision engine with smoothing enabled
    engine = MediaPipeVisionEngine(
        camera_id=0,
        fps=30,
        max_queue_size=2,
        enable_smoothing=True,
        smoothing_factor=0.3,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize
    if not engine.initialize():
        logger.error("Failed to initialize Vision Engine")
        return

    logger.info("Vision Engine initialized successfully")

    # Start capture
    engine.start_capture()
    logger.info("Vision Engine capture started")

    # Display window
    window_name = "Vision Engine Demo - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    try:
        while True:
            # Get vision data with timeout
            vision_data = engine.get_vision_data(timeout=0.1)

            if vision_data:
                frame = vision_data.frame.copy()
                landmarks = vision_data.landmarks

                # Draw landmarks on frame
                frame = draw_landmarks_on_frame(frame, landmarks)

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed
                    start_time = time.time()

                # Draw FPS and hand count
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Hands: {len(landmarks)}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Print landmark info for first hand
                if landmarks:
                    hand = landmarks[0]
                    logger.debug(
                        f"Detected {hand['handedness']} hand with "
                        f"{len(hand['landmarks'])} landmarks "
                        f"(confidence: {hand['confidence']:.2f})"
                    )

                # Display frame
                cv2.imshow(window_name, frame)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested")
                break
            elif key == ord("s"):
                # Toggle smoothing (thread-safe)
                current = engine.enable_smoothing
                engine.set_smoothing(not current)
                logger.info(f"Smoothing: {'enabled' if not current else 'disabled'}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        cv2.destroyAllWindows()
        engine.cleanup()
        logger.info("Demo finished")


if __name__ == "__main__":
    main()
