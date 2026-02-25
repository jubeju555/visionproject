#!/usr/bin/env python3
"""
Demo script to test Mode Router functionality without UI.

This script demonstrates:
1. Mode router initialization
2. Mode switching via both palms open gesture simulation
3. Gesture routing to different handlers based on mode
4. Mode change callbacks
"""

import time
import logging
from src.core.state_manager import StateManager
from src.core.mode_router import ApplicationMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def audio_play_handler(data):
    """Handler for audio play gesture in AUDIO_CONTROL mode."""
    logger.info(f"🎵 AUDIO: Play handler triggered with data: {data}")


def audio_pause_handler(data):
    """Handler for audio pause gesture in AUDIO_CONTROL mode."""
    logger.info(f"🎵 AUDIO: Pause handler triggered with data: {data}")


def image_zoom_handler(data):
    """Handler for image zoom gesture in IMAGE_EDITING mode."""
    logger.info(f"🖼️  IMAGE: Zoom handler triggered with data: {data}")


def image_rotate_handler(data):
    """Handler for image rotate gesture in IMAGE_EDITING mode."""
    logger.info(f"🖼️  IMAGE: Rotate handler triggered with data: {data}")


def neutral_handler(data):
    """Handler for gestures in NEUTRAL mode."""
    logger.info(f"💤 NEUTRAL: Gesture received (no action) with data: {data}")


def mode_change_callback(new_mode: ApplicationMode):
    """Callback for mode changes."""
    mode_emoji = {
        ApplicationMode.NEUTRAL: "💤",
        ApplicationMode.AUDIO_CONTROL: "🎵",
        ApplicationMode.IMAGE_EDITING: "🖼️"
    }
    emoji = mode_emoji.get(new_mode, "❓")
    logger.info(f"{'='*60}")
    logger.info(f"{emoji} MODE CHANGED TO: {new_mode.value.upper()} {emoji}")
    logger.info(f"{'='*60}")


def main():
    """Run the mode router demo."""
    logger.info("="*60)
    logger.info("MODE ROUTER DEMO")
    logger.info("="*60)
    
    # Create and initialize mode router
    router = StateManager(mode_switch_duration=2.0)
    router.initialize()
    
    # Register mode change callback
    router.register_mode_change_callback(mode_change_callback)
    
    # Register handlers for NEUTRAL mode
    router.register_handler(ApplicationMode.NEUTRAL, "open_palm", neutral_handler)
    router.register_handler(ApplicationMode.NEUTRAL, "fist", neutral_handler)
    
    # Register handlers for AUDIO_CONTROL mode
    router.register_handler(ApplicationMode.AUDIO_CONTROL, "open_palm", audio_play_handler)
    router.register_handler(ApplicationMode.AUDIO_CONTROL, "fist", audio_pause_handler)
    
    # Register handlers for IMAGE_EDITING mode
    router.register_handler(ApplicationMode.IMAGE_EDITING, "pinch", image_zoom_handler)
    router.register_handler(ApplicationMode.IMAGE_EDITING, "swipe_right", image_rotate_handler)
    
    logger.info("\n")
    logger.info("Starting demo sequence...")
    logger.info("\n")
    
    # DEMO SEQUENCE
    
    # 1. Test gestures in NEUTRAL mode
    logger.info("Step 1: Testing gestures in NEUTRAL mode")
    time.sleep(1)
    router.route_gesture("open_palm", {"hand_id": "left"})
    time.sleep(0.5)
    router.route_gesture("fist", {"hand_id": "right"})
    time.sleep(1)
    
    # 2. Switch to AUDIO_CONTROL mode
    logger.info("\nStep 2: Switching to AUDIO_CONTROL mode (simulating both palms open for 2 seconds)")
    gesture_data = {"hand_id": "both", "both_hands": True}
    
    # Start the gesture
    router.route_gesture("open_palm", gesture_data)
    logger.info("  - Both palms detected, timer started...")
    time.sleep(1)
    logger.info("  - 1 second elapsed...")
    time.sleep(1)
    logger.info("  - 2 seconds elapsed...")
    
    # Trigger mode switch
    router.route_gesture("open_palm", gesture_data)
    time.sleep(0.5)
    
    # 3. Test gestures in AUDIO_CONTROL mode
    logger.info("\nStep 3: Testing gestures in AUDIO_CONTROL mode")
    time.sleep(0.5)
    router.route_gesture("open_palm", {"hand_id": "left"})
    time.sleep(0.5)
    router.route_gesture("fist", {"hand_id": "right"})
    time.sleep(1)
    
    # 4. Switch to IMAGE_EDITING mode
    logger.info("\nStep 4: Switching to IMAGE_EDITING mode")
    router.route_gesture("open_palm", gesture_data)
    time.sleep(2.1)
    router.route_gesture("open_palm", gesture_data)
    time.sleep(0.5)
    
    # 5. Test gestures in IMAGE_EDITING mode
    logger.info("\nStep 5: Testing gestures in IMAGE_EDITING mode")
    time.sleep(0.5)
    router.route_gesture("pinch", {"hand_id": "left"})
    time.sleep(0.5)
    router.route_gesture("swipe_right", {"hand_id": "right"})
    time.sleep(1)
    
    # 6. Cycle back to NEUTRAL
    logger.info("\nStep 6: Cycling back to NEUTRAL mode")
    router.route_gesture("open_palm", gesture_data)
    time.sleep(2.1)
    router.route_gesture("open_palm", gesture_data)
    time.sleep(0.5)
    
    # 7. Verify we're back in NEUTRAL
    logger.info("\nStep 7: Verifying NEUTRAL mode with gesture")
    router.route_gesture("open_palm", {"hand_id": "left"})
    time.sleep(1)
    
    # Cleanup
    logger.info("\n")
    logger.info("Demo complete! Cleaning up...")
    router.cleanup()
    
    logger.info("="*60)
    logger.info("DEMO FINISHED SUCCESSFULLY")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)
