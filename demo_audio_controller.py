#!/usr/bin/env python3
"""
Demo script for Audio Control Module.

This script demonstrates the AudioControlModule functionality with:
- Simulated gesture events
- Real-time audio control
- PulseAudio integration (Linux)
- Visual feedback
"""

import time
import logging
import sys
from src.audio.audio_controller_module import AudioControlModule
from src.core.state_manager import StateManager
from src.core.mode_router import ApplicationMode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print("  Audio Control Module Demo")
    print("="*60)
    print("\nThis demo shows the AudioControlModule with simulated gestures.")
    print("\nFeatures demonstrated:")
    print("  • Play/Pause control")
    print("  • Volume control via pinch distance")
    print("  • Tempo control via vertical position (stubbed)")
    print("  • Pitch control via horizontal position (stubbed)")
    print("  • Track navigation")
    print("\nNote: PulseAudio commands will only work on Linux systems")
    print("      with PulseAudio installed.")
    print("="*60 + "\n")


def print_state(audio_module: AudioControlModule):
    """Print current audio state."""
    state = audio_module.get_state()
    print(f"\n--- Audio State ---")
    print(f"Playing:  {'▶ Yes' if state.is_playing else '⏸ No'}")
    print(f"Volume:   {'█' * int(state.volume * 20):20s} {state.volume:.2f}")
    print(f"Tempo:    {state.tempo:.2f}x")
    print(f"Pitch:    {state.pitch:+.1f} semitones")
    print("-" * 40)


def demo_discrete_controls(audio_module: AudioControlModule):
    """Demonstrate discrete controls (play/pause, track skip)."""
    print("\n\n=== DEMO 1: Discrete Controls ===\n")
    
    print("1. Testing Play/Pause (fist gesture)...")
    audio_module.handle_play_pause()
    time.sleep(0.5)
    print_state(audio_module)
    
    time.sleep(1)
    
    print("\n2. Pause again...")
    audio_module.handle_play_pause()
    time.sleep(0.5)
    print_state(audio_module)
    
    time.sleep(1)
    
    print("\n3. Testing Next Track (swipe right)...")
    audio_module.handle_next_track()
    time.sleep(0.5)
    print("   → Skipped to next track")
    
    time.sleep(1)
    
    print("\n4. Testing Previous Track (swipe left)...")
    audio_module.handle_previous_track()
    time.sleep(0.5)
    print("   → Returned to previous track")


def demo_volume_control(audio_module: AudioControlModule):
    """Demonstrate volume control via pinch distance."""
    print("\n\n=== DEMO 2: Volume Control (Pinch Distance) ===\n")
    
    pinch_values = [
        (0.0, "Minimum volume (tight pinch)"),
        (0.05, "Low volume"),
        (0.1, "Medium volume"),
        (0.15, "High volume"),
        (0.2, "Maximum volume (wide pinch)")
    ]
    
    for pinch_distance, description in pinch_values:
        print(f"\n{description}:")
        print(f"  Pinch distance: {pinch_distance:.3f}")
        audio_module.handle_volume_control({'pinch_distance': pinch_distance})
        time.sleep(0.3)
        print_state(audio_module)
        time.sleep(0.5)


def demo_tempo_control(audio_module: AudioControlModule):
    """Demonstrate tempo control via vertical position."""
    print("\n\n=== DEMO 3: Tempo Control (Vertical Position) ===\n")
    print("NOTE: Tempo control is stubbed by default (real-time DSP can be unstable)")
    
    positions = [
        (0.0, "Top of screen → Fast tempo"),
        (0.5, "Middle of screen → Normal tempo"),
        (1.0, "Bottom of screen → Slow tempo")
    ]
    
    for vertical_pos, description in positions:
        print(f"\n{description}:")
        print(f"  Vertical position: {vertical_pos:.2f}")
        audio_module.handle_tempo_control({'vertical_position': vertical_pos})
        time.sleep(0.3)
        print(f"  Current tempo: {audio_module.get_tempo():.2f}x (stubbed)")
        time.sleep(0.5)


def demo_pitch_control(audio_module: AudioControlModule):
    """Demonstrate pitch control via horizontal position."""
    print("\n\n=== DEMO 4: Pitch Control (Horizontal Position) ===\n")
    print("NOTE: Pitch control is stubbed by default (real-time DSP can be unstable)")
    
    positions = [
        (0.0, "Left side → Lower pitch"),
        (0.5, "Center → No pitch shift"),
        (1.0, "Right side → Higher pitch")
    ]
    
    for horizontal_pos, description in positions:
        print(f"\n{description}:")
        print(f"  Horizontal position: {horizontal_pos:.2f}")
        audio_module.handle_pitch_control({'horizontal_position': horizontal_pos})
        time.sleep(0.3)
        print(f"  Current pitch: {audio_module.get_pitch():+.1f} semitones (stubbed)")
        time.sleep(0.5)


def demo_mode_router_integration(audio_module: AudioControlModule):
    """Demonstrate integration with ModeRouter."""
    print("\n\n=== DEMO 5: Mode Router Integration ===\n")
    
    # Create state manager
    state_manager = StateManager()
    state_manager.initialize()
    
    print("1. Registering audio control handlers with ModeRouter...")
    
    # Register handlers for AUDIO_CONTROL mode
    state_manager.register_handler(
        ApplicationMode.AUDIO_CONTROL,
        "fist",
        audio_module.handle_play_pause
    )
    state_manager.register_handler(
        ApplicationMode.AUDIO_CONTROL,
        "pinch",
        audio_module.handle_volume_control
    )
    state_manager.register_handler(
        ApplicationMode.AUDIO_CONTROL,
        "swipe_right",
        audio_module.handle_next_track
    )
    state_manager.register_handler(
        ApplicationMode.AUDIO_CONTROL,
        "swipe_left",
        audio_module.handle_previous_track
    )
    
    print("   ✓ Handlers registered")
    
    # Switch to AUDIO_CONTROL mode
    print("\n2. Switching to AUDIO_CONTROL mode...")
    state_manager.set_mode(ApplicationMode.AUDIO_CONTROL)
    print(f"   Current mode: {state_manager.get_mode().value}")
    
    # Simulate gesture events
    print("\n3. Routing gesture events through ModeRouter...")
    
    print("\n   • Fist gesture (play/pause):")
    state_manager.route_gesture("fist", {})
    time.sleep(0.5)
    print_state(audio_module)
    
    print("\n   • Pinch gesture (volume):")
    state_manager.route_gesture("pinch", {'pinch_distance': 0.12})
    time.sleep(0.5)
    print_state(audio_module)
    
    print("\n   • Swipe right (next track):")
    state_manager.route_gesture("swipe_right", {})
    time.sleep(0.5)
    print("     → Next track")
    
    # Cleanup
    state_manager.cleanup()
    print("\n   ✓ ModeRouter integration test complete")


def main():
    """Main demo function."""
    print_banner()
    
    # Create and initialize audio control module
    print("Initializing Audio Control Module...")
    audio_module = AudioControlModule(stub_dsp=True)
    
    if not audio_module.initialize():
        logger.error("Failed to initialize AudioControlModule")
        return 1
    
    print("✓ AudioControlModule initialized\n")
    
    try:
        # Run demos
        demo_discrete_controls(audio_module)
        time.sleep(1)
        
        demo_volume_control(audio_module)
        time.sleep(1)
        
        demo_tempo_control(audio_module)
        time.sleep(1)
        
        demo_pitch_control(audio_module)
        time.sleep(1)
        
        demo_mode_router_integration(audio_module)
        
        # Final state
        print("\n\n=== Final State ===")
        print_state(audio_module)
        
        print("\n\n" + "="*60)
        print("  Demo Complete!")
        print("="*60)
        print("\nThe AudioControlModule is ready for integration with")
        print("real gesture recognition and media players.\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        print("\nCleaning up...")
        audio_module.cleanup()
        print("✓ Cleanup complete\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
