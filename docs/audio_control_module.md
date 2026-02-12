# Audio Control Module - Technical Documentation

## Overview

The Audio Control Module provides gesture-controlled audio playback with discrete and continuous controls. It integrates seamlessly with the gesture recognition system and mode router to provide real-time, non-blocking audio control.

## Architecture

### Components

1. **AudioControlModule**: Main controller class
2. **AudioState**: Data class for audio state
3. **AudioCommand**: Enum for audio commands
4. **Worker Thread**: Async command execution

### Design Principles

- **Non-blocking**: All operations are async via worker thread
- **Thread-safe**: State access protected by locks
- **Real-time**: Minimal latency for gesture-to-audio control
- **Modular**: Clean interfaces for gesture handlers
- **Stubbed DSP**: Tempo/pitch are stubbed by default (real-time DSP unstable)

## Features

### Discrete Controls

- **Play/Pause**: Toggle playback state
- **Next Track**: Skip to next track
- **Previous Track**: Go to previous track

### Continuous Controls

- **Volume**: Mapped to pinch distance (0.0 - 1.0)
- **Tempo**: Mapped to vertical hand position (0.5x - 2.0x) [stubbed]
- **Pitch**: Mapped to horizontal hand position (-12 to +12 semitones) [stubbed]

### PulseAudio Integration

The module uses PulseAudio commands for Linux audio control:

- `pactl set-sink-volume @DEFAULT_SINK@ <value>`: Set volume
- `pactl suspend-sink 0 0|1`: Resume/suspend playback

## Usage

### Basic Initialization

```python
from src.audio.audio_controller_module import AudioControlModule

# Create module
audio_module = AudioControlModule(
    volume_sensitivity=1.0,
    tempo_sensitivity=1.0,
    pitch_sensitivity=1.0,
    stub_dsp=True  # Stub tempo/pitch by default
)

# Initialize
if not audio_module.initialize():
    logger.error("Failed to initialize audio module")
    sys.exit(1)

# Use module...

# Cleanup when done
audio_module.cleanup()
```

### Integration with ModeRouter

```python
from src.core.state_manager import StateManager
from src.core.mode_router import ApplicationMode
from src.audio.audio_controller_module import AudioControlModule

# Create components
state_manager = StateManager()
audio_module = AudioControlModule()

# Initialize
state_manager.initialize()
audio_module.initialize()

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

# Switch to audio control mode
state_manager.set_mode(ApplicationMode.AUDIO_CONTROL)

# Now gestures will be routed to audio module
```

### Gesture Event Format

The AudioControlModule expects gesture events with the following data:

#### Volume Control (pinch gesture)
```python
{
    'pinch_distance': 0.0 - 0.2  # Normalized distance between thumb and index
}
```

#### Tempo Control (any gesture with hand position)
```python
{
    'vertical_position': 0.0 - 1.0  # 0.0 = top, 1.0 = bottom
}
```

#### Pitch Control (any gesture with hand position)
```python
{
    'horizontal_position': 0.0 - 1.0  # 0.0 = left, 1.0 = right
}
```

### Enhanced Gesture Recognition

The GestureRecognitionEngine has been extended to include continuous control data in gesture events:

```python
# In gesture_recognition_engine.py
def _extract_continuous_control_data(self, hand_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract continuous control data from hand landmarks.
    
    Returns:
        Dictionary with:
        - pinch_distance: Distance between thumb and index tips
        - vertical_position: Hand wrist y-coordinate (0.0-1.0)
        - horizontal_position: Hand wrist x-coordinate (0.0-1.0)
    """
```

This data is automatically included in all gesture events under `additional_data`.

## API Reference

### AudioControlModule

#### `__init__(volume_sensitivity=1.0, tempo_sensitivity=1.0, pitch_sensitivity=1.0, stub_dsp=True)`
Initialize the audio control module.

**Parameters:**
- `volume_sensitivity` (float): Multiplier for volume sensitivity (default: 1.0)
- `tempo_sensitivity` (float): Multiplier for tempo sensitivity (default: 1.0)
- `pitch_sensitivity` (float): Multiplier for pitch sensitivity (default: 1.0)
- `stub_dsp` (bool): If True, stub tempo/pitch controls (default: True)

#### `initialize() -> bool`
Initialize the module and start worker thread.

**Returns:** `True` if successful, `False` otherwise

#### `cleanup() -> None`
Clean up resources and stop worker thread.

#### Gesture Handlers

##### `handle_play_pause(data: Optional[Dict[str, Any]] = None) -> None`
Handle play/pause toggle gesture.

##### `handle_next_track(data: Optional[Dict[str, Any]] = None) -> None`
Handle next track gesture.

##### `handle_previous_track(data: Optional[Dict[str, Any]] = None) -> None`
Handle previous track gesture.

##### `handle_volume_control(data: Optional[Dict[str, Any]] = None) -> None`
Handle volume control via pinch distance.

**Required data:** `{'pinch_distance': float}`

##### `handle_tempo_control(data: Optional[Dict[str, Any]] = None) -> None`
Handle tempo control via vertical position. (Stubbed by default)

**Required data:** `{'vertical_position': float}`

##### `handle_pitch_control(data: Optional[Dict[str, Any]] = None) -> None`
Handle pitch control via horizontal position. (Stubbed by default)

**Required data:** `{'horizontal_position': float}`

#### State Access

##### `get_state() -> AudioState`
Get current audio state (thread-safe copy).

##### `get_volume() -> float`
Get current volume (0.0 to 1.0).

##### `get_tempo() -> float`
Get current tempo multiplier (0.5 to 2.0).

##### `get_pitch() -> float`
Get current pitch shift in semitones (-12 to +12).

##### `is_playing() -> bool`
Check if audio is currently playing.

### AudioState

Data class representing current audio state:

```python
@dataclass
class AudioState:
    volume: float = 0.5          # 0.0 to 1.0
    tempo: float = 1.0           # 0.5 to 2.0 (1.0 = normal)
    pitch: float = 0.0           # -12 to +12 semitones
    is_playing: bool = False
    current_track: Optional[str] = None
```

### AudioCommand

Enum for audio commands:

```python
class AudioCommand(Enum):
    PLAY = "play"
    PAUSE = "pause"
    NEXT_TRACK = "next_track"
    PREVIOUS_TRACK = "previous_track"
    SET_VOLUME = "set_volume"
    SET_TEMPO = "set_tempo"
    SET_PITCH = "set_pitch"
```

## Configuration

### Sensitivity Tuning

Adjust sensitivity multipliers to fine-tune control responsiveness:

```python
audio_module = AudioControlModule(
    volume_sensitivity=2.0,   # More sensitive volume control
    tempo_sensitivity=0.5,    # Less sensitive tempo control
    pitch_sensitivity=1.5     # More sensitive pitch control
)
```

### Enabling Real-time DSP

By default, tempo and pitch controls are stubbed. To enable them:

```python
audio_module = AudioControlModule(stub_dsp=False)
```

**Warning:** Real-time tempo/pitch DSP can be unstable and require additional audio processing libraries.

## PulseAudio Commands

The module uses these PulseAudio commands on Linux:

### Volume Control
```bash
pactl set-sink-volume @DEFAULT_SINK@ <0-65536>
# 0 = 0%, 65536 = 100%
```

### Playback Control
```bash
# Resume playback
pactl suspend-sink 0 0

# Suspend playback
pactl suspend-sink 0 1
```

### Check PulseAudio Status
```bash
pactl --version
pactl list sinks short
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_audio_controller_module.py -v
```

Run the demo:

```bash
python demo_audio_controller.py
```

## Thread Safety

The AudioControlModule is thread-safe:

- All state access is protected by locks
- Worker thread handles async command execution
- Non-blocking design ensures real-time responsiveness
- Queue-based command processing prevents race conditions

## Performance Considerations

- **Latency**: < 10ms gesture-to-command
- **CPU Usage**: Minimal (single worker thread)
- **Memory**: ~1KB state + command queue
- **PulseAudio calls**: Async, non-blocking

## Future Enhancements

1. **MPRIS Integration**: Control media players via D-Bus
2. **Real-time DSP**: Implement tempo/pitch with audio libraries
3. **Multi-sink Support**: Control multiple audio devices
4. **Audio Visualization**: Add real-time audio spectrum display
5. **Gesture Learning**: Customize gesture-to-action mappings

## Troubleshooting

### PulseAudio Not Available

**Symptom:** Warning: "PulseAudio not detected on system"

**Solution:**
1. Install PulseAudio: `sudo apt install pulseaudio`
2. Check status: `pactl --version`
3. Restart service: `systemctl --user restart pulseaudio`

### Commands Not Executing

**Issue:** Audio commands not having effect

**Check:**
1. Verify PulseAudio is running
2. Check default sink: `pactl list sinks short`
3. Test manually: `pactl set-sink-volume @DEFAULT_SINK@ 32768`

### Tempo/Pitch Not Working

This is expected - tempo and pitch are stubbed by default. Real-time DSP requires additional libraries and can be unstable.

## License

Same as project license (TBD)
