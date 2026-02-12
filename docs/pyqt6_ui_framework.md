# PyQt6 UI Framework Documentation

## Overview

This document describes the PyQt6-based UI framework implementation for the gesture-media-interface project.

## Architecture

The PyQt6 UI system follows a modern, thread-safe architecture that ensures:
- Non-blocking UI updates
- Thread-safe signal/slot communication
- Smooth frame rate performance
- Clean separation of concerns

### Component Hierarchy

```
PyQt6UI (AppUI implementation)
  └── PyQt6MainWindow (QMainWindow)
      ├── CameraWidget (QLabel)
      │   └── Live camera feed with landmark overlay
      ├── StatusPanel (QGroupBox)
      │   ├── Current Mode Indicator
      │   ├── Detected Gesture Label
      │   ├── Confidence Score
      │   ├── Audio Parameters (stubbed)
      │   ├── Image Editing Status (stubbed)
      │   ├── FPS Display
      │   ├── Latency Estimate
      │   └── Hands Detected Count
      └── ControlsPanel (QGroupBox)
          ├── Debug Mode Toggle
          ├── Smoothing Toggle
          └── Reset System State Button
```

### Threading Architecture

```
Main Thread (UI)
  ├── PyQt6MainWindow
  │   ├── Renders UI
  │   ├── Handles user input
  │   └── Updates display widgets
  │
  └── VisionWorker (QThread)
      ├── Runs in separate thread
      ├── Gets VisionData from engine
      └── Emits signals to main thread
          └── vision_data_ready signal
              └── Handled by _on_vision_data slot
```

## Key Components

### 1. PyQt6UI

Main implementation of the `AppUI` interface using PyQt6.

**Features:**
- Implements AppUI abstract interface
- Creates and manages PyQt6MainWindow
- Initializes VisionEngine if not provided
- Provides clean initialization and cleanup

**Usage:**
```python
from src.ui.pyqt6_ui import PyQt6UI

# Create UI
ui = PyQt6UI()

# Initialize
if ui.initialize():
    # Run UI (blocks until window closed)
    ui.run()
    
    # Cleanup
    ui.cleanup()
```

### 2. PyQt6MainWindow

Main window class that manages all UI components.

**Features:**
- Left panel: Live camera feed with landmark overlay
- Right panel: Status display and controls
- Thread-safe frame updates via Qt signals/slots
- FPS and latency tracking
- Interactive controls with signal-based communication

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│                  Gesture Media Interface                    │
├───────────────────────────┬─────────────────────────────────┤
│                           │   System Status                 │
│                           ├─────────────────────────────────┤
│                           │ Current Mode: IDLE              │
│                           │                                 │
│    Camera Feed            │ Detected Gesture: None          │
│    (640x480)              │ Confidence: 0.00                │
│                           │                                 │
│    With hand landmarks    │ Audio Parameters:               │
│    overlay                │   Volume: --                    │
│                           │   Tempo: --                     │
│                           │   Pitch: --                     │
│                           │                                 │
│                           │ Image Editing: Not Active       │
│                           │                                 │
│                           │ FPS: 30.0                       │
│                           │ Latency: 33 ms                  │
│                           │ Hands Detected: 2               │
│                           ├─────────────────────────────────┤
│                           │   Controls                      │
│                           ├─────────────────────────────────┤
│                           │ [Toggle Debug Mode]             │
│                           │ [Toggle Smoothing]              │
│                           │ [Reset System State]            │
│                           │                                 │
│                           │ Status: Ready                   │
└───────────────────────────┴─────────────────────────────────┘
```

### 3. VisionWorker

Worker thread that interfaces with VisionEngine.

**Features:**
- Runs in separate QThread
- Non-blocking frame capture
- Emits Qt signals for thread-safe communication
- Handles errors gracefully

**Signal Flow:**
```
VisionEngine (separate thread)
  → VisionWorker.run()
    → vision_data = engine.get_vision_data()
    → vision_data_ready.emit(vision_data)
      → PyQt6MainWindow._on_vision_data(vision_data)
        → Update camera widget
        → Update status panel
        → Calculate FPS/latency
```

### 4. CameraWidget

Displays live camera feed with hand landmarks.

**Features:**
- QLabel-based display
- Converts BGR (OpenCV) to RGB (Qt)
- Scales to fit widget size
- Minimum size: 640x480

**Methods:**
- `display_frame(frame)`: Display a BGR frame from OpenCV

### 5. StatusPanel

Displays system status and metrics.

**Features:**
- Current mode indicator
- Detected gesture and confidence
- Audio parameters (stubbed for now)
- Image editing status (stubbed for now)
- FPS display
- Latency estimate
- Hands detected count

**Methods:**
- `update_mode(mode)`: Update current mode
- `update_gesture(gesture, confidence)`: Update gesture info
- `update_fps(fps)`: Update FPS display
- `update_latency(latency_ms)`: Update latency
- `update_hands(count)`: Update hands count

### 6. ControlsPanel

Interactive controls for the system.

**Features:**
- Debug mode toggle (changes landmark color)
- Smoothing toggle (enables/disables landmark smoothing)
- Reset system state button
- Status label showing control actions

**Signals:**
- `debug_toggled(bool)`: Emitted when debug mode changes
- `smoothing_toggled(bool)`: Emitted when smoothing changes
- `reset_requested()`: Emitted when reset button clicked

## Thread Safety

The UI system ensures thread safety through:

1. **Qt Signal/Slot Mechanism**: All cross-thread communication uses Qt's signal/slot system
2. **Worker Thread**: VisionEngine runs in separate thread, never blocks UI
3. **Thread-Safe Engine Methods**: VisionEngine provides thread-safe methods (e.g., `set_smoothing()`)
4. **Frame Copying**: Frames are copied before display to prevent race conditions

## Performance

### Target Metrics
- **FPS**: 25-30 FPS (camera-limited)
- **Latency**: < 50ms (capture to display)
- **UI Responsiveness**: 60 FPS UI updates

### Optimization Techniques
1. **Queue-based data transfer**: Prevents blocking
2. **Frame dropping**: Old frames dropped if queue full
3. **Efficient rendering**: QPixmap caching
4. **Minimal processing in UI thread**: Heavy lifting in worker thread

## Usage Examples

### Basic Usage

```python
from src.ui.pyqt6_ui import PyQt6UI

# Create and run UI
ui = PyQt6UI()
if ui.initialize():
    ui.run()  # Blocks until window closed
    ui.cleanup()
```

### With Custom VisionEngine

```python
from src.ui.pyqt6_ui import PyQt6UI
from src.vision.vision_engine_impl import MediaPipeVisionEngine

# Create custom vision engine
engine = MediaPipeVisionEngine(
    camera_id=0,
    fps=30,
    max_num_hands=2,
    enable_smoothing=True,
    smoothing_factor=0.5
)

# Initialize engine
engine.initialize()

# Create UI with engine
ui = PyQt6UI(vision_engine=engine)
if ui.initialize():
    ui.run()
    ui.cleanup()
```

### Updating Mode and Gesture

```python
# Update mode display
frame = get_current_frame()
ui.display_mode(frame, "AUDIO")

# Update gesture display
ui.display_gesture(frame, "Thumbs Up")
```

## Demo Application

A demo application is provided in `demo_pyqt6_ui.py`:

```bash
python demo_pyqt6_ui.py
```

This demonstrates:
- Live camera feed
- Hand landmark detection
- Real-time FPS display
- Interactive controls
- Thread-safe operation

## Requirements

### Python Packages
- PyQt6 >= 6.6.0
- mediapipe == 0.10.13
- opencv-python >= 4.8.0
- numpy >= 1.24.0

### System Dependencies (Linux)
```bash
sudo apt-get install -y \
    libegl1 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxcb-cursor0 \
    x11-utils
```

## Testing

### Running Tests

```bash
# Run with virtual display (headless)
xvfb-run -a python -m pytest tests/test_pyqt6_ui.py -v

# Run all tests
python -m pytest tests/ -v
```

### Test Coverage
- **25 unit tests** covering all major components
- CameraWidget: Display and rendering
- StatusPanel: All update methods
- ControlsPanel: Signal emission
- PyQt6MainWindow: Component integration
- VisionWorker: Threading behavior
- PyQt6UI: Initialization and cleanup

## Known Limitations

1. **Headless Environments**: Requires X11 display or Xvfb for testing
2. **Camera Access**: Requires camera hardware or mocked camera
3. **Audio/Image Editing**: Currently stubbed (not implemented)
4. **Multi-monitor**: Not explicitly tested

## Future Enhancements

1. **Audio Controls**: Implement audio parameter displays and controls
2. **Image Editing**: Implement image editing status and controls
3. **Gesture Visualization**: Enhanced gesture visualization
4. **Recording**: Add video recording capability
5. **Configuration UI**: Add settings dialog
6. **Keyboard Shortcuts**: Add hotkeys for common actions
7. **Theme Support**: Add dark/light theme toggle

## Troubleshooting

### Issue: "libEGL.so.1: cannot open shared object file"
**Solution**: Install system dependencies:
```bash
sudo apt-get install -y libegl1
```

### Issue: Tests crash with "Aborted (core dumped)"
**Solution**: Run tests with xvfb:
```bash
xvfb-run -a python -m pytest tests/test_pyqt6_ui.py -v
```

### Issue: Low FPS
**Solution**: 
- Check camera FPS: `v4l2-ctl --device=/dev/video0 --all`
- Reduce resolution if needed
- Disable smoothing if too slow

### Issue: High Latency
**Solution**:
- Reduce queue size: `max_queue_size=1`
- Increase capture thread priority
- Check system load

## API Reference

### PyQt6UI

```python
class PyQt6UI(AppUI):
    def __init__(self, vision_engine: Optional[MediaPipeVisionEngine] = None)
    def initialize(self) -> bool
    def render_frame(self, frame: np.ndarray) -> np.ndarray
    def draw_landmarks(self, frame: np.ndarray, landmarks: Any) -> np.ndarray
    def display_mode(self, frame: np.ndarray, mode: str) -> np.ndarray
    def display_gesture(self, frame: np.ndarray, gesture: str) -> np.ndarray
    def display_message(self, frame: np.ndarray, message: str) -> np.ndarray
    def show_frame(self, frame: np.ndarray) -> None
    def handle_key_input(self) -> Optional[str]
    def run(self)
    def cleanup(self) -> None
```

### PyQt6MainWindow

```python
class PyQt6MainWindow(QMainWindow):
    def __init__(self, vision_engine: MediaPipeVisionEngine)
    def start_vision_engine(self)
    def stop_vision_engine(self)
    def _on_vision_data(self, vision_data: VisionData)  # Slot
    def _on_debug_toggled(self, enabled: bool)  # Slot
    def _on_smoothing_toggled(self, enabled: bool)  # Slot
    def _on_reset_requested(self)  # Slot
```

### VisionWorker

```python
class VisionWorker(QThread):
    vision_data_ready = pyqtSignal(object)  # VisionData
    error_occurred = pyqtSignal(str)
    
    def __init__(self, vision_engine: MediaPipeVisionEngine)
    def run(self)
    def stop(self)
```

## Conclusion

The PyQt6 UI Framework provides a modern, thread-safe, and performant user interface for the gesture-media-interface project. It successfully integrates with the VisionEngine to provide real-time hand tracking visualization with interactive controls and comprehensive status displays.

All requirements from SEGMENT 3 have been met:
- ✅ Main window with left and right panels
- ✅ Live camera feed with hand landmark overlay
- ✅ Status displays (mode, gesture, confidence, FPS, latency)
- ✅ Interactive controls (debug, smoothing, reset)
- ✅ Thread-safe signal/slot communication
- ✅ Non-blocking capture thread
- ✅ Frame rate updates
- ✅ Stub gesture/mode data
