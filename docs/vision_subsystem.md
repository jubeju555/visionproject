# Vision Subsystem Documentation

## Overview

The Vision Subsystem provides real-time webcam capture with hand landmark detection using MediaPipe Hands. It runs in a separate thread to ensure non-blocking operation and provides structured data through a thread-safe queue.

## Architecture

### Components

1. **MediaPipeVisionEngine**: Main vision engine implementation
   - Captures frames at 30 FPS (configurable)
   - Detects hand landmarks using MediaPipe
   - Runs in separate thread
   - Provides queue-based output

2. **VisionData**: Structured data container
   - Contains both BGR and RGB frames
   - Contains hand landmarks (up to 2 hands)
   - Includes timestamp for synchronization

### Thread Model

```
Main Thread                    Vision Thread
    |                               |
    | start_capture()              |
    |----------------------------->|
    |                              | [Capture Loop]
    |                              |  - Read frame from camera
    |                              |  - Convert BGR → RGB
    |                              |  - Process with MediaPipe
    |                              |  - Extract landmarks
    |                              |  - Apply smoothing (optional)
    |                              |  - Push to queue
    |                              |     ↓
    | get_vision_data()            |
    |<-----------------------------|
    |                              |
    | stop_capture()               |
    |----------------------------->|
    |                              | [Exit]
```

## Usage

### Basic Usage

```python
from src.vision import MediaPipeVisionEngine, VisionData

# Create engine
engine = MediaPipeVisionEngine(
    camera_id=0,
    fps=30,
    max_num_hands=2,
    enable_smoothing=True
)

# Initialize
if engine.initialize():
    # Start capture thread
    engine.start_capture()
    
    # Main processing loop
    while True:
        # Get vision data from queue
        vision_data = engine.get_vision_data(timeout=0.1)
        
        if vision_data:
            frame = vision_data.frame
            landmarks = vision_data.landmarks
            
            # Process landmarks
            for hand_data in landmarks:
                handedness = hand_data['handedness']  # 'Left' or 'Right'
                confidence = hand_data['confidence']
                hand_landmarks = hand_data['landmarks']  # 21 landmarks
                
                # Each landmark has x, y, z coordinates
                for lm in hand_landmarks:
                    x, y, z = lm['x'], lm['y'], lm['z']
                    # Process landmark...
        
        # Exit condition
        if should_exit:
            break
    
    # Cleanup
    engine.stop_capture()
    engine.cleanup()
```

### Configuration Options

```python
engine = MediaPipeVisionEngine(
    camera_id=0,                      # Camera device ID
    fps=30,                           # Target frames per second
    max_queue_size=2,                 # Max queue size before dropping frames
    enable_smoothing=True,            # Enable exponential smoothing
    smoothing_factor=0.5,             # Smoothing factor (0-1)
    max_num_hands=2,                  # Max hands to detect (1 or 2)
    min_detection_confidence=0.5,     # Min confidence for detection
    min_tracking_confidence=0.5       # Min confidence for tracking
)
```

### Landmark Structure

Each detected hand provides:

```python
{
    'handedness': 'Right',           # 'Left' or 'Right'
    'confidence': 0.95,              # Detection confidence (0-1)
    'landmarks': [                   # 21 landmarks in pixel coordinates
        {'x': 320, 'y': 240, 'z': 0.0},
        {'x': 325, 'y': 245, 'z': 0.01},
        # ... 19 more landmarks
    ],
    'landmarks_normalized': [        # 21 landmarks normalized (0-1)
        {'x': 0.5, 'y': 0.5, 'z': 0.0},
        {'x': 0.51, 'y': 0.51, 'z': 0.01},
        # ... 19 more landmarks
    ]
}
```

### Landmark Indices

MediaPipe Hands provides 21 landmarks per hand:

- **0**: Wrist
- **1-4**: Thumb (from palm to tip)
- **5-8**: Index finger (from palm to tip)
- **9-12**: Middle finger (from palm to tip)
- **13-16**: Ring finger (from palm to tip)
- **17-20**: Pinky finger (from palm to tip)

## Features

### Frame Capture
- Captures frames at specified FPS (default 30)
- Automatic BGR to RGB conversion for MediaPipe
- Thread-safe frame access via `get_frame()`

### Hand Detection
- Detects up to 2 hands simultaneously
- 21 landmarks per hand
- Both pixel and normalized coordinates
- Left/right hand classification
- Confidence scores for each detection

### Smoothing
- Optional exponential smoothing for stable tracking
- Configurable smoothing factor
- Applied to both pixel and normalized coordinates
- Formula: `smoothed = alpha * current + (1 - alpha) * previous`
  where `alpha = 1 - smoothing_factor`

### Thread Safety
- Runs in separate thread for non-blocking operation
- Thread-safe queue for output data
- Thread-safe frame access with locks
- Graceful shutdown with timeout

### Queue Management
- Configurable queue size
- Automatic dropping of old frames when queue is full
- Non-blocking data retrieval with timeout

## Integration with Main Application

The Vision Engine can be used in two ways:

### Option 1: Direct Usage (Recommended for new code)

```python
from src.vision import MediaPipeVisionEngine

class GestureMediaInterface:
    def initialize(self):
        self.vision_engine = MediaPipeVisionEngine(
            camera_id=0,
            fps=30,
            enable_smoothing=True
        )
        return self.vision_engine.initialize()
    
    def run(self):
        self.vision_engine.start_capture()
        
        while self._running:
            vision_data = self.vision_engine.get_vision_data(timeout=0.1)
            if vision_data:
                # Process vision data
                self.process_vision_data(vision_data)
```

### Option 2: Via Abstract Interface

The MediaPipeVisionEngine implements the VisionEngine abstract interface, so it can be used wherever a VisionEngine is expected.

## Performance

- **Typical FPS**: 25-30 FPS on modern hardware
- **Latency**: < 50ms from capture to landmark output
- **CPU Usage**: Moderate (MediaPipe is CPU-optimized)
- **Memory**: ~50-100 MB depending on resolution

## Error Handling

The engine handles common errors gracefully:

- Camera not available: `initialize()` returns `False`
- Frame read failure: Logs warning, continues operation
- MediaPipe errors: Logs error, returns empty landmarks
- Thread shutdown: Ensures clean resource cleanup

## Testing

Comprehensive unit tests are provided in `tests/test_vision_engine.py`:

```bash
# Run all vision tests
python -m pytest tests/test_vision_engine.py -v

# Run specific test
python -m pytest tests/test_vision_engine.py::TestMediaPipeVisionEngine::test_initialize_success -v
```

## Demo

A demo script is provided to test the vision engine:

```bash
python demo_vision_engine.py
```

Features:
- Real-time hand landmark visualization
- FPS counter
- Hand count display
- Toggle smoothing with 's' key
- Quit with 'q' key

## Troubleshooting

### Camera not opening
- Check camera permissions
- Verify camera is not in use by another application
- Try different camera_id (0, 1, 2, etc.)

### Low FPS
- Reduce target FPS
- Reduce max_num_hands to 1
- Lower camera resolution (via cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)

### Landmarks not detected
- Ensure good lighting
- Place hands clearly in view
- Lower min_detection_confidence
- Check that camera is working

### Import errors
- Ensure MediaPipe 0.10.13 is installed: `pip install mediapipe==0.10.13`
- Verify all dependencies: `pip install -r requirements.txt`

## Dependencies

- Python 3.11+
- OpenCV >= 4.8.0
- MediaPipe == 0.10.13 (locked for API stability)
- NumPy >= 1.24.0

## Future Enhancements

Potential improvements for future versions:

1. **GPU Acceleration**: Add GPU support for better performance
2. **Multi-camera Support**: Support multiple cameras simultaneously
3. **Recording**: Add capability to record vision data
4. **Calibration**: Add camera calibration support
5. **Advanced Smoothing**: Additional smoothing algorithms (Kalman filter, etc.)
6. **Pose Detection**: Extend to full body pose detection
7. **Face Detection**: Add face landmark detection

## Notes

- **No Gesture Classification**: This module only detects landmarks. Gesture classification is the responsibility of the GestureEngine.
- **Thread Safety**: All public methods are thread-safe and can be called from any thread.
- **Resource Cleanup**: Always call `cleanup()` to release camera and MediaPipe resources.
- **MediaPipe Version**: Locked to 0.10.13 for API stability (uses `solutions` API).
