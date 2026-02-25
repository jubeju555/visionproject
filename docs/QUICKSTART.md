# Quick Start Guide - Vision Subsystem

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.vision import MediaPipeVisionEngine; print('✅ Vision Subsystem ready!')"
```

## Basic Usage

### Simple Example

```python
from src.vision import MediaPipeVisionEngine

# Create engine
engine = MediaPipeVisionEngine(camera_id=0, fps=30, max_num_hands=2)

# Initialize
if not engine.initialize():
    print("Failed to initialize camera")
    exit(1)

# Start capture
engine.start_capture()

# Main loop
try:
    while True:
        # Get latest data
        vision_data = engine.get_vision_data(timeout=0.1)
        
        if vision_data:
            print(f"Detected {len(vision_data.landmarks)} hand(s)")
            
            for hand in vision_data.landmarks:
                print(f"  {hand['handedness']} hand ({hand['confidence']:.2f})")
                # Process 21 landmarks...
                
finally:
    # Always cleanup
    engine.cleanup()
```

### With Smoothing

```python
engine = MediaPipeVisionEngine(
    enable_smoothing=True,    # Enable smoothing
    smoothing_factor=0.5      # Higher = more smoothing
)
```

### Access Individual Frame

```python
# Get just the frame without landmark processing
frame = engine.get_frame()
if frame is not None:
    cv2.imshow("Frame", frame)
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/test_vision_engine.py -v

# Run specific test
python -m pytest tests/test_vision_engine.py::TestMediaPipeVisionEngine::test_initialize_success -v

# Run with coverage
python -m pytest tests/test_vision_engine.py --cov=src.vision --cov-report=html
```

## Running Demo

```bash
# Run interactive demo
python demo_vision_engine.py

# Controls:
# - 'q': Quit
# - 's': Toggle smoothing
```

## Common Issues

### Camera not opening
```python
# Try different camera IDs
for cam_id in range(5):
    engine = MediaPipeVisionEngine(camera_id=cam_id)
    if engine.initialize():
        print(f"Camera {cam_id} works!")
        break
```

### Low FPS
```python
# Reduce to 1 hand for better performance
engine = MediaPipeVisionEngine(max_num_hands=1)
```

### Import Errors
```bash
# Make sure you're in the project root
cd /path/to/visionproject

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Integration Example

```python
from src.vision import MediaPipeVisionEngine
from src.gesture import HandTracker  # Future segment

class MyApp:
    def __init__(self):
        self.vision = MediaPipeVisionEngine()
        self.gesture = HandTracker()
        
    def run(self):
        self.vision.initialize()
        self.vision.start_capture()
        
        while True:
            vision_data = self.vision.get_vision_data()
            if vision_data:
                # Pass landmarks to gesture engine
                gesture = self.gesture.process(vision_data.landmarks)
                # Handle gesture...
```

## API Reference

### MediaPipeVisionEngine

**Constructor Parameters:**
- `camera_id`: int = 0
- `fps`: int = 30
- `max_queue_size`: int = 2
- `enable_smoothing`: bool = False
- `smoothing_factor`: float = 0.5
- `max_num_hands`: int = 2
- `min_detection_confidence`: float = 0.5
- `min_tracking_confidence`: float = 0.5

**Methods:**
- `initialize() -> bool`: Initialize camera and MediaPipe
- `start_capture() -> None`: Start capture thread
- `stop_capture() -> None`: Stop capture thread
- `get_frame() -> Optional[np.ndarray]`: Get latest frame
- `get_vision_data(timeout=None) -> Optional[VisionData]`: Get vision data from queue
- `set_smoothing(enabled: bool) -> None`: Toggle smoothing (thread-safe)
- `is_running() -> bool`: Check if capture thread is running
- `cleanup() -> None`: Release all resources

### VisionData

**Attributes:**
- `frame`: BGR frame (numpy array)
- `frame_rgb`: RGB frame (numpy array)
- `landmarks`: List of hand data dictionaries
- `timestamp`: Frame timestamp

**Landmark Structure:**
```python
{
    'handedness': 'Right' | 'Left',
    'confidence': 0.0 - 1.0,
    'landmarks': [                    # 21 landmarks
        {'x': int, 'y': int, 'z': float},
        ...
    ],
    'landmarks_normalized': [         # 21 normalized
        {'x': float, 'y': float, 'z': float},
        ...
    ]
}
```

## Performance Tips

1. **Lower FPS**: Set `fps=20` for slower systems
2. **One hand**: Set `max_num_hands=1` if you don't need dual-hand
3. **Lower confidence**: Set `min_detection_confidence=0.3` for easier detection
4. **Disable smoothing**: Set `enable_smoothing=False` for lowest latency

## Documentation

- Full docs: `docs/vision_subsystem.md`
- Implementation summary: `IMPLEMENTATION_SUMMARY.md`
- Source code: `src/vision/vision_engine_impl.py`
- Tests: `tests/test_vision_engine.py`

## Support

For issues or questions:
1. Check `docs/vision_subsystem.md` for troubleshooting
2. Review test cases in `tests/test_vision_engine.py`
3. Run the demo to verify setup: `python demo_vision_engine.py`
