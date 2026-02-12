# Module Dependencies

## Import Hierarchy

```
main.py
├── src.core (Abstract Interfaces)
│   ├── VisionEngine
│   ├── GestureEngine
│   ├── ModeRouter & ApplicationMode
│   ├── AudioController & PlaybackState
│   ├── ImageEditor & ImageOperation
│   ├── AppUI
│   └── StateManager (Implementation)
│
├── src.vision (Vision Module)
│   └── CameraCapture → implements VisionEngine
│
├── src.gesture (Gesture Module)
│   ├── HandTracker → implements GestureEngine
│   └── GestureClassifier (Helper)
│
├── src.audio (Audio Module)
│   └── AudioPlayer → implements AudioController
│
├── src.image (Image Module)
│   └── ImageManipulator → implements ImageEditor
│
└── src.ui (UI Module)
    └── UIRenderer → implements AppUI
```

## Dependency Rules

### Core Module
- **No dependencies** on implementation modules
- Only external dependencies: numpy, abc, typing, enum, queue, threading
- Defines all abstract interfaces

### Implementation Modules
- **Only depend on** core interfaces
- Can use external libraries (OpenCV, MediaPipe, pygame, PIL)
- No cross-dependencies between implementation modules

### Main Application
- Imports from both core and implementation modules
- Wires everything together
- Manages lifecycle of all components

## Clean Boundaries

✅ **Vision** → Only imports from core.vision_engine
✅ **Gesture** → Only imports from core.gesture_engine  
✅ **Audio** → Only imports from core.audio_controller
✅ **Image** → Only imports from core.image_editor
✅ **UI** → Only imports from core.app_ui

This ensures:
- Easy testing (mock interfaces)
- Loose coupling
- Independent development
- Clear contracts between modules

## Extension Points

To add new functionality:

1. **New Gesture Type**: Extend GestureClassifier
2. **New Mode**: Add to ApplicationMode enum, register handlers
3. **New Action Module**: Create new interface in core, implement in separate module
4. **New UI Elements**: Extend UIRenderer methods

## Thread Safety

- VisionEngine: Thread-safe frame capture
- ModeRouter: Thread-safe event queue
- All components: Can be called from multiple threads via ModeRouter
