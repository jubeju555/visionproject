# Architecture Documentation

## Overview

Gesture Media Interface is a modular, production-grade Python application for real-time gesture-controlled multimedia and image manipulation.

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Application                          │
│                  (main.py - Coordinator)                     │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ VisionEngine │  │GestureEngine │  │  ModeRouter  │
│              │  │              │  │              │
│ - Camera     │  │ - MediaPipe  │  │ - State Mgmt │
│ - Threading  │  │ - Landmarks  │  │ - Event Queue│
└──────────────┘  └──────────────┘  └──────────────┘
                                           │
                    ┌──────────────────────┼──────────────────┐
                    ▼                      ▼                  ▼
            ┌──────────────┐      ┌──────────────┐  ┌──────────────┐
            │AudioController│     │ ImageEditor  │  │    AppUI     │
            │              │      │              │  │              │
            │ - Playback   │      │ - Transform  │  │ - Rendering  │
            │ - Volume     │      │ - Filters    │  │ - Display    │
            └──────────────┘      └──────────────┘  └──────────────┘
```

### Processing Pipeline

1. **Frame Capture** (VisionEngine)
   - Runs in separate thread
   - Captures frames from camera
   - Provides thread-safe frame access

2. **Hand Detection** (GestureEngine)
   - Processes frames with MediaPipe
   - Detects hand landmarks
   - Tracks hand position and orientation

3. **Gesture Classification** (GestureEngine)
   - Analyzes landmark patterns
   - Classifies recognized gestures
   - Provides confidence scores

4. **Command Routing** (ModeRouter)
   - Maintains application state
   - Routes gestures to handlers
   - Manages event queue

5. **Action Execution** (Action Modules)
   - AudioController: Controls media playback
   - ImageEditor: Applies image operations
   - Mode-specific actions

6. **UI Rendering** (AppUI)
   - Displays camera feed
   - Overlays hand landmarks
   - Shows mode indicators
   - Renders gesture feedback

## Module Details

### Core Module (`src/core/`)

Contains abstract base classes defining the system interfaces:

- **VisionEngine**: Camera input and frame capture
- **GestureEngine**: Hand tracking and gesture recognition
- **ModeRouter**: State management and command routing
- **AudioController**: Audio playback control
- **ImageEditor**: Image manipulation operations
- **AppUI**: User interface rendering

### Implementation Modules

- **src/vision/**: Camera capture implementation
- **src/gesture/**: Hand tracking and classification
- **src/audio/**: Audio playback
- **src/image/**: Image editing operations
- **src/ui/**: UI rendering

## Threading Model

The application uses a multithreaded architecture:

1. **Main Thread**: Coordinates subsystems and processes events
2. **Capture Thread**: Continuously captures frames from camera
3. **Event Queue**: Thread-safe communication between components

## Extension Points

The modular architecture allows easy extension:

1. Add new gesture types by extending GestureClassifier
2. Add new modes by extending ApplicationMode enum
3. Register new handlers in ModeRouter
4. Implement additional action modules

## Next Steps

See individual module documentation for implementation details.
