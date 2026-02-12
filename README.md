# Gesture Media Interface

A production-grade, modular Python application for real-time gesture-controlled multimedia and image manipulation.

## Overview

This system provides a comprehensive gesture control interface for multimedia playback and image manipulation using computer vision and hand tracking technology.

## Architecture

### High-Level Pipeline

```
Camera Input
  → Frame Capture Thread
  → Hand Landmark Detection (MediaPipe)
  → Gesture Classification Engine
  → State Manager / Mode Router
  → Action Modules
  → UI Rendering Layer
```

## Project Structure

```
gesture-media-interface/
├── src/
│   ├── core/          # Core abstract interfaces and base classes
│   ├── vision/        # Camera input and frame capture
│   ├── gesture/       # Hand landmark detection and gesture classification
│   ├── audio/         # Audio playback and control
│   ├── image/         # Image manipulation operations
│   └── ui/            # UI rendering layer
├── tests/             # Unit and integration tests
├── docs/              # Documentation
├── main.py            # Application entry point
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Requirements

- Python 3.11+
- MediaPipe for hand tracking
- OpenCV for video capture and rendering
- Modern Linux system with camera access

## Installation

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Features

### Modular Architecture
- Clean separation of concerns
- Abstract interfaces for extensibility
- Thread-safe event dispatch via queue
- Multithreaded processing pipeline

### Core Components

1. **VisionEngine**: Handles camera input and frame capture
2. **GestureEngine**: Processes hand landmarks and classifies gestures
3. **ModeRouter**: Manages application state and routes commands
4. **AudioController**: Controls audio playback
5. **ImageEditor**: Performs image manipulation operations
6. **AppUI**: Renders the user interface

## Development

This project follows a modular architecture with clean boundaries between subsystems. Each module is independently testable and can be extended without affecting other components.

## License

TBD

## Contributors

TBD
