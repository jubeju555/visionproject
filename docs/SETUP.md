# Setup Guide

## Prerequisites

- Python 3.11 or higher
- Linux operating system
- Camera device (webcam)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jubeju555/visionproject.git
cd visionproject
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure

```
gesture-media-interface/
├── src/                    # Source code
│   ├── core/              # Core abstract interfaces
│   │   ├── vision_engine.py
│   │   ├── gesture_engine.py
│   │   ├── mode_router.py
│   │   ├── audio_controller.py
│   │   ├── image_editor.py
│   │   ├── app_ui.py
│   │   └── state_manager.py
│   ├── vision/            # Camera capture implementation
│   ├── gesture/           # Hand tracking and classification
│   ├── audio/             # Audio playback
│   ├── image/             # Image editing
│   └── ui/                # UI rendering
├── tests/                 # Test suite
├── docs/                  # Documentation
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
└── README.md            # Project overview
```

## Running the Application

Currently, the application is in scaffold mode. The main components are defined but not yet fully implemented.

To verify the scaffold:

```bash
# Activate virtual environment
source venv/bin/activate

# Run syntax check
python3 -m py_compile main.py

# Run tests (once dependencies are installed)
pytest tests/
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## Next Steps

The current implementation includes:
- ✅ Complete folder structure
- ✅ Abstract interfaces for all components
- ✅ Skeleton implementations with docstrings
- ✅ Main application wiring
- ✅ Documentation structure

To be implemented:
- [ ] Camera capture functionality
- [ ] MediaPipe hand tracking
- [ ] Gesture classification logic
- [ ] Audio playback system
- [ ] Image manipulation operations
- [ ] UI rendering
- [ ] Event processing loop
- [ ] Comprehensive tests

## Contributing

(To be added)

## License

(To be added)
