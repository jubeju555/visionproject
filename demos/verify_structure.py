#!/usr/bin/env python3
"""
Verification script for gesture-media-interface project structure.

This script verifies that all required files and directories exist
and that the project structure is correct.
"""

import os
import sys
from pathlib import Path


def check_structure():
    """Verify project structure."""
    print("Verifying gesture-media-interface project structure...\n")
    
    errors = []
    warnings = []
    
    # Required directories
    required_dirs = [
        'src',
        'src/core',
        'src/vision',
        'src/gesture',
        'src/audio',
        'src/image',
        'src/ui',
        'tests',
        'docs',
    ]
    
    # Required files
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        'src/__init__.py',
        'src/core/__init__.py',
        'src/core/vision_engine.py',
        'src/core/gesture_engine.py',
        'src/core/mode_router.py',
        'src/core/audio_controller.py',
        'src/core/image_editor.py',
        'src/core/app_ui.py',
        'src/core/state_manager.py',
        'src/vision/__init__.py',
        'src/vision/camera_capture.py',
        'src/gesture/__init__.py',
        'src/gesture/hand_tracker.py',
        'src/gesture/gesture_classifier.py',
        'src/audio/__init__.py',
        'src/audio/player.py',
        'src/image/__init__.py',
        'src/image/editor.py',
        'src/ui/__init__.py',
        'src/ui/renderer.py',
        'tests/__init__.py',
        'tests/test_core.py',
        'docs/ARCHITECTURE.md',
        'docs/SETUP.md',
        'docs/DEPENDENCIES.md',
    ]
    
    # Check directories
    print("Checking directories...")
    for directory in required_dirs:
        if not Path(directory).is_dir():
            errors.append(f"Missing directory: {directory}")
        else:
            print(f"  ✓ {directory}")
    
    print()
    
    # Check files
    print("Checking files...")
    for file in required_files:
        if not Path(file).is_file():
            errors.append(f"Missing file: {file}")
        else:
            print(f"  ✓ {file}")
    
    print()
    
    # Check Python syntax
    print("Checking Python syntax...")
    import py_compile
    for file in required_files:
        if file.endswith('.py'):
            try:
                py_compile.compile(file, doraise=True)
                print(f"  ✓ {file} - valid syntax")
            except py_compile.PyCompileError as e:
                errors.append(f"Syntax error in {file}: {e}")
    
    print()
    
    # Report results
    if errors:
        print("❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✅ All checks passed!")
    print(f"\nProject structure is valid with {len(required_files)} files and {len(required_dirs)} directories.")
    return True


if __name__ == "__main__":
    success = check_structure()
    sys.exit(0 if success else 1)
