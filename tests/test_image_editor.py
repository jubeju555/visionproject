"""
Tests for ImageEditor implementation.

Tests the gesture-controlled image editing module including:
- Frame capture and freezing
- Layer management (base, selection, transform)
- Transform operations (translate, rotate, scale)
- Brightness/contrast adjustments
- Undo/redo functionality
- Filter application
- Thread safety
"""

import pytest
import numpy as np
import cv2
import threading
import time
from pathlib import Path

from src.image.editor import ImageManipulator, TransformType, EditorState


class TestImageEditorInitialization:
    """Test ImageEditor initialization and basic setup."""
    
    def test_initialization(self):
        """Test basic initialization."""
        editor = ImageManipulator()
        assert editor.initialize() is True
        editor.cleanup()
    
    def test_multiple_initialization(self):
        """Test that multiple initializations work correctly."""
        editor = ImageManipulator()
        assert editor.initialize() is True
        assert editor.initialize() is True  # Should work
        editor.cleanup()
    
    def test_get_image_before_load(self):
        """Test that get_image returns None before any image is loaded."""
        editor = ImageManipulator()
        editor.initialize()
        assert editor.get_image() is None
        editor.cleanup()


class TestFrameCapture:
    """Test freeze frame capture functionality."""
    
    def test_freeze_frame_basic(self):
        """Test basic freeze frame capture."""
        editor = ImageManipulator()
        editor.initialize()
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Freeze frame
        assert editor.freeze_frame(frame) is True
        
        # Get image
        img = editor.get_image()
        assert img is not None
        assert img.shape == frame.shape
        
        editor.cleanup()
    
    def test_freeze_frame_none(self):
        """Test that freezing None frame fails gracefully."""
        editor = ImageManipulator()
        editor.initialize()
        
        assert editor.freeze_frame(None) is False
        
        editor.cleanup()
    
    def test_freeze_frame_empty(self):
        """Test that freezing empty frame fails gracefully."""
        editor = ImageManipulator()
        editor.initialize()
        
        empty_frame = np.array([])
        assert editor.freeze_frame(empty_frame) is False
        
        editor.cleanup()
    
    def test_freeze_frame_updates_image(self):
        """Test that freezing a new frame updates the current image."""
        editor = ImageManipulator()
        editor.initialize()
        
        # First frame
        frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 50
        editor.freeze_frame(frame1)
        img1 = editor.get_image()
        
        # Second frame
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        editor.freeze_frame(frame2)
        img2 = editor.get_image()
        
        # Images should be different
        assert not np.array_equal(img1, img2)
        
        editor.cleanup()


class TestLayerManagement:
    """Test layer management functionality."""
    
    def test_get_layers(self):
        """Test getting all layers."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        layers = editor.get_layers()
        
        assert 'base' in layers
        assert 'transform' in layers
        assert 'selection_mask' in layers
        assert layers['base'] is not None
        assert layers['selection_mask'] is not None
        
        editor.cleanup()
    
    def test_selection_mask(self):
        """Test setting selection mask."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Create custom mask (circle selection)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)
        
        assert editor.set_selection_mask(mask) is True
        
        layers = editor.get_layers()
        assert layers['selection_mask'] is not None
        
        editor.cleanup()
    
    def test_selection_mask_wrong_size(self):
        """Test that wrong-sized mask is rejected."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Wrong size mask
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert editor.set_selection_mask(mask) is False
        
        editor.cleanup()


class TestTransformOperations:
    """Test geometric transform operations."""
    
    def test_translate(self):
        """Test translation operation."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Apply translation
        editor.translate(10, 20)
        
        # Get transform matrix
        matrix = editor.get_transform_matrix()
        assert matrix is not None
        assert matrix.shape == (3, 3)
        
        # Check translation components
        assert matrix[0, 2] == 10  # dx
        assert matrix[1, 2] == 20  # dy
        
        editor.cleanup()
    
    def test_rotate(self):
        """Test rotation operation."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Apply rotation
        editor.rotate(45)
        
        # Image should still be retrievable
        img = editor.get_image()
        assert img is not None
        assert img.shape == frame.shape
        
        editor.cleanup()
    
    def test_scale(self):
        """Test scaling operation."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Apply scale
        editor.scale(1.5)
        
        # Image should still be retrievable
        img = editor.get_image()
        assert img is not None
        
        editor.cleanup()
    
    def test_multiple_transforms(self):
        """Test applying multiple transforms in sequence."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Apply multiple transforms
        editor.translate(10, 0)
        editor.rotate(30)
        editor.scale(1.2)
        
        # Image should still be retrievable
        img = editor.get_image()
        assert img is not None
        
        editor.cleanup()
    
    def test_transform_matrix_composition(self):
        """Test that transforms compose correctly."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Get initial matrix (identity)
        initial_matrix = editor.get_transform_matrix()
        assert np.allclose(initial_matrix, np.eye(3))
        
        # Apply translation
        editor.translate(5, 5)
        
        # Matrix should have changed
        new_matrix = editor.get_transform_matrix()
        assert not np.allclose(new_matrix, initial_matrix)
        
        editor.cleanup()


class TestBrightnessContrast:
    """Test brightness and contrast adjustments."""
    
    def test_brightness_increase(self):
        """Test increasing brightness."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
        editor.freeze_frame(frame)
        
        img_before = editor.get_image()
        
        # Increase brightness
        editor.adjust_brightness(0.5)
        
        img_after = editor.get_image()
        assert img_after is not None
        
        # Average brightness should increase
        assert np.mean(img_after) > np.mean(img_before)
        
        editor.cleanup()
    
    def test_brightness_decrease(self):
        """Test decreasing brightness."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
        editor.freeze_frame(frame)
        
        img_before = editor.get_image()
        
        # Decrease brightness
        editor.adjust_brightness(-0.5)
        
        img_after = editor.get_image()
        assert img_after is not None
        
        # Average brightness should decrease
        assert np.mean(img_after) < np.mean(img_before)
        
        editor.cleanup()
    
    def test_brightness_clamp(self):
        """Test that brightness values are clamped."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
        editor.freeze_frame(frame)
        
        # Try extreme values (should be clamped)
        editor.adjust_brightness(2.0)  # Should clamp to 1.0
        img1 = editor.get_image()
        
        editor.freeze_frame(frame)
        editor.adjust_brightness(-2.0)  # Should clamp to -1.0
        img2 = editor.get_image()
        
        assert img1 is not None
        assert img2 is not None
        
        editor.cleanup()
    
    def test_contrast(self):
        """Test contrast adjustment."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(50, 150, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Adjust contrast
        editor.adjust_contrast(1.5)
        
        img = editor.get_image()
        assert img is not None
        
        editor.cleanup()


class TestCropOperation:
    """Test image cropping."""
    
    def test_crop_basic(self):
        """Test basic crop operation."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Crop to smaller region
        editor.crop(25, 25, 50, 50)
        
        img = editor.get_image()
        assert img is not None
        assert img.shape == (50, 50, 3)
        
        editor.cleanup()
    
    def test_crop_resets_transforms(self):
        """Test that crop resets transform matrix."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Apply transforms
        editor.translate(10, 10)
        editor.rotate(45)
        
        # Crop
        editor.crop(10, 10, 80, 80)
        
        # Transform matrix should be reset
        matrix = editor.get_transform_matrix()
        assert np.allclose(matrix, np.eye(3))
        
        editor.cleanup()


class TestFilters:
    """Test filter application."""
    
    def test_blur_filter(self):
        """Test blur filter."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        img_before = editor.get_image()
        
        editor.apply_filter('blur')
        
        img_after = editor.get_image()
        assert img_after is not None
        assert img_after.shape == img_before.shape
        
        editor.cleanup()
    
    def test_sharpen_filter(self):
        """Test sharpen filter."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        editor.apply_filter('sharpen')
        
        img = editor.get_image()
        assert img is not None
        
        editor.cleanup()
    
    def test_grayscale_filter(self):
        """Test grayscale filter."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        editor.apply_filter('grayscale')
        
        img = editor.get_image()
        assert img is not None
        
        # Check that all channels are equal (grayscale)
        if img.shape[2] == 3:
            assert np.allclose(img[:, :, 0], img[:, :, 1])
            assert np.allclose(img[:, :, 1], img[:, :, 2])
        
        editor.cleanup()
    
    def test_edge_filter(self):
        """Test edge detection filter."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        editor.apply_filter('edge')
        
        img = editor.get_image()
        assert img is not None
        
        editor.cleanup()
    
    def test_sepia_filter(self):
        """Test sepia filter."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        editor.apply_filter('sepia')
        
        img = editor.get_image()
        assert img is not None
        
        editor.cleanup()
    
    def test_unknown_filter(self):
        """Test that unknown filter is handled gracefully."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        img_before = editor.get_image()
        
        # Apply unknown filter
        editor.apply_filter('nonexistent_filter')
        
        # Image should remain unchanged
        img_after = editor.get_image()
        assert np.array_equal(img_before, img_after)
        
        editor.cleanup()


class TestUndoRedo:
    """Test undo/redo functionality."""
    
    def test_undo_single_operation(self):
        """Test undo of single operation."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
        editor.freeze_frame(frame)
        
        img_original = editor.get_image()
        
        # Apply operation
        editor.adjust_brightness(0.5)
        img_modified = editor.get_image()
        
        # Undo
        editor.undo()
        img_undone = editor.get_image()
        
        # Should be back to original
        assert np.allclose(img_undone, img_original, atol=1)
        assert not np.array_equal(img_modified, img_undone)
        
        editor.cleanup()
    
    def test_undo_multiple_operations(self):
        """Test undo of multiple operations."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Apply multiple operations
        editor.translate(10, 0)
        editor.rotate(30)
        editor.scale(1.2)
        
        # Undo all
        editor.undo()  # Undo scale
        editor.undo()  # Undo rotate
        editor.undo()  # Undo translate
        
        # Transform matrix should be back to identity
        matrix = editor.get_transform_matrix()
        assert np.allclose(matrix, np.eye(3), atol=0.1)
        
        editor.cleanup()
    
    def test_redo_operation(self):
        """Test redo of undone operation."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
        editor.freeze_frame(frame)
        
        # Apply operation
        editor.adjust_brightness(0.5)
        img_modified = editor.get_image()
        
        # Undo
        editor.undo()
        
        # Redo
        editor.redo()
        img_redone = editor.get_image()
        
        # Should match modified version
        assert np.allclose(img_redone, img_modified, atol=1)
        
        editor.cleanup()
    
    def test_undo_with_no_history(self):
        """Test that undo with no history doesn't crash."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        img_before = editor.get_image()
        
        # Undo with no history
        editor.undo()
        
        img_after = editor.get_image()
        assert np.array_equal(img_before, img_after)
        
        editor.cleanup()
    
    def test_redo_with_no_history(self):
        """Test that redo with no history doesn't crash."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        img_before = editor.get_image()
        
        # Redo with no history
        editor.redo()
        
        img_after = editor.get_image()
        assert np.array_equal(img_before, img_after)
        
        editor.cleanup()
    
    def test_new_operation_clears_redo_stack(self):
        """Test that new operation clears redo stack."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Apply, undo
        editor.translate(10, 0)
        editor.undo()
        
        # Apply new operation
        editor.rotate(45)
        
        # Redo should do nothing
        img_before_redo = editor.get_image()
        editor.redo()
        img_after_redo = editor.get_image()
        
        assert np.array_equal(img_before_redo, img_after_redo)
        
        editor.cleanup()


class TestReset:
    """Test reset functionality."""
    
    def test_reset_to_base(self):
        """Test reset to base layer."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        img_original = editor.get_image()
        
        # Apply multiple operations
        editor.translate(10, 10)
        editor.rotate(45)
        editor.adjust_brightness(0.5)
        
        # Reset
        editor.reset()
        
        img_reset = editor.get_image()
        
        # Should be back to original (approximately)
        assert np.allclose(img_reset, img_original, atol=1)
        
        editor.cleanup()


class TestFileOperations:
    """Test loading and saving images."""
    
    def test_load_image(self, tmp_path):
        """Test loading image from file."""
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_file = tmp_path / "test_image.png"
        cv2.imwrite(str(test_file), test_image)
        
        # Load with editor
        editor = ImageManipulator()
        editor.initialize()
        
        assert editor.load_image(str(test_file)) is True
        
        img = editor.get_image()
        assert img is not None
        assert img.shape == test_image.shape
        
        editor.cleanup()
    
    def test_save_image(self, tmp_path):
        """Test saving image to file."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        # Save image
        output_file = tmp_path / "output.png"
        assert editor.save_image(str(output_file)) is True
        assert output_file.exists()
        
        # Load and verify
        loaded = cv2.imread(str(output_file))
        assert loaded is not None
        
        editor.cleanup()
    
    def test_save_without_image(self, tmp_path):
        """Test that saving without image fails gracefully."""
        editor = ImageManipulator()
        editor.initialize()
        
        output_file = tmp_path / "output.png"
        assert editor.save_image(str(output_file)) is False
        
        editor.cleanup()


class TestThreadSafety:
    """Test thread safety of operations."""
    
    def test_concurrent_transforms(self):
        """Test concurrent transform operations."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        errors = []
        
        def apply_transforms():
            try:
                for _ in range(10):
                    editor.translate(1, 1)
                    editor.rotate(5)
                    editor.scale(1.01)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = [threading.Thread(target=apply_transforms) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0
        
        # Image should still be retrievable
        img = editor.get_image()
        assert img is not None
        
        editor.cleanup()
    
    def test_concurrent_get_image(self):
        """Test concurrent get_image calls."""
        editor = ImageManipulator()
        editor.initialize()
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        editor.freeze_frame(frame)
        
        results = []
        errors = []
        
        def get_images():
            try:
                for _ in range(20):
                    img = editor.get_image()
                    results.append(img is not None)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = [threading.Thread(target=get_images) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0
        assert all(results)
        
        editor.cleanup()


class TestEditorState:
    """Test EditorState dataclass."""
    
    def test_editor_state_creation(self):
        """Test creating EditorState."""
        base = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        matrix = np.eye(3, dtype=np.float32)
        
        state = EditorState(
            base_layer=base,
            transform_layer=None,
            selection_mask=mask,
            transform_matrix=matrix,
            brightness=0.0,
            contrast=1.0,
            operation_name="test"
        )
        
        assert state.base_layer is not None
        assert state.operation_name == "test"
        assert state.brightness == 0.0
        assert state.contrast == 1.0


class TestTransformType:
    """Test TransformType enum."""
    
    def test_transform_type_enum(self):
        """Test TransformType enum values."""
        assert TransformType.TRANSLATE.value == "translate"
        assert TransformType.SCALE.value == "scale"
        assert TransformType.ROTATE.value == "rotate"
        assert TransformType.BRIGHTNESS.value == "brightness"
        assert TransformType.CONTRAST.value == "contrast"
        assert TransformType.CROP.value == "crop"
        assert TransformType.FILTER.value == "filter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
