#!/usr/bin/env python3
"""
Demo script for ImageEditor module.

Demonstrates gesture-controlled image editing capabilities including:
- Freeze frame capture
- Multi-layer management
- Transform operations (translate, rotate, scale)
- Brightness/contrast adjustments
- Undo/redo functionality
- Filter application
"""

import sys
import logging
import cv2
import numpy as np
from pathlib import Path

from src.image.editor import ImageManipulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image() -> np.ndarray:
    """
    Create a test image with various features for demonstration.
    
    Returns:
        Test image as BGR numpy array
    """
    # Create a blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Add colored rectangles
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)  # Blue
    cv2.rectangle(img, (250, 50), (400, 200), (0, 255, 0), -1)  # Green
    cv2.rectangle(img, (450, 50), (590, 200), (0, 0, 255), -1)  # Red
    
    # Add circles
    cv2.circle(img, (125, 300), 50, (255, 255, 0), -1)  # Cyan
    cv2.circle(img, (325, 300), 50, (255, 0, 255), -1)  # Magenta
    cv2.circle(img, (520, 300), 50, (0, 255, 255), -1)  # Yellow
    
    # Add text
    cv2.putText(img, "Image Editor Demo", (150, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    return img


def demonstrate_basic_operations():
    """Demonstrate basic ImageEditor operations."""
    logger.info("=== Demonstrating Basic Operations ===")
    
    # Initialize editor
    editor = ImageManipulator()
    editor.initialize()
    
    # Create test image
    test_img = create_test_image()
    logger.info(f"Created test image: {test_img.shape}")
    
    # Freeze frame
    editor.freeze_frame(test_img)
    logger.info("Freeze frame captured")
    
    # Get and display original
    original = editor.get_image()
    cv2.imshow("Original Image", original)
    cv2.waitKey(1000)
    
    # Cleanup
    editor.cleanup()
    cv2.destroyAllWindows()
    logger.info("Basic operations complete\n")


def demonstrate_transforms():
    """Demonstrate geometric transformations."""
    logger.info("=== Demonstrating Geometric Transforms ===")
    
    editor = ImageManipulator()
    editor.initialize()
    
    test_img = create_test_image()
    editor.freeze_frame(test_img)
    
    # Translation
    logger.info("Applying translation...")
    editor.translate(50, 30)
    img_translated = editor.get_image()
    cv2.imshow("Translated (50, 30)", img_translated)
    cv2.waitKey(1000)
    
    # Reset for next demo
    editor.reset()
    
    # Rotation
    logger.info("Applying rotation...")
    editor.rotate(30)
    img_rotated = editor.get_image()
    cv2.imshow("Rotated 30 degrees", img_rotated)
    cv2.waitKey(1000)
    
    # Reset for next demo
    editor.reset()
    
    # Scale
    logger.info("Applying scale...")
    editor.scale(1.5)
    img_scaled = editor.get_image()
    cv2.imshow("Scaled 1.5x", img_scaled)
    cv2.waitKey(1000)
    
    # Combined transforms
    logger.info("Applying combined transforms...")
    editor.reset()
    editor.translate(20, 20)
    editor.rotate(15)
    editor.scale(1.2)
    img_combined = editor.get_image()
    cv2.imshow("Combined Transforms", img_combined)
    cv2.waitKey(1000)
    
    editor.cleanup()
    cv2.destroyAllWindows()
    logger.info("Transform demonstrations complete\n")


def demonstrate_brightness_contrast():
    """Demonstrate brightness and contrast adjustments."""
    logger.info("=== Demonstrating Brightness & Contrast ===")
    
    editor = ImageManipulator()
    editor.initialize()
    
    test_img = create_test_image()
    editor.freeze_frame(test_img)
    
    # Brightness adjustments
    logger.info("Increasing brightness...")
    editor.adjust_brightness(0.3)
    img_bright = editor.get_image()
    cv2.imshow("Brightness +0.3", img_bright)
    cv2.waitKey(1000)
    
    editor.reset()
    logger.info("Decreasing brightness...")
    editor.adjust_brightness(-0.3)
    img_dark = editor.get_image()
    cv2.imshow("Brightness -0.3", img_dark)
    cv2.waitKey(1000)
    
    # Contrast adjustments
    editor.reset()
    logger.info("Increasing contrast...")
    editor.adjust_contrast(1.5)
    img_contrast = editor.get_image()
    cv2.imshow("Contrast 1.5x", img_contrast)
    cv2.waitKey(1000)
    
    editor.cleanup()
    cv2.destroyAllWindows()
    logger.info("Brightness & contrast demonstrations complete\n")


def demonstrate_filters():
    """Demonstrate filter applications."""
    logger.info("=== Demonstrating Filters ===")
    
    editor = ImageManipulator()
    editor.initialize()
    
    test_img = create_test_image()
    editor.freeze_frame(test_img)
    
    # Original
    original = editor.get_image()
    cv2.imshow("Original", original)
    cv2.waitKey(1000)
    
    # Blur
    logger.info("Applying blur filter...")
    editor.apply_filter('blur')
    img_blur = editor.get_image()
    cv2.imshow("Blur Filter", img_blur)
    cv2.waitKey(1000)
    
    # Sharpen
    editor.freeze_frame(test_img)
    logger.info("Applying sharpen filter...")
    editor.apply_filter('sharpen')
    img_sharpen = editor.get_image()
    cv2.imshow("Sharpen Filter", img_sharpen)
    cv2.waitKey(1000)
    
    # Grayscale
    editor.freeze_frame(test_img)
    logger.info("Applying grayscale filter...")
    editor.apply_filter('grayscale')
    img_gray = editor.get_image()
    cv2.imshow("Grayscale Filter", img_gray)
    cv2.waitKey(1000)
    
    # Edge detection
    editor.freeze_frame(test_img)
    logger.info("Applying edge detection filter...")
    editor.apply_filter('edge')
    img_edge = editor.get_image()
    cv2.imshow("Edge Detection Filter", img_edge)
    cv2.waitKey(1000)
    
    # Sepia
    editor.freeze_frame(test_img)
    logger.info("Applying sepia filter...")
    editor.apply_filter('sepia')
    img_sepia = editor.get_image()
    cv2.imshow("Sepia Filter", img_sepia)
    cv2.waitKey(1000)
    
    editor.cleanup()
    cv2.destroyAllWindows()
    logger.info("Filter demonstrations complete\n")


def demonstrate_undo_redo():
    """Demonstrate undo/redo functionality."""
    logger.info("=== Demonstrating Undo/Redo ===")
    
    editor = ImageManipulator()
    editor.initialize()
    
    test_img = create_test_image()
    editor.freeze_frame(test_img)
    
    # Show original
    original = editor.get_image()
    cv2.imshow("Step 0: Original", original)
    cv2.waitKey(1000)
    
    # Step 1: Translate
    logger.info("Step 1: Translate")
    editor.translate(50, 0)
    img1 = editor.get_image()
    cv2.imshow("Step 1: Translated", img1)
    cv2.waitKey(1000)
    
    # Step 2: Rotate
    logger.info("Step 2: Rotate")
    editor.rotate(30)
    img2 = editor.get_image()
    cv2.imshow("Step 2: + Rotated", img2)
    cv2.waitKey(1000)
    
    # Step 3: Brightness
    logger.info("Step 3: Brightness")
    editor.adjust_brightness(0.3)
    img3 = editor.get_image()
    cv2.imshow("Step 3: + Brightness", img3)
    cv2.waitKey(1000)
    
    # Undo step 3
    logger.info("Undo step 3")
    editor.undo()
    img_undo1 = editor.get_image()
    cv2.imshow("Undo 1: Back to step 2", img_undo1)
    cv2.waitKey(1000)
    
    # Undo step 2
    logger.info("Undo step 2")
    editor.undo()
    img_undo2 = editor.get_image()
    cv2.imshow("Undo 2: Back to step 1", img_undo2)
    cv2.waitKey(1000)
    
    # Redo step 2
    logger.info("Redo step 2")
    editor.redo()
    img_redo1 = editor.get_image()
    cv2.imshow("Redo 1: Forward to step 2", img_redo1)
    cv2.waitKey(1000)
    
    # Redo step 3
    logger.info("Redo step 3")
    editor.redo()
    img_redo2 = editor.get_image()
    cv2.imshow("Redo 2: Forward to step 3", img_redo2)
    cv2.waitKey(1000)
    
    editor.cleanup()
    cv2.destroyAllWindows()
    logger.info("Undo/redo demonstrations complete\n")


def demonstrate_crop():
    """Demonstrate cropping functionality."""
    logger.info("=== Demonstrating Crop ===")
    
    editor = ImageManipulator()
    editor.initialize()
    
    test_img = create_test_image()
    editor.freeze_frame(test_img)
    
    # Show original
    original = editor.get_image()
    cv2.imshow("Original (640x480)", original)
    cv2.waitKey(1000)
    
    # Crop to center region
    logger.info("Cropping to center region...")
    editor.crop(160, 120, 320, 240)
    img_cropped = editor.get_image()
    cv2.imshow("Cropped (320x240)", img_cropped)
    cv2.waitKey(1000)
    
    editor.cleanup()
    cv2.destroyAllWindows()
    logger.info("Crop demonstration complete\n")


def demonstrate_file_operations(tmp_dir="/tmp"):
    """Demonstrate loading and saving images."""
    logger.info("=== Demonstrating File Operations ===")
    
    editor = ImageManipulator()
    editor.initialize()
    
    # Create and save test image
    test_img = create_test_image()
    test_path = f"{tmp_dir}/test_image_editor.png"
    
    logger.info(f"Saving test image to {test_path}...")
    cv2.imwrite(test_path, test_img)
    
    # Load image
    logger.info("Loading image...")
    success = editor.load_image(test_path)
    if success:
        logger.info("Image loaded successfully")
        loaded_img = editor.get_image()
        cv2.imshow("Loaded Image", loaded_img)
        cv2.waitKey(1000)
        
        # Apply some edits
        logger.info("Applying edits...")
        editor.rotate(45)
        editor.adjust_brightness(0.2)
        
        # Save edited image
        output_path = f"{tmp_dir}/edited_image.png"
        logger.info(f"Saving edited image to {output_path}...")
        editor.save_image(output_path)
        
        # Load and display edited image
        edited_img = cv2.imread(output_path)
        if edited_img is not None:
            cv2.imshow("Saved & Reloaded Edited Image", edited_img)
            cv2.waitKey(1000)
            logger.info(f"Edited image saved successfully")
    else:
        logger.error("Failed to load image")
    
    editor.cleanup()
    cv2.destroyAllWindows()
    logger.info("File operations demonstration complete\n")


def demonstrate_layer_management():
    """Demonstrate layer management functionality."""
    logger.info("=== Demonstrating Layer Management ===")
    
    editor = ImageManipulator()
    editor.initialize()
    
    test_img = create_test_image()
    editor.freeze_frame(test_img)
    
    # Get layers
    layers = editor.get_layers()
    
    logger.info(f"Base layer shape: {layers['base'].shape if layers['base'] is not None else 'None'}")
    logger.info(f"Selection mask shape: {layers['selection_mask'].shape if layers['selection_mask'] is not None else 'None'}")
    logger.info(f"Transform layer: {layers['transform'] is not None}")
    
    # Display base layer
    if layers['base'] is not None:
        cv2.imshow("Base Layer", layers['base'])
        cv2.waitKey(1000)
    
    # Create and set custom selection mask (circular region)
    h, w = test_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 3, 255, -1)
    
    logger.info("Setting custom selection mask...")
    editor.set_selection_mask(mask)
    
    # Display mask
    cv2.imshow("Selection Mask", mask)
    cv2.waitKey(1000)
    
    # Get transform matrix
    matrix = editor.get_transform_matrix()
    logger.info(f"Transform matrix:\n{matrix}")
    
    editor.cleanup()
    cv2.destroyAllWindows()
    logger.info("Layer management demonstration complete\n")


def main():
    """Run all demonstrations."""
    logger.info("Starting ImageEditor demonstration\n")
    
    try:
        # Basic operations
        demonstrate_basic_operations()
        
        # Geometric transforms
        demonstrate_transforms()
        
        # Brightness and contrast
        demonstrate_brightness_contrast()
        
        # Filters
        demonstrate_filters()
        
        # Undo/redo
        demonstrate_undo_redo()
        
        # Crop
        demonstrate_crop()
        
        # File operations
        demonstrate_file_operations()
        
        # Layer management
        demonstrate_layer_management()
        
        logger.info("All demonstrations complete!")
        logger.info("\nPress any key to exit...")
        
        # Final display
        img = create_test_image()
        cv2.putText(img, "Demo Complete!", (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        cv2.imshow("Demo Complete", img)
        cv2.waitKey(0)
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        return 1
    finally:
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
