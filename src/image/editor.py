"""
Image editor implementation with gesture-based manipulation.

This module provides comprehensive image editing functionality including:
- Freeze frame capture from VisionEngine
- Multi-layer management (base, selection mask, transform)
- Transform matrix tracking for non-destructive editing
- Undo/redo stack
- Gesture-based controls
- OpenCV-powered transformations
"""

from typing import Optional, List, Tuple, Dict, Any
import threading
import logging
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum

from src.core.image_editor import ImageEditor

logger = logging.getLogger(__name__)


class TransformType(Enum):
    """Types of image transformations."""
    TRANSLATE = "translate"
    SCALE = "scale"
    ROTATE = "rotate"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    CROP = "crop"
    FILTER = "filter"


@dataclass
class EditorState:
    """
    Snapshot of editor state for undo/redo functionality.
    
    Stores all necessary information to reconstruct editor state.
    """
    base_layer: np.ndarray
    transform_layer: Optional[np.ndarray]
    selection_mask: Optional[np.ndarray]
    transform_matrix: np.ndarray
    brightness: float
    contrast: float
    operation_name: str


class ImageManipulator(ImageEditor):
    """
    Gesture-controlled image editor with non-blocking operations.
    
    Features:
    - Captures freeze frames from VisionEngine
    - Maintains multi-layer structure (base, selection, transform)
    - Tracks transform matrices for non-destructive editing
    - Supports gesture-based manipulation
    - Thread-safe operations
    - Comprehensive undo/redo stack
    """
    
    def __init__(self, max_undo_stack: int = 50):
        """
        Initialize image editor.
        
        Args:
            max_undo_stack: Maximum number of undo states to keep
        """
        super().__init__()
        
        # Layer management
        self._base_layer: Optional[np.ndarray] = None
        self._selection_mask: Optional[np.ndarray] = None
        self._transform_layer: Optional[np.ndarray] = None
        
        # Transform state
        self._transform_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        self._brightness: float = 0.0
        self._contrast: float = 1.0
        
        # Undo/redo stack
        self._undo_stack: List[EditorState] = []
        self._redo_stack: List[EditorState] = []
        self._max_undo_stack = max_undo_stack
        
        # Thread safety
        self._lock = threading.Lock()
        
        # State tracking
        self._initialized = False
        self._current_image: Optional[np.ndarray] = None
        
    def initialize(self) -> bool:
        """
        Initialize the image editor.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing ImageEditor")
            
            with self._lock:
                # Reset all state
                self._base_layer = None
                self._selection_mask = None
                self._transform_layer = None
                self._transform_matrix = np.eye(3, dtype=np.float32)
                self._brightness = 0.0
                self._contrast = 1.0
                self._undo_stack.clear()
                self._redo_stack.clear()
                self._current_image = None
                self._initialized = True
                
            logger.info("ImageEditor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ImageEditor: {e}", exc_info=True)
            return False
    
    def freeze_frame(self, frame: np.ndarray) -> bool:
        """
        Capture a freeze frame from VisionEngine.
        
        Args:
            frame: Frame to capture (BGR numpy array)
            
        Returns:
            bool: True if captured successfully
        """
        if frame is None or frame.size == 0:
            logger.error("Cannot freeze empty frame")
            return False
            
        try:
            with self._lock:
                # Store original frame as base layer
                self._base_layer = frame.copy()
                
                # Initialize selection mask (full image selected)
                h, w = frame.shape[:2]
                self._selection_mask = np.ones((h, w), dtype=np.uint8) * 255
                
                # Reset transform layer
                self._transform_layer = None
                
                # Reset transform matrix
                self._transform_matrix = np.eye(3, dtype=np.float32)
                self._brightness = 0.0
                self._contrast = 1.0
                
                # Clear undo/redo stacks
                self._undo_stack.clear()
                self._redo_stack.clear()
                
                # Update current image
                self._current_image = self._composite_image()
                
            logger.info(f"Freeze frame captured: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to freeze frame: {e}", exc_info=True)
            return False
    
    def load_image(self, filepath: str) -> bool:
        """
        Load an image from file.
        
        Args:
            filepath: Path to image file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            logger.info(f"Loading image from {filepath}")
            frame = cv2.imread(filepath)
            
            if frame is None:
                logger.error(f"Failed to read image from {filepath}")
                return False
                
            return self.freeze_frame(frame)
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}", exc_info=True)
            return False
    
    def save_image(self, filepath: str) -> bool:
        """
        Save current image to file.
        
        Args:
            filepath: Path where to save
            
        Returns:
            bool: True if saved successfully
        """
        try:
            with self._lock:
                if self._current_image is None:
                    logger.error("No image to save")
                    return False
                    
                success = cv2.imwrite(filepath, self._current_image)
                
                if success:
                    logger.info(f"Image saved to {filepath}")
                else:
                    logger.error(f"Failed to save image to {filepath}")
                    
                return success
                
        except Exception as e:
            logger.error(f"Failed to save image: {e}", exc_info=True)
            return False
    
    def get_image(self) -> Optional[np.ndarray]:
        """
        Get current composited image (thread-safe).
        
        Returns:
            Current image or None if unavailable
        """
        with self._lock:
            return self._current_image.copy() if self._current_image is not None else None
    
    def translate(self, dx: float, dy: float) -> None:
        """
        Translate (move) the image or selected region.
        
        Corresponds to pinch + drag gesture.
        
        Args:
            dx: Horizontal translation in pixels
            dy: Vertical translation in pixels
        """
        if not self._check_initialized():
            return
            
        try:
            with self._lock:
                # Save current state for undo
                self._save_state("translate")
                
                # Create translation matrix
                translation = np.array([
                    [1, 0, dx],
                    [0, 1, dy],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Apply to transform matrix
                self._transform_matrix = translation @ self._transform_matrix
                
                # Update current image
                self._current_image = self._composite_image()
                
                logger.debug(f"Applied translation: dx={dx}, dy={dy}")
                
        except Exception as e:
            logger.error(f"Failed to translate: {e}", exc_info=True)
    
    def rotate(self, angle: float, center: Optional[Tuple[float, float]] = None) -> None:
        """
        Rotate image by specified angle.
        
        Corresponds to circular motion gesture.
        
        Args:
            angle: Rotation angle in degrees (positive = counter-clockwise)
            center: Rotation center (x, y). If None, uses image center.
        """
        if not self._check_initialized():
            return
            
        try:
            with self._lock:
                # Save current state for undo
                self._save_state("rotate")
                
                # Get image dimensions
                h, w = self._base_layer.shape[:2]
                
                # Use image center if not specified
                if center is None:
                    center = (w / 2, h / 2)
                
                # Create rotation matrix
                rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Convert to 3x3 matrix
                rotation_3x3 = np.vstack([rotation, [0, 0, 1]])
                
                # Apply to transform matrix
                self._transform_matrix = rotation_3x3 @ self._transform_matrix
                
                # Update current image
                self._current_image = self._composite_image()
                
                logger.debug(f"Applied rotation: angle={angle}°, center={center}")
                
        except Exception as e:
            logger.error(f"Failed to rotate: {e}", exc_info=True)
    
    def scale(self, factor: float, center: Optional[Tuple[float, float]] = None) -> None:
        """
        Scale image by specified factor.
        
        Corresponds to two-hand stretch gesture.
        
        Args:
            factor: Scaling factor (1.0 = original, >1.0 = zoom in, <1.0 = zoom out)
            center: Scaling center (x, y). If None, uses image center.
        """
        if not self._check_initialized():
            return
            
        try:
            with self._lock:
                # Save current state for undo
                self._save_state("scale")
                
                # Get image dimensions
                h, w = self._base_layer.shape[:2]
                
                # Use image center if not specified
                if center is None:
                    center = (w / 2, h / 2)
                
                cx, cy = center
                
                # Create scaling matrix with center offset
                # Scale around center: T(center) * S(factor) * T(-center)
                scale_matrix = np.array([
                    [factor, 0, cx * (1 - factor)],
                    [0, factor, cy * (1 - factor)],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Apply to transform matrix
                self._transform_matrix = scale_matrix @ self._transform_matrix
                
                # Update current image
                self._current_image = self._composite_image()
                
                logger.debug(f"Applied scale: factor={factor}, center={center}")
                
        except Exception as e:
            logger.error(f"Failed to scale: {e}", exc_info=True)
    
    def adjust_brightness(self, value: float) -> None:
        """
        Adjust image brightness.
        
        Corresponds to palm tilt gesture.
        
        Args:
            value: Brightness adjustment (-1.0 to 1.0)
                   Negative values darken, positive values brighten
        """
        if not self._check_initialized():
            return
            
        try:
            # Clamp value to valid range
            value = max(-1.0, min(1.0, value))
            
            with self._lock:
                # Save current state for undo
                self._save_state("brightness")
                
                # Update brightness value
                self._brightness = value
                
                # Update current image
                self._current_image = self._composite_image()
                
                logger.debug(f"Applied brightness adjustment: {value}")
                
        except Exception as e:
            logger.error(f"Failed to adjust brightness: {e}", exc_info=True)
    
    def adjust_contrast(self, value: float) -> None:
        """
        Adjust image contrast.
        
        Args:
            value: Contrast adjustment (0.0 to 2.0)
                   1.0 = original, <1.0 = reduce, >1.0 = increase
        """
        if not self._check_initialized():
            return
            
        try:
            # Clamp value to valid range
            value = max(0.0, min(2.0, value))
            
            with self._lock:
                # Save current state for undo
                self._save_state("contrast")
                
                # Update contrast value
                self._contrast = value
                
                # Update current image
                self._current_image = self._composite_image()
                
                logger.debug(f"Applied contrast adjustment: {value}")
                
        except Exception as e:
            logger.error(f"Failed to adjust contrast: {e}", exc_info=True)
    
    def crop(self, x: int, y: int, width: int, height: int) -> None:
        """
        Crop image to specified region.
        
        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of crop region
            height: Height of crop region
        """
        if not self._check_initialized():
            return
            
        try:
            with self._lock:
                # Save current state for undo
                self._save_state("crop")
                
                h, w = self._base_layer.shape[:2]
                
                # Clamp crop region to image bounds
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                width = max(1, min(width, w - x))
                height = max(1, min(height, h - y))
                
                # Crop base layer
                self._base_layer = self._base_layer[y:y+height, x:x+width].copy()
                
                # Crop selection mask
                if self._selection_mask is not None:
                    self._selection_mask = self._selection_mask[y:y+height, x:x+width].copy()
                
                # Reset transform matrix after crop
                self._transform_matrix = np.eye(3, dtype=np.float32)
                
                # Update current image
                self._current_image = self._composite_image()
                
                logger.debug(f"Applied crop: x={x}, y={y}, w={width}, h={height}")
                
        except Exception as e:
            logger.error(f"Failed to crop: {e}", exc_info=True)
    
    def apply_filter(self, filter_name: str) -> None:
        """
        Apply a filter to the image.
        
        Supported filters:
        - 'blur': Gaussian blur
        - 'sharpen': Sharpen filter
        - 'edge': Edge detection
        - 'grayscale': Convert to grayscale
        - 'sepia': Sepia tone effect
        
        Args:
            filter_name: Name of filter to apply
        """
        if not self._check_initialized():
            return
            
        try:
            with self._lock:
                # Save current state for undo
                self._save_state(f"filter_{filter_name}")
                
                # Get current composited image for filter application
                img = self._composite_image()
                
                # Apply filter
                if filter_name == 'blur':
                    img = cv2.GaussianBlur(img, (15, 15), 0)
                elif filter_name == 'sharpen':
                    kernel = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
                    img = cv2.filter2D(img, -1, kernel)
                elif filter_name == 'edge':
                    img = cv2.Canny(img, 100, 200)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif filter_name == 'grayscale':
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                elif filter_name == 'sepia':
                    kernel = np.array([[0.272, 0.534, 0.131],
                                      [0.349, 0.686, 0.168],
                                      [0.393, 0.769, 0.189]])
                    img = cv2.transform(img, kernel)
                    img = np.clip(img, 0, 255).astype(np.uint8)
                else:
                    logger.warning(f"Unknown filter: {filter_name}")
                    return
                
                # Update base layer with filtered image
                self._base_layer = img
                
                # Reset transform state
                self._transform_matrix = np.eye(3, dtype=np.float32)
                self._brightness = 0.0
                self._contrast = 1.0
                
                # Update current image
                self._current_image = self._composite_image()
                
                logger.debug(f"Applied filter: {filter_name}")
                
        except Exception as e:
            logger.error(f"Failed to apply filter: {e}", exc_info=True)
    
    def set_selection_mask(self, mask: np.ndarray) -> bool:
        """
        Set the selection mask for region-specific editing.
        
        Args:
            mask: Binary mask (uint8, 0 or 255)
            
        Returns:
            bool: True if set successfully
        """
        try:
            with self._lock:
                if self._base_layer is None:
                    logger.error("No base layer to apply mask to")
                    return False
                
                h, w = self._base_layer.shape[:2]
                mh, mw = mask.shape[:2]
                
                if (h, w) != (mh, mw):
                    logger.error(f"Mask size {(mh, mw)} does not match image size {(h, w)}")
                    return False
                
                self._selection_mask = mask.copy()
                logger.debug("Selection mask updated")
                return True
                
        except Exception as e:
            logger.error(f"Failed to set selection mask: {e}", exc_info=True)
            return False
    
    def undo(self) -> None:
        """
        Undo last operation.
        
        Corresponds to swipe left gesture.
        """
        try:
            with self._lock:
                if not self._undo_stack:
                    logger.debug("Nothing to undo")
                    return
                
                # Save current state to redo stack
                current_state = self._capture_state("redo")
                self._redo_stack.append(current_state)
                
                # Restore previous state
                prev_state = self._undo_stack.pop()
                self._restore_state(prev_state)
                
                # Update current image
                self._current_image = self._composite_image()
                
                logger.debug(f"Undo: restored '{prev_state.operation_name}'")
                
        except Exception as e:
            logger.error(f"Failed to undo: {e}", exc_info=True)
    
    def redo(self) -> None:
        """Redo last undone operation."""
        try:
            with self._lock:
                if not self._redo_stack:
                    logger.debug("Nothing to redo")
                    return
                
                # Save current state to undo stack
                current_state = self._capture_state("undo")
                self._undo_stack.append(current_state)
                
                # Restore next state
                next_state = self._redo_stack.pop()
                self._restore_state(next_state)
                
                # Update current image
                self._current_image = self._composite_image()
                
                logger.debug(f"Redo: restored '{next_state.operation_name}'")
                
        except Exception as e:
            logger.error(f"Failed to redo: {e}", exc_info=True)
    
    def reset(self) -> None:
        """Reset image to original base layer state."""
        try:
            with self._lock:
                if self._base_layer is None:
                    logger.debug("No image to reset")
                    return
                
                # Save current state for undo
                self._save_state("reset")
                
                # Reset all transforms
                self._transform_matrix = np.eye(3, dtype=np.float32)
                self._brightness = 0.0
                self._contrast = 1.0
                self._transform_layer = None
                
                # Update current image
                self._current_image = self._composite_image()
                
                logger.debug("Image reset to base layer")
                
        except Exception as e:
            logger.error(f"Failed to reset: {e}", exc_info=True)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            logger.info("Cleaning up ImageEditor")
            
            with self._lock:
                self._base_layer = None
                self._selection_mask = None
                self._transform_layer = None
                self._current_image = None
                self._undo_stack.clear()
                self._redo_stack.clear()
                self._initialized = False
                
            logger.info("ImageEditor cleanup complete")
            
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}", exc_info=True)
    
    # ==================== Internal Helper Methods ====================
    
    def _check_initialized(self) -> bool:
        """Check if editor is initialized with an image."""
        if not self._initialized or self._base_layer is None:
            logger.warning("ImageEditor not initialized or no image loaded")
            return False
        return True
    
    def _composite_image(self) -> np.ndarray:
        """
        Composite final image from layers and transforms.
        
        Returns:
            Final composited image
        """
        if self._base_layer is None:
            return None
        
        # Start with base layer
        img = self._base_layer.copy()
        
        # Apply geometric transforms (translation, rotation, scale)
        h, w = img.shape[:2]
        
        # Extract 2x3 affine matrix from 3x3 matrix
        transform_2x3 = self._transform_matrix[:2, :]
        
        # Apply affine transformation
        img = cv2.warpAffine(img, transform_2x3, (w, h), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
        
        # Apply brightness/contrast adjustments
        if self._brightness != 0.0 or self._contrast != 1.0:
            img = self._apply_brightness_contrast(img, self._brightness, self._contrast)
        
        # Apply selection mask if present
        if self._selection_mask is not None:
            # Blend with base layer using mask
            mask_3ch = cv2.cvtColor(self._selection_mask, cv2.COLOR_GRAY2BGR) / 255.0
            img = (img * mask_3ch + self._base_layer * (1 - mask_3ch)).astype(np.uint8)
        
        return img
    
    def _apply_brightness_contrast(self, img: np.ndarray, 
                                   brightness: float, contrast: float) -> np.ndarray:
        """
        Apply brightness and contrast adjustments.
        
        Args:
            img: Input image
            brightness: Brightness adjustment (-1.0 to 1.0)
            contrast: Contrast multiplier (0.0 to 2.0)
            
        Returns:
            Adjusted image
        """
        # Convert brightness from [-1, 1] to [0, 255] offset
        beta = brightness * 127.0
        
        # Apply contrast and brightness
        # formula: output = alpha * input + beta
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=beta)
        
        return img
    
    def _save_state(self, operation_name: str) -> None:
        """
        Save current state to undo stack.
        
        Args:
            operation_name: Name of the operation being performed
        """
        state = self._capture_state(operation_name)
        
        # Add to undo stack
        self._undo_stack.append(state)
        
        # Limit stack size
        if len(self._undo_stack) > self._max_undo_stack:
            self._undo_stack.pop(0)
        
        # Clear redo stack when new operation is performed
        self._redo_stack.clear()
    
    def _capture_state(self, operation_name: str) -> EditorState:
        """
        Capture current editor state.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            EditorState snapshot
        """
        return EditorState(
            base_layer=self._base_layer.copy() if self._base_layer is not None else None,
            transform_layer=self._transform_layer.copy() if self._transform_layer is not None else None,
            selection_mask=self._selection_mask.copy() if self._selection_mask is not None else None,
            transform_matrix=self._transform_matrix.copy(),
            brightness=self._brightness,
            contrast=self._contrast,
            operation_name=operation_name
        )
    
    def _restore_state(self, state: EditorState) -> None:
        """
        Restore editor state from snapshot.
        
        Args:
            state: EditorState to restore
        """
        self._base_layer = state.base_layer.copy() if state.base_layer is not None else None
        self._transform_layer = state.transform_layer.copy() if state.transform_layer is not None else None
        self._selection_mask = state.selection_mask.copy() if state.selection_mask is not None else None
        self._transform_matrix = state.transform_matrix.copy()
        self._brightness = state.brightness
        self._contrast = state.contrast
    
    def get_transform_matrix(self) -> np.ndarray:
        """
        Get current transform matrix (thread-safe).
        
        Returns:
            3x3 transform matrix
        """
        with self._lock:
            return self._transform_matrix.copy()
    
    def get_layers(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Get all layers (thread-safe).
        
        Returns:
            Dictionary with 'base', 'transform', 'selection_mask' keys
        """
        with self._lock:
            return {
                'base': self._base_layer.copy() if self._base_layer is not None else None,
                'transform': self._transform_layer.copy() if self._transform_layer is not None else None,
                'selection_mask': self._selection_mask.copy() if self._selection_mask is not None else None,
            }
