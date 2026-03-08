"""
Mask Post-processing
====================

OpenCV-based mask cleaning and refinement operations:
- Morphological closing (fill holes)
- Hole filling
- Small component removal
- Contour smoothing
- Edge feathering
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass

from src.core import get_logger

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PostprocessConfig:
    """Configuration for mask post-processing."""
    # Morphological operations
    closing_kernel_size: int = 15
    opening_kernel_size: int = 5
    
    # Hole filling
    fill_holes: bool = True
    
    # Component filtering
    min_component_area: int = 500
    keep_largest_only: bool = True
    
    # Smoothing
    smooth_contours: bool = True
    contour_epsilon: float = 0.005  # Relative to contour perimeter
    
    # Edge feathering
    feather_edges: bool = True
    feather_amount: int = 3
    
    # Gaussian blur for edges
    edge_blur_kernel: int = 5


# ============================================================================
# Mask Post-processor
# ============================================================================

class MaskPostprocessor:
    """
    Post-processes masks using OpenCV operations.
    
    Produces clean, smooth masks suitable for garment extraction.
    """
    
    def __init__(self, config: Optional[PostprocessConfig] = None):
        """
        Initialize the post-processor.
        
        Args:
            config: Post-processing configuration
        """
        self.config = config or PostprocessConfig()
        logger.info("MaskPostprocessor initialized")
    
    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply full post-processing pipeline to a mask.
        
        Pipeline:
        1. Morphological closing (fill small holes)
        2. Morphological opening (remove noise)
        3. Fill interior holes
        4. Remove small components
        5. Smooth contours
        6. Feather edges (optional)
        
        Args:
            mask: Input binary mask (0 or 255)
            
        Returns:
            Cleaned binary mask
        """
        # Ensure binary mask
        mask = self._ensure_binary(mask)
        
        # Step 1: Morphological closing
        mask = self._morphological_close(mask)
        
        # Step 2: Morphological opening
        mask = self._morphological_open(mask)
        
        # Step 3: Fill holes
        if self.config.fill_holes:
            mask = self._fill_holes(mask)
        
        # Step 4: Remove small components
        mask = self._remove_small_components(mask)
        
        # Step 5: Smooth contours
        if self.config.smooth_contours:
            mask = self._smooth_contours(mask)
        
        # Step 6: Edge smoothing with Gaussian blur
        mask = self._smooth_edges(mask)
        
        return mask
    
    def process_batch(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process multiple masks.
        
        Args:
            masks: List of input masks
            
        Returns:
            List of processed masks
        """
        return [self.process(mask) for mask in masks]
    
    def _ensure_binary(self, mask: np.ndarray) -> np.ndarray:
        """Ensure mask is binary (0 or 255)."""
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        if mask.max() == 1:
            mask = mask * 255
        elif mask.max() > 1 and mask.max() <= 255:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _morphological_close(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological closing to fill small holes.
        
        Closing = Dilation followed by Erosion
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.config.closing_kernel_size, self.config.closing_kernel_size)
        )
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    def _morphological_open(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening to remove noise.
        
        Opening = Erosion followed by Dilation
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.opening_kernel_size, self.config.opening_kernel_size)
        )
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes inside the mask using flood fill.
        """
        # Create a padded version for flood fill
        h, w = mask.shape
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = mask
        
        # Flood fill from corner (exterior)
        flood_fill_mask = padded.copy()
        cv2.floodFill(flood_fill_mask, None, (0, 0), 255)
        
        # Invert flood filled to get holes
        holes = cv2.bitwise_not(flood_fill_mask)[1:-1, 1:-1]
        
        # Combine with original mask
        filled = cv2.bitwise_or(mask, holes)
        
        return filled
    
    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove small connected components.
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        if num_labels <= 1:
            return mask
        
        # Get component areas (excluding background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        
        if self.config.keep_largest_only:
            # Keep only the largest component
            largest_idx = np.argmax(areas) + 1  # +1 because we excluded background
            result = np.where(labels == largest_idx, 255, 0).astype(np.uint8)
        else:
            # Keep all components above minimum area
            result = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= self.config.min_component_area:
                    result = np.where(labels == i, 255, result).astype(np.uint8)
        
        return result
    
    def _smooth_contours(self, mask: np.ndarray) -> np.ndarray:
        """
        Smooth contours using polygon approximation.
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return mask
        
        # Create smoothed mask
        smoothed = np.zeros_like(mask)
        
        for contour in contours:
            # Calculate epsilon based on contour perimeter
            perimeter = cv2.arcLength(contour, True)
            epsilon = self.config.contour_epsilon * perimeter
            
            # Approximate polygon
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw filled polygon
            cv2.drawContours(smoothed, [approx], -1, 255, -1)
        
        return smoothed
    
    def _smooth_edges(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to smooth edges, then threshold.
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            mask, 
            (self.config.edge_blur_kernel, self.config.edge_blur_kernel), 
            0
        )
        
        # Threshold to get binary mask back
        _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        return result
    
    def feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Create a feathered (soft-edged) mask for smooth blending.
        
        Returns a mask with gradient edges (0-255).
        """
        if not self.config.feather_edges:
            return mask
        
        # Create distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Normalize and create alpha gradient at edges
        feather = self.config.feather_amount
        dist_normalized = np.clip(dist / feather, 0, 1)
        
        return (dist_normalized * 255).astype(np.uint8)
    
    def get_bounding_box(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the bounding box of the mask content.
        
        Returns:
            (x, y, width, height) or None if mask is empty
        """
        coords = cv2.findNonZero(mask)
        
        if coords is None:
            return None
        
        return cv2.boundingRect(coords)
    
    def crop_to_content(
        self, 
        mask: np.ndarray, 
        padding: int = 0
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop mask to content bounding box.
        
        Args:
            mask: Input mask
            padding: Extra padding around content
            
        Returns:
            Tuple of (cropped mask, bounding box)
        """
        bbox = self.get_bounding_box(mask)
        
        if bbox is None:
            return mask, (0, 0, mask.shape[1], mask.shape[0])
        
        x, y, w, h = bbox
        h_img, w_img = mask.shape
        
        # Apply padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        
        cropped = mask[y1:y2, x1:x2]
        
        return cropped, (x1, y1, x2 - x1, y2 - y1)


# ============================================================================
# Convenience Functions
# ============================================================================

def clean_mask(
    mask: np.ndarray,
    config: Optional[PostprocessConfig] = None
) -> np.ndarray:
    """
    Convenience function to clean a mask.
    
    Args:
        mask: Input binary mask
        config: Optional configuration
        
    Returns:
        Cleaned binary mask
    """
    processor = MaskPostprocessor(config)
    return processor.process(mask)


def clean_masks(
    masks: List[np.ndarray],
    config: Optional[PostprocessConfig] = None
) -> List[np.ndarray]:
    """
    Convenience function to clean multiple masks.
    
    Args:
        masks: List of input masks
        config: Optional configuration
        
    Returns:
        List of cleaned masks
    """
    processor = MaskPostprocessor(config)
    return processor.process_batch(masks)
