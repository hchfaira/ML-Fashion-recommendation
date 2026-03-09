"""
Clothing Detector
=================

Detects clothing items in user photos (optional analysis).
Extracts clothing type, colors, and patterns.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from PIL import Image
from dataclasses import dataclass

from .models import DetectedClothing

logger = logging.getLogger(__name__)


@dataclass
class ClothingRegion:
    """A detected clothing region."""
    category: str           # top, bottom, dress, jacket, etc.
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    dominant_color: Tuple[int, int, int]  # RGB
    pattern: str            # solid, striped, plaid, etc.
    confidence: float


class ClothingDetector:
    """
    Detects and analyzes clothing in user photos.
    
    Uses person segmentation to isolate body region,
    then estimates clothing regions based on body proportions.
    """
    
    # Clothing region estimations (relative to body bbox)
    REGION_ESTIMATES = {
        "top": {
            "y_start": 0.15,   # Below head
            "y_end": 0.45,     # Above waist
            "x_margin": 0.1,
        },
        "bottom": {
            "y_start": 0.45,   # Below waist
            "y_end": 0.85,     # Above feet
            "x_margin": 0.15,
        },
        "full_body": {
            "y_start": 0.15,
            "y_end": 0.85,
            "x_margin": 0.1,
        }
    }
    
    # Common pattern characteristics
    PATTERN_THRESHOLDS = {
        "solid_variance": 500,      # Low color variance = solid
        "striped_freq": 0.3,        # Regular vertical/horizontal patterns
        "plaid_freq": 0.2,          # Both directions
    }
    
    # Basic color name mapping
    COLOR_NAMES = {
        (0, 0, 0): "black",
        (255, 255, 255): "white",
        (128, 128, 128): "grey",
        (255, 0, 0): "red",
        (0, 255, 0): "green",
        (0, 0, 255): "blue",
        (255, 255, 0): "yellow",
        (255, 165, 0): "orange",
        (128, 0, 128): "purple",
        (255, 192, 203): "pink",
        (165, 42, 42): "brown",
        (0, 128, 128): "teal",
        (0, 0, 128): "navy",
        (245, 245, 220): "beige",
    }
    
    def __init__(self):
        """Initialize clothing detector."""
        self._segmenter = None
    
    def _ensure_mediapipe(self):
        """Lazy load MediaPipe Selfie Segmentation."""
        if self._segmenter is None:
            try:
                import mediapipe as mp
                selfie_seg = mp.solutions.selfie_segmentation
                self._segmenter = selfie_seg.SelfieSegmentation(model_selection=1)
                logger.info("MediaPipe Selfie Segmentation initialized for clothing detection")
            except ImportError:
                logger.warning("MediaPipe not installed, using fallback clothing detection")
                self._segmenter = "fallback"
    
    def detect(
        self,
        image: Image.Image,
        body_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[DetectedClothing]:
        """
        Detect clothing from user image.
        
        Args:
            image: PIL Image containing person
            body_bbox: Optional body bounding box (x, y, w, h)
            
        Returns:
            DetectedClothing with detected items
        """
        self._ensure_mediapipe()
        
        img_rgb = np.array(image.convert("RGB"))
        height, width = img_rgb.shape[:2]
        
        # Get person mask
        person_mask = self._get_person_mask(img_rgb)
        
        if person_mask is None:
            # Fall back to body bbox or full image
            if body_bbox:
                x, y, w, h = body_bbox
                person_mask = np.zeros((height, width), dtype=np.float32)
                person_mask[y:y+h, x:x+w] = 1.0
            else:
                person_mask = np.ones((height, width), dtype=np.float32)
        
        # Get body bounding box
        if body_bbox is None:
            body_bbox = self._get_body_bbox(person_mask)
        
        if body_bbox is None:
            logger.warning("Could not detect body region")
            return None
        
        # Detect clothing regions
        regions = self._detect_clothing_regions(img_rgb, person_mask, body_bbox)
        
        if not regions:
            logger.warning("No clothing regions detected")
            return None
        
        # Build detected clothing
        detected = DetectedClothing()
        
        for region in regions:
            if region.category == "top":
                detected.has_top = True
                detected.top_type = self._infer_top_type(region)
                detected.top_color = self._get_color_name(region.dominant_color)
                detected.top_pattern = region.pattern
            elif region.category == "bottom":
                detected.has_bottom = True
                detected.bottom_type = self._infer_bottom_type(region)
                detected.bottom_color = self._get_color_name(region.dominant_color)
                detected.bottom_pattern = region.pattern
            elif region.category == "full_body":
                detected.has_full_body = True
                detected.full_body_type = "dress"
                detected.full_body_color = self._get_color_name(region.dominant_color)
        
        # Compute overall confidence
        detected.confidence = np.mean([r.confidence for r in regions])
        
        return detected
    
    def _get_person_mask(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Get person segmentation mask."""
        if self._segmenter == "fallback":
            return None
        
        try:
            results = self._segmenter.process(img_rgb)
            if results.segmentation_mask is not None:
                return results.segmentation_mask
        except Exception as e:
            logger.warning(f"Person segmentation failed: {e}")
        
        return None
    
    def _get_body_bbox(
        self,
        mask: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """Get body bounding box from mask."""
        # Find non-zero regions
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y_min = np.argmax(rows)
        y_max = len(rows) - np.argmax(rows[::-1])
        x_min = np.argmax(cols)
        x_max = len(cols) - np.argmax(cols[::-1])
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _detect_clothing_regions(
        self,
        img_rgb: np.ndarray,
        person_mask: np.ndarray,
        body_bbox: Tuple[int, int, int, int]
    ) -> List[ClothingRegion]:
        """Detect individual clothing regions."""
        regions = []
        bx, by, bw, bh = body_bbox
        height, width = img_rgb.shape[:2]
        
        # Check if wearing full body (dress) or separates
        is_full_body = self._detect_full_body_garment(img_rgb, person_mask, body_bbox)
        
        if is_full_body:
            # Detect full body garment
            region_def = self.REGION_ESTIMATES["full_body"]
            y_start = int(by + bh * region_def["y_start"])
            y_end = int(by + bh * region_def["y_end"])
            x_margin = int(bw * region_def["x_margin"])
            x_start = bx + x_margin
            x_end = bx + bw - x_margin
            
            region_pixels = img_rgb[y_start:y_end, x_start:x_end]
            
            if region_pixels.size > 0:
                color, pattern = self._analyze_region(region_pixels)
                regions.append(ClothingRegion(
                    category="full_body",
                    bbox=(x_start, y_start, x_end - x_start, y_end - y_start),
                    dominant_color=color,
                    pattern=pattern,
                    confidence=0.7,
                ))
        else:
            # Detect top
            region_def = self.REGION_ESTIMATES["top"]
            y_start = int(by + bh * region_def["y_start"])
            y_end = int(by + bh * region_def["y_end"])
            x_margin = int(bw * region_def["x_margin"])
            x_start = bx + x_margin
            x_end = bx + bw - x_margin
            
            region_pixels = img_rgb[y_start:y_end, x_start:x_end]
            
            if region_pixels.size > 0:
                color, pattern = self._analyze_region(region_pixels)
                regions.append(ClothingRegion(
                    category="top",
                    bbox=(x_start, y_start, x_end - x_start, y_end - y_start),
                    dominant_color=color,
                    pattern=pattern,
                    confidence=0.75,
                ))
            
            # Detect bottom
            region_def = self.REGION_ESTIMATES["bottom"]
            y_start = int(by + bh * region_def["y_start"])
            y_end = int(by + bh * region_def["y_end"])
            x_margin = int(bw * region_def["x_margin"])
            x_start = bx + x_margin
            x_end = bx + bw - x_margin
            
            region_pixels = img_rgb[y_start:y_end, x_start:x_end]
            
            if region_pixels.size > 0:
                color, pattern = self._analyze_region(region_pixels)
                regions.append(ClothingRegion(
                    category="bottom",
                    bbox=(x_start, y_start, x_end - x_start, y_end - y_start),
                    dominant_color=color,
                    pattern=pattern,
                    confidence=0.7,
                ))
        
        return regions
    
    def _detect_full_body_garment(
        self,
        img_rgb: np.ndarray,
        person_mask: np.ndarray,
        body_bbox: Tuple[int, int, int, int]
    ) -> bool:
        """
        Detect if person is wearing a full-body garment (dress).
        
        Uses color continuity between top and bottom regions.
        """
        bx, by, bw, bh = body_bbox
        
        # Sample from top region
        top_region = self.REGION_ESTIMATES["top"]
        top_y = int(by + bh * (top_region["y_start"] + top_region["y_end"]) / 2)
        top_x_start = bx + int(bw * 0.3)
        top_x_end = bx + int(bw * 0.7)
        
        # Sample from bottom region
        bottom_region = self.REGION_ESTIMATES["bottom"]
        bottom_y = int(by + bh * (bottom_region["y_start"] + bottom_region["y_end"]) / 2)
        bottom_x_start = bx + int(bw * 0.3)
        bottom_x_end = bx + int(bw * 0.7)
        
        # Get sample strips
        top_strip = img_rgb[top_y:top_y+10, top_x_start:top_x_end]
        bottom_strip = img_rgb[bottom_y:bottom_y+10, bottom_x_start:bottom_x_end]
        
        if top_strip.size == 0 or bottom_strip.size == 0:
            return False
        
        # Compare colors
        top_mean = np.mean(top_strip.reshape(-1, 3), axis=0)
        bottom_mean = np.mean(bottom_strip.reshape(-1, 3), axis=0)
        
        # If colors are very similar, likely full body
        color_diff = np.sqrt(np.sum((top_mean - bottom_mean) ** 2))
        
        return color_diff < 40  # Threshold for color similarity
    
    def _analyze_region(
        self,
        pixels: np.ndarray
    ) -> Tuple[Tuple[int, int, int], str]:
        """Analyze a clothing region for color and pattern."""
        # Reshape
        flat = pixels.reshape(-1, 3)
        
        # Get dominant color (median for robustness)
        dominant = tuple(np.median(flat, axis=0).astype(int))
        
        # Detect pattern
        pattern = self._detect_pattern(pixels)
        
        return dominant, pattern
    
    def _detect_pattern(self, pixels: np.ndarray) -> str:
        """Detect pattern type in clothing region."""
        # Convert to grayscale for pattern analysis
        gray = np.mean(pixels, axis=2)
        
        # Compute variance
        variance = np.var(gray)
        
        if variance < self.PATTERN_THRESHOLDS["solid_variance"]:
            return "solid"
        
        # Check for stripes (high frequency in one direction)
        # Simple heuristic: variance in rows vs columns
        row_var = np.var(np.mean(gray, axis=1))
        col_var = np.var(np.mean(gray, axis=0))
        
        if row_var > col_var * 2:
            return "horizontal_stripes"
        elif col_var > row_var * 2:
            return "vertical_stripes"
        elif row_var > 100 and col_var > 100:
            return "plaid"
        elif variance > 2000:
            return "print"
        
        return "patterned"
    
    def _infer_top_type(self, region: ClothingRegion) -> str:
        """Infer top type from region."""
        # This would need more sophisticated analysis
        # For now, return generic type
        return "shirt"
    
    def _infer_bottom_type(self, region: ClothingRegion) -> str:
        """Infer bottom type from region."""
        # This would need more sophisticated analysis
        return "pants"
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get color name from RGB value."""
        # Find closest named color
        min_dist = float("inf")
        closest_name = "unknown"
        
        for color_rgb, name in self.COLOR_NAMES.items():
            dist = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        
        return closest_name
    
    def close(self):
        """Release resources."""
        if self._segmenter and self._segmenter != "fallback":
            self._segmenter.close()
            self._segmenter = None
