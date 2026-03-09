"""
Color Analyzer
==============

Detects skin tone and undertone from face region.
Uses perceptual color space (LAB) for accurate analysis.
"""

import logging
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
from dataclasses import dataclass

from .models import SkinAnalysis, SkinTone, Undertone

logger = logging.getLogger(__name__)


@dataclass
class ColorSample:
    """A color sample with lab and rgb values."""
    lab: Tuple[float, float, float]  # L, a, b
    rgb: Tuple[int, int, int]        # R, G, B
    hex_code: str


class ColorAnalyzer:
    """
    Analyzes skin tone and undertone from skin regions.
    
    Uses LAB color space for perceptual accuracy.
    Samples from multiple face regions for robust detection.
    """
    
    # Skin tone L* ranges in LAB space
    # L* = 0 (black) to 100 (white)
    SKIN_TONE_RANGES = {
        SkinTone.VERY_LIGHT: (85, 100),
        SkinTone.LIGHT: (70, 85),
        SkinTone.MEDIUM_LIGHT: (55, 70),
        SkinTone.MEDIUM: (40, 55),
        SkinTone.MEDIUM_DARK: (25, 40),
        SkinTone.DARK: (15, 25),
        SkinTone.VERY_DARK: (0, 15),
    }
    
    # Undertone detection thresholds in LAB space
    # a* = green(-) to red(+), b* = blue(-) to yellow(+)
    UNDERTONE_THRESHOLDS = {
        "warm_a_min": 8,    # High red component
        "warm_b_min": 15,   # High yellow component
        "cool_a_max": 5,    # Low red component
        "cool_b_max": 10,   # Lower yellow component
    }
    
    # Face sampling regions (relative coordinates)
    # These regions typically have exposed skin
    SAMPLE_REGIONS = [
        {"name": "forehead", "x": 0.5, "y": 0.2, "size": 0.15},
        {"name": "left_cheek", "x": 0.3, "y": 0.5, "size": 0.12},
        {"name": "right_cheek", "x": 0.7, "y": 0.5, "size": 0.12},
        {"name": "chin", "x": 0.5, "y": 0.85, "size": 0.10},
    ]
    
    def __init__(self, face_bbox: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize color analyzer.
        
        Args:
            face_bbox: Optional face bounding box (x, y, width, height)
        """
        self.face_bbox = face_bbox
    
    def analyze(
        self,
        image: Image.Image,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> SkinAnalysis:
        """
        Analyze skin color from image.
        
        Args:
            image: PIL Image (full image or face crop)
            face_bbox: Optional face bounding box (x, y, w, h)
            
        Returns:
            SkinAnalysis with skin tone and undertone
        """
        bbox = face_bbox or self.face_bbox
        
        # Crop to face region if bbox provided
        if bbox:
            x, y, w, h = bbox
            face_image = image.crop((x, y, x + w, y + h))
        else:
            face_image = image
        
        # Sample skin colors from multiple regions
        samples = self._sample_skin_colors(face_image)
        
        if not samples:
            logger.warning("Could not sample skin colors, using defaults")
            return SkinAnalysis(
                skin_tone=SkinTone.MEDIUM,
                undertone=Undertone.NEUTRAL,
                dominant_skin_lab=(50, 15, 20),
                dominant_skin_rgb=(180, 150, 130),
                skin_tone_confidence=0.3,
                undertone_confidence=0.3,
            )
        
        # Compute dominant color
        avg_lab = self._compute_average_lab(samples)
        avg_rgb = self._lab_to_rgb(avg_lab)
        
        # Classify skin tone
        skin_tone = self._classify_skin_tone(avg_lab)
        
        # Determine undertone
        undertone = self._determine_undertone(avg_lab)
        
        # Confidence based on sample consistency
        confidence = self._compute_confidence(samples)
        
        return SkinAnalysis(
            skin_tone=skin_tone,
            undertone=undertone,
            dominant_skin_lab=avg_lab,
            dominant_skin_rgb=avg_rgb,
            skin_tone_confidence=confidence,
            undertone_confidence=confidence,
        )
    
    def _sample_skin_colors(
        self,
        face_image: Image.Image
    ) -> List[ColorSample]:
        """Sample skin colors from predefined face regions."""
        samples = []
        img_rgb = np.array(face_image.convert("RGB"))
        height, width = img_rgb.shape[:2]
        
        for region in self.SAMPLE_REGIONS:
            # Compute region center and size
            cx = int(region["x"] * width)
            cy = int(region["y"] * height)
            size = int(region["size"] * min(width, height))
            half_size = size // 2
            
            # Extract region
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(width, cx + half_size)
            y2 = min(height, cy + half_size)
            
            region_pixels = img_rgb[y1:y2, x1:x2]
            
            if region_pixels.size == 0:
                continue
            
            # Filter out non-skin pixels (simple heuristic)
            filtered = self._filter_skin_pixels(region_pixels)
            
            if len(filtered) == 0:
                continue
            
            # Get median color (more robust than mean)
            median_rgb = np.median(filtered, axis=0).astype(int)
            median_rgb = tuple(median_rgb)
            
            # Convert to LAB
            lab = self._rgb_to_lab(median_rgb)
            hex_code = self._rgb_to_hex(median_rgb)
            
            samples.append(ColorSample(
                lab=lab,
                rgb=median_rgb,
                hex_code=hex_code
            ))
        
        return samples
    
    def _filter_skin_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """
        Filter pixels to keep only skin-like colors.
        
        Uses YCrCb color space thresholds.
        """
        # Reshape for easier processing
        flat = pixels.reshape(-1, 3)
        
        if len(flat) == 0:
            return flat
        
        # Convert to YCrCb
        r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
        
        # YCrCb conversion
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = (r - y) * 0.713 + 128
        cb = (b - y) * 0.564 + 128
        
        # Skin detection thresholds in YCrCb
        # These are empirical values that work across skin tones
        skin_mask = (
            (cr >= 133) & (cr <= 173) &
            (cb >= 77) & (cb <= 127) &
            (y > 40)
        )
        
        filtered = flat[skin_mask]
        
        # If filter is too aggressive, return some pixels
        if len(filtered) < len(flat) * 0.1:
            # Relax constraints
            return flat[y > 30]
        
        return filtered
    
    def _compute_average_lab(
        self,
        samples: List[ColorSample]
    ) -> Tuple[float, float, float]:
        """Compute average LAB color from samples."""
        if not samples:
            return (50.0, 15.0, 20.0)  # Default neutral skin
        
        l_sum = sum(s.lab[0] for s in samples)
        a_sum = sum(s.lab[1] for s in samples)
        b_sum = sum(s.lab[2] for s in samples)
        n = len(samples)
        
        return (l_sum / n, a_sum / n, b_sum / n)
    
    def _classify_skin_tone(
        self,
        lab: Tuple[float, float, float]
    ) -> SkinTone:
        """Classify skin tone based on L* value."""
        l_value = lab[0]
        
        for tone, (min_l, max_l) in self.SKIN_TONE_RANGES.items():
            if min_l <= l_value < max_l:
                return tone
        
        # Edge cases
        if l_value >= 100:
            return SkinTone.VERY_LIGHT
        if l_value <= 0:
            return SkinTone.VERY_DARK
        
        return SkinTone.MEDIUM
    
    def _determine_undertone(
        self,
        lab: Tuple[float, float, float]
    ) -> Undertone:
        """
        Determine undertone from LAB values.
        
        a* > 0 = more red (warm)
        b* > 0 = more yellow (warm)
        """
        l, a, b = lab
        
        # Warm undertone: high red and/or yellow
        if (a >= self.UNDERTONE_THRESHOLDS["warm_a_min"] and 
            b >= self.UNDERTONE_THRESHOLDS["warm_b_min"]):
            return Undertone.WARM
        
        # Cool undertone: lower red and yellow
        if (a <= self.UNDERTONE_THRESHOLDS["cool_a_max"] or 
            b <= self.UNDERTONE_THRESHOLDS["cool_b_max"]):
            return Undertone.COOL
        
        # Olive: moderate a, moderate-high b (greenish tint)
        if -5 < a < 10 and 15 < b < 30:
            return Undertone.OLIVE
        
        # Neutral: balanced
        return Undertone.NEUTRAL
    
    def _compute_confidence(self, samples: List[ColorSample]) -> float:
        """Compute confidence based on sample consistency."""
        if len(samples) < 2:
            return 0.5
        
        # Compute standard deviation of L values
        l_values = [s.lab[0] for s in samples]
        l_std = np.std(l_values)
        
        # Low std = high consistency = high confidence
        if l_std < 5:
            return 0.95
        elif l_std < 10:
            return 0.85
        elif l_std < 15:
            return 0.70
        else:
            return 0.50
    
    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to LAB color space."""
        r, g, b = [x / 255.0 for x in rgb]
        
        # RGB to XYZ
        def gamma(c):
            return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
        
        r, g, b = gamma(r), gamma(g), gamma(b)
        
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # XYZ to LAB (D65 illuminant)
        x /= 0.95047
        y /= 1.00000
        z /= 1.08883
        
        def f(t):
            return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16 / 116)
        
        l = 116 * f(y) - 16
        a = 500 * (f(x) - f(y))
        b_val = 200 * (f(y) - f(z))
        
        return (l, a, b_val)
    
    def _lab_to_rgb(self, lab: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert LAB to RGB color space."""
        l, a, b = lab
        
        # LAB to XYZ
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b / 200
        
        def f_inv(t):
            return t ** 3 if t > 0.206893 else (t - 16 / 116) / 7.787
        
        x = 0.95047 * f_inv(x)
        y = 1.00000 * f_inv(y)
        z = 1.08883 * f_inv(z)
        
        # XYZ to RGB
        r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
        g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
        b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
        
        def gamma_inv(c):
            return 1.055 * (c ** (1/2.4)) - 0.055 if c > 0.0031308 else 12.92 * c
        
        r = int(max(0, min(255, gamma_inv(r) * 255)))
        g = int(max(0, min(255, gamma_inv(g) * 255)))
        b = int(max(0, min(255, gamma_inv(b) * 255)))
        
        return (r, g, b)
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex color code."""
        return "#{:02X}{:02X}{:02X}".format(*rgb)
