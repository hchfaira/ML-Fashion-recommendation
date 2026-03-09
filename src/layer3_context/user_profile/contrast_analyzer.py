"""
Contrast Analyzer
=================

Computes contrast level between features (hair, skin, eyes).
Uses color science principles for perceptual contrast.
"""

import logging
from typing import Optional, Tuple
import numpy as np

from .models import (
    ContrastLevel,
    SkinAnalysis,
    HairAnalysis,
)

logger = logging.getLogger(__name__)


class ContrastAnalyzer:
    """
    Analyzes contrast between user features.
    
    Contrast is computed between:
    - Hair and skin
    - Hair and eyes (if available)
    
    Uses Delta E (LAB color difference) for perceptual accuracy.
    """
    
    # Contrast level thresholds (Delta E values)
    CONTRAST_THRESHOLDS = {
        "very_low": 10,     # Delta E < 10: barely perceptible
        "low": 25,          # Delta E 10-25: low contrast
        "medium": 45,       # Delta E 25-45: medium contrast
        "high": 70,         # Delta E 45-70: high contrast
        # > 70: very high contrast
    }
    
    # Default LAB values for estimation
    DEFAULT_VALUES = {
        "light_skin": (70, 15, 20),
        "medium_skin": (55, 15, 20),
        "dark_skin": (35, 15, 20),
        "black_hair": (15, 0, 0),
        "brown_hair": (35, 10, 15),
        "blonde_hair": (70, 5, 25),
    }
    
    def analyze(
        self,
        skin_analysis: Optional[SkinAnalysis] = None,
        hair_analysis: Optional[HairAnalysis] = None,
        skin_lab: Optional[Tuple[float, float, float]] = None,
        hair_lab: Optional[Tuple[float, float, float]] = None,
    ) -> ContrastLevel:
        """
        Analyze contrast between hair and skin.
        
        Args:
            skin_analysis: Skin analysis with skin tone
            hair_analysis: Hair analysis with hair color
            skin_lab: Optional direct LAB values for skin
            hair_lab: Optional direct LAB values for hair
            
        Returns:
            ContrastLevel classification
        """
        # Get LAB values
        if skin_lab is None:
            if skin_analysis and skin_analysis.dominant_skin_lab:
                skin_lab = skin_analysis.dominant_skin_lab
            else:
                # Estimate from skin tone if available
                skin_lab = self._estimate_skin_lab(skin_analysis)
        
        if hair_lab is None:
            if hair_analysis and hair_analysis.dominant_hair_rgb:
                hair_lab = self._rgb_to_lab(hair_analysis.dominant_hair_rgb)
            else:
                # Estimate from hair color if available
                hair_lab = self._estimate_hair_lab(hair_analysis)
        
        # Compute Delta E (CIE76)
        delta_e = self._compute_delta_e(skin_lab, hair_lab)
        
        # Classify contrast level
        return self._classify_contrast(delta_e)
    
    def compute_feature_contrast(
        self,
        feature1_lab: Tuple[float, float, float],
        feature2_lab: Tuple[float, float, float],
    ) -> Tuple[float, ContrastLevel]:
        """
        Compute contrast between any two features.
        
        Args:
            feature1_lab: LAB color of first feature
            feature2_lab: LAB color of second feature
            
        Returns:
            Tuple of (delta_e value, contrast level)
        """
        delta_e = self._compute_delta_e(feature1_lab, feature2_lab)
        level = self._classify_contrast(delta_e)
        return delta_e, level
    
    def _compute_delta_e(
        self,
        lab1: Tuple[float, float, float],
        lab2: Tuple[float, float, float]
    ) -> float:
        """
        Compute Delta E (CIE76) color difference.
        
        Delta E represents perceptual color difference:
        - < 1: Not perceptible by human eyes
        - 1-2: Perceptible through close observation
        - 2-10: Perceptible at a glance
        - 11-49: Colors are more similar than opposite
        - 100: Opposite colors
        """
        l1, a1, b1 = lab1
        l2, a2, b2 = lab2
        
        delta_e = np.sqrt(
            (l2 - l1) ** 2 +
            (a2 - a1) ** 2 +
            (b2 - b1) ** 2
        )
        
        return delta_e
    
    def _compute_delta_e_2000(
        self,
        lab1: Tuple[float, float, float],
        lab2: Tuple[float, float, float]
    ) -> float:
        """
        Compute Delta E 2000 (more perceptually uniform).
        
        This is a more advanced formula that accounts for
        human perception variations across the color space.
        """
        l1, a1, b1 = lab1
        l2, a2, b2 = lab2
        
        # Average L
        l_bar = (l1 + l2) / 2
        
        # Compute C (chroma)
        c1 = np.sqrt(a1**2 + b1**2)
        c2 = np.sqrt(a2**2 + b2**2)
        c_bar = (c1 + c2) / 2
        
        # Compute G
        g = 0.5 * (1 - np.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
        
        # Adjust a values
        a1_prime = a1 * (1 + g)
        a2_prime = a2 * (1 + g)
        
        # Compute C' and h'
        c1_prime = np.sqrt(a1_prime**2 + b1**2)
        c2_prime = np.sqrt(a2_prime**2 + b2**2)
        
        h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
        h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
        
        # Delta values
        delta_l = l2 - l1
        delta_c = c2_prime - c1_prime
        
        # Delta h
        if c1_prime * c2_prime == 0:
            delta_h = 0
        else:
            dh = h2_prime - h1_prime
            if dh > 180:
                dh -= 360
            elif dh < -180:
                dh += 360
            delta_h = 2 * np.sqrt(c1_prime * c2_prime) * np.sin(np.radians(dh / 2))
        
        # Averages
        l_bar_prime = l_bar
        c_bar_prime = (c1_prime + c2_prime) / 2
        
        if c1_prime * c2_prime == 0:
            h_bar_prime = h1_prime + h2_prime
        else:
            if abs(h1_prime - h2_prime) <= 180:
                h_bar_prime = (h1_prime + h2_prime) / 2
            elif h1_prime + h2_prime < 360:
                h_bar_prime = (h1_prime + h2_prime + 360) / 2
            else:
                h_bar_prime = (h1_prime + h2_prime - 360) / 2
        
        # Weighting functions
        t = (1 - 0.17 * np.cos(np.radians(h_bar_prime - 30)) +
             0.24 * np.cos(np.radians(2 * h_bar_prime)) +
             0.32 * np.cos(np.radians(3 * h_bar_prime + 6)) -
             0.20 * np.cos(np.radians(4 * h_bar_prime - 63)))
        
        s_l = 1 + (0.015 * (l_bar_prime - 50)**2) / np.sqrt(20 + (l_bar_prime - 50)**2)
        s_c = 1 + 0.045 * c_bar_prime
        s_h = 1 + 0.015 * c_bar_prime * t
        
        # Rotation term
        delta_theta = 30 * np.exp(-((h_bar_prime - 275) / 25)**2)
        r_c = 2 * np.sqrt(c_bar_prime**7 / (c_bar_prime**7 + 25**7))
        r_t = -r_c * np.sin(np.radians(2 * delta_theta))
        
        # Final Delta E 2000
        k_l, k_c, k_h = 1, 1, 1  # Weighting factors
        
        delta_e = np.sqrt(
            (delta_l / (k_l * s_l))**2 +
            (delta_c / (k_c * s_c))**2 +
            (delta_h / (k_h * s_h))**2 +
            r_t * (delta_c / (k_c * s_c)) * (delta_h / (k_h * s_h))
        )
        
        return delta_e
    
    def _classify_contrast(self, delta_e: float) -> ContrastLevel:
        """Classify contrast level based on Delta E value."""
        if delta_e < self.CONTRAST_THRESHOLDS["very_low"]:
            return ContrastLevel.VERY_LOW
        elif delta_e < self.CONTRAST_THRESHOLDS["low"]:
            return ContrastLevel.LOW
        elif delta_e < self.CONTRAST_THRESHOLDS["medium"]:
            return ContrastLevel.MEDIUM
        elif delta_e < self.CONTRAST_THRESHOLDS["high"]:
            return ContrastLevel.HIGH
        else:
            return ContrastLevel.VERY_HIGH
    
    def _estimate_skin_lab(
        self,
        skin_analysis: Optional[SkinAnalysis]
    ) -> Tuple[float, float, float]:
        """Estimate skin LAB values from analysis."""
        if skin_analysis is None:
            return self.DEFAULT_VALUES["medium_skin"]
        
        # Use skin tone to estimate
        from .models import SkinTone
        
        tone = skin_analysis.skin_tone
        
        if tone in [SkinTone.VERY_LIGHT, SkinTone.LIGHT]:
            return self.DEFAULT_VALUES["light_skin"]
        elif tone in [SkinTone.MEDIUM_LIGHT, SkinTone.MEDIUM]:
            return self.DEFAULT_VALUES["medium_skin"]
        else:
            return self.DEFAULT_VALUES["dark_skin"]
    
    def _estimate_hair_lab(
        self,
        hair_analysis: Optional[HairAnalysis]
    ) -> Tuple[float, float, float]:
        """Estimate hair LAB values from analysis."""
        if hair_analysis is None:
            return self.DEFAULT_VALUES["brown_hair"]
        
        from .models import HairColor
        
        color = hair_analysis.hair_color
        
        if color in [HairColor.BLACK]:
            return self.DEFAULT_VALUES["black_hair"]
        elif color in [HairColor.BLONDE, HairColor.PLATINUM, HairColor.WHITE]:
            return self.DEFAULT_VALUES["blonde_hair"]
        else:
            return self.DEFAULT_VALUES["brown_hair"]
    
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
        
        l_val = 116 * f(y) - 16
        a_val = 500 * (f(x) - f(y))
        b_val = 200 * (f(y) - f(z))
        
        return (l_val, a_val, b_val)
