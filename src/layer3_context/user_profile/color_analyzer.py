"""
Color Analyzer
==============

Detects skin tone and undertone from face region.
Uses perceptual color space (LAB) for accurate analysis.
"""

import logging
import math
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image
from dataclasses import dataclass

from .models import SkinAnalysis, SkinTone, Undertone

logger = logging.getLogger(__name__)

# Conditional import – MediaPipe FaceMesh is optional
try:
    import mediapipe as mp
    _FACE_MESH_AVAILABLE = True
except ImportError:
    _FACE_MESH_AVAILABLE = False


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
        
        # ---- 12-season colour analysis ----
        chroma = self._compute_chroma(avg_lab)
        depth = self._skin_tone_to_depth(skin_tone)
        season_sub, season_confidence = self._classify_season_12(
            undertone, depth, chroma
        )
        
        return SkinAnalysis(
            skin_tone=skin_tone,
            undertone=undertone,
            dominant_skin_lab=avg_lab,
            dominant_skin_rgb=avg_rgb,
            skin_tone_confidence=confidence,
            undertone_confidence=confidence,
            chroma=chroma,
            season_sub=season_sub,
            season_confidence=season_confidence,
        )
    
    def _sample_skin_colors(
        self,
        face_image: Image.Image
    ) -> List[ColorSample]:
        """Sample skin colors from face regions (FaceMesh when available, else fixed regions)."""
        regions = self._detect_face_zones(face_image)
        
        samples = []
        img_rgb = np.array(face_image.convert("RGB"))
        height, width = img_rgb.shape[:2]
        
        for region in regions:
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

    # ------------------------------------------------------------------
    # MediaPipe FaceMesh – adaptive face zone detection
    # ------------------------------------------------------------------

    # FaceMesh landmark indices for skin zones (468-point model).
    _FOREHEAD_LANDMARKS = [10, 67, 109, 338, 297]
    _LEFT_CHEEK_LANDMARKS = [116, 123, 147, 187, 205]
    _RIGHT_CHEEK_LANDMARKS = [345, 352, 376, 411, 425]
    _CHIN_LANDMARKS = [152, 175, 199, 200, 18]

    def _detect_face_zones(
        self,
        face_image: Image.Image,
    ) -> List[Dict[str, float]]:
        """
        Return sampling regions as dicts with keys *x*, *y*, *size* (all relative).

        When MediaPipe FaceMesh is available the zones are derived from actual
        landmark positions; otherwise we fall back to the fixed ``SAMPLE_REGIONS``.
        """
        if not _FACE_MESH_AVAILABLE:
            return self.SAMPLE_REGIONS

        try:
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
            img_rgb = np.array(face_image.convert("RGB"))
            results = face_mesh.process(img_rgb)
            face_mesh.close()

            if not results.multi_face_landmarks:
                logger.debug("FaceMesh detected no face – falling back to fixed regions")
                return self.SAMPLE_REGIONS

            lms = results.multi_face_landmarks[0].landmark

            def _zone_center(indices):
                xs = [lms[i].x for i in indices if i < len(lms)]
                ys = [lms[i].y for i in indices if i < len(lms)]
                if not xs:
                    return None
                return (sum(xs) / len(xs), sum(ys) / len(ys))

            zones = []
            for name, indices, default_size in [
                ("forehead", self._FOREHEAD_LANDMARKS, 0.15),
                ("left_cheek", self._LEFT_CHEEK_LANDMARKS, 0.12),
                ("right_cheek", self._RIGHT_CHEEK_LANDMARKS, 0.12),
                ("chin", self._CHIN_LANDMARKS, 0.10),
            ]:
                center = _zone_center(indices)
                if center:
                    zones.append({"name": name, "x": center[0], "y": center[1], "size": default_size})

            if not zones:
                return self.SAMPLE_REGIONS
            return zones

        except Exception as exc:  # noqa: BLE001 – never crash, just fall back
            logger.warning("FaceMesh detection failed (%s) – using fixed regions", exc)
            return self.SAMPLE_REGIONS

    # ------------------------------------------------------------------
    # 12-season colour classification
    # ------------------------------------------------------------------

    # Mapping from (undertone_bucket, depth, chroma) → season sub-name
    _SEASON_12_MATRIX: Dict[Tuple[str, str, str], str] = {
        ("warm", "light", "clear"):  "Light Spring",
        ("warm", "light", "muted"):  "True Spring",
        ("warm", "medium", "clear"): "Warm Spring",
        ("warm", "medium", "muted"): "True Autumn",
        ("warm", "deep", "clear"):   "Deep Autumn",
        ("warm", "deep", "muted"):   "Warm Autumn",
        ("cool", "light", "clear"):  "Light Summer",
        ("cool", "light", "muted"):  "True Summer",
        ("cool", "medium", "clear"): "Cool Summer",
        ("cool", "medium", "muted"): "True Winter",
        ("cool", "deep", "clear"):   "Deep Winter",
        ("cool", "deep", "muted"):   "Cool Winter",
    }

    @staticmethod
    def _compute_chroma(lab: Tuple[float, float, float]) -> str:
        """Return ``'clear'`` or ``'muted'`` based on LAB chroma distance."""
        _, a, b = lab
        c = math.sqrt(a * a + b * b)
        return "clear" if c >= 20 else "muted"

    @staticmethod
    def _skin_tone_to_depth(tone: SkinTone) -> str:
        """Map a :class:`SkinTone` value to a three-level depth bucket."""
        light = {SkinTone.VERY_LIGHT, SkinTone.LIGHT}
        medium = {SkinTone.MEDIUM_LIGHT, SkinTone.MEDIUM}
        # MEDIUM_DARK, DARK, VERY_DARK → deep
        if tone in light:
            return "light"
        if tone in medium:
            return "medium"
        return "deep"

    def _classify_season_12(
        self,
        undertone: Undertone,
        depth: str,
        chroma: str,
    ) -> Tuple[str, float]:
        """
        Classify into one of 12 colour-seasons.

        Returns:
            (season_sub_name, confidence) – confidence is 0.0-1.0.
        """
        # Map undertone to warm/cool bucket; NEUTRAL and OLIVE → cool as safe default
        if undertone == Undertone.WARM:
            tone_bucket = "warm"
            base_confidence = 0.85
        elif undertone in (Undertone.COOL,):
            tone_bucket = "cool"
            base_confidence = 0.85
        else:
            # Neutral / Olive – less certainty, lean cool
            tone_bucket = "cool"
            base_confidence = 0.55

        key = (tone_bucket, depth, chroma)
        season_sub = self._SEASON_12_MATRIX.get(key)

        if season_sub is None:
            logger.warning("No 12-season match for key %s", key)
            return ("Unknown", 0.0)

        return (season_sub, round(base_confidence, 2))
