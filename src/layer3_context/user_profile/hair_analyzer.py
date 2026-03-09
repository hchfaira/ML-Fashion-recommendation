"""
Hair Analyzer
=============

Segments hair region and extracts hair color.
Uses MediaPipe Selfie Segmentation for hair detection.
"""

import logging
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image

from .models import HairAnalysis, HairColor

logger = logging.getLogger(__name__)


class HairAnalyzer:
    """
    Analyzes hair region and extracts hair color.
    
    Uses segmentation to isolate hair region,
    then extracts dominant color for classification.
    """
    
    # Hair color LAB ranges
    # Organized by L* (lightness), a* (red-green), b* (yellow-blue)
    HAIR_COLOR_RANGES = {
        HairColor.BLACK: {
            "l_range": (0, 25),
            "description": "Very dark, low lightness"
        },
        HairColor.DARK_BROWN: {
            "l_range": (15, 35),
            "a_range": (3, 15),
            "b_range": (10, 30),
            "description": "Dark with brown undertone"
        },
        HairColor.MEDIUM_BROWN: {
            "l_range": (30, 50),
            "a_range": (5, 20),
            "b_range": (15, 35),
            "description": "Medium brown"
        },
        HairColor.LIGHT_BROWN: {
            "l_range": (45, 65),
            "a_range": (5, 18),
            "b_range": (20, 40),
            "description": "Light brown"
        },
        HairColor.BLONDE: {
            "l_range": (65, 90),
            "a_range": (-2, 15),
            "b_range": (25, 50),
            "description": "Blonde, high lightness with yellow"
        },
        HairColor.RED: {
            "l_range": (30, 60),
            "a_range": (20, 45),
            "b_range": (20, 45),
            "description": "Red, high a* value"
        },
        HairColor.AUBURN: {
            "l_range": (25, 45),
            "a_range": (15, 35),
            "b_range": (15, 35),
            "description": "Auburn, red-brown mix"
        },
        HairColor.GREY: {
            "l_range": (40, 80),
            "a_range": (-5, 5),
            "b_range": (-5, 10),
            "description": "Grey, neutral with high L"
        },
        HairColor.WHITE: {
            "l_range": (85, 100),
            "description": "Very light, nearly white"
        },
        HairColor.PLATINUM: {
            "l_range": (80, 95),
            "a_range": (-5, 5),
            "b_range": (-5, 15),
            "description": "Platinum blonde"
        },
    }
    
    def __init__(self, model_selection: int = 1):
        """
        Initialize hair analyzer.
        
        Args:
            model_selection: 0 for general, 1 for landscape (better for portraits)
        """
        self.model_selection = model_selection
        self._segmenter = None
        self._mp = None
    
    def _ensure_mediapipe(self):
        """Lazy load MediaPipe Image Segmenter (Tasks API)."""
        if self._segmenter is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                self._mp = mp
                
                # Get model path
                model_path = self._get_model_path()
                
                # Create options
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.ImageSegmenterOptions(
                    base_options=base_options,
                    output_category_mask=True,
                )
                
                self._segmenter = vision.ImageSegmenter.create_from_options(options)
                logger.info("MediaPipe ImageSegmenter initialized")
            except ImportError:
                logger.error("MediaPipe not installed. Install with: pip install mediapipe")
                raise ImportError("MediaPipe is required for hair analysis")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe: {e}")
                raise
    
    def _get_model_path(self) -> str:
        """Get path to selfie segmentation model, downloading if needed."""
        import urllib.request
        from pathlib import Path
        
        cache_dir = Path.home() / ".cache" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "selfie_segmenter.tflite"
        
        if not model_path.exists():
            logger.info("Downloading selfie segmenter model...")
            url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Model downloaded to {model_path}")
        
        return str(model_path)
    
    def analyze(
        self,
        image: Image.Image,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[HairAnalysis]:
        """
        Analyze hair from image.
        
        Args:
            image: PIL Image containing person
            face_bbox: Optional face bounding box (x, y, w, h) for region targeting
            
        Returns:
            HairAnalysis with hair color, or None if no hair detected
        """
        # Get hair mask
        hair_mask = self._segment_hair(image, face_bbox)
        
        if hair_mask is None or np.sum(hair_mask) < 100:
            logger.warning("No hair region detected")
            return None
        
        # Extract dominant hair color
        img_rgb = np.array(image.convert("RGB"))
        hair_color_rgb, hair_color_lab = self._extract_dominant_color(img_rgb, hair_mask)
        
        if hair_color_rgb is None:
            return None
        
        # Classify hair color
        hair_color = self._classify_hair_color(hair_color_lab)
        
        # Compute coverage
        coverage = np.sum(hair_mask > 0.5) / hair_mask.size
        
        return HairAnalysis(
            hair_color=hair_color,
            dominant_hair_rgb=hair_color_rgb,
            dominant_hair_lab=hair_color_lab,
            hair_mask_coverage=coverage,
            hair_color_confidence=self._estimate_confidence(hair_mask, hair_color_lab),
        )
    
    def _segment_hair(
        self,
        image: Image.Image,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[np.ndarray]:
        """
        Segment hair region from image.
        
        Uses MediaPipe for person segmentation, then estimates
        hair region as the area above the face.
        """
        self._ensure_mediapipe()
        
        img_rgb = np.array(image.convert("RGB"))
        height, width = img_rgb.shape[:2]
        
        # Create MediaPipe Image and get segmentation
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=img_rgb)
        segmentation_result = self._segmenter.segment(mp_image)
        
        # Get category mask (person mask)
        if not segmentation_result.category_mask:
            return None
        
        # Convert to numpy - category mask is a MediaPipe Image
        # Use numpy_view() or convert properly
        category_mask_raw = segmentation_result.category_mask.numpy_view()
        
        # Handle different output formats
        if len(category_mask_raw.shape) == 3:
            # If 3D, take first channel
            category_mask_raw = category_mask_raw[:, :, 0]
        
        # Convert to float32 person mask (person pixels are non-zero)
        person_mask = (category_mask_raw > 0).astype(np.float32)
        
        # Estimate hair region
        # Hair is typically above the face bounding box
        hair_mask = np.zeros_like(person_mask)
        
        if face_bbox:
            fx, fy, fw, fh = face_bbox
            # Hair region: above face, within face width (with margin)
            hair_top = 0
            hair_bottom = fy + int(fh * 0.2)  # Top 20% of face may have hair
            hair_left = max(0, fx - int(fw * 0.3))
            hair_right = min(width, fx + fw + int(fw * 0.3))
            
            # Combine with person mask
            region_mask = np.zeros_like(person_mask)
            region_mask[hair_top:hair_bottom, hair_left:hair_right] = 1.0
            hair_mask = person_mask * region_mask
        else:
            # Without face bbox, estimate hair as top portion of person
            # Find top of person mask
            person_rows = np.any(person_mask > 0.5, axis=1)
            if not np.any(person_rows):
                return None
            
            top_row = np.argmax(person_rows)
            person_height = np.sum(person_rows)
            
            # Hair is roughly top 20% of person
            hair_end = top_row + int(person_height * 0.25)
            
            hair_mask[:hair_end, :] = person_mask[:hair_end, :]
        
        # Apply threshold
        hair_mask = (hair_mask > 0.5).astype(np.float32)
        
        return hair_mask
    
    def _extract_dominant_color(
        self,
        img_rgb: np.ndarray,
        hair_mask: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[float, float, float]]]:
        """Extract dominant color from hair region."""
        # Get hair pixels
        mask_bool = hair_mask > 0.5
        
        if img_rgb.shape[:2] != hair_mask.shape:
            # Resize mask to match image
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((hair_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((img_rgb.shape[1], img_rgb.shape[0]))
            mask_bool = np.array(mask_pil) > 127
        
        hair_pixels = img_rgb[mask_bool]
        
        if len(hair_pixels) < 10:
            return None, None
        
        # Use k-means to find dominant color
        dominant_rgb = self._kmeans_dominant(hair_pixels, k=3)
        
        # Convert to LAB
        dominant_lab = self._rgb_to_lab(dominant_rgb)
        
        return dominant_rgb, dominant_lab
    
    def _kmeans_dominant(
        self,
        pixels: np.ndarray,
        k: int = 3
    ) -> Tuple[int, int, int]:
        """Find dominant color using k-means clustering."""
        # Simple k-means implementation
        pixels = pixels.astype(np.float32)
        n_pixels = len(pixels)
        
        if n_pixels < k:
            return tuple(np.median(pixels, axis=0).astype(int))
        
        # Initialize centroids randomly
        np.random.seed(42)
        indices = np.random.choice(n_pixels, k, replace=False)
        centroids = pixels[indices].copy()
        
        for _ in range(10):  # Max iterations
            # Assign pixels to nearest centroid
            distances = np.zeros((n_pixels, k))
            for i in range(k):
                distances[:, i] = np.sum((pixels - centroids[i]) ** 2, axis=1)
            
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = []
            for i in range(k):
                cluster_pixels = pixels[labels == i]
                if len(cluster_pixels) > 0:
                    new_centroids.append(cluster_pixels.mean(axis=0))
                else:
                    new_centroids.append(centroids[i])
            
            centroids = np.array(new_centroids)
        
        # Find largest cluster (dominant color)
        cluster_sizes = [np.sum(labels == i) for i in range(k)]
        dominant_idx = np.argmax(cluster_sizes)
        dominant_color = centroids[dominant_idx]
        
        return tuple(dominant_color.astype(int))
    
    def _classify_hair_color(
        self,
        lab: Tuple[float, float, float]
    ) -> HairColor:
        """Classify hair color based on LAB values."""
        l, a, b = lab
        
        # Check for extreme lightness first
        if l >= 85:
            if a < 5 and b < 15:
                return HairColor.WHITE
            return HairColor.PLATINUM
        
        if l < 20:
            return HairColor.BLACK
        
        # Check for red (high a*)
        if a >= 20:
            if l >= 35:
                return HairColor.RED
            return HairColor.AUBURN
        
        # Check for grey (neutral a*, b*)
        if -5 <= a <= 5 and -5 <= b <= 15 and l >= 40:
            return HairColor.GREY
        
        # Check for blonde (high L*, high b*)
        if l >= 60 and b >= 20:
            return HairColor.BLONDE
        
        # Brown shades based on lightness
        if l >= 45:
            return HairColor.LIGHT_BROWN
        elif l >= 30:
            return HairColor.MEDIUM_BROWN
        else:
            return HairColor.DARK_BROWN
    
    def _estimate_confidence(
        self,
        hair_mask: np.ndarray,
        hair_lab: Tuple[float, float, float]
    ) -> float:
        """Estimate confidence based on mask quality and color distinctness."""
        # Check mask size
        mask_coverage = np.sum(hair_mask > 0.5) / hair_mask.size
        
        # Small coverage = less confident
        if mask_coverage < 0.01:
            return 0.3
        elif mask_coverage < 0.05:
            return 0.5
        elif mask_coverage < 0.15:
            return 0.7
        
        # Check if color is distinctive
        l, a, b = hair_lab
        
        # Very neutral colors are harder to classify
        if abs(a) < 3 and abs(b) < 5:
            return 0.6
        
        return 0.85
    
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
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex color code."""
        return "#{:02X}{:02X}{:02X}".format(*rgb)
    
    def close(self):
        """Release resources."""
        if self._segmenter:
            self._segmenter.close()
            self._segmenter = None
