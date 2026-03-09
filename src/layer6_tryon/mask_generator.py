"""
Mask Generator for Virtual Try-On
=================================

Generates body region masks for inpainting.
Uses rembg for silhouette detection, with geometric fallback.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional, Tuple, Dict
from io import BytesIO
import logging

from .models import GarmentType

logger = logging.getLogger(__name__)


class MaskGenerator:
    """
    Generates agnostic body masks for virtual try-on.
    
    The mask identifies the region where the garment should be placed,
    allowing the diffusion model to inpaint that area.
    """
    
    def __init__(self, use_rembg: bool = True, cache_silhouettes: bool = True):
        """
        Initialize the mask generator.
        
        Args:
            use_rembg: Whether to use rembg for precise silhouette detection
            cache_silhouettes: Whether to cache person silhouettes
        """
        self.use_rembg = use_rembg
        self.cache_silhouettes = cache_silhouettes
        self._silhouette_cache: Dict[str, np.ndarray] = {}
        self._rembg_available: Optional[bool] = None
    
    def _check_rembg(self) -> bool:
        """Check if rembg is available."""
        if self._rembg_available is not None:
            return self._rembg_available
        
        try:
            from rembg import remove
            self._rembg_available = True
        except ImportError:
            logger.warning("rembg not available, using geometric fallback masks")
            self._rembg_available = False
        
        return self._rembg_available
    
    def generate_mask(
        self,
        person_img: Image.Image,
        garment_type: GarmentType,
        cache_key: Optional[str] = None
    ) -> Image.Image:
        """
        Generate a body region mask for the specified garment type.
        
        Args:
            person_img: PIL Image of the person
            garment_type: Type of garment to mask for
            cache_key: Optional key for caching silhouette
            
        Returns:
            Grayscale mask image (white = region to replace)
        """
        if self.use_rembg and self._check_rembg():
            return self._generate_mask_rembg(person_img, garment_type, cache_key)
        else:
            return self._generate_mask_geometric(person_img, garment_type)
    
    def _generate_mask_rembg(
        self,
        person_img: Image.Image,
        garment_type: GarmentType,
        cache_key: Optional[str] = None
    ) -> Image.Image:
        """Generate mask using rembg for silhouette detection."""
        from rembg import remove
        
        # Check cache
        if cache_key and cache_key in self._silhouette_cache:
            alpha = self._silhouette_cache[cache_key]
        else:
            # Get person silhouette
            buf = BytesIO()
            person_img.save(buf, format="PNG")
            seg_bytes = remove(buf.getvalue())
            seg = Image.open(BytesIO(seg_bytes)).convert("RGBA")
            alpha = np.array(seg)[:, :, 3]
            
            # Cache if requested
            if cache_key and self.cache_silhouettes:
                self._silhouette_cache[cache_key] = alpha
        
        # Find body bounding box
        rows = np.any(alpha > 10, axis=1)
        cols = np.any(alpha > 10, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            logger.warning("No body detected, using geometric fallback")
            return self._generate_mask_geometric(person_img, garment_type)
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h_body = rmax - rmin
        w_body = cmax - cmin
        
        # Define garment region based on type
        mask = Image.new("L", person_img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        region = self._get_region_coords(
            garment_type, cmin, cmax, rmin, rmax, h_body, w_body
        )
        
        draw.rectangle(region, fill=255)
        
        # Soft edges via blur
        mask = mask.filter(ImageFilter.GaussianBlur(10))
        
        logger.debug(f"Generated rembg mask for {garment_type.value}")
        return mask
    
    def _generate_mask_geometric(
        self,
        person_img: Image.Image,
        garment_type: GarmentType
    ) -> Image.Image:
        """Generate mask using simple geometric assumptions."""
        w, h = person_img.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        if garment_type == GarmentType.UPPER_BODY:
            # Torso region: roughly top 20% to 65%
            region = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.65))
        elif garment_type == GarmentType.LOWER_BODY:
            # Hip to ankle region
            region = (int(w * 0.15), int(h * 0.5), int(w * 0.85), int(h * 0.95))
        elif garment_type == GarmentType.FULL_BODY:
            # Full body from shoulders down
            region = (int(w * 0.15), int(h * 0.2), int(w * 0.85), int(h * 0.95))
        else:
            # Default: small center region for accessories
            region = (int(w * 0.35), int(h * 0.4), int(w * 0.65), int(h * 0.6))
        
        draw.rectangle(region, fill=255)
        
        # Apply blur for soft edges
        mask = mask.filter(ImageFilter.GaussianBlur(8))
        
        logger.debug(f"Generated geometric mask for {garment_type.value}")
        return mask
    
    def _get_region_coords(
        self,
        garment_type: GarmentType,
        cmin: int, cmax: int,
        rmin: int, rmax: int,
        h_body: int, w_body: int
    ) -> Tuple[int, int, int, int]:
        """Calculate region coordinates based on garment type and body bounds."""
        
        if garment_type == GarmentType.UPPER_BODY:
            # Torso: from ~20% to ~65% of body height
            return (
                int(cmin + w_body * 0.05),
                int(rmin + h_body * 0.18),
                int(cmax - w_body * 0.05),
                int(rmin + h_body * 0.65),
            )
        
        elif garment_type == GarmentType.LOWER_BODY:
            # Hips to ankles
            return (
                int(cmin),
                int(rmin + h_body * 0.50),
                int(cmax),
                int(rmax),
            )
        
        elif garment_type == GarmentType.FULL_BODY:
            # Full body from shoulders
            return (
                int(cmin + w_body * 0.03),
                int(rmin + h_body * 0.18),
                int(cmax - w_body * 0.03),
                int(rmax),
            )
        
        else:
            # Accessories - small center region
            return (
                int(cmin + w_body * 0.3),
                int(rmin + h_body * 0.3),
                int(cmax - w_body * 0.3),
                int(rmin + h_body * 0.7),
            )
    
    def clear_cache(self):
        """Clear the silhouette cache."""
        self._silhouette_cache.clear()
        logger.debug("Silhouette cache cleared")
