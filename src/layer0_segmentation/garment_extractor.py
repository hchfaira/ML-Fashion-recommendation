"""
Garment Extractor
=================

Extracts individual garments from images as RGBA PNGs with
transparent backgrounds.

Responsibilities:
- Apply mask to original image
- Generate RGBA image with transparency
- Tight crop around garment
- Resize and center on canvas
- Store garment class from GARMENT_TAXONOMY
- Save with proper naming convention
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict
from dataclasses import dataclass

from src.core import get_logger
from .taxonomy import GarmentCategory
from .models import (
    FusedMask, ExtractedGarment, ExtractionResult, BoundingBox
)

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CANVAS_SIZE = 512
DEFAULT_PADDING = 10


@dataclass
class ExtractorConfig:
    """Configuration for garment extraction."""
    canvas_size: int = DEFAULT_CANVAS_SIZE
    padding: int = DEFAULT_PADDING
    maintain_aspect_ratio: bool = True
    center_on_canvas: bool = True
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 0)  # Transparent


# ============================================================================
# Garment Extractor
# ============================================================================

class GarmentExtractor:
    """
    Extracts garments from images using masks.
    
    Produces RGBA images with transparent backgrounds, properly
    cropped and sized for downstream processing.
    """
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        """
        Initialize the garment extractor.
        
        Args:
            config: Extraction configuration
        """
        self.config = config or ExtractorConfig()
        logger.info(
            f"GarmentExtractor initialized "
            f"(canvas_size={self.config.canvas_size})"
        )
    
    def extract(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        mask: FusedMask,
        source_path: Optional[Path] = None
    ) -> ExtractedGarment:
        """
        Extract a single garment from an image using a mask.
        
        Args:
            image: Source image
            mask: Fused mask for the garment
            source_path: Optional path to source image
            
        Returns:
            ExtractedGarment with RGBA image
        """
        # Load image
        image_np = self._load_image(image)
        
        # Apply mask to create RGBA
        rgba = self._apply_mask(image_np, mask.mask)
        
        # Crop to content
        cropped, bbox = self._crop_to_content(rgba, padding=self.config.padding)
        
        # Convert to PIL
        pil_image = Image.fromarray(cropped)
        
        # Resize and center
        final_image = self._resize_and_center(pil_image)
        
        # Create bounding box object
        bounding_box = BoundingBox(
            x1=bbox[0], y1=bbox[1],
            x2=bbox[0] + bbox[2], y2=bbox[1] + bbox[3]
        )
        
        return ExtractedGarment(
            image=final_image,
            category=mask.category,
            label=mask.label,
            confidence=mask.confidence,
            source_path=source_path,
            bounding_box=bounding_box,
            mask=mask.mask,
            area=int(np.sum(mask.mask > 0))
        )
    
    def extract_all(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        masks: List[FusedMask],
        source_path: Optional[Path] = None
    ) -> ExtractionResult:
        """
        Extract all garments from an image.
        
        Args:
            image: Source image
            masks: List of fused masks
            source_path: Optional path to source image
            
        Returns:
            ExtractionResult with all extracted garments
        """
        import time
        start_time = time.time()
        
        garments = []
        errors = []
        
        # Load image once
        image_np = self._load_image(image)
        
        for i, mask in enumerate(masks):
            try:
                garment = self.extract(image_np, mask, source_path)
                garments.append(garment)
            except Exception as e:
                error_msg = f"Failed to extract garment {i} ({mask.label}): {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return ExtractionResult(
            garments=garments,
            source_image_path=source_path,
            processing_time_ms=processing_time,
            errors=errors
        )
    
    def _load_image(
        self, 
        image: Union[np.ndarray, Image.Image, Path, str]
    ) -> np.ndarray:
        """Load image as numpy RGB array."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        # Ensure RGB (not RGBA or grayscale)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        return image
    
    def _apply_mask(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply mask to image to create RGBA with transparency.
        
        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W)
            
        Returns:
            RGBA image (H, W, 4)
        """
        # Ensure mask is correct shape
        if mask.shape != image.shape[:2]:
            # Resize mask to match image
            mask = self._resize_mask(mask, image.shape[:2])
        
        # Ensure mask is binary (0 or 255)
        if mask.max() == 1:
            mask = mask * 255
        
        # Create RGBA
        rgba = np.dstack((image, mask))
        
        return rgba
    
    def _resize_mask(
        self, 
        mask: np.ndarray, 
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Resize mask to target shape."""
        from PIL import Image as PILImage
        
        mask_pil = PILImage.fromarray(mask)
        resized = mask_pil.resize(
            (target_shape[1], target_shape[0]),  # PIL uses (W, H)
            PILImage.NEAREST
        )
        return np.array(resized)
    
    def _crop_to_content(
        self, 
        rgba: np.ndarray, 
        padding: int = 0
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop RGBA image to content bounding box.
        
        Args:
            rgba: RGBA image (H, W, 4)
            padding: Extra padding around content
            
        Returns:
            Tuple of (cropped image, bounding box as x, y, w, h)
        """
        # Get alpha channel
        alpha = rgba[:, :, 3]
        
        # Find non-zero pixels
        coords = np.argwhere(alpha > 0)
        
        if len(coords) == 0:
            # Empty mask, return original
            return rgba, (0, 0, rgba.shape[1], rgba.shape[0])
        
        # Get bounding box
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        
        # Add padding
        h, w = rgba.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding + 1)
        y2 = min(h, y2 + padding + 1)
        
        cropped = rgba[y1:y2, x1:x2]
        bbox = (x1, y1, x2 - x1, y2 - y1)
        
        return cropped, bbox
    
    def _resize_and_center(self, img: Image.Image) -> Image.Image:
        """
        Resize image and center on canvas.
        
        Args:
            img: PIL RGBA image
            
        Returns:
            Resized and centered image on canvas
        """
        w, h = img.size
        canvas_size = self.config.canvas_size
        
        if w == 0 or h == 0:
            return Image.new("RGBA", (canvas_size, canvas_size), self.config.background_color)
        
        if self.config.maintain_aspect_ratio:
            # Calculate scale to fit within canvas
            scale = min(canvas_size / w, canvas_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            img = img.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
        
        if self.config.center_on_canvas:
            canvas = Image.new("RGBA", (canvas_size, canvas_size), self.config.background_color)
            x = (canvas_size - img.width) // 2
            y = (canvas_size - img.height) // 2
            canvas.paste(img, (x, y), img)
            return canvas
        
        return img
    
    def save_garment(
        self,
        garment: ExtractedGarment,
        output_dir: Path,
        index: int = 1
    ) -> Path:
        """
        Save a garment to file with proper naming.
        
        Naming convention: {category}_{index:03d}_{label}.png
        
        Args:
            garment: Extracted garment
            output_dir: Output directory
            index: Index for naming
            
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = garment.get_filename(index)
        output_path = output_dir / filename
        
        garment.save(output_path)
        logger.info(f"Saved: {output_path}")
        
        return output_path
    
    def save_all(
        self,
        result: ExtractionResult,
        output_dir: Path
    ) -> List[Path]:
        """
        Save all garments from an extraction result.
        
        Args:
            result: Extraction result
            output_dir: Output directory
            
        Returns:
            List of saved file paths
        """
        return result.save_all(output_dir)


# ============================================================================
# Convenience Functions
# ============================================================================

def extract_garment(
    image: Union[np.ndarray, Image.Image, Path, str],
    mask: np.ndarray,
    category: GarmentCategory = GarmentCategory.UNKNOWN,
    label: str = "garment",
    canvas_size: int = DEFAULT_CANVAS_SIZE
) -> ExtractedGarment:
    """
    Convenience function to extract a single garment.
    
    Args:
        image: Source image
        mask: Binary mask
        category: Garment category
        label: Garment label
        canvas_size: Output canvas size
        
    Returns:
        ExtractedGarment
    """
    config = ExtractorConfig(canvas_size=canvas_size)
    extractor = GarmentExtractor(config)
    
    fused = FusedMask(
        mask=mask,
        category=category,
        label=label,
        confidence=1.0
    )
    
    return extractor.extract(image, fused)


def extract_and_save(
    image: Union[np.ndarray, Image.Image, Path, str],
    masks: List[FusedMask],
    output_dir: Path,
    canvas_size: int = DEFAULT_CANVAS_SIZE
) -> ExtractionResult:
    """
    Extract all garments and save to directory.
    
    Args:
        image: Source image
        masks: List of fused masks
        output_dir: Output directory
        canvas_size: Output canvas size
        
    Returns:
        ExtractionResult with all extracted garments
    """
    config = ExtractorConfig(canvas_size=canvas_size)
    extractor = GarmentExtractor(config)
    
    result = extractor.extract_all(image, masks)
    result.save_all(output_dir)
    
    return result
