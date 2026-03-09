"""
Layer 6: Virtual Try-On Models
==============================

Data models for virtual try-on operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from PIL import Image


class TryOnBackend(str, Enum):
    """Available try-on backends."""
    CATVTON = "catvton"      # CatVTON (ICLR 2025) - Local GPU
    IDMVTON = "idmvton"      # IDM-VTON (ECCV 2024) - Local GPU
    REPLICATE = "replicate"  # Replicate API - Cloud


class GarmentType(str, Enum):
    """Garment type for try-on positioning."""
    UPPER_BODY = "upper_body"   # Tops, shirts, jackets
    LOWER_BODY = "lower_body"   # Pants, skirts
    FULL_BODY = "dresses"       # Dresses, jumpsuits
    FOOTWEAR = "footwear"       # Shoes (limited support)
    ACCESSORY = "accessory"     # Accessories (limited support)


@dataclass
class TryOnConfig:
    """Configuration for virtual try-on."""
    backend: TryOnBackend = TryOnBackend.CATVTON
    
    # Model settings
    device: str = "cuda"  # "cuda" or "cpu"
    dtype: str = "float16"  # "float16" or "float32"
    
    # Resolution
    target_width: int = 768
    target_height: int = 1024
    
    # Diffusion settings
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.99
    
    # Replicate API settings
    replicate_token: Optional[str] = None
    
    # Output settings
    output_dir: Optional[str] = None
    save_intermediate: bool = False
    create_comparison_panel: bool = True
    
    # Caching
    cache_masks: bool = True
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.replicate_token is None:
            import os
            self.replicate_token = os.environ.get("REPLICATE_API_TOKEN", "")


@dataclass
class TryOnResult:
    """Result of a single try-on operation."""
    # The generated image
    image: Image.Image
    
    # Metadata
    garment_id: str
    garment_type: GarmentType
    backend: TryOnBackend
    
    # Processing info
    processing_time_ms: float
    success: bool = True
    error_message: Optional[str] = None
    
    # Paths (if saved)
    result_path: Optional[str] = None
    mask_path: Optional[str] = None
    
    # Model info
    model_name: Optional[str] = None
    inference_steps: int = 30
    
    def save(self, path: str) -> str:
        """Save the result image."""
        self.image.save(path)
        self.result_path = path
        return path


@dataclass 
class OutfitTryOnResult:
    """Result of trying on a complete outfit."""
    # Outfit identification
    outfit_name: str
    outfit_score: float
    
    # Individual garment results
    garment_results: List[TryOnResult] = field(default_factory=list)
    
    # Composite result (all garments combined)
    composite_image: Optional[Image.Image] = None
    comparison_panel: Optional[Image.Image] = None
    
    # Processing info
    total_processing_time_ms: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)
    
    # Paths
    composite_path: Optional[str] = None
    panel_path: Optional[str] = None
    
    @property
    def num_garments_processed(self) -> int:
        """Number of garments successfully processed."""
        return sum(1 for r in self.garment_results if r.success)
    
    @property
    def num_garments_failed(self) -> int:
        """Number of garments that failed processing."""
        return sum(1 for r in self.garment_results if not r.success)


@dataclass
class TryOnError(Exception):
    """Exception for try-on errors."""
    message: str
    backend: Optional[TryOnBackend] = None
    garment_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.backend:
            parts.append(f"Backend: {self.backend.value}")
        if self.garment_id:
            parts.append(f"Garment: {self.garment_id}")
        return " | ".join(parts)


# =============================================================================
# Category Mapping Helpers
# =============================================================================

def category_to_garment_type(category: str) -> GarmentType:
    """
    Map garment category to try-on garment type.
    
    Args:
        category: Category string from GarmentAttributes
        
    Returns:
        GarmentType for try-on positioning
    """
    category_lower = category.lower()
    
    # Upper body items
    if category_lower in ["top", "tops", "shirt", "blouse", "t-shirt", "sweater", 
                          "jacket", "coat", "blazer", "outerwear", "cardigan",
                          "tank_top", "hoodie", "vest"]:
        return GarmentType.UPPER_BODY
    
    # Lower body items
    if category_lower in ["bottom", "bottoms", "pants", "trousers", "jeans",
                          "skirt", "shorts", "leggings"]:
        return GarmentType.LOWER_BODY
    
    # Full body items
    if category_lower in ["dress", "dresses", "full_body", "jumpsuit", "romper",
                          "gown", "maxi_dress"]:
        return GarmentType.FULL_BODY
    
    # Footwear
    if category_lower in ["shoes", "footwear", "sneakers", "boots", "heels",
                          "sandals", "flats", "loafers"]:
        return GarmentType.FOOTWEAR
    
    # Default to accessory
    return GarmentType.ACCESSORY


def is_tryonable(category: str) -> bool:
    """
    Check if a garment category can be tried on.
    
    Currently, try-on works best for upper_body, lower_body, and dresses.
    Shoes and accessories have limited support.
    """
    garment_type = category_to_garment_type(category)
    return garment_type in [
        GarmentType.UPPER_BODY,
        GarmentType.LOWER_BODY,
        GarmentType.FULL_BODY,
    ]
