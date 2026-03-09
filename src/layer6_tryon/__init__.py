"""
Layer 6: Virtual Try-On
=======================

Virtual try-on using diffusion models to visualize outfits on user photos.

Supported backends:
- CatVTON (ICLR 2025): Best open-source, local GPU
- IDM-VTON (ECCV 2024): Highest fidelity, local GPU
- Replicate API: Cloud-based, no GPU required

Example usage:
    from src.layer6_tryon import VirtualTryOnService, TryOnBackend
    
    service = VirtualTryOnService(backend=TryOnBackend.CATVTON)
    
    # Try on a single garment
    result = service.try_on_garment(person_image_path, garment)
    
    # Try on a complete outfit
    results = service.try_on_outfit(person_image_path, outfit_garments)
    
    # Try on multiple ranked outfits
    all_results = service.try_on_outfits(
        person_image_path,
        outfits=[("Casual Look", garments1, 0.85), ("Formal Look", garments2, 0.78)]
    )
"""

from .models import (
    TryOnBackend,
    TryOnConfig,
    GarmentType,
    TryOnResult,
    OutfitTryOnResult,
    TryOnError,
)
from .tryon_service import VirtualTryOnService
from .catvton_backend import CatVTONBackend
from .replicate_backend import ReplicateBackend
from .mask_generator import MaskGenerator

__all__ = [
    # Enums and Config
    "TryOnBackend",
    "TryOnConfig",
    "GarmentType",
    # Results
    "TryOnResult",
    "OutfitTryOnResult",
    "TryOnError",
    # Services
    "VirtualTryOnService",
    "CatVTONBackend",
    "ReplicateBackend",
    "MaskGenerator",
]
