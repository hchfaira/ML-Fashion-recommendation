"""
Virtual Try-On Service
======================

Main service for performing virtual try-on operations.
Takes outfits and user images, generates try-on visualizations.
"""

import logging
import time
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from src.core.models import Garment
from .models import (
    TryOnBackend,
    TryOnConfig,
    GarmentType,
    TryOnResult,
    OutfitTryOnResult,
    TryOnError,
    category_to_garment_type,
    is_tryonable,
)
from .catvton_backend import CatVTONBackend
from .replicate_backend import ReplicateBackend
from .mask_generator import MaskGenerator

logger = logging.getLogger(__name__)


# Type alias for outfit tuple: (name, garments, score)
OutfitTuple = Tuple[str, List[Garment], float]


class VirtualTryOnService:
    """
    Virtual Try-On Service.
    
    Generates virtual try-on images for outfits on user photos.
    
    Example:
        service = VirtualTryOnService(backend=TryOnBackend.CATVTON)
        
        # Single garment try-on
        result = service.try_on_garment(person_path, garment)
        
        # Full outfit try-on
        results = service.try_on_outfit(person_path, "Summer Look", garments, 0.85)
        
        # Multiple outfits
        all_results = service.try_on_outfits(person_path, [
            ("Casual", garments1, 0.9),
            ("Formal", garments2, 0.8),
        ])
    """
    
    def __init__(
        self,
        backend: TryOnBackend = TryOnBackend.CATVTON,
        config: Optional[TryOnConfig] = None,
    ):
        """
        Initialize the Virtual Try-On Service.
        
        Args:
            backend: Which backend to use (CATVTON, IDMVTON, REPLICATE)
            config: Optional configuration
        """
        self.config = config or TryOnConfig(backend=backend)
        self.config.backend = backend
        
        # Initialize backend
        self._backend = self._create_backend()
        self._initialized = False
    
    def _create_backend(self) -> Union[CatVTONBackend, ReplicateBackend]:
        """Create the appropriate backend instance."""
        if self.config.backend == TryOnBackend.CATVTON:
            return CatVTONBackend(self.config)
        elif self.config.backend == TryOnBackend.IDMVTON:
            # IDM-VTON uses similar pipeline to CatVTON locally
            # For full IDM-VTON, use Replicate
            logger.warning("IDM-VTON local not implemented, using CatVTON")
            return CatVTONBackend(self.config)
        elif self.config.backend == TryOnBackend.REPLICATE:
            return ReplicateBackend(self.config)
        else:
            raise TryOnError(
                message=f"Unknown backend: {self.config.backend}",
                backend=self.config.backend,
            )
    
    def initialize(self):
        """Initialize the backend (load models)."""
        if not self._initialized:
            self._backend.initialize()
            self._initialized = True
    
    def try_on_garment(
        self,
        person_image: Union[str, Path, Image.Image],
        garment: Garment,
        description: Optional[str] = None,
    ) -> TryOnResult:
        """
        Try on a single garment.
        
        Args:
            person_image: Path to person image or PIL Image
            garment: Garment to try on
            description: Optional garment description
            
        Returns:
            TryOnResult with generated image
        """
        # Ensure initialized
        self.initialize()
        
        # Load images
        person_img = self._load_image(person_image)
        garment_img = self._load_garment_image(garment)
        
        if garment_img is None:
            return TryOnResult(
                image=person_img,
                garment_id=garment.id,
                garment_type=GarmentType.UPPER_BODY,
                backend=self.config.backend,
                processing_time_ms=0,
                success=False,
                error_message="Could not load garment image",
            )
        
        # Determine garment type
        category = garment.attributes.category.value if hasattr(garment.attributes.category, 'value') else str(garment.attributes.category)
        garment_type = category_to_garment_type(category)
        
        # Build description
        if description is None:
            desc_parts = []
            if garment.attributes.color:
                color = garment.attributes.color.primary if hasattr(garment.attributes.color, 'primary') else str(garment.attributes.color)
                desc_parts.append(color)
            if garment.attributes.subcategory:
                desc_parts.append(garment.attributes.subcategory)
            description = " ".join(desc_parts) if desc_parts else None
        
        # Perform try-on
        return self._backend.try_on(
            person_image=person_img,
            garment_image=garment_img,
            garment_type=garment_type,
            garment_id=garment.id,
            garment_description=description,
        )
    
    def try_on_outfit(
        self,
        person_image: Union[str, Path, Image.Image],
        outfit_name: str,
        garments: List[Garment],
        outfit_score: float = 0.0,
        sequential: bool = True,
    ) -> OutfitTryOnResult:
        """
        Try on a complete outfit.
        
        Processes each tryonable garment in the outfit.
        For best results, garments are applied sequentially to build up the look.
        
        Args:
            person_image: Path to person image or PIL Image
            outfit_name: Name of the outfit
            garments: List of garments in the outfit
            outfit_score: Score of the outfit
            sequential: If True, apply garments sequentially (each builds on previous)
            
        Returns:
            OutfitTryOnResult with all garment results
        """
        start_time = time.time()
        
        # Ensure initialized
        self.initialize()
        
        # Load person image
        person_img = self._load_image(person_image)
        current_image = person_img.copy()
        
        garment_results: List[TryOnResult] = []
        errors: List[str] = []
        
        # Sort garments by try-on order: lower_body first, then upper_body, then full_body
        # This ensures proper layering
        sorted_garments = self._sort_garments_for_tryon(garments)
        
        logger.info(f"Trying on outfit '{outfit_name}' with {len(sorted_garments)} garments")
        
        for garment in sorted_garments:
            # Check if garment can be tried on
            category = garment.attributes.category.value if hasattr(garment.attributes.category, 'value') else str(garment.attributes.category)
            
            if not is_tryonable(category):
                logger.debug(f"Skipping non-tryonable garment: {category}")
                continue
            
            # Load garment image
            garment_img = self._load_garment_image(garment)
            if garment_img is None:
                errors.append(f"Could not load image for garment {garment.id}")
                continue
            
            garment_type = category_to_garment_type(category)
            
            # Perform try-on
            result = self._backend.try_on(
                person_image=current_image if sequential else person_img,
                garment_image=garment_img,
                garment_type=garment_type,
                garment_id=garment.id,
            )
            
            garment_results.append(result)
            
            if result.success and sequential:
                current_image = result.image
            
            if not result.success:
                errors.append(result.error_message or f"Failed to try on {garment.id}")
        
        total_time = (time.time() - start_time) * 1000
        
        # Create comparison panel
        comparison_panel = None
        if self.config.create_comparison_panel and garment_results:
            comparison_panel = self._create_comparison_panel(
                person_img,
                current_image if sequential else garment_results[-1].image,
                outfit_name,
                outfit_score,
            )
        
        return OutfitTryOnResult(
            outfit_name=outfit_name,
            outfit_score=outfit_score,
            garment_results=garment_results,
            composite_image=current_image if sequential else None,
            comparison_panel=comparison_panel,
            total_processing_time_ms=total_time,
            success=len(errors) == 0,
            errors=errors,
        )
    
    def try_on_outfits(
        self,
        person_image: Union[str, Path, Image.Image],
        outfits: List[OutfitTuple],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[OutfitTryOnResult]:
        """
        Try on multiple outfits.
        
        Args:
            person_image: Path to person image or PIL Image
            outfits: List of (name, garments, score) tuples
            output_dir: Optional directory to save results
            
        Returns:
            List of OutfitTryOnResult for each outfit
        """
        results: List[OutfitTryOnResult] = []
        
        logger.info(f"Processing {len(outfits)} outfits for virtual try-on")
        
        for i, (name, garments, score) in enumerate(outfits):
            logger.info(f"Processing outfit {i+1}/{len(outfits)}: {name} (score: {score:.2f})")
            
            result = self.try_on_outfit(
                person_image=person_image,
                outfit_name=name,
                garments=garments,
                outfit_score=score,
            )
            
            # Save if output_dir provided
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                safe_name = name.replace(" ", "_").replace("/", "-")[:50]
                
                if result.composite_image:
                    result.composite_path = str(output_path / f"{i+1:02d}_{safe_name}_tryon.png")
                    result.composite_image.save(result.composite_path)
                
                if result.comparison_panel:
                    result.panel_path = str(output_path / f"{i+1:02d}_{safe_name}_comparison.png")
                    result.comparison_panel.save(result.panel_path)
            
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Virtual try-on complete: {successful}/{len(results)} outfits successful")
        
        return results
    
    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load an image from path or return if already PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(str(image)).convert("RGB")
    
    def _load_garment_image(self, garment: Garment) -> Optional[Image.Image]:
        """Load garment image from its path or URL."""
        try:
            if garment.image_path:
                return Image.open(garment.image_path).convert("RGB")
            elif garment.image_url:
                import requests
                response = requests.get(garment.image_url, timeout=30)
                from io import BytesIO
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                logger.warning(f"No image path/URL for garment {garment.id}")
                return None
        except Exception as e:
            logger.error(f"Failed to load garment image: {e}")
            return None
    
    def _sort_garments_for_tryon(self, garments: List[Garment]) -> List[Garment]:
        """Sort garments in optimal order for sequential try-on."""
        # Priority: lower_body (1), upper_body (2), full_body (3)
        def priority(g: Garment) -> int:
            cat = g.attributes.category.value if hasattr(g.attributes.category, 'value') else str(g.attributes.category)
            gtype = category_to_garment_type(cat)
            if gtype == GarmentType.LOWER_BODY:
                return 1
            elif gtype == GarmentType.UPPER_BODY:
                return 2
            elif gtype == GarmentType.FULL_BODY:
                return 3
            else:
                return 4
        
        return sorted(garments, key=priority)
    
    def _create_comparison_panel(
        self,
        original: Image.Image,
        result: Image.Image,
        outfit_name: str,
        score: float,
    ) -> Image.Image:
        """Create a side-by-side comparison panel."""
        SIZE = (512, 640)
        
        orig_resized = original.resize(SIZE, Image.LANCZOS)
        result_resized = result.resize(SIZE, Image.LANCZOS)
        
        # Panel dimensions
        panel_w = SIZE[0] * 2 + 30
        panel_h = SIZE[1] + 100
        
        panel = Image.new("RGB", (panel_w, panel_h), (248, 248, 250))
        draw = ImageDraw.Draw(panel)
        
        # Paste images
        panel.paste(orig_resized, (10, 50))
        panel.paste(result_resized, (SIZE[0] + 20, 50))
        
        # Add labels
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Title
        title = f"{outfit_name} (Score: {score:.2f})"
        draw.text((panel_w // 2, 15), title, fill=(30, 30, 30), font=font, anchor="mm")
        
        # Labels
        draw.text((10 + SIZE[0] // 2, SIZE[1] + 65), "Original", fill=(100, 100, 100), font=small_font, anchor="mm")
        draw.text((SIZE[0] + 20 + SIZE[0] // 2, SIZE[1] + 65), "Virtual Try-On ✨", fill=(30, 140, 80), font=small_font, anchor="mm")
        
        # Separator line
        draw.line([(SIZE[0] + 15, 50), (SIZE[0] + 15, SIZE[1] + 50)], fill=(200, 200, 200), width=2)
        
        return panel
    
    def unload(self):
        """Unload backend models to free memory."""
        if hasattr(self._backend, 'unload'):
            self._backend.unload()
        self._initialized = False
