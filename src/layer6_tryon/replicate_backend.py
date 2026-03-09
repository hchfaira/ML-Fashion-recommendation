"""
Replicate Backend for Virtual Try-On
=====================================

Cloud-based virtual try-on using Replicate API.
No GPU required. ~$0.005 per image, ~30s latency.

Uses hosted IDM-VTON model.
"""

import logging
import base64
import time
from typing import Optional
from pathlib import Path
from io import BytesIO
from PIL import Image

from .models import TryOnConfig, TryOnResult, GarmentType, TryOnBackend, TryOnError

logger = logging.getLogger(__name__)


class ReplicateBackend:
    """
    Replicate API backend for virtual try-on.
    
    Uses hosted IDM-VTON model. No local GPU required.
    Cost: ~$0.005 per image
    """
    
    def __init__(self, config: Optional[TryOnConfig] = None):
        """
        Initialize Replicate backend.
        
        Args:
            config: Try-on configuration with replicate_token
        """
        self.config = config or TryOnConfig(backend=TryOnBackend.REPLICATE)
        self._replicate = None
    
    def _ensure_token(self):
        """Ensure Replicate API token is available."""
        if not self.config.replicate_token:
            raise TryOnError(
                message="REPLICATE_API_TOKEN not set. Get your token at https://replicate.com/account/api-tokens",
                backend=TryOnBackend.REPLICATE,
            )
    
    def _ensure_dependencies(self):
        """Ensure required packages are installed."""
        try:
            import replicate
            import requests
            return True
        except ImportError as e:
            logger.error(f"Missing dependencies for Replicate: {e}")
            logger.info("Install with: pip install replicate requests")
            raise TryOnError(
                message="Missing Replicate dependencies",
                backend=TryOnBackend.REPLICATE,
                details={"missing": str(e)}
            )
    
    def initialize(self):
        """Initialize the Replicate client."""
        self._ensure_token()
        self._ensure_dependencies()
        
        import replicate
        import os
        
        # Set token in environment for replicate library
        os.environ["REPLICATE_API_TOKEN"] = self.config.replicate_token
        self._replicate = replicate
        
        logger.info("Replicate backend initialized")
    
    def _image_to_data_uri(self, image: Image.Image) -> str:
        """Convert PIL Image to data URI."""
        buf = BytesIO()
        image.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    
    def try_on(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_type: GarmentType,
        garment_id: str = "unknown",
        garment_description: Optional[str] = None,
    ) -> TryOnResult:
        """
        Perform virtual try-on via Replicate API.
        
        Args:
            person_image: PIL Image of the person
            garment_image: PIL Image of the garment
            garment_type: Type of garment
            garment_id: ID for tracking
            garment_description: Optional description
            
        Returns:
            TryOnResult with generated image
        """
        start_time = time.time()
        
        # Initialize if needed
        if self._replicate is None:
            self.initialize()
        
        import requests
        
        # Convert images to data URIs
        person_uri = self._image_to_data_uri(person_image)
        garment_uri = self._image_to_data_uri(garment_image)
        
        # Map garment type to Replicate category
        category = {
            GarmentType.UPPER_BODY: "upper_body",
            GarmentType.LOWER_BODY: "lower_body",
            GarmentType.FULL_BODY: "dresses",
            GarmentType.FOOTWEAR: "upper_body",  # Fallback
            GarmentType.ACCESSORY: "upper_body",  # Fallback
        }.get(garment_type, "upper_body")
        
        # Build garment description
        desc = garment_description or f"A {garment_type.value.replace('_', ' ')} garment"
        desc += ", fabric texture, high quality"
        
        logger.info(f"Running IDM-VTON on Replicate ({self.config.num_inference_steps} steps)...")
        
        try:
            output = self._replicate.run(
                "cuuupid/idm-vton:906425dbca90663ff5427624839572cc56ea7d380343d13e2a4c4b09d3f0c30f",
                input={
                    "human_img": person_uri,
                    "garm_img": garment_uri,
                    "garment_des": desc,
                    "is_checked": True,
                    "is_checked_crop": False,
                    "denoise_steps": self.config.num_inference_steps,
                    "seed": 42,
                    "category": category,
                }
            )
            
            # Download result
            img_url = output[0] if isinstance(output, list) else str(output)
            logger.debug(f"Result URL: {img_url}")
            
            response = requests.get(img_url, timeout=60)
            response.raise_for_status()
            
            generated_image = Image.open(BytesIO(response.content)).convert("RGB")
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Replicate inference complete in {processing_time:.1f}ms")
            
            return TryOnResult(
                image=generated_image,
                garment_id=garment_id,
                garment_type=garment_type,
                backend=TryOnBackend.REPLICATE,
                processing_time_ms=processing_time,
                success=True,
                model_name="IDM-VTON (Replicate)",
                inference_steps=self.config.num_inference_steps,
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Replicate inference failed: {e}")
            
            return TryOnResult(
                image=person_image,  # Return original on failure
                garment_id=garment_id,
                garment_type=garment_type,
                backend=TryOnBackend.REPLICATE,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e),
            )
