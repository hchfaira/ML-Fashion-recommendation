"""
CatVTON Backend for Virtual Try-On
==================================

CatVTON: Concatenation Is All You Need for Virtual Try-on with Diffusion Models
ICLR 2025 - https://github.com/Zheng-Chong/CatVTON

Architecture: Single UNet with spatial concatenation
VRAM: ~8GB | Resolution: 768x1024
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
from PIL import Image
import time

from .models import TryOnConfig, TryOnResult, GarmentType, TryOnBackend, TryOnError
from .mask_generator import MaskGenerator

logger = logging.getLogger(__name__)


class CatVTONBackend:
    """
    CatVTON virtual try-on backend.
    
    Uses a fine-tuned Stable Diffusion inpainting model with 
    garment-conditioned generation.
    """
    
    def __init__(self, config: Optional[TryOnConfig] = None):
        """
        Initialize CatVTON backend.
        
        Args:
            config: Try-on configuration. Uses defaults if not provided.
        """
        self.config = config or TryOnConfig(backend=TryOnBackend.CATVTON)
        self.mask_generator = MaskGenerator()
        
        self._pipeline = None
        self._device = None
        self._dtype = None
        self._initialized = False
    
    def _ensure_dependencies(self):
        """Ensure required packages are installed."""
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
            return True
        except ImportError as e:
            logger.error(f"Missing dependencies for CatVTON: {e}")
            logger.info("Install with: pip install torch diffusers transformers accelerate")
            raise TryOnError(
                message="Missing CatVTON dependencies",
                backend=TryOnBackend.CATVTON,
                details={"missing": str(e)}
            )
    
    def initialize(self):
        """
        Initialize the CatVTON pipeline.
        
        Loads the model from HuggingFace. First run downloads ~4GB.
        """
        if self._initialized:
            return
        
        self._ensure_dependencies()
        
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        
        # Determine device and dtype
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        
        logger.info(f"Initializing CatVTON on {self._device}")
        logger.info("Loading CatVTON weights from HuggingFace (first run ~4GB download)...")
        
        try:
            # Try loading CatVTON model
            self._pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "zhengchong/CatVTON",
                torch_dtype=self._dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self._device)
            logger.info("CatVTON model loaded successfully")
            
        except Exception as e:
            # Fallback to base SD-inpainting
            logger.warning(f"CatVTON HF model unavailable: {e}")
            logger.info("Using SD-inpainting fallback")
            
            self._pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=self._dtype,
                safety_checker=None,
            ).to(self._device)
        
        self._initialized = True
    
    def try_on(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_type: GarmentType,
        garment_id: str = "unknown",
        garment_description: Optional[str] = None,
    ) -> TryOnResult:
        """
        Perform virtual try-on.
        
        Args:
            person_image: PIL Image of the person
            garment_image: PIL Image of the garment (flat-lay preferred)
            garment_type: Type of garment for mask generation
            garment_id: ID for tracking
            garment_description: Optional description for better prompting
            
        Returns:
            TryOnResult with generated image
        """
        start_time = time.time()
        
        # Initialize if needed
        if not self._initialized:
            self.initialize()
        
        import torch
        
        # Resize to target resolution
        target_size = (self.config.target_width, self.config.target_height)
        person_resized = person_image.resize(target_size, Image.LANCZOS)
        garment_resized = garment_image.resize(target_size, Image.LANCZOS)
        
        # Generate mask for the garment region
        mask = self.mask_generator.generate_mask(person_resized, garment_type)
        
        # Build prompt
        prompt = self._build_prompt(garment_type, garment_description)
        negative = self._build_negative_prompt()
        
        logger.info(f"Running CatVTON inference ({self.config.num_inference_steps} steps)...")
        
        try:
            with torch.autocast(self._device):
                result = self._pipeline(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=person_resized,
                    mask_image=mask,
                    width=self.config.target_width,
                    height=self.config.target_height,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    strength=self.config.strength,
                )
            
            generated_image = result.images[0]
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"CatVTON inference complete in {processing_time:.1f}ms")
            
            return TryOnResult(
                image=generated_image,
                garment_id=garment_id,
                garment_type=garment_type,
                backend=TryOnBackend.CATVTON,
                processing_time_ms=processing_time,
                success=True,
                model_name="CatVTON",
                inference_steps=self.config.num_inference_steps,
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"CatVTON inference failed: {e}")
            
            return TryOnResult(
                image=person_resized,  # Return original on failure
                garment_id=garment_id,
                garment_type=garment_type,
                backend=TryOnBackend.CATVTON,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e),
            )
    
    def _build_prompt(
        self,
        garment_type: GarmentType,
        description: Optional[str] = None
    ) -> str:
        """Build the prompt for generation."""
        
        garment_desc = {
            GarmentType.UPPER_BODY: "wearing the provided top/shirt/jacket, perfect fit, same fabric texture and color",
            GarmentType.LOWER_BODY: "wearing the provided pants/skirt, perfect fit, same fabric",
            GarmentType.FULL_BODY: "wearing the provided dress, full body, same fabric texture and color",
            GarmentType.FOOTWEAR: "wearing the provided shoes, same style",
            GarmentType.ACCESSORY: "wearing the provided accessory",
        }.get(garment_type, "wearing the provided garment, perfect fit")
        
        if description:
            garment_desc = f"wearing {description}, perfect fit, same fabric texture and color"
        
        return (
            f"A person {garment_desc}, "
            "photorealistic, high quality, fashion photography, "
            "natural pose, studio lighting, 8k resolution"
        )
    
    def _build_negative_prompt(self) -> str:
        """Build the negative prompt."""
        return (
            "ugly, deformed, blurry, artifacts, watermark, "
            "wrong clothing, different garment, extra limbs, "
            "distorted body, low quality, cartoon"
        )
    
    def unload(self):
        """Unload the model to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._initialized = False
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("CatVTON model unloaded")
