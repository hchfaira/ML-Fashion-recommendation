"""
Garment Extraction Pipeline
============================

Main orchestrator for the GroundingDINO + SCHP + SAM garment extraction pipeline.

Pipeline Stages:
1. GroundingDINO → Detect garments with text prompts from GARMENT_TAXONOMY
2. SCHP → Parse human figure for clothing region classification
3. SAM → Refine mask edges with precise segmentation
4. Mask Fusion → Combine detection + parsing + segmentation
5. OpenCV → Clean and post-process masks
6. Extraction → Generate RGBA transparent PNGs per garment
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass, field
import time

from src.core import get_logger
from .taxonomy import (
    GARMENT_TAXONOMY,
    GarmentCategory,
    generate_detection_prompt
)
from .models import (
    DetectionResult, ParsingResult, RefinedMask, FusedMask,
    ExtractedGarment, ExtractionResult, BoundingBox
)
from .groundingdino_detector import (
    GroundingDINODetector, FallbackDetector, get_detector, BaseDetector
)
from .schp_parser import SCHPParser, FallbackParser, get_parser, BaseHumanParser
from .sam_refiner import SAMRefiner, FallbackRefiner, get_refiner, BaseMaskRefiner
from .mask_fusion import MaskFusion, fuse_masks
from .mask_postprocess import MaskPostprocessor, PostprocessConfig, clean_mask
from .garment_extractor import GarmentExtractor, ExtractorConfig

logger = get_logger(__name__)


# ============================================================================
# Pipeline Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""
    
    # Model enablement
    use_grounding_dino: bool = True
    use_schp: bool = True
    use_sam: bool = True
    
    # Detection settings
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    
    # Extraction settings
    canvas_size: int = 512
    
    # Post-processing settings
    postprocess_config: Optional[PostprocessConfig] = None
    
    # Output settings
    save_intermediate: bool = False
    output_dir: Optional[Path] = None
    
    # Device settings
    device: Optional[str] = None
    
    # Categories to detect (None = all)
    categories: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.postprocess_config is None:
            self.postprocess_config = PostprocessConfig()
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)


# ============================================================================
# Extraction Pipeline
# ============================================================================

class GarmentExtractionPipeline:
    """
    Complete garment extraction pipeline.
    
    Orchestrates all stages:
    1. GroundingDINO detection
    2. SCHP human parsing
    3. SAM mask refinement
    4. Mask fusion
    5. OpenCV post-processing
    6. Garment extraction
    
    Designed to be modular and testable - each component can be
    replaced or mocked for testing.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        detector: Optional[BaseDetector] = None,
        parser: Optional[BaseHumanParser] = None,
        refiner: Optional[BaseMaskRefiner] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            detector: Custom detector (optional)
            parser: Custom parser (optional)
            refiner: Custom mask refiner (optional)
        """
        self.config = config or PipelineConfig()
        
        # Initialize components (lazy or injected)
        self._detector = detector
        self._parser = parser
        self._refiner = refiner
        
        # Always initialize these
        self._fusion = MaskFusion(use_schp_validation=self.config.use_schp)
        self._postprocessor = MaskPostprocessor(self.config.postprocess_config)
        self._extractor = GarmentExtractor(
            ExtractorConfig(canvas_size=self.config.canvas_size)
        )
        
        logger.info("GarmentExtractionPipeline initialized")
        logger.info(f"  - GroundingDINO: {self.config.use_grounding_dino}")
        logger.info(f"  - SCHP: {self.config.use_schp}")
        logger.info(f"  - SAM: {self.config.use_sam}")
    
    # -------------------------------------------------------------------------
    # Component Accessors (Lazy Loading)
    # -------------------------------------------------------------------------
    
    @property
    def detector(self) -> BaseDetector:
        """Get or create the detector."""
        if self._detector is None:
            self._detector = get_detector(
                use_grounding_dino=self.config.use_grounding_dino,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                device=self.config.device
            )
        return self._detector
    
    @property
    def parser(self) -> BaseHumanParser:
        """Get or create the parser."""
        if self._parser is None:
            self._parser = get_parser(
                use_schp=self.config.use_schp,
                device=self.config.device
            )
        return self._parser
    
    @property
    def refiner(self) -> BaseMaskRefiner:
        """Get or create the mask refiner."""
        if self._refiner is None:
            self._refiner = get_refiner(
                use_sam=self.config.use_sam,
                device=self.config.device
            )
        return self._refiner
    
    # -------------------------------------------------------------------------
    # Main Pipeline
    # -------------------------------------------------------------------------
    
    def process(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        categories: Optional[List[str]] = None,
        save_output: bool = False,
        output_dir: Optional[Path] = None
    ) -> ExtractionResult:
        """
        Run the complete extraction pipeline on an image.
        
        Args:
            image: Input image (array, PIL Image, or path)
            categories: Categories to detect (None = all from config or taxonomy)
            save_output: Whether to save extracted garments
            output_dir: Output directory (overrides config)
            
        Returns:
            ExtractionResult with all extracted garments
        """
        start_time = time.time()
        errors = []
        
        # Determine source path if applicable
        source_path = None
        if isinstance(image, (str, Path)):
            source_path = Path(image)
        
        # Load image
        image_np = self._load_image(image)
        image_shape = image_np.shape[:2]  # (H, W)
        
        # Categories to detect
        cats = categories or self.config.categories
        prompt = generate_detection_prompt(cats)
        
        logger.info(f"Processing image {source_path or '(array)'}")
        logger.debug(f"Image shape: {image_shape}")
        
        # Stage 1: Detection
        logger.debug("Stage 1: GroundingDINO detection")
        try:
            detections = self.detector.detect(image_np, prompt=prompt)
            logger.info(f"  Detected {len(detections)} garments")
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            errors.append(f"Detection: {e}")
            detections = DetectionResult()
        
        if len(detections) == 0:
            logger.warning("No garments detected")
            return ExtractionResult(
                garments=[],
                source_image_path=source_path,
                processing_time_ms=(time.time() - start_time) * 1000,
                errors=errors + ["No garments detected"]
            )
        
        # Stage 2: Human Parsing (optional)
        parsing = None
        if self.config.use_schp:
            logger.debug("Stage 2: SCHP human parsing")
            try:
                parsing = self.parser.parse(image_np)
                logger.debug(f"  Parsed {len(parsing.get_present_categories())} categories")
            except Exception as e:
                logger.warning(f"Parsing failed: {e}")
                errors.append(f"Parsing: {e}")
        
        # Stage 3: Mask Refinement
        logger.debug("Stage 3: SAM mask refinement")
        try:
            boxes = [d.box for d in detections]
            refined_masks = self.refiner.refine_batch(image_np, boxes)
            logger.debug(f"  Refined {len(refined_masks)} masks")
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            errors.append(f"Refinement: {e}")
            # Create fallback masks from boxes
            refined_masks = [
                RefinedMask(
                    mask=self._create_box_mask(image_shape, d.box),
                    score=0.5,
                    box=d.box
                )
                for d in detections
            ]
        
        # Stage 4: Mask Fusion
        logger.debug("Stage 4: Mask fusion")
        try:
            if parsing is not None:
                fused_masks = self._fusion.fuse(detections, parsing, refined_masks)
            else:
                fused_masks = self._fusion.fuse_simple(
                    detections, refined_masks, image_shape
                )
            logger.debug(f"  Fused {len(fused_masks)} masks")
        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            errors.append(f"Fusion: {e}")
            fused_masks = []
        
        # Stage 5: Post-processing
        logger.debug("Stage 5: OpenCV post-processing")
        for mask in fused_masks:
            try:
                mask.mask = self._postprocessor.process(mask.mask)
            except Exception as e:
                logger.warning(f"Post-processing failed for {mask.label}: {e}")
        
        # Stage 6: Garment Extraction
        logger.debug("Stage 6: Garment extraction")
        result = self._extractor.extract_all(image_np, fused_masks, source_path)
        result.errors.extend(errors)
        
        # Calculate total time
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Extracted {len(result)} garments in "
            f"{result.processing_time_ms:.1f}ms"
        )
        
        # Save if requested
        if save_output:
            out_dir = output_dir or self.config.output_dir
            if out_dir:
                saved = result.save_all(out_dir)
                logger.info(f"Saved {len(saved)} garments to {out_dir}")
        
        return result
    
    def process_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, Path, str]],
        categories: Optional[List[str]] = None
    ) -> List[ExtractionResult]:
        """
        Process multiple images.
        
        Args:
            images: List of input images
            categories: Categories to detect
            
        Returns:
            List of ExtractionResults
        """
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            try:
                result = self.process(image, categories)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                results.append(ExtractionResult(errors=[str(e)]))
        
        return results
    
    def process_folder(
        self,
        folder_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
        categories: Optional[List[str]] = None
    ) -> Dict[str, ExtractionResult]:
        """
        Process all images in a folder.
        
        Args:
            folder_path: Path to folder
            output_dir: Output directory
            extensions: Image file extensions to process
            categories: Categories to detect
            
        Returns:
            Dictionary mapping filename to ExtractionResult
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Find image files
        image_files = [
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        logger.info(f"Found {len(image_files)} images in {folder_path}")
        
        results = {}
        for img_path in sorted(image_files):
            try:
                # Create per-image output dir if specified
                img_out_dir = None
                if output_dir:
                    img_out_dir = Path(output_dir) / img_path.stem
                
                result = self.process(
                    img_path,
                    categories=categories,
                    save_output=bool(output_dir),
                    output_dir=img_out_dir
                )
                results[img_path.name] = result
                
            except Exception as e:
                logger.error(f"Failed: {img_path.name}: {e}")
                results[img_path.name] = ExtractionResult(errors=[str(e)])
        
        return results
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _load_image(
        self, 
        image: Union[np.ndarray, Image.Image, Path, str]
    ) -> np.ndarray:
        """Load image as numpy RGB array."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        # Ensure RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        return image
    
    def _create_box_mask(
        self, 
        shape: tuple, 
        box: BoundingBox
    ) -> np.ndarray:
        """Create a rectangular mask from bounding box."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1 = max(0, box.x1), max(0, box.y1)
        x2, y2 = min(w, box.x2), min(h, box.y2)
        mask[y1:y2, x1:x2] = 255
        return mask


# ============================================================================
# Factory Function
# ============================================================================

def create_pipeline(
    mode: str = "auto",
    canvas_size: int = 512,
    **kwargs
) -> GarmentExtractionPipeline:
    """
    Create a garment extraction pipeline.
    
    Args:
        mode: Pipeline mode
            - "full": Use all models (GroundingDINO + SCHP + SAM)
            - "fast": Use SAM only (faster, less accurate)
            - "simple": Use OpenCV only (no ML models)
            - "auto": Try full, fallback to simple
        canvas_size: Output canvas size
        **kwargs: Additional configuration
        
    Returns:
        Configured pipeline instance
    """
    if mode == "full":
        config = PipelineConfig(
            use_grounding_dino=True,
            use_schp=True,
            use_sam=True,
            canvas_size=canvas_size,
            **kwargs
        )
    elif mode == "fast":
        config = PipelineConfig(
            use_grounding_dino=False,
            use_schp=False,
            use_sam=True,
            canvas_size=canvas_size,
            **kwargs
        )
    elif mode == "simple":
        config = PipelineConfig(
            use_grounding_dino=False,
            use_schp=False,
            use_sam=False,
            canvas_size=canvas_size,
            **kwargs
        )
    else:  # auto
        # Try to detect available models
        try:
            import groundingdino
            from segment_anything import sam_model_registry
            config = PipelineConfig(
                use_grounding_dino=True,
                use_schp=True,
                use_sam=True,
                canvas_size=canvas_size,
                **kwargs
            )
        except ImportError:
            logger.warning("Full pipeline not available, using simple mode")
            config = PipelineConfig(
                use_grounding_dino=False,
                use_schp=False,
                use_sam=False,
                canvas_size=canvas_size,
                **kwargs
            )
    
    return GarmentExtractionPipeline(config)


# ============================================================================
# Convenience Function
# ============================================================================

def extract_garments(
    image: Union[np.ndarray, Image.Image, Path, str],
    output_dir: Optional[Path] = None,
    categories: Optional[List[str]] = None,
    canvas_size: int = 512,
    mode: str = "auto"
) -> ExtractionResult:
    """
    Convenience function to extract garments from an image.
    
    Args:
        image: Input image
        output_dir: Optional output directory
        categories: Categories to detect
        canvas_size: Output canvas size
        mode: Pipeline mode
        
    Returns:
        ExtractionResult with all extracted garments
    """
    pipeline = create_pipeline(mode=mode, canvas_size=canvas_size)
    return pipeline.process(
        image,
        categories=categories,
        save_output=bool(output_dir),
        output_dir=output_dir
    )
