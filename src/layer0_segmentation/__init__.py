"""
Layer 0: Garment Extraction Pipeline
=====================================

Advanced garment extraction using a multi-stage pipeline:

Pipeline Architecture:
    Input Image
        ↓
    GroundingDINO (text-prompt detection using GARMENT_TAXONOMY)
        ↓
    SCHP (Self-Correction Human Parsing for clothing classification)
        ↓
    SAM (Segment Anything Model for precise mask refinement)
        ↓
    Mask Fusion (merge detection + parsing + segmentation)
        ↓
    OpenCV Post-processing (mask cleaning and smoothing)
        ↓
    Garment Extraction (RGBA transparent PNG per garment)

Supported Garment Categories (GARMENT_TAXONOMY):
    - tops: t-shirt, shirt, blouse, sweater, hoodie, etc.
    - bottoms: pants, jeans, shorts, skirt, leggings
    - full_body: dress, jumpsuit, romper
    - outerwear: jacket, coat, parka, trench coat
    - footwear: sneakers, boots, sandals, heels, loafers
    - accessories: bag, handbag, backpack, belt, scarf, hat, etc.

Usage:
    # Simple extraction
    from src.layer0_segmentation import extract_garments
    result = extract_garments("image.jpg", output_dir="output/garments")
    
    # Pipeline with configuration
    from src.layer0_segmentation import GarmentExtractionPipeline, PipelineConfig
    config = PipelineConfig(canvas_size=512, use_schp=True)
    pipeline = GarmentExtractionPipeline(config)
    result = pipeline.process("image.jpg")
    
    # Access individual components
    from src.layer0_segmentation import GroundingDINODetector, SAMRefiner
    detector = GroundingDINODetector()
    detections = detector.detect(image)

Modules:
    - taxonomy: GARMENT_TAXONOMY and classification utilities
    - models: Data classes (Detection, FusedMask, ExtractedGarment, etc.)
    - groundingdino_detector: GroundingDINO-based garment detection
    - schp_parser: SCHP human parsing
    - sam_refiner: SAM mask refinement
    - mask_fusion: Multi-source mask fusion
    - mask_postprocess: OpenCV mask cleaning
    - garment_extractor: RGBA garment extraction
    - pipeline: Main orchestrator
"""

# =============================================================================
# Taxonomy
# =============================================================================
from .taxonomy import (
    GARMENT_TAXONOMY,
    GarmentCategory,
    SCHP_LABELS,
    SCHP_TO_TAXONOMY,
    generate_detection_prompt,
    generate_category_prompt,
    get_category_from_label,
    get_all_garment_labels,
    get_schp_category,
    get_schp_label_name,
    is_garment_schp_label
)

# =============================================================================
# Data Models
# =============================================================================
from .models import (
    # Bounding box
    BoundingBox,
    # Detection
    Detection,
    DetectionResult,
    # Parsing
    ParsingResult,
    # Masks
    RefinedMask,
    FusedMask,
    # Garment outputs
    ExtractedGarment,
    ExtractionResult
)

# =============================================================================
# Components
# =============================================================================

# Detector
from .groundingdino_detector import (
    BaseDetector,
    GroundingDINODetector,
    FallbackDetector,
    get_detector
)

# Parser
from .schp_parser import (
    BaseHumanParser,
    SCHPParser,
    FallbackParser,
    get_parser
)

# Refiner
from .sam_refiner import (
    BaseMaskRefiner,
    SAMRefiner,
    FallbackRefiner,
    get_refiner
)

# Mask Fusion
from .mask_fusion import (
    MaskFusion,
    fuse_masks
)

# Post-processing
from .mask_postprocess import (
    PostprocessConfig,
    MaskPostprocessor,
    clean_mask,
    clean_masks
)

# Extraction
from .garment_extractor import (
    ExtractorConfig,
    GarmentExtractor,
    extract_garment
)

# =============================================================================
# Pipeline
# =============================================================================
from .pipeline import (
    PipelineConfig,
    GarmentExtractionPipeline,
    create_pipeline,
    extract_garments
)

# =============================================================================
# Quality Checking & Import Session
# =============================================================================
from .quality_checker import (
    ProblemCode,
    Severity,
    GarmentStatus,
    QualityWarning,
    GarmentReport,
    ImportSessionReport,
    QualityThresholds,
    QualityChecker,
)
from .session_manager import (
    SessionState,
    GarmentDecision,
    GarmentDecisionRecord,
    ImportSession,
    ImportSessionManager,
)


# =============================================================================
# Legacy Compatibility Layer
# =============================================================================

# Keep old names for backward compatibility
DEFAULT_CANVAS_SIZE = 512

# Legacy GarmentType enum (maps to new GarmentCategory)
from enum import Enum

class GarmentType(str, Enum):
    """Legacy enum - use GarmentCategory instead."""
    TOP = "tops"
    BOTTOM = "bottoms"
    DRESS = "full_body"
    OUTERWEAR = "outerwear"
    SHOES = "footwear"
    ACCESSORIES = "accessories"
    UNKNOWN = "unknown"


from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
from PIL import Image
import numpy as np

@dataclass
class SegmentedGarment:
    """
    Legacy dataclass - use ExtractedGarment instead.
    
    Kept for backward compatibility with existing code.
    """
    image: Image.Image
    original_path: Optional[Path] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    area: int = 0
    confidence: float = 1.0
    garment_type: GarmentType = GarmentType.UNKNOWN
    label: str = ""
    mask: Optional[np.ndarray] = None
    
    @classmethod
    def from_extracted(cls, extracted: ExtractedGarment) -> "SegmentedGarment":
        """Convert from new ExtractedGarment."""
        # Map category to legacy type
        category_to_type = {
            GarmentCategory.TOPS: GarmentType.TOP,
            GarmentCategory.BOTTOMS: GarmentType.BOTTOM,
            GarmentCategory.FULL_BODY: GarmentType.DRESS,
            GarmentCategory.OUTERWEAR: GarmentType.OUTERWEAR,
            GarmentCategory.FOOTWEAR: GarmentType.SHOES,
            GarmentCategory.ACCESSORIES: GarmentType.ACCESSORIES,
            GarmentCategory.UNKNOWN: GarmentType.UNKNOWN
        }
        
        bbox = None
        if extracted.bounding_box:
            bbox = (
                extracted.bounding_box.x1,
                extracted.bounding_box.y1,
                extracted.bounding_box.width,
                extracted.bounding_box.height
            )
        
        return cls(
            image=extracted.image,
            original_path=extracted.source_path,
            bounding_box=bbox,
            area=extracted.area,
            confidence=extracted.confidence,
            garment_type=category_to_type.get(
                extracted.category, GarmentType.UNKNOWN
            ),
            label=extracted.label,
            mask=extracted.mask
        )


class SimpleGarmentSegmenter:
    """
    Legacy simple segmenter - use GarmentExtractionPipeline instead.
    
    Uses fallback detection without ML models.
    """
    
    def __init__(self, canvas_size: int = DEFAULT_CANVAS_SIZE):
        self.canvas_size = canvas_size
        self._pipeline = create_pipeline(mode="simple", canvas_size=canvas_size)
    
    def segment_garment(self, image_path) -> SegmentedGarment:
        """Segment a single garment from image."""
        result = self._pipeline.process(image_path)
        
        if len(result) > 0:
            return SegmentedGarment.from_extracted(result[0])
        
        # Return empty result
        from PIL import Image
        empty_img = Image.new(
            "RGBA", 
            (self.canvas_size, self.canvas_size), 
            (255, 255, 255, 0)
        )
        return SegmentedGarment(image=empty_img)
    
    def segment_folder(self, folder_path, extensions=(".jpg", ".jpeg", ".png", ".webp")):
        """Segment all images in folder."""
        results_dict = self._pipeline.process_folder(
            folder_path, extensions=extensions
        )
        
        segmented = []
        for result in results_dict.values():
            for extracted in result:
                segmented.append(SegmentedGarment.from_extracted(extracted))
        
        return segmented


class AdvancedGarmentSegmenter:
    """
    Legacy advanced segmenter - use GarmentExtractionPipeline instead.
    
    Uses full GroundingDINO + SCHP + SAM pipeline.
    """
    
    def __init__(
        self,
        canvas_size: int = DEFAULT_CANVAS_SIZE,
        device: Optional[str] = None,
        use_schp: bool = True,
        use_grounding_dino: bool = True,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25
    ):
        self.canvas_size = canvas_size
        config = PipelineConfig(
            use_grounding_dino=use_grounding_dino,
            use_schp=use_schp,
            use_sam=True,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            canvas_size=canvas_size,
            device=device
        )
        self._pipeline = GarmentExtractionPipeline(config)
    
    def segment_garment(
        self,
        image_path,
        garment_prompt: Optional[str] = None,
        refine_edges: bool = True
    ) -> SegmentedGarment:
        """Segment a garment from image."""
        result = self._pipeline.process(image_path)
        
        if len(result) > 0:
            return SegmentedGarment.from_extracted(result[0])
        
        from PIL import Image
        empty_img = Image.new(
            "RGBA",
            (self.canvas_size, self.canvas_size),
            (255, 255, 255, 0)
        )
        return SegmentedGarment(image=empty_img)
    
    def segment_folder(self, folder_path, extensions=(".jpg", ".jpeg", ".png", ".webp")):
        """Segment all images in folder."""
        results_dict = self._pipeline.process_folder(
            folder_path, extensions=extensions
        )
        
        segmented = []
        for result in results_dict.values():
            for extracted in result:
                segmented.append(SegmentedGarment.from_extracted(extracted))
        
        return segmented
    
    def save_segmented(self, garment: SegmentedGarment, output_path) -> Path:
        """Save segmented garment."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        garment.image.save(output_path, "PNG")
        return output_path


class GarmentSegmenter:
    """
    Legacy SAM-only segmenter - use GarmentExtractionPipeline instead.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_h",
        device: Optional[str] = None,
        canvas_size: int = DEFAULT_CANVAS_SIZE
    ):
        self.canvas_size = canvas_size
        config = PipelineConfig(
            use_grounding_dino=False,
            use_schp=False,
            use_sam=True,
            canvas_size=canvas_size,
            device=device
        )
        self._pipeline = GarmentExtractionPipeline(config)
    
    def segment_garment(self, image_path) -> SegmentedGarment:
        """Segment garment from image."""
        result = self._pipeline.process(image_path)
        
        if len(result) > 0:
            return SegmentedGarment.from_extracted(result[0])
        
        from PIL import Image
        empty_img = Image.new(
            "RGBA",
            (self.canvas_size, self.canvas_size),
            (255, 255, 255, 0)
        )
        return SegmentedGarment(image=empty_img)
    
    def segment_folder(self, folder_path, extensions=(".jpg", ".jpeg", ".png", ".webp")):
        """Segment all images in folder."""
        results_dict = self._pipeline.process_folder(
            folder_path, extensions=extensions
        )
        
        segmented = []
        for result in results_dict.values():
            for extracted in result:
                segmented.append(SegmentedGarment.from_extracted(extracted))
        
        return segmented
    
    def save_segmented(self, garment: SegmentedGarment, output_path) -> Path:
        """Save segmented garment."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        garment.image.save(output_path, "PNG")
        return output_path


def get_segmenter(
    mode: str = "auto",
    use_sam: bool = True,
    **kwargs
):
    """
    Legacy factory function - use create_pipeline instead.
    
    Args:
        mode: "advanced", "sam", "simple", or "auto"
        use_sam: Legacy parameter
        **kwargs: Additional arguments
    """
    canvas_size = kwargs.pop("canvas_size", DEFAULT_CANVAS_SIZE)
    
    if mode == "auto" and not use_sam:
        mode = "simple"
    
    if mode == "advanced":
        return AdvancedGarmentSegmenter(canvas_size=canvas_size, **kwargs)
    elif mode == "sam":
        return GarmentSegmenter(canvas_size=canvas_size, **kwargs)
    elif mode == "simple":
        return SimpleGarmentSegmenter(canvas_size=canvas_size)
    else:  # auto
        try:
            import torch
            from groundingdino.util.inference import load_model
            return AdvancedGarmentSegmenter(canvas_size=canvas_size, **kwargs)
        except ImportError:
            pass
        
        try:
            import torch
            from segment_anything import sam_model_registry
            return GarmentSegmenter(canvas_size=canvas_size, **kwargs)
        except ImportError:
            pass
        
        return SimpleGarmentSegmenter(canvas_size=canvas_size)


def segment_garment(
    image_path,
    output_path=None,
    mode: str = "auto",
    **kwargs
) -> SegmentedGarment:
    """
    Legacy convenience function - use extract_garments instead.
    """
    segmenter = get_segmenter(mode=mode, **kwargs)
    result = segmenter.segment_garment(image_path)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(output_path, "PNG")
    
    return result


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Taxonomy
    "GARMENT_TAXONOMY",
    "GarmentCategory",
    "SCHP_LABELS",
    "SCHP_TO_TAXONOMY",
    "generate_detection_prompt",
    "generate_category_prompt",
    "get_category_from_label",
    "get_all_garment_labels",
    "get_schp_category",
    "get_schp_label_name",
    "is_garment_schp_label",
    
    # Data Models
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "ParsingResult",
    "RefinedMask",
    "FusedMask",
    "ExtractedGarment",
    "ExtractionResult",
    
    # Components
    "BaseDetector",
    "GroundingDINODetector",
    "FallbackDetector",
    "get_detector",
    "BaseHumanParser",
    "SCHPParser",
    "FallbackParser",
    "get_parser",
    "BaseMaskRefiner",
    "SAMRefiner",
    "FallbackRefiner",
    "get_refiner",
    "MaskFusion",
    "fuse_masks",
    "PostprocessConfig",
    "MaskPostprocessor",
    "clean_mask",
    "clean_masks",
    "ExtractorConfig",
    "GarmentExtractor",
    "extract_garment",
    
    # Pipeline
    "PipelineConfig",
    "GarmentExtractionPipeline",
    "create_pipeline",
    "extract_garments",
    
    # Legacy (backward compatibility)
    "DEFAULT_CANVAS_SIZE",
    "GarmentType",
    "SegmentedGarment",
    "SimpleGarmentSegmenter",
    "AdvancedGarmentSegmenter",
    "GarmentSegmenter",
    "get_segmenter",
    "segment_garment",
]
