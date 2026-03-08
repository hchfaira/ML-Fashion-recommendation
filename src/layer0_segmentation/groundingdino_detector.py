"""
GroundingDINO Detector
======================

Detects garments in images using GroundingDINO with text prompts
generated from GARMENT_TAXONOMY.

GroundingDINO is a zero-shot object detector that uses text descriptions
to detect objects without requiring class-specific training.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union, Tuple
from abc import ABC, abstractmethod

from src.core import get_logger
from .taxonomy import (
    generate_detection_prompt, 
    get_category_from_label,
    GarmentCategory
)
from .models import Detection, DetectionResult, BoundingBox

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Model paths (can be overridden)
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"

# Default thresholds
DEFAULT_BOX_THRESHOLD = 0.3
DEFAULT_TEXT_THRESHOLD = 0.25


# ============================================================================
# Abstract Base Class
# ============================================================================

class BaseDetector(ABC):
    """Abstract base class for garment detectors."""
    
    @abstractmethod
    def detect(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        prompt: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> DetectionResult:
        """
        Detect garments in an image.
        
        Args:
            image: Input image (numpy array, PIL Image, or path)
            prompt: Custom text prompt (optional)
            categories: List of taxonomy categories to detect (optional)
            
        Returns:
            DetectionResult containing all detections
        """
        pass


# ============================================================================
# GroundingDINO Detector
# ============================================================================

class GroundingDINODetector(BaseDetector):
    """
    Garment detector using GroundingDINO.
    
    Uses text prompts derived from GARMENT_TAXONOMY to detect garments
    in images with zero-shot capability.
    """
    
    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        checkpoint_path: str = GROUNDING_DINO_CHECKPOINT,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        device: Optional[str] = None
    ):
        """
        Initialize the GroundingDINO detector.
        
        Args:
            config_path: Path to GroundingDINO config file
            checkpoint_path: Path to model checkpoint
            box_threshold: Confidence threshold for box detection
            text_threshold: Confidence threshold for text matching
            device: Device to run on (cuda/cpu/mps, auto-detected if None)
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._device = device
        
        # Lazy-loaded model
        self._model = None
        
        logger.info(
            f"GroundingDINODetector initialized "
            f"(box_threshold={box_threshold}, text_threshold={text_threshold})"
        )
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self._device:
            return self._device
        
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"
        
        return self._device
    
    def _load_model(self) -> None:
        """Load the GroundingDINO model."""
        if self._model is not None:
            return
        
        try:
            from groundingdino.util.inference import load_model
        except ImportError:
            raise ImportError(
                "GroundingDINO is required for detection.\n"
                "Install with: pip install groundingdino-py\n"
                "Or clone: https://github.com/IDEA-Research/GroundingDINO"
            )
        
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"GroundingDINO checkpoint not found: {self.checkpoint_path}\n"
                "Download from: https://github.com/IDEA-Research/GroundingDINO#model-checkpoints"
            )
        
        logger.info("Loading GroundingDINO model...")
        self._model = load_model(
            self.config_path,
            self.checkpoint_path,
            device=self._get_device()
        )
        logger.info(f"GroundingDINO loaded on {self._get_device()}")
    
    def _load_image(
        self, 
        image: Union[np.ndarray, Image.Image, Path, str]
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Load and prepare image for detection.
        
        Returns:
            Tuple of (numpy RGB array, PIL Image)
        """
        if isinstance(image, (str, Path)):
            image_pil = Image.open(image).convert("RGB")
            image_np = np.array(image_pil)
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
            image_np = np.array(image_pil)
        else:
            # Assume numpy array
            if image.ndim == 3 and image.shape[2] == 4:
                # RGBA to RGB
                image_np = image[:, :, :3]
            elif image.ndim == 3 and image.shape[2] == 3:
                image_np = image
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            image_pil = Image.fromarray(image_np)
        
        return image_np, image_pil
    
    def detect(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        prompt: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> DetectionResult:
        """
        Detect garments in an image using GroundingDINO.
        
        Args:
            image: Input image
            prompt: Custom text prompt (overrides categories)
            categories: List of taxonomy categories to detect
            
        Returns:
            DetectionResult with all detected garments
        """
        self._load_model()
        
        from groundingdino.util.inference import predict
        
        # Load image
        image_np, image_pil = self._load_image(image)
        h, w = image_np.shape[:2]
        
        # Generate prompt
        if prompt is None:
            prompt = generate_detection_prompt(categories)
        
        logger.debug(f"Detecting with prompt: {prompt[:100]}...")
        
        # Run detection
        boxes, logits, phrases = predict(
            model=self._model,
            image=image_pil,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self._get_device()
        )
        
        # Convert to Detection objects
        detections = []
        for box, score, phrase in zip(boxes, logits.tolist(), phrases):
            # Convert normalized coordinates to pixel coordinates
            x1, y1, x2, y2 = box
            bbox = BoundingBox(
                x1=int(x1 * w),
                y1=int(y1 * h),
                x2=int(x2 * w),
                y2=int(y2 * h)
            )
            
            # Determine category from label
            category = get_category_from_label(phrase)
            
            detection = Detection(
                label=phrase,
                box=bbox,
                score=float(score),
                category=category
            )
            detections.append(detection)
        
        result = DetectionResult(
            detections=detections,
            image_size=(h, w)
        )
        
        logger.info(f"Detected {len(detections)} garments")
        return result
    
    def detect_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, Path, str]],
        prompt: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> List[DetectionResult]:
        """
        Detect garments in multiple images.
        
        Args:
            images: List of input images
            prompt: Custom text prompt
            categories: List of taxonomy categories
            
        Returns:
            List of DetectionResult, one per image
        """
        results = []
        for i, image in enumerate(images):
            try:
                result = self.detect(image, prompt, categories)
                results.append(result)
            except Exception as e:
                logger.error(f"Detection failed for image {i}: {e}")
                results.append(DetectionResult())
        
        return results


# ============================================================================
# Fallback Detector (no model required)
# ============================================================================

class FallbackDetector(BaseDetector):
    """
    Simple fallback detector that returns the full image as a single detection.
    Use when GroundingDINO is not available.
    """
    
    def __init__(self, default_label: str = "garment"):
        self.default_label = default_label
        logger.info("FallbackDetector initialized (no model)")
    
    def detect(
        self,
        image: Union[np.ndarray, Image.Image, Path, str],
        prompt: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> DetectionResult:
        """Return full image as single detection."""
        # Get image dimensions
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                w, h = img.size
        elif isinstance(image, Image.Image):
            w, h = image.size
        else:
            h, w = image.shape[:2]
        
        detection = Detection(
            label=self.default_label,
            box=BoundingBox(x1=0, y1=0, x2=w, y2=h),
            score=1.0,
            category=GarmentCategory.UNKNOWN
        )
        
        return DetectionResult(
            detections=[detection],
            image_size=(h, w)
        )


# ============================================================================
# Factory Function
# ============================================================================

def get_detector(
    use_grounding_dino: bool = True,
    **kwargs
) -> BaseDetector:
    """
    Factory function to get the appropriate detector.
    
    Args:
        use_grounding_dino: Whether to try loading GroundingDINO
        **kwargs: Additional arguments for the detector
        
    Returns:
        Detector instance
    """
    if use_grounding_dino:
        try:
            from groundingdino.util.inference import load_model
            return GroundingDINODetector(**kwargs)
        except ImportError:
            logger.warning("GroundingDINO not available, using fallback detector")
    
    return FallbackDetector()
