"""
SAM Mask Refiner
================

Segment Anything Model (SAM) for high-precision mask refinement.

Uses bounding boxes from GroundingDINO as prompts to generate
precise segmentation masks for individual garments.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, List, Tuple
from abc import ABC, abstractmethod

from src.core import get_logger
from .models import BoundingBox, RefinedMask

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# SAM model configuration
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

# Available model types
AVAILABLE_SAM_MODELS = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth"
}


# ============================================================================
# Abstract Base Class
# ============================================================================

class BaseMaskRefiner(ABC):
    """Abstract base class for mask refinement."""
    
    @abstractmethod
    def refine(
        self,
        image: Union[np.ndarray, Image.Image],
        box: BoundingBox,
        initial_mask: Optional[np.ndarray] = None
    ) -> RefinedMask:
        """
        Refine mask for a detected region.
        
        Args:
            image: Input image
            box: Bounding box for the region
            initial_mask: Optional initial mask to refine
            
        Returns:
            RefinedMask with precise segmentation
        """
        pass
    
    @abstractmethod
    def refine_batch(
        self,
        image: Union[np.ndarray, Image.Image],
        boxes: List[BoundingBox]
    ) -> List[RefinedMask]:
        """
        Refine masks for multiple boxes in the same image.
        
        Args:
            image: Input image
            boxes: List of bounding boxes
            
        Returns:
            List of RefinedMasks
        """
        pass


# ============================================================================
# SAM Mask Refiner
# ============================================================================

class SAMRefiner(BaseMaskRefiner):
    """
    Mask refiner using Segment Anything Model (SAM).
    
    Uses bounding boxes as prompts to generate precise masks.
    SAM excels at boundary accuracy and fine detail preservation.
    """
    
    def __init__(
        self,
        checkpoint_path: str = SAM_CHECKPOINT,
        model_type: str = SAM_MODEL_TYPE,
        device: Optional[str] = None,
        multimask_output: bool = True
    ):
        """
        Initialize the SAM refiner.
        
        Args:
            checkpoint_path: Path to SAM checkpoint
            model_type: SAM model variant (vit_h, vit_l, vit_b)
            device: Device to run on (cuda/cpu/mps)
            multimask_output: Whether to output multiple mask candidates
        """
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self._device = device
        self.multimask_output = multimask_output
        
        # Lazy-loaded model
        self._sam = None
        self._predictor = None
        self._current_image = None  # Cache for batch processing
        
        logger.info(f"SAMRefiner initialized (model_type={model_type})")
    
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
        """Load the SAM model."""
        if self._sam is not None:
            return
        
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment_anything is required for mask refinement.\n"
                "Install with: pip install segment-anything"
            )
        
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {self.checkpoint_path}\n"
                "Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )
        
        logger.info(f"Loading SAM model ({self.model_type})...")
        
        self._sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self._sam.to(self._get_device())
        self._predictor = SamPredictor(self._sam)
        
        logger.info(f"SAM model loaded on {self._get_device()}")
    
    def _prepare_image(
        self, 
        image: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Prepare image for SAM."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        # Ensure RGB
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        return image
    
    def set_image(self, image: Union[np.ndarray, Image.Image]) -> None:
        """
        Set the image for prediction (use for batch processing).
        
        Args:
            image: Input image
        """
        self._load_model()
        
        image_np = self._prepare_image(image)
        self._predictor.set_image(image_np)
        self._current_image = image_np
    
    def refine(
        self,
        image: Union[np.ndarray, Image.Image],
        box: BoundingBox,
        initial_mask: Optional[np.ndarray] = None
    ) -> RefinedMask:
        """
        Refine mask for a detected region using SAM.
        
        Args:
            image: Input image
            box: Bounding box for the region
            initial_mask: Optional initial mask to refine
            
        Returns:
            RefinedMask with precise segmentation
        """
        self._load_model()
        
        # Prepare image
        image_np = self._prepare_image(image)
        
        # Set image (caches embedding)
        self._predictor.set_image(image_np)
        
        # Prepare box prompt
        box_array = np.array([[box.x1, box.y1, box.x2, box.y2]])
        
        # Prepare mask prompt if provided
        mask_input = None
        if initial_mask is not None:
            # SAM expects logits, so convert binary mask
            mask_input = (initial_mask.astype(np.float32) / 255.0 * 20 - 10)
            mask_input = mask_input[None, :, :]  # Add batch dimension
        
        # Run prediction
        masks, scores, _ = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_array,
            mask_input=mask_input,
            multimask_output=self.multimask_output
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8) * 255
        score = float(scores[best_idx])
        
        return RefinedMask(
            mask=mask,
            score=score,
            box=box
        )
    
    def refine_batch(
        self,
        image: Union[np.ndarray, Image.Image],
        boxes: List[BoundingBox]
    ) -> List[RefinedMask]:
        """
        Refine masks for multiple boxes in the same image.
        
        More efficient than calling refine() multiple times as
        the image embedding is computed only once.
        
        Args:
            image: Input image
            boxes: List of bounding boxes
            
        Returns:
            List of RefinedMasks
        """
        self._load_model()
        
        # Prepare and set image once
        image_np = self._prepare_image(image)
        self._predictor.set_image(image_np)
        
        results = []
        for box in boxes:
            try:
                # Prepare box prompt
                box_array = np.array([[box.x1, box.y1, box.x2, box.y2]])
                
                # Run prediction
                masks, scores, _ = self._predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_array,
                    multimask_output=self.multimask_output
                )
                
                # Select best mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx].astype(np.uint8) * 255
                score = float(scores[best_idx])
                
                results.append(RefinedMask(
                    mask=mask,
                    score=score,
                    box=box
                ))
                
            except Exception as e:
                logger.warning(f"SAM refinement failed for box {box}: {e}")
                # Create fallback box mask
                results.append(self._create_box_mask(image_np.shape[:2], box))
        
        return results
    
    def _create_box_mask(
        self, 
        shape: Tuple[int, int], 
        box: BoundingBox
    ) -> RefinedMask:
        """Create a simple rectangular mask as fallback."""
        mask = np.zeros(shape, dtype=np.uint8)
        mask[box.y1:box.y2, box.x1:box.x2] = 255
        return RefinedMask(mask=mask, score=0.5, box=box)
    
    def refine_with_points(
        self,
        image: Union[np.ndarray, Image.Image],
        points: np.ndarray,
        labels: np.ndarray,
        box: Optional[BoundingBox] = None
    ) -> RefinedMask:
        """
        Refine mask using point prompts (in addition to box).
        
        Args:
            image: Input image
            points: Array of point coordinates (N, 2)
            labels: Array of point labels (N,) - 1 for foreground, 0 for background
            box: Optional bounding box
            
        Returns:
            RefinedMask with precise segmentation
        """
        self._load_model()
        
        image_np = self._prepare_image(image)
        self._predictor.set_image(image_np)
        
        box_array = None
        if box is not None:
            box_array = np.array([[box.x1, box.y1, box.x2, box.y2]])
        
        masks, scores, _ = self._predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box_array,
            multimask_output=self.multimask_output
        )
        
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8) * 255
        
        return RefinedMask(
            mask=mask,
            score=float(scores[best_idx]),
            box=box
        )


# ============================================================================
# Fallback Refiner (no SAM required)
# ============================================================================

class FallbackRefiner(BaseMaskRefiner):
    """
    Simple fallback refiner that creates rectangular masks from boxes.
    Use when SAM is not available.
    """
    
    def __init__(self):
        logger.info("FallbackRefiner initialized (box-based masks)")
    
    def _prepare_image(
        self, 
        image: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Prepare image."""
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, :3]
        return image
    
    def refine(
        self,
        image: Union[np.ndarray, Image.Image],
        box: BoundingBox,
        initial_mask: Optional[np.ndarray] = None
    ) -> RefinedMask:
        """Create rectangular mask from box."""
        image_np = self._prepare_image(image)
        h, w = image_np.shape[:2]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[box.y1:box.y2, box.x1:box.x2] = 255
        
        return RefinedMask(
            mask=mask,
            score=0.7,
            box=box
        )
    
    def refine_batch(
        self,
        image: Union[np.ndarray, Image.Image],
        boxes: List[BoundingBox]
    ) -> List[RefinedMask]:
        """Create rectangular masks for all boxes."""
        return [self.refine(image, box) for box in boxes]


# ============================================================================
# Factory Function
# ============================================================================

def get_refiner(
    use_sam: bool = True,
    model_type: str = SAM_MODEL_TYPE,
    **kwargs
) -> BaseMaskRefiner:
    """
    Factory function to get the appropriate mask refiner.
    
    Args:
        use_sam: Whether to try loading SAM model
        model_type: SAM model variant
        **kwargs: Additional arguments for the refiner
        
    Returns:
        Refiner instance
    """
    if use_sam:
        try:
            import torch
            from segment_anything import sam_model_registry
            return SAMRefiner(model_type=model_type, **kwargs)
        except ImportError:
            logger.warning("SAM not available, using fallback refiner")
    
    return FallbackRefiner()
