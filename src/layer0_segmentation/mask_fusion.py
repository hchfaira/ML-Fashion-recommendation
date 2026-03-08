"""
Mask Fusion
============

Combines masks from multiple sources:
- GroundingDINO bounding boxes
- SCHP semantic regions
- SAM refined masks

Strategy:
1. Use SAM mask as the base (best edge precision)
2. Filter using SCHP class region (semantic consistency)
3. Restrict to detection bounding box
4. Resolve overlapping garments
5. Map to GARMENT_TAXONOMY categories
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2

from src.core import get_logger
from .taxonomy import GarmentCategory, SCHP_TO_TAXONOMY, get_schp_category
from .models import (
    Detection, DetectionResult, ParsingResult, 
    RefinedMask, FusedMask, BoundingBox
)

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Minimum overlap ratio for SCHP validation
MIN_SCHP_OVERLAP = 0.3

# IoU threshold for resolving overlapping masks
IOU_THRESHOLD = 0.5

# Minimum mask area (pixels) to keep
MIN_MASK_AREA = 100


# ============================================================================
# Mask Fusion Logic
# ============================================================================

class MaskFusion:
    """
    Fuses masks from detection, parsing, and refinement stages.
    
    Combines the strengths of each source:
    - GroundingDINO: Object-level detection with labels
    - SCHP: Semantic clothing classification
    - SAM: Precise mask boundaries
    """
    
    def __init__(
        self,
        min_schp_overlap: float = MIN_SCHP_OVERLAP,
        iou_threshold: float = IOU_THRESHOLD,
        min_area: int = MIN_MASK_AREA,
        use_schp_validation: bool = True
    ):
        """
        Initialize the mask fusion module.
        
        Args:
            min_schp_overlap: Minimum overlap ratio with SCHP region
            iou_threshold: IoU threshold for overlap resolution
            min_area: Minimum mask area to keep
            use_schp_validation: Whether to validate against SCHP regions
        """
        self.min_schp_overlap = min_schp_overlap
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        self.use_schp_validation = use_schp_validation
        
        logger.info(
            f"MaskFusion initialized "
            f"(schp_overlap={min_schp_overlap}, iou={iou_threshold})"
        )
    
    def fuse(
        self,
        detections: DetectionResult,
        parsing: ParsingResult,
        refined_masks: List[RefinedMask]
    ) -> List[FusedMask]:
        """
        Fuse masks from all sources.
        
        Args:
            detections: Results from GroundingDINO
            parsing: Results from SCHP
            refined_masks: Results from SAM
            
        Returns:
            List of FusedMask objects
        """
        if len(detections) == 0:
            logger.warning("No detections to fuse")
            return []
        
        if len(refined_masks) != len(detections):
            logger.warning(
                f"Mismatch: {len(detections)} detections, "
                f"{len(refined_masks)} refined masks"
            )
        
        fused_masks = []
        
        for i, detection in enumerate(detections):
            # Get corresponding SAM mask
            if i < len(refined_masks):
                sam_mask = refined_masks[i]
            else:
                # Create fallback mask from box
                sam_mask = self._create_box_mask(
                    parsing.shape, detection.box
                )
            
            # Step 1: Start with SAM mask as base
            base_mask = sam_mask.mask.copy()
            
            # Step 2: Filter using SCHP if enabled
            if self.use_schp_validation:
                base_mask = self._apply_schp_filter(
                    base_mask, parsing, detection.category
                )
            
            # Step 3: Restrict to detection bounding box
            base_mask = self._apply_box_restriction(
                base_mask, detection.box
            )
            
            # Step 4: Validate mask area
            if np.sum(base_mask > 0) < self.min_area:
                logger.debug(f"Mask too small for {detection.label}, skipping")
                continue
            
            fused = FusedMask(
                mask=base_mask,
                detection=detection,
                category=detection.category,
                label=detection.label,
                confidence=detection.score * sam_mask.score
            )
            fused_masks.append(fused)
        
        # Step 5: Resolve overlapping masks
        fused_masks = self._resolve_overlaps(fused_masks)
        
        logger.info(f"Fused {len(fused_masks)} masks")
        return fused_masks
    
    def fuse_simple(
        self,
        detections: DetectionResult,
        refined_masks: List[RefinedMask],
        image_shape: Tuple[int, int]
    ) -> List[FusedMask]:
        """
        Simple fusion without SCHP parsing.
        
        Args:
            detections: Results from GroundingDINO
            refined_masks: Results from SAM
            image_shape: (H, W) of the image
            
        Returns:
            List of FusedMask objects
        """
        if len(detections) == 0:
            return []
        
        fused_masks = []
        
        for i, detection in enumerate(detections):
            if i < len(refined_masks):
                sam_mask = refined_masks[i]
            else:
                sam_mask = self._create_box_mask(image_shape, detection.box)
            
            base_mask = sam_mask.mask.copy()
            
            # Restrict to box
            base_mask = self._apply_box_restriction(base_mask, detection.box)
            
            if np.sum(base_mask > 0) < self.min_area:
                continue
            
            fused = FusedMask(
                mask=base_mask,
                detection=detection,
                category=detection.category,
                label=detection.label,
                confidence=detection.score * sam_mask.score
            )
            fused_masks.append(fused)
        
        fused_masks = self._resolve_overlaps(fused_masks)
        return fused_masks
    
    def _create_box_mask(
        self, 
        shape: Tuple[int, int], 
        box: BoundingBox
    ) -> RefinedMask:
        """Create a simple rectangular mask."""
        mask = np.zeros(shape, dtype=np.uint8)
        x1, y1 = max(0, box.x1), max(0, box.y1)
        x2, y2 = min(shape[1], box.x2), min(shape[0], box.y2)
        mask[y1:y2, x1:x2] = 255
        return RefinedMask(mask=mask, score=0.5, box=box)
    
    def _apply_schp_filter(
        self,
        mask: np.ndarray,
        parsing: ParsingResult,
        category: GarmentCategory
    ) -> np.ndarray:
        """
        Filter mask using SCHP semantic regions.
        
        Keep only pixels that match the expected clothing category.
        """
        # Get SCHP mask for this category
        schp_mask = parsing.get_mask_for_category(category)
        
        if schp_mask is None or not np.any(schp_mask):
            # No SCHP data for this category, keep original mask
            return mask
        
        # Calculate overlap
        intersection = np.logical_and(mask > 0, schp_mask > 0)
        mask_area = np.sum(mask > 0)
        
        if mask_area == 0:
            return mask
        
        overlap_ratio = np.sum(intersection) / mask_area
        
        if overlap_ratio < self.min_schp_overlap:
            # Low overlap - might be wrong category, but keep mask
            logger.debug(
                f"Low SCHP overlap ({overlap_ratio:.2f}) for {category.value}"
            )
            return mask
        
        # Apply SCHP filter - keep only overlapping region
        filtered_mask = np.where(
            schp_mask > 0, mask, 0
        ).astype(np.uint8)
        
        return filtered_mask
    
    def _apply_box_restriction(
        self,
        mask: np.ndarray,
        box: BoundingBox
    ) -> np.ndarray:
        """Restrict mask to bounding box region."""
        h, w = mask.shape
        
        # Create box mask
        box_mask = np.zeros_like(mask)
        x1, y1 = max(0, box.x1), max(0, box.y1)
        x2, y2 = min(w, box.x2), min(h, box.y2)
        box_mask[y1:y2, x1:x2] = 255
        
        # Apply restriction
        restricted = np.where(box_mask > 0, mask, 0).astype(np.uint8)
        
        return restricted
    
    def _resolve_overlaps(
        self,
        masks: List[FusedMask]
    ) -> List[FusedMask]:
        """
        Resolve overlapping masks by keeping higher confidence regions.
        
        For each pair of overlapping masks, assign disputed pixels
        to the mask with higher confidence.
        """
        if len(masks) <= 1:
            return masks
        
        # Sort by confidence (highest first)
        masks = sorted(masks, key=lambda m: m.confidence, reverse=True)
        
        # Create combined mask to track assigned pixels
        if len(masks) > 0:
            h, w = masks[0].mask.shape
            assigned = np.zeros((h, w), dtype=np.uint8)
            
            resolved = []
            for mask in masks:
                # Remove already assigned pixels
                current = mask.mask.copy()
                current = np.where(assigned > 0, 0, current).astype(np.uint8)
                
                # Check if enough remains
                if np.sum(current > 0) < self.min_area:
                    logger.debug(
                        f"Mask for {mask.label} too small after overlap removal"
                    )
                    continue
                
                # Mark pixels as assigned
                assigned = np.where(current > 0, 255, assigned).astype(np.uint8)
                
                # Update mask
                mask.mask = current
                resolved.append(mask)
            
            return resolved
        
        return masks
    
    def compute_mask_iou(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> float:
        """Compute Intersection over Union for two masks."""
        intersection = np.logical_and(mask1 > 0, mask2 > 0)
        union = np.logical_or(mask1 > 0, mask2 > 0)
        
        union_area = np.sum(union)
        if union_area == 0:
            return 0.0
        
        return np.sum(intersection) / union_area


# ============================================================================
# Convenience Functions
# ============================================================================

def fuse_masks(
    detections: DetectionResult,
    parsing: Optional[ParsingResult],
    refined_masks: List[RefinedMask],
    image_shape: Optional[Tuple[int, int]] = None,
    **kwargs
) -> List[FusedMask]:
    """
    Convenience function to fuse masks from all sources.
    
    Args:
        detections: Results from GroundingDINO
        parsing: Results from SCHP (optional)
        refined_masks: Results from SAM
        image_shape: Image shape if parsing not provided
        **kwargs: Additional arguments for MaskFusion
        
    Returns:
        List of FusedMask objects
    """
    fusion = MaskFusion(**kwargs)
    
    if parsing is not None:
        return fusion.fuse(detections, parsing, refined_masks)
    elif image_shape is not None:
        return fusion.fuse_simple(detections, refined_masks, image_shape)
    else:
        raise ValueError("Either parsing or image_shape must be provided")
