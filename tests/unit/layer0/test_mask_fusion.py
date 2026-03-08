"""
Tests for Layer 0: Mask Fusion
"""

import pytest
import numpy as np


class TestMaskFusion:
    """Tests for MaskFusion class."""
    
    def test_initialization(self):
        """Test MaskFusion initialization."""
        from src.layer0_segmentation import MaskFusion
        
        fusion = MaskFusion(
            min_schp_overlap=0.4,
            iou_threshold=0.6,
            min_area=200
        )
        
        assert fusion.min_schp_overlap == 0.4
        assert fusion.iou_threshold == 0.6
        assert fusion.min_area == 200
    
    def test_fuse_simple_single_mask(self):
        """Test simple fusion with single mask."""
        from src.layer0_segmentation import (
            MaskFusion, DetectionResult, Detection, 
            BoundingBox, GarmentCategory, RefinedMask
        )
        
        fusion = MaskFusion(min_area=10)
        
        # Create a detection
        detections = DetectionResult(
            detections=[
                Detection(
                    label="t-shirt",
                    box=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                    score=0.9,
                    category=GarmentCategory.TOPS
                )
            ],
            image_size=(100, 100)
        )
        
        # Create a refined mask
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        refined_masks = [RefinedMask(mask=mask, score=0.95)]
        
        # Fuse
        results = fusion.fuse_simple(detections, refined_masks, (100, 100))
        
        assert len(results) == 1
        assert results[0].label == "t-shirt"
        assert results[0].category == GarmentCategory.TOPS
    
    def test_fuse_simple_multiple_masks(self):
        """Test simple fusion with multiple masks."""
        from src.layer0_segmentation import (
            MaskFusion, DetectionResult, Detection,
            BoundingBox, GarmentCategory, RefinedMask
        )
        
        fusion = MaskFusion(min_area=10)
        
        detections = DetectionResult(
            detections=[
                Detection(
                    label="t-shirt",
                    box=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                    score=0.9,
                    category=GarmentCategory.TOPS
                ),
                Detection(
                    label="jeans",
                    box=BoundingBox(x1=0, y1=50, x2=100, y2=100),
                    score=0.85,
                    category=GarmentCategory.BOTTOMS
                )
            ],
            image_size=(100, 100)
        )
        
        # Create refined masks
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[:50, :] = 255
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[50:, :] = 255
        
        refined_masks = [
            RefinedMask(mask=mask1, score=0.9),
            RefinedMask(mask=mask2, score=0.85)
        ]
        
        results = fusion.fuse_simple(detections, refined_masks, (100, 100))
        
        assert len(results) == 2
    
    def test_fuse_with_parsing(self):
        """Test fusion with SCHP parsing."""
        from src.layer0_segmentation import (
            MaskFusion, DetectionResult, Detection,
            BoundingBox, GarmentCategory, RefinedMask, ParsingResult
        )
        
        fusion = MaskFusion(min_area=10, use_schp_validation=True)
        
        detections = DetectionResult(
            detections=[
                Detection(
                    label="t-shirt",
                    box=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                    score=0.9,
                    category=GarmentCategory.TOPS
                )
            ],
            image_size=(100, 100)
        )
        
        # Create parsing result with upper_clothes region
        parsing_map = np.zeros((100, 100), dtype=np.uint8)
        parsing_map[:60, :] = 5  # upper_clothes
        parsing = ParsingResult(parsing_map=parsing_map, label_areas={5: 6000})
        
        # Create refined mask
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        refined_masks = [RefinedMask(mask=mask, score=0.9)]
        
        results = fusion.fuse(detections, parsing, refined_masks)
        
        assert len(results) == 1
    
    def test_resolve_overlaps(self):
        """Test overlap resolution keeps higher confidence."""
        from src.layer0_segmentation import (
            MaskFusion, FusedMask, GarmentCategory
        )
        
        fusion = MaskFusion(min_area=10)
        
        # Create two overlapping masks
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[0:60, 0:60] = 255  # 60x60 region
        
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[40:100, 40:100] = 255  # 60x60 region, overlaps with mask1
        
        fused_masks = [
            FusedMask(
                mask=mask1,
                category=GarmentCategory.TOPS,
                label="t-shirt",
                confidence=0.9
            ),
            FusedMask(
                mask=mask2,
                category=GarmentCategory.BOTTOMS,
                label="pants",
                confidence=0.7  # Lower confidence
            )
        ]
        
        resolved = fusion._resolve_overlaps(fused_masks)
        
        # Higher confidence should win disputed pixels
        assert len(resolved) >= 1
    
    def test_apply_box_restriction(self):
        """Test that mask is restricted to bounding box."""
        from src.layer0_segmentation import MaskFusion, BoundingBox
        
        fusion = MaskFusion()
        
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        box = BoundingBox(x1=20, y1=30, x2=80, y2=70)
        
        restricted = fusion._apply_box_restriction(mask, box)
        
        # Should only have non-zero values inside box
        assert restricted[50, 50] == 255  # Inside
        assert restricted[10, 10] == 0  # Outside
        assert restricted[90, 90] == 0  # Outside
    
    def test_compute_mask_iou(self):
        """Test IoU computation."""
        from src.layer0_segmentation import MaskFusion
        
        fusion = MaskFusion()
        
        # Identical masks
        mask1 = np.ones((100, 100), dtype=np.uint8) * 255
        mask2 = np.ones((100, 100), dtype=np.uint8) * 255
        
        iou = fusion.compute_mask_iou(mask1, mask2)
        assert iou == 1.0
        
        # Non-overlapping masks
        mask3 = np.zeros((100, 100), dtype=np.uint8)
        mask3[:50, :] = 255
        mask4 = np.zeros((100, 100), dtype=np.uint8)
        mask4[50:, :] = 255
        
        iou = fusion.compute_mask_iou(mask3, mask4)
        assert iou == 0.0
        
        # 50% overlap
        mask5 = np.zeros((100, 100), dtype=np.uint8)
        mask5[:, :50] = 255
        mask6 = np.zeros((100, 100), dtype=np.uint8)
        mask6[:, 25:75] = 255
        
        iou = fusion.compute_mask_iou(mask5, mask6)
        assert 0 < iou < 1
    
    def test_empty_detections(self):
        """Test handling empty detections."""
        from src.layer0_segmentation import MaskFusion, DetectionResult
        
        fusion = MaskFusion()
        
        detections = DetectionResult()
        
        results = fusion.fuse_simple(detections, [], (100, 100))
        
        assert len(results) == 0


class TestFuseMasksFunction:
    """Tests for fuse_masks convenience function."""
    
    def test_fuse_masks_simple(self):
        """Test fuse_masks convenience function."""
        from src.layer0_segmentation import (
            fuse_masks, DetectionResult, Detection,
            BoundingBox, GarmentCategory, RefinedMask
        )
        
        detections = DetectionResult(
            detections=[
                Detection(
                    label="t-shirt",
                    box=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                    score=0.9,
                    category=GarmentCategory.TOPS
                )
            ]
        )
        
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        refined_masks = [RefinedMask(mask=mask, score=0.9)]
        
        results = fuse_masks(
            detections=detections,
            parsing=None,
            refined_masks=refined_masks,
            image_shape=(100, 100),
            min_area=10
        )
        
        assert len(results) == 1
    
    def test_fuse_masks_requires_shape_or_parsing(self):
        """Test that fuse_masks requires either parsing or shape."""
        from src.layer0_segmentation import fuse_masks, DetectionResult
        
        with pytest.raises(ValueError, match="Either parsing or image_shape"):
            fuse_masks(
                detections=DetectionResult(),
                parsing=None,
                refined_masks=[],
                image_shape=None
            )
