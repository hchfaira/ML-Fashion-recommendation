"""
Tests for Layer 0: SAM Mask Refiner
"""

import pytest
import numpy as np
from PIL import Image


class TestFallbackRefiner:
    """Tests for FallbackRefiner (no model required)."""
    
    def test_refine_creates_box_mask(self):
        """Test that fallback refiner creates rectangular mask."""
        from src.layer0_segmentation import FallbackRefiner, BoundingBox, RefinedMask
        
        refiner = FallbackRefiner()
        
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        box = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        
        result = refiner.refine(image, box)
        
        assert isinstance(result, RefinedMask)
        assert result.shape == (200, 300)
        assert result.mask[100, 100] == 255  # Inside box
        assert result.mask[10, 10] == 0  # Outside box
        assert result.score == 0.7
    
    def test_refine_with_pil_image(self):
        """Test refining with PIL Image."""
        from src.layer0_segmentation import FallbackRefiner, BoundingBox
        
        refiner = FallbackRefiner()
        
        image = Image.new("RGB", (300, 200))
        box = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        
        result = refiner.refine(image, box)
        
        assert result.shape == (200, 300)
    
    def test_refine_batch(self):
        """Test refining multiple boxes."""
        from src.layer0_segmentation import FallbackRefiner, BoundingBox
        
        refiner = FallbackRefiner()
        
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        boxes = [
            BoundingBox(x1=0, y1=0, x2=100, y2=100),
            BoundingBox(x1=100, y1=100, x2=200, y2=200)
        ]
        
        results = refiner.refine_batch(image, boxes)
        
        assert len(results) == 2
        assert results[0].mask[50, 50] == 255
        assert results[1].mask[150, 150] == 255


class TestSAMRefiner:
    """Tests for SAMRefiner (model-based, needs mocking without model)."""
    
    def test_initialization(self):
        """Test refiner initialization."""
        from src.layer0_segmentation.sam_refiner import SAMRefiner
        
        refiner = SAMRefiner(model_type="vit_h")
        
        assert refiner.model_type == "vit_h"
        assert refiner._sam is None  # Not loaded yet
        assert refiner.multimask_output is True
    
    def test_get_device(self):
        """Test device detection."""
        from src.layer0_segmentation.sam_refiner import SAMRefiner
        
        refiner = SAMRefiner()
        
        device = refiner._get_device()
        
        assert device in ["cuda", "mps", "cpu"]
    
    def test_prepare_image_rgb(self):
        """Test preparing RGB image."""
        from src.layer0_segmentation.sam_refiner import SAMRefiner
        
        refiner = SAMRefiner()
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        prepared = refiner._prepare_image(image)
        
        assert prepared.shape == (100, 100, 3)
    
    def test_prepare_image_rgba(self):
        """Test preparing RGBA image."""
        from src.layer0_segmentation.sam_refiner import SAMRefiner
        
        refiner = SAMRefiner()
        
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        prepared = refiner._prepare_image(image)
        
        assert prepared.shape == (100, 100, 3)  # Converted to RGB
    
    def test_prepare_image_pil(self):
        """Test preparing PIL Image."""
        from src.layer0_segmentation.sam_refiner import SAMRefiner
        
        refiner = SAMRefiner()
        
        image = Image.new("RGB", (100, 100))
        prepared = refiner._prepare_image(image)
        
        assert prepared.shape == (100, 100, 3)
    
    def test_create_box_mask(self):
        """Test creating box mask as fallback."""
        from src.layer0_segmentation.sam_refiner import SAMRefiner
        from src.layer0_segmentation import BoundingBox
        
        refiner = SAMRefiner()
        
        box = BoundingBox(x1=10, y1=20, x2=50, y2=80)
        result = refiner._create_box_mask((100, 100), box)
        
        assert result.mask.shape == (100, 100)
        assert result.mask[50, 30] == 255  # Inside box
        assert result.mask[5, 5] == 0  # Outside box
        assert result.score == 0.5


class TestGetRefiner:
    """Tests for get_refiner factory function."""
    
    def test_returns_fallback_when_disabled(self):
        """Test that factory returns FallbackRefiner when SAM disabled."""
        from src.layer0_segmentation import get_refiner, FallbackRefiner
        
        refiner = get_refiner(use_sam=False)
        
        assert isinstance(refiner, FallbackRefiner)
    
    def test_returns_refiner_when_enabled(self):
        """Test that factory returns a refiner when enabled."""
        from src.layer0_segmentation import get_refiner, FallbackRefiner, SAMRefiner
        
        refiner = get_refiner(use_sam=True)
        
        # Will be SAMRefiner if SAM available, FallbackRefiner otherwise
        assert isinstance(refiner, (SAMRefiner, FallbackRefiner))
