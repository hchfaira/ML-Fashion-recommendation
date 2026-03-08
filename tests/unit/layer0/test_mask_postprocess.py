"""
Tests for Layer 0: Mask Post-processing
"""

import pytest
import numpy as np


class TestMaskPostprocessor:
    """Tests for MaskPostprocessor class."""
    
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        assert processor.config is not None
        assert processor.config.closing_kernel_size == 15
        assert processor.config.fill_holes is True
    
    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        from src.layer0_segmentation import MaskPostprocessor, PostprocessConfig
        
        config = PostprocessConfig(
            closing_kernel_size=10,
            fill_holes=False,
            keep_largest_only=False
        )
        
        processor = MaskPostprocessor(config)
        
        assert processor.config.closing_kernel_size == 10
        assert processor.config.fill_holes is False
    
    def test_ensure_binary(self):
        """Test binary mask conversion."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        # Test with 0-1 mask
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        result = processor._ensure_binary(mask)
        
        assert result.max() == 255
        assert result.min() == 0
        
        # Test with 0-255 mask
        mask = np.array([[0, 255], [128, 50]], dtype=np.uint8)
        result = processor._ensure_binary(mask)
        
        assert result[0, 0] == 0
        assert result[0, 1] == 255
        assert result[1, 0] == 255  # 128 > 127
        assert result[1, 1] == 0  # 50 < 127
    
    def test_morphological_close(self):
        """Test morphological closing fills small holes."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        # Create mask with small hole
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        mask[24:26, 24:26] = 0  # Small 2x2 hole
        
        result = processor._morphological_close(mask)
        
        # Small hole should be filled
        assert result[25, 25] == 255
    
    def test_fill_holes(self):
        """Test hole filling."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        # Create mask with interior hole
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        mask[20:30, 20:30] = 0  # Interior hole
        
        result = processor._fill_holes(mask)
        
        # Interior hole should be filled
        assert result[25, 25] == 255
    
    def test_remove_small_components(self):
        """Test small component removal."""
        from src.layer0_segmentation import MaskPostprocessor, PostprocessConfig
        
        config = PostprocessConfig(min_component_area=100, keep_largest_only=True)
        processor = MaskPostprocessor(config)
        
        # Create mask with large and small components
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:60, 10:60] = 255  # Large component (50x50 = 2500 pixels)
        mask[80:85, 80:85] = 255  # Small component (5x5 = 25 pixels)
        
        result = processor._remove_small_components(mask)
        
        # Large component should remain
        assert result[35, 35] == 255
        # Small component should be removed
        assert result[82, 82] == 0
    
    def test_smooth_contours(self):
        """Test contour smoothing."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        # Create mask with jagged edges
        mask = np.zeros((100, 100), dtype=np.uint8)
        for i in range(20, 80):
            for j in range(20, 80):
                # Add some noise to edges
                if 20 <= i <= 79 and 20 <= j <= 79:
                    mask[i, j] = 255
        
        result = processor._smooth_contours(mask)
        
        # Result should still have mask content
        assert result.sum() > 0
    
    def test_process_pipeline(self):
        """Test full processing pipeline."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        # Create realistic mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        mask[45:55, 45:55] = 0  # Interior hole
        
        result = processor.process(mask)
        
        assert result.shape == (100, 100)
        assert result.max() == 255
        # Hole should be filled
        assert result[50, 50] == 255
    
    def test_process_batch(self):
        """Test batch processing."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        masks = [
            np.ones((50, 50), dtype=np.uint8) * 255,
            np.ones((50, 50), dtype=np.uint8) * 255
        ]
        
        results = processor.process_batch(masks)
        
        assert len(results) == 2
    
    def test_get_bounding_box(self):
        """Test bounding box extraction."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 255
        
        bbox = processor.get_bounding_box(mask)
        
        assert bbox is not None
        x, y, w, h = bbox
        assert x == 30
        assert y == 20
        assert w == 40
        assert h == 40
    
    def test_get_bounding_box_empty_mask(self):
        """Test bounding box extraction with empty mask."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        bbox = processor.get_bounding_box(mask)
        
        assert bbox is None
    
    def test_crop_to_content(self):
        """Test cropping to content."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 255
        
        cropped, bbox = processor.crop_to_content(mask, padding=0)
        
        assert cropped.shape == (40, 40)
        assert bbox == (30, 20, 40, 40)
    
    def test_crop_to_content_with_padding(self):
        """Test cropping with padding."""
        from src.layer0_segmentation import MaskPostprocessor
        
        processor = MaskPostprocessor()
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 30:60] = 255
        
        cropped, bbox = processor.crop_to_content(mask, padding=5)
        
        assert cropped.shape == (40, 40)  # 30 + 2*5
    
    def test_feather_mask(self):
        """Test mask feathering."""
        from src.layer0_segmentation import MaskPostprocessor, PostprocessConfig
        
        config = PostprocessConfig(feather_edges=True, feather_amount=5)
        processor = MaskPostprocessor(config)
        
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[15:35, 15:35] = 255
        
        feathered = processor.feather_mask(mask)
        
        # Center should be fully opaque
        assert feathered[25, 25] == 255
        # Should have gradient at edges
        # (exact values depend on distance transform)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_clean_mask(self):
        """Test clean_mask function."""
        from src.layer0_segmentation import clean_mask
        
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        mask[24:26, 24:26] = 0  # Small hole
        
        result = clean_mask(mask)
        
        assert result.shape == (50, 50)
        # Hole should be filled
        assert result[25, 25] == 255
    
    def test_clean_masks(self):
        """Test clean_masks function."""
        from src.layer0_segmentation import clean_masks
        
        masks = [
            np.ones((50, 50), dtype=np.uint8) * 255,
            np.ones((50, 50), dtype=np.uint8) * 255
        ]
        
        results = clean_masks(masks)
        
        assert len(results) == 2
