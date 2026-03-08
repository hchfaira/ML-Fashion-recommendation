"""
Tests for Layer 0: Garment Extractor
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path


class TestGarmentExtractor:
    """Tests for GarmentExtractor class."""
    
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        from src.layer0_segmentation import GarmentExtractor
        
        extractor = GarmentExtractor()
        
        assert extractor.config.canvas_size == 512
        assert extractor.config.padding == 10
    
    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        from src.layer0_segmentation import GarmentExtractor, ExtractorConfig
        
        config = ExtractorConfig(
            canvas_size=256,
            padding=5
        )
        
        extractor = GarmentExtractor(config)
        
        assert extractor.config.canvas_size == 256
        assert extractor.config.padding == 5
    
    def test_load_image_numpy(self):
        """Test loading numpy image."""
        from src.layer0_segmentation import GarmentExtractor
        
        extractor = GarmentExtractor()
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = extractor._load_image(image)
        
        assert result.shape == (100, 100, 3)
    
    def test_load_image_rgba_to_rgb(self):
        """Test loading RGBA image converts to RGB."""
        from src.layer0_segmentation import GarmentExtractor
        
        extractor = GarmentExtractor()
        
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        result = extractor._load_image(image)
        
        assert result.shape == (100, 100, 3)
    
    def test_load_image_pil(self):
        """Test loading PIL image."""
        from src.layer0_segmentation import GarmentExtractor
        
        extractor = GarmentExtractor()
        
        image = Image.new("RGB", (100, 100))
        result = extractor._load_image(image)
        
        assert result.shape == (100, 100, 3)
    
    def test_load_image_from_path(self, tmp_path):
        """Test loading image from path."""
        from src.layer0_segmentation import GarmentExtractor
        
        extractor = GarmentExtractor()
        
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(img_path)
        
        result = extractor._load_image(img_path)
        
        assert result.shape == (100, 100, 3)
    
    def test_apply_mask(self):
        """Test applying mask to create RGBA."""
        from src.layer0_segmentation import GarmentExtractor
        
        extractor = GarmentExtractor()
        
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        mask[50:, :] = 0  # Bottom half transparent
        
        rgba = extractor._apply_mask(image, mask)
        
        assert rgba.shape == (100, 100, 4)
        assert rgba[25, 25, 3] == 255  # Top half opaque
        assert rgba[75, 25, 3] == 0  # Bottom half transparent
    
    def test_crop_to_content(self):
        """Test cropping to content."""
        from src.layer0_segmentation import GarmentExtractor
        
        extractor = GarmentExtractor()
        
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[20:60, 30:70, :3] = 128  # RGB content
        rgba[20:60, 30:70, 3] = 255  # Alpha
        
        cropped, bbox = extractor._crop_to_content(rgba, padding=0)
        
        assert cropped.shape == (40, 40, 4)
        assert bbox == (30, 20, 40, 40)
    
    def test_crop_to_content_empty(self):
        """Test cropping empty image."""
        from src.layer0_segmentation import GarmentExtractor
        
        extractor = GarmentExtractor()
        
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        
        cropped, bbox = extractor._crop_to_content(rgba)
        
        # Should return original
        assert cropped.shape == (100, 100, 4)
    
    def test_resize_and_center(self):
        """Test resize and center on canvas."""
        from src.layer0_segmentation import GarmentExtractor, ExtractorConfig
        
        config = ExtractorConfig(canvas_size=256)
        extractor = GarmentExtractor(config)
        
        img = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
        
        result = extractor._resize_and_center(img)
        
        assert result.size == (256, 256)
        assert result.mode == "RGBA"
    
    def test_resize_maintains_aspect_ratio(self):
        """Test that resize maintains aspect ratio."""
        from src.layer0_segmentation import GarmentExtractor, ExtractorConfig
        
        config = ExtractorConfig(canvas_size=256, maintain_aspect_ratio=True)
        extractor = GarmentExtractor(config)
        
        # Wide image
        img = Image.new("RGBA", (200, 100), (255, 0, 0, 255))
        
        result = extractor._resize_and_center(img)
        
        # Should fit within 256x256 maintaining aspect ratio
        assert result.size == (256, 256)
    
    def test_extract_single_garment(self):
        """Test extracting single garment."""
        from src.layer0_segmentation import (
            GarmentExtractor, FusedMask, GarmentCategory, ExtractorConfig
        )
        
        config = ExtractorConfig(canvas_size=128)
        extractor = GarmentExtractor(config)
        
        # Create test image and mask
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        
        fused = FusedMask(
            mask=mask,
            category=GarmentCategory.TOPS,
            label="t-shirt",
            confidence=0.9
        )
        
        result = extractor.extract(image, fused)
        
        assert result.image.size == (128, 128)
        assert result.image.mode == "RGBA"
        assert result.category == GarmentCategory.TOPS
        assert result.label == "t-shirt"
        assert result.confidence == 0.9
    
    def test_extract_all(self):
        """Test extracting multiple garments."""
        from src.layer0_segmentation import (
            GarmentExtractor, FusedMask, GarmentCategory, ExtractorConfig
        )
        
        config = ExtractorConfig(canvas_size=128)
        extractor = GarmentExtractor(config)
        
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        
        masks = [
            FusedMask(
                mask=np.ones((100, 100), dtype=np.uint8) * 255,
                category=GarmentCategory.TOPS,
                label="t-shirt",
                confidence=0.9
            ),
            FusedMask(
                mask=np.ones((100, 100), dtype=np.uint8) * 255,
                category=GarmentCategory.BOTTOMS,
                label="jeans",
                confidence=0.85
            )
        ]
        
        result = extractor.extract_all(image, masks)
        
        assert len(result) == 2
        assert result.processing_time_ms > 0
    
    def test_save_garment(self, tmp_path):
        """Test saving garment to file."""
        from src.layer0_segmentation import (
            GarmentExtractor, ExtractedGarment, GarmentCategory
        )
        
        extractor = GarmentExtractor()
        
        garment = ExtractedGarment(
            image=Image.new("RGBA", (100, 100)),
            category=GarmentCategory.TOPS,
            label="t-shirt",
            confidence=0.9
        )
        
        output_path = extractor.save_garment(garment, tmp_path, index=1)
        
        assert output_path.exists()
        assert output_path.name == "tops_001_t_shirt.png"
    
    def test_save_all(self, tmp_path):
        """Test saving all garments."""
        from src.layer0_segmentation import (
            GarmentExtractor, ExtractionResult, ExtractedGarment, GarmentCategory
        )
        
        extractor = GarmentExtractor()
        
        garments = [
            ExtractedGarment(
                image=Image.new("RGBA", (100, 100)),
                category=GarmentCategory.TOPS,
                label="t-shirt",
                confidence=0.9
            ),
            ExtractedGarment(
                image=Image.new("RGBA", (100, 100)),
                category=GarmentCategory.BOTTOMS,
                label="jeans",
                confidence=0.85
            )
        ]
        
        result = ExtractionResult(garments=garments)
        
        paths = extractor.save_all(result, tmp_path)
        
        assert len(paths) == 2
        assert all(p.exists() for p in paths)


class TestExtractGarmentFunction:
    """Tests for extract_garment convenience function."""
    
    def test_extract_garment(self):
        """Test extract_garment function."""
        from src.layer0_segmentation import extract_garment, GarmentCategory
        
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        result = extract_garment(
            image=image,
            mask=mask,
            category=GarmentCategory.TOPS,
            label="sweater",
            canvas_size=256
        )
        
        assert result.image.size == (256, 256)
        assert result.category == GarmentCategory.TOPS
        assert result.label == "sweater"
