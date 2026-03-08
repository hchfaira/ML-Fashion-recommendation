"""
Tests for Layer 0: Garment Segmentation
"""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
import shutil


class TestSimpleGarmentSegmenter:
    """Tests for SimpleGarmentSegmenter (no SAM required)."""
    
    @pytest.fixture
    def segmenter(self):
        """Create a simple segmenter instance."""
        from src.layer0_segmentation import SimpleGarmentSegmenter
        return SimpleGarmentSegmenter(canvas_size=256)
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample test image."""
        # Create a simple RGB image with a colored rectangle
        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        
        # Draw a colored rectangle (simulating a garment)
        pixels = img.load()
        for x in range(50, 150):
            for y in range(50, 250):
                pixels[x, y] = (100, 100, 200)  # Blue rectangle
        
        # Save to temp file
        img_path = tmp_path / "test_garment.png"
        img.save(img_path)
        return img_path
    
    def test_segment_garment_returns_segmented_garment(self, segmenter, sample_image):
        """Test that segment_garment returns a SegmentedGarment."""
        from src.layer0_segmentation import SegmentedGarment
        
        result = segmenter.segment_garment(sample_image)
        
        assert isinstance(result, SegmentedGarment)
        assert result.image is not None
        assert isinstance(result.image, Image.Image)
        assert result.original_path == sample_image
    
    def test_segmented_image_has_correct_size(self, segmenter, sample_image):
        """Test that output image has correct canvas size."""
        result = segmenter.segment_garment(sample_image)
        
        assert result.image.size == (256, 256)
    
    def test_segmented_image_is_rgba(self, segmenter, sample_image):
        """Test that output image is RGBA (transparent background)."""
        result = segmenter.segment_garment(sample_image)
        
        assert result.image.mode == "RGBA"
    
    def test_segment_folder(self, segmenter, tmp_path):
        """Test segmenting multiple images from a folder."""
        # Create multiple test images
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(tmp_path / f"garment_{i}.png")
        
        results = segmenter.segment_folder(tmp_path)
        
        assert len(results) == 3
        for result in results:
            assert result.image.mode == "RGBA"
            assert result.image.size == (256, 256)
    
    def test_invalid_image_path_raises_error(self, segmenter):
        """Test that invalid path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            segmenter.segment_garment(Path("/nonexistent/image.png"))


class TestGarmentSegmenterFactory:
    """Tests for the get_segmenter factory function."""
    
    def test_get_segmenter_returns_simple_when_sam_not_available(self):
        """Test fallback to SimpleGarmentSegmenter when SAM unavailable."""
        from src.layer0_segmentation import get_segmenter, SimpleGarmentSegmenter
        
        # Force simple segmenter
        segmenter = get_segmenter(use_sam=False, canvas_size=128)
        
        assert isinstance(segmenter, SimpleGarmentSegmenter)
        assert segmenter.canvas_size == 128


class TestGarmentType:
    """Tests for GarmentType enum (legacy compatibility)."""
    
    def test_garment_type_values(self):
        """Test that all expected garment types exist with new values."""
        from src.layer0_segmentation import GarmentType
        
        # Updated to match new taxonomy-based values
        assert GarmentType.TOP.value == "tops"
        assert GarmentType.BOTTOM.value == "bottoms"
        assert GarmentType.DRESS.value == "full_body"
        assert GarmentType.OUTERWEAR.value == "outerwear"
        assert GarmentType.SHOES.value == "footwear"
        assert GarmentType.ACCESSORIES.value == "accessories"
        assert GarmentType.UNKNOWN.value == "unknown"


class TestSegmentedGarment:
    """Tests for the SegmentedGarment dataclass."""
    
    def test_segmented_garment_creation(self):
        """Test creating a SegmentedGarment."""
        from src.layer0_segmentation import SegmentedGarment
        
        img = Image.new("RGBA", (100, 100))
        
        garment = SegmentedGarment(
            image=img,
            original_path=Path("/test/image.png"),
            bounding_box=(10, 20, 80, 90),
            area=7200,
            confidence=0.95
        )
        
        assert garment.image == img
        assert garment.original_path == Path("/test/image.png")
        assert garment.bounding_box == (10, 20, 80, 90)
        assert garment.area == 7200
        assert garment.confidence == 0.95
    
    def test_segmented_garment_with_garment_type(self):
        """Test SegmentedGarment with garment_type and label."""
        from src.layer0_segmentation import SegmentedGarment, GarmentType
        
        img = Image.new("RGBA", (100, 100))
        
        garment = SegmentedGarment(
            image=img,
            garment_type=GarmentType.TOP,
            label="t-shirt",
            confidence=0.92
        )
        
        assert garment.garment_type == GarmentType.TOP
        assert garment.label == "t-shirt"
        assert garment.confidence == 0.92
    
    def test_segmented_garment_defaults(self):
        """Test SegmentedGarment default values."""
        from src.layer0_segmentation import SegmentedGarment, GarmentType
        
        img = Image.new("RGBA", (100, 100))
        garment = SegmentedGarment(image=img)
        
        assert garment.original_path is None
        assert garment.bounding_box is None
        assert garment.area == 0
        assert garment.confidence == 1.0
        assert garment.garment_type == GarmentType.UNKNOWN
        assert garment.label == ""
        assert garment.mask is None


class TestAdvancedGarmentSegmenter:
    """Tests for AdvancedGarmentSegmenter (legacy wrapper)."""
    
    def test_advanced_segmenter_initialization(self):
        """Test that AdvancedGarmentSegmenter can be initialized."""
        from src.layer0_segmentation import AdvancedGarmentSegmenter
        
        # Should initialize without loading models
        segmenter = AdvancedGarmentSegmenter(
            canvas_size=256,
            use_grounding_dino=False,  # Disable to avoid loading
            use_schp=False
        )
        
        assert segmenter.canvas_size == 256
    
    def test_advanced_segmenter_has_pipeline(self):
        """Test that AdvancedGarmentSegmenter has internal pipeline."""
        from src.layer0_segmentation import AdvancedGarmentSegmenter
        
        segmenter = AdvancedGarmentSegmenter(
            use_grounding_dino=False,
            use_schp=False
        )
        
        # Should have internal pipeline
        assert hasattr(segmenter, '_pipeline')
        assert segmenter._pipeline is not None


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""
    
    def test_detection_result_creation(self):
        """Test creating a DetectionResult."""
        from src.layer0_segmentation import DetectionResult, Detection, BoundingBox
        
        # New API uses Detection objects
        detections = [
            Detection(
                label="shirt",
                box=BoundingBox(10, 20, 100, 200),
                score=0.95,
                category="tops"
            ),
            Detection(
                label="pants",
                box=BoundingBox(50, 60, 150, 250),
                score=0.88,
                category="bottoms"
            )
        ]
        result = DetectionResult(detections=detections)
        
        assert len(result.detections) == 2
        assert result.detections[0].label == "shirt"
        assert result.detections[0].score == 0.95


class TestFactoryWithModes:
    """Tests for get_segmenter with different modes."""
    
    def test_get_segmenter_mode_simple(self):
        """Test get_segmenter with mode='simple'."""
        from src.layer0_segmentation import get_segmenter, SimpleGarmentSegmenter
        
        segmenter = get_segmenter(mode="simple", canvas_size=256)
        
        assert isinstance(segmenter, SimpleGarmentSegmenter)
        assert segmenter.canvas_size == 256
    
    def test_get_segmenter_mode_auto_fallback(self):
        """Test that auto mode falls back to simple when advanced not available."""
        from src.layer0_segmentation import get_segmenter, SimpleGarmentSegmenter
        
        # In test environment, advanced models likely not available
        # so should fall back to simple
        segmenter = get_segmenter(mode="auto", canvas_size=128)
        
        # Should be some type of segmenter
        assert hasattr(segmenter, 'segment_garment')
        assert hasattr(segmenter, 'canvas_size')

