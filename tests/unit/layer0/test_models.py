"""
Tests for Layer 0: Data Models
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""
    
    def test_create_bounding_box(self):
        """Test creating a BoundingBox."""
        from src.layer0_segmentation import BoundingBox
        
        box = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        
        assert box.x1 == 10
        assert box.y1 == 20
        assert box.x2 == 100
        assert box.y2 == 150
    
    def test_width_and_height(self):
        """Test width and height properties."""
        from src.layer0_segmentation import BoundingBox
        
        box = BoundingBox(x1=10, y1=20, x2=110, y2=170)
        
        assert box.width == 100
        assert box.height == 150
    
    def test_area(self):
        """Test area property."""
        from src.layer0_segmentation import BoundingBox
        
        box = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        
        assert box.area == 5000
    
    def test_center(self):
        """Test center property."""
        from src.layer0_segmentation import BoundingBox
        
        box = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        
        assert box.center == (50, 50)
    
    def test_to_tuple(self):
        """Test to_tuple method."""
        from src.layer0_segmentation import BoundingBox
        
        box = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        
        assert box.to_tuple() == (10, 20, 100, 150)
    
    def test_to_xywh(self):
        """Test to_xywh method."""
        from src.layer0_segmentation import BoundingBox
        
        box = BoundingBox(x1=10, y1=20, x2=110, y2=170)
        
        assert box.to_xywh() == (10, 20, 100, 150)
    
    def test_from_tuple(self):
        """Test from_tuple class method."""
        from src.layer0_segmentation import BoundingBox
        
        box = BoundingBox.from_tuple((10, 20, 100, 150))
        
        assert box.x1 == 10
        assert box.y1 == 20
        assert box.x2 == 100
        assert box.y2 == 150
    
    def test_intersects_true(self):
        """Test intersects method when boxes overlap."""
        from src.layer0_segmentation import BoundingBox
        
        box1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        box2 = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        
        assert box1.intersects(box2) is True
        assert box2.intersects(box1) is True
    
    def test_intersects_false(self):
        """Test intersects method when boxes don't overlap."""
        from src.layer0_segmentation import BoundingBox
        
        box1 = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        box2 = BoundingBox(x1=100, y1=100, x2=150, y2=150)
        
        assert box1.intersects(box2) is False
    
    def test_iou_calculation(self):
        """Test intersection over union calculation."""
        from src.layer0_segmentation import BoundingBox
        
        box1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        box2 = BoundingBox(x1=0, y1=0, x2=100, y2=100)  # Identical
        
        assert box1.intersection_over_union(box2) == 1.0
        
        # Non-overlapping
        box3 = BoundingBox(x1=200, y1=200, x2=300, y2=300)
        assert box1.intersection_over_union(box3) == 0.0


class TestDetection:
    """Tests for Detection dataclass."""
    
    def test_create_detection(self):
        """Test creating a Detection."""
        from src.layer0_segmentation import Detection, BoundingBox, GarmentCategory
        
        box = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        detection = Detection(
            label="t-shirt",
            box=box,
            score=0.95,
            category=GarmentCategory.TOPS
        )
        
        assert detection.label == "t-shirt"
        assert detection.box == box
        assert detection.score == 0.95
        assert detection.category == GarmentCategory.TOPS
    
    def test_to_dict(self):
        """Test converting detection to dict."""
        from src.layer0_segmentation import Detection, BoundingBox, GarmentCategory
        
        box = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        detection = Detection(
            label="t-shirt",
            box=box,
            score=0.95,
            category=GarmentCategory.TOPS
        )
        
        d = detection.to_dict()
        
        assert d["class"] == "t-shirt"
        assert d["box"] == (10, 20, 100, 150)
        assert d["score"] == 0.95
        assert d["category"] == "tops"


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""
    
    def test_create_empty_result(self):
        """Test creating empty DetectionResult."""
        from src.layer0_segmentation import DetectionResult
        
        result = DetectionResult()
        
        assert len(result) == 0
        assert result.image_size == (0, 0)
    
    def test_create_result_with_detections(self):
        """Test creating DetectionResult with detections."""
        from src.layer0_segmentation import (
            DetectionResult, Detection, BoundingBox, GarmentCategory
        )
        
        detections = [
            Detection(
                label="t-shirt",
                box=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                score=0.9,
                category=GarmentCategory.TOPS
            ),
            Detection(
                label="jeans",
                box=BoundingBox(x1=0, y1=100, x2=100, y2=200),
                score=0.85,
                category=GarmentCategory.BOTTOMS
            )
        ]
        
        result = DetectionResult(detections=detections, image_size=(300, 200))
        
        assert len(result) == 2
        assert result.image_size == (300, 200)
    
    def test_iteration(self):
        """Test iterating over DetectionResult."""
        from src.layer0_segmentation import (
            DetectionResult, Detection, BoundingBox, GarmentCategory
        )
        
        detections = [
            Detection(
                label="t-shirt",
                box=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                score=0.9,
                category=GarmentCategory.TOPS
            )
        ]
        result = DetectionResult(detections=detections)
        
        items = list(result)
        assert len(items) == 1
        assert items[0].label == "t-shirt"
    
    def test_filter_by_score(self):
        """Test filtering detections by score."""
        from src.layer0_segmentation import (
            DetectionResult, Detection, BoundingBox, GarmentCategory
        )
        
        detections = [
            Detection(
                label="t-shirt",
                box=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                score=0.9,
                category=GarmentCategory.TOPS
            ),
            Detection(
                label="jeans",
                box=BoundingBox(x1=0, y1=100, x2=100, y2=200),
                score=0.5,
                category=GarmentCategory.BOTTOMS
            )
        ]
        result = DetectionResult(detections=detections)
        
        filtered = result.filter_by_score(0.7)
        
        assert len(filtered) == 1
        assert filtered[0].label == "t-shirt"
    
    def test_get_best_detection(self):
        """Test getting best detection."""
        from src.layer0_segmentation import (
            DetectionResult, Detection, BoundingBox, GarmentCategory
        )
        
        detections = [
            Detection(
                label="t-shirt",
                box=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                score=0.7,
                category=GarmentCategory.TOPS
            ),
            Detection(
                label="jeans",
                box=BoundingBox(x1=0, y1=100, x2=100, y2=200),
                score=0.9,
                category=GarmentCategory.BOTTOMS
            )
        ]
        result = DetectionResult(detections=detections)
        
        best = result.get_best_detection()
        
        assert best.label == "jeans"
        assert best.score == 0.9


class TestParsingResult:
    """Tests for ParsingResult dataclass."""
    
    def test_create_parsing_result(self):
        """Test creating ParsingResult."""
        from src.layer0_segmentation import ParsingResult
        
        parsing_map = np.zeros((100, 100), dtype=np.uint8)
        parsing_map[:50, :] = 5  # upper_clothes
        parsing_map[50:, :] = 9  # pants
        
        result = ParsingResult(
            parsing_map=parsing_map,
            label_areas={5: 5000, 9: 5000}
        )
        
        assert result.shape == (100, 100)
        assert 5 in result.label_areas
        assert 9 in result.label_areas
    
    def test_get_mask_for_label(self):
        """Test getting mask for specific label."""
        from src.layer0_segmentation import ParsingResult
        
        parsing_map = np.zeros((100, 100), dtype=np.uint8)
        parsing_map[:50, :] = 5  # upper_clothes
        parsing_map[50:, :] = 9  # pants
        
        result = ParsingResult(parsing_map=parsing_map)
        
        mask = result.get_mask_for_label(5)
        
        assert mask.shape == (100, 100)
        assert mask[:50, :].sum() == 50 * 100 * 255
        assert mask[50:, :].sum() == 0
    
    def test_get_present_labels(self):
        """Test getting present labels."""
        from src.layer0_segmentation import ParsingResult
        
        parsing_map = np.zeros((100, 100), dtype=np.uint8)
        parsing_map[:30, :] = 5
        parsing_map[30:60, :] = 9
        parsing_map[60:, :] = 18
        
        result = ParsingResult(parsing_map=parsing_map)
        
        labels = result.get_present_labels()
        
        assert 5 in labels
        assert 9 in labels
        assert 18 in labels


class TestRefinedMask:
    """Tests for RefinedMask dataclass."""
    
    def test_create_refined_mask(self):
        """Test creating RefinedMask."""
        from src.layer0_segmentation import RefinedMask, BoundingBox
        
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        box = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        
        refined = RefinedMask(mask=mask, score=0.95, box=box)
        
        assert refined.shape == (100, 100)
        assert refined.score == 0.95
        assert refined.area == 10000
    
    def test_to_binary(self):
        """Test converting to binary mask."""
        from src.layer0_segmentation import RefinedMask
        
        mask = np.array([[0, 128, 200], [50, 255, 127]], dtype=np.uint8)
        refined = RefinedMask(mask=mask)
        
        binary = refined.to_binary()
        
        assert binary[0, 0] == 0  # 0 < 127
        assert binary[0, 1] == 255  # 128 > 127
        assert binary[0, 2] == 255  # 200 > 127
        assert binary[1, 0] == 0  # 50 < 127
        assert binary[1, 1] == 255  # 255 > 127
        assert binary[1, 2] == 0  # 127 == 127 (not > 127)


class TestExtractedGarment:
    """Tests for ExtractedGarment dataclass."""
    
    def test_create_extracted_garment(self):
        """Test creating ExtractedGarment."""
        from src.layer0_segmentation import ExtractedGarment, GarmentCategory
        
        image = Image.new("RGBA", (256, 256), (255, 255, 255, 0))
        
        garment = ExtractedGarment(
            image=image,
            category=GarmentCategory.TOPS,
            label="t-shirt",
            confidence=0.95
        )
        
        assert garment.image == image
        assert garment.category == GarmentCategory.TOPS
        assert garment.label == "t-shirt"
        assert garment.confidence == 0.95
    
    def test_get_filename(self):
        """Test generating filename."""
        from src.layer0_segmentation import ExtractedGarment, GarmentCategory
        
        image = Image.new("RGBA", (256, 256))
        
        garment = ExtractedGarment(
            image=image,
            category=GarmentCategory.TOPS,
            label="t-shirt",
            confidence=0.95
        )
        
        filename = garment.get_filename(index=1)
        
        assert filename == "tops_001_t_shirt.png"
    
    def test_to_dict(self):
        """Test converting to dict."""
        from src.layer0_segmentation import ExtractedGarment, GarmentCategory
        
        image = Image.new("RGBA", (256, 256))
        
        garment = ExtractedGarment(
            image=image,
            category=GarmentCategory.BOTTOMS,
            label="jeans",
            confidence=0.9,
            area=5000
        )
        
        d = garment.to_dict()
        
        assert d["category"] == "bottoms"
        assert d["label"] == "jeans"
        assert d["confidence"] == 0.9
        assert d["area"] == 5000


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""
    
    def test_create_empty_result(self):
        """Test creating empty ExtractionResult."""
        from src.layer0_segmentation import ExtractionResult
        
        result = ExtractionResult()
        
        assert len(result) == 0
        assert result.processing_time_ms == 0.0
    
    def test_filter_by_category(self):
        """Test filtering by category."""
        from src.layer0_segmentation import (
            ExtractionResult, ExtractedGarment, GarmentCategory
        )
        
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
        
        tops = result.filter_by_category(GarmentCategory.TOPS)
        
        assert len(tops) == 1
        assert tops[0].label == "t-shirt"
    
    def test_to_summary(self):
        """Test getting summary."""
        from src.layer0_segmentation import (
            ExtractionResult, ExtractedGarment, GarmentCategory
        )
        
        garments = [
            ExtractedGarment(
                image=Image.new("RGBA", (100, 100)),
                category=GarmentCategory.TOPS,
                label="t-shirt",
                confidence=0.9
            ),
            ExtractedGarment(
                image=Image.new("RGBA", (100, 100)),
                category=GarmentCategory.TOPS,
                label="sweater",
                confidence=0.8
            ),
            ExtractedGarment(
                image=Image.new("RGBA", (100, 100)),
                category=GarmentCategory.BOTTOMS,
                label="jeans",
                confidence=0.85
            )
        ]
        
        result = ExtractionResult(
            garments=garments,
            processing_time_ms=150.5
        )
        
        summary = result.to_summary()
        
        assert summary["total_garments"] == 3
        assert summary["categories"]["tops"] == 2
        assert summary["categories"]["bottoms"] == 1
        assert summary["processing_time_ms"] == 150.5
