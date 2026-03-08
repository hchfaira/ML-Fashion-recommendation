"""
Tests for Layer 0: Garment Extraction Pipeline
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""
    
    def test_default_config(self):
        """Test default pipeline configuration."""
        from src.layer0_segmentation import PipelineConfig
        
        config = PipelineConfig()
        
        assert config.use_grounding_dino is True
        assert config.use_schp is True
        assert config.use_sam is True
        assert config.canvas_size == 512
        assert config.box_threshold == 0.3
        assert config.text_threshold == 0.25
    
    def test_custom_config(self):
        """Test custom pipeline configuration."""
        from src.layer0_segmentation import PipelineConfig
        
        config = PipelineConfig(
            use_grounding_dino=False,
            use_schp=False,
            use_sam=True,
            canvas_size=256,
            box_threshold=0.4
        )
        
        assert config.use_grounding_dino is False
        assert config.canvas_size == 256
    
    def test_postprocess_config_auto_created(self):
        """Test that postprocess config is auto-created."""
        from src.layer0_segmentation import PipelineConfig, PostprocessConfig
        
        config = PipelineConfig()
        
        assert config.postprocess_config is not None
        assert isinstance(config.postprocess_config, PostprocessConfig)


class TestGarmentExtractionPipeline:
    """Tests for GarmentExtractionPipeline class."""
    
    def test_initialization_default(self):
        """Test default pipeline initialization."""
        from src.layer0_segmentation import GarmentExtractionPipeline
        
        pipeline = GarmentExtractionPipeline()
        
        assert pipeline.config is not None
        assert pipeline._detector is None  # Lazy loaded
        assert pipeline._parser is None
        assert pipeline._refiner is None
    
    def test_initialization_with_config(self):
        """Test pipeline initialization with custom config."""
        from src.layer0_segmentation import (
            GarmentExtractionPipeline, PipelineConfig
        )
        
        config = PipelineConfig(
            use_grounding_dino=False,
            use_schp=False,
            use_sam=False,
            canvas_size=256
        )
        
        pipeline = GarmentExtractionPipeline(config)
        
        assert pipeline.config.canvas_size == 256
        assert pipeline.config.use_grounding_dino is False
    
    def test_initialization_with_injected_components(self):
        """Test pipeline with injected components."""
        from src.layer0_segmentation import (
            GarmentExtractionPipeline, FallbackDetector,
            FallbackParser, FallbackRefiner
        )
        
        detector = FallbackDetector()
        parser = FallbackParser()
        refiner = FallbackRefiner()
        
        pipeline = GarmentExtractionPipeline(
            detector=detector,
            parser=parser,
            refiner=refiner
        )
        
        assert pipeline._detector is detector
        assert pipeline._parser is parser
        assert pipeline._refiner is refiner
    
    def test_detector_property_lazy_loads(self):
        """Test that detector property triggers lazy loading."""
        from src.layer0_segmentation import (
            GarmentExtractionPipeline, PipelineConfig, BaseDetector
        )
        
        config = PipelineConfig(use_grounding_dino=False)
        pipeline = GarmentExtractionPipeline(config)
        
        detector = pipeline.detector
        
        assert detector is not None
        assert isinstance(detector, BaseDetector)
    
    def test_parser_property_lazy_loads(self):
        """Test that parser property triggers lazy loading."""
        from src.layer0_segmentation import (
            GarmentExtractionPipeline, PipelineConfig, BaseHumanParser
        )
        
        config = PipelineConfig(use_schp=False)
        pipeline = GarmentExtractionPipeline(config)
        
        parser = pipeline.parser
        
        assert parser is not None
        assert isinstance(parser, BaseHumanParser)
    
    def test_refiner_property_lazy_loads(self):
        """Test that refiner property triggers lazy loading."""
        from src.layer0_segmentation import (
            GarmentExtractionPipeline, PipelineConfig, BaseMaskRefiner
        )
        
        config = PipelineConfig(use_sam=False)
        pipeline = GarmentExtractionPipeline(config)
        
        refiner = pipeline.refiner
        
        assert refiner is not None
        assert isinstance(refiner, BaseMaskRefiner)
    
    def test_load_image_rgb(self):
        """Test loading RGB image."""
        from src.layer0_segmentation import GarmentExtractionPipeline
        
        pipeline = GarmentExtractionPipeline()
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline._load_image(image)
        
        assert result.shape == (100, 100, 3)
    
    def test_load_image_rgba(self):
        """Test loading RGBA image."""
        from src.layer0_segmentation import GarmentExtractionPipeline
        
        pipeline = GarmentExtractionPipeline()
        
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        result = pipeline._load_image(image)
        
        assert result.shape == (100, 100, 3)
    
    def test_load_image_grayscale(self):
        """Test loading grayscale image."""
        from src.layer0_segmentation import GarmentExtractionPipeline
        
        pipeline = GarmentExtractionPipeline()
        
        image = np.zeros((100, 100), dtype=np.uint8)
        result = pipeline._load_image(image)
        
        assert result.shape == (100, 100, 3)
    
    def test_create_box_mask(self):
        """Test creating box mask."""
        from src.layer0_segmentation import (
            GarmentExtractionPipeline, BoundingBox
        )
        
        pipeline = GarmentExtractionPipeline()
        
        box = BoundingBox(x1=10, y1=20, x2=50, y2=80)
        mask = pipeline._create_box_mask((100, 100), box)
        
        assert mask.shape == (100, 100)
        assert mask[50, 30] == 255  # Inside box
        assert mask[5, 5] == 0  # Outside box
    
    def test_process_simple_mode(self, tmp_path):
        """Test processing in simple mode (no ML models)."""
        from src.layer0_segmentation import (
            GarmentExtractionPipeline, PipelineConfig
        )
        
        config = PipelineConfig(
            use_grounding_dino=False,
            use_schp=False,
            use_sam=False,
            canvas_size=128
        )
        pipeline = GarmentExtractionPipeline(config)
        
        # Create test image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        result = pipeline.process(image)
        
        assert result is not None
        assert len(result) >= 0  # May be 0 or more depending on fallback
        assert result.processing_time_ms > 0
    
    def test_process_with_path(self, tmp_path):
        """Test processing with image path."""
        from src.layer0_segmentation import (
            GarmentExtractionPipeline, PipelineConfig
        )
        
        config = PipelineConfig(
            use_grounding_dino=False,
            use_schp=False,
            use_sam=False
        )
        pipeline = GarmentExtractionPipeline(config)
        
        # Create test image file
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        img_path = tmp_path / "test.png"
        img.save(img_path)
        
        result = pipeline.process(img_path)
        
        assert result is not None
        assert result.source_image_path == img_path
    
    def test_process_with_save_output(self, tmp_path):
        """Test processing with output saving."""
        from src.layer0_segmentation import (
            GarmentExtractionPipeline, PipelineConfig, 
            FallbackDetector, FallbackRefiner,
            BoundingBox, DetectionResult, Detection, GarmentCategory
        )
        
        # Create a mock detector that returns detections
        class MockDetector(FallbackDetector):
            def detect(self, image, prompt=None, categories=None):
                return DetectionResult(
                    detections=[
                        Detection(
                            label="t-shirt",
                            box=BoundingBox(x1=10, y1=10, x2=190, y2=190),
                            score=0.9,
                            category=GarmentCategory.TOPS
                        )
                    ],
                    image_size=(200, 200)
                )
        
        config = PipelineConfig(
            use_grounding_dino=False,
            use_schp=False,
            use_sam=False
        )
        pipeline = GarmentExtractionPipeline(
            config,
            detector=MockDetector()
        )
        
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        output_dir = tmp_path / "garments"
        
        result = pipeline.process(
            image,
            save_output=True,
            output_dir=output_dir
        )
        
        assert len(result) >= 1
        assert output_dir.exists()


class TestCreatePipeline:
    """Tests for create_pipeline factory function."""
    
    def test_create_full_pipeline(self):
        """Test creating full pipeline."""
        from src.layer0_segmentation import create_pipeline
        
        pipeline = create_pipeline(mode="full", canvas_size=256)
        
        assert pipeline.config.use_grounding_dino is True
        assert pipeline.config.use_schp is True
        assert pipeline.config.use_sam is True
        assert pipeline.config.canvas_size == 256
    
    def test_create_fast_pipeline(self):
        """Test creating fast pipeline."""
        from src.layer0_segmentation import create_pipeline
        
        pipeline = create_pipeline(mode="fast")
        
        assert pipeline.config.use_grounding_dino is False
        assert pipeline.config.use_schp is False
        assert pipeline.config.use_sam is True
    
    def test_create_simple_pipeline(self):
        """Test creating simple pipeline."""
        from src.layer0_segmentation import create_pipeline
        
        pipeline = create_pipeline(mode="simple")
        
        assert pipeline.config.use_grounding_dino is False
        assert pipeline.config.use_schp is False
        assert pipeline.config.use_sam is False
    
    def test_create_auto_pipeline(self):
        """Test creating auto pipeline."""
        from src.layer0_segmentation import create_pipeline
        
        pipeline = create_pipeline(mode="auto")
        
        # Should return a valid pipeline
        assert pipeline is not None


class TestExtractGarmentsFunction:
    """Tests for extract_garments convenience function."""
    
    def test_extract_garments(self):
        """Test extract_garments function."""
        from src.layer0_segmentation import extract_garments
        
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        result = extract_garments(
            image=image,
            canvas_size=128,
            mode="simple"
        )
        
        assert result is not None
    
    def test_extract_garments_with_output(self, tmp_path):
        """Test extract_garments with output directory."""
        from src.layer0_segmentation import extract_garments
        
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        output_dir = tmp_path / "output"
        
        result = extract_garments(
            image=image,
            output_dir=output_dir,
            mode="simple"
        )
        
        assert result is not None


class TestLegacyCompatibility:
    """Tests for legacy API compatibility."""
    
    def test_simple_garment_segmenter(self, tmp_path):
        """Test legacy SimpleGarmentSegmenter."""
        from src.layer0_segmentation import SimpleGarmentSegmenter, SegmentedGarment
        
        segmenter = SimpleGarmentSegmenter(canvas_size=128)
        
        # Create test image
        img = Image.new("RGB", (100, 100), color=(128, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(img_path)
        
        result = segmenter.segment_garment(img_path)
        
        assert isinstance(result, SegmentedGarment)
        assert result.image.size == (128, 128)
    
    def test_garment_type_enum(self):
        """Test legacy GarmentType enum."""
        from src.layer0_segmentation import GarmentType
        
        assert GarmentType.TOP.value == "tops"
        assert GarmentType.BOTTOM.value == "bottoms"
        assert GarmentType.DRESS.value == "full_body"
        assert GarmentType.UNKNOWN.value == "unknown"
    
    def test_segmented_garment_dataclass(self):
        """Test legacy SegmentedGarment dataclass."""
        from src.layer0_segmentation import SegmentedGarment, GarmentType
        
        img = Image.new("RGBA", (100, 100))
        
        garment = SegmentedGarment(
            image=img,
            garment_type=GarmentType.TOP,
            label="t-shirt",
            confidence=0.9
        )
        
        assert garment.image == img
        assert garment.garment_type == GarmentType.TOP
    
    def test_get_segmenter_simple(self):
        """Test legacy get_segmenter function."""
        from src.layer0_segmentation import get_segmenter, SimpleGarmentSegmenter
        
        segmenter = get_segmenter(use_sam=False)
        
        assert isinstance(segmenter, SimpleGarmentSegmenter)
    
    def test_default_canvas_size(self):
        """Test DEFAULT_CANVAS_SIZE constant."""
        from src.layer0_segmentation import DEFAULT_CANVAS_SIZE
        
        assert DEFAULT_CANVAS_SIZE == 512
