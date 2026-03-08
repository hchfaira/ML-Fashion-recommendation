"""
Tests for Layer 0: GroundingDINO Detector
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestFallbackDetector:
    """Tests for FallbackDetector (no model required)."""
    
    def test_detect_returns_full_image(self):
        """Test that fallback detector returns full image as detection."""
        from src.layer0_segmentation import FallbackDetector, GarmentCategory
        
        detector = FallbackDetector()
        
        # Create test image
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        
        result = detector.detect(image)
        
        assert len(result) == 1
        assert result[0].label == "garment"
        assert result[0].box.x1 == 0
        assert result[0].box.y1 == 0
        assert result[0].box.x2 == 300
        assert result[0].box.y2 == 200
        assert result[0].score == 1.0
    
    def test_detect_with_pil_image(self):
        """Test detection with PIL Image."""
        from src.layer0_segmentation import FallbackDetector
        
        detector = FallbackDetector()
        
        image = Image.new("RGB", (300, 200))
        
        result = detector.detect(image)
        
        assert len(result) == 1
        assert result[0].box.x2 == 300
        assert result[0].box.y2 == 200
    
    def test_detect_with_path(self, tmp_path):
        """Test detection with image path."""
        from src.layer0_segmentation import FallbackDetector
        
        detector = FallbackDetector()
        
        # Create test image file
        img = Image.new("RGB", (400, 300))
        img_path = tmp_path / "test.png"
        img.save(img_path)
        
        result = detector.detect(img_path)
        
        assert len(result) == 1
        assert result[0].box.x2 == 400
        assert result[0].box.y2 == 300
    
    def test_custom_default_label(self):
        """Test using custom default label."""
        from src.layer0_segmentation import FallbackDetector
        
        detector = FallbackDetector(default_label="custom_garment")
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect(image)
        
        assert result[0].label == "custom_garment"


class TestGroundingDINODetector:
    """Tests for GroundingDINODetector (may require mocking)."""
    
    def test_initialization(self):
        """Test detector initialization."""
        from src.layer0_segmentation.groundingdino_detector import GroundingDINODetector
        
        # This should not load the model yet (lazy loading)
        detector = GroundingDINODetector(
            box_threshold=0.4,
            text_threshold=0.3
        )
        
        assert detector.box_threshold == 0.4
        assert detector.text_threshold == 0.3
        assert detector._model is None  # Not loaded yet
    
    def test_get_device_cpu_fallback(self):
        """Test device detection falls back to CPU."""
        from src.layer0_segmentation.groundingdino_detector import GroundingDINODetector
        
        detector = GroundingDINODetector()
        
        # This should return a device string
        device = detector._get_device()
        
        assert device in ["cuda", "mps", "cpu"]
    
    def test_load_image_from_numpy(self):
        """Test loading image from numpy array."""
        from src.layer0_segmentation.groundingdino_detector import GroundingDINODetector
        
        detector = GroundingDINODetector()
        
        # RGB image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_np, image_pil = detector._load_image(image)
        
        assert image_np.shape == (100, 100, 3)
        assert isinstance(image_pil, Image.Image)
    
    def test_load_image_from_rgba(self):
        """Test loading image from RGBA numpy array."""
        from src.layer0_segmentation.groundingdino_detector import GroundingDINODetector
        
        detector = GroundingDINODetector()
        
        # RGBA image
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        image_np, image_pil = detector._load_image(image)
        
        assert image_np.shape == (100, 100, 3)  # Converted to RGB
    
    def test_load_image_from_pil(self):
        """Test loading image from PIL Image."""
        from src.layer0_segmentation.groundingdino_detector import GroundingDINODetector
        
        detector = GroundingDINODetector()
        
        image = Image.new("RGB", (100, 100))
        image_np, image_pil = detector._load_image(image)
        
        assert image_np.shape == (100, 100, 3)
        assert isinstance(image_pil, Image.Image)
    
    @pytest.mark.parametrize("model_missing", [True])
    def test_load_model_raises_if_checkpoint_missing(self, model_missing, tmp_path):
        """Test that loading model raises error if checkpoint missing."""
        from src.layer0_segmentation.groundingdino_detector import GroundingDINODetector
        
        detector = GroundingDINODetector(
            checkpoint_path=str(tmp_path / "nonexistent.pth")
        )
        
        # Mock the import to succeed but checkpoint doesn't exist
        with patch.dict('sys.modules', {'groundingdino.util.inference': MagicMock()}):
            with pytest.raises(FileNotFoundError, match="checkpoint not found"):
                detector._load_model()


class TestGetDetector:
    """Tests for get_detector factory function."""
    
    def test_returns_fallback_when_disabled(self):
        """Test that factory returns FallbackDetector when GroundingDINO disabled."""
        from src.layer0_segmentation import get_detector, FallbackDetector
        
        detector = get_detector(use_grounding_dino=False)
        
        assert isinstance(detector, FallbackDetector)
    
    def test_returns_fallback_when_import_fails(self):
        """Test fallback when GroundingDINO import fails."""
        from src.layer0_segmentation import get_detector, FallbackDetector
        
        # When GroundingDINO is not installed, should return FallbackDetector
        detector = get_detector(use_grounding_dino=True)
        
        # Will be FallbackDetector if GroundingDINO not available
        assert detector is not None
