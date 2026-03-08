"""
Tests for Layer 0: SCHP Human Parser
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path


class TestFallbackParser:
    """Tests for FallbackParser (no model required)."""
    
    def test_parse_returns_parsing_result(self):
        """Test that fallback parser returns ParsingResult."""
        from src.layer0_segmentation import FallbackParser, ParsingResult
        
        parser = FallbackParser()
        
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        
        result = parser.parse(image)
        
        assert isinstance(result, ParsingResult)
        assert result.shape == (200, 100)
    
    def test_parse_creates_vertical_regions(self):
        """Test that parser creates vertical region segmentation."""
        from src.layer0_segmentation import FallbackParser
        
        parser = FallbackParser()
        
        image = np.zeros((100, 50, 3), dtype=np.uint8)
        
        result = parser.parse(image)
        
        # Check that different regions have different labels
        # Upper region should have label 5 (upper_clothes)
        assert result.parsing_map[10, 25] == 5
        # Lower region should have label 9 (pants)
        assert result.parsing_map[70, 25] == 9
        # Bottom should have label 18 (shoes)
        assert result.parsing_map[90, 25] == 18
    
    def test_parse_with_pil_image(self):
        """Test parsing with PIL Image."""
        from src.layer0_segmentation import FallbackParser
        
        parser = FallbackParser()
        
        image = Image.new("RGB", (100, 200))
        
        result = parser.parse(image)
        
        assert result.shape == (200, 100)
    
    def test_parse_with_path(self, tmp_path):
        """Test parsing with image path."""
        from src.layer0_segmentation import FallbackParser
        
        parser = FallbackParser()
        
        img = Image.new("RGB", (100, 200))
        img_path = tmp_path / "test.png"
        img.save(img_path)
        
        result = parser.parse(img_path)
        
        assert result.shape == (200, 100)
    
    def test_label_areas_populated(self):
        """Test that label areas are populated."""
        from src.layer0_segmentation import FallbackParser
        
        parser = FallbackParser()
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = parser.parse(image)
        
        # Should have areas for the labels
        assert len(result.label_areas) > 0


class TestSCHPParser:
    """Tests for SCHPParser (model-based, uses fallback if not available)."""
    
    def test_initialization(self):
        """Test parser initialization."""
        from src.layer0_segmentation.schp_parser import SCHPParser
        
        parser = SCHPParser()
        
        assert parser._model is None  # Not loaded yet
        assert parser.input_size == (473, 473)
    
    def test_get_device(self):
        """Test device detection."""
        from src.layer0_segmentation.schp_parser import SCHPParser
        
        parser = SCHPParser()
        
        device = parser._get_device()
        
        assert device in ["cuda", "mps", "cpu"]
    
    def test_heuristic_parsing(self):
        """Test heuristic-based parsing fallback."""
        from src.layer0_segmentation.schp_parser import SCHPParser
        
        parser = SCHPParser()
        
        image = np.zeros((300, 200, 3), dtype=np.uint8)
        
        result = parser._heuristic_parsing(image)
        
        assert result.shape == (300, 200)
        # Check regions are assigned
        assert 5 in result.label_areas  # upper_clothes
        assert 9 in result.label_areas  # pants
    
    def test_parse_falls_back_to_heuristic(self):
        """Test that parse falls back to heuristic when model unavailable."""
        from src.layer0_segmentation.schp_parser import SCHPParser
        
        parser = SCHPParser()
        
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        
        # Should work even without model (uses heuristic)
        result = parser.parse(image)
        
        assert result.shape == (200, 100)
    
    def test_get_garment_masks(self):
        """Test extracting garment masks from parsing result."""
        from src.layer0_segmentation.schp_parser import SCHPParser, GarmentCategory
        from src.layer0_segmentation import ParsingResult
        
        parser = SCHPParser()
        
        # Create a parsing map with specific labels
        parsing_map = np.zeros((100, 100), dtype=np.uint8)
        parsing_map[:50, :] = 5  # upper_clothes -> TOPS
        parsing_map[50:, :] = 9  # pants -> BOTTOMS
        
        parsing_result = ParsingResult(
            parsing_map=parsing_map,
            label_areas={5: 5000, 9: 5000}
        )
        
        masks = parser.get_garment_masks(parsing_result)
        
        assert GarmentCategory.TOPS in masks
        assert GarmentCategory.BOTTOMS in masks
        assert masks[GarmentCategory.TOPS][:50, :].sum() > 0
        assert masks[GarmentCategory.BOTTOMS][50:, :].sum() > 0


class TestGetParser:
    """Tests for get_parser factory function."""
    
    def test_returns_fallback_when_disabled(self):
        """Test that factory returns FallbackParser when SCHP disabled."""
        from src.layer0_segmentation import get_parser, FallbackParser
        
        parser = get_parser(use_schp=False)
        
        assert isinstance(parser, FallbackParser)
    
    def test_returns_parser_when_enabled(self):
        """Test that factory returns a parser when enabled."""
        from src.layer0_segmentation import get_parser, FallbackParser, SCHPParser
        
        parser = get_parser(use_schp=True)
        
        # Will be SCHPParser if PyTorch available, FallbackParser otherwise
        assert isinstance(parser, (SCHPParser, FallbackParser))
