"""
Tests for Layer 1: Attribute Extractor - Unit Tests

Tests cover:
- ExtractionMode enum
- AttributeExtractor initialization
- Mode configuration
- Error handling

Note: Integration tests with real API calls are in tests/integration/
"""
import pytest
import json
from unittest.mock import MagicMock, patch

from src.layer1_vision.attribute_extractor import (
    AttributeExtractor, 
    ExtractionMode, 
    DEFAULT_EXTRACTION_MODE
)
from src.core.models import GarmentCategory
from src.core.exceptions import VisionProcessingError


# ============== Mock Data ==============

MOCK_SEGMENTATION_RESPONSE = [
    {
        "item_number": 1,
        "category": "top",
        "specific_type": "t-shirt",
        "position": "upper_body",
        "visibility": "full"
    },
    {
        "item_number": 2,
        "category": "bottom",
        "specific_type": "jeans",
        "position": "lower_body",
        "visibility": "full"
    }
]

MOCK_TOP_ATTRIBUTES = {
    "taxonomy": {
        "category": "top",
        "subcategory": "t-shirt",
        "product_type": "Basic T-Shirt",
        "outfit_role": "base_layer"
    },
    "color_profile": {
        "primary_color": "white",
        "color_temperature": "neutral",
        "color_depth": "light"
    },
    "material_profile": {
        "material": "cotton",
        "fabric_weight": "light"
    },
    "occasion_profile": {
        "formality_level": "casual"
    },
    "confidence_score": 0.9
}


# ============== Fixtures ==============

@pytest.fixture
def mock_gemini():
    """Mock the Gemini API calls."""
    with patch('src.layer1_vision.attribute_extractor.genai') as mock_genai:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        yield mock_genai, mock_model


# ============== Test Classes ==============

@pytest.mark.unit
class TestExtractionModeEnum:
    """Tests for ExtractionMode enum and defaults."""
    
    def test_extraction_mode_full_outfit_value(self):
        """Test that FULL_OUTFIT has expected value."""
        assert ExtractionMode.FULL_OUTFIT.value == "full_outfit"
    
    def test_extraction_mode_specific_garment_value(self):
        """Test that SPECIFIC_GARMENT has expected value."""
        assert ExtractionMode.SPECIFIC_GARMENT.value == "specific_garment"
    
    def test_default_extraction_mode_is_full_outfit(self):
        """Test that default extraction mode is FULL_OUTFIT."""
        assert DEFAULT_EXTRACTION_MODE == ExtractionMode.FULL_OUTFIT
    
    def test_extraction_mode_string_representation(self):
        """Test that ExtractionMode has correct string representation."""
        assert "FULL_OUTFIT" in str(ExtractionMode.FULL_OUTFIT)


@pytest.mark.unit
class TestAttributeExtractorInit:
    """Tests for AttributeExtractor initialization."""
    
    def test_default_initialization(self, mock_gemini):
        """Test default initialization uses FULL_OUTFIT mode."""
        extractor = AttributeExtractor()
        
        assert extractor.extraction_mode == ExtractionMode.FULL_OUTFIT
        assert extractor.target_category is None
        assert extractor.target_categories is None
    
    def test_explicit_full_outfit_mode(self, mock_gemini):
        """Test explicit FULL_OUTFIT mode initialization."""
        extractor = AttributeExtractor(extraction_mode=ExtractionMode.FULL_OUTFIT)
        
        assert extractor.extraction_mode == ExtractionMode.FULL_OUTFIT
        assert extractor.target_category is None
    
    def test_specific_garment_mode_single_category(self, mock_gemini):
        """Test SPECIFIC_GARMENT mode with single target_category."""
        extractor = AttributeExtractor(
            extraction_mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=GarmentCategory.BOTTOM
        )
        
        assert extractor.extraction_mode == ExtractionMode.SPECIFIC_GARMENT
        assert extractor.target_category == GarmentCategory.BOTTOM
        assert extractor.target_categories == [GarmentCategory.BOTTOM]
    
    def test_specific_garment_mode_multiple_categories(self, mock_gemini):
        """Test SPECIFIC_GARMENT mode with multiple target_categories."""
        categories = [GarmentCategory.TOP, GarmentCategory.BOTTOM]
        extractor = AttributeExtractor(
            extraction_mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=categories
        )
        
        assert extractor.extraction_mode == ExtractionMode.SPECIFIC_GARMENT
        assert extractor.target_categories == categories
        assert extractor.target_category == GarmentCategory.TOP
    
    def test_specific_garment_without_target_raises_error(self, mock_gemini):
        """Test that SPECIFIC_GARMENT mode without target raises error."""
        with pytest.raises(ValueError) as exc_info:
            AttributeExtractor(extraction_mode=ExtractionMode.SPECIFIC_GARMENT)
        
        assert "target_categories must be specified" in str(exc_info.value)


@pytest.mark.unit
class TestTargetCategoryValidation:
    """Tests for target category validation."""
    
    def test_all_garment_categories_valid(self, mock_gemini):
        """Test initialization with each garment category."""
        categories = [
            GarmentCategory.TOP,
            GarmentCategory.BOTTOM,
            GarmentCategory.DRESS,
            GarmentCategory.OUTERWEAR,
            GarmentCategory.SHOES,
            GarmentCategory.ACCESSORY,
            GarmentCategory.BAG
        ]
        
        for category in categories:
            extractor = AttributeExtractor(
                extraction_mode=ExtractionMode.SPECIFIC_GARMENT,
                target_categories=category
            )
            assert extractor.target_category == category
    
    def test_multiple_categories_list(self, mock_gemini):
        """Test initialization with multiple categories list."""
        categories = [
            GarmentCategory.TOP,
            GarmentCategory.BOTTOM,
            GarmentCategory.SHOES
        ]
        extractor = AttributeExtractor(
            extraction_mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=categories
        )
        
        assert extractor.target_categories == categories
        assert len(extractor.target_categories) == 3


@pytest.mark.unit
class TestMockedExtraction:
    """Tests using mocked API responses."""
    
    @pytest.mark.asyncio
    async def test_extract_with_mode_override(self, mock_gemini):
        """Test mode override at extraction time."""
        mock_genai, mock_model = mock_gemini
        
        # Setup response
        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_TOP_ATTRIBUTES)
        mock_model.generate_content.return_value = mock_response
        
        # Create extractor with SPECIFIC_GARMENT mode
        extractor = AttributeExtractor(
            extraction_mode=ExtractionMode.SPECIFIC_GARMENT,
            target_categories=GarmentCategory.TOP
        )
        
        assert extractor.extraction_mode == ExtractionMode.SPECIFIC_GARMENT
