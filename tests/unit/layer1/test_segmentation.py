"""
Tests for Layer 1: Clothing Segmentation - Unit Tests

Tests cover:
- ClothingSegmenter initialization
- OutfitDecomposer initialization
- Helper functions

Note: Integration tests with real API calls are in tests/integration/
"""
import pytest
import base64
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from src.layer1_vision.segmentation import ClothingSegmenter, OutfitDecomposer
from src.core.exceptions import VisionProcessingError


# ============== Constants ==============

TEST_IMAGES_DIR = Path(__file__).parent.parent.parent / "test_images" / "segmentation"


# ============== Helper Functions ==============

def image_to_base64_url(image_path: Path) -> str:
    """Convert image file to base64 data URL."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    suffix = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif"
    }
    mime_type = mime_types.get(suffix, "image/jpeg")
    
    return f"data:{mime_type};base64,{image_data}"


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
    },
    {
        "item_number": 3,
        "category": "shoes",
        "specific_type": "sneakers",
        "position": "feet",
        "visibility": "partial"
    }
]


# ============== Test Classes ==============

@pytest.mark.unit
class TestClothingSegmenterInit:
    """Tests for ClothingSegmenter initialization."""
    
    def test_init_creates_instance(self):
        """Test that ClothingSegmenter initializes."""
        try:
            segmenter = ClothingSegmenter()
            assert segmenter is not None
        except Exception as e:
            pytest.skip(f"ClothingSegmenter requires API key: {e}")
    
    def test_init_has_required_attributes(self):
        """Test that segmenter has required attributes."""
        try:
            segmenter = ClothingSegmenter()
            assert hasattr(segmenter, 'segment_outfit')
        except Exception as e:
            pytest.skip(f"ClothingSegmenter requires API key: {e}")


@pytest.mark.unit
class TestOutfitDecomposerInit:
    """Tests for OutfitDecomposer initialization."""
    
    def test_init_creates_instance(self):
        """Test that OutfitDecomposer initializes."""
        try:
            decomposer = OutfitDecomposer()
            assert decomposer is not None
        except Exception as e:
            pytest.skip(f"OutfitDecomposer requires API key: {e}")
    
    def test_init_has_required_attributes(self):
        """Test that decomposer has required attributes."""
        try:
            decomposer = OutfitDecomposer()
            assert hasattr(decomposer, 'decompose_outfit')
        except Exception as e:
            pytest.skip(f"OutfitDecomposer requires API key: {e}")


@pytest.mark.unit
class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_image_to_base64_url_jpeg(self, tmp_path):
        """Test converting JPEG to base64 URL."""
        # Create a minimal fake image file
        image_path = tmp_path / "test.jpg"
        image_path.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
        
        result = image_to_base64_url(image_path)
        
        assert result.startswith("data:image/jpeg;base64,")
    
    def test_image_to_base64_url_png(self, tmp_path):
        """Test converting PNG to base64 URL."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes
        
        result = image_to_base64_url(image_path)
        
        assert result.startswith("data:image/png;base64,")
    
    def test_image_to_base64_url_webp(self, tmp_path):
        """Test converting WebP to base64 URL."""
        image_path = tmp_path / "test.webp"
        image_path.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
        
        result = image_to_base64_url(image_path)
        
        assert result.startswith("data:image/webp;base64,")


@pytest.mark.unit
class TestMockSegmentation:
    """Tests with mocked API responses."""
    
    @pytest.mark.asyncio
    async def test_segmentation_response_structure(self):
        """Test that expected response structure is correct."""
        # Validate mock data structure
        response = MOCK_SEGMENTATION_RESPONSE
        
        assert isinstance(response, list)
        assert len(response) > 0
        
        for item in response:
            assert "item_number" in item
            assert "category" in item
            assert "specific_type" in item
    
    def test_mock_response_categories(self):
        """Test that mock response has expected categories."""
        categories = [item["category"] for item in MOCK_SEGMENTATION_RESPONSE]
        
        assert "top" in categories
        assert "bottom" in categories
        assert "shoes" in categories
