"""
Integration tests for Vision Services (Layer 1).

These tests require API keys and are marked as slow/integration.
"""
import pytest
import json
import base64
from pathlib import Path
from datetime import datetime

from src.layer1_vision.attribute_extractor import (
    AttributeExtractor, 
    ExtractionMode
)
from src.layer1_vision.segmentation import ClothingSegmenter, OutfitDecomposer
from src.layer1_vision.embedding_generator import EmbeddingGenerator
from src.core.models import GarmentCategory
from src.core.exceptions import VisionProcessingError


# ============== Configuration ==============

TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images" / "segmentation"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "integration"


# ============== Helper Functions ==============

def get_test_images():
    """Get all test images from test_images folder."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    images = []
    
    if TEST_IMAGES_DIR.exists():
        for file in TEST_IMAGES_DIR.iterdir():
            if file.suffix.lower() in image_extensions:
                images.append(file)
    
    return images


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


# ============== Fixtures ==============

@pytest.fixture
def segmenter():
    """Create ClothingSegmenter instance."""
    try:
        return ClothingSegmenter()
    except Exception as e:
        pytest.skip(f"ClothingSegmenter requires API key: {e}")


@pytest.fixture
def decomposer():
    """Create OutfitDecomposer instance."""
    try:
        return OutfitDecomposer()
    except Exception as e:
        pytest.skip(f"OutfitDecomposer requires API key: {e}")


@pytest.fixture
def extractor():
    """Create AttributeExtractor instance."""
    try:
        from config import get_settings
        settings = get_settings()
        if not settings.gemini_api_key:
            pytest.skip("GEMINI_API_KEY not configured")
        return AttributeExtractor()
    except Exception as e:
        pytest.skip(f"AttributeExtractor requires API key: {e}")


@pytest.fixture
def embedding_generator():
    """Create EmbeddingGenerator instance."""
    try:
        return EmbeddingGenerator()
    except Exception as e:
        pytest.skip(f"EmbeddingGenerator requires API key: {e}")


@pytest.fixture
def test_images():
    """Get all test images."""
    images = get_test_images()
    if not images:
        pytest.skip(f"No test images found in {TEST_IMAGES_DIR}")
    return images


# ============== Test Classes ==============

@pytest.mark.integration
@pytest.mark.slow
class TestClothingSegmentation:
    """Integration tests for ClothingSegmenter."""
    
    @pytest.mark.asyncio
    async def test_segment_outfit_basic(self, segmenter, test_images):
        """Test basic outfit segmentation."""
        image_path = test_images[0]
        image_url = image_to_base64_url(image_path)
        
        items = await segmenter.segment_outfit(image_url)
        
        assert isinstance(items, list)
        assert len(items) > 0
    
    @pytest.mark.asyncio
    async def test_segment_identifies_categories(self, segmenter, test_images):
        """Test that segmentation identifies clothing categories."""
        image_path = test_images[0]
        image_url = image_to_base64_url(image_path)
        
        items = await segmenter.segment_outfit(image_url)
        
        for item in items:
            assert "category" in item
            assert item["category"] in ["top", "bottom", "shoes", "outerwear", "dress", "accessory", "bag"]


@pytest.mark.integration
@pytest.mark.slow
class TestOutfitDecomposition:
    """Integration tests for OutfitDecomposer."""
    
    @pytest.mark.asyncio
    async def test_decompose_outfit_basic(self, decomposer, test_images):
        """Test basic outfit decomposition."""
        image_path = test_images[0]
        image_url = image_to_base64_url(image_path)
        
        result = await decomposer.decompose_outfit(image_url)
        
        assert "items" in result or "item_count" in result
    
    @pytest.mark.asyncio
    async def test_decompose_provides_metadata(self, decomposer, test_images):
        """Test that decomposition provides metadata."""
        image_path = test_images[0]
        image_url = image_to_base64_url(image_path)
        
        result = await decomposer.decompose_outfit(image_url)
        
        assert "item_count" in result
        assert "categories_present" in result


@pytest.mark.integration
@pytest.mark.slow
class TestAttributeExtraction:
    """Integration tests for AttributeExtractor."""
    
    @pytest.mark.asyncio
    async def test_extract_full_outfit(self, extractor, test_images):
        """Test full outfit attribute extraction."""
        image_path = test_images[0]
        image_url = image_to_base64_url(image_path)
        
        result = await extractor.extract_attributes(image_url)
        
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestEmbeddingGeneration:
    """Integration tests for EmbeddingGenerator."""
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, embedding_generator, garment_factory):
        """Test generating embedding for a garment."""
        garment = garment_factory.create_basic_top(color="white")
        
        embedding = await embedding_generator.generate_embedding(garment)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
    
    @pytest.mark.asyncio
    async def test_embedding_dimension(self, embedding_generator, garment_factory):
        """Test embedding has expected dimension."""
        garment = garment_factory.create_basic_top(color="blue")
        
        embedding = await embedding_generator.generate_embedding(garment)
        
        # Gemini embedding dimension is typically 3072
        assert len(embedding) == 3072
