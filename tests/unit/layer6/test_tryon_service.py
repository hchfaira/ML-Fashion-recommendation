"""
Unit Tests for Layer 6: Virtual Try-On Service
===============================================

Tests for the main VirtualTryOnService class.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List
import tempfile
from pathlib import Path

from src.layer6_tryon.tryon_service import VirtualTryOnService, OutfitTuple
from src.layer6_tryon.models import (
    TryOnBackend,
    TryOnConfig,
    GarmentType,
    TryOnResult,
    OutfitTryOnResult,
    TryOnError,
)


# =============================================================================
# Mock Classes
# =============================================================================

@dataclass
class MockColorProfile:
    primary: str = "blue"
    secondary: Optional[str] = None


@dataclass
class MockPatternInfo:
    type: str = "solid"


@dataclass
class MockGarmentAttributes:
    category: str = "top"
    subcategory: str = "t-shirt"
    color: MockColorProfile = None
    pattern: MockPatternInfo = None
    
    def __post_init__(self):
        if self.color is None:
            self.color = MockColorProfile()
        if self.pattern is None:
            self.pattern = MockPatternInfo()


@dataclass
class MockGarment:
    id: str
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    attributes: MockGarmentAttributes = None
    embedding: Optional[List[float]] = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = MockGarmentAttributes()
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dummy_image():
    """Create a dummy PIL Image."""
    return Image.new("RGB", (768, 1024), color="white")


@pytest.fixture
def temp_image_file(dummy_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        dummy_image.save(f.name)
        return f.name


@pytest.fixture
def mock_garment(temp_image_file):
    """Create a mock garment with image path."""
    return MockGarment(
        id="garment_123",
        image_path=temp_image_file,
        attributes=MockGarmentAttributes(
            category="top",
            subcategory="shirt",
            color=MockColorProfile(primary="blue")
        )
    )


@pytest.fixture
def mock_outfit(temp_image_file):
    """Create a mock outfit with multiple garments."""
    return [
        MockGarment(
            id="top_1",
            image_path=temp_image_file,
            attributes=MockGarmentAttributes(category="top")
        ),
        MockGarment(
            id="bottom_1",
            image_path=temp_image_file,
            attributes=MockGarmentAttributes(category="bottom")
        ),
        MockGarment(
            id="shoes_1",
            image_path=temp_image_file,
            attributes=MockGarmentAttributes(category="shoes")
        ),
    ]


@pytest.fixture
def mock_tryon_result(dummy_image):
    """Create a mock TryOnResult."""
    return TryOnResult(
        image=dummy_image,
        garment_id="test_garment",
        garment_type=GarmentType.UPPER_BODY,
        backend=TryOnBackend.CATVTON,
        processing_time_ms=1000.0,
        success=True,
    )


# =============================================================================
# Service Initialization Tests
# =============================================================================

class TestServiceInit:
    """Tests for VirtualTryOnService initialization."""
    
    def test_default_init(self):
        """Test default initialization with CatVTON backend."""
        service = VirtualTryOnService()
        
        assert service.config.backend == TryOnBackend.CATVTON
        assert service._initialized == False
    
    def test_replicate_backend_init(self):
        """Test initialization with Replicate backend."""
        service = VirtualTryOnService(backend=TryOnBackend.REPLICATE)
        
        assert service.config.backend == TryOnBackend.REPLICATE
    
    def test_custom_config_init(self):
        """Test initialization with custom config."""
        config = TryOnConfig(
            backend=TryOnBackend.CATVTON,
            num_inference_steps=50,
            target_width=512,
        )
        service = VirtualTryOnService(config=config)
        
        assert service.config.num_inference_steps == 50
        assert service.config.target_width == 512


# =============================================================================
# Image Loading Tests
# =============================================================================

class TestImageLoading:
    """Tests for image loading utilities."""
    
    def test_load_pil_image(self, dummy_image):
        """Test loading from PIL Image."""
        service = VirtualTryOnService()
        loaded = service._load_image(dummy_image)
        
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
    
    def test_load_from_path(self, temp_image_file):
        """Test loading from file path."""
        service = VirtualTryOnService()
        loaded = service._load_image(temp_image_file)
        
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
    
    def test_load_from_pathlib(self, temp_image_file):
        """Test loading from pathlib.Path."""
        service = VirtualTryOnService()
        loaded = service._load_image(Path(temp_image_file))
        
        assert isinstance(loaded, Image.Image)


# =============================================================================
# Garment Sorting Tests
# =============================================================================

class TestGarmentSorting:
    """Tests for garment sorting order."""
    
    def test_sorting_order(self):
        """Test garments are sorted in correct order for try-on."""
        service = VirtualTryOnService()
        
        garments = [
            MockGarment("top", attributes=MockGarmentAttributes(category="top")),
            MockGarment("dress", attributes=MockGarmentAttributes(category="dress")),
            MockGarment("bottom", attributes=MockGarmentAttributes(category="bottom")),
        ]
        
        sorted_garments = service._sort_garments_for_tryon(garments)
        
        # Lower body (bottom) should come first, then upper body (top), then full body (dress)
        categories = [g.attributes.category for g in sorted_garments]
        assert categories.index("bottom") < categories.index("top")
    
    def test_sorting_with_accessories(self):
        """Test sorting with non-tryonable items."""
        service = VirtualTryOnService()
        
        garments = [
            MockGarment("hat", attributes=MockGarmentAttributes(category="hat")),
            MockGarment("top", attributes=MockGarmentAttributes(category="top")),
            MockGarment("bottom", attributes=MockGarmentAttributes(category="bottom")),
        ]
        
        sorted_garments = service._sort_garments_for_tryon(garments)
        
        # Accessories should come last
        categories = [g.attributes.category for g in sorted_garments]
        assert categories[-1] == "hat"


# =============================================================================
# Try-On Single Garment Tests (Mocked)
# =============================================================================

class TestTryOnGarmentMocked:
    """Tests for single garment try-on with mocked backend."""
    
    def test_try_on_garment_success(self, mock_garment, dummy_image, mock_tryon_result):
        """Test successful single garment try-on."""
        service = VirtualTryOnService()
        
        # Mock the backend
        service._backend = MagicMock()
        service._backend.try_on.return_value = mock_tryon_result
        service._initialized = True
        
        result = service.try_on_garment(dummy_image, mock_garment)
        
        assert result.success == True
        assert result.garment_id == "test_garment"
        service._backend.try_on.assert_called_once()
    
    def test_try_on_garment_with_description(self, mock_garment, dummy_image, mock_tryon_result):
        """Test try-on with custom description."""
        service = VirtualTryOnService()
        
        service._backend = MagicMock()
        service._backend.try_on.return_value = mock_tryon_result
        service._initialized = True
        
        result = service.try_on_garment(
            dummy_image, 
            mock_garment, 
            description="blue silk shirt"
        )
        
        # Check description was passed to backend
        call_kwargs = service._backend.try_on.call_args[1]
        assert call_kwargs.get("garment_description") == "blue silk shirt"
    
    def test_try_on_garment_missing_image(self, dummy_image):
        """Test try-on with garment missing image."""
        service = VirtualTryOnService()
        service._initialized = True
        
        # Garment with no image path
        garment = MockGarment(
            id="no_image",
            image_path=None,
            image_url=None,
        )
        
        result = service.try_on_garment(dummy_image, garment)
        
        assert result.success == False
        assert "Could not load garment image" in result.error_message


# =============================================================================
# Try-On Outfit Tests (Mocked)
# =============================================================================

class TestTryOnOutfitMocked:
    """Tests for outfit try-on with mocked backend."""
    
    def test_try_on_outfit_success(self, mock_outfit, dummy_image):
        """Test successful outfit try-on."""
        service = VirtualTryOnService()
        
        # Mock backend to return success
        mock_result = TryOnResult(
            image=dummy_image,
            garment_id="test",
            garment_type=GarmentType.UPPER_BODY,
            backend=TryOnBackend.CATVTON,
            processing_time_ms=500.0,
            success=True,
        )
        
        service._backend = MagicMock()
        service._backend.try_on.return_value = mock_result
        service._initialized = True
        
        result = service.try_on_outfit(
            person_image=dummy_image,
            outfit_name="Test Outfit",
            garments=mock_outfit,
            outfit_score=0.85,
        )
        
        assert result.outfit_name == "Test Outfit"
        assert result.outfit_score == 0.85
        # Should have tried on 2 garments (top and bottom, shoes are skipped)
        assert len(result.garment_results) == 2
    
    def test_try_on_outfit_skips_non_tryonable(self, dummy_image, temp_image_file):
        """Test that non-tryonable items are skipped."""
        service = VirtualTryOnService()
        
        mock_result = TryOnResult(
            image=dummy_image,
            garment_id="test",
            garment_type=GarmentType.UPPER_BODY,
            backend=TryOnBackend.CATVTON,
            processing_time_ms=500.0,
            success=True,
        )
        
        service._backend = MagicMock()
        service._backend.try_on.return_value = mock_result
        service._initialized = True
        
        # Include non-tryonable items
        outfit = [
            MockGarment("top", image_path=temp_image_file, 
                       attributes=MockGarmentAttributes(category="top")),
            MockGarment("hat", image_path=temp_image_file,
                       attributes=MockGarmentAttributes(category="hat")),
            MockGarment("watch", image_path=temp_image_file,
                       attributes=MockGarmentAttributes(category="watch")),
        ]
        
        result = service.try_on_outfit(
            person_image=dummy_image,
            outfit_name="With Accessories",
            garments=outfit,
            outfit_score=0.75,
        )
        
        # Only top should be tried on
        assert len(result.garment_results) == 1
    
    def test_try_on_outfit_sequential_mode(self, mock_outfit, dummy_image):
        """Test sequential try-on mode."""
        service = VirtualTryOnService()
        
        # Track input images to verify sequential behavior
        input_images = []
        
        def mock_try_on(**kwargs):
            input_images.append(kwargs['person_image'])
            return TryOnResult(
                image=Image.new("RGB", (100, 100), color="green"),  # Different image
                garment_id=kwargs['garment_id'],
                garment_type=kwargs['garment_type'],
                backend=TryOnBackend.CATVTON,
                processing_time_ms=500.0,
                success=True,
            )
        
        service._backend = MagicMock()
        service._backend.try_on.side_effect = mock_try_on
        service._initialized = True
        
        result = service.try_on_outfit(
            person_image=dummy_image,
            outfit_name="Sequential",
            garments=mock_outfit[:2],  # Just top and bottom
            outfit_score=0.8,
            sequential=True,
        )
        
        # Second call should use result from first call
        assert len(input_images) == 2
        # First image should be original
        assert input_images[0].size == dummy_image.size


# =============================================================================
# Try-On Multiple Outfits Tests
# =============================================================================

class TestTryOnMultipleOutfits:
    """Tests for trying on multiple outfits."""
    
    def test_try_on_outfits_returns_all_results(self, dummy_image, temp_image_file):
        """Test that all outfits are processed."""
        service = VirtualTryOnService()
        
        mock_result = TryOnResult(
            image=dummy_image,
            garment_id="test",
            garment_type=GarmentType.UPPER_BODY,
            backend=TryOnBackend.CATVTON,
            processing_time_ms=500.0,
            success=True,
        )
        
        service._backend = MagicMock()
        service._backend.try_on.return_value = mock_result
        service._initialized = True
        
        outfits: List[OutfitTuple] = [
            ("Outfit A", [MockGarment("top_a", image_path=temp_image_file, 
                                      attributes=MockGarmentAttributes(category="top"))], 0.9),
            ("Outfit B", [MockGarment("top_b", image_path=temp_image_file,
                                      attributes=MockGarmentAttributes(category="top"))], 0.8),
            ("Outfit C", [MockGarment("top_c", image_path=temp_image_file,
                                      attributes=MockGarmentAttributes(category="top"))], 0.7),
        ]
        
        results = service.try_on_outfits(dummy_image, outfits)
        
        assert len(results) == 3
        assert results[0].outfit_name == "Outfit A"
        assert results[1].outfit_name == "Outfit B"
        assert results[2].outfit_name == "Outfit C"
    
    def test_try_on_outfits_saves_to_output_dir(self, dummy_image, temp_image_file):
        """Test saving results to output directory."""
        service = VirtualTryOnService()
        
        mock_result = TryOnResult(
            image=dummy_image,
            garment_id="test",
            garment_type=GarmentType.UPPER_BODY,
            backend=TryOnBackend.CATVTON,
            processing_time_ms=500.0,
            success=True,
        )
        
        service._backend = MagicMock()
        service._backend.try_on.return_value = mock_result
        service._initialized = True
        
        outfits: List[OutfitTuple] = [
            ("Test Outfit", [MockGarment("top", image_path=temp_image_file,
                                         attributes=MockGarmentAttributes(category="top"))], 0.85),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = service.try_on_outfits(dummy_image, outfits, output_dir=tmpdir)
            
            # Check that files were saved
            output_path = Path(tmpdir)
            assert any(output_path.iterdir())


# =============================================================================
# Comparison Panel Tests
# =============================================================================

class TestComparisonPanel:
    """Tests for comparison panel generation."""
    
    def test_create_comparison_panel(self, dummy_image):
        """Test creating a comparison panel."""
        service = VirtualTryOnService()
        
        original = Image.new("RGB", (768, 1024), color="white")
        result = Image.new("RGB", (768, 1024), color="blue")
        
        panel = service._create_comparison_panel(
            original=original,
            result=result,
            outfit_name="Test Outfit",
            score=0.85,
        )
        
        assert isinstance(panel, Image.Image)
        # Panel should be wider than single image (side-by-side)
        assert panel.width > 512
        # Panel should have some height for labels
        assert panel.height > 640


# =============================================================================
# Unload Tests
# =============================================================================

class TestServiceUnload:
    """Tests for service unloading."""
    
    def test_unload_calls_backend_unload(self):
        """Test that unload calls backend unload."""
        service = VirtualTryOnService()
        service._backend = MagicMock()
        service._backend.unload = MagicMock()
        service._initialized = True
        
        service.unload()
        
        service._backend.unload.assert_called_once()
        assert service._initialized == False
