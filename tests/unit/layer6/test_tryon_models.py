"""
Unit Tests for Layer 6: Virtual Try-On Models
==============================================

Tests for data models, enums, and helper functions.
"""

import pytest
from PIL import Image
from dataclasses import dataclass

from src.layer6_tryon.models import (
    TryOnBackend,
    TryOnConfig,
    GarmentType,
    TryOnResult,
    OutfitTryOnResult,
    TryOnError,
    category_to_garment_type,
    is_tryonable,
)


# =============================================================================
# TryOnBackend Tests
# =============================================================================

class TestTryOnBackend:
    """Tests for TryOnBackend enum."""
    
    def test_backend_values(self):
        """Test backend enum values."""
        assert TryOnBackend.CATVTON.value == "catvton"
        assert TryOnBackend.IDMVTON.value == "idmvton"
        assert TryOnBackend.REPLICATE.value == "replicate"
    
    def test_backend_from_string(self):
        """Test creating backend from string."""
        assert TryOnBackend("catvton") == TryOnBackend.CATVTON
        assert TryOnBackend("replicate") == TryOnBackend.REPLICATE


# =============================================================================
# GarmentType Tests
# =============================================================================

class TestGarmentType:
    """Tests for GarmentType enum."""
    
    def test_garment_type_values(self):
        """Test garment type enum values."""
        assert GarmentType.UPPER_BODY.value == "upper_body"
        assert GarmentType.LOWER_BODY.value == "lower_body"
        assert GarmentType.FULL_BODY.value == "dresses"
        assert GarmentType.FOOTWEAR.value == "footwear"
        assert GarmentType.ACCESSORY.value == "accessory"


# =============================================================================
# TryOnConfig Tests
# =============================================================================

class TestTryOnConfig:
    """Tests for TryOnConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TryOnConfig()
        
        assert config.backend == TryOnBackend.CATVTON
        assert config.device == "cuda"
        assert config.dtype == "float16"
        assert config.target_width == 768
        assert config.target_height == 1024
        assert config.num_inference_steps == 30
        assert config.guidance_scale == 7.5
        assert config.create_comparison_panel == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TryOnConfig(
            backend=TryOnBackend.REPLICATE,
            num_inference_steps=50,
            target_width=512,
            target_height=768,
        )
        
        assert config.backend == TryOnBackend.REPLICATE
        assert config.num_inference_steps == 50
        assert config.target_width == 512
        assert config.target_height == 768
    
    def test_config_with_replicate_token(self):
        """Test config with Replicate token."""
        config = TryOnConfig(
            backend=TryOnBackend.REPLICATE,
            replicate_token="test_token_123"
        )
        
        assert config.replicate_token == "test_token_123"


# =============================================================================
# TryOnResult Tests
# =============================================================================

class TestTryOnResult:
    """Tests for TryOnResult dataclass."""
    
    @pytest.fixture
    def dummy_image(self):
        """Create a dummy PIL Image."""
        return Image.new("RGB", (100, 100), color="red")
    
    def test_result_creation(self, dummy_image):
        """Test creating a TryOnResult."""
        result = TryOnResult(
            image=dummy_image,
            garment_id="garment_123",
            garment_type=GarmentType.UPPER_BODY,
            backend=TryOnBackend.CATVTON,
            processing_time_ms=1500.0,
            success=True,
        )
        
        assert result.image == dummy_image
        assert result.garment_id == "garment_123"
        assert result.garment_type == GarmentType.UPPER_BODY
        assert result.backend == TryOnBackend.CATVTON
        assert result.processing_time_ms == 1500.0
        assert result.success == True
        assert result.error_message is None
    
    def test_result_with_error(self, dummy_image):
        """Test TryOnResult with error."""
        result = TryOnResult(
            image=dummy_image,
            garment_id="garment_456",
            garment_type=GarmentType.LOWER_BODY,
            backend=TryOnBackend.REPLICATE,
            processing_time_ms=500.0,
            success=False,
            error_message="API timeout",
        )
        
        assert result.success == False
        assert result.error_message == "API timeout"
    
    def test_result_save(self, dummy_image, tmp_path):
        """Test saving result image."""
        result = TryOnResult(
            image=dummy_image,
            garment_id="garment_789",
            garment_type=GarmentType.UPPER_BODY,
            backend=TryOnBackend.CATVTON,
            processing_time_ms=1000.0,
        )
        
        save_path = str(tmp_path / "test_result.png")
        returned_path = result.save(save_path)
        
        assert returned_path == save_path
        assert result.result_path == save_path
        
        # Verify file was created
        loaded = Image.open(save_path)
        assert loaded.size == (100, 100)


# =============================================================================
# OutfitTryOnResult Tests
# =============================================================================

class TestOutfitTryOnResult:
    """Tests for OutfitTryOnResult dataclass."""
    
    @pytest.fixture
    def dummy_image(self):
        """Create a dummy PIL Image."""
        return Image.new("RGB", (100, 100), color="blue")
    
    @pytest.fixture
    def sample_garment_results(self, dummy_image):
        """Create sample garment results."""
        return [
            TryOnResult(
                image=dummy_image,
                garment_id="top_1",
                garment_type=GarmentType.UPPER_BODY,
                backend=TryOnBackend.CATVTON,
                processing_time_ms=1000.0,
                success=True,
            ),
            TryOnResult(
                image=dummy_image,
                garment_id="bottom_1",
                garment_type=GarmentType.LOWER_BODY,
                backend=TryOnBackend.CATVTON,
                processing_time_ms=1200.0,
                success=True,
            ),
        ]
    
    def test_outfit_result_creation(self, sample_garment_results, dummy_image):
        """Test creating an OutfitTryOnResult."""
        result = OutfitTryOnResult(
            outfit_name="Summer Casual",
            outfit_score=0.85,
            garment_results=sample_garment_results,
            composite_image=dummy_image,
            total_processing_time_ms=2200.0,
            success=True,
        )
        
        assert result.outfit_name == "Summer Casual"
        assert result.outfit_score == 0.85
        assert len(result.garment_results) == 2
        assert result.composite_image == dummy_image
        assert result.success == True
    
    def test_outfit_result_garment_counts(self, dummy_image):
        """Test garment count properties."""
        results = [
            TryOnResult(
                image=dummy_image,
                garment_id="g1",
                garment_type=GarmentType.UPPER_BODY,
                backend=TryOnBackend.CATVTON,
                processing_time_ms=100.0,
                success=True,
            ),
            TryOnResult(
                image=dummy_image,
                garment_id="g2",
                garment_type=GarmentType.LOWER_BODY,
                backend=TryOnBackend.CATVTON,
                processing_time_ms=100.0,
                success=False,
                error_message="Failed",
            ),
            TryOnResult(
                image=dummy_image,
                garment_id="g3",
                garment_type=GarmentType.FULL_BODY,
                backend=TryOnBackend.CATVTON,
                processing_time_ms=100.0,
                success=True,
            ),
        ]
        
        outfit_result = OutfitTryOnResult(
            outfit_name="Test",
            outfit_score=0.75,
            garment_results=results,
        )
        
        assert outfit_result.num_garments_processed == 2
        assert outfit_result.num_garments_failed == 1


# =============================================================================
# TryOnError Tests
# =============================================================================

class TestTryOnError:
    """Tests for TryOnError exception."""
    
    def test_error_creation(self):
        """Test creating a TryOnError."""
        error = TryOnError(
            message="Model not loaded",
            backend=TryOnBackend.CATVTON,
        )
        
        assert error.message == "Model not loaded"
        assert error.backend == TryOnBackend.CATVTON
        assert "Model not loaded" in str(error)
        assert "catvton" in str(error)
    
    def test_error_with_garment_id(self):
        """Test TryOnError with garment ID."""
        error = TryOnError(
            message="Invalid image format",
            backend=TryOnBackend.REPLICATE,
            garment_id="garment_xyz",
        )
        
        error_str = str(error)
        assert "Invalid image format" in error_str
        assert "garment_xyz" in error_str
        assert "replicate" in error_str
    
    def test_error_with_details(self):
        """Test TryOnError with details."""
        error = TryOnError(
            message="API error",
            backend=TryOnBackend.REPLICATE,
            details={"status_code": 429, "retry_after": 60},
        )
        
        assert error.details["status_code"] == 429
        assert error.details["retry_after"] == 60


# =============================================================================
# Category Mapping Tests
# =============================================================================

class TestCategoryMapping:
    """Tests for category to garment type mapping."""
    
    def test_upper_body_categories(self):
        """Test mapping upper body categories."""
        upper_body_cats = [
            "top", "tops", "shirt", "blouse", "t-shirt", 
            "sweater", "jacket", "coat", "blazer", "outerwear",
            "cardigan", "tank_top", "hoodie", "vest"
        ]
        
        for cat in upper_body_cats:
            assert category_to_garment_type(cat) == GarmentType.UPPER_BODY, f"Failed for {cat}"
    
    def test_lower_body_categories(self):
        """Test mapping lower body categories."""
        lower_body_cats = [
            "bottom", "bottoms", "pants", "trousers", "jeans",
            "skirt", "shorts", "leggings"
        ]
        
        for cat in lower_body_cats:
            assert category_to_garment_type(cat) == GarmentType.LOWER_BODY, f"Failed for {cat}"
    
    def test_full_body_categories(self):
        """Test mapping full body categories."""
        full_body_cats = [
            "dress", "dresses", "full_body", "jumpsuit", 
            "romper", "gown", "maxi_dress"
        ]
        
        for cat in full_body_cats:
            assert category_to_garment_type(cat) == GarmentType.FULL_BODY, f"Failed for {cat}"
    
    def test_footwear_categories(self):
        """Test mapping footwear categories."""
        footwear_cats = [
            "shoes", "footwear", "sneakers", "boots", 
            "heels", "sandals", "flats", "loafers"
        ]
        
        for cat in footwear_cats:
            assert category_to_garment_type(cat) == GarmentType.FOOTWEAR, f"Failed for {cat}"
    
    def test_unknown_category_defaults_to_accessory(self):
        """Test that unknown categories default to accessory."""
        unknown_cats = ["hat", "scarf", "belt", "jewelry", "unknown"]
        
        for cat in unknown_cats:
            assert category_to_garment_type(cat) == GarmentType.ACCESSORY, f"Failed for {cat}"
    
    def test_case_insensitivity(self):
        """Test that mapping is case-insensitive."""
        assert category_to_garment_type("TOP") == GarmentType.UPPER_BODY
        assert category_to_garment_type("Dress") == GarmentType.FULL_BODY
        assert category_to_garment_type("PANTS") == GarmentType.LOWER_BODY


# =============================================================================
# is_tryonable Tests
# =============================================================================

class TestIsTryonable:
    """Tests for is_tryonable helper function."""
    
    def test_tryonable_categories(self):
        """Test that main clothing categories are tryonable."""
        tryonable = ["top", "bottom", "dress", "shirt", "pants", "jumpsuit"]
        
        for cat in tryonable:
            assert is_tryonable(cat) == True, f"{cat} should be tryonable"
    
    def test_non_tryonable_categories(self):
        """Test that accessories and footwear have limited support."""
        non_tryonable = ["shoes", "hat", "bag", "jewelry", "watch"]
        
        for cat in non_tryonable:
            assert is_tryonable(cat) == False, f"{cat} should not be tryonable"
