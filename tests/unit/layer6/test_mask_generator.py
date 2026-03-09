"""
Unit Tests for Layer 6: Mask Generator
=======================================

Tests for the body mask generation component.
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from src.layer6_tryon.mask_generator import MaskGenerator
from src.layer6_tryon.models import GarmentType


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mask_generator():
    """Create a MaskGenerator with rembg disabled."""
    return MaskGenerator(use_rembg=False)


@pytest.fixture
def rembg_mask_generator():
    """Create a MaskGenerator with rembg enabled."""
    return MaskGenerator(use_rembg=True, cache_silhouettes=True)


@pytest.fixture
def sample_person_image():
    """Create a sample person image (simple rectangle)."""
    img = Image.new("RGB", (768, 1024), color="white")
    # Draw a simple "body" shape
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Head
    draw.ellipse([334, 50, 434, 150], fill="beige")
    # Body
    draw.rectangle([284, 150, 484, 650], fill="blue")
    # Legs
    draw.rectangle([284, 650, 384, 950], fill="blue")
    draw.rectangle([384, 650, 484, 950], fill="blue")
    return img


# =============================================================================
# MaskGenerator Initialization Tests
# =============================================================================

class TestMaskGeneratorInit:
    """Tests for MaskGenerator initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        gen = MaskGenerator()
        assert gen.use_rembg == True
        assert gen.cache_silhouettes == True
    
    def test_custom_init(self):
        """Test custom initialization."""
        gen = MaskGenerator(use_rembg=False, cache_silhouettes=False)
        assert gen.use_rembg == False
        assert gen.cache_silhouettes == False
    
    def test_cache_starts_empty(self):
        """Test that silhouette cache starts empty."""
        gen = MaskGenerator()
        assert len(gen._silhouette_cache) == 0


# =============================================================================
# Geometric Mask Generation Tests
# =============================================================================

class TestGeometricMaskGeneration:
    """Tests for geometric fallback mask generation."""
    
    def test_upper_body_mask_dimensions(self, mask_generator, sample_person_image):
        """Test upper body mask has correct dimensions."""
        mask = mask_generator.generate_mask(
            sample_person_image,
            GarmentType.UPPER_BODY
        )
        
        assert mask.mode == "L"  # Grayscale
        assert mask.size == sample_person_image.size
    
    def test_upper_body_mask_covers_torso(self, mask_generator, sample_person_image):
        """Test upper body mask covers torso region."""
        mask = mask_generator.generate_mask(
            sample_person_image,
            GarmentType.UPPER_BODY
        )
        
        # Convert to numpy for analysis
        mask_arr = np.array(mask)
        
        # Check that center-top area has white pixels
        h, w = mask_arr.shape
        torso_region = mask_arr[int(h*0.3):int(h*0.5), int(w*0.3):int(w*0.7)]
        
        # Should have significant white area
        assert np.mean(torso_region) > 100  # Should be mostly white
    
    def test_lower_body_mask(self, mask_generator, sample_person_image):
        """Test lower body mask covers legs region."""
        mask = mask_generator.generate_mask(
            sample_person_image,
            GarmentType.LOWER_BODY
        )
        
        mask_arr = np.array(mask)
        h, w = mask_arr.shape
        
        # Lower region should be white
        lower_region = mask_arr[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
        assert np.mean(lower_region) > 100
        
        # Upper region should be black
        upper_region = mask_arr[int(h*0.1):int(h*0.3), int(w*0.3):int(w*0.7)]
        assert np.mean(upper_region) < 50
    
    def test_full_body_mask(self, mask_generator, sample_person_image):
        """Test full body mask covers entire body."""
        mask = mask_generator.generate_mask(
            sample_person_image,
            GarmentType.FULL_BODY
        )
        
        mask_arr = np.array(mask)
        h, w = mask_arr.shape
        
        # Both torso and legs should be white
        body_region = mask_arr[int(h*0.25):int(h*0.85), int(w*0.3):int(w*0.7)]
        assert np.mean(body_region) > 100
    
    def test_accessory_mask_is_small(self, mask_generator, sample_person_image):
        """Test accessory mask covers small center area."""
        mask = mask_generator.generate_mask(
            sample_person_image,
            GarmentType.ACCESSORY
        )
        
        mask_arr = np.array(mask)
        
        # Count white pixels (above threshold due to blur)
        white_pixels = np.sum(mask_arr > 128)
        total_pixels = mask_arr.size
        
        # Should cover less than 20% of image
        coverage = white_pixels / total_pixels
        assert coverage < 0.25
    
    def test_mask_has_soft_edges(self, mask_generator, sample_person_image):
        """Test that mask has soft (blurred) edges."""
        mask = mask_generator.generate_mask(
            sample_person_image,
            GarmentType.UPPER_BODY
        )
        
        mask_arr = np.array(mask)
        
        # Check for gradient values (not just 0 and 255)
        unique_values = np.unique(mask_arr)
        
        # Should have intermediate values due to blur
        assert len(unique_values) > 2


# =============================================================================
# Different Image Size Tests
# =============================================================================

class TestDifferentImageSizes:
    """Tests for mask generation with different image sizes."""
    
    @pytest.mark.parametrize("size", [
        (512, 512),
        (768, 1024),
        (1024, 768),
        (256, 512),
    ])
    def test_mask_matches_input_size(self, mask_generator, size):
        """Test mask size matches input image size."""
        img = Image.new("RGB", size, color="white")
        mask = mask_generator.generate_mask(img, GarmentType.UPPER_BODY)
        
        assert mask.size == size
    
    def test_small_image(self, mask_generator):
        """Test mask generation for small images."""
        small_img = Image.new("RGB", (100, 150), color="white")
        mask = mask_generator.generate_mask(small_img, GarmentType.UPPER_BODY)
        
        assert mask.size == (100, 150)
    
    def test_large_image(self, mask_generator):
        """Test mask generation for large images."""
        large_img = Image.new("RGB", (2048, 2048), color="white")
        mask = mask_generator.generate_mask(large_img, GarmentType.FULL_BODY)
        
        assert mask.size == (2048, 2048)


# =============================================================================
# Cache Tests
# =============================================================================

class TestMaskCache:
    """Tests for silhouette caching."""
    
    def test_clear_cache(self, rembg_mask_generator):
        """Test clearing the cache."""
        # Manually add something to cache
        rembg_mask_generator._silhouette_cache["test_key"] = np.zeros((100, 100))
        
        assert len(rembg_mask_generator._silhouette_cache) == 1
        
        rembg_mask_generator.clear_cache()
        
        assert len(rembg_mask_generator._silhouette_cache) == 0


# =============================================================================
# Region Coordinates Tests
# =============================================================================

class TestRegionCoordinates:
    """Tests for region coordinate calculation."""
    
    def test_upper_body_region(self, mask_generator):
        """Test upper body region coordinates."""
        # Body bounds: x=100-300, y=50-400
        region = mask_generator._get_region_coords(
            GarmentType.UPPER_BODY,
            cmin=100, cmax=300,
            rmin=50, rmax=400,
            h_body=350, w_body=200
        )
        
        # Should be within body bounds
        assert region[0] >= 100  # left
        assert region[2] <= 300  # right
        assert region[1] >= 50   # top
        assert region[3] <= 400  # bottom
        
        # Should cover torso (roughly 18% to 65% of body height)
        expected_top = int(50 + 350 * 0.18)
        expected_bottom = int(50 + 350 * 0.65)
        assert abs(region[1] - expected_top) < 5
        assert abs(region[3] - expected_bottom) < 5
    
    def test_lower_body_region(self, mask_generator):
        """Test lower body region coordinates."""
        region = mask_generator._get_region_coords(
            GarmentType.LOWER_BODY,
            cmin=100, cmax=300,
            rmin=50, rmax=400,
            h_body=350, w_body=200
        )
        
        # Should start at ~50% of body height
        expected_top = int(50 + 350 * 0.50)
        assert abs(region[1] - expected_top) < 5
        
        # Should extend to bottom
        assert region[3] == 400
    
    def test_full_body_region(self, mask_generator):
        """Test full body region coordinates."""
        region = mask_generator._get_region_coords(
            GarmentType.FULL_BODY,
            cmin=100, cmax=300,
            rmin=50, rmax=400,
            h_body=350, w_body=200
        )
        
        # Should cover most of body from shoulders to bottom
        expected_top = int(50 + 350 * 0.18)
        assert abs(region[1] - expected_top) < 5
        assert region[3] == 400


# =============================================================================
# Edge Cases
# =============================================================================

class TestMaskEdgeCases:
    """Tests for edge cases in mask generation."""
    
    def test_very_narrow_image(self, mask_generator):
        """Test mask for very narrow image."""
        narrow_img = Image.new("RGB", (50, 500), color="white")
        mask = mask_generator.generate_mask(narrow_img, GarmentType.UPPER_BODY)
        
        assert mask.size == (50, 500)
    
    def test_very_wide_image(self, mask_generator):
        """Test mask for very wide image."""
        wide_img = Image.new("RGB", (1000, 100), color="white")
        mask = mask_generator.generate_mask(wide_img, GarmentType.UPPER_BODY)
        
        assert mask.size == (1000, 100)
    
    def test_square_image(self, mask_generator):
        """Test mask for square image."""
        square_img = Image.new("RGB", (512, 512), color="white")
        mask = mask_generator.generate_mask(square_img, GarmentType.FULL_BODY)
        
        assert mask.size == (512, 512)
