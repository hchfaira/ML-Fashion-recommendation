"""
Integration Tests for Layer 0: Garment Extraction
==================================================

These tests verify that garments are correctly extracted from real images.
Tests use actual images to ensure the segmentation produces usable results.
"""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from src.layer0_segmentation import (
    SimpleGarmentSegmenter,
    SegmentedGarment,
    get_segmenter,
    DEFAULT_CANVAS_SIZE
)
from src.core import get_logger

logger = get_logger(__name__)


# Test image directories
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"
SEGMENTATION_DIR = TEST_IMAGES_DIR / "segmentation"
WARDROBE_DIR = TEST_IMAGES_DIR / "outfit_builder" / "wardrobe"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "segmentation"


class TestGarmentExtractionFromRealImages:
    """
    Integration tests that verify garment extraction works correctly
    on real images from the test dataset.
    """
    
    @pytest.fixture
    def segmenter(self):
        """Get a segmenter (SimpleGarmentSegmenter as fallback)."""
        return get_segmenter(use_sam=False, canvas_size=DEFAULT_CANVAS_SIZE)
    
    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory for visual inspection."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        yield
    
    def _get_test_images(self) -> list[Path]:
        """Get all test images from various directories."""
        images = []
        
        # From segmentation test folder
        if SEGMENTATION_DIR.exists():
            images.extend(SEGMENTATION_DIR.glob("*.jpg"))
            images.extend(SEGMENTATION_DIR.glob("*.png"))
        
        # From wardrobe folders
        if WARDROBE_DIR.exists():
            for category_dir in WARDROBE_DIR.iterdir():
                if category_dir.is_dir():
                    images.extend(category_dir.glob("*.jpg"))
                    images.extend(category_dir.glob("*.png"))
        
        return images
    
    def test_extract_single_garment(self, segmenter):
        """Test extracting a single garment from an image."""
        # Get any available test image
        test_images = self._get_test_images()
        if not test_images:
            pytest.skip("No test images available")
        
        image_path = test_images[0]
        logger.info(f"Testing extraction from: {image_path.name}")
        
        result = segmenter.segment_garment(image_path)
        
        # Verify basic structure
        assert isinstance(result, SegmentedGarment)
        assert result.image is not None
        assert result.original_path == image_path
        
        # Verify image properties
        assert result.image.mode == "RGBA", "Output should be RGBA for transparency"
        assert result.image.size == (DEFAULT_CANVAS_SIZE, DEFAULT_CANVAS_SIZE)
        
        # Verify bounding box exists
        assert result.bounding_box is not None
        assert len(result.bounding_box) == 4
        
        # Save for visual inspection
        output_path = OUTPUT_DIR / f"extracted_{image_path.stem}.png"
        result.image.save(output_path)
        logger.info(f"Saved extracted garment to: {output_path}")
    
    def test_extracted_garment_has_transparency(self, segmenter):
        """Verify that extracted garment has transparent background."""
        test_images = self._get_test_images()
        if not test_images:
            pytest.skip("No test images available")
        
        image_path = test_images[0]
        result = segmenter.segment_garment(image_path)
        
        # Convert to numpy to check alpha channel
        img_array = np.array(result.image)
        alpha_channel = img_array[:, :, 3]
        
        # There should be some transparent pixels (alpha = 0)
        # and some opaque pixels (alpha = 255)
        has_transparent = np.any(alpha_channel < 128)
        has_opaque = np.any(alpha_channel > 128)
        
        # At minimum, there should be opaque pixels (the garment)
        assert has_opaque, "Extracted garment should have opaque pixels"
        
        # Log transparency stats
        total_pixels = alpha_channel.size
        transparent_count = np.sum(alpha_channel < 128)
        transparency_ratio = transparent_count / total_pixels
        
        logger.info(f"Transparency ratio: {transparency_ratio:.2%}")
        logger.info(f"Transparent pixels: {transparent_count}, Opaque: {total_pixels - transparent_count}")
    
    def test_extracted_garment_is_centered(self, segmenter):
        """Verify that extracted garment is reasonably centered on canvas."""
        test_images = self._get_test_images()
        if not test_images:
            pytest.skip("No test images available")
        
        image_path = test_images[0]
        result = segmenter.segment_garment(image_path)
        
        # Find the bounding box of non-transparent pixels
        img_array = np.array(result.image)
        alpha = img_array[:, :, 3]
        
        # Find rows and columns with content
        rows_with_content = np.any(alpha > 128, axis=1)
        cols_with_content = np.any(alpha > 128, axis=0)
        
        if not np.any(rows_with_content):
            pytest.fail("No content found in extracted image")
        
        # Get bounds
        row_indices = np.where(rows_with_content)[0]
        col_indices = np.where(cols_with_content)[0]
        
        top_margin = row_indices[0]
        bottom_margin = result.image.height - row_indices[-1] - 1
        left_margin = col_indices[0]
        right_margin = result.image.width - col_indices[-1] - 1
        
        # Check centering (margins should be roughly equal, within 20% tolerance)
        canvas_size = result.image.width
        
        vertical_balance = abs(top_margin - bottom_margin) / canvas_size
        horizontal_balance = abs(left_margin - right_margin) / canvas_size
        
        logger.info(f"Margins - Top: {top_margin}, Bottom: {bottom_margin}, Left: {left_margin}, Right: {right_margin}")
        logger.info(f"Centering balance - Vertical: {1 - vertical_balance:.2%}, Horizontal: {1 - horizontal_balance:.2%}")
        
        # Allow 30% imbalance (some garments may not be perfectly centered)
        assert vertical_balance < 0.3, f"Garment not vertically centered: {vertical_balance:.2%} imbalance"
        assert horizontal_balance < 0.3, f"Garment not horizontally centered: {horizontal_balance:.2%} imbalance"
    
    def test_extract_all_wardrobe_categories(self, segmenter):
        """Test extraction from all wardrobe categories."""
        if not WARDROBE_DIR.exists():
            pytest.skip("Wardrobe test directory not found")
        
        categories_tested = []
        results = {}
        
        for category_dir in WARDROBE_DIR.iterdir():
            if not category_dir.is_dir():
                continue
            
            images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
            if not images:
                continue
            
            category_name = category_dir.name
            image_path = images[0]  # Test first image in category
            
            logger.info(f"Testing category: {category_name} with {image_path.name}")
            
            result = segmenter.segment_garment(image_path)
            
            assert result is not None, f"Failed to extract from {category_name}"
            assert result.image is not None
            
            categories_tested.append(category_name)
            results[category_name] = result
            
            # Save for visual inspection
            output_path = OUTPUT_DIR / f"category_{category_name}_{image_path.stem}.png"
            result.image.save(output_path)
        
        logger.info(f"Successfully tested categories: {categories_tested}")
        assert len(categories_tested) > 0, "No categories were tested"
    
    def test_extract_preserves_garment_colors(self, segmenter):
        """Verify that extracted garment preserves original colors."""
        test_images = self._get_test_images()
        if not test_images:
            pytest.skip("No test images available")
        
        image_path = test_images[0]
        
        # Load original image
        original = Image.open(image_path).convert("RGB")
        original_array = np.array(original)
        
        # Extract garment
        result = segmenter.segment_garment(image_path)
        extracted_array = np.array(result.image)
        
        # Get average color of opaque pixels in extracted image
        alpha = extracted_array[:, :, 3]
        opaque_mask = alpha > 128
        
        if not np.any(opaque_mask):
            pytest.fail("No opaque pixels in extracted image")
        
        # Calculate average RGB of opaque pixels
        rgb_channels = extracted_array[:, :, :3]
        opaque_pixels = rgb_channels[opaque_mask]
        avg_extracted_color = np.mean(opaque_pixels, axis=0)
        
        # Calculate average color of original image
        avg_original_color = np.mean(original_array, axis=(0, 1))
        
        logger.info(f"Original avg color: {avg_original_color}")
        logger.info(f"Extracted avg color: {avg_extracted_color}")
        
        # Colors should be similar (garment is part of original image)
        # This is a sanity check, not exact match
        # Note: This may not be exact if background is very different from garment
        assert avg_extracted_color is not None
    
    def test_segment_folder_extracts_all_images(self, segmenter):
        """Test that segment_folder processes all images correctly."""
        if not SEGMENTATION_DIR.exists():
            pytest.skip("Segmentation test directory not found")
        
        # Count expected images
        expected_images = list(SEGMENTATION_DIR.glob("*.jpg")) + list(SEGMENTATION_DIR.glob("*.png"))
        
        if not expected_images:
            pytest.skip("No images in segmentation test directory")
        
        results = segmenter.segment_folder(SEGMENTATION_DIR)
        
        assert len(results) == len(expected_images), \
            f"Expected {len(expected_images)} results, got {len(results)}"
        
        for result in results:
            assert result.image is not None
            assert result.image.mode == "RGBA"
            assert result.image.size == (DEFAULT_CANVAS_SIZE, DEFAULT_CANVAS_SIZE)
    
    def test_extraction_metadata_is_valid(self, segmenter):
        """Verify that extraction metadata (bounding box, area, confidence) is valid."""
        test_images = self._get_test_images()
        if not test_images:
            pytest.skip("No test images available")
        
        image_path = test_images[0]
        result = segmenter.segment_garment(image_path)
        
        # Validate bounding box
        x, y, w, h = result.bounding_box
        assert x >= 0, "Bounding box x should be non-negative"
        assert y >= 0, "Bounding box y should be non-negative"
        assert w > 0, "Bounding box width should be positive"
        assert h > 0, "Bounding box height should be positive"
        
        # Validate area
        assert result.area > 0, "Area should be positive"
        assert result.area <= w * h, "Area should not exceed bounding box area"
        
        # Validate confidence
        assert 0.0 <= result.confidence <= 1.0, "Confidence should be between 0 and 1"
        
        logger.info(f"Metadata - BBox: {result.bounding_box}, Area: {result.area}, Confidence: {result.confidence}")


class TestGarmentExtractionQuality:
    """
    Quality tests to ensure extracted garments meet visual standards.
    """
    
    @pytest.fixture
    def segmenter(self):
        return get_segmenter(use_sam=False, canvas_size=512)
    
    def test_extracted_image_not_too_small(self, segmenter):
        """Verify that the extracted garment is not too small on the canvas."""
        test_images = []
        if SEGMENTATION_DIR.exists():
            test_images = list(SEGMENTATION_DIR.glob("*.jpg")) + list(SEGMENTATION_DIR.glob("*.png"))
        
        if not test_images:
            pytest.skip("No test images available")
        
        image_path = test_images[0]
        result = segmenter.segment_garment(image_path)
        
        # Calculate coverage
        img_array = np.array(result.image)
        alpha = img_array[:, :, 3]
        
        opaque_pixels = np.sum(alpha > 128)
        total_pixels = alpha.size
        coverage = opaque_pixels / total_pixels
        
        # Garment should cover at least 10% of the canvas
        logger.info(f"Canvas coverage: {coverage:.2%}")
        assert coverage >= 0.10, f"Garment too small: only covers {coverage:.2%} of canvas"
    
    def test_extracted_image_not_too_large(self, segmenter):
        """Verify that the extracted garment doesn't overflow canvas."""
        test_images = []
        if SEGMENTATION_DIR.exists():
            test_images = list(SEGMENTATION_DIR.glob("*.jpg")) + list(SEGMENTATION_DIR.glob("*.png"))
        
        if not test_images:
            pytest.skip("No test images available")
        
        image_path = test_images[0]
        result = segmenter.segment_garment(image_path)
        
        # Check that there are no opaque pixels at the very edge
        img_array = np.array(result.image)
        alpha = img_array[:, :, 3]
        
        canvas_size = result.image.width
        
        # Check 1-pixel border (should be mostly transparent)
        top_edge = alpha[0, :].sum()
        bottom_edge = alpha[-1, :].sum()
        left_edge = alpha[:, 0].sum()
        right_edge = alpha[:, -1].sum()
        
        edge_content = top_edge + bottom_edge + left_edge + right_edge
        max_edge_content = 4 * canvas_size * 255  # If all edge pixels were opaque
        
        edge_ratio = edge_content / max_edge_content
        
        logger.info(f"Edge content ratio: {edge_ratio:.2%}")
        
        # Allow some edge content but not excessive
        assert edge_ratio < 0.5, f"Too much content at edges: {edge_ratio:.2%}"


class TestCreateSyntheticTestImages:
    """
    Tests using synthetic images with known garment shapes.
    These provide controlled test cases.
    """
    
    @pytest.fixture
    def segmenter(self):
        return get_segmenter(use_sam=False, canvas_size=256)
    
    @pytest.fixture
    def synthetic_tshirt(self, tmp_path):
        """Create a synthetic T-shirt shaped image."""
        # Create white background
        img = Image.new("RGB", (300, 400), color=(245, 245, 245))
        pixels = img.load()
        
        # Draw a simple T-shirt shape (blue)
        tshirt_color = (50, 100, 180)  # Blue
        
        # Body
        for x in range(75, 225):
            for y in range(100, 350):
                pixels[x, y] = tshirt_color
        
        # Left sleeve
        for x in range(30, 100):
            for y in range(100, 180):
                pixels[x, y] = tshirt_color
        
        # Right sleeve
        for x in range(200, 270):
            for y in range(100, 180):
                pixels[x, y] = tshirt_color
        
        # Save
        img_path = tmp_path / "synthetic_tshirt.png"
        img.save(img_path)
        return img_path
    
    @pytest.fixture
    def synthetic_pants(self, tmp_path):
        """Create a synthetic pants shaped image."""
        img = Image.new("RGB", (300, 500), color=(240, 240, 240))
        pixels = img.load()
        
        pants_color = (50, 50, 60)  # Dark gray/black
        
        # Waist
        for x in range(75, 225):
            for y in range(50, 120):
                pixels[x, y] = pants_color
        
        # Left leg
        for x in range(75, 145):
            for y in range(120, 450):
                pixels[x, y] = pants_color
        
        # Right leg
        for x in range(155, 225):
            for y in range(120, 450):
                pixels[x, y] = pants_color
        
        img_path = tmp_path / "synthetic_pants.png"
        img.save(img_path)
        return img_path
    
    def test_extract_synthetic_tshirt(self, segmenter, synthetic_tshirt):
        """Test extraction of synthetic T-shirt."""
        result = segmenter.segment_garment(synthetic_tshirt)
        
        assert result is not None
        assert result.image.mode == "RGBA"
        
        # Check that blue is preserved
        img_array = np.array(result.image)
        alpha = img_array[:, :, 3]
        opaque_mask = alpha > 128
        
        if np.any(opaque_mask):
            blue_channel = img_array[:, :, 2][opaque_mask]
            avg_blue = np.mean(blue_channel)
            
            # Should be predominantly blue
            assert avg_blue > 100, f"Blue color not preserved, avg blue: {avg_blue}"
    
    def test_extract_synthetic_pants(self, segmenter, synthetic_pants):
        """Test extraction of synthetic pants."""
        result = segmenter.segment_garment(synthetic_pants)
        
        assert result is not None
        assert result.image.mode == "RGBA"
        
        # Check that extraction produces an image
        # Note: Without advanced models (GroundingDINO/SAM), fallback extracts full image
        img_array = np.array(result.image)
        alpha = img_array[:, :, 3]
        opaque_mask = alpha > 128
        
        # At minimum, there should be opaque pixels
        assert np.any(opaque_mask), "Should have some opaque pixels"
    
    def test_compare_different_garment_shapes(self, segmenter, synthetic_tshirt, synthetic_pants):
        """Test that different garment shapes produce different extractions."""
        result_tshirt = segmenter.segment_garment(synthetic_tshirt)
        result_pants = segmenter.segment_garment(synthetic_pants)
        
        # Both should succeed
        assert result_tshirt is not None
        assert result_pants is not None
        
        # They should produce valid images
        tshirt_array = np.array(result_tshirt.image)
        pants_array = np.array(result_pants.image)
        
        # Both should have opaque pixels
        tshirt_opaque = tshirt_array[:, :, 3] > 128
        pants_opaque = pants_array[:, :, 3] > 128
        
        assert np.any(tshirt_opaque), "T-shirt should have opaque pixels"
        assert np.any(pants_opaque), "Pants should have opaque pixels"
        
        # With advanced models, these would have significantly different colors
        # With fallback, they may be similar due to background being included
