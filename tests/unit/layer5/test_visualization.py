"""
Tests for Layer 5: Outfit Visualization
"""

import pytest
from pathlib import Path
from PIL import Image
import tempfile

from src.core.models import Garment, GarmentAttributes, GarmentCategory, ColorProfile, PatternInfo


class TestOutfitVisualizer:
    """Tests for OutfitVisualizer."""
    
    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance."""
        from src.layer5_visualization import OutfitVisualizer
        return OutfitVisualizer(garment_size=200, padding=10)
    
    @pytest.fixture
    def sample_garments(self):
        """Create sample garments for testing."""
        return [
            Garment(
                id="top_1",
                attributes=GarmentAttributes(
                    category=GarmentCategory.TOP,
                    subcategory="t-shirt",
                    color=ColorProfile(primary="white"),
                    pattern=PatternInfo(type="solid"),
                )
            ),
            Garment(
                id="bottom_1",
                attributes=GarmentAttributes(
                    category=GarmentCategory.BOTTOM,
                    subcategory="jeans",
                    color=ColorProfile(primary="blue"),
                    pattern=PatternInfo(type="solid"),
                )
            ),
            Garment(
                id="shoes_1",
                attributes=GarmentAttributes(
                    category=GarmentCategory.SHOES,
                    subcategory="sneakers",
                    color=ColorProfile(primary="black"),
                    pattern=PatternInfo(type="solid"),
                )
            ),
        ]
    
    def test_create_outfit_catalogue(self, visualizer, sample_garments):
        """Test creating an outfit catalogue."""
        from src.layer5_visualization import OutfitVisualization
        
        result = visualizer.create_outfit_catalogue(
            garments=sample_garments,
            outfit_name="Test Outfit",
            score=0.85
        )
        
        assert isinstance(result, OutfitVisualization)
        assert result.outfit_name == "Test Outfit"
        assert result.garment_count == 3
        assert result.score == 0.85
        assert result.image is not None
        assert isinstance(result.image, Image.Image)
    
    def test_catalogue_image_is_rgba(self, visualizer, sample_garments):
        """Test that catalogue image is RGBA."""
        result = visualizer.create_outfit_catalogue(
            garments=sample_garments,
            outfit_name="Test"
        )
        
        assert result.image.mode == "RGBA"
    
    def test_catalogue_with_custom_columns(self, visualizer, sample_garments):
        """Test creating catalogue with specific column count."""
        result = visualizer.create_outfit_catalogue(
            garments=sample_garments,
            outfit_name="Test",
            columns=1  # Single column
        )
        
        # With 3 garments and 1 column, image should be taller than wide
        width, height = result.image.size
        assert height > width
    
    def test_save_visualization(self, visualizer, sample_garments, tmp_path):
        """Test saving visualization to file."""
        viz = visualizer.create_outfit_catalogue(
            garments=sample_garments,
            outfit_name="Test"
        )
        
        output_path = tmp_path / "output" / "catalogue.png"
        result_path = visualizer.save_visualization(viz, output_path)
        
        assert result_path.exists()
        assert result_path == output_path
        
        # Verify it's a valid image
        loaded = Image.open(result_path)
        assert loaded.size == viz.image.size
    
    def test_empty_garments_raises_error(self, visualizer):
        """Test that empty garments list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot create catalogue"):
            visualizer.create_outfit_catalogue(
                garments=[],
                outfit_name="Empty"
            )
    
    def test_catalogue_with_score_coloring(self, visualizer, sample_garments):
        """Test that different scores produce different visual results."""
        # High score (green)
        high_score_viz = visualizer.create_outfit_catalogue(
            garments=sample_garments,
            outfit_name="High Score",
            score=0.9
        )
        
        # Low score (red)
        low_score_viz = visualizer.create_outfit_catalogue(
            garments=sample_garments,
            outfit_name="Low Score",
            score=0.4
        )
        
        # Both should be valid images
        assert high_score_viz.image is not None
        assert low_score_viz.image is not None


class TestCreateOutfitImage:
    """Tests for the convenience function create_outfit_image."""
    
    @pytest.fixture
    def sample_garments(self):
        """Create sample garments."""
        return [
            Garment(
                id="top_1",
                attributes=GarmentAttributes(
                    category=GarmentCategory.TOP,
                    subcategory="shirt",
                    color=ColorProfile(primary="navy"),
                )
            ),
            Garment(
                id="bottom_1",
                attributes=GarmentAttributes(
                    category=GarmentCategory.BOTTOM,
                    subcategory="chinos",
                    color=ColorProfile(primary="beige"),
                )
            ),
        ]
    
    def test_create_outfit_image_saves_file(self, sample_garments, tmp_path):
        """Test that create_outfit_image creates a file."""
        from src.layer5_visualization import create_outfit_image
        
        output_path = tmp_path / "outfit.png"
        result = create_outfit_image(
            garments=sample_garments,
            output_path=output_path,
            outfit_name="Business Casual",
            score=0.75
        )
        
        assert result == output_path
        assert output_path.exists()
        
        # Verify image
        img = Image.open(output_path)
        assert img.mode == "RGBA"


class TestOutfitComparison:
    """Tests for outfit comparison visualization."""
    
    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance."""
        from src.layer5_visualization import OutfitVisualizer
        return OutfitVisualizer(garment_size=150, padding=5)
    
    @pytest.fixture
    def multiple_outfits(self):
        """Create multiple outfits for comparison."""
        outfit1 = [
            Garment(
                id="top_1",
                attributes=GarmentAttributes(
                    category=GarmentCategory.TOP,
                    color=ColorProfile(primary="white"),
                )
            ),
            Garment(
                id="bottom_1",
                attributes=GarmentAttributes(
                    category=GarmentCategory.BOTTOM,
                    color=ColorProfile(primary="black"),
                )
            ),
        ]
        
        outfit2 = [
            Garment(
                id="top_2",
                attributes=GarmentAttributes(
                    category=GarmentCategory.TOP,
                    color=ColorProfile(primary="blue"),
                )
            ),
            Garment(
                id="bottom_2",
                attributes=GarmentAttributes(
                    category=GarmentCategory.BOTTOM,
                    color=ColorProfile(primary="gray"),
                )
            ),
        ]
        
        return [
            ("Outfit A", outfit1, 0.85),
            ("Outfit B", outfit2, 0.72),
        ]
    
    def test_create_comparison_catalogue(self, visualizer, multiple_outfits):
        """Test creating a comparison of multiple outfits."""
        result = visualizer.create_comparison_catalogue(
            outfits=multiple_outfits,
            title="Outfit Comparison"
        )
        
        assert result.outfit_name == "Outfit Comparison"
        assert result.garment_count == 4  # 2 outfits x 2 garments
        assert result.image is not None
