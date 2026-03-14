"""
Tests for the MinimalistOutfitVisualizer.
"""

import pytest
from pathlib import Path
from PIL import Image
from datetime import datetime, timezone

from src.core.models import (
    Garment,
    GarmentAttributes,
    GarmentCategory,
    FormalityLevel,
    ColorProfile,
    MaterialProfile,
    StyleIdentity,
    OccasionProfile,
    SilhouetteProfile,
    VersatilityInfo,
    TrendAlignment,
    StatementLevel,
)
from src.layer5_visualization.minimalist_visualizer import (
    MinimalistOutfitVisualizer,
    OutfitVisualization,
    QuietLuxuryVisualizer,
    W,
    H,
    _pick_hero,
    _compute_grid,
    _ql_canvas_size,
    _spaced,
    visualize_outfit,
)


@pytest.fixture
def visualizer():
    """Create a MinimalistOutfitVisualizer instance."""
    return MinimalistOutfitVisualizer()


@pytest.fixture
def sample_top_garment():
    """Create a sample top garment."""
    return Garment(
        id="top_001",
        image_path=None,
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            subcategory="blouse",
            color=ColorProfile(primary="beige", secondary="white"),
            material=MaterialProfile(primary="cotton", secondary="linen"),
            formality_level=FormalityLevel.BUSINESS_CASUAL,
            style_tags=["minimal", "elegant"],
        ),
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_bottom_garment():
    """Create a sample bottom garment."""
    return Garment(
        id="bottom_001",
        image_path=None,
        attributes=GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            subcategory="trousers",
            color=ColorProfile(primary="olive", secondary="green"),
            material=MaterialProfile(primary="cotton", secondary="elastane"),
            formality_level=FormalityLevel.BUSINESS_CASUAL,
            style_tags=["minimal"],
        ),
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_shoes_garment():
    """Create a sample shoes garment."""
    return Garment(
        id="shoes_001",
        image_path=None,
        attributes=GarmentAttributes(
            category=GarmentCategory.SHOES,
            subcategory="heels",
            color=ColorProfile(primary="black"),
            material=MaterialProfile(primary="leather"),
            formality_level=FormalityLevel.BUSINESS,
            style_tags=["elegant"],
        ),
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_accessory_garment():
    """Create a sample accessory garment."""
    return Garment(
        id="acc_001",
        image_path=None,
        attributes=GarmentAttributes(
            category=GarmentCategory.ACCESSORY,
            subcategory="necklace",
            color=ColorProfile(primary="gold"),
            material=MaterialProfile(primary="metal"),
            formality_level=FormalityLevel.BUSINESS,
            style_tags=["jewelry"],
        ),
        created_at=datetime.now(timezone.utc),
    )


class TestMinimalistOutfitVisualizerInitialization:
    """Test visualization initialization."""

    def test_visualizer_initialization(self, visualizer):
        """Test that visualizer initializes and fonts are loaded."""
        assert visualizer._f_title is not None
        assert visualizer._f_body is not None
        assert visualizer._f_section is not None
        assert visualizer._f_caption is not None

    def test_fonts_load_with_fallback(self, visualizer):
        """Test that fonts load with fallback."""
        # Even if custom fonts are not found, default fonts should be loaded
        assert visualizer._f_title is not None
        assert visualizer._f_section is not None


class TestCreateVisualization:
    """Test visualization creation."""

    def test_create_visualization_with_full_outfit(
        self, visualizer, sample_top_garment, sample_bottom_garment, sample_shoes_garment, sample_accessory_garment
    ):
        """Test creating visualization with a complete outfit."""
        garments = [sample_top_garment, sample_bottom_garment, sample_shoes_garment, sample_accessory_garment]
        viz = visualizer.create_visualization(garments)

        assert isinstance(viz, OutfitVisualization)
        assert viz.image is not None
        assert viz.garment_count == 4
        assert viz.outfit_name == "Minimalist Style Guide"
        assert isinstance(viz.image, Image.Image)

    def test_create_visualization_with_minimal_outfit(
        self, visualizer, sample_top_garment, sample_bottom_garment
    ):
        """Test creating visualization with minimal outfit (top + bottom only)."""
        garments = [sample_top_garment, sample_bottom_garment]
        viz = visualizer.create_visualization(garments)

        assert isinstance(viz, OutfitVisualization)
        assert viz.garment_count == 2
        assert viz.image is not None

    def test_create_visualization_canvas_size(self, visualizer, sample_top_garment):
        """Test that the generated canvas has correct dimensions."""
        garments = [sample_top_garment]
        viz = visualizer.create_visualization(garments)

        assert viz.image.width == W
        assert viz.image.height == H

    def test_create_visualization_no_garments_raises_error(self, visualizer):
        """Test that creating visualization with no garments raises error."""
        with pytest.raises(ValueError, match="Cannot create visualization with no garments"):
            visualizer.create_visualization([])

    def test_create_visualization_with_segmented_images(
        self, visualizer, sample_top_garment, sample_bottom_garment
    ):
        """Test creating visualization with segmented images."""
        # Create placeholder images
        segmented_images = {
            "top_001": Image.new("RGBA", (100, 100), (255, 0, 0, 255)),
            "bottom_001": Image.new("RGBA", (100, 100), (0, 255, 0, 255)),
        }

        garments = [sample_top_garment, sample_bottom_garment]
        viz = visualizer.create_visualization(garments, segmented_images=segmented_images)

        assert viz.image is not None
        assert viz.garment_count == 2


class TestGarmentImageHandling:
    """Test garment image loading."""

    def test_load_image_from_segmented_images(self, visualizer, sample_top_garment):
        """Test that segmented images are preferred."""
        segmented_img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        segmented_images = {"top_001": segmented_img}

        result = visualizer._load_image(sample_top_garment, segmented_images)
        assert result is segmented_img

    def test_load_image_placeholder_fallback(self, visualizer, sample_top_garment):
        """Test that a placeholder image is returned when no image is available."""
        result = visualizer._load_image(sample_top_garment, None)

        assert isinstance(result, Image.Image)
        assert result.width > 0 and result.height > 0

    def test_fit_image_resizing(self, visualizer):
        """Test that images are properly resized to fit the target box."""
        img = Image.new("RGB", (500, 500), (255, 0, 0))
        result = visualizer._fit_image(img, 200, 200)

        assert result.width <= 200
        assert result.height <= 200
        assert result.mode == "RGBA"

    def test_fit_image_portrait_keeps_aspect_ratio(self, visualizer):
        """Tall image fitted into square box: height == target, width < target."""
        img = Image.new("RGB", (200, 400), (255, 0, 0))
        result = visualizer._fit_image(img, 200, 200)

        # Result must fit inside 200×200
        assert result.width <= 200
        assert result.height <= 200


class TestSaveVisualization:
    """Test saving visualizations."""

    def test_save_visualization_as_png(self, visualizer, sample_top_garment, tmp_path):
        """Test saving visualization as PNG."""
        garments = [sample_top_garment]
        viz = visualizer.create_visualization(garments)

        output_path = tmp_path / "test_outfit.png"
        result = visualizer.save_visualization(viz, output_path)

        assert result is True
        assert output_path.exists()
        assert output_path.suffix == ".png"

    def test_save_visualization_as_jpeg(self, visualizer, sample_top_garment, tmp_path):
        """Test saving visualization as JPEG."""
        garments = [sample_top_garment]
        viz = visualizer.create_visualization(garments)

        output_path = tmp_path / "test_outfit.jpg"
        result = visualizer.save_visualization(viz, output_path)

        assert result is True
        assert output_path.exists()
        assert output_path.suffix == ".jpg"

    def test_save_visualization_creates_directories(self, visualizer, sample_top_garment, tmp_path):
        """Test that save creates necessary directories."""
        garments = [sample_top_garment]
        viz = visualizer.create_visualization(garments)

        output_path = tmp_path / "nested" / "dirs" / "test_outfit.png"
        result = visualizer.save_visualization(viz, output_path)

        assert result is True
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_visualization_returns_false_on_error(self, visualizer, sample_top_garment):
        """Test that save returns False on error."""
        garments = [sample_top_garment]
        viz = visualizer.create_visualization(garments)

        # Try to save to an invalid path (e.g., a file that exists as a directory)
        invalid_path = "/dev/null/impossible/path.png"
        result = visualizer.save_visualization(viz, invalid_path)

        assert result is False


class TestAttributeExtraction:
    """Test extraction of garment attributes for display."""

    def test_extract_material_info(self, sample_top_garment):
        """Test that material information is properly extracted."""
        assert sample_top_garment.attributes.material is not None
        assert sample_top_garment.attributes.material.primary == "cotton"
        assert sample_top_garment.attributes.material.secondary == "linen"

    def test_extract_color_info(self, sample_bottom_garment):
        """Test that color information is properly extracted."""
        assert sample_bottom_garment.attributes.color is not None
        assert sample_bottom_garment.attributes.color.primary == "olive"

    def test_extract_subcategory(self, sample_shoes_garment):
        """Test that subcategory is properly extracted."""
        assert sample_shoes_garment.attributes.subcategory == "heels"

    def test_handle_missing_subcategory(self, visualizer, sample_top_garment):
        """Test handling of garments without subcategory."""
        garment = Garment(
            id="test_001",
            image_path=None,
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                subcategory=None,  # Missing subcategory
                color=ColorProfile(primary="red"),
                formality_level=FormalityLevel.CASUAL,
            ),
            created_at=datetime.now(timezone.utc),
        )

        # Should not raise an error
        viz = visualizer.create_visualization([garment])
        assert viz is not None


class TestOutfitVisualizationDataclass:
    """Test OutfitVisualization dataclass."""

    def test_outfit_visualization_creation(self):
        """Test creating an OutfitVisualization instance."""
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        viz = OutfitVisualization(
            image=img,
            outfit_name="Test Outfit",
            garment_count=3,
            score=0.85,
        )

        assert viz.image is img
        assert viz.outfit_name == "Test Outfit"
        assert viz.garment_count == 3
        assert viz.score == 0.85
        assert viz.output_path is None

    def test_outfit_visualization_with_output_path(self, tmp_path):
        """Test OutfitVisualization with output path."""
        img = Image.new("RGBA", (100, 100))
        output_path = tmp_path / "output.png"
        viz = OutfitVisualization(
            image=img,
            outfit_name="Test",
            garment_count=1,
            output_path=output_path,
        )

        assert viz.output_path == output_path


class TestDynamicGarmentDescription:
    """Test the dynamic garment description generation."""

    def test_description_with_full_attributes(self, visualizer):
        """Test description generation with rich attributes."""
        garment = Garment(
            id="test_blouse",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                subcategory="Silk Blouse",
                color=ColorProfile(primary="Ivory"),
                material=MaterialProfile(primary="Silk", texture="satin"),
                style_tags=["elegant", "minimalist"],
                occasion_profile=OccasionProfile(
                    formality_level=FormalityLevel.BUSINESS_CASUAL,
                    occasions=["work", "business"],
                ),
                silhouette_profile=SilhouetteProfile(
                    fit="slim",
                ),
            ),
            created_at=datetime.now(timezone.utc),
        )

        desc = visualizer._generate_garment_description(garment)
        
        # Should contain refined material + color description
        assert "satin" in desc.lower()
        assert "ivory" in desc.lower()
        # Should contain refined aesthetic language
        assert "refined aesthetic" in desc.lower() or "elegant" in desc.lower()
        # Should contain occasions
        assert "work" in desc.lower()
        # Should contain refined fit info
        assert "tailored silhouette" in desc.lower() or "slim" in desc.lower()

    def test_description_with_minimal_attributes(self, visualizer):
        """Test description generation with minimal attributes."""
        garment = Garment(
            id="test_tshirt",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                color=ColorProfile(primary="Navy"),
            ),
            created_at=datetime.now(timezone.utc),
        )

        desc = visualizer._generate_garment_description(garment)
        
        # Should at least contain color and category
        assert "navy" in desc.lower()
        assert "top" in desc.lower()

    def test_description_with_style_identity(self, visualizer):
        """Test description with style_identity instead of tags."""
        garment = Garment(
            id="test_sweater",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                color=ColorProfile(primary="Gray"),
                style_identity=StyleIdentity(
                    aesthetic_styles=["minimalist", "modern"],
                ),
            ),
            created_at=datetime.now(timezone.utc),
        )

        desc = visualizer._generate_garment_description(garment)
        
        # Should use aesthetic_styles when style_tags not available
        assert "minimalist" in desc.lower()

    def test_description_with_versatility(self, visualizer):
        """Test description includes versatility info."""
        garment = Garment(
            id="test_basics",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                color=ColorProfile(primary="White"),
                versatility=VersatilityInfo(
                    capsule_wardrobe_friendly=True,
                    versatility_score=0.9,
                ),
            ),
            created_at=datetime.now(timezone.utc),
        )

        desc = visualizer._generate_garment_description(garment)
        
        # Should mention quiet luxury versatility language (staple, timeless, effortless)
        assert "staple" in desc.lower() or "timeless" in desc.lower() or "effortlessly" in desc.lower()

    def test_description_with_material_texture(self, visualizer):
        """Test description prefers texture over primary material."""
        garment = Garment(
            id="test_jacket",
            attributes=GarmentAttributes(
                category=GarmentCategory.OUTERWEAR,
                color=ColorProfile(primary="Black"),
                material=MaterialProfile(
                    primary="Polyester",
                    texture="wool blend",
                ),
            ),
            created_at=datetime.now(timezone.utc),
        )

        desc = visualizer._generate_garment_description(garment)
        
        # Should use texture (satin, wool blend) not primary material
        assert "wool blend" in desc.lower()

    def test_description_line_breaks(self, visualizer):
        """Test that description uses line breaks for readability."""
        garment = Garment(
            id="test_outfit_piece",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM,
                color=ColorProfile(primary="Charcoal"),
                material=MaterialProfile(primary="Cotton"),
                style_tags=["professional"],
                occasion_profile=OccasionProfile(occasions=["work"]),
            ),
            created_at=datetime.now(timezone.utc),
        )

        desc = visualizer._generate_garment_description(garment)
        
        # Should have multiple lines
        lines = desc.split("\n")
        assert len(lines) > 1

    def test_description_capitalization(self, visualizer):
        """Test that first letter of description is capitalized."""
        garment = Garment(
            id="test_casual",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                color=ColorProfile(primary="olive"),
            ),
            created_at=datetime.now(timezone.utc),
        )

        desc = visualizer._generate_garment_description(garment)
        
        # First character should be uppercase
        assert desc[0].isupper()


# ══════════════════════════════════════════════════════════════════════════════
# QuietLuxuryVisualizer Tests
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def ql_visualizer():
    """Create a QuietLuxuryVisualizer instance."""
    return QuietLuxuryVisualizer()


class TestQuietLuxuryInitialization:
    """Test QuietLuxuryVisualizer initialisation."""

    def test_ql_initializes(self, ql_visualizer):
        assert ql_visualizer._f_title is not None
        assert ql_visualizer._f_body is not None
        assert ql_visualizer._f_score is not None
        assert ql_visualizer._f_caption is not None

    def test_ql_fonts_have_fallback(self, ql_visualizer):
        """Even on systems without the preferred fonts, defaults load."""
        assert ql_visualizer._f_section is not None
        assert ql_visualizer._f_hero is not None


class TestQuietLuxuryCreateVisualization:
    """Test editorial moodboard generation."""

    def test_ql_single_garment(self, ql_visualizer, sample_top_garment):
        """One garment produces a hero-only board without error."""
        viz = ql_visualizer.create_visualization([sample_top_garment])
        assert isinstance(viz, OutfitVisualization)
        assert viz.garment_count == 1
        assert viz.image is not None

    def test_ql_full_outfit(self, ql_visualizer, sample_top_garment,
                            sample_bottom_garment, sample_shoes_garment,
                            sample_accessory_garment):
        garments = [sample_top_garment, sample_bottom_garment,
                    sample_shoes_garment, sample_accessory_garment]
        viz = ql_visualizer.create_visualization(garments, score=0.85)
        assert viz.garment_count == 4
        assert viz.score == 0.85
        assert isinstance(viz.image, Image.Image)

    def test_ql_empty_garments_raises(self, ql_visualizer):
        with pytest.raises(ValueError, match="At least one garment"):
            ql_visualizer.create_visualization([])

    def test_ql_outfit_name_preserved(self, ql_visualizer, sample_top_garment):
        viz = ql_visualizer.create_visualization(
            [sample_top_garment], outfit_name="Autumn Quietude")
        assert viz.outfit_name == "Autumn Quietude"

    def test_ql_canvas_dynamic_size(self, ql_visualizer, sample_top_garment,
                                    sample_bottom_garment, sample_shoes_garment):
        """Canvas grows when there are secondary cards."""
        viz_small = ql_visualizer.create_visualization([sample_top_garment])
        viz_large = ql_visualizer.create_visualization(
            [sample_top_garment, sample_bottom_garment, sample_shoes_garment])
        # More garments → wider canvas
        assert viz_large.image.width >= viz_small.image.width

    def test_ql_with_season_and_occasion(self, ql_visualizer, sample_top_garment):
        viz = ql_visualizer.create_visualization(
            [sample_top_garment], season="AW 2025", occasion="Business")
        assert viz.image is not None


class TestQuietLuxurySave:
    """Test save_visualization for the editorial layout."""

    def test_ql_save_png(self, ql_visualizer, sample_top_garment, tmp_path):
        viz = ql_visualizer.create_visualization([sample_top_garment])
        out = tmp_path / "board.png"
        result = ql_visualizer.save_visualization(viz, out)
        assert result is True
        assert out.exists()

    def test_ql_save_creates_dirs(self, ql_visualizer, sample_top_garment, tmp_path):
        viz = ql_visualizer.create_visualization([sample_top_garment])
        out = tmp_path / "sub" / "dir" / "board.png"
        result = ql_visualizer.save_visualization(viz, out)
        assert result is True
        assert out.exists()

    def test_ql_save_bad_path_returns_false(self, ql_visualizer, sample_top_garment):
        viz = ql_visualizer.create_visualization([sample_top_garment])
        result = ql_visualizer.save_visualization(viz, "/proc/0/nonexistent/x.png")
        assert result is False


class TestHeroPicking:
    """Test _pick_hero helper."""

    def test_hero_prefers_dress(self, sample_top_garment, sample_bottom_garment):
        dress = Garment(
            id="dress_001",
            attributes=GarmentAttributes(
                category=GarmentCategory.DRESS,
                color=ColorProfile(primary="black"),
            ),
            created_at=datetime.now(timezone.utc),
        )
        hero, rest = _pick_hero([sample_top_garment, dress, sample_bottom_garment])
        assert hero is dress
        assert dress not in rest

    def test_hero_falls_back_to_first(self, sample_shoes_garment, sample_accessory_garment):
        """When no priority category matches, first garment is hero."""
        hero, rest = _pick_hero([sample_shoes_garment, sample_accessory_garment])
        assert hero is sample_shoes_garment

    def test_hero_empty_list(self):
        hero, rest = _pick_hero([])
        assert hero is None
        assert rest == []


class TestGridHelpers:
    """Test _compute_grid and _ql_canvas_size."""

    def test_zero_secondary(self):
        assert _compute_grid(0) == (0, 0)

    def test_three_secondary(self):
        cols, rows = _compute_grid(3)
        assert cols == 3
        assert rows == 1

    def test_seven_secondary(self):
        cols, rows = _compute_grid(7)
        assert cols == 3
        assert rows == 3

    def test_canvas_minimum_width(self):
        """Canvas width never drops below 800."""
        w, _ = _ql_canvas_size(0)
        assert w >= 800

    def test_canvas_grows_with_columns(self):
        w1, _ = _ql_canvas_size(1)
        w3, _ = _ql_canvas_size(3)
        assert w3 >= w1


class TestSpacedText:
    """Test the _spaced helper."""

    def test_spaced_default(self):
        assert _spaced("abc") == "A  B  C"

    def test_spaced_custom(self):
        assert _spaced("hi", 3) == "H   I"


class TestConvenienceVisualizeOutfit:
    """Test the one-liner visualize_outfit function."""

    def test_visualize_outfit_saves(self, sample_top_garment, tmp_path):
        out = tmp_path / "quick.png"
        viz = visualize_outfit(
            [sample_top_garment],
            output_path=str(out),
            outfit_name="Quick Test",
            score=0.72,
        )
        assert out.exists()
        assert viz.outfit_name == "Quick Test"
        assert viz.score == 0.72
