"""
Tests for Color Harmony Analyzer
Layer 2 Style - Color Analysis

Tests cover:
- Color pair scoring
- Outfit color harmony analysis
- Complementary color detection
- Neutral color handling
"""
import pytest
from unittest.mock import MagicMock

from src.layer2_style.color_harmony import ColorHarmonyAnalyzer
from src.core.models import Garment, GarmentAttributes, GarmentCategory, ColorInfo


# ============== Fixtures ==============

@pytest.fixture
def analyzer():
    """Create a ColorHarmonyAnalyzer instance."""
    return ColorHarmonyAnalyzer()


@pytest.fixture
def white_garment():
    """Create a white garment."""
    return Garment(
        id="white_top",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorInfo(primary="white", hex_codes=[])
        )
    )


@pytest.fixture
def black_garment():
    """Create a black garment."""
    return Garment(
        id="black_bottom",
        attributes=GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            color=ColorInfo(primary="black", hex_codes=[])
        )
    )


@pytest.fixture
def blue_garment():
    """Create a blue garment."""
    return Garment(
        id="blue_item",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorInfo(primary="blue", hex_codes=[])
        )
    )


@pytest.fixture
def red_garment():
    """Create a red garment."""
    return Garment(
        id="red_item",
        attributes=GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            color=ColorInfo(primary="red", hex_codes=[])
        )
    )


@pytest.fixture
def orange_garment():
    """Create an orange garment."""
    return Garment(
        id="orange_item",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorInfo(primary="orange", hex_codes=[])
        )
    )


# ============== Test Classes ==============

class TestColorHarmonyAnalyzerInit:
    """Tests for ColorHarmonyAnalyzer initialization."""
    
    def test_init_creates_color_map(self, analyzer):
        """Test that initialization creates color HSL map."""
        assert hasattr(analyzer, 'color_hsl_map')
        assert len(analyzer.color_hsl_map) > 0
    
    def test_init_defines_neutrals(self, analyzer):
        """Test that neutrals set is defined."""
        assert hasattr(analyzer, 'neutrals')
        assert "black" in analyzer.neutrals
        assert "white" in analyzer.neutrals
        assert "gray" in analyzer.neutrals
        assert "beige" in analyzer.neutrals


class TestColorPairScoring:
    """Tests for scoring color pairs."""
    
    def test_same_color_monochromatic(self, analyzer):
        """Test that same colors score high (monochromatic)."""
        color1 = ColorInfo(primary="blue", hex_codes=[])
        color2 = ColorInfo(primary="blue", hex_codes=[])
        
        score = analyzer.score_color_pair(color1, color2)
        
        assert score >= 0.8
    
    def test_both_neutrals_high_score(self, analyzer):
        """Test that two neutrals score high."""
        color1 = ColorInfo(primary="black", hex_codes=[])
        color2 = ColorInfo(primary="white", hex_codes=[])
        
        score = analyzer.score_color_pair(color1, color2)
        
        assert score >= 0.85
    
    def test_one_neutral_scores_well(self, analyzer):
        """Test that one neutral + any color scores well."""
        color1 = ColorInfo(primary="black", hex_codes=[])
        color2 = ColorInfo(primary="red", hex_codes=[])
        
        score = analyzer.score_color_pair(color1, color2)
        
        assert score >= 0.8
    
    def test_score_is_normalized(self, analyzer):
        """Test that score is between 0 and 1."""
        color1 = ColorInfo(primary="purple", hex_codes=[])
        color2 = ColorInfo(primary="orange", hex_codes=[])
        
        score = analyzer.score_color_pair(color1, color2)
        
        assert 0.0 <= score <= 1.0


class TestOutfitColorAnalysis:
    """Tests for analyzing outfit color harmony."""
    
    def test_single_item_returns_default_score(self, analyzer, white_garment):
        """Test that single item returns default score."""
        score = analyzer.analyze_outfit_colors([white_garment])
        
        assert score == 0.7  # Default for single item
    
    def test_all_neutrals_scores_high(self, analyzer, white_garment, black_garment):
        """Test that all-neutral outfit scores high."""
        score = analyzer.analyze_outfit_colors([white_garment, black_garment])
        
        assert score >= 0.85
    
    def test_neutrals_with_accent_scores_high(
        self, analyzer, white_garment, black_garment, blue_garment
    ):
        """Test that neutrals + 1 accent color scores high."""
        score = analyzer.analyze_outfit_colors([
            white_garment, black_garment, blue_garment
        ])
        
        assert score >= 0.8
    
    def test_empty_outfit_handled(self, analyzer):
        """Test that empty outfit is handled gracefully."""
        score = analyzer.analyze_outfit_colors([])
        
        assert 0.0 <= score <= 1.0
    
    def test_secondary_colors_considered(self, analyzer):
        """Test that secondary colors are included in analysis."""
        garment = Garment(
            id="multi_color",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                color=ColorInfo(primary="blue", secondary="white", hex_codes=[])
            )
        )
        
        # Should not raise and should handle secondary
        score = analyzer.analyze_outfit_colors([garment])
        assert 0.0 <= score <= 1.0


class TestComplementaryColors:
    """Tests for complementary color detection."""
    
    def test_neutral_returns_any_works_message(self, analyzer):
        """Test that neutral colors return 'any works' message."""
        result = analyzer.get_complementary_colors("white")
        
        assert "any color works with neutrals" in result
    
    def test_non_neutral_returns_colors(self, analyzer):
        """Test that non-neutral colors return complementary options."""
        result = analyzer.get_complementary_colors("blue")
        
        assert len(result) > 0
        # Should include neutrals as safe options
        assert any(c in ["white", "black", "gray"] for c in result)
    
    def test_unknown_color_returns_defaults(self, analyzer):
        """Test that unknown colors return safe defaults."""
        result = analyzer.get_complementary_colors("xyzabc123")  # Fake color
        
        # Should return safe defaults
        assert any(c in ["navy", "white", "black"] for c in result)


class TestColorCategories:
    """Tests for color category recognition."""
    
    def test_black_is_neutral(self, analyzer):
        """Test that black is recognized as neutral."""
        assert "black" in analyzer.neutrals
    
    def test_white_is_neutral(self, analyzer):
        """Test that white is recognized as neutral."""
        assert "white" in analyzer.neutrals
    
    def test_navy_is_neutral(self, analyzer):
        """Test that navy is recognized as neutral."""
        assert "navy" in analyzer.neutrals
    
    def test_beige_is_neutral(self, analyzer):
        """Test that beige is recognized as neutral."""
        assert "beige" in analyzer.neutrals
    
    def test_taupe_is_neutral(self, analyzer):
        """Test that taupe is recognized as neutral."""
        assert "taupe" in analyzer.neutrals
