"""
Tests for Pattern Mixing Scorer
Layer 2 Style - Pattern Combination Analysis

Tests cover:
- Pattern family classification
- Scale contrast rules
- 60/40 balance rule
- Pattern compatibility scoring
"""
import pytest

from src.layer2_style.pattern_mixing_scorer import (
    PatternMixingScorer, PatternFamily, PatternScale, PatternMixingResult
)
from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, 
    ColorInfo, PatternInfo
)


# ============== Fixtures ==============

@pytest.fixture
def scorer():
    """Create a PatternMixingScorer instance."""
    return PatternMixingScorer()


@pytest.fixture
def solid_white_top():
    """Create a solid white top."""
    return Garment(
        id="solid_top",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorInfo(primary="white", hex_codes=[]),
            pattern=PatternInfo(type="solid", scale="none")
        )
    )


@pytest.fixture
def solid_navy_bottom():
    """Create solid navy pants."""
    return Garment(
        id="solid_bottom",
        attributes=GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            color=ColorInfo(primary="navy", hex_codes=[]),
            pattern=PatternInfo(type="solid", scale="none")
        )
    )


@pytest.fixture
def striped_shirt():
    """Create a striped shirt with medium scale."""
    return Garment(
        id="striped_shirt",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorInfo(primary="blue", hex_codes=[]),
            pattern=PatternInfo(type="stripes", scale="medium")
        )
    )


@pytest.fixture
def small_check_shirt():
    """Create a small gingham check shirt."""
    return Garment(
        id="check_shirt",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorInfo(primary="blue", hex_codes=[]),
            pattern=PatternInfo(type="gingham", scale="small")
        )
    )


@pytest.fixture
def large_floral_dress():
    """Create a large floral dress."""
    return Garment(
        id="floral_dress",
        attributes=GarmentAttributes(
            category=GarmentCategory.DRESS,
            color=ColorInfo(primary="red", hex_codes=[]),
            pattern=PatternInfo(type="floral", scale="large")
        )
    )


@pytest.fixture
def leopard_shoes():
    """Create leopard print shoes."""
    return Garment(
        id="leopard_shoes",
        attributes=GarmentAttributes(
            category=GarmentCategory.SHOES,
            color=ColorInfo(primary="brown", hex_codes=[]),
            pattern=PatternInfo(type="leopard", scale="small")
        )
    )


# ============== Test Classes ==============

class TestPatternMixingScorerInit:
    """Tests for scorer initialization."""
    
    def test_init_successful(self, scorer):
        """Test that scorer initializes correctly."""
        assert scorer is not None
    
    def test_has_analyze_outfit_method(self, scorer):
        """Test that analyze_outfit method exists."""
        assert hasattr(scorer, 'analyze_outfit')


class TestPatternFamilyEnum:
    """Tests for PatternFamily enum."""
    
    def test_solid_exists(self):
        """Test that SOLID pattern family exists."""
        assert PatternFamily.SOLID.value == "solid"
    
    def test_stripes_exists(self):
        """Test that STRIPES pattern family exists."""
        assert PatternFamily.STRIPES.value == "stripes"
    
    def test_floral_exists(self):
        """Test that FLORAL pattern family exists."""
        assert PatternFamily.FLORAL.value == "floral"
    
    def test_animal_exists(self):
        """Test that ANIMAL pattern family exists."""
        assert PatternFamily.ANIMAL.value == "animal"
    
    def test_checks_exists(self):
        """Test that CHECKS pattern family exists."""
        assert PatternFamily.CHECKS.value == "checks"


class TestPatternScaleEnum:
    """Tests for PatternScale enum."""
    
    def test_micro_scale(self):
        """Test that MICRO scale exists."""
        assert PatternScale.MICRO.value == "micro"
    
    def test_small_scale(self):
        """Test that SMALL scale exists."""
        assert PatternScale.SMALL.value == "small"
    
    def test_medium_scale(self):
        """Test that MEDIUM scale exists."""
        assert PatternScale.MEDIUM.value == "medium"
    
    def test_large_scale(self):
        """Test that LARGE scale exists."""
        assert PatternScale.LARGE.value == "large"
    
    def test_oversized_scale(self):
        """Test that OVERSIZED scale exists."""
        assert PatternScale.OVERSIZED.value == "oversized"


class TestAllSolidsOutfit:
    """Tests for all-solid outfits."""
    
    def test_all_solids_is_valid(self, scorer, solid_white_top, solid_navy_bottom):
        """Test that all solid outfit is considered valid."""
        outfit = [solid_white_top, solid_navy_bottom]
        
        result = scorer.analyze_outfit(outfit)
        
        # All solids should score well (safe choice)
        assert result.overall_score >= 0.6


class TestMixedPatternOutfit:
    """Tests for mixed pattern outfits."""
    
    def test_one_pattern_one_solid(self, scorer, striped_shirt, solid_navy_bottom):
        """Test outfit with one pattern and one solid."""
        outfit = [striped_shirt, solid_navy_bottom]
        
        result = scorer.analyze_outfit(outfit)
        
        # Pattern + solid is a good combination
        assert result.overall_score >= 0.7
    
    def test_two_patterns_different_scales(
        self, scorer, striped_shirt, leopard_shoes, solid_navy_bottom
    ):
        """Test two patterns with different scales (should work)."""
        outfit = [striped_shirt, solid_navy_bottom, leopard_shoes]
        
        result = scorer.analyze_outfit(outfit)
        
        # Different scales can work together
        assert result.overall_score >= 0.5


class TestScaleContrastRule:
    """Tests for scale contrast scoring."""
    
    def test_different_scales_score_well(self, scorer):
        """Test that different pattern scales score well."""
        # Create patterns with different scales
        small_pattern = Garment(
            id="small",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                color=ColorInfo(primary="blue", hex_codes=[]),
                pattern=PatternInfo(type="stripes", scale="small")
            )
        )
        large_pattern = Garment(
            id="large",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM,
                color=ColorInfo(primary="blue", hex_codes=[]),
                pattern=PatternInfo(type="floral", scale="large")
            )
        )
        
        result = scorer.analyze_outfit([small_pattern, large_pattern])
        
        # Different scales create visual interest without clashing
        assert result.scale_score >= 3  # Out of 5 points


class TestBalanceRule:
    """Tests for 60/40 pattern balance."""
    
    def test_mostly_solids_with_pattern_accent(
        self, scorer, solid_white_top, solid_navy_bottom, leopard_shoes
    ):
        """Test mostly solids with one pattern accent."""
        outfit = [solid_white_top, solid_navy_bottom, leopard_shoes]
        
        result = scorer.analyze_outfit(outfit)
        
        # 2 solids, 1 pattern = good balance
        assert result.balance_score >= 3  # Out of 5 points
