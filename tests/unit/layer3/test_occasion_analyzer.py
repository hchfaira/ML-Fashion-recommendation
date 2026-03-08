"""
Tests for Layer 3: Context Engine - Occasion Analyzer

Tests cover:
- Occasion scoring for outfits
- Occasion requirements retrieval
- Formality matching for occasions
"""
import pytest

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory,
    ColorInfo, FormalityLevel, Season, Occasion
)
from src.layer3_context import OccasionAnalyzer


# ============== Fixtures ==============

@pytest.fixture
def analyzer():
    """Create an OccasionAnalyzer instance."""
    return OccasionAnalyzer()


@pytest.fixture
def casual_outfit(garment_factory):
    """Create a casual outfit for testing."""
    return [
        garment_factory.create_basic_top(
            color="white",
            formality=FormalityLevel.CASUAL
        ),
        garment_factory.create_basic_bottom(
            color="blue",
            formality=FormalityLevel.CASUAL
        )
    ]


@pytest.fixture
def business_outfit(garment_factory):
    """Create a business outfit for testing."""
    return [
        garment_factory.create_dress_shirt(
            color="white"
        ),
        garment_factory.create_dress_pants(
            color="navy"
        )
    ]


# ============== Test Classes ==============

@pytest.mark.unit
class TestOccasionAnalyzerInit:
    """Tests for OccasionAnalyzer initialization."""
    
    def test_init_creates_instance(self, analyzer):
        """Test that OccasionAnalyzer initializes correctly."""
        assert analyzer is not None
        assert isinstance(analyzer, OccasionAnalyzer)


@pytest.mark.unit
class TestOccasionScoring:
    """Tests for occasion-based outfit scoring."""
    
    def test_casual_outfit_scores_well_for_casual(self, analyzer, casual_outfit):
        """Test that casual outfit scores well for casual occasion."""
        score = analyzer.score_for_occasion(casual_outfit, Occasion.CASUAL)
        assert score >= 0.7
    
    def test_casual_outfit_scores_poorly_for_formal(self, analyzer, casual_outfit):
        """Test that casual outfit scores poorly for formal occasion."""
        score = analyzer.score_for_occasion(casual_outfit, Occasion.FORMAL)
        assert score < 0.6
    
    def test_business_outfit_scores_well_for_business(self, analyzer, business_outfit):
        """Test that business outfit scores well for business occasion."""
        score = analyzer.score_for_occasion(business_outfit, Occasion.BUSINESS)
        assert score >= 0.7
    
    def test_score_is_normalized(self, analyzer, casual_outfit):
        """Test that scores are normalized between 0 and 1."""
        for occasion in [Occasion.CASUAL, Occasion.BUSINESS, Occasion.FORMAL]:
            score = analyzer.score_for_occasion(casual_outfit, occasion)
            assert 0.0 <= score <= 1.0


@pytest.mark.unit
class TestOccasionRequirements:
    """Tests for occasion requirements retrieval."""
    
    def test_get_occasion_requirements_business(self, analyzer):
        """Test getting requirements for business occasion."""
        reqs = analyzer.get_occasion_requirements(Occasion.BUSINESS)
        
        assert "target_formality" in reqs
        assert "preferred_styles" in reqs
        assert reqs["target_formality"] == "business"
    
    def test_get_occasion_requirements_casual(self, analyzer):
        """Test getting requirements for casual occasion."""
        reqs = analyzer.get_occasion_requirements(Occasion.CASUAL)
        
        assert "target_formality" in reqs
        assert reqs["target_formality"] == "casual"
    
    def test_get_occasion_requirements_formal(self, analyzer):
        """Test getting requirements for formal occasion."""
        reqs = analyzer.get_occasion_requirements(Occasion.FORMAL)
        
        assert "target_formality" in reqs
        assert reqs["target_formality"] == "formal"
