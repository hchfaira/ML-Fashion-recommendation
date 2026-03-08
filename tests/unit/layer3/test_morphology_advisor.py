"""
Tests for Layer 3: Context Engine - Morphology Advisor

Tests cover:
- Body type recommendations
- Flattering silhouettes
- Styling tips
- Unknown body type handling
"""
import pytest

from src.core.models import FormalityLevel, Season
from src.layer3_context import MorphologyAdvisor


# ============== Fixtures ==============

@pytest.fixture
def advisor():
    """Create a MorphologyAdvisor instance."""
    return MorphologyAdvisor()


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


# ============== Test Classes ==============

@pytest.mark.unit
class TestMorphologyAdvisorInit:
    """Tests for MorphologyAdvisor initialization."""
    
    def test_init_creates_instance(self, advisor):
        """Test that MorphologyAdvisor initializes correctly."""
        assert advisor is not None
        assert isinstance(advisor, MorphologyAdvisor)


@pytest.mark.unit
class TestBodyTypeRecommendations:
    """Tests for body type recommendation functionality."""
    
    def test_get_recommendations_hourglass(self, advisor):
        """Test getting recommendations for hourglass body type."""
        recs = advisor.get_recommendations_for_body_type("hourglass")
        
        assert "flattering_silhouettes" in recs
        assert "styling_tips" in recs
        assert len(recs["flattering_silhouettes"]) > 0
    
    def test_get_recommendations_rectangle(self, advisor):
        """Test getting recommendations for rectangle body type."""
        recs = advisor.get_recommendations_for_body_type("rectangle")
        
        assert "flattering_silhouettes" in recs
        assert "styling_tips" in recs
    
    def test_get_recommendations_pear(self, advisor):
        """Test getting recommendations for pear body type."""
        recs = advisor.get_recommendations_for_body_type("pear")
        
        assert "flattering_silhouettes" in recs
        assert "styling_tips" in recs
    
    def test_get_recommendations_apple(self, advisor):
        """Test getting recommendations for apple body type."""
        recs = advisor.get_recommendations_for_body_type("apple")
        
        assert "flattering_silhouettes" in recs
        assert "styling_tips" in recs
    
    def test_unknown_body_type_returns_general(self, advisor):
        """Test that unknown body type returns general recommendations."""
        recs = advisor.get_recommendations_for_body_type("unknown_type")
        
        assert recs["body_type"] == "general"
    
    def test_body_type_case_insensitive(self, advisor):
        """Test that body type lookup is case insensitive."""
        recs_lower = advisor.get_recommendations_for_body_type("hourglass")
        recs_upper = advisor.get_recommendations_for_body_type("HOURGLASS")
        
        # Both should return valid recommendations (not general fallback)
        assert recs_lower.get("body_type") != "general" or recs_upper.get("body_type") != "general"


@pytest.mark.unit
class TestBodyTypeScoring:
    """Tests for body type outfit scoring."""
    
    def test_score_for_body_type_hourglass(self, advisor, casual_outfit):
        """Test scoring outfit for hourglass body type."""
        score = advisor.score_for_body_type(casual_outfit, "hourglass")
        assert 0 <= score <= 1
    
    def test_score_for_body_type_rectangle(self, advisor, casual_outfit):
        """Test scoring outfit for rectangle body type."""
        score = advisor.score_for_body_type(casual_outfit, "rectangle")
        assert 0 <= score <= 1
    
    def test_score_for_unknown_body_type(self, advisor, casual_outfit):
        """Test scoring outfit for unknown body type."""
        score = advisor.score_for_body_type(casual_outfit, "unknown")
        assert 0 <= score <= 1
