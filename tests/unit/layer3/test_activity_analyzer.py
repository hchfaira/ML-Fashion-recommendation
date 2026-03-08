"""
Tests for Layer 3: Activity Analyzer

Tests cover:
- Activity context analysis
- Garment scoring for activities
- Transport mode considerations
- Outfit scoring for activities
- Activity recommendations
"""
import pytest

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, ColorProfile,
    UserContext, Occasion, FormalityLevel, Season,
    ActivityLevel, TransportMode
)
from src.layer3_context.activity_analyzer import ActivityAnalyzer


# ============== Fixtures ==============

@pytest.fixture
def analyzer():
    """Create an ActivityAnalyzer instance."""
    return ActivityAnalyzer()


@pytest.fixture
def sample_garments():
    """Create sample garments for testing."""
    return [
        Garment(
            id="top-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                color=ColorProfile(primary="white"),
                formality_level=FormalityLevel.CASUAL,
                season_suitable=[Season.SPRING, Season.SUMMER]
            )
        ),
        Garment(
            id="blazer-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.OUTERWEAR,
                color=ColorProfile(primary="navy"),
                formality_level=FormalityLevel.BUSINESS_CASUAL,
                season_suitable=[Season.FALL, Season.SPRING]
            )
        ),
        Garment(
            id="jeans-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM,
                color=ColorProfile(primary="blue"),
                formality_level=FormalityLevel.CASUAL,
                season_suitable=[Season.SPRING, Season.FALL, Season.WINTER]
            )
        ),
        Garment(
            id="heels-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.SHOES,
                subcategory="heels",
                product_type="high heels",
                color=ColorProfile(primary="black"),
                formality_level=FormalityLevel.FORMAL
            )
        ),
        Garment(
            id="sneakers-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.SHOES,
                subcategory="sneakers",
                product_type="casual sneakers",
                color=ColorProfile(primary="white"),
                formality_level=FormalityLevel.CASUAL
            )
        ),
        Garment(
            id="dress-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.DRESS,
                color=ColorProfile(primary="floral"),
                fit="flowy",
                formality_level=FormalityLevel.SMART_CASUAL,
                season_suitable=[Season.SUMMER]
            )
        ),
        Garment(
            id="activewear-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM,
                subcategory="activewear",
                color=ColorProfile(primary="black"),
                formality_level=FormalityLevel.VERY_CASUAL
            )
        ),
    ]


@pytest.fixture
def basic_context():
    """Create a basic user context."""
    return UserContext(
        user_id="test-user",
        occasion=Occasion.WORK
    )


# ============== Test Classes ==============

@pytest.mark.unit
class TestActivityAnalyzerInit:
    """Tests for ActivityAnalyzer initialization."""
    
    def test_init_creates_instance(self, analyzer):
        """Test that ActivityAnalyzer initializes correctly."""
        assert analyzer is not None
        assert isinstance(analyzer, ActivityAnalyzer)


@pytest.mark.unit
class TestActivityContextAnalysis:
    """Tests for activity context analysis."""
    
    def test_analyze_activity_context(self, analyzer):
        """Test basic activity context analysis."""
        context = UserContext(
            user_id="test-user",
            activity_level=ActivityLevel.MODERATE,
            transport_mode=TransportMode.WALKING,
            duration_hours=8.0
        )
        
        result = analyzer.analyze_activity_context(context)
        
        assert result["activity_level"] == ActivityLevel.MODERATE
        assert result["transport_mode"] == TransportMode.WALKING
        assert result["duration_hours"] == 8.0
        assert "requirements" in result


@pytest.mark.unit
class TestGarmentActivityScoring:
    """Tests for garment scoring based on activity."""
    
    def test_score_activewear_for_intense_activity(
        self, analyzer, sample_garments
    ):
        """Test activewear scores well for intense activity."""
        context = UserContext(
            user_id="test-user",
            activity_level=ActivityLevel.INTENSE
        )
        
        activewear = next(g for g in sample_garments if g.id == "activewear-1")
        
        score = analyzer.score_garment_for_activity(activewear, context)
        
        assert score >= 0.5
    
    def test_score_formal_for_sedentary(self, analyzer, sample_garments):
        """Test formal items score well for sedentary activity."""
        context = UserContext(
            user_id="test-user",
            activity_level=ActivityLevel.SEDENTARY
        )
        
        blazer = next(g for g in sample_garments if g.id == "blazer-1")
        heels = next(g for g in sample_garments if g.id == "heels-1")
        
        blazer_score = analyzer.score_garment_for_activity(blazer, context)
        heels_score = analyzer.score_garment_for_activity(heels, context)
        
        assert blazer_score >= 0.6
        assert heels_score >= 0.6


@pytest.mark.unit
class TestTransportModeScoring:
    """Tests for transport mode based scoring."""
    
    def test_walking_prefers_comfortable_footwear(
        self, analyzer, sample_garments
    ):
        """Test walking mode prefers comfortable footwear."""
        context = UserContext(
            user_id="test-user",
            transport_mode=TransportMode.WALKING
        )
        
        heels = next(g for g in sample_garments if g.id == "heels-1")
        sneakers = next(g for g in sample_garments if g.id == "sneakers-1")
        
        heels_score = analyzer.score_garment_for_activity(heels, context)
        sneakers_score = analyzer.score_garment_for_activity(sneakers, context)
        
        assert sneakers_score >= heels_score
    
    def test_cycling_penalizes_flowy(self, analyzer, sample_garments):
        """Test cycling mode penalizes flowy items."""
        context = UserContext(
            user_id="test-user",
            transport_mode=TransportMode.CYCLING
        )
        
        flowy_dress = next(g for g in sample_garments if g.id == "dress-1")
        
        score = analyzer.score_garment_for_activity(flowy_dress, context)
        
        assert score < 0.8


@pytest.mark.unit
class TestOutfitActivityScoring:
    """Tests for outfit scoring based on activity."""
    
    def test_score_outfit_for_activity(self, analyzer, sample_garments):
        """Test scoring entire outfit for activity."""
        context = UserContext(
            user_id="test-user",
            activity_level=ActivityLevel.MODERATE,
            duration_hours=6.0
        )
        
        outfit = sample_garments[:3]
        
        score, breakdown = analyzer.score_outfit_for_activity(outfit, context)
        
        assert 0 <= score <= 1.0
        assert "overall" in breakdown
        assert "garment_scores" in breakdown
        assert "requirements_met" in breakdown


@pytest.mark.unit
class TestActivityRecommendations:
    """Tests for activity-based recommendations."""
    
    def test_get_activity_recommendations(self, analyzer):
        """Test getting activity recommendations."""
        context = UserContext(
            user_id="test-user",
            activity_level=ActivityLevel.INTENSE,
            transport_mode=TransportMode.WALKING,
            duration_hours=12.0
        )
        
        recommendations = analyzer.get_activity_recommendations(context)
        
        assert len(recommendations) > 0


@pytest.mark.unit
class TestRequirementChecking:
    """Tests for requirement checking functionality."""
    
    def test_comfortable_footwear_requirement(
        self, analyzer, sample_garments
    ):
        """Test comfortable footwear requirement checking."""
        context = UserContext(
            user_id="test-user",
            transport_mode=TransportMode.WALKING
        )
        
        top = sample_garments[0]
        sneakers = next(g for g in sample_garments if g.id == "sneakers-1")
        outfit_with_sneakers = [top, sneakers]
        
        _, breakdown = analyzer.score_outfit_for_activity(
            outfit_with_sneakers, context
        )
        
        assert breakdown["requirements_met"]["comfortable_footwear"] == True
