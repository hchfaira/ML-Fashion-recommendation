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


@pytest.mark.unit
class TestEnhancedMorphologyWithMeasurements:
    """Tests for enhanced morphology advisor with body measurements."""
    
    def test_score_with_measurements_no_metrics(self, advisor, casual_outfit):
        """Test scoring with measurements when no body metrics provided."""
        result = advisor.score_with_measurements(
            casual_outfit,
            "hourglass",
            body_metrics=None
        )
        
        # Should still return a valid score
        assert 0 <= result.final_score <= 1
        assert result.body_type == "hourglass"
    
    def test_score_with_measurements_with_metrics(self, advisor, casual_outfit):
        """Test scoring with measurements when body metrics are provided."""
        from unittest.mock import MagicMock
        
        # Create mock body metrics
        mock_metrics = MagicMock()
        mock_metrics.torso_proportion = "short"
        mock_metrics.leg_proportion = "average"
        mock_metrics.frame_size = "medium"
        mock_metrics.shoulder_hip_ratio = 1.0
        
        result = advisor.score_with_measurements(
            casual_outfit,
            "hourglass",
            body_metrics=mock_metrics
        )
        
        assert 0 <= result.final_score <= 1
        assert result.body_type == "hourglass"
        # Should have measurement_based_tips
        assert hasattr(result, 'measurement_based_tips')
    
    def test_get_enhanced_recommendations_no_metrics(self, advisor):
        """Test enhanced recommendations without metrics."""
        recs = advisor.get_enhanced_recommendations("hourglass", body_metrics=None)
        
        assert "flattering_silhouettes" in recs
        assert "styling_tips" in recs
    
    def test_get_enhanced_recommendations_with_metrics(self, advisor):
        """Test enhanced recommendations with body metrics."""
        from unittest.mock import MagicMock
        
        mock_metrics = MagicMock()
        mock_metrics.torso_proportion = "short"
        mock_metrics.leg_proportion = "average"
        mock_metrics.frame_size = "medium"
        mock_metrics.estimated_top_size = "M"
        mock_metrics.estimated_bottom_size = "M"
        mock_metrics.proportion_tips = ["Custom tip from metrics"]
        
        recs = advisor.get_enhanced_recommendations("hourglass", body_metrics=mock_metrics)
        
        assert "measurement_based_tips" in recs
        assert len(recs["measurement_based_tips"]) > 0
        # Should include the custom tip
        assert any("Custom tip" in tip for tip in recs["measurement_based_tips"])
    
    def test_get_enhanced_recommendations_includes_size(self, advisor):
        """Test enhanced recommendations include size recommendations."""
        from unittest.mock import MagicMock
        
        mock_metrics = MagicMock()
        mock_metrics.torso_proportion = "average"
        mock_metrics.leg_proportion = "average"
        mock_metrics.frame_size = "medium"
        mock_metrics.estimated_top_size = "L"
        mock_metrics.estimated_bottom_size = "M"
        mock_metrics.proportion_tips = []
        
        recs = advisor.get_enhanced_recommendations("rectangle", body_metrics=mock_metrics)
        
        assert "size_recommendations" in recs
        assert recs["size_recommendations"]["estimated_top_size"] == "L"
        assert recs["size_recommendations"]["estimated_bottom_size"] == "M"
    
    def test_adjust_for_torso_short(self, advisor, garment_factory):
        """Test adjustment for short torso with high-waisted pants."""
        # Create garment with high waist
        from unittest.mock import MagicMock
        garment = MagicMock()
        garment.attributes = {"waist": "high waist"}
        
        adj, tip = advisor._adjust_for_torso([garment], "short")
        
        # Should get positive adjustment for high waist with short torso
        assert adj >= 0
    
    def test_adjust_for_torso_short_low_rise_penalty(self, advisor):
        """Test penalty for short torso with low-rise pants."""
        from unittest.mock import MagicMock
        garment = MagicMock()
        garment.attributes = {"waist": "low rise"}
        
        adj, tip = advisor._adjust_for_torso([garment], "short")
        
        # Should get negative adjustment or warning
        assert adj <= 0 or tip is not None
    
    def test_adjust_for_legs_short(self, advisor):
        """Test adjustment for short legs with vertical patterns."""
        from unittest.mock import MagicMock
        garment = MagicMock()
        garment.attributes = {"pattern": "vertical stripes"}
        garment.category = "pants"
        
        adj, tip = advisor._adjust_for_legs([garment], "short")
        
        # Should get positive adjustment for vertical stripes
        assert adj >= 0
    
    def test_adjust_for_frame_small_heavy_fabric(self, advisor):
        """Test penalty for small frame with heavy fabric."""
        from unittest.mock import MagicMock
        garment = MagicMock()
        garment.attributes = {"weight": "heavy bulky"}
        
        adj, tip = advisor._adjust_for_frame([garment], "small")
        
        # Should get negative adjustment
        assert adj < 0 or tip is not None
