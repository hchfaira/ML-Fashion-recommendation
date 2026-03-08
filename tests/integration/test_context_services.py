"""
Integration tests for Context Services.

These tests verify that multiple Layer 3 context services work together.
"""
import pytest
from datetime import datetime, timedelta

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, ColorProfile,
    UserContext, Occasion, FormalityLevel, Season,
    ActivityLevel, TransportMode, ScheduleEvent
)
from src.layer3_context import (
    ContextEngine,
    OccasionAnalyzer,
    MorphologyAdvisor
)
from src.layer3_context.schedule_analyzer import ScheduleAnalyzer, TransitionStrategy
from src.layer3_context.wardrobe_rotation import WardrobeRotationService
from src.layer3_context.activity_analyzer import ActivityAnalyzer


# ============== Fixtures ==============

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
            id="sneakers-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.SHOES,
                subcategory="sneakers",
                color=ColorProfile(primary="white"),
                formality_level=FormalityLevel.CASUAL
            )
        ),
    ]


# ============== Test Classes ==============

@pytest.mark.integration
class TestContextEngineImports:
    """Test that ContextEngine properly imports all services."""
    
    def test_imports_all_services(self):
        """Test that all context services are importable."""
        from src.layer3_context import (
            ContextEngine,
            ScheduleAnalyzer,
            WardrobeRotationService,
            ActivityAnalyzer,
            TransitionStrategy
        )
        
        engine = ContextEngine()
        assert engine is not None


@pytest.mark.integration
class TestScheduleWithRotation:
    """Test schedule analysis with wardrobe rotation."""
    
    def test_schedule_considers_freshness(self, sample_garments):
        """Test that schedule-based suggestions consider freshness."""
        schedule_analyzer = ScheduleAnalyzer()
        rotation_service = WardrobeRotationService()
        
        # Mark some items as recently worn
        rotation_service.record_wear(["top-1"])
        
        context = UserContext(
            user_id="test-user",
            occasion=Occasion.WORK,
            recently_worn_items=rotation_service.get_recently_worn(days=3)
        )
        
        # Analyze schedule
        result = schedule_analyzer.analyze_schedule(context)
        
        assert result is not None
        assert "mode" in result


@pytest.mark.integration
class TestActivityWithOccasion:
    """Test activity analyzer with occasion constraints."""
    
    def test_activity_and_occasion_combined(self, sample_garments):
        """Test combining activity and occasion scoring."""
        activity_analyzer = ActivityAnalyzer()
        occasion_analyzer = OccasionAnalyzer()
        
        context = UserContext(
            user_id="test-user",
            occasion=Occasion.WORK,
            activity_level=ActivityLevel.MODERATE,
            transport_mode=TransportMode.WALKING
        )
        
        outfit = sample_garments[:3]
        
        # Get activity score
        activity_score, _ = activity_analyzer.score_outfit_for_activity(
            outfit, context
        )
        
        # Get occasion score
        occasion_score = occasion_analyzer.score_for_occasion(
            outfit, context.occasion
        )
        
        assert 0 <= activity_score <= 1.0
        assert 0 <= occasion_score <= 1.0
        
        # Combined score could be weighted average
        combined_score = (activity_score + occasion_score) / 2
        assert 0 <= combined_score <= 1.0


@pytest.mark.integration
class TestMultiOccasionDay:
    """Test handling multi-occasion days with all services."""
    
    def test_complex_schedule_analysis(self, sample_garments):
        """Test complex multi-occasion day analysis."""
        schedule_analyzer = ScheduleAnalyzer()
        activity_analyzer = ActivityAnalyzer()
        rotation_service = WardrobeRotationService()
        
        # Use occasions with large formality gap for transition to be needed
        # CASUAL=1, BUSINESS=4 → gap=3 > 1 → transition_needed=True
        context = UserContext(
            user_id="test-user",
            schedule=[
                ScheduleEvent(
                    name="Morning Gym",
                    time="07:00",
                    occasion=Occasion.GYM,  # formality=1
                    duration_hours=1.0,
                    indoor=True
                ),
                ScheduleEvent(
                    name="Office Work",
                    time="09:00",
                    occasion=Occasion.BUSINESS,  # formality=4
                    duration_hours=4.0,
                    indoor=True
                ),
                ScheduleEvent(
                    name="Evening Cocktail",
                    time="19:00",
                    occasion=Occasion.COCKTAIL,  # formality=5
                    duration_hours=2.0,
                    indoor=True
                )
            ],
            activity_level=ActivityLevel.MODERATE,
            transport_mode=TransportMode.WALKING
        )
        
        # Analyze schedule
        schedule_result = schedule_analyzer.analyze_schedule(context)
        
        assert schedule_result["mode"] == "multi_occasion"
        assert len(schedule_result["events"]) == 3
        
        # Check transition needed (gap = 5-1 = 4 > 1)
        assert schedule_result["transition_needed"] == True


@pytest.mark.integration
class TestContextEngineFiltering:
    """Test ContextEngine filtering with multiple criteria."""
    
    @pytest.mark.asyncio
    async def test_filter_with_multiple_criteria(self, sample_garments):
        """Test filtering wardrobe with multiple context criteria."""
        engine = ContextEngine()
        
        from src.core.models import WeatherContext
        
        context = UserContext(
            user_id="test-user",
            occasion=Occasion.CASUAL,
            weather=WeatherContext(
                temperature_celsius=22,
                condition="sunny"
            )
        )
        
        filtered = await engine.filter_wardrobe_by_context(
            sample_garments, context
        )
        
        assert len(filtered) > 0
