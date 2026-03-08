"""
Tests for Layer 3: Schedule Analyzer

Tests cover:
- Single occasion analysis
- Multi-occasion scheduling
- Transition strategy determination
- Formality range calculation
"""
import pytest
from datetime import datetime, timedelta

from src.core.models import (
    UserContext, Occasion, FormalityLevel,
    ScheduleEvent
)
from src.layer3_context.schedule_analyzer import ScheduleAnalyzer, TransitionStrategy


# ============== Fixtures ==============

@pytest.fixture
def analyzer():
    """Create a ScheduleAnalyzer instance."""
    return ScheduleAnalyzer()


@pytest.fixture
def single_occasion_context():
    """Create a context with single occasion."""
    return UserContext(
        user_id="test-user",
        occasion=Occasion.WORK
    )


@pytest.fixture
def multi_occasion_context():
    """Create a context with multiple scheduled events."""
    return UserContext(
        user_id="test-user",
        occasion=Occasion.WORK,
        schedule=[
            ScheduleEvent(
                name="Morning Meeting",
                time="09:00",
                occasion=Occasion.BUSINESS,
                duration_hours=2.0,
                indoor=True
            ),
            ScheduleEvent(
                name="Lunch Date",
                time="12:30",
                occasion=Occasion.DATE,
                duration_hours=1.5,
                indoor=True
            ),
            ScheduleEvent(
                name="Evening Cocktail",
                time="18:00",
                occasion=Occasion.COCKTAIL,
                duration_hours=3.0,
                indoor=True
            )
        ]
    )


@pytest.fixture
def similar_formality_context():
    """Create a context with similar formality events."""
    return UserContext(
        user_id="test-user",
        schedule=[
            ScheduleEvent(
                name="Office Work",
                time="09:00",
                occasion=Occasion.WORK,
                duration_hours=4.0,
                indoor=True
            ),
            ScheduleEvent(
                name="Business Lunch",
                time="12:00",
                occasion=Occasion.BUSINESS,
                duration_hours=2.0,
                indoor=True
            )
        ]
    )


# ============== Test Classes ==============

@pytest.mark.unit
class TestScheduleAnalyzerInit:
    """Tests for ScheduleAnalyzer initialization."""
    
    def test_init_creates_instance(self, analyzer):
        """Test that ScheduleAnalyzer initializes correctly."""
        assert analyzer is not None
        assert isinstance(analyzer, ScheduleAnalyzer)


@pytest.mark.unit
class TestSingleOccasionAnalysis:
    """Tests for single occasion schedule analysis."""
    
    def test_single_occasion_mode(self, analyzer, single_occasion_context):
        """Test analysis with no schedule returns single occasion mode."""
        result = analyzer.analyze_schedule(single_occasion_context)
        
        assert result["mode"] == "single_occasion"
    
    def test_single_occasion_primary_occasion(self, analyzer, single_occasion_context):
        """Test that primary occasion is correctly identified."""
        result = analyzer.analyze_schedule(single_occasion_context)
        
        assert result["primary_occasion"] == Occasion.WORK
    
    def test_single_occasion_no_transition(self, analyzer, single_occasion_context):
        """Test that no transition is needed for single occasion."""
        result = analyzer.analyze_schedule(single_occasion_context)
        
        assert result["transition_needed"] == False
    
    def test_single_occasion_strategy(self, analyzer, single_occasion_context):
        """Test strategy is single outfit for single occasion."""
        result = analyzer.analyze_schedule(single_occasion_context)
        
        assert result["strategy"] == TransitionStrategy.SINGLE_OUTFIT


@pytest.mark.unit
class TestMultiOccasionAnalysis:
    """Tests for multi-occasion schedule analysis."""
    
    def test_multi_occasion_mode(self, analyzer, multi_occasion_context):
        """Test analysis with schedule returns multi occasion mode."""
        result = analyzer.analyze_schedule(multi_occasion_context)
        
        assert result["mode"] == "multi_occasion"
    
    def test_multi_occasion_event_count(self, analyzer, multi_occasion_context):
        """Test that all events are analyzed."""
        result = analyzer.analyze_schedule(multi_occasion_context)
        
        assert len(result["events"]) == 3
    
    def test_multi_occasion_transition_needed(self, analyzer, multi_occasion_context):
        """Test that transition is needed for varied formality."""
        result = analyzer.analyze_schedule(multi_occasion_context)
        
        assert result["transition_needed"] == True
    
    def test_multi_occasion_formality_gap(self, analyzer, multi_occasion_context):
        """Test that formality gap is calculated."""
        result = analyzer.analyze_schedule(multi_occasion_context)
        
        assert result["formality_range"]["gap"] > 0


@pytest.mark.unit
class TestSimilarFormalityAnalysis:
    """Tests for similar formality occasion analysis."""
    
    def test_similar_formality_suggests_single_or_layers(
        self, analyzer, similar_formality_context
    ):
        """Test that similar formality events suggest single outfit or layers."""
        result = analyzer.analyze_schedule(similar_formality_context)
        
        assert result["strategy"] in [
            TransitionStrategy.SINGLE_OUTFIT,
            TransitionStrategy.SMART_LAYERS
        ]


@pytest.mark.unit
class TestTransitionStrategy:
    """Tests for TransitionStrategy enum."""
    
    def test_single_outfit_strategy_exists(self):
        """Test that SINGLE_OUTFIT strategy exists."""
        assert hasattr(TransitionStrategy, 'SINGLE_OUTFIT')
    
    def test_smart_layers_strategy_exists(self):
        """Test that SMART_LAYERS strategy exists."""
        assert hasattr(TransitionStrategy, 'SMART_LAYERS')
