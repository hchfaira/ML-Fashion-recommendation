"""
Tests for Layer 3: Context Engine

Tests cover:
- Wardrobe filtering by context
- Weather-based filtering
- Occasion-based filtering
- Context combinations
"""
import pytest

from src.core.models import (
    FormalityLevel, Season, UserContext, WeatherContext, Occasion
)
from src.layer3_context import ContextEngine


# ============== Fixtures ==============

@pytest.fixture
def engine():
    """Create a ContextEngine instance."""
    return ContextEngine()


@pytest.fixture
def casual_outfit(garment_factory):
    """Create a casual outfit for testing."""
    return [
        garment_factory.create_basic_top(
            garment_id="casual_top",
            color="white",
            formality=FormalityLevel.CASUAL
        ),
        garment_factory.create_basic_bottom(
            garment_id="casual_bottom",
            color="blue",
            formality=FormalityLevel.CASUAL
        )
    ]


@pytest.fixture
def business_outfit(garment_factory):
    """Create a business outfit for testing."""
    return [
        garment_factory.create_dress_shirt(
            garment_id="business_top",
            color="white"
        ),
        garment_factory.create_dress_pants(
            garment_id="business_bottom",
            color="navy"
        )
    ]


@pytest.fixture
def summer_context():
    """Create a summer weather context."""
    return UserContext(
        weather=WeatherContext(
            temperature_celsius=28,
            condition="sunny"
        )
    )


@pytest.fixture
def winter_context():
    """Create a winter weather context."""
    return UserContext(
        weather=WeatherContext(
            temperature_celsius=5,
            condition="cloudy"
        )
    )


@pytest.fixture
def business_context():
    """Create a business occasion context."""
    return UserContext(occasion=Occasion.BUSINESS)


@pytest.fixture
def casual_context():
    """Create a casual occasion context."""
    return UserContext(occasion=Occasion.CASUAL)


# ============== Test Classes ==============

@pytest.mark.unit
class TestContextEngineInit:
    """Tests for ContextEngine initialization."""
    
    def test_init_creates_instance(self, engine):
        """Test that ContextEngine initializes correctly."""
        assert engine is not None
        assert isinstance(engine, ContextEngine)


@pytest.mark.unit
class TestWeatherFiltering:
    """Tests for weather-based wardrobe filtering."""
    
    @pytest.mark.asyncio
    async def test_filter_by_summer_weather(
        self, engine, casual_outfit, business_outfit, summer_context
    ):
        """Test filtering wardrobe by summer weather."""
        all_items = casual_outfit + business_outfit
        filtered = await engine.filter_wardrobe_by_context(all_items, summer_context)
        
        # Summer items should be included
        summer_ids = [
            g.id for g in filtered 
            if Season.SUMMER in g.attributes.season_suitable
        ]
        assert len(summer_ids) > 0
    
    @pytest.mark.asyncio
    async def test_filter_by_winter_weather(
        self, engine, casual_outfit, business_outfit, winter_context
    ):
        """Test filtering wardrobe by winter weather."""
        all_items = casual_outfit + business_outfit
        filtered = await engine.filter_wardrobe_by_context(all_items, winter_context)
        
        # Winter items should be included
        winter_ids = [
            g.id for g in filtered 
            if Season.WINTER in g.attributes.season_suitable
        ]
        assert len(winter_ids) > 0


@pytest.mark.unit
class TestOccasionFiltering:
    """Tests for occasion-based wardrobe filtering."""
    
    @pytest.mark.asyncio
    async def test_filter_by_business_occasion(
        self, engine, casual_outfit, business_outfit, business_context
    ):
        """Test filtering wardrobe by business occasion."""
        all_items = casual_outfit + business_outfit
        filtered = await engine.filter_wardrobe_by_context(all_items, business_context)
        
        # Business items should be prioritized
        business_ids = [
            g.id for g in filtered 
            if g.attributes.formality_level in [
                FormalityLevel.BUSINESS, 
                FormalityLevel.BUSINESS_CASUAL
            ]
        ]
        assert len(business_ids) > 0
    
    @pytest.mark.asyncio
    async def test_filter_by_casual_occasion(
        self, engine, casual_outfit, business_outfit, casual_context
    ):
        """Test filtering wardrobe by casual occasion."""
        all_items = casual_outfit + business_outfit
        filtered = await engine.filter_wardrobe_by_context(all_items, casual_context)
        
        # Casual items should be prioritized
        casual_ids = [
            g.id for g in filtered 
            if g.attributes.formality_level == FormalityLevel.CASUAL
        ]
        assert len(casual_ids) > 0


@pytest.mark.unit
class TestCombinedFiltering:
    """Tests for combined context filtering."""
    
    @pytest.mark.asyncio
    async def test_filter_by_weather_and_occasion(
        self, engine, casual_outfit, business_outfit
    ):
        """Test filtering by both weather and occasion."""
        all_items = casual_outfit + business_outfit
        
        context = UserContext(
            weather=WeatherContext(
                temperature_celsius=20,
                condition="sunny"
            ),
            occasion=Occasion.CASUAL
        )
        
        filtered = await engine.filter_wardrobe_by_context(all_items, context)
        
        # Should return relevant items
        assert len(filtered) > 0
