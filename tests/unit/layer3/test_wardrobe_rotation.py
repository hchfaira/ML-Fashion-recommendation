"""
Tests for Layer 3: Wardrobe Rotation Service

Tests cover:
- Wear recording and retrieval
- Freshness scoring
- Forgotten gems discovery
- Featured item suggestions
- Wear statistics
"""
import pytest
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, ColorProfile,
    UserContext, Occasion, FormalityLevel
)
from src.layer3_context.wardrobe_rotation import WardrobeRotationService


# ============== Fixtures ==============

@pytest.fixture
def service():
    """Create a WardrobeRotationService instance."""
    return WardrobeRotationService()


@pytest.fixture
def sample_garments():
    """Create sample garments for testing."""
    return [
        Garment(
            id="top-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                color=ColorProfile(primary="white"),
                formality_level=FormalityLevel.CASUAL
            )
        ),
        Garment(
            id="blazer-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.OUTERWEAR,
                color=ColorProfile(primary="navy"),
                formality_level=FormalityLevel.BUSINESS_CASUAL
            )
        ),
        Garment(
            id="jeans-1",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM,
                color=ColorProfile(primary="blue"),
                formality_level=FormalityLevel.CASUAL
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
class TestWardrobeRotationInit:
    """Tests for WardrobeRotationService initialization."""
    
    def test_init_creates_instance(self, service):
        """Test that service initializes correctly."""
        assert service is not None
        assert isinstance(service, WardrobeRotationService)


@pytest.mark.unit
class TestWearRecording:
    """Tests for wear recording functionality."""
    
    def test_record_and_retrieve_wear(self, service, sample_garments):
        """Test recording and retrieving wear history."""
        service.record_wear(["top-1", "jeans-1"])
        
        recently_worn = service.get_recently_worn(days=7)
        
        assert "top-1" in recently_worn
        assert "jeans-1" in recently_worn
        assert "blazer-1" not in recently_worn
    
    def test_record_wear_updates_history(self, service):
        """Test that recording wear updates history."""
        service.record_wear(["top-1"])
        recently_worn = service.get_recently_worn(days=1)
        
        assert "top-1" in recently_worn


@pytest.mark.unit
class TestFreshnessScoring:
    """Tests for freshness score calculation."""
    
    def test_freshness_score_never_worn(
        self, service, sample_garments, basic_context
    ):
        """Test freshness score for never worn item is 1.0."""
        score = service.calculate_freshness_score(
            sample_garments[0], basic_context
        )
        
        assert score == 1.0
    
    def test_freshness_score_recently_worn(
        self, service, sample_garments, basic_context
    ):
        """Test freshness score for recently worn item is low."""
        service.record_wear([sample_garments[0].id])
        
        score = service.calculate_freshness_score(
            sample_garments[0], basic_context
        )
        
        assert score < 0.5
    
    def test_freshness_score_with_context_recently_worn(
        self, service, sample_garments
    ):
        """Test freshness using context's recently worn list."""
        context = UserContext(
            user_id="test-user",
            recently_worn_items=["top-1"]
        )
        
        score = service.calculate_freshness_score(
            sample_garments[0], context
        )
        
        assert score == 0.2


@pytest.mark.unit
class TestOutfitFreshness:
    """Tests for outfit freshness scoring."""
    
    def test_outfit_freshness_score_range(
        self, service, sample_garments, basic_context
    ):
        """Test that outfit freshness score is between 0 and 1."""
        outfit = sample_garments[:3]
        
        score = service.score_outfit_freshness(outfit, basic_context)
        
        assert 0 <= score <= 1.0


@pytest.mark.unit
class TestForgottenGems:
    """Tests for forgotten gems discovery."""
    
    def test_get_forgotten_gems_old_wear(self, service, sample_garments):
        """Test finding items worn long ago."""
        old_date = datetime.now() - timedelta(days=60)
        service.record_wear(["top-1"], old_date)
        service.record_wear(["blazer-1"])  # Recent
        
        gems = service.get_forgotten_gems(sample_garments, min_days=30)
        
        gem_ids = [g.id for g in gems]
        assert "top-1" in gem_ids
        assert "blazer-1" not in gem_ids


@pytest.mark.unit
class TestFeaturedItem:
    """Tests for featured item suggestions."""
    
    def test_suggest_featured_item(self, service, sample_garments, basic_context):
        """Test suggesting a featured item."""
        featured = service.suggest_featured_item(sample_garments, basic_context)
        
        assert featured is not None
        assert featured in sample_garments
    
    def test_suggest_featured_item_respects_preference(self, service, sample_garments):
        """Test that featured item respects user preference."""
        context = UserContext(
            user_id="test-user",
            items_to_feature=["blazer-1"]
        )
        
        featured = service.suggest_featured_item(sample_garments, context)
        
        assert featured.id == "blazer-1"


@pytest.mark.unit
class TestWearStatistics:
    """Tests for wear statistics."""
    
    def test_get_wear_statistics(self, service, sample_garments):
        """Test wardrobe usage statistics."""
        service.record_wear(["top-1"])
        service.record_wear(["top-1"])
        service.record_wear(["jeans-1"])
        
        stats = service.get_wear_statistics(sample_garments)
        
        assert stats["total_items"] == len(sample_garments)
        assert stats["worn_this_week"] > 0
        assert "by_category" in stats
