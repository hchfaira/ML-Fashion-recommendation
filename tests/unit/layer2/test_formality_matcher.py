"""
Tests for Layer 2: Formality Matcher

Tests cover:
- Formality consistency checking
- Compatible formality levels
- Formality scoring
"""
import pytest

from src.core.models import FormalityLevel
from src.layer2_style import FormalityMatcher


# ============== Fixtures ==============

@pytest.fixture
def matcher():
    """Create a FormalityMatcher instance."""
    return FormalityMatcher()


@pytest.fixture
def casual_outfit(garment_factory):
    """Create a casual outfit."""
    return [
        garment_factory.create_basic_top(formality=FormalityLevel.CASUAL),
        garment_factory.create_basic_bottom(formality=FormalityLevel.CASUAL)
    ]


@pytest.fixture
def business_outfit(garment_factory):
    """Create a business outfit."""
    return [
        garment_factory.create_dress_shirt(),
        garment_factory.create_dress_pants()
    ]


@pytest.fixture
def mixed_formality_outfit(garment_factory):
    """Create an outfit with mixed formality levels."""
    return [
        garment_factory.create_basic_top(formality=FormalityLevel.CASUAL),
        garment_factory.create_dress_pants()  # Business
    ]


# ============== Test Classes ==============

@pytest.mark.unit
class TestFormalityMatcherInit:
    """Tests for FormalityMatcher initialization."""
    
    def test_init_creates_instance(self, matcher):
        """Test that FormalityMatcher initializes correctly."""
        assert matcher is not None
        assert isinstance(matcher, FormalityMatcher)


@pytest.mark.unit
class TestFormalityConsistency:
    """Tests for formality consistency checking."""
    
    def test_same_formality_scores_high(self, matcher, casual_outfit):
        """Test that same formality level scores 1.0."""
        score = matcher.check_formality_consistency(casual_outfit)
        assert score == 1.0
    
    def test_mismatched_formality_scores_lower(self, matcher, mixed_formality_outfit):
        """Test that mismatched formality scores lower."""
        score = matcher.check_formality_consistency(mixed_formality_outfit)
        assert score < 0.8
    
    def test_business_outfit_consistency(self, matcher, business_outfit):
        """Test business outfit formality consistency."""
        score = matcher.check_formality_consistency(business_outfit)
        assert score == 1.0


@pytest.mark.unit
class TestCompatibleFormalities:
    """Tests for compatible formality level retrieval."""
    
    def test_smart_casual_compatible_with_casual(self, matcher):
        """Test that smart casual is compatible with casual."""
        compatible = matcher.get_compatible_formalities(FormalityLevel.SMART_CASUAL)
        assert FormalityLevel.CASUAL in compatible
    
    def test_smart_casual_compatible_with_business_casual(self, matcher):
        """Test that smart casual is compatible with business casual."""
        compatible = matcher.get_compatible_formalities(FormalityLevel.SMART_CASUAL)
        assert FormalityLevel.BUSINESS_CASUAL in compatible
    
    def test_smart_casual_not_compatible_with_black_tie(self, matcher):
        """Test that smart casual is not compatible with black tie."""
        compatible = matcher.get_compatible_formalities(FormalityLevel.SMART_CASUAL)
        assert FormalityLevel.BLACK_TIE not in compatible
    
    def test_casual_compatible_formalities(self, matcher):
        """Test casual compatible formalities."""
        compatible = matcher.get_compatible_formalities(FormalityLevel.CASUAL)
        
        assert FormalityLevel.SMART_CASUAL in compatible
        assert FormalityLevel.BLACK_TIE not in compatible
    
    def test_formal_compatible_formalities(self, matcher):
        """Test formal compatible formalities."""
        compatible = matcher.get_compatible_formalities(FormalityLevel.FORMAL)
        
        assert FormalityLevel.BUSINESS in compatible or len(compatible) > 0


@pytest.mark.unit
class TestFormalityScoring:
    """Tests for formality score normalization."""
    
    def test_score_is_normalized(self, matcher, casual_outfit):
        """Test that scores are normalized between 0 and 1."""
        score = matcher.check_formality_consistency(casual_outfit)
        assert 0.0 <= score <= 1.0
    
    def test_score_with_empty_outfit(self, matcher):
        """Test scoring with empty outfit."""
        score = matcher.check_formality_consistency([])
        # Should handle gracefully
        assert score >= 0.0
