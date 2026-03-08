"""
Tests for Layer 2: Compatibility Scorer

Tests cover:
- Pair scoring
- Multi-item scoring
- Score normalization
"""
import pytest

from src.layer2_style import CompatibilityScorer


# ============== Fixtures ==============

@pytest.fixture
def scorer():
    """Create a CompatibilityScorer instance."""
    return CompatibilityScorer()


@pytest.fixture
def sample_top(garment_factory):
    """Create a sample top."""
    return garment_factory.create_basic_top(color="white")


@pytest.fixture
def sample_bottom(garment_factory):
    """Create a sample bottom."""
    return garment_factory.create_basic_bottom(color="blue")


@pytest.fixture
def formal_blazer(garment_factory):
    """Create a formal blazer."""
    return garment_factory.create_outerwear(
        subcategory="blazer",
        color="navy"
    )


# ============== Test Classes ==============

@pytest.mark.unit
class TestCompatibilityScorerInit:
    """Tests for CompatibilityScorer initialization."""
    
    def test_init_creates_instance(self, scorer):
        """Test that CompatibilityScorer initializes correctly."""
        assert scorer is not None
        assert isinstance(scorer, CompatibilityScorer)


@pytest.mark.unit
class TestPairScoring:
    """Tests for pair compatibility scoring."""
    
    @pytest.mark.asyncio
    async def test_score_pair_returns_valid_score(
        self, scorer, sample_top, sample_bottom
    ):
        """Test that score_pair returns a valid score."""
        score = await scorer.score_pair(sample_top, sample_bottom)
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_score_pair_casual_items(
        self, scorer, sample_top, sample_bottom
    ):
        """Test scoring casual top and bottom."""
        score = await scorer.score_pair(sample_top, sample_bottom)
        
        # Casual items should score reasonably well together
        assert score >= 0.5


@pytest.mark.unit
class TestMultiItemScoring:
    """Tests for multi-item compatibility scoring."""
    
    @pytest.mark.asyncio
    async def test_score_items_returns_valid_score(
        self, scorer, sample_top, sample_bottom
    ):
        """Test that score_items returns a valid score."""
        score = await scorer.score_items([sample_top, sample_bottom])
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_score_multiple_items(
        self, scorer, sample_top, sample_bottom, formal_blazer
    ):
        """Test scoring multiple items."""
        score = await scorer.score_items([sample_top, sample_bottom, formal_blazer])
        
        assert 0 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_score_single_item(self, scorer, sample_top):
        """Test scoring a single item."""
        score = await scorer.score_items([sample_top])
        
        # Single item should be compatible with itself
        assert score >= 0.0


@pytest.mark.unit
class TestScoreNormalization:
    """Tests for score normalization."""
    
    @pytest.mark.asyncio
    async def test_scores_between_zero_and_one(
        self, scorer, sample_top, sample_bottom
    ):
        """Test that all scores are normalized."""
        pair_score = await scorer.score_pair(sample_top, sample_bottom)
        items_score = await scorer.score_items([sample_top, sample_bottom])
        
        assert 0.0 <= pair_score <= 1.0
        assert 0.0 <= items_score <= 1.0
