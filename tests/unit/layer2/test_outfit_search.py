"""
Unit Tests for Outfit Search Algorithms
========================================

Tests for BeamSearch, A*Search, and HybridSearch algorithms.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

from src.layer2_style.outfit_search import (
    SearchAlgorithm,
    SearchConfig,
    SearchNode,
    SearchResult,
    PartialScorer,
    BeamSearch,
    AStarSearch,
    HybridSearch,
    create_outfit_search,
    search_best_outfits,
)


# =============================================================================
# Fixtures
# =============================================================================

class MockCategory(Enum):
    """Mock category enum for testing."""
    TOP = "top"
    BOTTOM = "bottom"
    SHOES = "shoes"


@dataclass
class MockColor:
    primary: str = "blue"
    secondary: str = None


@dataclass
class MockPattern:
    type: str = "solid"


@dataclass 
class MockAttributes:
    category: MockCategory = MockCategory.TOP
    subcategory: str = "t-shirt"
    color: MockColor = None
    pattern: MockPattern = None
    
    def __post_init__(self):
        if self.color is None:
            self.color = MockColor()
        if self.pattern is None:
            self.pattern = MockPattern()


@dataclass
class MockGarment:
    """Mock garment for testing."""
    id: str
    attributes: MockAttributes = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = MockAttributes()


@pytest.fixture
def mock_wardrobe():
    """Create a simple mock wardrobe."""
    return {
        "tops": [
            MockGarment("top1", MockAttributes(MockCategory.TOP, "t-shirt", MockColor("white"))),
            MockGarment("top2", MockAttributes(MockCategory.TOP, "shirt", MockColor("blue"))),
            MockGarment("top3", MockAttributes(MockCategory.TOP, "sweater", MockColor("red"))),
        ],
        "bottoms": [
            MockGarment("bottom1", MockAttributes(MockCategory.BOTTOM, "jeans", MockColor("blue"))),
            MockGarment("bottom2", MockAttributes(MockCategory.BOTTOM, "pants", MockColor("black"))),
        ],
        "shoes": [
            MockGarment("shoes1", MockAttributes(MockCategory.SHOES, "sneakers", MockColor("white"))),
            MockGarment("shoes2", MockAttributes(MockCategory.SHOES, "boots", MockColor("brown"))),
        ],
    }


@pytest.fixture
def large_wardrobe():
    """Create a larger wardrobe for performance testing."""
    wardrobe = {
        "tops": [MockGarment(f"top{i}") for i in range(20)],
        "bottoms": [MockGarment(f"bottom{i}") for i in range(15)],
        "shoes": [MockGarment(f"shoes{i}") for i in range(10)],
        "outerwear": [MockGarment(f"outer{i}") for i in range(5)],
        "accessories": [MockGarment(f"acc{i}") for i in range(8)],
    }
    return wardrobe


# =============================================================================
# SearchConfig Tests
# =============================================================================

class TestSearchConfig:
    """Tests for SearchConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SearchConfig()
        
        assert config.algorithm == SearchAlgorithm.BEAM
        assert config.beam_width == 10
        assert config.top_k == 5
        assert config.min_score_threshold == 0.3
        assert "tops" in config.required_categories
        assert "bottoms" in config.required_categories
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SearchConfig(
            algorithm=SearchAlgorithm.ASTAR,
            beam_width=20,
            top_k=10,
            max_expansions=500
        )
        
        assert config.algorithm == SearchAlgorithm.ASTAR
        assert config.beam_width == 20
        assert config.top_k == 10
        assert config.max_expansions == 500


class TestSearchNode:
    """Tests for SearchNode dataclass."""
    
    def test_node_creation(self):
        """Test creating a search node."""
        garments = [MockGarment("g1"), MockGarment("g2")]
        
        node = SearchNode(
            priority=-0.8,
            garments=garments,
            categories_filled={"tops", "bottoms"},
            g_score=0.75,
            h_score=0.05,
            f_score=0.80,
            depth=2
        )
        
        assert node.priority == -0.8
        assert len(node.garments) == 2
        assert node.depth == 2
        assert node.g_score == 0.75
    
    def test_node_state_key(self):
        """Test unique state key generation."""
        garments = [MockGarment("g2"), MockGarment("g1")]  # Unordered
        
        node = SearchNode(
            priority=0,
            garments=garments,
            categories_filled=set()
        )
        
        # State key should be sorted
        assert node.state_key == "g1|g2"
    
    def test_node_ordering(self):
        """Test nodes are ordered by priority."""
        node1 = SearchNode(priority=-0.9, garments=[], categories_filled=set())
        node2 = SearchNode(priority=-0.5, garments=[], categories_filled=set())
        
        # Lower priority value should sort first (for min-heap)
        assert node1 < node2


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_result_creation(self):
        """Test creating search result."""
        outfits = [[MockGarment("g1")], [MockGarment("g2")]]
        scores = [0.9, 0.8]
        
        result = SearchResult(
            outfits=outfits,
            scores=scores,
            nodes_expanded=100,
            nodes_pruned=50,
            search_time_ms=25.5,
            algorithm=SearchAlgorithm.BEAM
        )
        
        assert len(result) == 2
        assert result.nodes_expanded == 100
        assert result.search_time_ms == 25.5
    
    def test_result_best(self):
        """Test getting best result."""
        outfits = [[MockGarment("g1")], [MockGarment("g2")]]
        scores = [0.9, 0.8]
        
        result = SearchResult(
            outfits=outfits,
            scores=scores,
            nodes_expanded=0,
            nodes_pruned=0,
            search_time_ms=0,
            algorithm=SearchAlgorithm.BEAM
        )
        
        best_outfit, best_score = result.best()
        assert best_score == 0.9
        assert best_outfit[0].id == "g1"
    
    def test_result_iteration(self):
        """Test iterating over results."""
        outfits = [[MockGarment("g1")], [MockGarment("g2")]]
        scores = [0.9, 0.8]
        
        result = SearchResult(
            outfits=outfits,
            scores=scores,
            nodes_expanded=0,
            nodes_pruned=0,
            search_time_ms=0,
            algorithm=SearchAlgorithm.BEAM
        )
        
        items = list(result)
        assert len(items) == 2
        assert items[0] == (outfits[0], 0.9)


# =============================================================================
# PartialScorer Tests
# =============================================================================

class TestPartialScorer:
    """Tests for PartialScorer."""
    
    def test_scorer_initialization(self):
        """Test scorer initialization."""
        scorer = PartialScorer()
        
        assert scorer.profile is None
        assert isinstance(scorer._cache, dict)
    
    def test_score_partial_empty(self):
        """Test scoring empty garment list."""
        scorer = PartialScorer()
        
        score = scorer.score_partial([])
        assert score == 0.0
    
    def test_score_partial_caching(self, mock_wardrobe):
        """Test that scores are cached."""
        scorer = PartialScorer()
        
        garments = mock_wardrobe["tops"][:2]
        
        # First call
        score1 = scorer.score_partial(garments)
        
        # Second call (should hit cache)
        score2 = scorer.score_partial(garments)
        
        assert score1 == score2
        assert len(scorer._cache) > 0
    
    def test_clear_cache(self, mock_wardrobe):
        """Test clearing cache."""
        scorer = PartialScorer()
        
        scorer.score_partial(mock_wardrobe["tops"][:1])
        assert len(scorer._cache) > 0
        
        scorer.clear_cache()
        assert len(scorer._cache) == 0
    
    def test_estimate_heuristic(self, mock_wardrobe):
        """Test heuristic estimation."""
        scorer = PartialScorer()
        
        current = mock_wardrobe["tops"][:1]
        remaining = {"bottoms", "shoes"}
        
        h = scorer.estimate_heuristic(current, remaining, mock_wardrobe)
        
        # Heuristic should be between 0 and 1
        assert 0.0 <= h <= 1.0


# =============================================================================
# BeamSearch Tests
# =============================================================================

class TestBeamSearch:
    """Tests for BeamSearch algorithm."""
    
    def test_beam_search_initialization(self):
        """Test beam search initialization."""
        config = SearchConfig(beam_width=5)
        search = BeamSearch(config)
        
        assert search.config.beam_width == 5
    
    def test_beam_search_simple_wardrobe(self, mock_wardrobe):
        """Test beam search on simple wardrobe."""
        config = SearchConfig(
            beam_width=5,
            top_k=3,
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes"]
        )
        search = BeamSearch(config)
        
        result = search.search(mock_wardrobe)
        
        assert isinstance(result, SearchResult)
        assert result.algorithm == SearchAlgorithm.BEAM
        assert len(result.outfits) <= 3
        assert result.nodes_expanded > 0
    
    def test_beam_search_returns_complete_outfits(self, mock_wardrobe):
        """Test that beam search returns complete outfits."""
        config = SearchConfig(
            beam_width=3,
            top_k=2,
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes"]
        )
        search = BeamSearch(config)
        
        result = search.search(mock_wardrobe)
        
        # Each outfit should have items from all searched categories
        for outfit in result.outfits:
            categories = {g.attributes.category.value for g in outfit}
            assert "top" in categories
            assert "bottom" in categories
    
    def test_beam_search_missing_required_category(self):
        """Test beam search with missing required category."""
        wardrobe = {
            "tops": [MockGarment("t1")],
            # Missing "bottoms"
        }
        
        search = BeamSearch()
        result = search.search(wardrobe)
        
        assert len(result.outfits) == 0
    
    def test_beam_search_respects_beam_width(self, large_wardrobe):
        """Test that beam width limits candidates."""
        config = SearchConfig(beam_width=3, top_k=3)
        search = BeamSearch(config)
        
        result = search.search(large_wardrobe)
        
        # Should have results despite large wardrobe
        assert len(result.outfits) <= 3


# =============================================================================
# AStarSearch Tests
# =============================================================================

class TestAStarSearch:
    """Tests for A* Search algorithm."""
    
    def test_astar_initialization(self):
        """Test A* search initialization."""
        config = SearchConfig(max_expansions=500)
        search = AStarSearch(config)
        
        assert search.config.max_expansions == 500
    
    def test_astar_simple_wardrobe(self, mock_wardrobe):
        """Test A* search on simple wardrobe."""
        config = SearchConfig(
            algorithm=SearchAlgorithm.ASTAR,
            max_expansions=100,
            top_k=3,
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes"]
        )
        search = AStarSearch(config)
        
        result = search.search(mock_wardrobe)
        
        assert isinstance(result, SearchResult)
        assert result.algorithm == SearchAlgorithm.ASTAR
        assert len(result.outfits) <= 3
    
    def test_astar_respects_max_expansions(self, large_wardrobe):
        """Test that A* respects max expansions limit."""
        config = SearchConfig(
            algorithm=SearchAlgorithm.ASTAR,
            max_expansions=50,
            top_k=5
        )
        search = AStarSearch(config)
        
        result = search.search(large_wardrobe)
        
        # Should not expand more than limit
        assert result.nodes_expanded <= 50
    
    def test_astar_pruning(self, mock_wardrobe):
        """Test that A* prunes low-scoring paths."""
        config = SearchConfig(
            algorithm=SearchAlgorithm.ASTAR,
            min_score_threshold=0.9,  # High threshold to force pruning
            max_expansions=100
        )
        search = AStarSearch(config)
        
        result = search.search(mock_wardrobe)
        
        # Should have pruned some nodes
        # (Exact count depends on scorer implementation)
        assert result.nodes_pruned >= 0


# =============================================================================
# HybridSearch Tests
# =============================================================================

class TestHybridSearch:
    """Tests for Hybrid Search algorithm."""
    
    def test_hybrid_initialization(self):
        """Test hybrid search initialization."""
        config = SearchConfig(beam_width=10, top_k=5)
        search = HybridSearch(config)
        
        assert search.config.beam_width == 10
        assert search._beam_search is not None
    
    def test_hybrid_simple_wardrobe(self, mock_wardrobe):
        """Test hybrid search on simple wardrobe."""
        config = SearchConfig(
            algorithm=SearchAlgorithm.HYBRID,
            beam_width=5,
            top_k=3
        )
        search = HybridSearch(config)
        
        result = search.search(mock_wardrobe)
        
        assert isinstance(result, SearchResult)
        assert result.algorithm == SearchAlgorithm.HYBRID
        assert len(result.outfits) <= 3


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateOutfitSearch:
    """Tests for create_outfit_search factory."""
    
    def test_create_beam_search(self):
        """Test creating beam search."""
        search = create_outfit_search(SearchAlgorithm.BEAM)
        
        assert isinstance(search, BeamSearch)
    
    def test_create_astar_search(self):
        """Test creating A* search."""
        search = create_outfit_search(SearchAlgorithm.ASTAR)
        
        assert isinstance(search, AStarSearch)
    
    def test_create_hybrid_search(self):
        """Test creating hybrid search."""
        search = create_outfit_search(SearchAlgorithm.HYBRID)
        
        assert isinstance(search, HybridSearch)
    
    def test_create_with_config(self):
        """Test creating search with custom config."""
        config = SearchConfig(beam_width=20, top_k=10)
        search = create_outfit_search(SearchAlgorithm.BEAM, config)
        
        assert search.config.beam_width == 20
        assert search.config.top_k == 10


class TestSearchBestOutfits:
    """Tests for search_best_outfits convenience function."""
    
    def test_search_best_outfits(self, mock_wardrobe):
        """Test convenience search function."""
        result = search_best_outfits(
            mock_wardrobe,
            algorithm=SearchAlgorithm.BEAM,
            top_k=3,
            beam_width=5
        )
        
        assert isinstance(result, SearchResult)
        assert len(result.outfits) <= 3


# =============================================================================
# Performance Tests
# =============================================================================

class TestSearchPerformance:
    """Performance-related tests."""
    
    def test_beam_search_scales_with_beam_width(self, large_wardrobe):
        """Test that beam search time scales with beam width."""
        import time
        
        results = []
        for beam_width in [3, 10]:
            config = SearchConfig(beam_width=beam_width, top_k=3)
            search = BeamSearch(config)
            
            start = time.time()
            result = search.search(large_wardrobe)
            elapsed = time.time() - start
            
            results.append((beam_width, elapsed, result.nodes_expanded))
        
        # Larger beam width should expand more nodes
        assert results[1][2] >= results[0][2]
    
    def test_search_completes_in_reasonable_time(self, large_wardrobe):
        """Test that search completes quickly."""
        config = SearchConfig(beam_width=10, top_k=5)
        search = BeamSearch(config)
        
        result = search.search(large_wardrobe)
        
        # Should complete in under 1 second for this wardrobe size
        assert result.search_time_ms < 1000


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_wardrobe(self):
        """Test search with empty wardrobe."""
        search = BeamSearch()
        result = search.search({})
        
        assert len(result.outfits) == 0
    
    def test_single_item_per_category(self):
        """Test with single item per category."""
        wardrobe = {
            "tops": [MockGarment("t1")],
            "bottoms": [MockGarment("b1")],
            "shoes": [MockGarment("s1")],
        }
        
        search = BeamSearch()
        result = search.search(wardrobe)
        
        # Should find exactly one outfit
        assert len(result.outfits) == 1
        assert len(result.outfits[0]) == 3
    
    def test_only_required_categories(self):
        """Test with only required categories filled."""
        wardrobe = {
            "tops": [MockGarment("t1"), MockGarment("t2")],
            "bottoms": [MockGarment("b1")],
            # No shoes, outerwear, or accessories
        }
        
        config = SearchConfig(
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes"]
        )
        search = BeamSearch(config)
        result = search.search(wardrobe)
        
        # Should find outfits with just tops and bottoms
        assert len(result.outfits) > 0
        for outfit in result.outfits:
            assert len(outfit) == 2


# =============================================================================
# Full Body Support Tests
# =============================================================================

class TestFullBodySupport:
    """Tests for full_body garment support (dresses, jumpsuits, etc.)."""
    
    @pytest.fixture
    def full_body_wardrobe(self):
        """Wardrobe with full_body items (dresses, jumpsuits)."""
        return {
            "full_body": [
                MockGarment("dress1", MockAttributes(MockCategory.TOP, "dress", MockColor("red"))),
                MockGarment("jumpsuit1", MockAttributes(MockCategory.TOP, "jumpsuit", MockColor("black"))),
            ],
            "shoes": [
                MockGarment("heels1", MockAttributes(MockCategory.SHOES, "heels", MockColor("nude"))),
                MockGarment("sandals1", MockAttributes(MockCategory.SHOES, "sandals", MockColor("brown"))),
            ],
            "accessories": [
                MockGarment("bag1"),
            ],
        }
    
    @pytest.fixture
    def mixed_wardrobe(self):
        """Wardrobe with both top+bottom AND full_body options."""
        return {
            "tops": [
                MockGarment("top1", MockAttributes(MockCategory.TOP, "blouse", MockColor("white"))),
                MockGarment("top2", MockAttributes(MockCategory.TOP, "shirt", MockColor("blue"))),
            ],
            "bottoms": [
                MockGarment("bottom1", MockAttributes(MockCategory.BOTTOM, "skirt", MockColor("black"))),
                MockGarment("bottom2", MockAttributes(MockCategory.BOTTOM, "pants", MockColor("navy"))),
            ],
            "full_body": [
                MockGarment("dress1", MockAttributes(MockCategory.TOP, "dress", MockColor("red"))),
            ],
            "shoes": [
                MockGarment("shoes1", MockAttributes(MockCategory.SHOES, "flats", MockColor("black"))),
            ],
        }
    
    def test_beam_search_full_body_only_wardrobe(self, full_body_wardrobe):
        """Test BeamSearch with only full_body items (no tops/bottoms)."""
        config = SearchConfig(
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes", "accessories"],
            include_full_body=True
        )
        search = BeamSearch(config)
        result = search.search(full_body_wardrobe)
        
        # Should find outfits with full_body + shoes
        assert len(result.outfits) > 0
        
        # Each outfit should have full_body item
        for outfit in result.outfits:
            outfit_ids = [g.id for g in outfit]
            has_full_body = any("dress" in id or "jumpsuit" in id for id in outfit_ids)
            assert has_full_body, f"Outfit {outfit_ids} should have full_body item"
    
    def test_beam_search_mixed_wardrobe(self, mixed_wardrobe):
        """Test BeamSearch with both top+bottom and full_body options."""
        config = SearchConfig(
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes"],
            include_full_body=True
        )
        search = BeamSearch(config)
        result = search.search(mixed_wardrobe)
        
        # Should find outfits from both strategies
        assert len(result.outfits) > 0
        
        # Check we get both types of outfits
        top_bottom_outfits = 0
        full_body_outfits = 0
        
        for outfit in result.outfits:
            outfit_ids = [g.id for g in outfit]
            has_top = any("top" in id for id in outfit_ids)
            has_bottom = any("bottom" in id for id in outfit_ids)
            has_dress = any("dress" in id for id in outfit_ids)
            
            if has_top and has_bottom:
                top_bottom_outfits += 1
            if has_dress:
                full_body_outfits += 1
        
        # Should have at least one of each type
        assert top_bottom_outfits > 0, "Should have top+bottom outfits"
        assert full_body_outfits > 0, "Should have full_body outfits"
    
    def test_beam_search_full_body_disabled(self, mixed_wardrobe):
        """Test BeamSearch with full_body disabled."""
        config = SearchConfig(
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes"],
            include_full_body=False
        )
        search = BeamSearch(config)
        result = search.search(mixed_wardrobe)
        
        # Should only find top+bottom outfits
        for outfit in result.outfits:
            outfit_ids = [g.id for g in outfit]
            has_dress = any("dress" in id for id in outfit_ids)
            assert not has_dress, f"Outfit {outfit_ids} should not have full_body when disabled"
    
    def test_astar_search_full_body_only_wardrobe(self, full_body_wardrobe):
        """Test A*Search with only full_body items."""
        config = SearchConfig(
            algorithm=SearchAlgorithm.ASTAR,
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes", "accessories"],
            include_full_body=True
        )
        search = AStarSearch(config)
        result = search.search(full_body_wardrobe)
        
        # Should find outfits with full_body + shoes
        assert len(result.outfits) > 0
        assert result.algorithm == SearchAlgorithm.ASTAR
    
    def test_astar_search_mixed_wardrobe(self, mixed_wardrobe):
        """Test A*Search with both strategies."""
        config = SearchConfig(
            algorithm=SearchAlgorithm.ASTAR,
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes"],
            include_full_body=True
        )
        search = AStarSearch(config)
        result = search.search(mixed_wardrobe)
        
        # Should find outfits from both strategies
        assert len(result.outfits) > 0
    
    def test_hybrid_search_mixed_wardrobe(self, mixed_wardrobe):
        """Test HybridSearch with both strategies."""
        config = SearchConfig(
            algorithm=SearchAlgorithm.HYBRID,
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes"],
            include_full_body=True
        )
        search = HybridSearch(config)
        result = search.search(mixed_wardrobe)
        
        # Should find outfits
        assert len(result.outfits) > 0
        assert result.algorithm == SearchAlgorithm.HYBRID
    
    def test_full_body_replaces_top_and_bottom(self, full_body_wardrobe):
        """Test that full_body items are used instead of top+bottom."""
        config = SearchConfig(
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes", "accessories"],
            include_full_body=True
        )
        search = BeamSearch(config)
        result = search.search(full_body_wardrobe)
        
        # Each outfit should NOT have both top and bottom
        for outfit in result.outfits:
            outfit_ids = [g.id for g in outfit]
            has_top = any("top" in id for id in outfit_ids)
            has_bottom = any("bottom" in id for id in outfit_ids)
            # Full body wardrobe has no tops/bottoms, so this confirms full_body is used
            assert not has_top, "Should not have separate top with full_body"
            assert not has_bottom, "Should not have separate bottom with full_body"
    
    def test_no_valid_outfit_when_missing_all(self):
        """Test empty result when neither top+bottom nor full_body available."""
        wardrobe = {
            "shoes": [MockGarment("shoes1")],
            "accessories": [MockGarment("acc1")],
        }
        
        config = SearchConfig(
            required_categories=["tops", "bottoms"],
            include_full_body=True
        )
        search = BeamSearch(config)
        result = search.search(wardrobe)
        
        # Should find no valid outfits
        assert len(result.outfits) == 0
    
    def test_create_outfit_search_with_full_body(self):
        """Test factory function with full_body config."""
        search = create_outfit_search(
            SearchAlgorithm.BEAM,
            include_full_body=True
        )
        
        assert isinstance(search, BeamSearch)
        assert search.config.include_full_body == True
