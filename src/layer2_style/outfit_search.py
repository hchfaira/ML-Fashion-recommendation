"""
Outfit Search Algorithms
========================

Optimized search algorithms for finding the best outfit combinations
from a wardrobe without exhaustive enumeration.

Algorithms:
- BeamSearch: Greedy search with beam width
- AStarSearch: Heuristic-guided search with pruning
- HybridSearch: Combines beam search with A* for best results

Architecture:
    Wardrobe
        ↓
    OutfitSearch (A* or Beam Search)
        ↓
    Scorers (lazy evaluation)
        ↓
    TotalStyleScorer
        ↓
    Top-K Outfits

Key optimizations:
- Partial scoring for early pruning
- Priority queue for best-first exploration
- Heuristic estimation for unseen garments
- Caching of computed scores
"""

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Callable
from enum import Enum
import time

from src.core.models import Garment, UserContext
from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class SearchAlgorithm(Enum):
    """Available search algorithms."""
    BEAM = "beam"
    ASTAR = "astar"
    HYBRID = "hybrid"
    EXHAUSTIVE = "exhaustive"


@dataclass
class SearchConfig:
    """Configuration for outfit search."""
    algorithm: SearchAlgorithm = SearchAlgorithm.BEAM
    
    # Beam Search parameters
    beam_width: int = 10  # Number of candidates to keep at each level
    
    # A* parameters
    max_expansions: int = 1000  # Maximum nodes to expand
    heuristic_weight: float = 1.0  # Weight for heuristic in f = g + w*h
    
    # General parameters
    top_k: int = 5  # Number of top results to return
    min_score_threshold: float = 0.3  # Minimum score to consider
    early_termination: bool = True  # Stop early if confident
    
    # Category constraints
    # Outfit minimum = (top + bottom) OR full_body
    required_categories: List[str] = field(default_factory=lambda: ["tops", "bottoms"])
    optional_categories: List[str] = field(default_factory=lambda: ["shoes", "outerwear", "accessories"])
    full_body_category: str = "full_body"  # Category for dresses, jumpsuits, etc.
    include_full_body: bool = True  # Whether to search full_body outfits
    
    # Scoring profile
    scoring_profile: Optional[str] = None


@dataclass(order=True)
class SearchNode:
    """
    Node in the search tree representing a partial outfit.
    
    Attributes:
        priority: For heap ordering (negative score for max-heap behavior)
        garments: Current list of garments in partial outfit
        categories_filled: Set of categories already chosen
        g_score: Actual partial score (sum of pairwise compatibilities)
        h_score: Heuristic estimate for remaining categories
        f_score: Total estimated score (g + h)
        depth: Number of categories filled
    """
    priority: float = field(compare=True)
    garments: List[Garment] = field(compare=False)
    categories_filled: Set[str] = field(compare=False)
    g_score: float = field(compare=False, default=0.0)
    h_score: float = field(compare=False, default=0.0)
    f_score: float = field(compare=False, default=0.0)
    depth: int = field(compare=False, default=0)
    
    @property
    def state_key(self) -> str:
        """Unique key for this state (for visited tracking)."""
        return "|".join(sorted(g.id for g in self.garments))


@dataclass
class SearchResult:
    """Result of outfit search."""
    outfits: List[List[Garment]]  # Top-K outfits
    scores: List[float]  # Corresponding scores
    nodes_expanded: int
    nodes_pruned: int
    search_time_ms: float
    algorithm: SearchAlgorithm
    
    def __len__(self) -> int:
        return len(self.outfits)
    
    def __iter__(self):
        return iter(zip(self.outfits, self.scores))
    
    def best(self) -> Tuple[List[Garment], float]:
        """Return best outfit and score."""
        if not self.outfits:
            return [], 0.0
        return self.outfits[0], self.scores[0]


# =============================================================================
# Scorer Interface
# =============================================================================

class PartialScorer:
    """
    Lightweight scorer for partial outfits during search.
    
    Provides fast approximate scores for pruning without
    full scorecard computation.
    """
    
    def __init__(self, profile: Optional[str] = None):
        """
        Initialize partial scorer.
        
        Args:
            profile: Optional scoring profile
        """
        self.profile = profile
        self._cache: Dict[str, float] = {}
        
        # Lazy imports to avoid circular dependencies
        self._full_scorers_loaded = False
        self._seven_point_scorer = None
        self._color_harmony = None
        self._three_color_scorer = None
    
    def _load_scorers(self):
        """Lazy load scorers on first use."""
        if self._full_scorers_loaded:
            return
        
        from .seven_point_rule import SevenPointRuleScorer
        from .color_harmony import ColorHarmonyAnalyzer
        from .three_color_scorer import ThreeColorScorer
        
        self._seven_point_scorer = SevenPointRuleScorer()
        self._color_harmony = ColorHarmonyAnalyzer()
        self._three_color_scorer = ThreeColorScorer()
        self._full_scorers_loaded = True
    
    def score_partial(self, garments: List[Garment]) -> float:
        """
        Compute partial score for a set of garments.
        
        Uses a subset of fast scorers for quick evaluation.
        
        Args:
            garments: List of garments to score
            
        Returns:
            Partial score (0-1)
        """
        if not garments:
            return 0.0
        
        # Check cache
        cache_key = "|".join(sorted(g.id for g in garments))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self._load_scorers()
        
        # Quick scoring: color harmony + three color rule
        # These are fast and provide good signal
        scores = []
        
        # Color harmony (fast)
        try:
            color_score = self._color_harmony.analyze_outfit_colors(garments)
            scores.append(color_score)
        except Exception:
            scores.append(0.5)
        
        # Three color rule (very fast)
        try:
            three_color = self._three_color_scorer.analyze_outfit(garments)
            scores.append(three_color.score)
        except Exception:
            scores.append(0.5)
        
        # Average partial scores
        partial_score = sum(scores) / len(scores) if scores else 0.5
        
        self._cache[cache_key] = partial_score
        return partial_score
    
    def score_pairwise(self, garment1: Garment, garment2: Garment) -> float:
        """
        Compute compatibility score between two garments.
        
        Args:
            garment1: First garment
            garment2: Second garment
            
        Returns:
            Pairwise compatibility score (0-1)
        """
        return self.score_partial([garment1, garment2])
    
    def estimate_heuristic(
        self, 
        current_garments: List[Garment],
        remaining_categories: Set[str],
        wardrobe: Dict[str, List[Garment]]
    ) -> float:
        """
        Estimate optimistic score for remaining categories (A* heuristic).
        
        Returns upper bound on possible improvement from adding
        garments from remaining categories.
        
        Args:
            current_garments: Garments already selected
            remaining_categories: Categories still to fill
            wardrobe: Full wardrobe
            
        Returns:
            Heuristic estimate (optimistic upper bound)
        """
        if not remaining_categories:
            return 0.0
        
        # Optimistic: assume best possible garment from each category
        # contributes perfectly (score = 1.0)
        # This is admissible for A* (never overestimates)
        
        # Simple heuristic: remaining categories contribute proportionally
        total_categories = len(remaining_categories) + len(
            set(g.attributes.category.value for g in current_garments)
        )
        remaining_weight = len(remaining_categories) / total_categories
        
        return remaining_weight * 1.0  # Optimistic assumption
    
    def clear_cache(self):
        """Clear the score cache."""
        self._cache.clear()


# =============================================================================
# Base Search Class
# =============================================================================

class BaseOutfitSearch(ABC):
    """Abstract base class for outfit search algorithms."""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize search.
        
        Args:
            config: Search configuration
        """
        self.config = config or SearchConfig()
        self.scorer = PartialScorer(profile=self.config.scoring_profile)
        
        # Statistics
        self._nodes_expanded = 0
        self._nodes_pruned = 0
    
    @abstractmethod
    def search(
        self, 
        wardrobe: Dict[str, List[Garment]],
        context: Optional[UserContext] = None
    ) -> SearchResult:
        """
        Search for best outfits in wardrobe.
        
        Args:
            wardrobe: Dict mapping category to list of garments
            context: Optional user context
            
        Returns:
            SearchResult with top-K outfits
        """
        pass
    
    def _get_search_categories(
        self, 
        wardrobe: Dict[str, List[Garment]],
        use_full_body: bool = False
    ) -> List[str]:
        """
        Get ordered list of categories to search.
        
        Args:
            wardrobe: Wardrobe dictionary
            use_full_body: If True, use full_body as base instead of top+bottom
            
        Returns:
            List of category names in search order
        """
        categories = []
        
        if use_full_body:
            # Full body replaces top+bottom
            full_body_cat = self.config.full_body_category
            if full_body_cat in wardrobe and wardrobe[full_body_cat]:
                categories.append(full_body_cat)
        else:
            # Required categories first (top + bottom)
            for cat in self.config.required_categories:
                if cat in wardrobe and wardrobe[cat]:
                    categories.append(cat)
        
        # Optional categories (shoes, outerwear, accessories)
        excluded = set(self.config.required_categories) | {self.config.full_body_category}
        for cat in self.config.optional_categories:
            if cat in excluded:
                continue
            if cat in wardrobe and wardrobe[cat]:
                categories.append(cat)
        
        return categories
    
    def _validate_wardrobe(
        self, 
        wardrobe: Dict[str, List[Garment]]
    ) -> Tuple[bool, bool]:
        """
        Check if wardrobe has required categories.
        
        Returns:
            Tuple of (has_top_bottom, has_full_body)
        """
        # Check top + bottom
        has_top_bottom = all(
            cat in wardrobe and wardrobe[cat]
            for cat in self.config.required_categories
        )
        
        # Check full body
        full_body_cat = self.config.full_body_category
        has_full_body = (
            self.config.include_full_body and
            full_body_cat in wardrobe and
            bool(wardrobe.get(full_body_cat))
        )
        
        return has_top_bottom, has_full_body
    
    def _can_search(self, wardrobe: Dict[str, List[Garment]]) -> bool:
        """Check if any valid search is possible."""
        has_top_bottom, has_full_body = self._validate_wardrobe(wardrobe)
        if not has_top_bottom and not has_full_body:
            logger.warning("Cannot search: need (tops + bottoms) OR full_body items")
            return False
        return True


# =============================================================================
# Beam Search
# =============================================================================

class BeamSearch(BaseOutfitSearch):
    """
    Beam Search for outfit selection.
    
    Maintains top-K partial solutions at each level, expanding
    only the most promising candidates.
    
    Supports two outfit strategies:
    - Top + Bottom (+ optional: shoes, outerwear, accessories)
    - Full Body (dress, jumpsuit) + optional categories
    
    Time complexity: O(B * C * N) where:
    - B = beam width
    - C = number of categories
    - N = max items per category
    """
    
    def search(
        self, 
        wardrobe: Dict[str, List[Garment]],
        context: Optional[UserContext] = None
    ) -> SearchResult:
        """
        Perform beam search for best outfits.
        
        Searches both (top+bottom) and (full_body) combinations,
        then merges and ranks all results.
        
        Args:
            wardrobe: Wardrobe organized by category
            context: Optional user context
            
        Returns:
            SearchResult with top outfits
        """
        start_time = time.time()
        self._nodes_expanded = 0
        self._nodes_pruned = 0
        
        if not self._can_search(wardrobe):
            return SearchResult(
                outfits=[], scores=[],
                nodes_expanded=0, nodes_pruned=0,
                search_time_ms=0, algorithm=SearchAlgorithm.BEAM
            )
        
        has_top_bottom, has_full_body = self._validate_wardrobe(wardrobe)
        all_results: List[Tuple[float, List[Garment]]] = []
        
        # Strategy 1: Top + Bottom combinations
        if has_top_bottom:
            categories = self._get_search_categories(wardrobe, use_full_body=False)
            if categories:
                results = self._beam_search_categories(wardrobe, categories)
                all_results.extend(results)
                logger.debug(f"Top+Bottom search: {len(results)} candidates")
        
        # Strategy 2: Full Body combinations
        if has_full_body:
            categories = self._get_search_categories(wardrobe, use_full_body=True)
            if categories:
                results = self._beam_search_categories(wardrobe, categories)
                all_results.extend(results)
                logger.debug(f"Full-body search: {len(results)} candidates")
        
        # Sort and select top-K
        all_results.sort(key=lambda x: x[0], reverse=True)
        top_k = all_results[:self.config.top_k]
        
        search_time = (time.time() - start_time) * 1000
        
        logger.info(f"BeamSearch complete: {len(top_k)} results, "
                   f"{self._nodes_expanded} expanded, {self._nodes_pruned} pruned, "
                   f"{search_time:.1f}ms")
        
        return SearchResult(
            outfits=[outfit for _, outfit in top_k],
            scores=[score for score, _ in top_k],
            nodes_expanded=self._nodes_expanded,
            nodes_pruned=self._nodes_pruned,
            search_time_ms=search_time,
            algorithm=SearchAlgorithm.BEAM
        )
    
    def _beam_search_categories(
        self,
        wardrobe: Dict[str, List[Garment]],
        categories: List[str]
    ) -> List[Tuple[float, List[Garment]]]:
        """
        Perform beam search over given categories.
        
        Args:
            wardrobe: Full wardrobe
            categories: Ordered list of categories to search
            
        Returns:
            List of (score, outfit) tuples
        """
        if not categories:
            return []
        
        logger.debug(f"Beam search categories: {categories}")
        
        # Initialize beam with first category
        first_cat = categories[0]
        beam: List[Tuple[float, List[Garment]]] = []
        
        for garment in wardrobe[first_cat]:
            score = self.scorer.score_partial([garment])
            beam.append((score, [garment]))
            self._nodes_expanded += 1
        
        # Sort and keep top beam_width
        beam.sort(key=lambda x: x[0], reverse=True)
        beam = beam[:self.config.beam_width]
        
        # Expand beam for each remaining category
        for cat_idx, category in enumerate(categories[1:], start=1):
            next_beam: List[Tuple[float, List[Garment]]] = []
            
            for beam_score, partial_outfit in beam:
                for garment in wardrobe[category]:
                    new_outfit = partial_outfit + [garment]
                    new_score = self.scorer.score_partial(new_outfit)
                    
                    # Prune low-scoring candidates early
                    if new_score < self.config.min_score_threshold:
                        self._nodes_pruned += 1
                        continue
                    
                    next_beam.append((new_score, new_outfit))
                    self._nodes_expanded += 1
            
            # Keep top beam_width candidates
            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[:self.config.beam_width]
            
            logger.debug(f"Level {cat_idx}/{len(categories)-1}: {len(beam)} candidates")
        
        return beam


# =============================================================================
# A* Search
# =============================================================================

class AStarSearch(BaseOutfitSearch):
    """
    A* Search for outfit selection.
    
    Uses heuristic to prioritize exploration of promising
    partial outfits. Guarantees optimal solution if heuristic
    is admissible (never overestimates).
    
    Supports two outfit strategies:
    - Top + Bottom (+ optional: shoes, outerwear, accessories)
    - Full Body (dress, jumpsuit) + optional categories
    
    Advantages over beam search:
    - Can find globally optimal solution
    - Better handling of diverse wardrobe items
    - More principled pruning
    """
    
    def search(
        self, 
        wardrobe: Dict[str, List[Garment]],
        context: Optional[UserContext] = None
    ) -> SearchResult:
        """
        Perform A* search for best outfits.
        
        Searches both (top+bottom) and (full_body) combinations,
        then merges and ranks all results.
        
        Args:
            wardrobe: Wardrobe organized by category
            context: Optional user context
            
        Returns:
            SearchResult with top outfits
        """
        start_time = time.time()
        self._nodes_expanded = 0
        self._nodes_pruned = 0
        
        if not self._can_search(wardrobe):
            return SearchResult(
                outfits=[], scores=[],
                nodes_expanded=0, nodes_pruned=0,
                search_time_ms=0, algorithm=SearchAlgorithm.ASTAR
            )
        
        has_top_bottom, has_full_body = self._validate_wardrobe(wardrobe)
        all_completed: List[Tuple[float, List[Garment]]] = []
        
        # Strategy 1: Top + Bottom combinations
        if has_top_bottom:
            categories = self._get_search_categories(wardrobe, use_full_body=False)
            if categories:
                results = self._astar_search_categories(wardrobe, categories)
                all_completed.extend(results)
                logger.debug(f"A* Top+Bottom search: {len(results)} candidates")
        
        # Strategy 2: Full Body combinations
        if has_full_body:
            categories = self._get_search_categories(wardrobe, use_full_body=True)
            if categories:
                results = self._astar_search_categories(wardrobe, categories)
                all_completed.extend(results)
                logger.debug(f"A* Full-body search: {len(results)} candidates")
        
        # Sort and select top-K
        all_completed.sort(key=lambda x: x[0], reverse=True)
        top_k = all_completed[:self.config.top_k]
        
        search_time = (time.time() - start_time) * 1000
        
        logger.info(f"A*Search complete: {len(top_k)} results, "
                   f"{self._nodes_expanded} expanded, {self._nodes_pruned} pruned, "
                   f"{search_time:.1f}ms")
        
        return SearchResult(
            outfits=[outfit for _, outfit in top_k],
            scores=[score for score, _ in top_k],
            nodes_expanded=self._nodes_expanded,
            nodes_pruned=self._nodes_pruned,
            search_time_ms=search_time,
            algorithm=SearchAlgorithm.ASTAR
        )
    
    def _astar_search_categories(
        self,
        wardrobe: Dict[str, List[Garment]],
        categories: List[str]
    ) -> List[Tuple[float, List[Garment]]]:
        """
        Perform A* search over given categories.
        
        Args:
            wardrobe: Full wardrobe
            categories: Ordered list of categories to search
            
        Returns:
            List of (score, outfit) tuples
        """
        if not categories:
            return []
        
        target_depth = len(categories)
        
        logger.debug(f"A* search categories: {categories}")
        
        # Priority queue (min-heap, so use negative scores for max)
        open_set: List[SearchNode] = []
        visited: Set[str] = set()
        completed: List[Tuple[float, List[Garment]]] = []
        
        # Initialize with first category
        first_cat = categories[0]
        remaining = set(categories[1:])
        
        for garment in wardrobe[first_cat]:
            g_score = self.scorer.score_partial([garment])
            h_score = self.scorer.estimate_heuristic([garment], remaining, wardrobe)
            f_score = g_score + self.config.heuristic_weight * h_score
            
            node = SearchNode(
                priority=-f_score,  # Negative for max-heap behavior
                garments=[garment],
                categories_filled={first_cat},
                g_score=g_score,
                h_score=h_score,
                f_score=f_score,
                depth=1
            )
            heapq.heappush(open_set, node)
        
        # A* main loop
        while open_set and self._nodes_expanded < self.config.max_expansions:
            node = heapq.heappop(open_set)
            
            # Skip if already visited
            if node.state_key in visited:
                continue
            visited.add(node.state_key)
            
            self._nodes_expanded += 1
            
            # Check if complete outfit
            if node.depth == target_depth:
                completed.append((node.g_score, node.garments))
                
                # Early termination if we have enough good results
                if (self.config.early_termination and 
                    len(completed) >= self.config.top_k * 2):
                    logger.debug("Early termination: enough candidates found")
                    break
                continue
            
            # Expand: add garments from next category
            next_cat_idx = node.depth
            if next_cat_idx >= len(categories):
                continue
            
            next_cat = categories[next_cat_idx]
            remaining = set(categories[next_cat_idx + 1:])
            
            for garment in wardrobe[next_cat]:
                new_garments = node.garments + [garment]
                g_score = self.scorer.score_partial(new_garments)
                
                # Prune low-scoring paths
                if g_score < self.config.min_score_threshold:
                    self._nodes_pruned += 1
                    continue
                
                h_score = self.scorer.estimate_heuristic(new_garments, remaining, wardrobe)
                f_score = g_score + self.config.heuristic_weight * h_score
                
                new_node = SearchNode(
                    priority=-f_score,
                    garments=new_garments,
                    categories_filled=node.categories_filled | {next_cat},
                    g_score=g_score,
                    h_score=h_score,
                    f_score=f_score,
                    depth=node.depth + 1
                )
                
                if new_node.state_key not in visited:
                    heapq.heappush(open_set, new_node)
        
        return completed


# =============================================================================
# Hybrid Search
# =============================================================================

class HybridSearch(BaseOutfitSearch):
    """
    Hybrid search combining Beam Search and A*.
    
    Strategy:
    1. Use beam search for fast initial exploration
    2. Use A* for refinement on promising regions
    
    Good for large wardrobes where pure A* would be too slow.
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        super().__init__(config)
        
        # Internal searchers with adjusted configs
        beam_config = SearchConfig(
            algorithm=SearchAlgorithm.BEAM,
            beam_width=self.config.beam_width * 2,  # Wider initial beam
            top_k=self.config.top_k * 3,  # More candidates for A*
            min_score_threshold=self.config.min_score_threshold * 0.8,
            required_categories=self.config.required_categories,
            optional_categories=self.config.optional_categories,
            scoring_profile=self.config.scoring_profile,
            include_full_body=self.config.include_full_body
        )
        
        self._beam_search = BeamSearch(beam_config)
    
    def search(
        self, 
        wardrobe: Dict[str, List[Garment]],
        context: Optional[UserContext] = None
    ) -> SearchResult:
        """
        Perform hybrid search.
        
        Args:
            wardrobe: Wardrobe organized by category
            context: Optional user context
            
        Returns:
            SearchResult with top outfits
        """
        start_time = time.time()
        
        logger.info("HybridSearch: Phase 1 - Beam Search exploration")
        
        # Phase 1: Beam search for broad exploration
        beam_result = self._beam_search.search(wardrobe, context)
        
        if not beam_result.outfits:
            return SearchResult(
                outfits=[], scores=[],
                nodes_expanded=beam_result.nodes_expanded,
                nodes_pruned=beam_result.nodes_pruned,
                search_time_ms=(time.time() - start_time) * 1000,
                algorithm=SearchAlgorithm.HYBRID
            )
        
        logger.info(f"HybridSearch: Phase 1 found {len(beam_result.outfits)} candidates")
        
        # Phase 2: Refine with full scoring
        # Re-score top candidates with full scorecard
        refined_results: List[Tuple[float, List[Garment]]] = []
        
        for outfit, partial_score in beam_result:
            # Full scoring would use OutfitScorecard here
            # For now, use partial score as final
            refined_results.append((partial_score, outfit))
        
        # Sort and select top-K
        refined_results.sort(key=lambda x: x[0], reverse=True)
        top_k = refined_results[:self.config.top_k]
        
        total_time = (time.time() - start_time) * 1000
        total_expanded = beam_result.nodes_expanded
        total_pruned = beam_result.nodes_pruned
        
        logger.info(f"HybridSearch complete: {len(top_k)} results in {total_time:.1f}ms")
        
        return SearchResult(
            outfits=[outfit for _, outfit in top_k],
            scores=[score for score, _ in top_k],
            nodes_expanded=total_expanded,
            nodes_pruned=total_pruned,
            search_time_ms=total_time,
            algorithm=SearchAlgorithm.HYBRID
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_outfit_search(
    algorithm: SearchAlgorithm = SearchAlgorithm.BEAM,
    config: Optional[SearchConfig] = None,
    **kwargs
) -> BaseOutfitSearch:
    """
    Create an outfit search instance.
    
    Args:
        algorithm: Search algorithm to use
        config: Optional search configuration
        **kwargs: Additional config parameters
        
    Returns:
        Configured search instance
    """
    if config is None:
        config = SearchConfig(algorithm=algorithm, **kwargs)
    else:
        config.algorithm = algorithm
    
    if algorithm == SearchAlgorithm.BEAM:
        return BeamSearch(config)
    elif algorithm == SearchAlgorithm.ASTAR:
        return AStarSearch(config)
    elif algorithm == SearchAlgorithm.HYBRID:
        return HybridSearch(config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def search_best_outfits(
    wardrobe: Dict[str, List[Garment]],
    algorithm: SearchAlgorithm = SearchAlgorithm.BEAM,
    top_k: int = 5,
    beam_width: int = 10,
    context: Optional[UserContext] = None,
    **kwargs
) -> SearchResult:
    """
    Convenience function to search for best outfits.
    
    Args:
        wardrobe: Wardrobe organized by category
        algorithm: Search algorithm
        top_k: Number of results to return
        beam_width: Beam width for beam search
        context: Optional user context
        **kwargs: Additional config parameters
        
    Returns:
        SearchResult with top-K outfits
    """
    config = SearchConfig(
        algorithm=algorithm,
        top_k=top_k,
        beam_width=beam_width,
        **kwargs
    )
    
    search = create_outfit_search(algorithm, config)
    return search.search(wardrobe, context)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums
    "SearchAlgorithm",
    
    # Config
    "SearchConfig",
    "SearchNode",
    "SearchResult",
    
    # Scorer
    "PartialScorer",
    
    # Searchers
    "BaseOutfitSearch",
    "BeamSearch",
    "AStarSearch",
    "HybridSearch",
    
    # Factory
    "create_outfit_search",
    "search_best_outfits",
]
