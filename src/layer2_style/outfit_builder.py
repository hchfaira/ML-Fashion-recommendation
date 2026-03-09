"""
Outfit Builder
Builds and evaluates outfit combinations from a wardrobe.

This module provides:
- OutfitBuilder: Main class for building and scoring outfits
- OutfitCandidate: Data class for scored outfit candidates
- Combination generation, scoring, and selection logic
- Optimized search using Beam Search and A* algorithms

Architecture:
    Wardrobe
        ↓
    OutfitSearch (Beam Search / A* / Hybrid)
        ↓
    Scorers (lazy evaluation)
        ↓
    TotalStyleScorer
        ↓
    Top-K Outfits
"""
from typing import List, Dict, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from itertools import product
from uuid import uuid4
import asyncio

from src.core.models import Garment, GarmentAttributes, UserContext, Outfit
from src.core import get_logger
from src.layer1_vision.attribute_extractor import AttributeExtractor, ExtractionMode
from .outfit_scorecard import OutfitScorecard
from .outfit_search import (
    SearchAlgorithm,
    SearchConfig,
    SearchResult,
    BeamSearch,
    AStarSearch,
    HybridSearch,
    create_outfit_search,
    search_best_outfits
)

logger = get_logger(__name__)


@dataclass
class OutfitCandidate:
    """
    A candidate outfit with its score.
    
    Attributes:
        garments: List of Garment objects in this outfit
        scorecard: Complete OutfitScorecard with all metrics
        overall_score: Quick access to overall score
        name: Human-readable name for the outfit
    """
    garments: List[Garment]
    scorecard: OutfitScorecard
    overall_score: float
    name: str
    
    def __lt__(self, other: "OutfitCandidate") -> bool:
        """Enable sorting by overall score."""
        return self.overall_score < other.overall_score
    
    def __repr__(self) -> str:
        return f"OutfitCandidate(name='{self.name}', score={self.overall_score:.1%})"


class OutfitBuilder:
    """
    Builds and evaluates outfit combinations from a wardrobe.
    
    Supports two main workflows:
    
    1. Single Outfit Scoring:
       - Load garment images from a folder
       - Extract attributes using Vision API
       - Score the outfit
       - Generate visualizations
    
    2. Best Outfit Selection:
       - Given a wardrobe of garments
       - Generate multiple outfit combinations
       - Score each combination
       - Select the BEST outfit automatically
    
    Usage:
        builder = OutfitBuilder(context)
        
        # From images (async)
        garments = await builder.load_outfit_from_folder(Path("my_outfit/"))
        scorecard = builder.score_outfit(garments)
        
        # From wardrobe
        combinations = builder.generate_outfit_combinations(wardrobe)
        candidates = builder.evaluate_all_combinations(combinations)
        best = builder.select_best_outfit(candidates)
    """
    
    def __init__(self, context: Optional[UserContext] = None):
        """
        Initialize the outfit builder.
        
        Args:
            context: Optional user context for personalized scoring
        """
        self.context = context
        self.extractor = AttributeExtractor()
    
    # ==================== Image Loading ====================
    
    async def extract_garment_from_image(self, image_path: Union[str, Path]) -> Garment:
        """
        Extract garment attributes from a SINGLE garment image using Vision API.
        
        This method is designed for images containing a single garment item.
        For extracting multiple garments from a full outfit image, use
        extract_full_outfit_from_image() instead.
        
        Args:
            image_path: Path to the garment image
            
        Returns:
            Garment object with extracted attributes
        """
        # Use internal single garment extraction for backward compatibility
        attributes = await self.extractor._extract_single_garment(str(image_path))
        return Garment(
            id=str(uuid4()),
            image_path=str(image_path),
            attributes=attributes
        )
    
    async def extract_full_outfit_from_image(self, image_path: Union[str, Path]) -> List[Garment]:
        """
        Extract all garments from a full outfit image.
        
        Uses segmentation to identify and extract attributes for each
        garment visible in the image.
        
        Args:
            image_path: Path to the outfit image
            
        Returns:
            List of Garment objects for all detected garments
        """
        attributes_list = await self.extractor.extract_attributes(
            str(image_path),
            mode=ExtractionMode.FULL_OUTFIT
        )
        
        garments = []
        for attrs in attributes_list:
            garments.append(Garment(
                id=str(uuid4()),
                image_path=str(image_path),
                attributes=attrs
            ))
        return garments
    
    async def load_outfit_from_folder(self, folder_path: Path) -> List[Garment]:
        """
        Load all garment images from a folder as ONE outfit.
        
        Args:
            folder_path: Path to folder containing garment images
            
        Returns:
            List of Garment objects
            
        Raises:
            FileNotFoundError: If folder doesn't exist
            ValueError: If no images found in folder
        """
        if not folder_path.exists():
            raise FileNotFoundError(f"Outfit folder not found: {folder_path}")
        
        # Find all images in the folder
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(folder_path.glob(ext))
        
        if not image_paths:
            raise ValueError(f"No images found in {folder_path}")
        
        logger.info(f"Loading {len(image_paths)} garments from {folder_path.name}/")
        
        garments = []
        for img_path in sorted(image_paths):
            logger.debug(f"Extracting: {img_path.name}...")
            garment = await self.extract_garment_from_image(img_path)
            garments.append(garment)
            logger.debug(f"  → {garment.attributes.category.value}: {garment.attributes.color.primary}")
        
        return garments
    
    async def load_wardrobe(self, wardrobe_path: Path) -> Dict[str, List[Garment]]:
        """
        Load wardrobe organized by category.
        
        Expected structure:
            wardrobe/
              tops/
              bottoms/
              shoes/
              outerwear/  (optional)
              accessories/ (optional)
        
        Args:
            wardrobe_path: Path to wardrobe folder
            
        Returns:
            Dict mapping category name to list of garments
            
        Raises:
            FileNotFoundError: If wardrobe folder doesn't exist
        """
        wardrobe: Dict[str, List[Garment]] = {
            "tops": [],
            "bottoms": [],
            "shoes": [],
            "outerwear": [],
            "accessories": []
        }
        
        if not wardrobe_path.exists():
            raise FileNotFoundError(f"Wardrobe folder not found: {wardrobe_path}")
        
        logger.info(f"Loading wardrobe from {wardrobe_path}/")
        
        for category in wardrobe.keys():
            category_path = wardrobe_path / category
            if category_path.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                    for img_path in category_path.glob(ext):
                        logger.debug(f"Extracting {category}/{img_path.name}...")
                        garment = await self.extract_garment_from_image(img_path)
                        wardrobe[category].append(garment)
        
        # Summary
        total = sum(len(items) for items in wardrobe.values())
        logger.info(f"Wardrobe loaded: {total} items total")
        for cat, items in wardrobe.items():
            if items:
                logger.info(f"  {cat}: {len(items)} items")
        
        return wardrobe
    
    # ==================== Combination Generation ====================
    
    def generate_outfit_combinations(
        self, 
        wardrobe: Dict[str, List[Garment]],
        require_categories: List[str] = None,
        optional_categories: List[str] = None,
        max_combinations: int = 50,
        include_full_body: bool = True
    ) -> List[List[Garment]]:
        """
        Generate all valid outfit combinations from wardrobe.
        
        Outfit rules:
        - Minimum outfit = (top + bottom) OR full_body
        - full_body (dress, jumpsuit, romper) replaces top + bottom
        - Optional: shoes, outerwear, accessories can be added
        
        Args:
            wardrobe: Dict of category -> garments
            require_categories: Categories that must be present in each outfit.
                              Defaults to ["tops", "bottoms"] - minimum for an outfit.
            optional_categories: Categories to include if available.
                              Defaults to ["shoes", "outerwear", "accessories"].
            max_combinations: Maximum number of combinations to generate
            include_full_body: Whether to include full_body items as outfit base
            
        Returns:
            List of outfit combinations (each is a list of garments)
        """
        # Default required: just tops and bottoms (minimum outfit)
        if require_categories is None:
            require_categories = ["tops", "bottoms"]
        
        # Default optional: shoes, outerwear, accessories
        if optional_categories is None:
            optional_categories = ["shoes", "outerwear", "accessories"]
        
        all_outfits = []
        
        # ========== Strategy 1: Top + Bottom combinations ==========
        has_tops = "tops" in wardrobe and wardrobe.get("tops")
        has_bottoms = "bottoms" in wardrobe and wardrobe.get("bottoms")
        
        if has_tops and has_bottoms:
            base_combinations = self._generate_top_bottom_combinations(
                wardrobe, optional_categories
            )
            all_outfits.extend(base_combinations)
            logger.info(f"Generated {len(base_combinations)} top+bottom combinations")
        
        # ========== Strategy 2: Full Body combinations ==========
        has_full_body = "full_body" in wardrobe and wardrobe.get("full_body")
        
        if include_full_body and has_full_body:
            full_body_combinations = self._generate_full_body_combinations(
                wardrobe, optional_categories
            )
            all_outfits.extend(full_body_combinations)
            logger.info(f"Generated {len(full_body_combinations)} full_body combinations")
        
        # Check if any outfits were generated
        if not all_outfits:
            logger.warning("No valid outfit combinations found. "
                          "Need (tops + bottoms) OR full_body items.")
            return []
        
        # Limit combinations
        if len(all_outfits) > max_combinations:
            logger.warning(f"Limiting to {max_combinations} combinations (of {len(all_outfits)} total)")
            all_outfits = all_outfits[:max_combinations]
        
        logger.info(f"Total outfit combinations: {len(all_outfits)}")
        return all_outfits
    
    def _generate_top_bottom_combinations(
        self,
        wardrobe: Dict[str, List[Garment]],
        optional_categories: List[str]
    ) -> List[List[Garment]]:
        """
        Generate combinations with top + bottom as base.
        
        Args:
            wardrobe: Full wardrobe
            optional_categories: Optional categories to include
            
        Returns:
            List of outfit combinations
        """
        tops = wardrobe.get("tops", [])
        bottoms = wardrobe.get("bottoms", [])
        
        if not tops or not bottoms:
            return []
        
        # Build category items list: [tops, bottoms, ...optionals]
        category_items = [tops, bottoms]
        categories_used = ["tops", "bottoms"]
        
        # Add optional categories
        for cat in optional_categories:
            items = wardrobe.get(cat, [])
            if items:
                category_items.append(items)
                categories_used.append(cat)
        
        logger.debug(f"Top+Bottom categories: {categories_used}")
        
        # Generate cartesian product
        all_combinations = list(product(*category_items))
        
        return [list(combo) for combo in all_combinations]
    
    def _generate_full_body_combinations(
        self,
        wardrobe: Dict[str, List[Garment]],
        optional_categories: List[str]
    ) -> List[List[Garment]]:
        """
        Generate combinations with full_body (dress, jumpsuit) as base.
        
        Note: full_body replaces top + bottom, so we don't include them.
        
        Args:
            wardrobe: Full wardrobe
            optional_categories: Optional categories to include
            
        Returns:
            List of outfit combinations
        """
        full_body_items = wardrobe.get("full_body", [])
        
        if not full_body_items:
            return []
        
        # Build category items: [full_body, ...optionals]
        # Note: We exclude "tops" and "bottoms" from optionals
        category_items = [full_body_items]
        categories_used = ["full_body"]
        
        # Add optional categories (but not tops/bottoms)
        excluded = {"tops", "bottoms", "full_body"}
        for cat in optional_categories:
            if cat in excluded:
                continue
            items = wardrobe.get(cat, [])
            if items:
                category_items.append(items)
                categories_used.append(cat)
        
        logger.debug(f"Full-body categories: {categories_used}")
        
        # Generate cartesian product
        all_combinations = list(product(*category_items))
        
        return [list(combo) for combo in all_combinations]
    
    def _get_optional_categories_items(
        self,
        wardrobe: Dict[str, List[Garment]],
        optional_categories: List[str],
        exclude: set = None
    ) -> Tuple[List[List[Garment]], List[str]]:
        """
        Get items from optional categories.
        
        Args:
            wardrobe: Full wardrobe
            optional_categories: List of optional category names
            exclude: Set of category names to exclude
            
        Returns:
            Tuple of (category_items, categories_used)
        """
        if exclude is None:
            exclude = set()
        
        category_items = []
        categories_used = []
        
        for cat in optional_categories:
            if cat in exclude:
                continue
            items = wardrobe.get(cat, [])
            if items:
                category_items.append(items)
                categories_used.append(cat)
        
        return category_items, categories_used
        
        logger.info(f"Generated {len(outfits)} outfit combinations")
        return outfits
    
    # ==================== Scoring ====================
    
    def score_outfit(
        self, 
        garments: List[Garment],
        profile: Optional[str] = None
    ) -> OutfitScorecard:
        """
        Score a single outfit.
        
        Args:
            garments: List of Garment objects forming the outfit
            profile: Optional scoring profile (default, minimalist, creative, business, casual)
            
        Returns:
            OutfitScorecard with all metrics calculated
        """
        scorecard = OutfitScorecard(garments=garments, profile=profile)
        scorecard.calculate_all_scores(self.context)
        return scorecard
    
    def evaluate_all_combinations(
        self, 
        combinations: List[List[Garment]],
        progress_callback: Optional[callable] = None,
        profile: Optional[str] = None
    ) -> List[OutfitCandidate]:
        """
        Score all outfit combinations and return ranked results.
        
        Args:
            combinations: List of outfit combinations to score
            progress_callback: Optional callback(current, total) for progress
            profile: Optional scoring profile (default, minimalist, creative, business, casual)
            
        Returns:
            List of OutfitCandidate sorted by score (highest first)
        """
        candidates = []
        
        logger.info(f"Scoring {len(combinations)} outfits with profile '{profile or 'default'}'...")
        
        if len(combinations) == 0:
            logger.warning("No outfit combinations to score")
            return candidates
        
        for i, garments in enumerate(combinations):
            # Create outfit name from garment subcategories
            name = " + ".join([
                g.attributes.subcategory or g.attributes.category.value 
                for g in garments
            ])
            
            scorecard = self.score_outfit(garments, profile=profile)
            
            candidate = OutfitCandidate(
                garments=garments,
                scorecard=scorecard,
                overall_score=scorecard.scores.get("overall", 0),
                name=name
            )
            candidates.append(candidate)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(combinations))
            elif (i + 1) % 10 == 0:
                logger.debug(f"Scored {i + 1}/{len(combinations)}...")
        
        # Sort by score (highest first)
        candidates.sort(reverse=True)
        
        if candidates:
            logger.info(f"Scoring complete. Best: {candidates[0].overall_score:.1%} ({candidates[0].name})")
        
        return candidates
    
    # ==================== Selection ====================
    
    def select_best_outfit(
        self, 
        candidates: List[OutfitCandidate],
        criteria: str = "overall"
    ) -> OutfitCandidate:
        """
        Select the best outfit based on criteria.
        
        Args:
            candidates: List of scored outfit candidates
            criteria: Which score to optimize for:
                     "overall", "color_harmony", "seven_point", 
                     "creativity", "pattern_mixing", etc.
                     
        Returns:
            Best OutfitCandidate based on criteria
            
        Raises:
            ValueError: If no candidates provided
        """
        if not candidates:
            raise ValueError("No candidates to select from")
        
        if criteria == "overall":
            # Already sorted by overall
            return candidates[0]
        else:
            # Sort by specific criteria
            return max(
                candidates, 
                key=lambda c: c.scorecard.scores.get(criteria, 0)
            )
    
    def select_top_n(
        self, 
        candidates: List[OutfitCandidate],
        n: int = 5,
        criteria: str = "overall"
    ) -> List[OutfitCandidate]:
        """
        Select top N outfits based on criteria.
        
        Args:
            candidates: List of scored outfit candidates
            n: Number of top candidates to return
            criteria: Which score to optimize for
            
        Returns:
            List of top N OutfitCandidate
        """
        if criteria == "overall":
            return candidates[:n]
        else:
            sorted_candidates = sorted(
                candidates,
                key=lambda c: c.scorecard.scores.get(criteria, 0),
                reverse=True
            )
            return sorted_candidates[:n]
    
    # ==================== Optimized Search ====================
    
    def search_best_outfits(
        self,
        wardrobe: Dict[str, List[Garment]],
        algorithm: SearchAlgorithm = SearchAlgorithm.BEAM,
        top_k: int = 5,
        beam_width: int = 10,
        profile: Optional[str] = None
    ) -> List[OutfitCandidate]:
        """
        Search for best outfits using optimized search algorithms.
        
        Uses Beam Search, A*, or Hybrid algorithms for efficient
        exploration without exhaustive enumeration.
        
        Args:
            wardrobe: Dict of category -> garments
            algorithm: Search algorithm (BEAM, ASTAR, HYBRID)
            top_k: Number of top results to return
            beam_width: Beam width for beam search
            profile: Scoring profile to use
            
        Returns:
            List of top OutfitCandidate sorted by score
        """
        config = SearchConfig(
            algorithm=algorithm,
            top_k=top_k,
            beam_width=beam_width,
            scoring_profile=profile
        )
        
        searcher = create_outfit_search(algorithm, config)
        result = searcher.search(wardrobe, self.context)
        
        logger.info(f"Search completed: {len(result.outfits)} outfits found "
                   f"({result.nodes_expanded} nodes, {result.search_time_ms:.1f}ms)")
        
        # Convert to OutfitCandidate with full scoring
        candidates = []
        for outfit, partial_score in result:
            name = " + ".join([
                g.attributes.subcategory or g.attributes.category.value 
                for g in outfit
            ])
            
            # Full scoring
            scorecard = self.score_outfit(outfit, profile=profile)
            
            candidate = OutfitCandidate(
                garments=outfit,
                scorecard=scorecard,
                overall_score=scorecard.scores.get("overall", 0),
                name=name
            )
            candidates.append(candidate)
        
        # Re-sort by full score
        candidates.sort(reverse=True)
        
        return candidates
    
    def beam_search_outfits(
        self,
        wardrobe: Dict[str, List[Garment]],
        beam_width: int = 10,
        top_k: int = 5,
        profile: Optional[str] = None
    ) -> List[OutfitCandidate]:
        """
        Find best outfits using Beam Search.
        
        Efficient greedy search that maintains top candidates at each level.
        
        Args:
            wardrobe: Dict of category -> garments
            beam_width: Number of candidates to keep at each level
            top_k: Number of final results
            profile: Scoring profile
            
        Returns:
            List of top OutfitCandidate
        """
        return self.search_best_outfits(
            wardrobe,
            algorithm=SearchAlgorithm.BEAM,
            top_k=top_k,
            beam_width=beam_width,
            profile=profile
        )
    
    def astar_search_outfits(
        self,
        wardrobe: Dict[str, List[Garment]],
        max_expansions: int = 1000,
        top_k: int = 5,
        profile: Optional[str] = None
    ) -> List[OutfitCandidate]:
        """
        Find best outfits using A* Search.
        
        Heuristic-guided search for optimal solutions.
        
        Args:
            wardrobe: Dict of category -> garments
            max_expansions: Maximum nodes to expand
            top_k: Number of final results
            profile: Scoring profile
            
        Returns:
            List of top OutfitCandidate
        """
        config = SearchConfig(
            algorithm=SearchAlgorithm.ASTAR,
            max_expansions=max_expansions,
            top_k=top_k,
            scoring_profile=profile
        )
        
        searcher = create_outfit_search(SearchAlgorithm.ASTAR, config)
        result = searcher.search(wardrobe, self.context)
        
        logger.info(f"A* Search: {result.nodes_expanded} nodes, "
                   f"{result.nodes_pruned} pruned, {result.search_time_ms:.1f}ms")
        
        # Convert to OutfitCandidate with full scoring
        candidates = []
        for outfit, _ in result:
            name = " + ".join([
                g.attributes.subcategory or g.attributes.category.value 
                for g in outfit
            ])
            
            scorecard = self.score_outfit(outfit, profile=profile)
            
            candidate = OutfitCandidate(
                garments=outfit,
                scorecard=scorecard,
                overall_score=scorecard.scores.get("overall", 0),
                name=name
            )
            candidates.append(candidate)
        
        candidates.sort(reverse=True)
        return candidates
    
    def find_best_outfit_fast(
        self,
        wardrobe: Dict[str, List[Garment]],
        profile: Optional[str] = None
    ) -> OutfitCandidate:
        """
        Quickly find the single best outfit using beam search.
        
        Optimized for speed when you only need the top result.
        
        Args:
            wardrobe: Dict of category -> garments
            profile: Scoring profile
            
        Returns:
            Best OutfitCandidate
            
        Raises:
            ValueError: If no valid outfit found
        """
        candidates = self.beam_search_outfits(
            wardrobe,
            beam_width=5,
            top_k=1,
            profile=profile
        )
        
        if not candidates:
            raise ValueError("No valid outfit combinations found")
        
        return candidates[0]
    
    # ==================== Reporting ====================
    
    def generate_selection_report(
        self, 
        candidates: List[OutfitCandidate],
        top_n: int = 5,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a report of top outfit selections.
        
        Args:
            candidates: List of scored candidates
            top_n: Number of top outfits to include
            save_path: Optional path to save report
            
        Returns:
            Report as string
        """
        lines = [
            "=" * 70,
            "🏆 OUTFIT SELECTION REPORT",
            "=" * 70,
            f"\nTotal combinations evaluated: {len(candidates)}",
            f"Top {min(top_n, len(candidates))} outfits:\n",
        ]
        
        for i, candidate in enumerate(candidates[:top_n], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            lines.append(f"{medal} {candidate.name}")
            lines.append(f"   Overall Score: {candidate.overall_score:.1%}")
            lines.append(f"   Grade: {candidate.scorecard.get_grade()}")
            
            # Key scores
            scores = candidate.scorecard.scores
            lines.append(f"   └─ Color: {scores.get('color_harmony', 0):.0%} | "
                        f"7-Point: {scores.get('seven_point', 0):.0%} | "
                        f"Pattern: {scores.get('pattern_mixing', 0):.0%}")
            lines.append("")
        
        # Comparison stats
        if len(candidates) > 1:
            scores = [c.overall_score for c in candidates]
            lines.append("-" * 40)
            lines.append("Statistics:")
            lines.append(f"   Best:  {max(scores):.1%}")
            lines.append(f"   Worst: {min(scores):.1%}")
            lines.append(f"   Avg:   {sum(scores)/len(scores):.1%}")
            lines.append(f"   Range: {max(scores) - min(scores):.1%}")
        
        lines.append("\n" + "=" * 70)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {save_path}")
        
        return report
    
    def plot_top_outfits(
        self, 
        candidates: List[OutfitCandidate],
        top_n: int = 5,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualize comparison of top outfits.
        
        Args:
            candidates: List of scored candidates
            top_n: Number of top outfits to compare
            save_path: Optional path to save chart
            show: Whether to display chart
        """
        top_candidates = candidates[:top_n]
        
        # Create comparison using first scorecard's method
        if len(top_candidates) >= 2:
            first = top_candidates[0].scorecard
            others = [c.scorecard for c in top_candidates[1:]]
            names = [f"#{i+1} ({c.overall_score:.0%})" for i, c in enumerate(top_candidates)]
            
            first.plot_comparison(
                other_scorecards=others,
                names=names,
                save_path=save_path,
                show=show
            )
