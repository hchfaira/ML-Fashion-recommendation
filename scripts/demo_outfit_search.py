#!/usr/bin/env python3
"""
Outfit Search Demo Script
=========================

Demonstrates the optimized outfit search algorithms:
- Beam Search
- A* Search  
- Hybrid Search

Compares performance and results with exhaustive search.

Usage:
    python scripts/demo_outfit_search.py [wardrobe_path] [--algorithm beam|astar|hybrid]
    
Examples:
    # Use default sample wardrobe
    python scripts/demo_outfit_search.py
    
    # Use custom wardrobe
    python scripts/demo_outfit_search.py /path/to/wardrobe
    
    # Compare all algorithms
    python scripts/demo_outfit_search.py --compare
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.layer2_style.outfit_builder import OutfitBuilder, OutfitCandidate
from src.layer2_style.outfit_search import (
    SearchAlgorithm,
    SearchConfig,
    SearchResult,
    BeamSearch,
    AStarSearch,
    HybridSearch,
    create_outfit_search,
    search_best_outfits,
)
from src.core.models import Garment, UserContext
from src.core import get_logger

logger = get_logger(__name__)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(result: SearchResult, algorithm_name: str):
    """Print search results."""
    print(f"\n📊 {algorithm_name} Results:")
    print(f"   └─ Outfits found: {len(result.outfits)}")
    print(f"   └─ Nodes expanded: {result.nodes_expanded}")
    print(f"   └─ Nodes pruned: {result.nodes_pruned}")
    print(f"   └─ Search time: {result.search_time_ms:.2f}ms")
    
    if result.outfits:
        print(f"\n   Top outfits:")
        for i, (outfit, score) in enumerate(zip(result.outfits[:5], result.scores[:5]), 1):
            items = " + ".join([
                f"{g.attributes.subcategory or g.attributes.category.value}"
                for g in outfit
            ])
            print(f"   {i}. [{score:.1%}] {items}")


def print_candidate(candidate: OutfitCandidate, rank: int):
    """Print an outfit candidate."""
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
    
    print(f"\n{medal} {candidate.name}")
    print(f"   Overall Score: {candidate.overall_score:.1%}")
    print(f"   Grade: {candidate.scorecard.get_grade()}")
    
    scores = candidate.scorecard.scores
    print(f"   └─ Color Harmony: {scores.get('color_harmony', 0):.0%}")
    print(f"   └─ 7-Point Rule: {scores.get('seven_point', 0):.0%}")
    print(f"   └─ Pattern Mixing: {scores.get('pattern_mixing', 0):.0%}")
    print(f"   └─ Creativity: {scores.get('creativity', 0):.0%}")


async def load_wardrobe(wardrobe_path: Optional[Path]) -> Dict[str, List[Garment]]:
    """Load wardrobe from path or use sample."""
    builder = OutfitBuilder()
    
    if wardrobe_path and wardrobe_path.exists():
        print(f"📂 Loading wardrobe from: {wardrobe_path}")
        wardrobe = await builder.load_wardrobe(wardrobe_path)
    else:
        # Use sample wardrobe
        sample_path = project_root / "tests" / "test_images" / "outfit_builder" / "wardrobe"
        if sample_path.exists():
            print(f"📂 Using sample wardrobe: {sample_path}")
            wardrobe = await builder.load_wardrobe(sample_path)
        else:
            print("⚠️  No wardrobe found. Using mock data.")
            wardrobe = create_mock_wardrobe()
    
    # Print wardrobe summary
    total = sum(len(items) for items in wardrobe.values())
    print(f"\n👕 Wardrobe loaded: {total} items")
    for cat, items in wardrobe.items():
        if items:
            print(f"   └─ {cat}: {len(items)} items")
    
    return wardrobe


def create_mock_wardrobe() -> Dict[str, List[Garment]]:
    """Create mock wardrobe for demo."""
    from src.core.models import GarmentAttributes, GarmentCategory, ColorInfo, PatternInfo
    from uuid import uuid4
    
    def make_garment(cat: str, subcat: str, color: str) -> Garment:
        return Garment(
            id=str(uuid4()),
            attributes=GarmentAttributes(
                category=GarmentCategory(cat),
                subcategory=subcat,
                color=ColorInfo(primary=color, hex_codes=[]),
                pattern=PatternInfo(type="solid"),
                formality_level="casual",
                season_suitable=["spring", "summer"],
                fit="regular"
            )
        )
    
    return {
        "tops": [
            make_garment("top", "t-shirt", "white"),
            make_garment("top", "shirt", "blue"),
            make_garment("top", "sweater", "gray"),
            make_garment("top", "polo", "navy"),
            make_garment("top", "blouse", "pink"),
        ],
        "bottoms": [
            make_garment("bottom", "jeans", "blue"),
            make_garment("bottom", "chinos", "beige"),
            make_garment("bottom", "pants", "black"),
            make_garment("bottom", "shorts", "khaki"),
        ],
        "shoes": [
            make_garment("shoes", "sneakers", "white"),
            make_garment("shoes", "boots", "brown"),
            make_garment("shoes", "loafers", "black"),
        ],
        "outerwear": [
            make_garment("outerwear", "jacket", "black"),
            make_garment("outerwear", "cardigan", "gray"),
        ],
        "accessories": [
            make_garment("accessories", "watch", "silver"),
            make_garment("accessories", "belt", "brown"),
        ],
    }


def run_algorithm_comparison(wardrobe: Dict[str, List[Garment]]):
    """Compare all search algorithms."""
    print_header("Algorithm Comparison")
    
    algorithms = [
        (SearchAlgorithm.BEAM, "Beam Search"),
        (SearchAlgorithm.ASTAR, "A* Search"),
        (SearchAlgorithm.HYBRID, "Hybrid Search"),
    ]
    
    results = []
    
    for algo, name in algorithms:
        config = SearchConfig(
            algorithm=algo,
            beam_width=10,
            max_expansions=500,
            top_k=5,
            required_categories=["tops", "bottoms"],
            optional_categories=["shoes", "outerwear", "accessories"]
        )
        
        search = create_outfit_search(algo, config)
        result = search.search(wardrobe)
        
        results.append((name, result))
        print_result(result, name)
    
    # Performance summary
    print_header("Performance Summary")
    print(f"\n{'Algorithm':<20} {'Outfits':<10} {'Expanded':<12} {'Pruned':<10} {'Time (ms)':<12}")
    print("-" * 64)
    
    for name, result in results:
        print(f"{name:<20} {len(result.outfits):<10} {result.nodes_expanded:<12} "
              f"{result.nodes_pruned:<10} {result.search_time_ms:<12.2f}")


def run_full_scoring_demo(wardrobe: Dict[str, List[Garment]]):
    """Demo with full outfit scoring."""
    print_header("Full Outfit Scoring Demo")
    
    builder = OutfitBuilder()
    
    print("\n🔍 Using Beam Search to find best outfits...")
    start = time.time()
    
    candidates = builder.beam_search_outfits(
        wardrobe,
        beam_width=10,
        top_k=5,
        profile="default"
    )
    
    elapsed = (time.time() - start) * 1000
    print(f"   Search completed in {elapsed:.2f}ms")
    
    print_header("Top 5 Outfits with Full Scoring")
    
    for i, candidate in enumerate(candidates, 1):
        print_candidate(candidate, i)
    
    # Generate report
    if candidates:
        report = builder.generate_selection_report(candidates, top_n=5)
        print("\n" + report)


def run_quick_demo(wardrobe: Dict[str, List[Garment]]):
    """Quick demo finding best outfit."""
    print_header("Quick Best Outfit Demo")
    
    builder = OutfitBuilder()
    
    print("\n⚡ Finding best outfit quickly...")
    start = time.time()
    
    try:
        best = builder.find_best_outfit_fast(wardrobe)
        elapsed = (time.time() - start) * 1000
        
        print(f"\n✅ Found in {elapsed:.2f}ms!")
        print_candidate(best, 1)
    except ValueError as e:
        print(f"\n❌ Error: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Outfit Search Demo")
    parser.add_argument(
        "wardrobe_path",
        nargs="?",
        type=Path,
        help="Path to wardrobe folder"
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        choices=["beam", "astar", "hybrid"],
        default="beam",
        help="Search algorithm to use"
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare all algorithms"
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick demo finding single best outfit"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of top results"
    )
    parser.add_argument(
        "--beam-width",
        "-b",
        type=int,
        default=10,
        help="Beam width for beam search"
    )
    
    args = parser.parse_args()
    
    print_header("🎨 AI Fashion Stylist - Outfit Search Demo")
    
    # Load wardrobe
    wardrobe = await load_wardrobe(args.wardrobe_path)
    
    if not any(wardrobe.values()):
        print("❌ No items in wardrobe!")
        return
    
    # Run demos
    if args.compare:
        run_algorithm_comparison(wardrobe)
    elif args.quick:
        run_quick_demo(wardrobe)
    else:
        run_full_scoring_demo(wardrobe)
    
    print("\n✨ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
