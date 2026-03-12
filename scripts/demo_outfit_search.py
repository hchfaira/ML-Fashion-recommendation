#!/usr/bin/env python3
"""
Outfit Search Demo Script
=========================

Demonstrates the optimized outfit search algorithms:
- Beam Search
- A* Search  
- Hybrid Search

Also demonstrates the HybridOutfitRecommender that combines
StyleIntelligenceModel (Layer 2) with ContextEngine (Layer 3) to produce
context-aware style rankings with dual scores.

Usage:
    python scripts/demo_outfit_search.py [wardrobe_path] [--algorithm beam|astar|hybrid]
    python scripts/demo_outfit_search.py --hybrid            # dual-engine demo
    python scripts/demo_outfit_search.py --hybrid --compare  # show style vs context breakdown

Examples:
    # Use default sample wardrobe
    python scripts/demo_outfit_search.py

    # Use custom wardrobe
    python scripts/demo_outfit_search.py /path/to/wardrobe

    # Compare all algorithms
    python scripts/demo_outfit_search.py --compare

    # Hybrid (StyleIntelligenceModel + ContextEngine) ranking
    python scripts/demo_outfit_search.py --hybrid

    # Hybrid with user photo for personalised scoring
    python scripts/demo_outfit_search.py --hybrid --user-photo selfie.jpg
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
from src.layer2_style.hybrid_recommender import (
    HybridOutfitRecommender,
    HybridScore,
    RankedOutfit,
)
from src.core.models import Garment, UserContext, Occasion, FormalityLevel
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


def print_ranked_outfit(ranked: "RankedOutfit", rank: int, show_breakdown: bool = False):
    """Print a HybridOutfitRecommender result."""
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
    s = ranked.score
    items = " + ".join(
        g.attributes.subcategory or g.attributes.category.value
        for g in ranked.garments
    )

    print(f"\n{medal} {items}")
    print(f"   Combined Score : {s.combined_score:.1%}  [{s.grade}]")
    print(f"   ├─ Style  ({s.style_weight:.0%})  : {s.style_score:.1%}")
    print(f"   └─ Context({s.context_weight:.0%}) : {s.context_score:.1%}")

    if show_breakdown:
        if s.style_breakdown:
            print("   Style breakdown:")
            for k, v in s.style_breakdown.items():
                if isinstance(v, (int, float)):
                    print(f"      • {k}: {v:.0%}")
        if s.context_breakdown:
            print("   Context breakdown:")
            for k, v in s.context_breakdown.items():
                print(f"      • {k}: {v}")
        if s.strengths:
            print(f"   ✅ Strengths: {', '.join(s.strengths[:2])}")
        if s.improvements:
            print(f"   💡 Tips: {', '.join(s.improvements[:2])}")


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
                season_suitable=["spring", "summer", "fall", "winter"],
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


async def run_hybrid_demo(
    wardrobe: Dict[str, List[Garment]],
    *,
    show_breakdown: bool = False,
    user_photo: Optional[Path] = None,
    top_k: int = 5,
):
    """
    Rank wardrobe combinations using the HybridOutfitRecommender
    (StyleIntelligenceModel + ContextEngine) and display a dual-score table.
    """
    print_header("🤖 Hybrid Ranking — Style × Context")

    # Optionally personalise from a user photo
    color_season = None
    body_shape_enum = None

    if user_photo and user_photo.exists():
        print(f"\n📸 Extracting user profile from: {user_photo}")
        try:
            from src.layer3_context.user_profile import StyleProfilePipeline, PipelineConfig
            from src.layer2_style.season_color_harmony import ColorSeason as CS
            from src.layer2_style.volume_balance_scorer import BodyShape as BS

            pipeline = StyleProfilePipeline(PipelineConfig())
            pr = pipeline.analyze(str(user_photo))
            profile = pr.profile

            if profile.skin_analysis and profile.skin_analysis.undertone:
                undertone = profile.skin_analysis.undertone.value.lower()
                season_map = {
                    "warm": CS.AUTUMN,
                    "cool": CS.WINTER,
                    "neutral": CS.SPRING,
                }
                color_season = season_map.get(undertone, CS.SPRING)
                print(f"   🎨 Detected color season: {color_season.value if hasattr(color_season, 'value') else color_season}")

            if profile.body_metrics and profile.body_metrics.body_shape:
                shape_val = profile.body_metrics.body_shape.value.upper().replace(" ", "_")
                try:
                    body_shape_enum = BS[shape_val]
                    print(f"   👤 Detected body shape: {body_shape_enum.value if hasattr(body_shape_enum, 'value') else body_shape_enum}")
                except KeyError:
                    pass

        except Exception as e:
            print(f"   ⚠️  Could not extract profile: {e}")
    else:
        print("\n💡 Tip: pass --user-photo <selfie.jpg> for personalised scores")

    # Build recommender
    recommender = HybridOutfitRecommender(style_weight=0.40, context_weight=0.60)
    if color_season or body_shape_enum:
        recommender.set_user_profile(color_season=color_season, body_shape=body_shape_enum)

    # Flatten wardrobe to flat list for recommender
    all_garments: List[Garment] = [g for items in wardrobe.values() for g in items]

    # Build a simple UserContext (casual occasion, no weather override)
    user_context = UserContext(
        occasion=Occasion.CASUAL,
        formality_preference=FormalityLevel.CASUAL,
    )

    print(f"\n⏳ Scoring combinations (top {top_k}) …")
    start = time.time()
    ranked_outfits = await recommender.recommend(
        all_garments,
        user_context,
        top_k=top_k,
    )
    elapsed = (time.time() - start) * 1000
    print(f"   Done in {elapsed:.0f} ms — {len(ranked_outfits)} outfits ranked\n")

    # ── Score comparison table ──────────────────────────────────────────
    print(f"{'Rank':<5} {'Outfit':<38} {'Style':>7} {'Context':>9} {'Combined':>10} {'Grade':>6}")
    print("─" * 80)
    for rank, ro in enumerate(ranked_outfits, 1):
        items_str = " + ".join(
            (g.attributes.subcategory or g.attributes.category.value)[:12]
            for g in ro.garments[:3]
        )
        if len(ro.garments) > 3:
            items_str += f" (+{len(ro.garments)-3})"
        marker = " ← ✅ RECOMMENDED" if rank == 1 else ""
        print(
            f"{rank:<5} {items_str:<38} {ro.score.style_score:>6.0%}  "
            f"{ro.score.context_score:>8.0%}  {ro.score.combined_score:>9.0%}  "
            f"{ro.score.grade:>5}{marker}"
        )

    # Detailed breakdown for top 3
    print_header(f"Top {min(3, len(ranked_outfits))} Outfits — Full Breakdown")
    for rank, ro in enumerate(ranked_outfits[:3], 1):
        print_ranked_outfit(ro, rank, show_breakdown=show_breakdown)

    print()


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
        "--hybrid",
        action="store_true",
        help="Run HybridOutfitRecommender demo (StyleIntelligenceModel + ContextEngine)"
    )
    parser.add_argument(
        "--user-photo",
        type=Path,
        default=None,
        metavar="PHOTO",
        help="User selfie for personalised hybrid scoring"
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
    if args.hybrid:
        await run_hybrid_demo(
            wardrobe,
            show_breakdown=args.compare,
            user_photo=args.user_photo,
            top_k=args.top_k,
        )
    elif args.compare:
        run_algorithm_comparison(wardrobe)
    elif args.quick:
        run_quick_demo(wardrobe)
    else:
        run_full_scoring_demo(wardrobe)
    
    print("\n✨ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
