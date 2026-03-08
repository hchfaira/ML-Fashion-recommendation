"""
Report Generator Module
=======================

Utilities for generating formatted console output and demo data.
"""

from src.core.models import Garment, GarmentAttributes, GarmentCategory, ColorProfile, PatternInfo
from src.layer2_style import OutfitScorecard


class ReportGenerator:
    """
    Helper class for generating formatted reports and demo data.
    """
    
    @staticmethod
    def create_demo_garments() -> list:
        """Create a set of demo garments for testing."""
        return [
            Garment(
                id="demo_top_1",
                attributes=GarmentAttributes(
                    category=GarmentCategory.TOP,
                    subcategory="t-shirt",
                    color=ColorProfile(primary="white"),
                    pattern=PatternInfo(type="solid"),
                    fit="regular",
                    formality_level="casual"
                )
            ),
            Garment(
                id="demo_pants_1",
                attributes=GarmentAttributes(
                    category=GarmentCategory.BOTTOM,
                    subcategory="jeans",
                    color=ColorProfile(primary="blue"),
                    pattern=PatternInfo(type="solid"),
                    fit="regular",
                    formality_level="casual"
                )
            ),
            Garment(
                id="demo_shoes_1",
                attributes=GarmentAttributes(
                    category=GarmentCategory.SHOES,
                    subcategory="sneakers",
                    color=ColorProfile(primary="white", secondary="black"),
                    pattern=PatternInfo(type="solid"),
                    fit="regular",
                    formality_level="casual"
                )
            )
        ]
    
    @staticmethod
    def print_scorecard(scorecard: OutfitScorecard, show_details: bool = True) -> None:
        """Print formatted scorecard to console."""
        overall = scorecard.scores.get("overall", 0)
        print(f"\n🎯 Overall Score: {overall:.1%}")
        print(f"📝 Grade: {scorecard.get_grade()}")
        
        scores = scorecard.get_filtered_scores()
        print(f"\n📊 Criteria evaluated ({len(scores)}):")
        for criterion, score in scores.items():
            display_name = scorecard._get_display_name(criterion)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"   {display_name:25} [{bar}] {score:.1%}")
        
        if show_details:
            ReportGenerator._print_details(scorecard)
    
    @staticmethod
    def _print_details(scorecard: OutfitScorecard) -> None:
        """Print score details."""
        details = scorecard.get_filtered_details()
        has_details = any(d for d in details.values() if d)
        
        if not has_details:
            return
        
        print("\n💡 Details:")
        for criterion, detail in details.items():
            if detail:
                display_name = scorecard._get_display_name(criterion)
                if isinstance(detail, dict):
                    print(f"\n   {display_name}:")
                    for k, v in list(detail.items())[:3]:
                        print(f"      • {k}: {v}")
                elif isinstance(detail, list):
                    print(f"\n   {display_name}:")
                    for item in detail[:3]:
                        print(f"      • {item}")


async def run_demo():
    """Run demo with synthetic data."""
    print("\n🎭 DEMO MODE - Using synthetic garments")
    print("=" * 60)
    print("(To test with real images, use --wardrobe or --outfit-image)")
    
    # Create demo garments
    demo_garments = ReportGenerator.create_demo_garments()
    
    print(f"\n✅ Demo outfit with {len(demo_garments)} garments:")
    for g in demo_garments:
        print(f"   • {g.attributes.category.value}: "
              f"{g.attributes.color.primary} {g.attributes.subcategory}")
    
    # Score with different profiles
    profiles = ["default", "minimalist", "casual"]
    
    for profile in profiles:
        print(f"\n{'='*60}")
        print(f"📊 Scoring with profile: {profile}")
        print("=" * 60)
        
        scorecard = OutfitScorecard(garments=demo_garments, profile=profile)
        result = scorecard.calculate_all_scores()
        
        ReportGenerator.print_scorecard(result, show_details=False)
