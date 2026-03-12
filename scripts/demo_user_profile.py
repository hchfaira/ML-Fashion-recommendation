#!/usr/bin/env python3
"""
User Profile Extraction Demo Script
====================================

Demonstrates the User Style Profile Extraction Pipeline.
Extracts styling-relevant attributes from a user photo.

Usage:
    python scripts/demo_user_profile.py <image_path> [--height HEIGHT_CM] [--weight WEIGHT_KG]
    
Examples:
    python scripts/demo_user_profile.py data/sample_wardrobe/image.png
    python scripts/demo_user_profile.py photo.jpg --height 168 --weight 60
    python scripts/demo_user_profile.py photo.jpg --height 175 --weight 70 --output profile.json
    python scripts/demo_user_profile.py photo.jpg --hybrid          # run both engines on a sample wardrobe
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.layer3_context.user_profile import (
    StyleProfilePipeline,
    PipelineConfig,
    extract_style_profile,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract style profile from a user photo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg
  %(prog)s photo.jpg --height 168 --weight 60
  %(prog)s photo.jpg --minimal
  %(prog)s photo.jpg --output profile.json
        """,
    )
    
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the user photo",
    )
    
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="User height in centimeters (optional)",
    )
    
    parser.add_argument(
        "--weight",
        type=float,
        default=None,
        help="User weight in kilograms (optional)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (optional, prints to stdout if not specified)",
    )
    
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Run minimal analysis (color only, faster)",
    )
    
    parser.add_argument(
        "--no-body",
        action="store_true",
        help="Disable body analysis",
    )
    
    parser.add_argument(
        "--no-face",
        action="store_true",
        help="Disable face analysis",
    )
    
    parser.add_argument(
        "--no-hair",
        action="store_true",
        help="Disable hair analysis",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including timing info",
    )
    
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help=(
            "After extracting the style profile, feed it into both "
            "StyleIntelligenceModel AND ContextEngine via HybridOutfitRecommender "
            "and show how a sample wardrobe is ranked by each engine."
        ),
    )
    
    return parser.parse_args()


async def run_hybrid_engines(profile):
    """
    Feed a StyleProfile into HybridOutfitRecommender and show how a sample
    wardrobe is ranked by StyleIntelligenceModel vs ContextEngine vs Combined.
    """
    from src.layer2_style.hybrid_recommender import HybridOutfitRecommender
    from src.layer2_style.season_color_harmony import ColorSeason
    from src.layer2_style.volume_balance_scorer import BodyShape as VolumeBodyShape
    from src.core.models import (
        Garment, GarmentAttributes, GarmentCategory,
        ColorInfo, PatternInfo, UserContext, Occasion, FormalityLevel,
    )
    from uuid import uuid4
    import time

    print()
    print("=" * 64)
    print("  🤖 Hybrid Engine Comparison — Style × Context")
    print("=" * 64)

    # ── Derive personalisation params from profile ────────────────────
    color_season = None
    body_shape_vol = None

    if profile.skin_analysis and profile.skin_analysis.undertone:
        undertone = profile.skin_analysis.undertone.value.lower()
        season_map = {
            "warm": ColorSeason.AUTUMN,
            "cool": ColorSeason.WINTER,
            "neutral": ColorSeason.SPRING,
        }
        color_season = season_map.get(undertone, ColorSeason.SPRING)
        print(f"\n🎨 Profile → colour season : {color_season.value if hasattr(color_season, 'value') else color_season}")

    if profile.body_metrics and profile.body_metrics.body_shape:
        shape_val = profile.body_metrics.body_shape.value.upper().replace(" ", "_")
        try:
            body_shape_vol = VolumeBodyShape[shape_val]
            print(f"👤 Profile → body shape   : {body_shape_vol.value if hasattr(body_shape_vol, 'value') else body_shape_vol}")
        except KeyError:
            pass

    # ── Sample wardrobe ───────────────────────────────────────────────
    def _mk(cat, sub, col, formality="casual"):
        return Garment(
            id=str(uuid4()),
            attributes=GarmentAttributes(
                category=GarmentCategory(cat),
                subcategory=sub,
                color=ColorInfo(primary=col, hex_codes=[]),
                pattern=PatternInfo(type="solid"),
                formality_level=formality,
                season_suitable=["spring", "summer", "fall", "winter"],
                fit="regular",
            ),
        )

    garments = [
        _mk("top", "white t-shirt", "white"),
        _mk("top", "navy blazer", "navy", "smart_casual"),
        _mk("top", "gray sweater", "gray"),
        _mk("bottom", "blue jeans", "blue"),
        _mk("bottom", "black trousers", "black", "smart_casual"),
        _mk("bottom", "beige chinos", "beige"),
        _mk("shoes", "white sneakers", "white"),
        _mk("shoes", "brown leather shoes", "brown", "smart_casual"),
        _mk("outerwear", "black jacket", "black"),
    ]

    # ── Recommender ───────────────────────────────────────────────────
    recommender = HybridOutfitRecommender(style_weight=0.40, context_weight=0.60)
    if color_season or body_shape_vol:
        recommender.set_user_profile(color_season=color_season, body_shape=body_shape_vol)

    user_context = UserContext(
        occasion=Occasion.CASUAL,
        formality_preference=FormalityLevel.CASUAL,
    )

    print("\n⏳ Scoring combinations …")
    t0 = time.time()
    ranked = await recommender.recommend(garments, user_context, top_k=5)
    elapsed = (time.time() - t0) * 1000
    print(f"   Done in {elapsed:.0f} ms\n")

    # ── Side-by-side table ────────────────────────────────────────────
    print(f"{'Rank':<5} {'Outfit':<35} {'Style':>7} {'Context':>9} {'Combined':>10} {'Grade':>6}")
    print("─" * 76)
    for i, ro in enumerate(ranked, 1):
        items = " + ".join(
            (g.attributes.subcategory or g.attributes.category.value)[:12]
            for g in ro.garments[:3]
        )
        if len(ro.garments) > 3:
            items += f" (+{len(ro.garments)-3})"
        marker = " ✅" if i == 1 else ""
        print(
            f"{i:<5} {items:<35} {ro.score.style_score:>6.0%}  "
            f"{ro.score.context_score:>8.0%}  {ro.score.combined_score:>9.0%}  "
            f"{ro.score.grade:>5}{marker}"
        )

    # ── Insight ───────────────────────────────────────────────────────
    if ranked:
        top = ranked[0]
        s = top.score
        print()
        print("📊 What this means for YOU:")
        if s.style_score > s.context_score + 0.1:
            print(
                "   The outfit scores higher on pure aesthetic rules than on "
                "contextual fit for your body/occasion — a visually strong but "
                "potentially less personalised choice."
            )
        elif s.context_score > s.style_score + 0.1:
            print(
                "   The outfit scores higher on contextual fit (body shape, "
                "occasion, colour season) than on pure style — very personal "
                "but might sacrifice a little aesthetic complexity."
            )
        else:
            print(
                "   Both engines agree — this outfit balances aesthetic quality "
                "and personalised fit equally well. ✨"
            )
        if s.strengths:
            print(f"   ✅ Strengths: {', '.join(s.strengths[:3])}")
        if s.improvements:
            print(f"   💡 Tips: {', '.join(s.improvements[:3])}")
    print()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"🖼️  Analyzing: {image_path}")
    
    if args.height:
        print(f"📏 Height: {args.height} cm")
    if args.weight:
        print(f"⚖️  Weight: {args.weight} kg")
    
    print()
    
    # Configure pipeline
    if args.minimal:
        print("🔧 Mode: Minimal (color analysis only)")
        config = PipelineConfig.minimal()
    else:
        config = PipelineConfig(
            enable_body_analysis=not args.no_body,
            enable_face_analysis=not args.no_face,
            enable_color_analysis=True,  # Always enabled
            enable_hair_analysis=not args.no_hair,
            enable_contrast_analysis=True,  # Always enabled if hair/color
            height_cm=args.height,
            weight_kg=args.weight,
        )
        
        enabled = []
        if config.enable_body_analysis:
            enabled.append("body")
        if config.enable_face_analysis:
            enabled.append("face")
        if config.enable_color_analysis:
            enabled.append("color")
        if config.enable_hair_analysis:
            enabled.append("hair")
        if config.enable_contrast_analysis:
            enabled.append("contrast")
        
        print(f"🔧 Mode: Full ({', '.join(enabled)})")
    
    print()
    print("⏳ Processing...")
    print()
    
    # Run pipeline
    try:
        pipeline = StyleProfilePipeline(config)
        result = pipeline.analyze(
            str(image_path),
            height_cm=args.height,
            weight_kg=args.weight,
        )
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Display results
    print("=" * 60)
    print("📊 STYLE PROFILE RESULTS")
    print("=" * 60)
    print()
    
    profile = result.profile
    
    # Body metrics
    if profile.body_metrics:
        print("👤 Body Analysis:")
        bm = profile.body_metrics
        if bm.body_shape:
            print(f"   Shape: {bm.body_shape.value}")
        if bm.height_cm:
            print(f"   Height: {bm.height_cm} cm")
        if bm.weight_kg:
            print(f"   Weight: {bm.weight_kg} kg")
        if bm.bmi:
            print(f"   BMI: {bm.bmi:.1f} ({bm.bmi_category or 'N/A'})")
        if bm.shoulder_hip_ratio:
            print(f"   Shoulder/Hip Ratio: {bm.shoulder_hip_ratio:.2f}")
        if bm.leg_torso_ratio:
            print(f"   Leg/Torso Ratio: {bm.leg_torso_ratio:.2f}")
        
        # Enhanced measurements (when height provided)
        if bm.estimated_shoulder_cm:
            print()
            print("   📐 Estimated Measurements:")
            print(f"      Shoulder Width: {bm.estimated_shoulder_cm:.1f} cm")
        if bm.estimated_hip_cm:
            print(f"      Hip Width: {bm.estimated_hip_cm:.1f} cm")
        if bm.estimated_torso_cm:
            print(f"      Torso Length: {bm.estimated_torso_cm:.1f} cm")
        if bm.estimated_leg_cm:
            print(f"      Leg Length: {bm.estimated_leg_cm:.1f} cm")
        if bm.estimated_inseam_cm:
            print(f"      Est. Inseam: {bm.estimated_inseam_cm:.1f} cm")
        if bm.estimated_arm_length_cm:
            print(f"      Est. Arm Length: {bm.estimated_arm_length_cm:.1f} cm")
        
        # Proportions
        if bm.torso_proportion or bm.leg_proportion or bm.frame_size:
            print()
            print("   📊 Body Proportions:")
            if bm.torso_proportion:
                print(f"      Torso: {bm.torso_proportion}")
            if bm.leg_proportion:
                print(f"      Legs: {bm.leg_proportion}")
            if bm.frame_size:
                print(f"      Frame Size: {bm.frame_size}")
        
        # Size estimations
        if bm.estimated_top_size or bm.estimated_bottom_size:
            print()
            print("   👕 Estimated Sizes:")
            if bm.estimated_top_size:
                print(f"      Top: {bm.estimated_top_size}")
            if bm.estimated_bottom_size:
                print(f"      Bottom: {bm.estimated_bottom_size}")
            if bm.estimated_dress_size:
                print(f"      Dress: {bm.estimated_dress_size}")
        
        # Styling tips
        if bm.proportion_tips:
            print()
            print("   💡 Proportion Tips:")
            for tip in bm.proportion_tips:
                print(f"      • {tip}")
        
        print()
    
    # Skin analysis
    if profile.skin_analysis:
        print("🎨 Skin Analysis:")
        sa = profile.skin_analysis
        if sa.skin_tone:
            print(f"   Tone: {sa.skin_tone.value}")
        if sa.undertone:
            print(f"   Undertone: {sa.undertone.value}")
        if sa.dominant_skin_rgb:
            r, g, b = sa.dominant_skin_rgb
            print(f"   Dominant Color: RGB({r}, {g}, {b})")
        print()
    
    # Hair analysis
    if profile.hair_analysis:
        print("💇 Hair Analysis:")
        ha = profile.hair_analysis
        if ha.hair_color:
            print(f"   Color: {ha.hair_color.value}")
        if ha.dominant_hair_rgb:
            r, g, b = ha.dominant_hair_rgb
            print(f"   Dominant Color: RGB({r}, {g}, {b})")
        print()
    
    # Contrast
    if profile.contrast_level:
        print("🎭 Contrast Analysis:")
        print(f"   Level: {profile.contrast_level.value}")
        print()
    
    # Visual weight
    if profile.visual_weight:
        print("⚖️  Visual Weight:")
        print(f"   Weight: {profile.visual_weight.value}")
        print()
    
    # Timing info
    if args.verbose:
        print("-" * 60)
        print("⏱️  Timing:")
        print(f"   Total: {result.processing_time_ms:.0f} ms")
        print(f"   Stages: {', '.join(result.stages_completed)}")
        print()
    
    # Output JSON
    profile_dict = profile.to_dict()
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(profile_dict, f, indent=2, default=str)
        print(f"💾 Saved to: {output_path}")
    elif args.verbose:
        print("-" * 60)
        print("📄 Full JSON:")
        print(json.dumps(profile_dict, indent=2, default=str))

    # ── Hybrid engine demo ────────────────────────────────────────────
    if args.hybrid:
        asyncio.run(run_hybrid_engines(profile))

    print()
    print("✅ Done!")


if __name__ == "__main__":
    main()
