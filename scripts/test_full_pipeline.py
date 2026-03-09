#!/usr/bin/env python3
"""
Full Pipeline Test Script
=========================

This script demonstrates the complete workflow:
1. Load garment images from a folder (your wardrobe)
2. Extract attributes using Vision API (Layer 1)
3. Generate outfit combinations
4. Score each combination (Layer 2)
5. Apply context-aware scoring (Layer 3) - including user profile from photo
6. Select the BEST outfit
7. Generate a detailed report

Usage:
    # Test with a folder of individual garment images
    python scripts/test_full_pipeline.py --wardrobe path/to/wardrobe_folder
    
    # Test with user profile photo for context-aware recommendations
    python scripts/test_full_pipeline.py --wardrobe path/to/wardrobe_folder --user-photo path/to/selfie.jpg
    
    # Test with user profile and measurements
    python scripts/test_full_pipeline.py --wardrobe ./wardrobe --user-photo selfie.jpg --height 170 --weight 65
    
    # Test with a single outfit image (full outfit photo)
    python scripts/test_full_pipeline.py --outfit-image path/to/outfit.jpg
    
    # Test with sample images
    python scripts/test_full_pipeline.py --demo
    
    # Save results for analysis
    python scripts/test_full_pipeline.py --wardrobe ./wardrobe --save-results

Requirements:
    - GOOGLE_API_KEY environment variable set (for Gemini Vision)
    - Images in supported formats (jpg, jpeg, png, webp)
"""
import asyncio
import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import get_logger

# Import from the pipeline package
from pipeline import (
    ResultsStorage,
    FullPipelineTester,
    run_demo,
)

# Import visualization from Layer 5
from src.layer5_visualization import OutfitVisualizer, create_outfit_image

# Import user profile pipeline (Layer 3)
from src.layer3_context.user_profile import (
    StyleProfilePipeline,
    PipelineConfig,
    StyleProfile,
)

# Import context engine components
from src.layer3_context import (
    ContextEngine,
    ContextCriteria,
    FitPredictor,
    ProportionHarmonizer,
    ColorHarmonyAdvisor,
)

# Import for demo
from src.layer2_style import OutfitScorecard
from scripts.pipeline.report_generator import ReportGenerator

logger = get_logger(__name__)


async def run_demo_with_result() -> Dict[str, Any]:
    """
    Run demo with synthetic data and return structured result.
    This allows context scoring to be applied to demo outfits.
    """
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
    best_result = None
    
    for profile in profiles:
        print(f"\n{'='*60}")
        print(f"📊 Scoring with profile: {profile}")
        print("=" * 60)
        
        scorecard = OutfitScorecard(garments=demo_garments, profile=profile)
        result = scorecard.calculate_all_scores()
        
        ReportGenerator.print_scorecard(result, show_details=False)
        
        # Keep the default profile result for context scoring
        if profile == "default":
            best_result = result
    
    # Return structured result for context scoring
    return {
        "best_outfit": {
            "name": "Demo Outfit",
            "overall_score": best_result.scores.get("overall", 0.5) if best_result else 0.5,
            "garments": [
                {
                    "id": g.id,
                    "category": g.attributes.category.value,
                    "color": g.attributes.color.primary if g.attributes.color else "unknown",
                    "size": getattr(g.attributes, "size", None),
                    "fit": getattr(g.attributes, "fit", "regular"),
                }
                for g in demo_garments
            ]
        },
        "profiles_tested": profiles
    }


def extract_user_profile(
    photo_path: Path,
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    verbose: bool = True
) -> Optional[StyleProfile]:
    """
    Extract user style profile from a photo.
    
    Args:
        photo_path: Path to user photo (selfie)
        height_cm: Optional height in cm
        weight_kg: Optional weight in kg
        verbose: Whether to print extraction progress
        
    Returns:
        StyleProfile or None if extraction failed
    """
    if verbose:
        print(f"\n👤 Extracting user profile from: {photo_path}")
        print("=" * 60)
    
    try:
        # Configure pipeline
        config = PipelineConfig(
            enable_body_analysis=True,
            enable_face_analysis=True,
            enable_color_analysis=True,
            enable_hair_analysis=True,
            enable_contrast_analysis=True,
            height_cm=height_cm,
            weight_kg=weight_kg,
        )
        
        if verbose:
            print("   🔧 Analyzing: body, face, color, hair, contrast")
            if height_cm:
                print(f"   📏 Height: {height_cm} cm")
            if weight_kg:
                print(f"   ⚖️  Weight: {weight_kg} kg")
            print()
        
        # Run pipeline
        pipeline = StyleProfilePipeline(config)
        result = pipeline.analyze(
            str(photo_path),
            height_cm=height_cm,
            weight_kg=weight_kg,
        )
        
        profile = result.profile
        
        if verbose:
            print("   ✅ Profile extracted successfully!")
            print()
            
            # Display key profile info
            if profile.body_metrics:
                bm = profile.body_metrics
                print("   👤 Body Analysis:")
                if bm.body_shape:
                    print(f"      Shape: {bm.body_shape.value}")
                if bm.shoulder_hip_ratio:
                    print(f"      Shoulder/Hip Ratio: {bm.shoulder_hip_ratio:.2f}")
                if bm.frame_size:
                    print(f"      Frame Size: {bm.frame_size}")
                if bm.estimated_top_size:
                    print(f"      Est. Top Size: {bm.estimated_top_size}")
                if bm.estimated_bottom_size:
                    print(f"      Est. Bottom Size: {bm.estimated_bottom_size}")
            
            if profile.skin_analysis:
                sa = profile.skin_analysis
                print("   🎨 Skin Analysis:")
                if sa.skin_tone:
                    print(f"      Skin Tone: {sa.skin_tone.value}")
                if sa.undertone:
                    print(f"      Undertone: {sa.undertone.value}")
            
            if profile.hair_analysis:
                ha = profile.hair_analysis
                print("   💇 Hair Analysis:")
                if ha.hair_color:
                    print(f"      Hair Color: {ha.hair_color.value}")
            
            if profile.contrast_level:
                print(f"   ⚡ Contrast Level: {profile.contrast_level.value}")
            
            print()
        
        return profile
        
    except Exception as e:
        logger.warning(f"Could not extract user profile: {e}")
        if verbose:
            print(f"   ⚠️  Warning: Could not extract profile: {e}")
        return None


def display_context_recommendations(
    style_profile: StyleProfile,
    color_advisor: ColorHarmonyAdvisor,
    proportion_harmonizer: ProportionHarmonizer
):
    """Display context-aware recommendations based on user profile."""
    print("\n💡 Context-Aware Recommendations")
    print("=" * 60)
    
    # Color recommendations
    if style_profile.skin_analysis:
        print("\n🎨 Color Recommendations (based on your coloring):")
        
        skin = style_profile.skin_analysis
        hair = style_profile.hair_analysis
        
        color_profile = color_advisor.create_color_profile(
            skin_tone=str(skin.skin_tone.value) if skin.skin_tone else "MEDIUM",
            undertone=str(skin.undertone.value) if skin.undertone else "NEUTRAL",
            hair_color=str(hair.hair_color.value) if hair and hair.hair_color else "BROWN",
            contrast_level=str(style_profile.contrast_level.value) if style_profile.contrast_level else "MEDIUM"
        )
        
        recs = color_advisor.get_color_recommendations(color_profile)
        
        print(f"   Color Season: {color_profile.season}")
        print(f"   Best Colors: {', '.join(recs.best_colors[:5])}")
        print(f"   Neutral Colors: {', '.join(recs.neutral_colors[:4])}")
        if recs.colors_to_avoid:
            print(f"   Colors to Avoid: {', '.join(recs.colors_to_avoid[:4])}")
        
        if recs.tips:
            print("\n   💡 Color Tips:")
            for tip in recs.tips[:2]:
                print(f"      • {tip}")
    
    # Proportion recommendations
    if style_profile.body_metrics:
        bm = style_profile.body_metrics
        print("\n📐 Proportion Recommendations (based on your body):")
        
        analysis = proportion_harmonizer.analyze_proportions(
            torso_ratio=getattr(bm, 'torso_leg_ratio', None),
            leg_ratio=None,  # Will use torso to infer
            frame_size=getattr(bm, 'frame_size', None),
            shoulder_hip_ratio=getattr(bm, 'shoulder_hip_ratio', None)
        )
        
        print(f"   Torso Proportion: {analysis.torso_proportion}")
        print(f"   Leg Proportion: {analysis.leg_proportion}")
        print(f"   Frame Size: {analysis.frame_size}")
        
        if analysis.recommended_silhouettes:
            print(f"   Recommended Silhouettes: {', '.join(analysis.recommended_silhouettes[:4])}")
        if analysis.avoid_silhouettes:
            print(f"   Silhouettes to Avoid: {', '.join(analysis.avoid_silhouettes[:3])}")
        
        tips = proportion_harmonizer.get_proportion_tips(
            analysis.torso_proportion,
            analysis.leg_proportion
        )
        if tips:
            print("\n   💡 Styling Tips:")
            for tip in tips[:2]:
                print(f"      • {tip}")
    
    # Size recommendations
    if style_profile.body_metrics:
        bm = style_profile.body_metrics
        if bm.estimated_top_size or bm.estimated_bottom_size:
            print("\n👕 Size Recommendations:")
            if bm.estimated_top_size:
                print(f"   Estimated Top Size: {bm.estimated_top_size}")
            if bm.estimated_bottom_size:
                print(f"   Estimated Bottom Size: {bm.estimated_bottom_size}")
    
    print()


async def apply_context_scoring(
    result: Dict[str, Any],
    style_profile: StyleProfile,
    body_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Apply context-aware scoring to outfit results using user profile.
    
    Args:
        result: Original pipeline result with best_outfit
        style_profile: User's style profile
        body_type: Optional override for body type
        
    Returns:
        Enhanced result with context scores
    """
    print("\n🎯 Applying Context-Aware Scoring")
    print("=" * 60)
    
    # Initialize context components
    fit_predictor = FitPredictor()
    proportion_harmonizer = ProportionHarmonizer()
    color_advisor = ColorHarmonyAdvisor()
    
    # Get best outfit garments
    best_outfit = result.get("best_outfit", {})
    garments_data = best_outfit.get("garments", [])
    
    if not garments_data:
        print("   ⚠️  No garments in best outfit to score")
        return result
    
    # Create mock garment objects for scoring
    from unittest.mock import MagicMock
    garments = []
    for g_data in garments_data:
        garment = MagicMock()
        garment.id = g_data.get("id", "unknown")
        garment.category = g_data.get("category", "top")
        garment.color = g_data.get("color", "unknown")
        garment.attributes = {
            "size": g_data.get("size"),
            "fit": g_data.get("fit"),
            "color": g_data.get("color"),
            "waist": g_data.get("waist_type"),
            "pattern": g_data.get("pattern"),
            "weight": g_data.get("weight"),
        }
        garments.append(garment)
    
    context_scores = {}
    
    # 1. Fit prediction
    if style_profile.body_metrics:
        bm = style_profile.body_metrics
        fit_result = fit_predictor.predict_outfit_fit(
            garments=garments,
            user_top_size=getattr(bm, 'estimated_top_size', None),
            user_bottom_size=getattr(bm, 'estimated_bottom_size', None),
            user_bmi_category=getattr(bm, 'bmi_category', None)
        )
        context_scores["fit"] = {
            "score": fit_result.overall_score,
            "fit_type": fit_result.average_fit_type,
            "notes": fit_result.fit_notes
        }
        print(f"   👕 Fit Score: {fit_result.overall_score:.2f} ({fit_result.average_fit_type})")
    
    # 2. Proportion harmony
    if style_profile.body_metrics:
        bm = style_profile.body_metrics
        analysis = proportion_harmonizer.analyze_proportions(
            torso_ratio=getattr(bm, 'torso_leg_ratio', None),
            leg_ratio=None,
            frame_size=getattr(bm, 'frame_size', None),
            shoulder_hip_ratio=getattr(bm, 'shoulder_hip_ratio', None)
        )
        prop_score = proportion_harmonizer.score_outfit_harmony(garments, analysis)
        context_scores["proportions"] = {
            "score": prop_score.score,
            "harmony_level": prop_score.harmony_level,
            "positive_factors": prop_score.positive_factors,
            "negative_factors": prop_score.negative_factors,
            "tips": prop_score.styling_tips
        }
        print(f"   📐 Proportion Score: {prop_score.score:.2f} ({prop_score.harmony_level})")
    
    # 3. Color harmony
    if style_profile.skin_analysis:
        skin = style_profile.skin_analysis
        hair = style_profile.hair_analysis
        
        color_profile = color_advisor.create_color_profile(
            skin_tone=str(skin.skin_tone.value) if skin.skin_tone else "MEDIUM",
            undertone=str(skin.undertone.value) if skin.undertone else "NEUTRAL",
            hair_color=str(hair.hair_color.value) if hair and hair.hair_color else "BROWN",
            contrast_level=str(style_profile.contrast_level.value) if style_profile.contrast_level else "MEDIUM"
        )
        
        color_score = color_advisor.score_outfit_colors(garments, color_profile)
        context_scores["color_harmony"] = {
            "score": color_score.score,
            "harmony_level": color_score.harmony_level,
            "matching_colors": color_score.matching_colors,
            "clashing_colors": color_score.clashing_colors,
            "notes": color_score.notes
        }
        print(f"   🎨 Color Harmony Score: {color_score.score:.2f} ({color_score.harmony_level})")
    
    # Calculate overall context score
    if context_scores:
        scores = [s["score"] for s in context_scores.values() if "score" in s]
        overall_context_score = sum(scores) / len(scores) if scores else 0.7
        context_scores["overall_context_score"] = overall_context_score
        print(f"\n   ⭐ Overall Context Score: {overall_context_score:.2f}")
        
        # Blend with original style score
        original_score = best_outfit.get("overall_score", 0.7)
        blended_score = 0.6 * original_score + 0.4 * overall_context_score
        context_scores["blended_score"] = blended_score
        print(f"   🔄 Blended Score (60% style + 40% context): {blended_score:.2f}")
    
    # Add context scores to result
    result["context_analysis"] = context_scores
    result["best_outfit"]["context_adjusted_score"] = context_scores.get("blended_score", best_outfit.get("overall_score", 0.7))
    
    print()
    
    return result


async def main():
    parser = argparse.ArgumentParser(
        description="Test the full outfit recommendation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with synthetic data
  python scripts/test_full_pipeline.py --demo
  
  # Test with a wardrobe folder (each image = one garment)
  python scripts/test_full_pipeline.py --wardrobe ./my_wardrobe/
  
  # Test with user profile photo for context-aware recommendations
  python scripts/test_full_pipeline.py --wardrobe ./wardrobe/ --user-photo selfie.jpg
  
  # Test with user profile and measurements
  python scripts/test_full_pipeline.py --wardrobe ./wardrobe/ --user-photo selfie.jpg --height 170 --weight 65
  
  # Test with a single outfit image
  python scripts/test_full_pipeline.py --outfit-image ./outfit.jpg
  
  # Test specific garment images
  python scripts/test_full_pipeline.py --images top.jpg pants.jpg shoes.jpg
  
  # Use a different scoring profile
  python scripts/test_full_pipeline.py --wardrobe ./wardrobe/ --profile business
  
  # Save results for analysis (garments JSON, graphs, reports)
  python scripts/test_full_pipeline.py --wardrobe ./wardrobe/ --save-results
  
  # Specify custom output directory for results
  python scripts/test_full_pipeline.py --wardrobe ./wardrobe/ --save-results --results-dir ./my_results/
        """
    )
    
    # Input options (mutually exclusive in practice)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with synthetic garments (no API calls)"
    )
    parser.add_argument(
        "--wardrobe",
        type=Path,
        help="Path to wardrobe folder containing garment images"
    )
    parser.add_argument(
        "--outfit-image",
        type=Path,
        help="Path to a single outfit image to analyze"
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="List of individual garment image paths"
    )
    
    # User profile options (for context-aware recommendations)
    user_profile_group = parser.add_argument_group('User Profile (for context-aware recommendations)')
    user_profile_group.add_argument(
        "--user-photo",
        type=Path,
        help="Path to user photo for profile extraction (enables context-aware scoring)"
    )
    user_profile_group.add_argument(
        "--height",
        type=float,
        help="User height in centimeters (used with --user-photo)"
    )
    user_profile_group.add_argument(
        "--weight",
        type=float,
        help="User weight in kilograms (used with --user-photo)"
    )
    user_profile_group.add_argument(
        "--body-type",
        type=str,
        choices=["rectangle", "hourglass", "pear", "apple", "inverted_triangle", "athletic"],
        help="Body type (if known, instead of auto-detection)"
    )
    
    # Configuration options
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=["default", "minimalist", "creative", "business", "casual"],
        help="Scoring profile to use (default: default)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        help="Save JSON report to file"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results (garment JSONs, score graphs, reports) for analysis"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("tests/output/pipeline_runs"),
        help="Directory to save results (default: tests/output/pipeline_runs)"
    )
    parser.add_argument(
        "--visualize",
        type=Path,
        help="Generate and save outfit visualization image to specified path (e.g., best_outfit.png)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.demo, args.wardrobe, args.outfit_image, args.images]):
        parser.print_help()
        print("\n⚠️  Please specify one of: --demo, --wardrobe, --outfit-image, or --images")
        sys.exit(1)
    
    # Validate user photo if provided
    style_profile = None
    if args.user_photo:
        if not args.user_photo.exists():
            print(f"❌ User photo not found: {args.user_photo}")
            sys.exit(1)
    
    try:
        result = None
        
        # Initialize results storage if saving is enabled
        storage = None
        if args.save_results:
            storage = ResultsStorage(args.results_dir)
        
        # Extract user profile from photo if provided
        if args.user_photo:
            style_profile = extract_user_profile(
                args.user_photo,
                height_cm=args.height,
                weight_kg=args.weight,
                verbose=True
            )
            
            # Display context-aware recommendations
            if style_profile:
                color_advisor = ColorHarmonyAdvisor()
                proportion_harmonizer = ProportionHarmonizer()
                display_context_recommendations(
                    style_profile,
                    color_advisor,
                    proportion_harmonizer
                )
                
                # Save profile if storage enabled
                if storage:
                    profile_path = storage.run_dir / "user_profile.json"
                    with open(profile_path, "w") as f:
                        json.dump(style_profile.to_dict(), f, indent=2, default=str)
                    print(f"💾 User profile saved to: {profile_path}")
        
        if args.demo:
            # Run demo mode
            demo_result = await run_demo_with_result()
            
            # Apply context scoring if style profile available
            if style_profile and demo_result:
                await apply_context_scoring(demo_result, style_profile, args.body_type)
        else:
            tester = FullPipelineTester(profile=args.profile, results_storage=storage)
            
            # Pass style profile to tester for context-aware scoring
            if style_profile:
                tester.style_profile = style_profile
                body_type = None
                if args.body_type:
                    body_type = args.body_type
                elif style_profile.body_metrics and style_profile.body_metrics.body_shape:
                    body_type = style_profile.body_metrics.body_shape.value
                tester.body_type = body_type
                print(f"\n🎯 Context-aware scoring enabled (body type: {body_type or 'auto'})")
            
            if args.wardrobe:
                if not args.wardrobe.exists():
                    print(f"❌ Wardrobe folder not found: {args.wardrobe}")
                    sys.exit(1)
                result = await tester.test_wardrobe_folder(args.wardrobe)
                
                # Apply context engine if style profile available
                if style_profile and result and "best_outfit" in result:
                    result = await apply_context_scoring(result, style_profile, args.body_type)
            
            elif args.outfit_image:
                if not args.outfit_image.exists():
                    print(f"❌ Image not found: {args.outfit_image}")
                    sys.exit(1)
                result = await tester.test_outfit_image(args.outfit_image)
            
            elif args.images:
                for img in args.images:
                    if not img.exists():
                        print(f"❌ Image not found: {img}")
                        sys.exit(1)
                result = await tester.test_individual_images(args.images)
            
            # Generate outfit visualization if requested
            if args.visualize and result and "best_outfit" in result:
                print(f"\n🖼️  Generating outfit visualization...")
                try:
                    # Extract garment info from result to create visualization
                    best_outfit = result["best_outfit"]
                    garments = best_outfit.get("garments", [])
                    score = best_outfit.get("overall_score", 0)
                    outfit_name = best_outfit.get("name", "Best Outfit")
                    
                    # Use the tester's segmented images if available
                    segmented_images = getattr(tester, 'segmented_images', {})
                    
                    # Create visualizer and generate image
                    visualizer = OutfitVisualizer()
                    
                    # We need to reconstruct Garment objects from the result
                    # or use the ones stored in the tester
                    if hasattr(tester, '_last_best_outfit') and tester._last_best_outfit:
                        # Use actual garments with full data
                        viz = visualizer.create_outfit_catalogue(
                            garments=tester._last_best_outfit.garments,
                            outfit_name=outfit_name,
                            score=score,
                            segmented_images=segmented_images
                        )
                    else:
                        # Fallback: create placeholder visualization from result data
                        from src.core.models import Garment, GarmentAttributes, GarmentCategory, ColorProfile
                        
                        reconstructed_garments = []
                        for g_data in garments:
                            garment = Garment(
                                id=g_data.get("id", "unknown"),
                                attributes=GarmentAttributes(
                                    category=GarmentCategory(g_data.get("category", "top")),
                                    subcategory=g_data.get("subcategory"),
                                    color=ColorProfile(
                                        primary=g_data.get("color", "unknown"),
                                        secondary=g_data.get("color_secondary")
                                    ) if g_data.get("color") else None
                                ),
                                image_path=g_data.get("image")
                            )
                            reconstructed_garments.append(garment)
                        
                        viz = visualizer.create_outfit_catalogue(
                            garments=reconstructed_garments,
                            outfit_name=outfit_name,
                            score=score,
                            segmented_images=segmented_images
                        )
                    
                    # Save the visualization
                    visualizer.save_visualization(viz, args.visualize)
                    print(f"   ✅ Outfit visualization saved to: {args.visualize}")
                    
                except Exception as e:
                    logger.warning(f"Could not generate visualization: {e}")
                    print(f"   ⚠️  Could not generate visualization: {e}")
        
        # Save output if requested
        if args.output and result:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Report saved to: {args.output}")
        
        print("\n✅ Pipeline test complete!")
        
        if storage:
            print(f"\n📁 All results saved to: {storage.run_dir}")
            print("   Contents:")
            print("   - garments/     : Individual garment JSON files")
            print("   - combinations/ : Outfit scores and rankings")
            print("   - graphs/       : Visual score comparisons")
            print("   - final_report.json : Summary report")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Pipeline test failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
