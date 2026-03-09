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
"""

import argparse
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
    
    return parser.parse_args()


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
    
    print()
    print("✅ Done!")


if __name__ == "__main__":
    main()
