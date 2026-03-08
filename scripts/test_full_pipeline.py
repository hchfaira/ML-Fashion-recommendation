#!/usr/bin/env python3
"""
Full Pipeline Test Script
=========================

This script demonstrates the complete workflow:
1. Load garment images from a folder (your wardrobe)
2. Extract attributes using Vision API (Layer 1)
3. Generate outfit combinations
4. Score each combination (Layer 2)
5. Select the BEST outfit
6. Generate a detailed report

Usage:
    # Test with a folder of individual garment images
    python scripts/test_full_pipeline.py --wardrobe path/to/wardrobe_folder
    
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

logger = get_logger(__name__)


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
    
    try:
        result = None
        
        # Initialize results storage if saving is enabled
        storage = None
        if args.save_results:
            storage = ResultsStorage(args.results_dir)
        
        if args.demo:
            await run_demo()
        else:
            tester = FullPipelineTester(profile=args.profile, results_storage=storage)
            
            if args.wardrobe:
                if not args.wardrobe.exists():
                    print(f"❌ Wardrobe folder not found: {args.wardrobe}")
                    sys.exit(1)
                result = await tester.test_wardrobe_folder(args.wardrobe)
            
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
