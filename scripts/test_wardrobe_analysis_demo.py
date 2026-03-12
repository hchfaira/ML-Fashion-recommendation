"""
Wardrobe Analysis Demo Script
==============================

Demonstrates the full wardrobe analysis pipeline:
1. Analyze wardrobe (distribution, gaps, coverage, versatility)
2. Improve outfit (diagnosis, suggestions)
3. Simulate addition (what-if analysis)

Results are saved as JSON and visualizations.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.models import (
    Garment,
    GarmentAttributes,
    GarmentCategory,
    ColorProfile,
    FormalityLevel,
    Season,
    PatternInfo,
    MaterialProfile,
    SeasonalityInfo,
)
from src.layer2_style.wardrobe_analyzer import WardrobeAnalyzer
from src.layer2_style.outfit_improver import OutfitImprover
from src.layer1_vision import AttributeExtractor
from src.core import get_logger

logger = get_logger(__name__)



# ============================================================================
# Configuration
# ============================================================================

SAMPLE_WARDROBE_DIR = Path("data/sample_wardrobe")
OUTPUT_DIR = Path("tests/output/wardrobe_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def create_sample_garment(
    image_path: str,
    garment_id: str = None,
    category: str = None,
    color: str = "unknown",
    formality: str = "casual",
) -> Garment:
    """Create a garment from basic info (simplified for demo)."""
    from uuid import uuid4
    
    return Garment(
        id=garment_id or f"g_{uuid4().hex[:8]}",
        image_path=image_path,
        attributes=GarmentAttributes(
            category=GarmentCategory[category.upper()] if category else GarmentCategory.TOP,
            subcategory=Path(image_path).stem,
            color=ColorProfile(primary=color, hex_codes=[]),
            formality_level=FormalityLevel[formality.upper()] if formality else FormalityLevel.CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
            seasonality=SeasonalityInfo(
                seasons=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER]
            ),
            pattern=PatternInfo(type="solid"),
            material=MaterialProfile(primary="fabric"),
        ),
    )


def load_sample_wardrobe() -> List[Garment]:
    """Load wardrobe from sample_wardrobe directory."""
    if not SAMPLE_WARDROBE_DIR.exists():
        logger.warning(f"Sample wardrobe directory not found: {SAMPLE_WARDROBE_DIR}")
        return _create_demo_wardrobe()
    
    logger.info(f"Loading wardrobe from {SAMPLE_WARDROBE_DIR}...")
    wardrobe = []
    
    # Map subdirectories to categories
    category_mapping = {
        "tops": "TOP",
        "bottoms": "BOTTOM",
        "shoes": "SHOES",
        "outerwear": "OUTERWEAR",
        "accessories": "ACCESSORY",
        "dresses": "DRESS",
    }
    
    for subdir, category in category_mapping.items():
        subdir_path = SAMPLE_WARDROBE_DIR / subdir
        if subdir_path.exists():
            logger.info(f"  Loading {subdir}...")
            for img_path in subdir_path.glob("*.png"):
                garment = create_sample_garment(
                    str(img_path),
                    category=category,
                    color=_infer_color_from_filename(img_path.name),
                )
                wardrobe.append(garment)
                logger.debug(f"    Added: {img_path.name}")
    
    if not wardrobe:
        logger.warning("No images found in sample wardrobe, using demo wardrobe")
        return _create_demo_wardrobe()
    
    logger.info(f"Loaded {len(wardrobe)} garments from sample wardrobe")
    return wardrobe


def _infer_color_from_filename(filename: str) -> str:
    """Infer color from filename."""
    filename_lower = filename.lower()
    color_map = {
        "black": "black",
        "white": "white",
        "blue": "blue",
        "red": "red",
        "green": "green",
        "yellow": "yellow",
        "navy": "navy",
        "grey": "grey",
        "gray": "gray",
        "brown": "brown",
        "beige": "beige",
        "pink": "pink",
    }
    for color_name, color_value in color_map.items():
        if color_name in filename_lower:
            return color_value
    return "multi"


def _create_demo_wardrobe() -> List[Garment]:
    """Create a demo wardrobe for testing."""
    from uuid import uuid4
    
    logger.info("Creating demo wardrobe...")
    garments = []
    
    # Tops
    for i, color in enumerate(["white", "navy", "black", "grey"], 1):
        garments.append(Garment(
            id=f"top_{i}",
            image_path=f"demo/top_{color}.png",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                subcategory="t-shirt" if i <= 2 else "dress_shirt",
                color=ColorProfile(primary=color, hex_codes=[]),
                formality_level=FormalityLevel.BUSINESS if i > 2 else FormalityLevel.CASUAL,
                season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
            ),
        ))
    
    # Bottoms
    for i, (color, form) in enumerate(
        [("blue", FormalityLevel.CASUAL), ("grey", FormalityLevel.SMART_CASUAL), ("black", FormalityLevel.BUSINESS)],
        1
    ):
        garments.append(Garment(
            id=f"bottom_{i}",
            image_path=f"demo/bottom_{color}.png",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM,
                subcategory="jeans" if i == 1 else "trousers",
                color=ColorProfile(primary=color, hex_codes=[]),
                formality_level=form,
                season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
            ),
        ))
    
    # Shoes
    for i, (color, form) in enumerate(
        [("white", FormalityLevel.CASUAL), ("brown", FormalityLevel.SMART_CASUAL), ("black", FormalityLevel.FORMAL)],
        1
    ):
        garments.append(Garment(
            id=f"shoes_{i}",
            image_path=f"demo/shoes_{color}.png",
            attributes=GarmentAttributes(
                category=GarmentCategory.SHOES,
                subcategory="sneakers" if i == 1 else "loafers" if i == 2 else "oxfords",
                color=ColorProfile(primary=color, hex_codes=[]),
                formality_level=form,
                season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
            ),
        ))
    
    # Outerwear
    garments.append(Garment(
        id="jacket_1",
        image_path="demo/jacket_black.png",
        attributes=GarmentAttributes(
            category=GarmentCategory.OUTERWEAR,
            subcategory="blazer",
            color=ColorProfile(primary="black", hex_codes=[]),
            formality_level=FormalityLevel.BUSINESS,
            season_suitable=[Season.FALL, Season.WINTER],
        ),
    ))
    
    # Accessories
    garments.append(Garment(
        id="acc_1",
        image_path="demo/watch_silver.png",
        attributes=GarmentAttributes(
            category=GarmentCategory.ACCESSORY,
            subcategory="watch",
            color=ColorProfile(primary="silver", hex_codes=[]),
            formality_level=FormalityLevel.SMART_CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
        ),
    ))
    
    logger.info(f"Created demo wardrobe with {len(garments)} garments")
    return garments


def save_json_result(data: Any, filename: str) -> Path:
    """Save result to JSON file."""
    output_path = OUTPUT_DIR / filename
    
    # Convert to JSON-serializable format
    json_data = _to_json_serializable(data)
    
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Saved JSON result to {output_path}")
    return output_path


def _to_json_serializable(obj: Any) -> Any:
    """Convert Pydantic models to JSON-serializable dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_json_serializable(item) for item in obj]
    elif hasattr(obj, "value"):
        return obj.value
    else:
        return obj


def create_summary_report(
    analysis_result: Dict[str, Any],
    improvement_result: Dict[str, Any],
    simulation_result: Dict[str, Any],
    removal_result: Dict[str, Any] = None,
) -> str:
    """Create a human-readable summary report."""
    report = []
    report.append("=" * 80)
    report.append("WARDROBE ANALYSIS SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # Wardrobe Analysis Summary
    report.append("1. WARDROBE ANALYSIS")
    report.append("-" * 80)
    if isinstance(analysis_result, dict) and "summary" in analysis_result:
        report.append(analysis_result["summary"])
    report.append("")
    
    if isinstance(analysis_result, dict) and "distribution" in analysis_result:
        dist = analysis_result["distribution"]
        if isinstance(dist, dict):
            report.append(f"Total items: {dist.get('total_items', 'N/A')}")
            if "by_category" in dist:
                report.append("Category breakdown:")
                for cat in dist["by_category"][:5]:
                    if isinstance(cat, dict):
                        report.append(
                            f"  • {cat.get('category', 'Unknown')}: "
                            f"{cat.get('count', 0)} items ({cat.get('percentage', 0):.1f}%)"
                        )
    report.append("")
    
    # Gaps
    if isinstance(analysis_result, dict) and "gaps" in analysis_result:
        gaps = analysis_result["gaps"]
        if gaps and isinstance(gaps, list):
            report.append(f"Identified gaps ({len(gaps)}):")
            for gap in gaps[:3]:
                if isinstance(gap, dict):
                    report.append(
                        f"  • [{gap.get('severity', 'unknown').upper()}] "
                        f"{gap.get('description', 'Unknown gap')}"
                    )
    report.append("")
    
    # Purchase Suggestions
    if isinstance(analysis_result, dict) and "purchase_suggestions" in analysis_result:
        suggestions = analysis_result["purchase_suggestions"]
        if suggestions and isinstance(suggestions, list):
            report.append(f"Top purchase suggestions ({len(suggestions)}):")
            for sugg in suggestions[:3]:
                if isinstance(sugg, dict):
                    report.append(
                        f"  {sugg.get('priority', 0)}. "
                        f"{sugg.get('category', 'Unknown')} - "
                        f"{sugg.get('description', 'No description')}"
                    )
    report.append("")
    
    # Outfit Improvement Summary
    report.append("2. OUTFIT IMPROVEMENT")
    report.append("-" * 80)
    if isinstance(improvement_result, dict) and "summary" in improvement_result:
        report.append(improvement_result["summary"])
    report.append("")
    
    if isinstance(improvement_result, dict) and "diagnosis" in improvement_result:
        diag = improvement_result["diagnosis"]
        if isinstance(diag, dict):
            report.append(
                f"Current outfit score: {diag.get('overall_score', 0):.0%} "
                f"(Grade: {diag.get('grade', 'N/A')})"
            )
    report.append("")
    
    # Simulation Summary
    report.append("3. ADDITION SIMULATION")
    report.append("-" * 80)
    if isinstance(simulation_result, dict):
        report.append(
            f"Simulated garment: {simulation_result.get('garment_description', 'Unknown')}"
        )
        report.append(
            f"Versatility score: {simulation_result.get('versatility_score', 0):.0%}"
        )
        report.append(
            f"New outfit combinations: {simulation_result.get('new_outfit_combinations', 0)}"
        )
        report.append(
            f"Gaps resolved: {simulation_result.get('gaps_resolved', 0)}"
        )
        report.append(
            f"Recommendation: {simulation_result.get('recommendation', 'No recommendation')}"
        )
    report.append("")
    
    # Removal Impact Summary
    report.append("4. REMOVAL IMPACT ANALYSIS")
    report.append("-" * 80)
    if removal_result and isinstance(removal_result, dict):
        if "summary" in removal_result:
            report.append(removal_result["summary"])
        else:
            report.append("No removal impact analysis available.")
    report.append("")
    
    report.append("=" * 80)
    return "\n".join(report)


# ============================================================================
# Main Demo Functions
# ============================================================================

def test_wardrobe_analysis():
    """Test 1: Full wardrobe analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: WARDROBE ANALYSIS")
    logger.info("=" * 80)
    
    wardrobe = load_sample_wardrobe()
    analyzer = WardrobeAnalyzer()
    
    logger.info(f"Analyzing wardrobe with {len(wardrobe)} garments...")
    result = analyzer.analyze(wardrobe, top_k_versatile=5)
    
    # Convert to serializable format
    result_dict = _to_json_serializable(result)
    
    # Save JSON
    json_path = save_json_result(result_dict, "01_wardrobe_analysis.json")
    logger.info(f"✓ Analysis complete. Saved to {json_path}")
    
    return result_dict, wardrobe


def test_outfit_improvement(wardrobe: List[Garment]):
    """Test 2: Outfit improvement."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: OUTFIT IMPROVEMENT")
    logger.info("=" * 80)
    
    # Create a sample outfit from the first 3 items
    outfit = wardrobe[:3]
    improver = OutfitImprover()
    
    logger.info(f"Improving outfit with {len(outfit)} garments...")
    result = improver.improve(outfit, wardrobe)
    
    # Convert to serializable format
    result_dict = _to_json_serializable(result)
    
    # Save JSON
    json_path = save_json_result(result_dict, "02_outfit_improvement.json")
    logger.info(f"✓ Improvement analysis complete. Saved to {json_path}")
    
    return result_dict


def test_simulate_addition(wardrobe: List[Garment]):
    """Test 3: Simulate addition."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: SIMULATE ADDITION")
    logger.info("=" * 80)
    
    # Create a new virtual garment to simulate
    from uuid import uuid4
    new_garment = Garment(
        id=f"virtual_{uuid4().hex[:8]}",
        image_path="virtual/new_dress_red.png",
        attributes=GarmentAttributes(
            category=GarmentCategory.DRESS,
            subcategory="cocktail_dress",
            color=ColorProfile(primary="red", hex_codes=[]),
            formality_level=FormalityLevel.SMART_CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL],
            pattern=PatternInfo(type="solid"),
            material=MaterialProfile(primary="silk"),
        ),
    )
    
    analyzer = WardrobeAnalyzer()
    
    logger.info(f"Simulating addition of: {new_garment.attributes.color.primary} dress...")
    result = analyzer.simulate_addition(wardrobe, new_garment)
    
    # Convert to serializable format
    result_dict = _to_json_serializable(result)
    
    # Save JSON
    json_path = save_json_result(result_dict, "03_simulate_addition.json")
    logger.info(f"✓ Simulation complete. Saved to {json_path}")
    
    return result_dict


def test_removal_impact_analysis(wardrobe: List[Garment]):
    """Test 4: Analyze removal impact - identify items safe to remove."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: REMOVAL IMPACT ANALYSIS")
    logger.info("=" * 80)
    
    improver = OutfitImprover()
    removal_impacts = []
    
    # Analyze impact of removing each item
    logger.info(f"Analyzing removal impact for {len(wardrobe)} items...")
    
    for i, garment in enumerate(wardrobe[:5]):  # Analyze first 5 items
        try:
            logger.info(
                f"  [{i+1}/{min(5, len(wardrobe))}] Analyzing: "
                f"{garment.attributes.color.primary} "
                f"{garment.attributes.category.value}..."
            )
            
            impact = improver.analyze_removal_impact(garment.id, wardrobe)
            impact_dict = _to_json_serializable(impact)
            
            removal_impacts.append({
                "garment_id": garment.id,
                "garment_description": (
                    f"{garment.attributes.color.primary} "
                    f"{garment.attributes.category.value}"
                ),
                "removal_impact": impact_dict
            })
        except Exception as e:
            logger.warning(f"  Could not analyze {garment.id}: {e}")
    
    # Compile results
    result_dict = {
        "total_items_analyzed": len(removal_impacts),
        "analysis_timestamp": datetime.now().isoformat(),
        "removal_impacts": removal_impacts,
        "summary": _generate_removal_summary(removal_impacts)
    }
    
    # Save JSON
    json_path = save_json_result(result_dict, "04_removal_impact_analysis.json")
    logger.info(f"✓ Removal impact analysis complete. Saved to {json_path}")
    
    return result_dict


def _generate_removal_summary(removal_impacts: List[Dict[str, Any]]) -> str:
    """Generate summary of removal impacts."""
    if not removal_impacts:
        return "No items analyzed."
    
    safe_to_remove = []
    critical_items = []
    moderate_items = []
    
    for item in removal_impacts:
        impact = item.get("removal_impact", {})
        is_critical = impact.get("is_critical", False)
        has_replacement = impact.get("has_replacement", False)
        
        if is_critical:
            critical_items.append(item["garment_description"])
        elif has_replacement:
            moderate_items.append(item["garment_description"])
        else:
            safe_to_remove.append(item["garment_description"])
    
    summary_lines = []
    
    if critical_items:
        summary_lines.append(
            f"🔴 CRITICAL ITEMS ({len(critical_items)}): "
            f"NOT recommended to remove - they fill essential gaps"
        )
        for item in critical_items:
            summary_lines.append(f"   • {item}")
    
    if moderate_items:
        summary_lines.append(
            f"🟡 REPLACEABLE ITEMS ({len(moderate_items)}): "
            f"Can be removed if replacement exists"
        )
        for item in moderate_items:
            summary_lines.append(f"   • {item}")
    
    if safe_to_remove:
        summary_lines.append(
            f"🟢 SAFE TO REMOVE ({len(safe_to_remove)}): "
            f"Can be removed without impact"
        )
        for item in safe_to_remove:
            summary_lines.append(f"   • {item}")
    
    return "\n".join(summary_lines)


def create_summary(analysis_result, improvement_result, simulation_result, removal_result=None):
    """Create and save summary report."""
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("=" * 80)
    
    report = create_summary_report(
        analysis_result, 
        improvement_result, 
        simulation_result,
        removal_result
    )
    
    # Save report
    report_path = OUTPUT_DIR / "00_SUMMARY_REPORT.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"✓ Summary report saved to {report_path}")
    
    # Print report
    print(report)
    
    return report_path


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tests."""
    logger.info("Starting Wardrobe Analysis Demo...")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    try:
        # Test 1: Wardrobe Analysis
        analysis_result, wardrobe = test_wardrobe_analysis()
        
        # Test 2: Outfit Improvement
        improvement_result = test_outfit_improvement(wardrobe)
        
        # Test 3: Simulate Addition
        simulation_result = test_simulate_addition(wardrobe)
        
        # Test 4: Removal Impact Analysis
        removal_result = test_removal_impact_analysis(wardrobe)
        
        # Create Summary
        summary_path = create_summary(
            analysis_result, 
            improvement_result, 
            simulation_result,
            removal_result
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        logger.info(f"  • 00_SUMMARY_REPORT.txt")
        logger.info(f"  • 01_wardrobe_analysis.json")
        logger.info(f"  • 02_outfit_improvement.json")
        logger.info(f"  • 03_simulate_addition.json")
        logger.info(f"  • 04_removal_impact_analysis.json")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
