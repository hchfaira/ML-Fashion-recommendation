"""
Pipeline Tester Module
======================

Main testing class for running the full outfit recommendation pipeline.

Pipeline Architecture:
- Layer 0: Garment Segmentation (SAM)
- Layer 1: Attribute Extraction (Vision AI)
- Layer 2: Style Scoring
- Layer 3: Context Engine
- Layer 4: LLM Explanations
- Layer 5: Outfit Visualization
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL import Image

from src.core.models import Garment, GarmentAttributes, GarmentCategory
from src.layer1_vision import AttributeExtractor, VisionService, EmbeddingGenerator
from src.layer2_style import OutfitBuilder, OutfitScorecard, get_scoring_config_service
from src.core import get_logger

# Import Layer 0 (segmentation) - optional, may require SAM
try:
    from src.layer0_segmentation import get_segmenter, SegmentedGarment
    SEGMENTATION_AVAILABLE = True
except ImportError:
    SEGMENTATION_AVAILABLE = False

# Import Layer 5 (visualization)
try:
    from src.layer5_visualization import OutfitVisualizer, create_outfit_image
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from .results_storage import ResultsStorage
from . import graph_generator

logger = get_logger(__name__)


class FullPipelineTester:
    """
    End-to-end pipeline tester for outfit recommendation.
    
    Full pipeline flow:
    1. Layer 0: Segment garments from images (extract clothing)
    2. Layer 1: Extract attributes (color, pattern, material, etc.)
    3. Layer 2: Score outfit combinations
    4. Layer 5: Generate visualization of best outfit
    
    Supports multiple input formats:
    - Wardrobe folder (organized or flat)
    - Single outfit image
    - Individual garment images
    """
    
    def __init__(
        self, 
        profile: str = "default", 
        results_storage: Optional[ResultsStorage] = None,
        use_segmentation: bool = True,
        use_visualization: bool = True
    ):
        """
        Initialize the tester.
        
        Args:
            profile: Scoring profile to use (default, minimalist, creative, business, casual)
            results_storage: Optional storage for saving results
            use_segmentation: Whether to use Layer 0 (SAM segmentation)
            use_visualization: Whether to use Layer 5 (outfit visualization)
        """
        self.profile = profile
        self.builder = OutfitBuilder()
        self.extractor = AttributeExtractor()
        self.embedding_gen = EmbeddingGenerator()
        self.storage = results_storage
        
        # Layer 0: Segmentation
        self.segmenter = None
        self.use_segmentation = use_segmentation and SEGMENTATION_AVAILABLE
        if self.use_segmentation:
            try:
                self.segmenter = get_segmenter(use_sam=False)  # Start with simple segmenter
                print("🔍 Layer 0 (Segmentation): Enabled")
            except Exception as e:
                logger.warning(f"Could not initialize segmenter: {e}")
                self.use_segmentation = False
        
        # Layer 5: Visualization
        self.visualizer = None
        self.use_visualization = use_visualization and VISUALIZATION_AVAILABLE
        if self.use_visualization:
            try:
                self.visualizer = OutfitVisualizer()
                print("🖼️  Layer 5 (Visualization): Enabled")
            except Exception as e:
                logger.warning(f"Could not initialize visualizer: {e}")
                self.use_visualization = False
        
        # Segmented images cache (garment.id -> PIL Image)
        self.segmented_images: Dict[str, Image.Image] = {}
        
        # Show available profiles
        config_service = get_scoring_config_service()
        print(f"\n📊 Available scoring profiles: {[p['key'] for p in config_service.list_profiles()]}")
        print(f"   Using profile: {profile}\n")
    
    async def test_wardrobe_folder(self, wardrobe_path: Path) -> dict:
        """
        Test with a folder containing individual garment images.
        
        Supports two structures:
        1. Flat folder with images (will extract, categorize, and find best combination)
        2. Organized by category (will find best combination)
        """
        print(f"🗂️  Loading wardrobe from: {wardrobe_path}")
        print("=" * 60)
        
        # Load wardrobe (handles both flat and organized structures)
        wardrobe = await self._load_wardrobe(wardrobe_path)
        
        # Display loaded garments
        self._display_loaded_garments(wardrobe)
        
        # Generate outfit combinations
        print(f"\n🎨 Generating outfit combinations...")
        combinations = self.builder.generate_outfit_combinations(wardrobe)
        print(f"   Generated {len(combinations)} possible combinations")
        
        if not combinations:
            return self._handle_no_combinations(wardrobe)
        
        # Get all garments list
        all_garments = self._flatten_wardrobe(wardrobe)
        
        # Save garments if storage enabled
        if self.storage:
            print("\n💾 Saving extracted garments...")
            self.storage.save_all_garments(all_garments)
            graph_generator.generate_color_distribution_graph(
                all_garments, 
                self.storage.run_dir / "graphs"
            )
        
        # Evaluate combinations
        print(f"\n📊 Scoring all combinations with profile '{self.profile}'...")
        candidates = self.builder.evaluate_all_combinations(
            combinations,
            profile=self.profile
        )
        
        # Select best outfit
        best = self.builder.select_best_outfit(candidates)
        
        # Save combinations and graphs
        if self.storage and candidates:
            print("\n💾 Saving combination scores and graphs...")
            self.storage.save_combinations(candidates, self.profile)
            graph_generator.generate_score_graph(
                candidates, 
                self.profile, 
                self.storage.run_dir / "graphs"
            )
            graph_generator.generate_criteria_breakdown_graph(
                candidates, 
                self.profile, 
                self.storage.run_dir / "graphs"
            )
        
        # Store best outfit for external access (e.g., visualization from CLI)
        self._last_best_outfit = best
        self._last_candidates = candidates
        
        report = self._generate_report(candidates, best)
        
        # Generate outfit visualization (Layer 5)
        if self.use_visualization and self.visualizer and best:
            print("\n🖼️  Generating outfit visualization (Layer 5)...")
            
            # Show mapping status for best outfit garments
            for g in best.garments:
                has_image = g.id in self.segmented_images
                status = "✓ segmented" if has_image else "○ placeholder"
                print(f"   {g.attributes.category.value:12} → {status}")
            
            try:
                viz = self.visualizer.create_outfit_catalogue(
                    garments=best.garments,
                    outfit_name=f"Best Outfit: {best.name}",
                    score=best.overall_score,
                    segmented_images=self.segmented_images
                )
                
                # Save visualization
                if self.storage:
                    viz_path = self.storage.run_dir / "best_outfit_catalogue.png"
                    self.visualizer.save_visualization(viz, viz_path)
                    report["visualization_path"] = str(viz_path)
                    print(f"   ✅ Saved outfit visualization to: {viz_path.name}")
            except Exception as e:
                logger.warning(f"Could not generate visualization: {e}")
        
        # Save final report
        if self.storage:
            self.storage.save_final_report(report, best, candidates)
        
        return report
    
    async def _load_wardrobe_with_segmentation(self, wardrobe_path: Path) -> Dict[str, List[Garment]]:
        """
        Load wardrobe with Layer 0 segmentation preprocessing.
        
        This applies SAM-based segmentation to each image before
        attribute extraction.
        
        Mapping Strategy:
        1. Layer 0 segments each image → stores by normalized path
        2. Layer 1 extracts attributes → creates Garment with image_path
        3. We match by normalizing both paths (resolve to absolute)
        4. Final mapping: garment.id → segmented PIL Image
        """
        print("\n🔍 Layer 0: Segmenting garments from images...")
        
        # Find all image files
        extensions = (".jpg", ".jpeg", ".png", ".webp")
        image_files = [
            f for f in wardrobe_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        # Dictionary to map normalized paths to segmented images
        # Key: normalized absolute path, Value: (SegmentedGarment, original_path)
        path_to_segmented: Dict[str, Any] = {}
        
        for img_path in sorted(image_files):
            try:
                # Segment the garment
                seg_result = self.segmenter.segment_garment(img_path)
                
                # Store by NORMALIZED path (absolute, resolved)
                normalized_path = str(img_path.resolve())
                path_to_segmented[normalized_path] = {
                    "segmented": seg_result,
                    "original_path": img_path,
                    "filename": img_path.name,
                }
                
                # Save segmented image for inspection
                if self.storage:
                    seg_path = self.storage.run_dir / "segmented" / f"{img_path.stem}_segmented.png"
                    seg_path.parent.mkdir(parents=True, exist_ok=True)
                    seg_result.image.save(seg_path, "PNG")
                
                print(f"   ✓ Segmented: {img_path.name}")
            except Exception as e:
                logger.warning(f"   ✗ Failed to segment {img_path.name}: {e}")
        
        # Now load and extract attributes (Layer 1)
        print("\n🔬 Layer 1: Extracting garment attributes...")
        garments = await self.builder.load_outfit_from_folder(wardrobe_path)
        
        # Map segmented images to garment IDs using multiple strategies
        print("\n🔗 Mapping Layer 0 → Layer 1...")
        matched = 0
        for garment in garments:
            seg_image = self._match_garment_to_segmented(garment, path_to_segmented)
            if seg_image:
                self.segmented_images[garment.id] = seg_image
                matched += 1
                logger.debug(f"   Matched: {garment.id} ← {garment.image_path}")
        
        print(f"   ✅ Matched {matched}/{len(garments)} garments to segmented images")
        
        return self._organize_garments_by_category(garments)
    
    def _match_garment_to_segmented(
        self, 
        garment: Garment, 
        path_to_segmented: Dict[str, Any]
    ) -> Optional[Image.Image]:
        """
        Match a garment to its segmented image using multiple strategies.
        
        Strategies (in order of priority):
        1. Exact path match (normalized absolute path)
        2. Filename match (basename)
        3. Stem match (filename without extension)
        
        Returns:
            PIL Image if matched, None otherwise
        """
        if not garment.image_path:
            return None
        
        garment_path = Path(garment.image_path)
        
        # Strategy 1: Exact normalized path match
        normalized = str(garment_path.resolve())
        if normalized in path_to_segmented:
            return path_to_segmented[normalized]["segmented"].image
        
        # Strategy 2: Filename match
        garment_filename = garment_path.name
        for data in path_to_segmented.values():
            if data["filename"] == garment_filename:
                return data["segmented"].image
        
        # Strategy 3: Stem match (handles extension differences)
        garment_stem = garment_path.stem
        for data in path_to_segmented.values():
            if data["original_path"].stem == garment_stem:
                return data["segmented"].image
        
        logger.warning(f"   No segmentation match for: {garment_path.name}")
        return None

    async def _load_wardrobe(self, wardrobe_path: Path) -> Dict[str, List[Garment]]:
        """Load wardrobe from folder, handling both flat and organized structures."""
        subdirs = [d for d in wardrobe_path.iterdir() if d.is_dir()]
        
        if subdirs:
            # Organized wardrobe
            print("\n📂 Detected organized wardrobe structure")
            return await self.builder.load_wardrobe(wardrobe_path)
        else:
            # Flat folder - check if segmentation is available
            if self.use_segmentation and self.segmenter:
                return await self._load_wardrobe_with_segmentation(wardrobe_path)
            else:
                print("\n📂 Detected flat folder - extracting and categorizing each garment")
                garments = await self.builder.load_outfit_from_folder(wardrobe_path)
                
                # Still cache images for visualization
                for garment in garments:
                    if garment.image_path and Path(garment.image_path).exists():
                        try:
                            img = Image.open(garment.image_path)
                            if img.mode != "RGBA":
                                img = img.convert("RGBA")
                            self.segmented_images[garment.id] = img
                        except Exception:
                            pass
                
                return self._organize_garments_by_category(garments)
    
    def _organize_garments_by_category(self, garments: List[Garment]) -> Dict[str, List[Garment]]:
        """Organize flat list of garments into category dict."""
        wardrobe = {
            "tops": [],
            "bottoms": [],
            "shoes": [],
            "outerwear": [],
            "accessories": [],
            "dresses": []
        }
        
        category_map = {
            "top": "tops",
            "bottom": "bottoms",
            "shoes": "shoes",
            "outerwear": "outerwear",
            "accessory": "accessories",
            "dress": "dresses",
            "bag": "accessories"
        }
        
        for g in garments:
            cat_key = category_map.get(g.attributes.category.value, "tops")
            wardrobe[cat_key].append(g)
        
        return wardrobe
    
    def _flatten_wardrobe(self, wardrobe: Dict[str, List[Garment]]) -> List[Garment]:
        """Flatten wardrobe dict into a list of all garments."""
        all_garments = []
        for items in wardrobe.values():
            all_garments.extend(items)
        return all_garments
    
    def _display_loaded_garments(self, wardrobe: Dict[str, List[Garment]]) -> None:
        """Display a summary of loaded garments."""
        total_items = sum(len(items) for items in wardrobe.values())
        print(f"\n✅ Loaded {total_items} garments:")
        print("-" * 50)
        
        for category, items in wardrobe.items():
            if items:
                print(f"\n   📁 {category.upper()} ({len(items)} items):")
                for item in items:
                    desc = self._get_garment_description(item, include_path=True)
                    print(f"      • {desc}")
    
    def _handle_no_combinations(self, wardrobe: Dict[str, List[Garment]]) -> dict:
        """Handle case when no valid combinations can be generated."""
        print("\n⚠️  No valid combinations found (need at least tops + bottoms)")
        print("   Scoring all garments as single outfit instead...")
        
        all_garments = self._flatten_wardrobe(wardrobe)
        scorecard = OutfitScorecard(garments=all_garments, profile=self.profile)
        result = scorecard.calculate_all_scores()
        return self._generate_single_outfit_report(result)
    
    async def test_outfit_image(self, image_path: Path) -> dict:
        """
        Test with a single full outfit image.
        The vision system will segment and extract each garment.
        """
        print(f"🖼️  Analyzing outfit image: {image_path}")
        print("=" * 60)
        
        # Extract all garments from the image
        print("\n🔍 Segmenting outfit image...")
        garments = await self.builder.extract_full_outfit_from_image(image_path)
        
        print(f"\n✅ Detected {len(garments)} garments:")
        for garment in garments:
            print(f"   • {garment.attributes.category.value}: "
                  f"{garment.attributes.color.primary} "
                  f"{garment.attributes.subcategory or ''}")
        
        # Score the outfit
        print(f"\n📊 Scoring outfit with profile '{self.profile}'...")
        scorecard = OutfitScorecard(garments=garments, profile=self.profile)
        result = scorecard.calculate_all_scores()
        
        return self._generate_single_outfit_report(result)
    
    async def test_individual_images(self, image_paths: List[Path]) -> dict:
        """Test with a list of individual garment images (one outfit)."""
        print(f"🖼️  Analyzing {len(image_paths)} garment images...")
        print("=" * 60)
        
        # Extract attributes from each image
        garments = []
        for img_path in image_paths:
            print(f"\n🔍 Extracting: {img_path.name}...")
            garment = await self.builder.extract_garment_from_image(img_path)
            garments.append(garment)
            print(f"   ✅ {garment.attributes.category.value}: "
                  f"{garment.attributes.color.primary} "
                  f"{garment.attributes.subcategory or ''}")
        
        # Score the outfit
        print(f"\n📊 Scoring outfit with profile '{self.profile}'...")
        scorecard = OutfitScorecard(garments=garments, profile=self.profile)
        result = scorecard.calculate_all_scores()
        
        return self._generate_single_outfit_report(result)
    
    def _get_garment_description(self, garment: Garment, include_path: bool = False) -> str:
        """Generate a clear description of a garment."""
        attrs = garment.attributes
        
        # Color
        color = "unknown"
        if attrs.color:
            color = attrs.color.primary or "unknown"
            if attrs.color.secondary:
                color = f"{color}/{attrs.color.secondary}"
        
        # Category and subcategory
        subcategory = attrs.subcategory or attrs.category.value
        
        # Pattern
        pattern = ""
        if attrs.pattern:
            pattern_type = attrs.pattern.type if hasattr(attrs.pattern, 'type') else str(attrs.pattern)
            if pattern_type and pattern_type != "solid":
                pattern = f" ({pattern_type})"
        
        # Build description
        desc = f"{color} {subcategory}{pattern}".strip()
        
        # Add image path if requested
        if include_path and garment.image_path:
            img_name = Path(garment.image_path).name
            desc = f"{desc} [{img_name}]"
        
        return desc

    def _generate_report(self, candidates: list, best) -> dict:
        """Generate report for best outfit selection."""
        print("\n" + "=" * 60)
        print("🏆 BEST OUTFIT FOUND!")
        print("=" * 60)
        
        print(f"\n📋 Outfit: {best.name}")
        print(f"🎯 Overall Score: {best.overall_score:.1%}")
        
        print("\n👔 Garments in Best Outfit:")
        for i, g in enumerate(best.garments, 1):
            desc = self._get_garment_description(g, include_path=True)
            print(f"   {i}. {g.attributes.category.value.upper():12} → {desc}")
        
        # Show score breakdown
        print("\n📊 Score Breakdown:")
        scores = best.scorecard.get_filtered_scores()
        for criterion, score in scores.items():
            display_name = best.scorecard._get_display_name(criterion)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"   {display_name:25} [{bar}] {score:.1%}")
        
        # Top alternatives
        print(f"\n📋 Top {min(5, len(candidates))} Outfit Combinations:")
        print("-" * 60)
        
        sorted_candidates = sorted(candidates, key=lambda c: c.overall_score, reverse=True)
        for i, candidate in enumerate(sorted_candidates[:5], 1):
            is_best = "🏆" if candidate == best else "  "
            print(f"\n{is_best} #{i} - Score: {candidate.overall_score:.1%}")
            
            for g in candidate.garments:
                desc = self._get_garment_description(g, include_path=True)
                print(f"      • {g.attributes.category.value:12} → {desc}")
        
        # Build JSON report
        return self._build_json_report(candidates, best, sorted_candidates)
    
    def _build_json_report(self, candidates: list, best, sorted_candidates: list) -> dict:
        """Build JSON report structure."""
        return {
            "best_outfit": {
                "name": best.name,
                "overall_score": best.overall_score,
                "garments": [
                    {
                        "id": g.id,
                        "image": Path(g.image_path).name if g.image_path else None,
                        "category": g.attributes.category.value,
                        "subcategory": g.attributes.subcategory,
                        "color": g.attributes.color.primary if g.attributes.color else None,
                        "color_secondary": g.attributes.color.secondary if g.attributes.color else None,
                        "pattern": g.attributes.pattern.type if g.attributes.pattern and hasattr(g.attributes.pattern, 'type') else None
                    }
                    for g in best.garments
                ],
                "scores": best.scorecard.to_json(include_all_scores=False)
            },
            "all_combinations": [
                {
                    "rank": i + 1,
                    "score": c.overall_score,
                    "garments": [
                        {
                            "category": g.attributes.category.value,
                            "description": self._get_garment_description(g),
                            "image": Path(g.image_path).name if g.image_path else None
                        }
                        for g in c.garments
                    ]
                }
                for i, c in enumerate(sorted_candidates)
            ],
            "total_combinations": len(candidates),
            "profile_used": self.profile
        }
    
    def _generate_single_outfit_report(self, scorecard: OutfitScorecard) -> dict:
        """Generate report for a single outfit."""
        print("\n" + "=" * 60)
        print("📊 OUTFIT ANALYSIS COMPLETE")
        print("=" * 60)
        
        overall = scorecard.scores.get("overall", 0)
        print(f"\n🎯 Overall Score: {overall:.1%}")
        print(f"📝 Grade: {scorecard.get_grade()}")
        
        # Show score breakdown
        print("\n📊 Score Breakdown:")
        scores = scorecard.get_filtered_scores()
        for criterion, score in scores.items():
            display_name = scorecard._get_display_name(criterion)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"   {display_name:25} [{bar}] {score:.1%}")
        
        # Show details
        self._print_score_details(scorecard)
        
        return scorecard.to_json(include_all_scores=False)
    
    def _print_score_details(self, scorecard: OutfitScorecard) -> None:
        """Print detailed scoring information."""
        print("\n💡 Details:")
        details = scorecard.get_filtered_details()
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
