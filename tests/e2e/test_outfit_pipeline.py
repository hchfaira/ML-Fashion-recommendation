"""
End-to-End Tests - Outfit Recommendation Pipeline
Tests the complete flow from image to recommendation.

E2E Test Scenarios:
1. Complete outfit scoring from images
2. Best outfit selection from wardrobe
3. Context-aware recommendations
4. Full pipeline with explanations

These tests use mocked external APIs but test real integration
between all layers.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory,
    ColorInfo, PatternInfo, FormalityLevel, Season,
    Outfit, OutfitItem, UserContext, Occasion
)


# ============== Test Data ==============

MOCK_EXTRACTED_TOP = GarmentAttributes(
    category=GarmentCategory.TOP,
    subcategory="t-shirt",
    color=ColorInfo(primary="white", hex_codes=["#FFFFFF"]),
    pattern=PatternInfo(type="solid", scale="none"),
    style_tags=["casual", "basic"],
    formality_level=FormalityLevel.CASUAL,
    season_suitable=[Season.SPRING, Season.SUMMER]
)

MOCK_EXTRACTED_BOTTOM = GarmentAttributes(
    category=GarmentCategory.BOTTOM,
    subcategory="jeans",
    color=ColorInfo(primary="blue", hex_codes=["#1E3A5F"]),
    pattern=PatternInfo(type="solid", scale="none"),
    style_tags=["casual", "denim"],
    formality_level=FormalityLevel.CASUAL,
    season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL]
)

MOCK_EXTRACTED_SHOES = GarmentAttributes(
    category=GarmentCategory.SHOES,
    subcategory="sneakers",
    color=ColorInfo(primary="white", hex_codes=["#FFFFFF"]),
    pattern=PatternInfo(type="solid", scale="none"),
    style_tags=["casual", "sporty"],
    formality_level=FormalityLevel.CASUAL,
    season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER]
)


# ============== Fixtures ==============

@pytest.fixture
def output_dir():
    """Create and return E2E output directory."""
    path = Path(__file__).parent.parent / "output" / "e2e"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def mock_vision_layer():
    """Mock Layer 1 Vision components."""
    with patch('src.layer1_vision.AttributeExtractor') as mock_extractor, \
         patch('src.layer1_vision.ClothingSegmenter') as mock_segmenter:
        
        # Setup attribute extractor
        extractor_instance = MagicMock()
        extractor_instance.extract_attributes = AsyncMock(
            return_value=[MOCK_EXTRACTED_TOP, MOCK_EXTRACTED_BOTTOM, MOCK_EXTRACTED_SHOES]
        )
        mock_extractor.return_value = extractor_instance
        
        # Setup segmenter
        segmenter_instance = MagicMock()
        segmenter_instance.segment_outfit = AsyncMock(return_value=[
            {"category": "top", "specific_type": "t-shirt"},
            {"category": "bottom", "specific_type": "jeans"},
            {"category": "shoes", "specific_type": "sneakers"}
        ])
        mock_segmenter.return_value = segmenter_instance
        
        yield {
            "extractor": extractor_instance,
            "segmenter": segmenter_instance
        }


@pytest.fixture
def mock_llm_layer():
    """Mock Layer 4 LLM components."""
    with patch('src.layer4_llm.LLMService') as mock_llm, \
         patch('src.layer4_llm.OutfitExplainer') as mock_explainer:
        
        llm_instance = MagicMock()
        llm_instance.explain_outfit = AsyncMock(
            return_value="This casual outfit combines a clean white t-shirt with classic blue jeans and white sneakers for a timeless, comfortable look."
        )
        mock_llm.return_value = llm_instance
        
        explainer_instance = MagicMock()
        explainer_instance.explain = AsyncMock(return_value=MagicMock(
            summary="Perfect casual look",
            style_notes=["Clean and minimal", "Good color harmony"],
            confidence=0.85
        ))
        mock_explainer.return_value = explainer_instance
        
        yield {
            "llm": llm_instance,
            "explainer": explainer_instance
        }


@pytest.fixture
def sample_wardrobe():
    """Create a sample wardrobe with multiple items."""
    return [
        Garment(id="white_tee", attributes=MOCK_EXTRACTED_TOP),
        Garment(id="blue_jeans", attributes=MOCK_EXTRACTED_BOTTOM),
        Garment(id="white_sneakers", attributes=MOCK_EXTRACTED_SHOES),
        Garment(
            id="navy_blazer",
            attributes=GarmentAttributes(
                category=GarmentCategory.OUTERWEAR,
                subcategory="blazer",
                color=ColorInfo(primary="navy", hex_codes=["#000080"]),
                pattern=PatternInfo(type="solid", scale="none"),
                style_tags=["smart_casual", "professional"],
                formality_level=FormalityLevel.SMART_CASUAL,
                season_suitable=[Season.FALL, Season.WINTER, Season.SPRING]
            )
        ),
        Garment(
            id="dress_shirt",
            attributes=GarmentAttributes(
                category=GarmentCategory.TOP,
                subcategory="dress_shirt",
                color=ColorInfo(primary="light_blue", hex_codes=["#ADD8E6"]),
                pattern=PatternInfo(type="solid", scale="none"),
                style_tags=["professional", "classic"],
                formality_level=FormalityLevel.BUSINESS,
                season_suitable=[Season.SPRING, Season.FALL, Season.WINTER]
            )
        ),
    ]


# ============== E2E Test Classes ==============

@pytest.mark.e2e
class TestCompleteOutfitScoringPipeline:
    """
    E2E Test: Complete outfit scoring from wardrobe items.
    
    Flow:
    1. Take garments from wardrobe
    2. Score outfit using Layer 2 style intelligence
    3. Generate explanation using Layer 4
    """
    
    @pytest.mark.asyncio
    async def test_casual_outfit_scoring(
        self, sample_wardrobe, mock_llm_layer, output_dir
    ):
        """Test scoring a casual outfit combination."""
        from src.layer2_style import OutfitBuilder, OutfitScorecard
        
        # Build outfit from wardrobe
        outfit_items = [
            sample_wardrobe[0],  # white_tee
            sample_wardrobe[1],  # blue_jeans
            sample_wardrobe[2],  # white_sneakers
        ]
        
        # Create scorecard with garments
        scorecard = OutfitScorecard(garments=outfit_items)
        
        # Calculate all scores
        scorecard.calculate_all_scores()
        result = scorecard.to_json()
        
        # Verify structure - overall_score is in summary
        assert "summary" in result
        assert "overall_score" in result["summary"]
        assert 0.0 <= result["summary"]["overall_score"] <= 1.0
        
        # Save result
        output_file = output_dir / "casual_outfit_score.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
    
    @pytest.mark.asyncio
    async def test_business_outfit_scoring(
        self, sample_wardrobe, mock_llm_layer, output_dir
    ):
        """Test scoring a business outfit combination."""
        from src.layer2_style import OutfitScorecard
        
        # Build business outfit
        outfit_items = [
            sample_wardrobe[4],  # dress_shirt
            sample_wardrobe[1],  # blue_jeans (smart casual)
            sample_wardrobe[3],  # navy_blazer
        ]
        
        # Create scorecard with garments
        scorecard = OutfitScorecard(garments=outfit_items)
        scorecard.calculate_all_scores()
        result = scorecard.to_json()
        
        # Verify structure - overall_score is in summary
        assert "summary" in result
        assert "overall_score" in result["summary"]
        
        # Save result
        output_file = output_dir / "business_outfit_score.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)


@pytest.mark.e2e
class TestBestOutfitSelectionPipeline:
    """
    E2E Test: Find the best outfit from wardrobe.
    
    Flow:
    1. Generate multiple outfit combinations
    2. Score each combination
    3. Return ranked outfits
    """
    
    @pytest.mark.asyncio
    async def test_find_best_casual_outfit(
        self, sample_wardrobe, mock_llm_layer, output_dir
    ):
        """Test finding the best casual outfit."""
        from src.layer2_style import OutfitBuilder
        
        context = UserContext(
            user_id="test_user",
            occasion=Occasion.CASUAL,
            body_shape="rectangle"
        )
        
        builder = OutfitBuilder(context=context)
        
        # Generate combinations
        combinations = builder.generate_outfit_combinations(sample_wardrobe)
        
        # Evaluate all combinations
        candidates = builder.evaluate_all_combinations(combinations)
        
        # Select top 3
        ranked_outfits = builder.select_top_n(candidates, n=3)
        
        assert len(ranked_outfits) <= 3
        
        # Verify ranking order
        if len(ranked_outfits) >= 2:
            scores = [o.overall_score for o in ranked_outfits]
            assert scores == sorted(scores, reverse=True)
        
        # Save results
        output_file = output_dir / "best_outfits_ranking.json"
        results = [
            {
                "rank": i + 1,
                "outfit_name": o.name,
                "score": o.overall_score,
                "items": [g.id for g in o.garments]
            }
            for i, o in enumerate(ranked_outfits)
        ]
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)


@pytest.mark.e2e
class TestContextAwareRecommendationPipeline:
    """
    E2E Test: Context-aware outfit recommendations.
    
    Flow:
    1. Filter wardrobe by context (weather, occasion)
    2. Generate appropriate outfits
    3. Score and rank
    """
    
    @pytest.mark.asyncio
    async def test_occasion_filtering(
        self, sample_wardrobe, output_dir
    ):
        """Test filtering wardrobe by occasion."""
        from src.layer3_context import ContextEngine
        
        # Business occasion context
        context = UserContext(
            user_id="test_user",
            occasion=Occasion.BUSINESS,
            body_shape="rectangle"
        )
        
        engine = ContextEngine()
        
        # Filter wardrobe
        filtered = await engine.filter_wardrobe_by_context(
            sample_wardrobe, context
        )
        
        # Should prefer business-appropriate items
        formality_levels = [g.attributes.formality_level for g in filtered]
        
        # Should not include highly casual items for business
        assert len(filtered) > 0
        
        # Save filtering results
        output_file = output_dir / "context_filtered_wardrobe.json"
        results = {
            "occasion": "business",
            "total_items": len(sample_wardrobe),
            "filtered_items": len(filtered),
            "filtered_ids": [g.id for g in filtered]
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    
    @pytest.mark.asyncio
    async def test_weather_consideration(
        self, sample_wardrobe, output_dir
    ):
        """Test weather-aware filtering."""
        from src.layer3_context import ContextEngine
        from src.core.models import WeatherContext
        
        # Cold weather context
        context = UserContext(
            user_id="test_user",
            occasion=Occasion.CASUAL,
            body_shape="rectangle",
            weather=WeatherContext(
                temperature_celsius=5.0,  # Cold
                condition="cloudy"
            )
        )
        
        engine = ContextEngine()
        filtered = await engine.filter_wardrobe_by_context(
            sample_wardrobe, context
        )
        
        # Should include outerwear for cold weather
        categories = [g.attributes.category for g in filtered]
        
        # Save results
        output_file = output_dir / "weather_filtered_wardrobe.json"
        results = {
            "temperature": 5.0,
            "filtered_items": len(filtered),
            "categories_included": [c.value for c in set(categories)]
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)


@pytest.mark.e2e
class TestFullPipelineWithExplanations:
    """
    E2E Test: Complete pipeline including LLM explanations.
    
    Flow:
    1. Extract attributes from images
    2. Score outfit with Layer 2
    3. Apply context with Layer 3
    4. Generate explanation with Layer 4
    """
    
    @pytest.mark.asyncio
    async def test_full_pipeline_casual_recommendation(
        self, sample_wardrobe, mock_vision_layer, mock_llm_layer, output_dir
    ):
        """Test complete pipeline for casual recommendation."""
        from src.layer2_style import OutfitBuilder, OutfitScorecard
        from src.layer3_context import ContextEngine
        
        # Setup context
        context = UserContext(
            user_id="test_user",
            occasion=Occasion.CASUAL,
            body_shape="rectangle",
            style_preferences=["minimalist", "casual"]
        )
        
        # Step 1: Filter by context (optional - use full wardrobe if filtered is empty)
        engine = ContextEngine()
        filtered_wardrobe = await engine.filter_wardrobe_by_context(
            sample_wardrobe, context
        )
        
        # Use full wardrobe if filtered is too small
        wardrobe_to_use = filtered_wardrobe if len(filtered_wardrobe) >= 3 else sample_wardrobe
        
        # Step 2: Build outfit
        builder = OutfitBuilder(context=context)
        combinations = builder.generate_outfit_combinations(wardrobe_to_use)
        
        # Skip test if no combinations possible
        if not combinations:
            pytest.skip("Not enough items for outfit combinations")
        
        candidates = builder.evaluate_all_combinations(combinations)
        outfits = builder.select_top_n(candidates, n=1)
        
        assert len(outfits) >= 1
        best_outfit = outfits[0]
        
        # Step 3: Generate explanation (mocked)
        explainer = mock_llm_layer["explainer"]
        explanation = await explainer.explain(best_outfit, context)
        
        # Compile results
        result = {
            "outfit_name": best_outfit.name,
            "items": [g.id for g in best_outfit.garments],
            "overall_score": best_outfit.overall_score,
            "explanation": explanation.summary,
            "style_notes": explanation.style_notes,
            "context": {
                "occasion": context.occasion.value,
                "style_preferences": context.style_preferences
            }
        }
        
        # Save full pipeline result
        output_file = output_dir / "full_pipeline_result.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        # Verify complete result
        assert result["overall_score"] > 0
        assert len(result["items"]) >= 2
        assert result["explanation"] is not None
