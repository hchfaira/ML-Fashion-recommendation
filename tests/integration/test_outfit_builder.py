"""
Integration tests for Outfit Builder.

Tests cover:
- Single outfit scoring
- Combination generation
- Best outfit selection
- Report generation
"""
import pytest
import json
from pathlib import Path
from uuid import uuid4

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory,
    ColorInfo, PatternInfo, FormalityLevel, Season,
    UserContext, Occasion
)
from src.layer2_style import (
    OutfitBuilder,
    OutfitScorecard,
    OutfitCandidate,
)


# ============== Configuration ==============

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "integration"


# ============== Fixtures ==============

@pytest.fixture
def outfit_builder():
    """Create outfit builder with default context."""
    context = UserContext(
        user_id="test_user",
        occasion=Occasion.CASUAL,
        body_shape="rectangle",
        style_preferences=["minimalist", "classic"],
        color_preferences=["neutral", "blue"],
        skin_undertone="warm",
        color_season="autumn"
    )
    return OutfitBuilder(context)


@pytest.fixture
def sample_wardrobe():
    """Create a mock wardrobe for testing."""
    white_tshirt = Garment(
        id=str(uuid4()),
        image_path="wardrobe/tops/white_tshirt.jpg",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            subcategory="t-shirt",
            color=ColorInfo(primary="white", hex_codes=["#FFFFFF"]),
            pattern=PatternInfo(type="solid"),
            formality_level=FormalityLevel.CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER],
            fit="regular"
        )
    )
    
    blue_shirt = Garment(
        id=str(uuid4()),
        image_path="wardrobe/tops/blue_shirt.jpg",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            subcategory="button_down",
            color=ColorInfo(primary="blue", hex_codes=["#3B82F6"]),
            pattern=PatternInfo(type="solid"),
            formality_level=FormalityLevel.SMART_CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL],
            fit="regular"
        )
    )
    
    striped_shirt = Garment(
        id=str(uuid4()),
        image_path="wardrobe/tops/striped_shirt.jpg",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            subcategory="button_down",
            color=ColorInfo(primary="white", secondary="navy", hex_codes=["#FFFFFF", "#1E3A5F"]),
            pattern=PatternInfo(type="stripes", scale="medium"),
            formality_level=FormalityLevel.SMART_CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER],
            fit="regular"
        )
    )
    
    blue_jeans = Garment(
        id=str(uuid4()),
        image_path="wardrobe/bottoms/blue_jeans.jpg",
        attributes=GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            subcategory="jeans",
            color=ColorInfo(primary="blue", hex_codes=["#1E40AF"]),
            pattern=PatternInfo(type="solid"),
            formality_level=FormalityLevel.CASUAL,
            season_suitable=[Season.SPRING, Season.FALL, Season.WINTER],
            fit="slim"
        )
    )
    
    khaki_chinos = Garment(
        id=str(uuid4()),
        image_path="wardrobe/bottoms/khaki_chinos.jpg",
        attributes=GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            subcategory="chinos",
            color=ColorInfo(primary="khaki", hex_codes=["#C3B091"]),
            pattern=PatternInfo(type="solid"),
            formality_level=FormalityLevel.SMART_CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL],
            fit="regular"
        )
    )
    
    charcoal_trousers = Garment(
        id=str(uuid4()),
        image_path="wardrobe/bottoms/charcoal_trousers.jpg",
        attributes=GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            subcategory="dress_pants",
            color=ColorInfo(primary="charcoal", hex_codes=["#36454F"]),
            pattern=PatternInfo(type="solid"),
            formality_level=FormalityLevel.BUSINESS,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
            fit="tailored"
        )
    )
    
    white_sneakers = Garment(
        id=str(uuid4()),
        image_path="wardrobe/shoes/white_sneakers.jpg",
        attributes=GarmentAttributes(
            category=GarmentCategory.SHOES,
            subcategory="sneakers",
            color=ColorInfo(primary="white", hex_codes=["#FFFFFF"]),
            pattern=PatternInfo(type="solid"),
            formality_level=FormalityLevel.CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL],
            fit="regular"
        )
    )
    
    brown_loafers = Garment(
        id=str(uuid4()),
        image_path="wardrobe/shoes/brown_loafers.jpg",
        attributes=GarmentAttributes(
            category=GarmentCategory.SHOES,
            subcategory="loafers",
            color=ColorInfo(primary="brown", hex_codes=["#8B4513"]),
            pattern=PatternInfo(type="solid"),
            formality_level=FormalityLevel.SMART_CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL],
            fit="regular"
        )
    )
    
    black_oxfords = Garment(
        id=str(uuid4()),
        image_path="wardrobe/shoes/black_oxfords.jpg",
        attributes=GarmentAttributes(
            category=GarmentCategory.SHOES,
            subcategory="oxford",
            color=ColorInfo(primary="black", hex_codes=["#000000"]),
            pattern=PatternInfo(type="solid"),
            formality_level=FormalityLevel.FORMAL,
            season_suitable=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
            fit="regular"
        )
    )
    
    return {
        "tops": [white_tshirt, blue_shirt, striped_shirt],
        "bottoms": [blue_jeans, khaki_chinos, charcoal_trousers],
        "shoes": [white_sneakers, brown_loafers, black_oxfords]
    }


# ============== Test Classes ==============

@pytest.mark.integration
class TestSingleOutfitScoring:
    """Tests for single outfit scoring."""
    
    def test_score_single_outfit(self, sample_wardrobe, outfit_builder):
        """Test scoring a single outfit."""
        outfit_garments = [
            sample_wardrobe["tops"][0],      # white t-shirt
            sample_wardrobe["bottoms"][0],   # blue jeans
            sample_wardrobe["shoes"][0]      # white sneakers
        ]
        
        scorecard = outfit_builder.score_outfit(outfit_garments)
        
        assert "overall" in scorecard.scores
        assert 0 <= scorecard.scores["overall"] <= 1
    
    def test_scorecard_report_generation(self, sample_wardrobe, outfit_builder):
        """Test scorecard report generation."""
        outfit_garments = [
            sample_wardrobe["tops"][0],
            sample_wardrobe["bottoms"][0],
            sample_wardrobe["shoes"][0]
        ]
        
        scorecard = outfit_builder.score_outfit(outfit_garments)
        report = scorecard.to_report()
        
        assert len(report) > 0
        assert "overall" in report.lower() or "score" in report.lower()
    
    def test_scorecard_json_save(self, sample_wardrobe, outfit_builder, tmp_path):
        """Test saving scorecard to JSON."""
        outfit_garments = [
            sample_wardrobe["tops"][0],
            sample_wardrobe["bottoms"][0],
            sample_wardrobe["shoes"][0]
        ]
        
        scorecard = outfit_builder.score_outfit(outfit_garments)
        
        json_path = tmp_path / "test_scorecard.json"
        scorecard.save_json(str(json_path))
        
        assert json_path.exists()
        
        with open(json_path) as f:
            data = json.load(f)
        
        assert "scores" in data or "overall" in str(data)


@pytest.mark.integration
class TestOutfitCombinations:
    """Tests for outfit combination generation."""
    
    def test_generate_combinations(self, sample_wardrobe, outfit_builder):
        """Test outfit combination generation."""
        combinations = outfit_builder.generate_outfit_combinations(sample_wardrobe)
        
        expected = 3 * 3 * 3  # 27 combinations
        assert len(combinations) == expected
    
    def test_each_combination_has_three_items(
        self, sample_wardrobe, outfit_builder
    ):
        """Test each combination has three items."""
        combinations = outfit_builder.generate_outfit_combinations(sample_wardrobe)
        
        for combo in combinations:
            assert len(combo) == 3


@pytest.mark.integration
class TestBestOutfitSelection:
    """Tests for best outfit selection."""
    
    def test_evaluate_all_combinations(self, sample_wardrobe, outfit_builder):
        """Test evaluating all combinations."""
        combinations = outfit_builder.generate_outfit_combinations(sample_wardrobe)
        candidates = outfit_builder.evaluate_all_combinations(combinations)
        
        assert len(candidates) == len(combinations)
        
        for candidate in candidates:
            assert hasattr(candidate, 'overall_score')
            assert 0 <= candidate.overall_score <= 1
    
    def test_select_best_outfit(self, sample_wardrobe, outfit_builder):
        """Test selecting best outfit."""
        combinations = outfit_builder.generate_outfit_combinations(sample_wardrobe)
        candidates = outfit_builder.evaluate_all_combinations(combinations)
        
        best = outfit_builder.select_best_outfit(candidates, criteria="overall")
        
        assert best is not None
        assert best.overall_score >= 0
    
    def test_generate_selection_report(self, sample_wardrobe, outfit_builder):
        """Test generating selection report."""
        combinations = outfit_builder.generate_outfit_combinations(sample_wardrobe)
        candidates = outfit_builder.evaluate_all_combinations(combinations)
        
        report = outfit_builder.generate_selection_report(candidates, top_n=5)
        
        assert len(report) > 0
