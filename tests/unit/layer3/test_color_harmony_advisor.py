"""
Tests for Color Harmony Advisor
"""
import pytest
from unittest.mock import MagicMock, patch

from src.layer3_context.color_harmony_advisor import (
    ColorHarmonyAdvisor,
    ColorProfile,
    ColorRecommendation,
    ColorHarmonyScore
)


class TestColorProfile:
    """Tests for ColorProfile dataclass."""
    
    def test_color_profile_creation(self):
        """Test creating a ColorProfile."""
        profile = ColorProfile(
            skin_tone="FAIR",
            undertone="COOL",
            hair_color="BLACK",
            contrast_level="HIGH"
        )
        
        assert profile.skin_tone == "FAIR"
        assert profile.undertone == "COOL"
        assert profile.hair_color == "BLACK"
        assert profile.contrast_level == "HIGH"
    
    def test_color_profile_season_cool_high_contrast(self):
        """Test color season for cool + high contrast = Winter."""
        profile = ColorProfile(
            skin_tone="FAIR",
            undertone="COOL",
            hair_color="BLACK",
            contrast_level="HIGH"
        )
        
        assert profile.season == "WINTER"
    
    def test_color_profile_season_cool_low_contrast(self):
        """Test color season for cool + low contrast = Summer."""
        profile = ColorProfile(
            skin_tone="LIGHT",
            undertone="COOL",
            hair_color="BLONDE",
            contrast_level="LOW"
        )
        
        assert profile.season == "SUMMER"
    
    def test_color_profile_season_warm_high_contrast(self):
        """Test color season for warm + high contrast = Autumn."""
        profile = ColorProfile(
            skin_tone="TAN",
            undertone="WARM",
            hair_color="BROWN",
            contrast_level="HIGH"
        )
        
        assert profile.season == "AUTUMN"
    
    def test_color_profile_season_warm_low_contrast(self):
        """Test color season for warm + low contrast = Spring."""
        profile = ColorProfile(
            skin_tone="LIGHT",
            undertone="WARM",
            hair_color="BLONDE",
            contrast_level="LOW"
        )
        
        assert profile.season == "SPRING"
    
    def test_color_profile_season_neutral_high(self):
        """Test color season for neutral + high contrast."""
        profile = ColorProfile(
            skin_tone="MEDIUM",
            undertone="NEUTRAL",
            hair_color="BLACK",
            contrast_level="HIGH"
        )
        
        assert profile.season == "WINTER"
    
    def test_color_profile_season_neutral_low(self):
        """Test color season for neutral + low contrast."""
        profile = ColorProfile(
            skin_tone="MEDIUM",
            undertone="NEUTRAL",
            hair_color="BROWN",
            contrast_level="MEDIUM"
        )
        
        assert profile.season == "SUMMER"


class TestColorRecommendation:
    """Tests for ColorRecommendation dataclass."""
    
    def test_color_recommendation_creation(self):
        """Test creating a ColorRecommendation."""
        rec = ColorRecommendation(
            best_colors=["navy", "blue", "purple"],
            good_colors=["white", "black"],
            colors_to_avoid=["orange", "peach"],
            neutral_colors=["white", "gray"],
            accent_colors=["fuchsia"],
            tips=["Silver jewelry complements your undertone"]
        )
        
        assert "navy" in rec.best_colors
        assert "orange" in rec.colors_to_avoid
        assert len(rec.tips) == 1
    
    def test_color_recommendation_defaults(self):
        """Test ColorRecommendation default values."""
        rec = ColorRecommendation(
            best_colors=["blue"],
            good_colors=[],
            colors_to_avoid=[],
            neutral_colors=["gray"],
            accent_colors=[]
        )
        
        assert rec.tips == []


class TestColorHarmonyScore:
    """Tests for ColorHarmonyScore dataclass."""
    
    def test_color_harmony_score_creation(self):
        """Test creating a ColorHarmonyScore."""
        score = ColorHarmonyScore(
            score=0.85,
            harmony_level="excellent",
            matching_colors=["navy", "blue"],
            clashing_colors=[],
            notes=["Good color choices"]
        )
        
        assert score.score == 0.85
        assert score.harmony_level == "excellent"
        assert len(score.matching_colors) == 2
    
    def test_color_harmony_score_defaults(self):
        """Test ColorHarmonyScore default values."""
        score = ColorHarmonyScore(
            score=0.7,
            harmony_level="good"
        )
        
        assert score.matching_colors == []
        assert score.clashing_colors == []
        assert score.notes == []


class TestColorHarmonyAdvisor:
    """Tests for ColorHarmonyAdvisor class."""
    
    @pytest.fixture
    def advisor(self):
        """Create a ColorHarmonyAdvisor instance."""
        with patch('src.layer3_context.color_harmony_advisor.get_config') as mock_config:
            mock_config.return_value.get_data.return_value = {}
            return ColorHarmonyAdvisor()
    
    @pytest.fixture
    def mock_garment(self):
        """Create a mock garment."""
        garment = MagicMock()
        garment.id = "test_garment_1"
        garment.color = "navy"
        garment.attributes = {"color": "navy blue"}
        return garment
    
    def test_init(self, advisor):
        """Test ColorHarmonyAdvisor initialization."""
        assert advisor is not None
        assert advisor.undertone_colors is not None
        assert advisor.contrast_matching is not None
        assert advisor.skin_tone_colors is not None
    
    def test_create_color_profile(self, advisor):
        """Test creating a color profile."""
        profile = advisor.create_color_profile(
            skin_tone="fair",
            undertone="cool",
            hair_color="black",
            contrast_level="high"
        )
        
        assert isinstance(profile, ColorProfile)
        assert profile.skin_tone == "FAIR"
        assert profile.undertone == "COOL"
    
    def test_get_color_recommendations_cool(self, advisor):
        """Test recommendations for cool undertone."""
        profile = advisor.create_color_profile(
            skin_tone="fair",
            undertone="cool",
            hair_color="black",
            contrast_level="high"
        )
        
        recs = advisor.get_color_recommendations(profile)
        
        assert isinstance(recs, ColorRecommendation)
        assert len(recs.best_colors) > 0
        # Cool undertones should have cool colors
        assert any(c in recs.best_colors for c in ["navy", "blue", "purple", "pink"])
    
    def test_get_color_recommendations_warm(self, advisor):
        """Test recommendations for warm undertone."""
        profile = advisor.create_color_profile(
            skin_tone="tan",
            undertone="warm",
            hair_color="brown",
            contrast_level="medium"
        )
        
        recs = advisor.get_color_recommendations(profile)
        
        assert len(recs.best_colors) > 0
        # Warm undertones should have warm colors
        assert any(c in recs.best_colors for c in ["orange", "coral", "peach", "gold", "olive"])
    
    def test_get_color_recommendations_neutral(self, advisor):
        """Test recommendations for neutral undertone."""
        profile = advisor.create_color_profile(
            skin_tone="medium",
            undertone="neutral",
            hair_color="brown",
            contrast_level="medium"
        )
        
        recs = advisor.get_color_recommendations(profile)
        
        assert len(recs.best_colors) > 0
        # Neutral can wear more colors
    
    def test_get_color_recommendations_includes_tips(self, advisor):
        """Test that recommendations include styling tips."""
        profile = advisor.create_color_profile(
            skin_tone="fair",
            undertone="cool",
            hair_color="black",
            contrast_level="high"
        )
        
        recs = advisor.get_color_recommendations(profile)
        
        assert len(recs.tips) > 0
    
    def test_get_color_recommendations_accent_colors(self, advisor):
        """Test that recommendations include accent colors."""
        profile = advisor.create_color_profile(
            skin_tone="fair",
            undertone="cool",
            hair_color="black",
            contrast_level="high"
        )
        
        recs = advisor.get_color_recommendations(profile)
        
        assert len(recs.accent_colors) > 0
    
    def test_score_outfit_colors_matching(self, advisor):
        """Test scoring outfit with matching colors."""
        profile = advisor.create_color_profile(
            skin_tone="fair",
            undertone="cool",
            hair_color="black",
            contrast_level="high"
        )
        
        garment = MagicMock()
        garment.color = "navy"
        garment.attributes = {}
        
        score = advisor.score_outfit_colors([garment], profile)
        
        assert isinstance(score, ColorHarmonyScore)
        assert score.score >= 0.7
        assert "navy" in score.matching_colors
    
    def test_score_outfit_colors_clashing(self, advisor):
        """Test scoring outfit with clashing colors."""
        profile = advisor.create_color_profile(
            skin_tone="fair",
            undertone="cool",
            hair_color="black",
            contrast_level="high"
        )
        
        garment = MagicMock()
        garment.color = "orange"
        garment.attributes = {}
        
        score = advisor.score_outfit_colors([garment], profile)
        
        # Orange should clash with cool undertones
        assert "orange" in score.clashing_colors or score.score < 0.7
    
    def test_score_outfit_colors_empty(self, advisor):
        """Test scoring empty outfit."""
        profile = advisor.create_color_profile(
            skin_tone="medium",
            undertone="neutral",
            hair_color="brown",
            contrast_level="medium"
        )
        
        score = advisor.score_outfit_colors([], profile)
        
        assert score.score == 0.7  # Base score
    
    def test_color_matches_exact(self, advisor):
        """Test exact color matching."""
        assert advisor._color_matches("navy", ["navy", "blue"]) is True
        assert advisor._color_matches("blue", ["navy", "blue"]) is True
    
    def test_color_matches_partial(self, advisor):
        """Test partial color matching."""
        assert advisor._color_matches("light blue", ["blue"]) is True
        assert advisor._color_matches("dark navy", ["navy"]) is True
    
    def test_color_matches_no_match(self, advisor):
        """Test non-matching colors."""
        assert advisor._color_matches("red", ["blue", "green"]) is False
    
    def test_analyze_outfit_contrast_high(self, advisor):
        """Test analyzing high contrast outfit."""
        colors = ["white", "black"]
        contrast = advisor._analyze_outfit_contrast(colors)
        assert contrast == "high_contrast"
    
    def test_analyze_outfit_contrast_medium(self, advisor):
        """Test analyzing medium contrast outfit."""
        colors = ["navy", "cream"]
        contrast = advisor._analyze_outfit_contrast(colors)
        # Should detect light (cream) or dark (navy)
        assert contrast in ["medium_contrast", "high_contrast"]
    
    def test_analyze_outfit_contrast_low(self, advisor):
        """Test analyzing low contrast outfit."""
        colors = ["beige", "tan"]
        contrast = advisor._analyze_outfit_contrast(colors)
        assert contrast == "low_contrast"
    
    def test_analyze_outfit_contrast_empty(self, advisor):
        """Test analyzing empty outfit."""
        contrast = advisor._analyze_outfit_contrast([])
        assert contrast == "medium_contrast"
    
    def test_classify_harmony_excellent(self, advisor):
        """Test harmony classification for excellent score."""
        assert advisor._classify_harmony(0.90) == "excellent"
        assert advisor._classify_harmony(0.85) == "excellent"
    
    def test_classify_harmony_good(self, advisor):
        """Test harmony classification for good score."""
        assert advisor._classify_harmony(0.80) == "good"
        assert advisor._classify_harmony(0.70) == "good"
    
    def test_classify_harmony_fair(self, advisor):
        """Test harmony classification for fair score."""
        assert advisor._classify_harmony(0.60) == "fair"
        assert advisor._classify_harmony(0.50) == "fair"
    
    def test_classify_harmony_poor(self, advisor):
        """Test harmony classification for poor score."""
        assert advisor._classify_harmony(0.40) == "poor"
        assert advisor._classify_harmony(0.20) == "poor"
    
    def test_get_season_accents_spring(self, advisor):
        """Test season accent colors for Spring."""
        accents = advisor._get_season_accents("SPRING")
        assert len(accents) > 0
        assert any("coral" in c or "peach" in c for c in accents)
    
    def test_get_season_accents_summer(self, advisor):
        """Test season accent colors for Summer."""
        accents = advisor._get_season_accents("SUMMER")
        assert len(accents) > 0
        assert any("lavender" in c or "pink" in c for c in accents)
    
    def test_get_season_accents_autumn(self, advisor):
        """Test season accent colors for Autumn."""
        accents = advisor._get_season_accents("AUTUMN")
        assert len(accents) > 0
        assert any("rust" in c or "mustard" in c for c in accents)
    
    def test_get_season_accents_winter(self, advisor):
        """Test season accent colors for Winter."""
        accents = advisor._get_season_accents("WINTER")
        assert len(accents) > 0
        assert any("fuchsia" in c or "royal" in c or "emerald" in c for c in accents)
    
    def test_get_color_tips_cool_undertone(self, advisor):
        """Test color tips for cool undertone."""
        profile = advisor.create_color_profile(
            skin_tone="fair",
            undertone="cool",
            hair_color="black",
            contrast_level="high"
        )
        
        tips = advisor.get_color_tips(profile)
        
        assert len(tips) > 0
        assert any("silver" in tip.lower() for tip in tips)
    
    def test_get_color_tips_warm_undertone(self, advisor):
        """Test color tips for warm undertone."""
        profile = advisor.create_color_profile(
            skin_tone="tan",
            undertone="warm",
            hair_color="brown",
            contrast_level="medium"
        )
        
        tips = advisor.get_color_tips(profile)
        
        assert len(tips) > 0
        assert any("gold" in tip.lower() for tip in tips)
    
    def test_get_color_tips_high_contrast(self, advisor):
        """Test color tips for high contrast."""
        profile = advisor.create_color_profile(
            skin_tone="fair",
            undertone="cool",
            hair_color="black",
            contrast_level="HIGH"
        )
        
        tips = advisor.get_color_tips(profile)
        
        assert any("bold" in tip.lower() or "contrast" in tip.lower() for tip in tips)
    
    def test_get_color_tips_low_contrast(self, advisor):
        """Test color tips for low contrast."""
        profile = advisor.create_color_profile(
            skin_tone="light",
            undertone="cool",
            hair_color="blonde",
            contrast_level="LOW"
        )
        
        tips = advisor.get_color_tips(profile)
        
        assert any("tonal" in tip.lower() or "contrast" in tip.lower() or "range" in tip.lower() for tip in tips)


class TestColorHarmonyAdvisorWithConfig:
    """Test ColorHarmonyAdvisor with configuration."""
    
    def test_load_config_from_file(self, tmp_path):
        """Test loading configuration from file."""
        config_file = tmp_path / "test_config.json"
        config_content = {
            "color_harmony": {
                "undertone_colors": {
                    "COOL": {
                        "best": ["custom_blue", "custom_purple"],
                        "good": ["custom_white"],
                        "avoid": ["custom_orange"]
                    }
                }
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_content))
        
        with patch('src.layer3_context.color_harmony_advisor.get_config') as mock_config:
            mock_config.return_value.get_data.return_value = {}
            advisor = ColorHarmonyAdvisor(config_path=config_file)
        
        assert "custom_blue" in advisor.undertone_colors["COOL"]["best"]
    
    def test_extract_outfit_colors_from_color_attr(self):
        """Test extracting colors from garment.color attribute."""
        with patch('src.layer3_context.color_harmony_advisor.get_config') as mock_config:
            mock_config.return_value.get_data.return_value = {}
            advisor = ColorHarmonyAdvisor()
        
        garment = MagicMock()
        garment.color = "navy"
        garment.attributes = {}
        
        colors = advisor._extract_outfit_colors([garment])
        assert "navy" in colors
    
    def test_extract_outfit_colors_from_attributes(self):
        """Test extracting colors from garment.attributes dict."""
        with patch('src.layer3_context.color_harmony_advisor.get_config') as mock_config:
            mock_config.return_value.get_data.return_value = {}
            advisor = ColorHarmonyAdvisor()
        
        garment = MagicMock()
        garment.color = None
        garment.attributes = {"color": "burgundy"}
        
        colors = advisor._extract_outfit_colors([garment])
        assert "burgundy" in colors
    
    def test_extract_outfit_colors_primary_secondary(self):
        """Test extracting primary and secondary colors."""
        with patch('src.layer3_context.color_harmony_advisor.get_config') as mock_config:
            mock_config.return_value.get_data.return_value = {}
            advisor = ColorHarmonyAdvisor()
        
        garment = MagicMock()
        garment.color = None
        garment.attributes = {
            "primary_color": "blue",
            "secondary_color": "white"
        }
        
        colors = advisor._extract_outfit_colors([garment])
        assert "blue" in colors
        assert "white" in colors
