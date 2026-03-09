"""
Tests for Proportion Harmonizer
"""
import pytest
from unittest.mock import MagicMock, patch

from src.layer3_context.proportion_harmonizer import (
    ProportionHarmonizer,
    ProportionAnalysis,
    ProportionScore
)


class TestProportionAnalysis:
    """Tests for ProportionAnalysis dataclass."""
    
    def test_proportion_analysis_creation(self):
        """Test creating a ProportionAnalysis."""
        analysis = ProportionAnalysis(
            torso_proportion="short",
            leg_proportion="long",
            frame_size="medium",
            balance_needed=["elongate_torso"],
            recommended_silhouettes=["high_waist", "v_neck"],
            avoid_silhouettes=["low_rise"]
        )
        
        assert analysis.torso_proportion == "short"
        assert analysis.leg_proportion == "long"
        assert analysis.frame_size == "medium"
        assert "elongate_torso" in analysis.balance_needed
        assert "high_waist" in analysis.recommended_silhouettes
    
    def test_proportion_analysis_defaults(self):
        """Test ProportionAnalysis default values."""
        analysis = ProportionAnalysis(
            torso_proportion="average",
            leg_proportion="average",
            frame_size="medium"
        )
        
        assert analysis.balance_needed == []
        assert analysis.recommended_silhouettes == []
        assert analysis.avoid_silhouettes == []


class TestProportionScore:
    """Tests for ProportionScore dataclass."""
    
    def test_proportion_score_creation(self):
        """Test creating a ProportionScore."""
        score = ProportionScore(
            score=0.85,
            harmony_level="excellent",
            positive_factors=["Good choice: high waist"],
            negative_factors=[],
            styling_tips=["Wear high-waisted bottoms"]
        )
        
        assert score.score == 0.85
        assert score.harmony_level == "excellent"
        assert len(score.positive_factors) == 1
    
    def test_proportion_score_defaults(self):
        """Test ProportionScore default values."""
        score = ProportionScore(
            score=0.7,
            harmony_level="good"
        )
        
        assert score.positive_factors == []
        assert score.negative_factors == []
        assert score.styling_tips == []


class TestProportionHarmonizer:
    """Tests for ProportionHarmonizer class."""
    
    @pytest.fixture
    def harmonizer(self):
        """Create a ProportionHarmonizer instance."""
        with patch('src.layer3_context.proportion_harmonizer.get_config') as mock_config:
            mock_config.return_value.get_data.return_value = {}
            return ProportionHarmonizer()
    
    @pytest.fixture
    def mock_garment(self):
        """Create a mock garment."""
        garment = MagicMock()
        garment.id = "test_garment_1"
        garment.category = "top"
        garment.attributes = {}
        return garment
    
    def test_init(self, harmonizer):
        """Test ProportionHarmonizer initialization."""
        assert harmonizer is not None
        assert harmonizer.proportion_rules is not None
        assert harmonizer.frame_size_matching is not None
    
    def test_classify_torso_short(self, harmonizer):
        """Test classifying short torso."""
        assert harmonizer._classify_torso(0.26) == "short"
        assert harmonizer._classify_torso(0.27) == "short"
    
    def test_classify_torso_average(self, harmonizer):
        """Test classifying average torso."""
        assert harmonizer._classify_torso(0.29) == "average"
        assert harmonizer._classify_torso(0.30) == "average"
        assert harmonizer._classify_torso(0.31) == "average"
    
    def test_classify_torso_long(self, harmonizer):
        """Test classifying long torso."""
        assert harmonizer._classify_torso(0.33) == "long"
        assert harmonizer._classify_torso(0.35) == "long"
    
    def test_classify_torso_none(self, harmonizer):
        """Test classifying torso with None ratio."""
        assert harmonizer._classify_torso(None) == "average"
    
    def test_classify_legs_short(self, harmonizer):
        """Test classifying short legs."""
        assert harmonizer._classify_legs(0.43) == "short"
        assert harmonizer._classify_legs(0.44) == "short"
    
    def test_classify_legs_average(self, harmonizer):
        """Test classifying average legs."""
        assert harmonizer._classify_legs(0.46) == "average"
        assert harmonizer._classify_legs(0.47) == "average"
        assert harmonizer._classify_legs(0.48) == "average"
    
    def test_classify_legs_long(self, harmonizer):
        """Test classifying long legs."""
        assert harmonizer._classify_legs(0.50) == "long"
        assert harmonizer._classify_legs(0.52) == "long"
    
    def test_classify_legs_none(self, harmonizer):
        """Test classifying legs with None ratio."""
        assert harmonizer._classify_legs(None) == "average"
    
    def test_analyze_proportions_short_torso(self, harmonizer):
        """Test analyzing proportions with short torso."""
        analysis = harmonizer.analyze_proportions(
            torso_ratio=0.26,
            leg_ratio=0.47,
            frame_size="medium"
        )
        
        assert analysis.torso_proportion == "short"
        assert analysis.leg_proportion == "average"
        assert "elongate_torso" in analysis.balance_needed
        assert "high_waist" in analysis.recommended_silhouettes
    
    def test_analyze_proportions_short_legs(self, harmonizer):
        """Test analyzing proportions with short legs."""
        analysis = harmonizer.analyze_proportions(
            torso_ratio=0.30,
            leg_ratio=0.43,
            frame_size="medium"
        )
        
        assert analysis.torso_proportion == "average"
        assert analysis.leg_proportion == "short"
        assert "elongate_legs" in analysis.balance_needed
        assert "vertical_lines" in analysis.recommended_silhouettes or "high_waist" in analysis.recommended_silhouettes
    
    def test_analyze_proportions_long_torso(self, harmonizer):
        """Test analyzing proportions with long torso."""
        analysis = harmonizer.analyze_proportions(
            torso_ratio=0.35,
            leg_ratio=0.47
        )
        
        assert analysis.torso_proportion == "long"
        assert "balance_torso" in analysis.balance_needed
    
    def test_analyze_proportions_long_legs(self, harmonizer):
        """Test analyzing proportions with long legs."""
        analysis = harmonizer.analyze_proportions(
            torso_ratio=0.30,
            leg_ratio=0.52
        )
        
        assert analysis.leg_proportion == "long"
        assert "balance_legs" in analysis.balance_needed
    
    def test_analyze_proportions_balanced(self, harmonizer):
        """Test analyzing balanced proportions."""
        analysis = harmonizer.analyze_proportions(
            torso_ratio=0.30,
            leg_ratio=0.47,
            frame_size="medium"
        )
        
        assert analysis.torso_proportion == "average"
        assert analysis.leg_proportion == "average"
        assert analysis.balance_needed == []
    
    def test_classify_harmony_excellent(self, harmonizer):
        """Test harmony classification for excellent score."""
        assert harmonizer._classify_harmony(0.90) == "excellent"
        assert harmonizer._classify_harmony(0.85) == "excellent"
    
    def test_classify_harmony_good(self, harmonizer):
        """Test harmony classification for good score."""
        assert harmonizer._classify_harmony(0.80) == "good"
        assert harmonizer._classify_harmony(0.70) == "good"
    
    def test_classify_harmony_fair(self, harmonizer):
        """Test harmony classification for fair score."""
        assert harmonizer._classify_harmony(0.60) == "fair"
        assert harmonizer._classify_harmony(0.50) == "fair"
    
    def test_classify_harmony_poor(self, harmonizer):
        """Test harmony classification for poor score."""
        assert harmonizer._classify_harmony(0.40) == "poor"
        assert harmonizer._classify_harmony(0.20) == "poor"
    
    def test_get_proportion_tips_short_torso(self, harmonizer):
        """Test getting tips for short torso."""
        tips = harmonizer.get_proportion_tips("short", "average")
        assert len(tips) > 0
        assert any("torso" in tip.lower() or "waist" in tip.lower() for tip in tips)
    
    def test_get_proportion_tips_short_legs(self, harmonizer):
        """Test getting tips for short legs."""
        tips = harmonizer.get_proportion_tips("average", "short")
        assert len(tips) > 0
        assert any("leg" in tip.lower() or "pants" in tip.lower() or "waist" in tip.lower() for tip in tips)
    
    def test_get_proportion_tips_balanced(self, harmonizer):
        """Test getting tips for balanced proportions."""
        tips = harmonizer.get_proportion_tips("average", "average")
        assert len(tips) > 0
        assert any("balanced" in tip.lower() or "most styles" in tip.lower() for tip in tips)
    
    def test_extract_outfit_features_high_waist(self, harmonizer):
        """Test extracting high waist feature."""
        garment = MagicMock()
        garment.attributes = {"waist": "high waist"}
        garment.category = "pants"
        
        features = harmonizer._extract_outfit_features([garment])
        assert "high_waist" in features
    
    def test_extract_outfit_features_v_neck(self, harmonizer):
        """Test extracting v-neck feature."""
        garment = MagicMock()
        garment.attributes = {"neckline": "v-neck"}
        garment.category = "top"
        
        features = harmonizer._extract_outfit_features([garment])
        assert "v_neck" in features
    
    def test_extract_outfit_features_vertical_stripes(self, harmonizer):
        """Test extracting vertical stripes feature."""
        garment = MagicMock()
        garment.attributes = {"pattern": "vertical stripes"}
        garment.category = "top"
        
        features = harmonizer._extract_outfit_features([garment])
        assert "vertical_stripes" in features
    
    def test_score_outfit_harmony_positive(self, harmonizer):
        """Test scoring outfit with positive harmony."""
        analysis = ProportionAnalysis(
            torso_proportion="short",
            leg_proportion="average",
            frame_size="medium",
            balance_needed=["elongate_torso"],
            recommended_silhouettes=["high_waist", "v_neck"],
            avoid_silhouettes=["low_rise"]
        )
        
        garment = MagicMock()
        garment.attributes = {"waist": "high waist"}
        garment.category = "pants"
        
        score = harmonizer.score_outfit_harmony([garment], analysis)
        
        assert score.score >= 0.7
        assert len(score.positive_factors) > 0
    
    def test_score_outfit_harmony_negative(self, harmonizer):
        """Test scoring outfit with negative harmony."""
        analysis = ProportionAnalysis(
            torso_proportion="short",
            leg_proportion="average",
            frame_size="medium",
            balance_needed=["elongate_torso"],
            recommended_silhouettes=["high_waist", "v_neck"],
            avoid_silhouettes=["low_rise", "long_tops"]
        )
        
        garment = MagicMock()
        garment.attributes = {"waist": "low rise"}
        garment.category = "pants"
        
        score = harmonizer.score_outfit_harmony([garment], analysis)
        
        # Should have lower score due to avoided feature
        assert len(score.negative_factors) > 0
    
    def test_score_outfit_harmony_empty(self, harmonizer):
        """Test scoring empty outfit."""
        analysis = ProportionAnalysis(
            torso_proportion="average",
            leg_proportion="average",
            frame_size="medium"
        )
        
        score = harmonizer.score_outfit_harmony([], analysis)
        
        assert score.score == 0.7  # Base score
        assert score.harmony_level in ["good", "fair"]
    
    def test_get_garment_weight(self, harmonizer):
        """Test getting garment weight."""
        garment = MagicMock()
        garment.attributes = {"weight": "lightweight"}
        
        weight = harmonizer._get_garment_weight(garment)
        assert weight == "lightweight"
    
    def test_get_garment_weight_none(self, harmonizer):
        """Test getting garment weight when not available."""
        garment = MagicMock()
        garment.attributes = {}
        
        weight = harmonizer._get_garment_weight(garment)
        assert weight is None


class TestProportionHarmonizerFrameSize:
    """Tests for frame size matching."""
    
    @pytest.fixture
    def harmonizer(self):
        """Create a ProportionHarmonizer instance."""
        with patch('src.layer3_context.proportion_harmonizer.get_config') as mock_config:
            mock_config.return_value.get_data.return_value = {}
            return ProportionHarmonizer()
    
    def test_small_frame_heavy_fabric_penalty(self, harmonizer):
        """Test penalty for heavy fabric on small frame."""
        analysis = ProportionAnalysis(
            torso_proportion="average",
            leg_proportion="average",
            frame_size="small"
        )
        
        garment = MagicMock()
        garment.attributes = {"weight": "heavy"}
        garment.category = "coat"
        
        score = harmonizer.score_outfit_harmony([garment], analysis)
        
        # Score should reflect penalty for heavy fabric on small frame
        assert score.score < 0.8 or len(score.negative_factors) > 0
    
    def test_large_frame_structured_bonus(self, harmonizer):
        """Test that large frames work well with structured pieces."""
        analysis = ProportionAnalysis(
            torso_proportion="average",
            leg_proportion="average",
            frame_size="large"
        )
        
        garment = MagicMock()
        garment.attributes = {"weight": "structured"}
        garment.category = "blazer"
        
        score = harmonizer.score_outfit_harmony([garment], analysis)
        
        # Should have positive factors for good match
        assert score.score >= 0.7


class TestProportionHarmonizerWithConfig:
    """Test ProportionHarmonizer with configuration."""
    
    def test_load_config_from_file(self, tmp_path):
        """Test loading configuration from file."""
        config_file = tmp_path / "test_config.json"
        config_content = {
            "proportion_harmonizer": {
                "short_torso": {
                    "recommended": ["high_waist", "empire_waist"],
                    "avoid": ["dropped_waist"],
                    "tips": ["Custom tip for short torso"]
                }
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_content))
        
        with patch('src.layer3_context.proportion_harmonizer.get_config') as mock_config:
            mock_config.return_value.get_data.return_value = {}
            harmonizer = ProportionHarmonizer(config_path=config_file)
        
        assert "high_waist" in harmonizer.proportion_rules["short_torso"]["recommended"]
        assert "Custom tip for short torso" in harmonizer.proportion_rules["short_torso"]["tips"]
