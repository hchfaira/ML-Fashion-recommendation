"""
Tests for Creativity Scorer — Layer 2 Style

Tests cover:
- Scorer initialization and config loading
- All domain knowledge is sourced from style_rules_config.json
- Rule-break detection (color, formality, pattern, texture, style fusion)
- Component score calculation
- CreativityLevel classification
- Recommendation generation
"""
import json
from pathlib import Path
from typing import List

import pytest

from src.layer2_style.creativity_scorer import (
    CreativityScorer,
    CreativityLevel,
    RuleBreakType,
    get_creativity_score,
    get_creativity_level,
    is_fashion_forward,
)
from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory, FormalityLevel,
    ColorInfo, PatternInfo,
)

_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "config" / "data" / "style_rules_config.json"
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def scorer():
    """Return a fresh CreativityScorer (loads from config)."""
    return CreativityScorer()


@pytest.fixture
def _cfg():
    """Return the raw creativity section of style_rules_config.json."""
    with open(_CONFIG_PATH) as f:
        return json.load(f)["creativity"]


@pytest.fixture
def casual_tshirt():
    return Garment(
        id="casual_tshirt",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorInfo(primary="white", hex_codes=[]),
            pattern=PatternInfo(type="solid"),
            style_tags=["casual", "basic"],
            formality_level=FormalityLevel.CASUAL,
        ),
    )


@pytest.fixture
def floral_blouse():
    return Garment(
        id="floral_blouse",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorInfo(primary="pink", hex_codes=[]),
            pattern=PatternInfo(type="floral", scale="medium"),
            style_tags=["feminine", "romantic"],
            formality_level=FormalityLevel.SMART_CASUAL,
        ),
    )


@pytest.fixture
def formal_blazer():
    return Garment(
        id="formal_blazer",
        attributes=GarmentAttributes(
            category=GarmentCategory.OUTERWEAR,
            subcategory="blazer",
            color=ColorInfo(primary="navy", hex_codes=[]),
            pattern=PatternInfo(type="solid"),
            style_tags=["tailored", "elegant"],
            formality_level=FormalityLevel.BUSINESS,
        ),
    )


@pytest.fixture
def sporty_hoodie():
    return Garment(
        id="sporty_hoodie",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorInfo(primary="grey", hex_codes=[]),
            pattern=PatternInfo(type="solid"),
            style_tags=["sporty", "athleisure", "activewear"],
            formality_level=FormalityLevel.VERY_CASUAL,
        ),
    )


@pytest.fixture
def camo_pants():
    return Garment(
        id="camo_pants",
        attributes=GarmentAttributes(
            category=GarmentCategory.BOTTOM,
            color=ColorInfo(primary="green", hex_codes=[]),
            pattern=PatternInfo(type="camo"),
            style_tags=["streetwear", "urban"],
            formality_level=FormalityLevel.CASUAL,
        ),
    )


# ============================================================
# Config-driven initialization tests
# ============================================================

@pytest.mark.unit
class TestCreativityScorerConfig:
    """Verify that all domain knowledge is sourced from config, not hardcoded."""

    def test_scoring_weights_match_config(self, scorer, _cfg):
        """Scoring weights come from config."""
        assert scorer.weights == _cfg["scoring_weights"]

    def test_creative_combos_color_clashes_match_config(self, scorer, _cfg):
        """Color-clash combos come from config."""
        cfg_clashes = [tuple(p) for p in _cfg["creative_combos"]["color_clashes_that_work"]]
        assert scorer.CREATIVE_COMBOS["color_clashes_that_work"] == cfg_clashes

    def test_style_fusions_match_config(self, scorer, _cfg):
        """Style fusion combos come from config."""
        cfg_fusions = [tuple(p) for p in _cfg["creative_combos"]["style_fusions"]]
        assert scorer.CREATIVE_COMBOS["style_fusions"] == cfg_fusions

    def test_risk_pieces_patterns_match_config(self, scorer, _cfg):
        """Risk patterns come from config."""
        assert scorer.RISK_PIECES["patterns"] == _cfg["risk_pieces"]["patterns"]

    def test_style_tribes_match_config(self, scorer, _cfg):
        """Style tribes come from config."""
        assert scorer._style_tribes == _cfg["style_tribes"]

    def test_novel_materials_match_config(self, scorer, _cfg):
        """Novel materials come from config."""
        assert scorer._novel_materials == _cfg["novel_materials"]

    def test_thresholds_match_config(self, scorer, _cfg):
        """Numeric thresholds come from config."""
        t = _cfg["thresholds"]
        assert scorer._t_safe_max      == t["safe_max_score"]
        assert scorer._t_moderate_max  == t["moderate_max_score"]
        assert scorer._t_creative_max  == t["creative_max_score"]
        assert scorer._t_success_creat == t["success_creativity_min"]
        assert scorer._t_success_style == t["success_style_min"]


# ============================================================
# Functional scoring tests
# ============================================================

@pytest.mark.unit
class TestCreativityScoring:
    """Functional tests for creativity scoring."""

    def test_empty_outfit_returns_zero(self, scorer):
        """Empty outfit → zero creativity score."""
        result = scorer.analyze_creativity([])
        assert result.creativity_score == 0.0
        assert result.creativity_level == CreativityLevel.SAFE

    def test_single_safe_item_is_safe(self, scorer, casual_tshirt):
        """Single boring item → SAFE creativity."""
        result = scorer.analyze_creativity([casual_tshirt])
        assert result.creativity_level == CreativityLevel.SAFE

    def test_score_in_range(self, scorer, casual_tshirt, floral_blouse, formal_blazer):
        """Score is always between 0 and 1."""
        result = scorer.analyze_creativity([casual_tshirt, floral_blouse, formal_blazer])
        assert 0.0 <= result.creativity_score <= 1.0

    def test_all_component_scores_in_range(self, scorer, camo_pants, sporty_hoodie):
        """All component scores are bounded [0, 1]."""
        result = scorer.analyze_creativity([camo_pants, sporty_hoodie])
        assert 0.0 <= result.rule_breaking_index <= 1.0
        assert 0.0 <= result.intentionality_score <= 1.0
        assert 0.0 <= result.novelty_score <= 1.0
        assert 0.0 <= result.risk_score <= 1.0

    def test_risk_piece_increases_risk_score(self, scorer, casual_tshirt, camo_pants):
        """Outfit with a risk pattern (camo) scores higher risk than plain outfit."""
        safe_result = scorer.analyze_creativity([casual_tshirt])
        risky_result = scorer.analyze_creativity([casual_tshirt, camo_pants])
        assert risky_result.risk_score >= safe_result.risk_score

    def test_formality_mix_detected(self, scorer, sporty_hoodie, formal_blazer):
        """Mixing very casual + business items creates a formality-mix rule break."""
        result = scorer.analyze_creativity([sporty_hoodie, formal_blazer])
        types = [rb.rule_type for rb in result.rules_broken]
        assert RuleBreakType.FORMALITY_MIX in types

    def test_fashion_forward_score_in_range(self, scorer, floral_blouse, formal_blazer):
        """Fashion-forward score is bounded [0, 1]."""
        result = scorer.analyze_creativity([floral_blouse, formal_blazer], style_score=0.8)
        assert 0.0 <= result.fashion_forward_score <= 1.0

    def test_recommendation_is_string(self, scorer, casual_tshirt):
        """Recommendation is always a non-empty string."""
        result = scorer.analyze_creativity([casual_tshirt])
        assert isinstance(result.recommendation, str)
        assert len(result.recommendation) > 0


@pytest.mark.unit
class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_creativity_score_returns_float(self, casual_tshirt):
        score = get_creativity_score([casual_tshirt])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_get_creativity_level_returns_string(self, casual_tshirt):
        level = get_creativity_level([casual_tshirt])
        assert isinstance(level, str)
        assert level in [lvl.value for lvl in CreativityLevel]

    def test_is_fashion_forward_returns_bool(self, floral_blouse, formal_blazer):
        result = is_fashion_forward([floral_blouse, formal_blazer], style_score=0.8)
        assert isinstance(result, bool)
