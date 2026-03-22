"""Unit tests for PromptAwareScorer (Layer 2)."""
from __future__ import annotations

import pytest

from src.layer2_style.prompt_aware_scorer import PromptAwareScorer
from src.layer3_context.prompt_to_filters import PromptToFilters, WardrobeFilters
from src.layer4_llm.prompt_parser import ParsedPrompt


def _filters(**kwargs) -> WardrobeFilters:
    return WardrobeFilters(**kwargs)


def _full_filters(
    colors=None, styles=None, occasion=None,
    formality_min=0.0, formality_max=1.0,
) -> WardrobeFilters:
    p = ParsedPrompt(
        raw_prompt="test",
        colors=colors or [],
        styles=styles or [],
        occasion=occasion,
        formality_range=(formality_min, formality_max),
        confidence=0.8,
    )
    return PromptToFilters().translate(p)


# ---------------------------------------------------------------------------
# Color sub-score
# ---------------------------------------------------------------------------

class TestColorScore:
    def setup_method(self):
        self.scorer = PromptAwareScorer()

    def test_no_color_filter_returns_neutral(self):
        f = _filters()
        score = self.scorer._color_score({"dominant_colors": ["red"]}, f)
        assert score == pytest.approx(0.5)

    def test_exact_match_high_score(self):
        f = _full_filters(colors=["blue"])
        score = self.scorer._color_score({"dominant_colors": ["blue"]}, f)
        assert score > 0.5

    def test_no_colors_in_outfit_penalised(self):
        f = _full_filters(colors=["blue"])
        score = self.scorer._color_score({}, f)
        assert score < 0.5

    def test_case_insensitive(self):
        f = _full_filters(colors=["blue"])
        score = self.scorer._color_score({"dominant_colors": ["Blue"]}, f)
        assert score > 0.5


# ---------------------------------------------------------------------------
# Style sub-score
# ---------------------------------------------------------------------------

class TestStyleScore:
    def setup_method(self):
        self.scorer = PromptAwareScorer()

    def test_no_style_filter_neutral(self):
        f = _filters()
        score = self.scorer._style_score({"dominant_styles": {"minimalist": 0.9}}, f)
        assert score == pytest.approx(0.5)

    def test_matching_style_high_score(self):
        f = _full_filters(styles=["minimalist"])
        score = self.scorer._style_score({"dominant_styles": {"minimalist": 0.9}}, f)
        assert score > 0.5

    def test_no_style_tags_penalised(self):
        f = _full_filters(styles=["minimalist"])
        score = self.scorer._style_score({}, f)
        assert score <= 0.3

    def test_style_tag_fallback(self):
        f = _full_filters(styles=["minimalist"])
        score = self.scorer._style_score({"style_tags": ["minimalist"]}, f)
        assert score > 0.5


# ---------------------------------------------------------------------------
# Occasion sub-score
# ---------------------------------------------------------------------------

class TestOccasionScore:
    def setup_method(self):
        self.scorer = PromptAwareScorer()

    def test_formality_in_range_perfect(self):
        f = _filters(formality_min=0.7, formality_max=1.0)
        score = self.scorer._occasion_score({"formality_score": 0.85}, f)
        assert score >= 0.5

    def test_formality_outside_range_penalised(self):
        f = _filters(formality_min=0.7, formality_max=1.0)
        score = self.scorer._occasion_score({"formality_score": 0.2}, f)
        assert score < 0.5

    def test_matching_occasion_tag_boosts_score(self):
        f = _filters(
            formality_min=0.7, formality_max=1.0,
            target_occasion="wedding",
        )
        score = self.scorer._occasion_score(
            {"formality_score": 0.85, "occasion_tags": ["wedding"]}, f
        )
        assert score >= 0.75

    def test_no_formality_neutral(self):
        f = _filters(formality_min=0.7, formality_max=1.0)
        score = self.scorer._occasion_score({}, f)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Full score_outfit
# ---------------------------------------------------------------------------

class TestScoreOutfit:
    def setup_method(self):
        self.scorer = PromptAwareScorer()

    def test_returns_float_in_0_1(self):
        f = _full_filters(colors=["blue"], occasion="wedding")
        score = self.scorer.score_outfit({}, f)
        assert 0.0 <= score <= 1.0

    def test_perfect_match_high_score(self):
        f = _full_filters(colors=["blue"], styles=["elegant"], occasion="wedding", formality_min=0.7, formality_max=1.0)
        outfit = {
            "dominant_colors": ["blue"],
            "dominant_styles": {"elegant": 0.9},
            "formality_score": 0.85,
            "occasion_tags": ["wedding"],
        }
        score = self.scorer.score_outfit(outfit, f)
        assert score >= 0.6

    def test_empty_outfit_gives_neutral_ish_score(self):
        f = _full_filters(colors=["blue"])
        score = self.scorer.score_outfit({}, f)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# apply_prompt_boost
# ---------------------------------------------------------------------------

class TestApplyPromptBoost:
    def setup_method(self):
        self.scorer = PromptAwareScorer()

    def test_preference_0_returns_base(self):
        assert self.scorer.apply_prompt_boost(0.8, 0.2, preference=0.0) == pytest.approx(0.8)

    def test_preference_1_returns_prompt(self):
        assert self.scorer.apply_prompt_boost(0.8, 0.2, preference=1.0) == pytest.approx(0.2)

    def test_default_preference_blends(self):
        result = self.scorer.apply_prompt_boost(0.6, 1.0, preference=0.4)
        assert result == pytest.approx(0.6 * 0.6 + 1.0 * 0.4)

    def test_clamped_to_0_1(self):
        result = self.scorer.apply_prompt_boost(-0.5, 2.0, preference=0.5)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# passes_hard_filters
# ---------------------------------------------------------------------------

class TestPassesHardFilters:
    def setup_method(self):
        self.scorer = PromptAwareScorer()

    def test_excluded_type_rejected(self):
        f = _filters(excluded_types=["skirt"])
        assert not self.scorer.passes_hard_filters({"garment_type": "skirt"}, f)

    def test_non_excluded_type_passes(self):
        f = _filters(excluded_types=["skirt"])
        assert self.scorer.passes_hard_filters({"garment_type": "dress"}, f)

    def test_no_exclusions_always_passes(self):
        f = _filters()
        assert self.scorer.passes_hard_filters({"garment_type": "anything"}, f)

    def test_case_insensitive_exclusion(self):
        f = _filters(excluded_types=["skirt"])
        assert not self.scorer.passes_hard_filters({"garment_type": "Skirt"}, f)


# ---------------------------------------------------------------------------
# get_dynamic_weights
# ---------------------------------------------------------------------------

class TestGetDynamicWeights:
    def setup_method(self):
        self.scorer = PromptAwareScorer()

    def test_wedding_boosts_occasion(self):
        f = _filters(target_occasion="wedding")
        weights = self.scorer.get_dynamic_weights(f)
        assert weights["occasion"] > 1.0

    def test_minimalist_boosts_three_color(self):
        f = _filters(target_styles=["minimalist"])
        weights = self.scorer.get_dynamic_weights(f)
        assert weights["three_color"] > 1.0

    def test_returns_all_keys(self):
        f = _filters()
        weights = self.scorer.get_dynamic_weights(f)
        for key in ["color", "style", "occasion", "formality", "creativity"]:
            assert key in weights


# ---------------------------------------------------------------------------
# rank_outfits
# ---------------------------------------------------------------------------

class TestRankOutfits:
    def setup_method(self):
        self.scorer = PromptAwareScorer()

    def test_sorted_descending_by_final_score(self):
        f = _full_filters(colors=["blue"])
        outfits = [
            {"dominant_colors": ["red"], "id": "low"},
            {"dominant_colors": ["blue"], "id": "high"},
        ]
        ranked = self.scorer.rank_outfits(outfits, f)
        assert ranked[0]["id"] == "high"

    def test_adds_prompt_score_key(self):
        f = _full_filters(colors=["blue"])
        ranked = self.scorer.rank_outfits([{"dominant_colors": ["blue"]}], f)
        assert "prompt_score" in ranked[0]
        assert "final_score" in ranked[0]

    def test_empty_outfits_returns_empty(self):
        f = _filters()
        assert self.scorer.rank_outfits([], f) == []

    def test_base_scores_respected(self):
        f = _full_filters(colors=["blue"])
        outfits = [
            {"dominant_colors": ["blue"], "id": "A"},
            {"dominant_colors": ["blue"], "id": "B"},
        ]
        # A gets a much higher base score
        ranked = self.scorer.rank_outfits(outfits, f, base_scores=[0.9, 0.1], preference=0.0)
        assert ranked[0]["id"] == "A"
