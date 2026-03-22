"""Unit tests for PromptToFilters (Layer 3)."""
from __future__ import annotations

import pytest

from src.layer3_context.prompt_to_filters import (
    PromptToFilters,
    WardrobeFilters,
    _COLOR_FAMILIES,
    _COLOR_TO_FAMILY,
)
from src.layer4_llm.prompt_parser import ParsedPrompt


def _parsed(**kwargs) -> ParsedPrompt:
    defaults = dict(raw_prompt="test", confidence=0.8)
    defaults.update(kwargs)
    return ParsedPrompt(**defaults)


# ---------------------------------------------------------------------------
# WardrobeFilters.to_dict
# ---------------------------------------------------------------------------

class TestWardrobeFiltersToDict:
    def test_contains_all_required_keys(self):
        f = WardrobeFilters()
        d = f.to_dict()
        for key in [
            "excluded_types", "formality_min", "formality_max",
            "target_colors", "color_families", "target_styles",
            "target_occasion", "preferred_patterns", "preferred_materials",
            "season_hint", "weight_color", "weight_style", "weight_occasion",
        ]:
            assert key in d

    def test_weights_sum_to_1(self):
        f = WardrobeFilters()
        assert abs(f.weight_color + f.weight_style + f.weight_occasion - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Off-topic passthrough
# ---------------------------------------------------------------------------

class TestOffTopicPassthrough:
    def setup_method(self):
        self.t = PromptToFilters()

    def test_off_topic_returns_empty_filters(self):
        p = _parsed(is_off_topic=True)
        f = self.t.translate(p)
        assert f.target_colors == []
        assert f.target_occasion is None
        assert f.excluded_types == []


# ---------------------------------------------------------------------------
# Formality translation
# ---------------------------------------------------------------------------

class TestFormalityTranslation:
    def setup_method(self):
        self.t = PromptToFilters()

    def test_wedding_formality(self):
        p = _parsed(occasion="wedding", formality_range=(0.7, 1.0))
        f = self.t.translate(p)
        assert f.formality_min == pytest.approx(0.7)
        assert f.formality_max == pytest.approx(1.0)

    def test_casual_formality(self):
        p = _parsed(occasion="casual", formality_range=(0.0, 0.4))
        f = self.t.translate(p)
        assert f.formality_max <= 0.45


# ---------------------------------------------------------------------------
# Color expansion
# ---------------------------------------------------------------------------

class TestColorExpansion:
    def setup_method(self):
        self.t = PromptToFilters()

    def test_navy_maps_to_blue_family(self):
        p = _parsed(colors=["navy"])
        f = self.t.translate(p)
        assert "blue" in f.color_families

    def test_unknown_color_kept_as_family(self):
        p = _parsed(colors=["fuchsia"])
        f = self.t.translate(p)
        # Not in map → kept directly
        assert "fuchsia" in f.color_families or "pink" in f.color_families

    def test_target_colors_preserved(self):
        p = _parsed(colors=["red", "black"])
        f = self.t.translate(p)
        assert "red" in f.target_colors
        assert "black" in f.target_colors

    def test_no_duplicate_families(self):
        # navy + cobalt → both blue family → deduplicated
        p = _parsed(colors=["navy", "cobalt"])
        f = self.t.translate(p)
        assert f.color_families.count("blue") == 1

    def test_get_color_shades_returns_family(self):
        shades = self.t.get_color_shades("navy")
        assert "cobalt" in shades
        assert "sky" in shades


# ---------------------------------------------------------------------------
# Style hints
# ---------------------------------------------------------------------------

class TestStyleHints:
    def setup_method(self):
        self.t = PromptToFilters()

    def test_minimalist_avoids_floral(self):
        p = _parsed(styles=["minimalist"])
        f = self.t.translate(p)
        assert "floral" in f.avoid_patterns

    def test_minimalist_prefers_solid(self):
        p = _parsed(styles=["minimalist"])
        f = self.t.translate(p)
        assert "solid" in f.preferred_patterns

    def test_bohemian_prefers_linen(self):
        p = _parsed(styles=["bohemian"])
        f = self.t.translate(p)
        assert "linen" in f.preferred_materials

    def test_no_duplicate_patterns(self):
        # Two styles that both prefer "solid"
        p = _parsed(styles=["minimalist", "classic"])
        f = self.t.translate(p)
        assert f.preferred_patterns.count("solid") == 1


# ---------------------------------------------------------------------------
# Mood / semantic expansion
# ---------------------------------------------------------------------------

class TestMoodExpansion:
    def setup_method(self):
        self.t = PromptToFilters()

    def test_comfortable_reduces_formality_max(self):
        p = _parsed(formality_range=(0.0, 1.0), mood="comfortable")
        f = self.t.translate(p)
        assert f.formality_max < 1.0

    def test_comfortable_prefers_cotton(self):
        p = _parsed(mood="comfortable")
        f = self.t.translate(p)
        assert "cotton" in f.preferred_materials

    def test_powerful_adds_bold_colors(self):
        p = _parsed(mood="powerful")
        f = self.t.translate(p)
        assert any(c in f.occasion_color_vibes for c in ["black", "navy", "burgundy"])

    def test_flowy_adds_silhouette_hint(self):
        p = _parsed(mood="flowy")
        f = self.t.translate(p)
        # flowy is in _SEMANTIC_EXPANSION under preferred_materials
        assert "chiffon" in f.preferred_materials or "silk" in f.preferred_materials


# ---------------------------------------------------------------------------
# garment_matches_color_filter
# ---------------------------------------------------------------------------

class TestGarmentMatchesColorFilter:
    def setup_method(self):
        self.t = PromptToFilters()

    def _filters_for(self, colors):
        p = _parsed(colors=colors)
        return self.t.translate(p)

    def test_exact_match_returns_high_score(self):
        f = self._filters_for(["blue"])
        score = self.t.garment_matches_color_filter(["blue"], f)
        assert score > 0.5

    def test_no_filter_returns_neutral(self):
        f = WardrobeFilters()  # no colors
        assert self.t.garment_matches_color_filter(["red"], f) == pytest.approx(0.5)

    def test_family_match_gives_score(self):
        f = self._filters_for(["blue"])
        score = self.t.garment_matches_color_filter(["navy"], f)
        assert score > 0.0

    def test_no_overlap_returns_0(self):
        f = self._filters_for(["blue"])
        score = self.t.garment_matches_color_filter(["red"], f, strict=True)
        assert score == 0.0

    def test_case_insensitive(self):
        f = self._filters_for(["blue"])
        score = self.t.garment_matches_color_filter(["Blue"], f)
        assert score > 0.0


# ---------------------------------------------------------------------------
# garment_matches_formality
# ---------------------------------------------------------------------------

class TestGarmentMatchesFormality:
    def setup_method(self):
        self.t = PromptToFilters()

    def test_within_range_passes(self):
        f = WardrobeFilters(formality_min=0.6, formality_max=1.0)
        assert self.t.garment_matches_formality(0.8, f) is True

    def test_below_range_fails(self):
        f = WardrobeFilters(formality_min=0.6, formality_max=1.0)
        assert self.t.garment_matches_formality(0.3, f) is False

    def test_none_formality_passes(self):
        f = WardrobeFilters(formality_min=0.6, formality_max=1.0)
        assert self.t.garment_matches_formality(None, f) is True


# ---------------------------------------------------------------------------
# Dynamic weight computation
# ---------------------------------------------------------------------------

class TestDynamicWeights:
    def setup_method(self):
        self.t = PromptToFilters()

    def test_color_only_boosts_color_weight(self):
        p = _parsed(colors=["blue"])
        f = self.t.translate(p)
        assert f.weight_color >= 0.45

    def test_occasion_only_boosts_occasion_weight(self):
        p = _parsed(occasion="wedding", formality_range=(0.7, 1.0))
        f = self.t.translate(p)
        assert f.weight_occasion >= 0.45

    def test_both_occasion_and_color_balances_weights(self):
        p = _parsed(colors=["blue"], occasion="wedding", formality_range=(0.7, 1.0))
        f = self.t.translate(p)
        total = f.weight_color + f.weight_style + f.weight_occasion
        assert abs(total - 1.0) < 1e-6
