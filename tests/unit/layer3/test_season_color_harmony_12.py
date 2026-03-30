"""
Tests for the 12-sub-season extension in SeasonColorHarmonyScorer.

Covers:
- ColorSeasonSub enum has 12 members
- sub-palettes & sub-clashes are initialised for all 12 subs
- analyze_outfit uses sub-palettes when season_sub is provided
- analyze_outfit falls back to 4-season when season_sub is None or unknown
"""

import pytest

from src.layer2_style.season_color_harmony import (
    ColorSeason,
    ColorSeasonSub,
    SeasonColorHarmonyScorer,
)


@pytest.fixture
def scorer():
    return SeasonColorHarmonyScorer()


# ---------------------------------------------------------------------------
# ColorSeasonSub enum
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestColorSeasonSubEnum:

    def test_has_12_members(self):
        assert len(ColorSeasonSub) == 12

    @pytest.mark.parametrize("name", [
        "Light Spring", "True Spring", "Warm Spring",
        "Light Summer", "True Summer", "Cool Summer",
        "True Autumn", "Warm Autumn", "Deep Autumn",
        "True Winter", "Cool Winter", "Deep Winter",
    ])
    def test_member_values(self, name):
        assert name in [m.value for m in ColorSeasonSub]


# ---------------------------------------------------------------------------
# Sub-palettes & sub-clashes initialisation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSubPalettesAndClashes:

    def test_all_12_sub_palettes_exist(self, scorer):
        for sub in ColorSeasonSub:
            assert sub.value in scorer.season_sub_palettes, f"Missing sub-palette for {sub.value}"
            assert len(scorer.season_sub_palettes[sub.value]) > 0

    def test_all_12_sub_clashes_exist(self, scorer):
        for sub in ColorSeasonSub:
            assert sub.value in scorer.season_sub_clashes, f"Missing sub-clash set for {sub.value}"
            assert len(scorer.season_sub_clashes[sub.value]) > 0


# ---------------------------------------------------------------------------
# analyze_outfit with season_sub
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnalyzeOutfitWithSeasonSub:
    """When season_sub is provided the finer sub-palette should be used."""

    @staticmethod
    def _make_garment(color: str):
        """Minimal garment stub with a primary color."""
        from unittest.mock import MagicMock
        g = MagicMock()
        g.attributes.color.primary = color
        return g

    def test_sub_palette_match_scores_higher(self, scorer):
        """A colour in the sub-palette should score >= a colour NOT in the sub-palette."""
        # "peach" is in Light Spring sub-palette
        items = [self._make_garment("peach")]
        result_with = scorer.analyze_outfit(
            items, ColorSeason.SPRING, season_sub="Light Spring"
        )
        # "marigold" is NOT in Light Spring sub-palette
        items2 = [self._make_garment("marigold")]
        result_without = scorer.analyze_outfit(
            items2, ColorSeason.SPRING, season_sub="Light Spring"
        )
        assert result_with.score >= result_without.score

    def test_fallback_when_season_sub_none(self, scorer):
        """Without season_sub the broad 4-season palette is used."""
        items = [self._make_garment("coral")]
        result = scorer.analyze_outfit(items, ColorSeason.SPRING, season_sub=None)
        # Should still get a valid result
        assert 0 <= result.score <= 1

    def test_fallback_when_season_sub_unknown(self, scorer):
        """An unrecognised sub-season string should fall back to 4-season."""
        items = [self._make_garment("coral")]
        result = scorer.analyze_outfit(
            items, ColorSeason.SPRING, season_sub="Nonexistent Sub"
        )
        assert 0 <= result.score <= 1

    def test_sub_clash_colour_penalised(self, scorer):
        """A colour in the sub-clash set should yield a lower score."""
        # "black" is in Light Spring clashes
        items = [self._make_garment("black")]
        result = scorer.analyze_outfit(
            items, ColorSeason.SPRING, season_sub="Light Spring"
        )
        # "peach" is in Light Spring palette
        items2 = [self._make_garment("peach")]
        result2 = scorer.analyze_outfit(
            items2, ColorSeason.SPRING, season_sub="Light Spring"
        )
        assert result.score < result2.score
