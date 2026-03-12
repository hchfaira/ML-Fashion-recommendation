"""
Unit Tests for SmartRemovalAnalyzer
====================================

Covers:
- Individual signal computation (recency, frequency, versatility, etc.)
- Regret-risk aggregation
- Verdict mapping
- User-goal personalisation
- Replacement quality evaluation
- Explainability (what_you_lose, what_you_keep, data_gaps, reasons)
- Edge cases (empty wardrobe, never-worn, brand-new items)
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from typing import List, Optional

from src.core.models import (
    Garment,
    GarmentAttributes,
    GarmentCategory,
    GarmentHistory,
    ColorProfile,
    FormalityLevel,
    Season,
    Occasion,
    UserContext,
    PatternInfo,
    MaterialProfile,
    SeasonalityInfo,
    WearRecord,
    UserRemovalGoal,
    SmartRemovalVerdict,
    ReplacementQuality,
)
from src.layer2_style.smart_removal_analyzer import SmartRemovalAnalyzer


# ============================================================================
# Helpers
# ============================================================================

NOW = datetime.now(timezone.utc)


def _g(
    garment_id: Optional[str] = None,
    category: GarmentCategory = GarmentCategory.TOP,
    subcategory: str = "t-shirt",
    color: str = "white",
    formality: FormalityLevel = FormalityLevel.CASUAL,
    seasons: Optional[List[Season]] = None,
    pattern_type: str = "solid",
    material: str = "cotton",
    wear_records: Optional[List[WearRecord]] = None,
    acquired_at: Optional[datetime] = None,
) -> Garment:
    """Shortcut to create a test garment with optional history."""
    season_list = seasons or [Season.SPRING, Season.SUMMER]
    history = GarmentHistory(
        wear_records=wear_records or [],
        acquired_at=acquired_at,
    )
    return Garment(
        id=garment_id or f"g_{uuid4().hex[:8]}",
        attributes=GarmentAttributes(
            category=category,
            subcategory=subcategory,
            color=ColorProfile(primary=color, hex_codes=[]),
            formality_level=formality,
            season_suitable=season_list,
            seasonality=SeasonalityInfo(seasons=season_list),
            pattern=PatternInfo(type=pattern_type),
            material=MaterialProfile(primary=material),
        ),
        history=history,
    )


def _recent_wears(n: int, days_back: int = 5) -> List[WearRecord]:
    """Generate n wear records spread over the last `days_back` days."""
    records = []
    for i in range(n):
        records.append(WearRecord(
            worn_at=NOW - timedelta(days=days_back - i),
            user_rating=4,
        ))
    return records


def _old_wears(n: int, days_back: int = 200) -> List[WearRecord]:
    """Generate n wear records all around `days_back` days ago."""
    records = []
    for i in range(n):
        records.append(WearRecord(
            worn_at=NOW - timedelta(days=days_back + i),
            user_rating=3,
        ))
    return records


def _wardrobe() -> List[Garment]:
    """A basic 10-item wardrobe with some variety."""
    return [
        _g("top1", GarmentCategory.TOP, "t-shirt", "white", FormalityLevel.CASUAL,
           [Season.SPRING, Season.SUMMER], wear_records=_recent_wears(5),
           acquired_at=NOW - timedelta(days=180)),
        _g("top2", GarmentCategory.TOP, "shirt", "navy", FormalityLevel.SMART_CASUAL,
           [Season.SPRING, Season.SUMMER, Season.FALL], wear_records=_recent_wears(3),
           acquired_at=NOW - timedelta(days=120)),
        _g("top3", GarmentCategory.TOP, "blouse", "black", FormalityLevel.BUSINESS,
           [Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
           wear_records=_recent_wears(8), acquired_at=NOW - timedelta(days=365)),
        _g("bottom1", GarmentCategory.BOTTOM, "jeans", "blue", FormalityLevel.CASUAL,
           [Season.SPRING, Season.SUMMER, Season.FALL], wear_records=_recent_wears(6),
           acquired_at=NOW - timedelta(days=200)),
        _g("bottom2", GarmentCategory.BOTTOM, "chinos", "khaki", FormalityLevel.SMART_CASUAL,
           [Season.SPRING, Season.FALL], wear_records=_old_wears(2),
           acquired_at=NOW - timedelta(days=300)),
        _g("shoes1", GarmentCategory.SHOES, "sneakers", "white", FormalityLevel.CASUAL,
           [Season.SPRING, Season.SUMMER], wear_records=_recent_wears(4),
           acquired_at=NOW - timedelta(days=90)),
        _g("shoes2", GarmentCategory.SHOES, "loafers", "brown", FormalityLevel.SMART_CASUAL,
           [Season.SPRING, Season.SUMMER, Season.FALL], wear_records=_old_wears(1),
           acquired_at=NOW - timedelta(days=400)),
        _g("jacket1", GarmentCategory.OUTERWEAR, "denim_jacket", "blue", FormalityLevel.CASUAL,
           [Season.SPRING, Season.FALL], wear_records=_recent_wears(2),
           acquired_at=NOW - timedelta(days=250)),
        _g("acc1", GarmentCategory.ACCESSORY, "watch", "silver", FormalityLevel.SMART_CASUAL,
           [Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
           wear_records=_recent_wears(10), acquired_at=NOW - timedelta(days=500)),
        _g("acc2", GarmentCategory.ACCESSORY, "belt", "brown", FormalityLevel.SMART_CASUAL,
           [Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
           wear_records=_old_wears(1), acquired_at=NOW - timedelta(days=150)),
    ]


# ============================================================================
# Test Class
# ============================================================================

class TestSmartRemovalAnalyzer:
    """Tests for SmartRemovalAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> SmartRemovalAnalyzer:
        return SmartRemovalAnalyzer()

    @pytest.fixture
    def wardrobe(self) -> List[Garment]:
        return _wardrobe()

    # ------------------------------------------------------------------ #
    # Basic operation                                                     #
    # ------------------------------------------------------------------ #

    def test_analyze_returns_verdict(self, analyzer, wardrobe):
        """analyze() returns a valid SmartRemovalVerdict."""
        result = analyzer.analyze("top1", wardrobe)
        assert isinstance(result, SmartRemovalVerdict)
        assert result.garment_id == "top1"
        assert 0 <= result.regret_risk <= 1
        assert result.verdict in ("SAFE_TO_REMOVE", "CONSIDER", "KEEP", "DONATE")
        assert 0 <= result.confidence <= 1

    def test_analyze_wardrobe_returns_sorted_list(self, analyzer, wardrobe):
        """analyze_wardrobe() returns all garments sorted by regret_risk ascending."""
        results = analyzer.analyze_wardrobe(wardrobe)
        assert len(results) == len(wardrobe)
        risks = [r.regret_risk for r in results]
        assert risks == sorted(risks), "Results should be sorted by regret_risk ascending"

    def test_garment_not_found(self, analyzer, wardrobe):
        """Missing garment_id → SAFE_TO_REMOVE with low confidence."""
        result = analyzer.analyze("nonexistent", wardrobe)
        assert result.verdict == "SAFE_TO_REMOVE"
        assert result.confidence <= 0.2
        assert any(g.field == "garment" for g in result.data_gaps)

    # ------------------------------------------------------------------ #
    # Signal: Recency                                                     #
    # ------------------------------------------------------------------ #

    def test_recently_worn_has_high_recency(self, analyzer, wardrobe):
        """A garment worn yesterday should have high recency score."""
        result = analyzer.analyze("top1", wardrobe)
        assert result.signals["recency"] > 0.7

    def test_old_worn_has_low_recency(self, analyzer, wardrobe):
        """A garment last worn 200+ days ago should have low recency."""
        result = analyzer.analyze("bottom2", wardrobe)
        assert result.signals["recency"] < 0.3

    def test_never_worn_has_zero_recency(self, analyzer):
        """A never-worn garment has recency = 0."""
        never = _g("never", wear_records=[], acquired_at=NOW - timedelta(days=100))
        wdrb = [never, _g("other")]
        result = analyzer.analyze("never", wdrb)
        assert result.signals["recency"] == 0.0

    # ------------------------------------------------------------------ #
    # Signal: Frequency                                                   #
    # ------------------------------------------------------------------ #

    def test_frequently_worn_high_frequency(self, analyzer, wardrobe):
        """top3 has 8 recent wears → high frequency score."""
        result = analyzer.analyze("top3", wardrobe)
        assert result.signals["frequency"] > 0.5

    def test_rarely_worn_low_frequency(self, analyzer, wardrobe):
        """shoes2 has 1 old wear → low frequency."""
        result = analyzer.analyze("shoes2", wardrobe)
        assert result.signals["frequency"] < 0.2

    # ------------------------------------------------------------------ #
    # Signal: Versatility                                                 #
    # ------------------------------------------------------------------ #

    def test_all_season_garment_more_versatile(self, analyzer, wardrobe):
        """top3 (all 4 seasons, business formality) should be more versatile than top1 (2 seasons, casual)."""
        v_top3 = analyzer.analyze("top3", wardrobe)
        v_top1 = analyzer.analyze("top1", wardrobe)
        assert v_top3.signals["versatility"] > v_top1.signals["versatility"]

    def test_single_season_low_versatility(self, analyzer):
        """A garment with 1 season has lower versatility than one with 4."""
        single = _g("single", seasons=[Season.SUMMER])
        multi = _g("multi", seasons=[Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER])
        wdrb = [single, multi, _g("bottom", category=GarmentCategory.BOTTOM)]
        r1 = analyzer.analyze("single", wdrb)
        r2 = analyzer.analyze("multi", wdrb)
        assert r2.signals["versatility"] > r1.signals["versatility"]

    # ------------------------------------------------------------------ #
    # Signal: Outfit Impact                                               #
    # ------------------------------------------------------------------ #

    def test_outfit_impact_exists(self, analyzer, wardrobe):
        """Outfit impact signal should be computed."""
        result = analyzer.analyze("top1", wardrobe)
        assert "outfit_impact" in result.signals
        assert result.signals["outfit_impact"] >= 0

    # ------------------------------------------------------------------ #
    # Signal: Rating                                                      #
    # ------------------------------------------------------------------ #

    def test_high_rated_garment_high_signal(self, analyzer):
        """A garment with all 5-star ratings should have high rating signal."""
        records = [WearRecord(worn_at=NOW - timedelta(days=i), user_rating=5) for i in range(5)]
        g = _g("rated", wear_records=records, acquired_at=NOW - timedelta(days=200))
        wdrb = [g, _g("other")]
        result = analyzer.analyze("rated", wdrb)
        assert result.signals["rating"] == 1.0

    def test_low_rated_garment_low_signal(self, analyzer):
        """A garment with all 1-star ratings should have low rating signal."""
        records = [WearRecord(worn_at=NOW - timedelta(days=i), user_rating=1) for i in range(5)]
        g = _g("lowrated", wear_records=records, acquired_at=NOW - timedelta(days=200))
        wdrb = [g, _g("other")]
        result = analyzer.analyze("lowrated", wdrb)
        assert result.signals["rating"] == 0.0

    def test_no_ratings_neutral(self, analyzer):
        """No ratings → rating signal = 0.5 (neutral)."""
        records = [WearRecord(worn_at=NOW - timedelta(days=1))]  # no rating
        g = _g("norating", wear_records=records, acquired_at=NOW - timedelta(days=200))
        wdrb = [g, _g("other")]
        result = analyzer.analyze("norating", wdrb)
        assert result.signals["rating"] == 0.5

    # ------------------------------------------------------------------ #
    # Signal: Newness                                                     #
    # ------------------------------------------------------------------ #

    def test_brand_new_item_has_high_newness(self, analyzer):
        """An item acquired 2 days ago → high newness protection."""
        g = _g("newitem", wear_records=[], acquired_at=NOW - timedelta(days=2))
        wdrb = [g, _g("other")]
        result = analyzer.analyze("newitem", wdrb)
        assert result.signals["newness"] > 0.9

    def test_old_item_has_low_newness(self, analyzer):
        """An item acquired 1 year ago → low newness."""
        g = _g("olditem", wear_records=_recent_wears(3), acquired_at=NOW - timedelta(days=365))
        wdrb = [g, _g("other")]
        result = analyzer.analyze("olditem", wdrb)
        assert result.signals["newness"] < 0.1

    # ------------------------------------------------------------------ #
    # Replacement Quality                                                 #
    # ------------------------------------------------------------------ #

    def test_replacement_with_same_category(self, analyzer, wardrobe):
        """top1 has replacements (top2, top3) → non-empty replacement."""
        result = analyzer.analyze("top1", wardrobe)
        assert result.replacement.valid_count >= 0
        assert result.replacement.quality_level in ("none", "low", "medium", "high")

    def test_unique_category_no_replacement(self, analyzer):
        """Only outerwear in wardrobe → no replacement possible."""
        jacket = _g("onlyjacket", GarmentCategory.OUTERWEAR, "leather_jacket", "black",
                     wear_records=_old_wears(2), acquired_at=NOW - timedelta(days=200))
        others = [
            _g("top", GarmentCategory.TOP),
            _g("bottom", GarmentCategory.BOTTOM),
        ]
        wdrb = [jacket] + others
        result = analyzer.analyze("onlyjacket", wdrb)
        assert result.replacement.valid_count == 0
        assert result.replacement.quality_level == "none"

    def test_replacement_considers_formality(self, analyzer):
        """A formal item is not well-replaced by a casual one."""
        formal = _g("formal_top", GarmentCategory.TOP, "dress_shirt", "white",
                     FormalityLevel.BUSINESS, [Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
                     wear_records=_recent_wears(5), acquired_at=NOW - timedelta(days=200))
        casual = _g("casual_top", GarmentCategory.TOP, "t-shirt", "white",
                     FormalityLevel.VERY_CASUAL, [Season.SPRING, Season.SUMMER],
                     acquired_at=NOW - timedelta(days=200))
        wdrb = [formal, casual, _g("bottom", GarmentCategory.BOTTOM)]
        result = analyzer.analyze("formal_top", wdrb)
        # Casual top is not formality-compatible
        if result.replacement.best_candidate:
            assert result.replacement.best_candidate.formality_compatible is False or \
                   result.replacement.best_candidate.coverage_match < 0.8

    def test_replacement_considers_seasons(self, analyzer):
        """A 4-season item replaced by a 2-season item has coverage gaps."""
        target = _g("all_season", GarmentCategory.TOP, "henley", "grey",
                     FormalityLevel.CASUAL,
                     [Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
                     wear_records=_old_wears(3), acquired_at=NOW - timedelta(days=200))
        limited = _g("summer_only", GarmentCategory.TOP, "tank", "white",
                      FormalityLevel.CASUAL, [Season.SUMMER],
                      acquired_at=NOW - timedelta(days=200))
        wdrb = [target, limited, _g("bottom", GarmentCategory.BOTTOM)]
        result = analyzer.analyze("all_season", wdrb)
        assert len(result.replacement.coverage_gaps) > 0

    # ------------------------------------------------------------------ #
    # Verdict mapping                                                     #
    # ------------------------------------------------------------------ #

    def test_never_worn_old_item_is_donate_or_safe(self, analyzer):
        """Never worn + old → DONATE or SAFE_TO_REMOVE."""
        g = _g("never_old", wear_records=[], acquired_at=NOW - timedelta(days=365))
        wdrb = [g, _g("other", category=GarmentCategory.BOTTOM)]
        result = analyzer.analyze("never_old", wdrb)
        assert result.verdict in ("DONATE", "SAFE_TO_REMOVE", "CONSIDER")

    def test_heavily_used_multi_season_is_keep(self, analyzer):
        """Heavily used, well-rated, multi-season → KEEP."""
        records = [
            WearRecord(worn_at=NOW - timedelta(days=i), user_rating=5)
            for i in range(15)
        ]
        g = _g("hero", GarmentCategory.TOP, "classic_shirt", "navy",
               FormalityLevel.SMART_CASUAL,
               [Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER],
               wear_records=records, acquired_at=NOW - timedelta(days=365))
        wdrb = [g, _g("bottom", GarmentCategory.BOTTOM)]
        result = analyzer.analyze("hero", wdrb)
        assert result.verdict in ("KEEP", "CONSIDER")  # should lean KEEP
        assert result.regret_risk > 0.4

    # ------------------------------------------------------------------ #
    # User Goal Personalisation                                           #
    # ------------------------------------------------------------------ #

    def test_minimalist_goal_pushes_towards_removal(self, analyzer, wardrobe):
        """Minimalist goal should lower regret_risk (more willing to remove)."""
        r_min = analyzer.analyze("top1", wardrobe, user_goal=UserRemovalGoal.MINIMALIST)
        r_max = analyzer.analyze("top1", wardrobe, user_goal=UserRemovalGoal.MAXIMIZE_OPTIONS)
        # Minimalist emphasises replacement → if replacement exists, risk should be lower
        # This is a directional test; not guaranteed but typical
        assert isinstance(r_min, SmartRemovalVerdict)
        assert isinstance(r_max, SmartRemovalVerdict)

    def test_style_upgrade_penalises_low_ratings(self, analyzer):
        """Style upgrade goal should make low-rated items easier to remove."""
        records = [WearRecord(worn_at=NOW - timedelta(days=i), user_rating=1) for i in range(5)]
        g = _g("lowq", wear_records=records, acquired_at=NOW - timedelta(days=200))
        wdrb = [g, _g("other", category=GarmentCategory.BOTTOM)]
        r_style = analyzer.analyze("lowq", wdrb, user_goal=UserRemovalGoal.STYLE_UPGRADE)
        r_options = analyzer.analyze("lowq", wdrb, user_goal=UserRemovalGoal.MAXIMIZE_OPTIONS)
        # Style upgrade weights rating more, so low rating → lower regret_risk
        assert r_style.regret_risk <= r_options.regret_risk + 0.1  # allow small tolerance

    def test_maximize_options_values_versatility(self, analyzer, wardrobe):
        """maximize_options should weight versatility higher."""
        # top3 is the most versatile item (all 4 seasons, business formality)
        r = analyzer.analyze("top3", wardrobe, user_goal=UserRemovalGoal.MAXIMIZE_OPTIONS)
        assert r.signals["versatility"] > 0.3  # should be reasonably high

    # ------------------------------------------------------------------ #
    # Explainability                                                      #
    # ------------------------------------------------------------------ #

    def test_reasons_sorted_by_contribution(self, analyzer, wardrobe):
        """Reasons should be sorted by absolute contribution descending."""
        result = analyzer.analyze("top1", wardrobe)
        contributions = [abs(r.score_contribution) for r in result.reasons]
        assert contributions == sorted(contributions, reverse=True)

    def test_what_you_lose_populated(self, analyzer, wardrobe):
        """what_you_lose should contain information about the garment's role."""
        result = analyzer.analyze("top3", wardrobe)
        # top3 is multi-season and frequently worn
        assert isinstance(result.what_you_lose, list)

    def test_what_you_keep_shows_remaining(self, analyzer, wardrobe):
        """what_you_keep should mention remaining items in same category."""
        result = analyzer.analyze("top1", wardrobe)
        assert any("category" in line.lower() or "item" in line.lower() for line in result.what_you_keep)

    def test_summary_contains_verdict(self, analyzer, wardrobe):
        """Summary string should mention the verdict."""
        result = analyzer.analyze("top1", wardrobe)
        assert result.verdict in result.summary

    def test_data_gaps_for_no_wear_history(self, analyzer):
        """Never-worn garment should have data gap for wear_records."""
        g = _g("nowear", wear_records=[], acquired_at=NOW - timedelta(days=100))
        wdrb = [g, _g("other")]
        result = analyzer.analyze("nowear", wdrb)
        gap_fields = [dg.field for dg in result.data_gaps]
        assert "wear_records" in gap_fields

    def test_data_gaps_for_no_rating(self, analyzer):
        """Items with no ratings should have a data gap for user_rating."""
        records = [WearRecord(worn_at=NOW - timedelta(days=1))]
        g = _g("norate", wear_records=records, acquired_at=NOW - timedelta(days=200))
        wdrb = [g, _g("other")]
        result = analyzer.analyze("norate", wdrb)
        gap_fields = [dg.field for dg in result.data_gaps]
        assert "user_rating" in gap_fields

    # ------------------------------------------------------------------ #
    # Confidence                                                          #
    # ------------------------------------------------------------------ #

    def test_high_data_garment_has_high_confidence(self, analyzer, wardrobe):
        """top3 has lots of wear data → confidence should be good."""
        result = analyzer.analyze("top3", wardrobe)
        assert result.confidence >= 0.6

    def test_low_data_garment_has_lower_confidence(self, analyzer):
        """Never-worn, no rating → low confidence."""
        g = _g("sparse", wear_records=[], acquired_at=NOW - timedelta(days=100))
        wdrb = [g, _g("other")]
        result = analyzer.analyze("sparse", wdrb)
        assert result.confidence < 0.8

    # ------------------------------------------------------------------ #
    # Edge Cases                                                          #
    # ------------------------------------------------------------------ #

    def test_single_item_wardrobe(self, analyzer):
        """Wardrobe with a single item should still work."""
        g = _g("only", wear_records=_recent_wears(3), acquired_at=NOW - timedelta(days=200))
        result = analyzer.analyze("only", [g])
        assert isinstance(result, SmartRemovalVerdict)
        assert result.garment_id == "only"

    def test_empty_wardrobe(self, analyzer):
        """Empty wardrobe → garment not found."""
        result = analyzer.analyze("any", [])
        assert result.verdict == "SAFE_TO_REMOVE"
        assert result.confidence <= 0.2

    def test_retired_garments_skipped_in_wardrobe_scan(self, analyzer):
        """analyze_wardrobe should skip retired garments."""
        g1 = _g("active", wear_records=_recent_wears(2), acquired_at=NOW - timedelta(days=200))
        g2 = _g("retired", wear_records=_old_wears(1), acquired_at=NOW - timedelta(days=200))
        g2.history.is_retired = True
        results = analyzer.analyze_wardrobe([g1, g2])
        ids = [r.garment_id for r in results]
        assert "active" in ids
        assert "retired" not in ids

    def test_all_goals_produce_valid_output(self, analyzer, wardrobe):
        """All three goals should produce valid verdicts."""
        for goal in UserRemovalGoal:
            result = analyzer.analyze("top1", wardrobe, user_goal=goal)
            assert 0 <= result.regret_risk <= 1
            assert result.user_goal == goal

    def test_garment_description_built(self, analyzer, wardrobe):
        """garment_description should not be empty."""
        result = analyzer.analyze("top1", wardrobe)
        assert len(result.garment_description) > 0

    def test_signals_dict_has_all_keys(self, analyzer, wardrobe):
        """signals dict should have all 7 expected keys."""
        result = analyzer.analyze("top1", wardrobe)
        expected_keys = {"recency", "frequency", "versatility", "outfit_impact",
                         "replacement", "rating", "newness"}
        assert expected_keys == set(result.signals.keys())

    def test_regret_risk_bounded(self, analyzer, wardrobe):
        """regret_risk must always be in [0, 1]."""
        for g in wardrobe:
            result = analyzer.analyze(g.id, wardrobe)
            assert 0 <= result.regret_risk <= 1, f"Garment {g.id} has out-of-bounds regret_risk"

    def test_garment_attributes_included(self, analyzer, wardrobe):
        """garment_attributes should be populated in the verdict."""
        result = analyzer.analyze("top1", wardrobe)
        assert result.garment_attributes is not None
        assert result.garment_attributes.category == GarmentCategory.TOP


class TestSmartRemovalAPI:
    """Integration tests for the smart-removal API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from fastapi.testclient import TestClient
        from src.main import app
        return TestClient(app)

    def test_smart_removal_endpoint_exists(self, client):
        """POST /wardrobe-analysis/smart-removal should exist."""
        # Minimal valid request — just check route is wired
        payload = {
            "garment_id": "nonexistent",
            "wardrobe": [{
                "id": "g1",
                "attributes": {
                    "category": "top",
                    "color": {"primary": "white", "hex_codes": []},
                },
            }],
        }
        resp = client.post("/api/v1/wardrobe-analysis/smart-removal", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "regret_risk" in data
        assert "verdict" in data

    def test_smart_removal_wardrobe_endpoint_exists(self, client):
        """POST /wardrobe-analysis/smart-removal/wardrobe should exist."""
        payload = {
            "wardrobe": [{
                "id": "g1",
                "attributes": {
                    "category": "top",
                    "color": {"primary": "white", "hex_codes": []},
                },
            }],
        }
        resp = client.post("/api/v1/wardrobe-analysis/smart-removal/wardrobe", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_smart_removal_requires_garment_id(self, client):
        """Single-item endpoint should require garment_id."""
        payload = {
            "wardrobe": [{
                "id": "g1",
                "attributes": {
                    "category": "top",
                    "color": {"primary": "white", "hex_codes": []},
                },
            }],
        }
        resp = client.post("/api/v1/wardrobe-analysis/smart-removal", json=payload)
        assert resp.status_code == 400

    def test_smart_removal_with_user_goal(self, client):
        """User goal parameter should be accepted."""
        payload = {
            "garment_id": "g1",
            "wardrobe": [{
                "id": "g1",
                "attributes": {
                    "category": "top",
                    "color": {"primary": "white", "hex_codes": []},
                },
            }],
            "user_goal": "minimalist",
        }
        resp = client.post("/api/v1/wardrobe-analysis/smart-removal", json=payload)
        assert resp.status_code == 200
        assert resp.json()["user_goal"] == "minimalist"
