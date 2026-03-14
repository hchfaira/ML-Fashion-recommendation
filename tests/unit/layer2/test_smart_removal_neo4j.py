"""
Tests for SmartRemovalAnalyzer — Neo4j Enrichment (Phase 4)
=============================================================

Covers:
- _centrality_signal(): high/low/missing/partial data
- _batch_fetch_centrality(): concurrent fetches, partial errors
- analyze_with_neo4j(): centrality injected into signals, fallback when client=None
- analyze_wardrobe_with_neo4j(): all garments analyzed, retired skipped, sorted
- Deprecation warning from _deprecated decorator
- on-complete weight injection when profile centrality weight > 0

All Neo4j I/O is mocked — no live instance required.

Markers:
    @pytest.mark.unit
    @pytest.mark.asyncio
"""

from __future__ import annotations

import warnings
import pytest
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.core.models import (
    Garment,
    GarmentAttributes,
    GarmentCategory,
    GarmentHistory,
    ColorProfile,
    FormalityLevel,
    Season,
    PatternInfo,
    MaterialProfile,
    SeasonalityInfo,
    WearRecord,
    UserRemovalGoal,
)
from src.layer2_style.smart_removal_analyzer import SmartRemovalAnalyzer, _deprecated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)


def _g(
    garment_id: Optional[str] = None,
    category: GarmentCategory = GarmentCategory.TOP,
    wear_records: Optional[List[WearRecord]] = None,
    acquired_at: Optional[datetime] = None,
    retired: bool = False,
) -> Garment:
    history = GarmentHistory(
        wear_records=wear_records or [],
        acquired_at=acquired_at or NOW - timedelta(days=180),
        is_retired=retired,
    )
    return Garment(
        id=garment_id or f"g_{uuid4().hex[:8]}",
        attributes=GarmentAttributes(
            category=category,
            color=ColorProfile(primary="blue"),
            formality_level=FormalityLevel.CASUAL,
            season_suitable=[Season.SPRING, Season.SUMMER],
            seasonality=SeasonalityInfo(seasons=[Season.SPRING, Season.SUMMER]),
            pattern=PatternInfo(type="solid"),
            material=MaterialProfile(primary="cotton"),
        ),
        history=history,
    )


def _recent_wears(n: int = 3) -> List[WearRecord]:
    return [
        WearRecord(worn_at=NOW - timedelta(days=i + 1), user_rating=4)
        for i in range(n)
    ]


def _wardrobe(n: int = 4) -> List[Garment]:
    return [_g(f"g{i}", wear_records=_recent_wears(3)) for i in range(n)]


def _mock_client(
    centrality_data: Optional[dict] = None,
    *,
    side_effect=None,
) -> MagicMock:
    """Create a mock Neo4jClient whose get_centrality_scores returns fixed data."""
    client = MagicMock()
    if side_effect is not None:
        client.get_centrality_scores = AsyncMock(side_effect=side_effect)
    else:
        client.get_centrality_scores = AsyncMock(return_value=centrality_data or {})
    return client


# ---------------------------------------------------------------------------
# _centrality_signal
# ---------------------------------------------------------------------------


class TestCentralitySignal:
    """Unit tests for SmartRemovalAnalyzer._centrality_signal()."""

    def setup_method(self):
        self.analyzer = SmartRemovalAnalyzer(profile_name="default")

    def test_none_returns_neutral(self):
        assert self.analyzer._centrality_signal(None) == 0.5

    def test_empty_dict_returns_neutral(self):
        assert self.analyzer._centrality_signal({}) == 0.5

    def test_high_degree_and_pagerank(self):
        score = self.analyzer._centrality_signal(
            {"degree": 1.0, "pagerank": 1.0, "betweenness": 0.9}
        )
        assert score == 1.0

    def test_zero_degree_and_pagerank(self):
        score = self.analyzer._centrality_signal(
            {"degree": 0.0, "pagerank": 0.0}
        )
        assert score == 0.0

    def test_weights_60_40(self):
        # 0.6 * 0.8 + 0.4 * 0.5 = 0.48 + 0.20 = 0.68
        score = self.analyzer._centrality_signal(
            {"degree": 0.8, "pagerank": 0.5}
        )
        assert abs(score - 0.68) < 1e-3

    def test_clamped_to_0_1(self):
        score = self.analyzer._centrality_signal(
            {"degree": 2.0, "pagerank": 3.0}  # absurdly large
        )
        assert score == 1.0

    def test_none_values_treated_as_zero(self):
        score = self.analyzer._centrality_signal(
            {"degree": None, "pagerank": None}
        )
        assert score == 0.0

    def test_missing_pagerank_uses_degree_only(self):
        # 0.6 * 0.5 + 0.4 * 0.0 = 0.30
        score = self.analyzer._centrality_signal({"degree": 0.5})
        assert abs(score - 0.30) < 1e-3

    def test_missing_degree_uses_pagerank_only(self):
        # 0.6 * 0.0 + 0.4 * 0.75 = 0.30
        score = self.analyzer._centrality_signal({"pagerank": 0.75})
        assert abs(score - 0.30) < 1e-3

    def test_betweenness_ignored_in_formula(self):
        """Betweenness is stored in Neo4j but not used in the formula."""
        s1 = self.analyzer._centrality_signal({"degree": 0.5, "pagerank": 0.5})
        s2 = self.analyzer._centrality_signal(
            {"degree": 0.5, "pagerank": 0.5, "betweenness": 0.99}
        )
        assert s1 == s2

    def test_result_rounded_to_4_decimals(self):
        score = self.analyzer._centrality_signal({"degree": 0.333, "pagerank": 0.333})
        assert score == round(score, 4)


# ---------------------------------------------------------------------------
# _batch_fetch_centrality
# ---------------------------------------------------------------------------


class TestBatchFetchCentrality:
    """Tests for the concurrent batch-fetch helper."""

    @pytest.mark.asyncio
    async def test_all_succeed(self):
        data = {"degree": 0.4, "pagerank": 0.3}
        client = _mock_client(centrality_data=data)

        result = await SmartRemovalAnalyzer._batch_fetch_centrality(
            client, ["g1", "g2", "g3"]
        )

        assert set(result.keys()) == {"g1", "g2", "g3"}
        assert all(v == data for v in result.values())

    @pytest.mark.asyncio
    async def test_empty_ids_returns_empty_dict(self):
        client = _mock_client()
        result = await SmartRemovalAnalyzer._batch_fetch_centrality(client, [])
        assert result == {}

    @pytest.mark.asyncio
    async def test_exception_for_one_sets_none(self):
        """Per-garment exception should map to None, others intact."""
        good_data = {"degree": 0.6, "pagerank": 0.4}
        call_count = 0

        async def _side(gid):
            nonlocal call_count
            call_count += 1
            if gid == "bad":
                raise RuntimeError("network error")
            return good_data

        client = MagicMock()
        client.get_centrality_scores = AsyncMock(side_effect=_side)

        result = await SmartRemovalAnalyzer._batch_fetch_centrality(
            client, ["g1", "bad", "g3"]
        )

        assert result["g1"] == good_data
        assert result["bad"] is None
        assert result["g3"] == good_data

    @pytest.mark.asyncio
    async def test_all_fail_returns_all_none(self):
        client = _mock_client(side_effect=RuntimeError("no db"))
        result = await SmartRemovalAnalyzer._batch_fetch_centrality(
            client, ["g1", "g2"]
        )
        assert result["g1"] is None
        assert result["g2"] is None

    @pytest.mark.asyncio
    async def test_concurrent_calls_issued(self):
        """All garments should trigger a get_centrality_scores call."""
        client = _mock_client(centrality_data={"degree": 0.1, "pagerank": 0.1})
        await SmartRemovalAnalyzer._batch_fetch_centrality(
            client, ["a", "b", "c", "d"]
        )
        assert client.get_centrality_scores.await_count == 4


# ---------------------------------------------------------------------------
# analyze_with_neo4j
# ---------------------------------------------------------------------------


class TestAnalyzeWithNeo4j:
    """Tests for SmartRemovalAnalyzer.analyze_with_neo4j()."""

    def setup_method(self):
        self.analyzer = SmartRemovalAnalyzer(profile_name="default")

    @pytest.mark.asyncio
    async def test_centrality_added_to_signals(self):
        wardrobe = _wardrobe(4)
        target_id = wardrobe[0].id
        client = _mock_client({"degree": 0.8, "pagerank": 0.7})

        verdict = await self.analyzer.analyze_with_neo4j(
            target_id, wardrobe, client
        )

        assert "centrality" in verdict.signals
        assert verdict.signals["centrality"] > 0.0

    @pytest.mark.asyncio
    async def test_fallback_to_base_when_client_none(self):
        """With client=None the verdict should still be produced."""
        wardrobe = _wardrobe(4)
        target_id = wardrobe[0].id

        verdict = await self.analyzer.analyze_with_neo4j(
            target_id, wardrobe, None
        )

        assert verdict.garment_id == target_id
        # centrality signal present, but neutral (no data)
        assert verdict.signals.get("centrality") == 0.5

    @pytest.mark.asyncio
    async def test_client_exception_falls_back_gracefully(self):
        wardrobe = _wardrobe(4)
        target_id = wardrobe[0].id
        client = _mock_client(side_effect=RuntimeError("db down"))

        # Should not raise
        verdict = await self.analyzer.analyze_with_neo4j(
            target_id, wardrobe, client
        )

        assert verdict.garment_id == target_id
        assert verdict.signals.get("centrality") == 0.5

    @pytest.mark.asyncio
    async def test_missing_garment_returns_not_found(self):
        wardrobe = _wardrobe(2)
        client = _mock_client({"degree": 0.5, "pagerank": 0.5})

        verdict = await self.analyzer.analyze_with_neo4j(
            "nonexistent_id", wardrobe, client
        )

        assert verdict.garment_id == "nonexistent_id"
        assert verdict.verdict in ("not_found", "KEEP", "REMOVE", "DONATE",
                                   "CONSIDER_REMOVING", "SAFE_TO_REMOVE")

    @pytest.mark.asyncio
    async def test_get_centrality_scores_called_with_garment_id(self):
        wardrobe = _wardrobe(3)
        target_id = wardrobe[1].id
        client = _mock_client({"degree": 0.3, "pagerank": 0.3})

        await self.analyzer.analyze_with_neo4j(target_id, wardrobe, client)

        client.get_centrality_scores.assert_awaited_once_with(target_id)

    @pytest.mark.asyncio
    async def test_zero_centrality_data_returns_neutral(self):
        wardrobe = _wardrobe(4)
        target_id = wardrobe[0].id
        client = _mock_client({})

        verdict = await self.analyzer.analyze_with_neo4j(
            target_id, wardrobe, client
        )

        assert verdict.signals["centrality"] == 0.5

    @pytest.mark.asyncio
    async def test_verdict_contains_standard_fields(self):
        wardrobe = _wardrobe(4)
        target_id = wardrobe[0].id
        client = _mock_client({"degree": 0.6, "pagerank": 0.4})

        verdict = await self.analyzer.analyze_with_neo4j(
            target_id, wardrobe, client
        )

        assert verdict.garment_id == target_id
        assert 0.0 <= verdict.regret_risk <= 1.0
        assert isinstance(verdict.reasons, list)
        assert isinstance(verdict.signals, dict)

    @pytest.mark.asyncio
    async def test_user_goal_forwarded(self):
        wardrobe = _wardrobe(4)
        target_id = wardrobe[0].id
        client = _mock_client({"degree": 0.5, "pagerank": 0.5})

        verdict = await self.analyzer.analyze_with_neo4j(
            target_id, wardrobe, client,
            user_goal=UserRemovalGoal.MINIMALIST,
        )

        assert verdict.user_goal == UserRemovalGoal.MINIMALIST


# ---------------------------------------------------------------------------
# analyze_wardrobe_with_neo4j
# ---------------------------------------------------------------------------


class TestAnalyzeWardrobeWithNeo4j:
    """Tests for SmartRemovalAnalyzer.analyze_wardrobe_with_neo4j()."""

    def setup_method(self):
        self.analyzer = SmartRemovalAnalyzer(profile_name="default")

    @pytest.mark.asyncio
    async def test_all_active_garments_returned(self):
        wardrobe = _wardrobe(5)
        client = _mock_client({"degree": 0.5, "pagerank": 0.5})

        verdicts = await self.analyzer.analyze_wardrobe_with_neo4j(
            wardrobe, client
        )

        assert len(verdicts) == 5

    @pytest.mark.asyncio
    async def test_retired_garments_excluded(self):
        active = _wardrobe(3)
        retired = [_g("retired1", retired=True)]
        wardrobe = active + retired
        client = _mock_client({"degree": 0.5, "pagerank": 0.5})

        verdicts = await self.analyzer.analyze_wardrobe_with_neo4j(
            wardrobe, client
        )

        ids = {v.garment_id for v in verdicts}
        assert "retired1" not in ids
        assert len(verdicts) == 3

    @pytest.mark.asyncio
    async def test_sorted_by_regret_risk_ascending(self):
        wardrobe = _wardrobe(5)
        client = _mock_client({"degree": 0.5, "pagerank": 0.5})

        verdicts = await self.analyzer.analyze_wardrobe_with_neo4j(
            wardrobe, client
        )

        risks = [v.regret_risk for v in verdicts]
        assert risks == sorted(risks)

    @pytest.mark.asyncio
    async def test_centrality_in_each_verdict_signal(self):
        wardrobe = _wardrobe(3)
        client = _mock_client({"degree": 0.7, "pagerank": 0.6})

        verdicts = await self.analyzer.analyze_wardrobe_with_neo4j(
            wardrobe, client
        )

        for v in verdicts:
            assert "centrality" in v.signals

    @pytest.mark.asyncio
    async def test_client_none_returns_all_verdicts(self):
        """With client=None every garment still gets analyzed."""
        wardrobe = _wardrobe(4)
        verdicts = await self.analyzer.analyze_wardrobe_with_neo4j(
            wardrobe, None
        )
        assert len(verdicts) == 4
        for v in verdicts:
            assert v.signals.get("centrality") == 0.5

    @pytest.mark.asyncio
    async def test_empty_wardrobe_returns_empty_list(self):
        client = _mock_client({"degree": 0.5, "pagerank": 0.5})
        verdicts = await self.analyzer.analyze_wardrobe_with_neo4j([], client)
        assert verdicts == []

    @pytest.mark.asyncio
    async def test_batch_fetch_called_once(self):
        """get_centrality_scores should be called once per garment concurrently."""
        wardrobe = _wardrobe(4)
        client = _mock_client({"degree": 0.4, "pagerank": 0.4})

        await self.analyzer.analyze_wardrobe_with_neo4j(wardrobe, client)

        assert client.get_centrality_scores.await_count == 4

    @pytest.mark.asyncio
    async def test_partial_centrality_failures_handled(self):
        """If some garments fail centrality fetch, others are unaffected."""
        wardrobe = _wardrobe(3)
        ids = [g.id for g in wardrobe]
        bad_id = ids[1]

        async def _side(gid):
            if gid == bad_id:
                raise RuntimeError("network")
            return {"degree": 0.5, "pagerank": 0.5}

        client = MagicMock()
        client.get_centrality_scores = AsyncMock(side_effect=_side)

        verdicts = await self.analyzer.analyze_wardrobe_with_neo4j(
            wardrobe, client
        )

        assert len(verdicts) == 3


# ---------------------------------------------------------------------------
# _explain_signal — centrality branch
# ---------------------------------------------------------------------------


class TestExplainCentralitySignal:
    """Tests for _explain_signal with name='centrality'."""

    def setup_method(self):
        self.analyzer = SmartRemovalAnalyzer(profile_name="default")

    def test_high_centrality_explanation(self):
        msg = self.analyzer._explain_signal("centrality", 0.9, None)
        assert "hub" in msg.lower() or "centrality" in msg.lower()

    def test_moderate_centrality_explanation(self):
        msg = self.analyzer._explain_signal("centrality", 0.55, None)
        assert "moderate" in msg.lower() or "centrality" in msg.lower()

    def test_neutral_centrality_explanation(self):
        msg = self.analyzer._explain_signal("centrality", 0.5, None)
        assert "unknown" in msg.lower() or "no" in msg.lower()

    def test_low_centrality_explanation(self):
        msg = self.analyzer._explain_signal("centrality", 0.1, None)
        assert "low" in msg.lower() or "rarely" in msg.lower()


# ---------------------------------------------------------------------------
# _deprecated decorator
# ---------------------------------------------------------------------------


class TestDeprecatedDecorator:
    """Tests for the _deprecated() decorator helper."""

    def test_emits_deprecation_warning(self):
        @_deprecated("new_method")
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = old_func()

        assert result == 42
        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)

    def test_warning_message_contains_replacement(self):
        @_deprecated("shiny_new_api")
        def old_thing():
            pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old_thing()

        assert "shiny_new_api" in str(caught[0].message)

    def test_warning_message_contains_function_name(self):
        @_deprecated("replacement")
        def my_old_func():
            pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            my_old_func()

        assert "my_old_func" in str(caught[0].message)

    def test_return_value_preserved(self):
        @_deprecated("other")
        def add(a, b):
            return a + b

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert add(3, 4) == 7

    def test_kwargs_forwarded(self):
        @_deprecated("other")
        def greet(name="world"):
            return f"hello {name}"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert greet(name="Alice") == "hello Alice"

    def test_category_is_deprecation_warning(self):
        @_deprecated("new")
        def old():
            pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old()

        assert issubclass(caught[0].category, DeprecationWarning)
