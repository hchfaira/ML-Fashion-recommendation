"""Integration tests for PromptSearchService."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from src.api.services.prompt_search_service import PromptSearchService, _conversation_mgr
from src.layer4_llm.prompt_parser import ParsedPrompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def svc() -> PromptSearchService:
    return PromptSearchService()


def _outfit(
    colors=None,
    styles=None,
    occasion_tags=None,
    formality=0.5,
    oid="outfit-1",
) -> Dict[str, Any]:
    return {
        "id": oid,
        "dominant_colors": colors or [],
        "dominant_styles": {s: 0.8 for s in (styles or [])},
        "occasion_tags": occasion_tags or [],
        "formality_score": formality,
        "title": f"outfit-{oid}",
    }


# ---------------------------------------------------------------------------
# parse_only
# ---------------------------------------------------------------------------

class TestParseOnly:
    def test_returns_parsed_prompt(self, svc):
        async def _run():
            result = await svc.parse_only("blue dress for a wedding")
            assert isinstance(result, ParsedPrompt)
            assert result.raw_prompt == "blue dress for a wedding"

        asyncio.get_event_loop().run_until_complete(_run())

    def test_off_topic_flagged(self, svc):
        async def _run():
            result = await svc.parse_only("pizza recipe please")
            assert result.is_off_topic

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# get_suggestions
# ---------------------------------------------------------------------------

class TestGetSuggestions:
    def test_returns_non_empty_list(self, svc):
        suggestions = svc.get_suggestions()
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_all_strings(self, svc):
        assert all(isinstance(s, str) for s in svc.get_suggestions())


# ---------------------------------------------------------------------------
# _apply_hard_filters
# ---------------------------------------------------------------------------

class TestHardFilters:
    def test_excluded_type_removed(self, svc):
        from src.layer3_context.prompt_to_filters import WardrobeFilters
        f = WardrobeFilters(excluded_types=["skirt"])
        outfits = [
            {"garment_type": "skirt", "id": "bad"},
            {"garment_type": "dress", "id": "good"},
        ]
        result = svc._apply_hard_filters(outfits, f)
        ids = [o["id"] for o in result]
        assert "bad" not in ids
        assert "good" in ids

    def test_no_exclusions_returns_all(self, svc):
        from src.layer3_context.prompt_to_filters import WardrobeFilters
        f = WardrobeFilters()
        outfits = [{"id": "a"}, {"id": "b"}]
        assert len(svc._apply_hard_filters(outfits, f)) == 2


# ---------------------------------------------------------------------------
# _identify_missing_aspects
# ---------------------------------------------------------------------------

class TestIdentifyMissingAspects:
    def test_empty_outfits_reports_all_missing(self, svc):
        p = ParsedPrompt(raw_prompt="x", colors=["blue"], occasion="wedding")
        missing = svc._identify_missing_aspects(p, [])
        assert any("color" in m for m in missing)
        assert any("occasion" in m for m in missing)

    def test_matching_outfit_no_missing(self, svc):
        p = ParsedPrompt(raw_prompt="x", colors=["blue"], occasion="wedding")
        outfit = _outfit(colors=["blue"], occasion_tags=["wedding"])
        missing = svc._identify_missing_aspects(p, [outfit])
        assert missing == []

    def test_wrong_color_reported(self, svc):
        p = ParsedPrompt(raw_prompt="x", colors=["blue"])
        outfit = _outfit(colors=["red"])
        missing = svc._identify_missing_aspects(p, [outfit])
        assert any("color" in m for m in missing)


# ---------------------------------------------------------------------------
# search — off-topic
# ---------------------------------------------------------------------------

class TestSearchOffTopic:
    def test_off_topic_returns_empty_outfits(self, svc):
        async def _run():
            result = await svc.search(
                "user1",
                "bitcoin investment strategy",
                candidate_outfits=[_outfit(colors=["red"])],
                explain=False,
            )
            assert result.outfits == []
            assert result.parsed_prompt.is_off_topic

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# search — ranking
# ---------------------------------------------------------------------------

class TestSearchRanking:
    def test_matching_outfit_ranked_first(self, svc):
        async def _run():
            candidates = [
                _outfit(colors=["red"], formality=0.2, oid="red"),
                _outfit(colors=["blue"], formality=0.85, occasion_tags=["wedding"], oid="blue"),
            ]
            result = await svc.search(
                "user1",
                "blue dress for a wedding",
                candidate_outfits=candidates,
                explain=False,
            )
            ids = [o["id"] for o in result.outfits]
            assert ids[0] == "blue"

        asyncio.get_event_loop().run_until_complete(_run())

    def test_result_has_prompt_score(self, svc):
        async def _run():
            candidates = [_outfit(colors=["blue"], oid="x")]
            result = await svc.search(
                "user1", "blue outfit", candidate_outfits=candidates, explain=False
            )
            if result.outfits:
                assert "prompt_score" in result.outfits[0]
                assert "final_score" in result.outfits[0]

        asyncio.get_event_loop().run_until_complete(_run())

    def test_max_results_respected(self, svc):
        async def _run():
            candidates = [_outfit(colors=["blue"], oid=str(i)) for i in range(20)]
            result = await svc.search(
                "user1", "blue", candidate_outfits=candidates,
                max_results=5, explain=False,
            )
            assert len(result.outfits) <= 5

        asyncio.get_event_loop().run_until_complete(_run())

    def test_excluded_type_removed_from_results(self, svc):
        async def _run():
            candidates = [
                {**_outfit(colors=["blue"], oid="skirt"), "garment_type": "skirt"},
                {**_outfit(colors=["blue"], oid="dress"), "garment_type": "dress"},
            ]
            result = await svc.search(
                "user1", "blue outfit no skirts",
                candidate_outfits=candidates, explain=False,
            )
            ids = [o["id"] for o in result.outfits]
            assert "skirt" not in ids

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# search — explanation
# ---------------------------------------------------------------------------

class TestSearchExplanation:
    def test_explanation_returned_as_string(self, svc):
        async def _run():
            candidates = [_outfit(colors=["blue"], formality=0.85, occasion_tags=["wedding"])]
            with patch(
                "src.api.services.prompt_search_service._explainer.explain_result",
                new_callable=AsyncMock,
                return_value="Great choice!",
            ):
                result = await svc.search(
                    "user1", "blue wedding", candidate_outfits=candidates, explain=True
                )
            assert result.explanation == "Great choice!"

        asyncio.get_event_loop().run_until_complete(_run())

    def test_no_outfits_triggers_missing_suggestion(self, svc):
        async def _run():
            with patch(
                "src.api.services.prompt_search_service._explainer.suggest_missing",
                new_callable=AsyncMock,
                return_value="Buy a blue gown.",
            ):
                result = await svc.search(
                    "user1", "blue wedding",
                    # No candidates → no outfits → suggest_missing called
                    candidate_outfits=[],
                    explain=True,
                )
            assert result.missing_piece_suggestion is not None

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Conversational search
# ---------------------------------------------------------------------------

class TestConversationalSearch:
    def test_start_conversation_returns_session_id(self, svc):
        async def _run():
            candidates = [_outfit(colors=["blue"])]
            result = await svc.start_conversation(
                "user1", "blue outfit", candidate_outfits=candidates
            )
            assert result.session_id is not None

        asyncio.get_event_loop().run_until_complete(_run())

    def test_refine_uses_session(self, svc):
        async def _run():
            candidates = [_outfit(colors=["blue"], formality=0.5, oid="low"),
                          _outfit(colors=["blue"], formality=0.9, oid="high")]
            start_result = await svc.start_conversation(
                "user1", "blue outfit", candidate_outfits=candidates
            )
            session_id = start_result.session_id
            refine_result = await svc.refine_conversation(
                session_id, "more formal", candidate_outfits=candidates
            )
            assert refine_result.session_id == session_id

        asyncio.get_event_loop().run_until_complete(_run())

    def test_refine_missing_session_raises(self, svc):
        async def _run():
            with pytest.raises(ValueError, match="not found"):
                await svc.refine_conversation(
                    "non-existent-session", "more formal", candidate_outfits=[]
                )

        asyncio.get_event_loop().run_until_complete(_run())

    def test_pagination_in_refine(self, svc):
        async def _run():
            candidates = [_outfit(colors=["blue"], oid=str(i)) for i in range(20)]
            start_result = await svc.start_conversation(
                "user1", "blue outfit", candidate_outfits=candidates, max_results=5
            )
            page1_ids = {o["id"] for o in start_result.outfits}
            refine_result = await svc.refine_conversation(
                start_result.session_id, "show me more",
                candidate_outfits=candidates, max_results=5,
            )
            page2_ids = {o["id"] for o in refine_result.outfits}
            # Pages should be different
            assert page1_ids != page2_ids or len(candidates) <= 5

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestPromptSearchResultToDict:
    def test_to_dict_has_required_keys(self, svc):
        async def _run():
            result = await svc.search(
                "user1", "blue outfit", candidate_outfits=[], explain=False
            )
            d = result.to_dict()
            for key in [
                "outfits", "prompt_interpretation", "filters", "explanation",
                "compromise_note", "missing_piece_suggestion", "session_id",
                "total_candidates", "confidence", "is_off_topic",
                "clarification_needed",
            ]:
                assert key in d

        asyncio.get_event_loop().run_until_complete(_run())
