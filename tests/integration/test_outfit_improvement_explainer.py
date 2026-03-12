"""
Integration tests for OutfitImprovementExplainer (Layer 4).

These tests verify that OutfitImprovementExplainer correctly orchestrates
OutfitImprover (Layer 2) without calling the real LLM (mocked).
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List
from uuid import uuid4

from src.core.models import (
    Garment,
    GarmentAttributes,
    GarmentCategory,
    FormalityLevel,
    UserContext,
    ColorProfile,
    PatternInfo,
    MaterialProfile,
)
from src.layer2_style.season_color_harmony import ColorSeason
from src.layer2_style.volume_balance_scorer import BodyShape
from src.layer4_llm.outfit_improvement_explainer import (
    OutfitImprovementExplainer,
    ImprovementExplanation,
    SuggestedPiece,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _garment(
    category: GarmentCategory,
    color: str = "black",
    formality: FormalityLevel = FormalityLevel.CASUAL,
    gid: str | None = None,
) -> Garment:
    """Build a minimal Garment for testing."""
    return Garment(
        id=gid or str(uuid4()),
        attributes=GarmentAttributes(
            category=category,
            color=ColorProfile(primary=color),
            pattern=PatternInfo(type="solid"),
            material=MaterialProfile(primary="cotton"),
            formality_level=formality,
            season_suitable=["spring", "summer", "fall", "winter"],
        ),
    )


def _make_wardrobe() -> List[Garment]:
    """Return a minimal but varied wardrobe."""
    return [
        _garment(GarmentCategory.TOP, "white"),
        _garment(GarmentCategory.TOP, "navy"),
        _garment(GarmentCategory.BOTTOM, "black"),
        _garment(GarmentCategory.BOTTOM, "beige"),
        _garment(GarmentCategory.SHOES, "black"),
        _garment(GarmentCategory.ACCESSORY, "gold"),
        _garment(GarmentCategory.OUTERWEAR, "charcoal"),
    ]


LLM_STUB = "Explication LLM simulée."


# ---------------------------------------------------------------------------
# Fake LLM service that never calls OpenAI
# ---------------------------------------------------------------------------

class _FakeLLMService:
    """Drop-in replacement for LLMService that never touches the network."""

    def __init__(self, response: str = LLM_STUB):
        self._response = response

    async def generate_completion(self, prompt: str, **kwargs) -> str:
        self._last_prompt = prompt
        return self._response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wardrobe() -> List[Garment]:
    return _make_wardrobe()


@pytest.fixture
def outfit(wardrobe) -> List[Garment]:
    """A simple top+bottom outfit."""
    tops    = [g for g in wardrobe if g.attributes.category == GarmentCategory.TOP]
    bottoms = [g for g in wardrobe if g.attributes.category == GarmentCategory.BOTTOM]
    return [tops[0], bottoms[0]]


@pytest.fixture
def context() -> UserContext:
    return UserContext()


@pytest.fixture
def explainer() -> OutfitImprovementExplainer:
    """OutfitImprovementExplainer with LLM mocked — no OpenAI key needed."""
    inst = OutfitImprovementExplainer.__new__(OutfitImprovementExplainer)
    inst._improver = _get_improver_class()()
    inst._llm = _FakeLLMService()
    inst._max_suggestions = 5
    return inst


def _get_improver_class():
    from src.layer2_style.outfit_improver import OutfitImprover
    return OutfitImprover


# ---------------------------------------------------------------------------
# Helper: swap LLM response per test
# ---------------------------------------------------------------------------

def _patch_llm(explainer: OutfitImprovementExplainer, response: str = LLM_STUB):
    explainer._llm = _FakeLLMService(response)


# ---------------------------------------------------------------------------
# TestImprovementExplanationContract
# ---------------------------------------------------------------------------

class TestImprovementExplanationContract:
    """ImprovementExplanation must always return a complete, valid object."""

    @pytest.mark.asyncio
    async def test_returns_improvement_explanation(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        assert isinstance(result, ImprovementExplanation)

    @pytest.mark.asyncio
    async def test_current_score_is_float(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        assert isinstance(result.current_score, float)
        assert 0.0 <= result.current_score <= 1.0

    @pytest.mark.asyncio
    async def test_grade_is_letter(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        assert result.grade in {"A", "B", "C", "D", "F"}

    @pytest.mark.asyncio
    async def test_potential_score_gte_current(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        assert result.potential_score >= result.current_score

    @pytest.mark.asyncio
    async def test_potential_score_capped_at_one(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        assert result.potential_score <= 1.0

    @pytest.mark.asyncio
    async def test_diagnosis_summary_is_string(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        assert isinstance(result.diagnosis_summary, str)
        assert len(result.diagnosis_summary) > 0

    @pytest.mark.asyncio
    async def test_to_dict_contains_required_keys(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        d = result.to_dict()
        for key in (
            "current_score", "grade", "potential_score",
            "diagnosis_summary", "weak_dimensions", "strong_dimensions",
            "suggested_pieces", "profile_note", "short_summary",
        ):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# TestSuggestedPieces
# ---------------------------------------------------------------------------

class TestSuggestedPieces:
    """Suggested pieces must be valid and ranked."""

    @pytest.mark.asyncio
    async def test_pieces_are_list(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        assert isinstance(result.suggested_pieces, list)

    @pytest.mark.asyncio
    async def test_pieces_respect_max_suggestions(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        assert len(result.suggested_pieces) <= explainer._max_suggestions

    @pytest.mark.asyncio
    async def test_each_piece_has_required_fields(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        for piece in result.suggested_pieces:
            assert isinstance(piece, SuggestedPiece)
            assert piece.suggestion_type in {"addition", "replacement", "purchase"}
            assert isinstance(piece.garment_description, str)
            assert isinstance(piece.llm_explanation, str)
            assert isinstance(piece.expected_score_change, float)

    @pytest.mark.asyncio
    async def test_llm_explanation_uses_stub(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer, "Réponse LLM test.")
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        if result.suggested_pieces:
            assert result.suggested_pieces[0].llm_explanation == "Réponse LLM test."

    @pytest.mark.asyncio
    async def test_priority_is_ascending(self, explainer, outfit, wardrobe, context):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        priorities = [p.priority for p in result.suggested_pieces]
        assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# TestPersonalisedNarration
# ---------------------------------------------------------------------------

class TestPersonalisedNarration:
    """Verify that season / body_shape are forwarded to LLM prompts."""

    @pytest.mark.asyncio
    async def test_narration_called_with_season_in_prompt(
        self, explainer, outfit, wardrobe, context
    ):
        captured: list[str] = []

        class CaptureLLM:
            async def generate_completion(self, prompt: str, **kwargs) -> str:
                captured.append(prompt)
                return LLM_STUB

        explainer._llm = CaptureLLM()

        await explainer.explain_improvements(
            outfit, wardrobe, context,
            user_season=ColorSeason.AUTUMN,
        )
        combined = "\n".join(captured)
        assert "autumn" in combined.lower() or "automne" in combined.lower()

    @pytest.mark.asyncio
    async def test_narration_called_with_body_shape_in_prompt(
        self, explainer, outfit, wardrobe, context
    ):
        captured: list[str] = []

        class CaptureLLM:
            async def generate_completion(self, prompt: str, **kwargs) -> str:
                captured.append(prompt)
                return LLM_STUB

        explainer._llm = CaptureLLM()

        await explainer.explain_improvements(
            outfit, wardrobe, context,
            body_shape=BodyShape.PEAR,
        )
        combined = "\n".join(captured)
        assert "pear" in combined.lower()

    @pytest.mark.asyncio
    async def test_no_profile_gives_empty_profile_note(
        self, explainer, outfit, wardrobe, context
    ):
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        assert result.profile_note == ""

    @pytest.mark.asyncio
    async def test_with_profile_gives_non_empty_profile_note(
        self, explainer, outfit, wardrobe, context
    ):
        _patch_llm(explainer, "Note profil LLM.")
        result = await explainer.explain_improvements(
            outfit, wardrobe, context,
            user_season=ColorSeason.SUMMER,
            body_shape=BodyShape.HOURGLASS,
        )
        assert len(result.profile_note) > 0


# ---------------------------------------------------------------------------
# TestEmptyInputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    """Edge cases: empty outfit, empty wardrobe."""

    @pytest.mark.asyncio
    async def test_empty_garments_returns_gracefully(
        self, explainer, wardrobe, context
    ):
        _patch_llm(explainer)
        result = await explainer.explain_improvements([], wardrobe, context)
        assert result.current_score == 0.0
        assert result.grade == "F"
        assert result.suggested_pieces == []

    @pytest.mark.asyncio
    async def test_empty_wardrobe_still_returns_result(
        self, explainer, outfit, context
    ):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, [], context)
        assert isinstance(result, ImprovementExplanation)
        # No additions/replacements possible from empty wardrobe
        wardrobe_pieces = [
            p for p in result.suggested_pieces
            if p.suggestion_type in {"addition", "replacement"}
        ]
        assert wardrobe_pieces == []

    @pytest.mark.asyncio
    async def test_no_context_still_works(self, explainer, outfit, wardrobe):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(outfit, wardrobe, context=None)
        assert isinstance(result, ImprovementExplanation)


# ---------------------------------------------------------------------------
# TestDetailLevel
# ---------------------------------------------------------------------------

class TestDetailLevel:
    """Different detail levels must all produce valid results."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("level", ["brief", "standard", "detailed"])
    async def test_detail_level_produces_result(
        self, level, explainer, outfit, wardrobe, context
    ):
        _patch_llm(explainer)
        result = await explainer.explain_improvements(
            outfit, wardrobe, context,
            detail_level=level,
        )
        assert isinstance(result, ImprovementExplanation)
        assert len(result.diagnosis_summary) > 0


# ---------------------------------------------------------------------------
# TestLLMFallback
# ---------------------------------------------------------------------------

class TestLLMFallback:
    """When LLM fails, the result should degrade gracefully."""

    @pytest.mark.asyncio
    async def test_llm_failure_uses_fallback_explanation(
        self, explainer, outfit, wardrobe, context
    ):
        from src.core.exceptions import LLMError

        class FailingLLM:
            async def generate_completion(self, prompt: str, **kwargs) -> str:
                raise LLMError("API unavailable")

        explainer._llm = FailingLLM()

        # Should NOT raise — should fall back to OutfitImprover's text
        result = await explainer.explain_improvements(outfit, wardrobe, context)
        assert isinstance(result, ImprovementExplanation)
        # Fallback explanations come from OutfitImprover (non-empty strings)
        for piece in result.suggested_pieces:
            assert isinstance(piece.llm_explanation, str)
