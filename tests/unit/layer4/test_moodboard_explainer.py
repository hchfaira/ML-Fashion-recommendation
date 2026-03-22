"""Unit tests for MoodBoardExplainer (layer4_llm/moodboard_explainer.py)."""
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.layer4_llm.moodboard_explainer import MoodBoardExplainer, _CacheEntry, _GeminiLLM
from src.database.models import MoodBoardStyleProfile


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_profile(
    board_id=None,
    dominant_styles=None,
    dominant_colors=None,
    formality_average=None,
    coherence_score=0.8,
    items_count=5,
) -> MoodBoardStyleProfile:
    p = MoodBoardStyleProfile()
    p.board_id = board_id or str(uuid4())
    p.dominant_styles = dominant_styles or {"minimalist": 0.8, "parisian": 0.4}
    p.dominant_colors = dominant_colors or [{"color": "beige", "frequency": 0.5}]
    p.formality_average = formality_average
    p.coherence_score = coherence_score
    p.items_count = items_count
    return p


# ── Tests: _CacheEntry ────────────────────────────────────────────────────────

class TestCacheEntry:
    def test_is_valid_when_fresh(self):
        entry = _CacheEntry("hello")
        assert entry.is_valid() is True

    def test_is_invalid_when_expired(self):
        entry = _CacheEntry("hello")
        entry.created_at = datetime.now(timezone.utc) - timedelta(days=31)
        assert entry.is_valid() is False


# ── Tests: fallback methods ───────────────────────────────────────────────────

class TestFallbacks:
    def setup_method(self):
        self.explainer = MoodBoardExplainer.__new__(MoodBoardExplainer)
        self.explainer._llm = MagicMock()
        self.explainer._cache = {}

    def test_fallback_recommendation_contains_style(self):
        profile = make_profile(dominant_styles={"minimalist": 0.9})
        result = self.explainer._fallback_recommendation(profile)
        assert "minimalist" in result

    def test_fallback_gap_contains_score(self):
        result = self.explainer._fallback_gap(72.5, "Summer Vibes")
        assert "72" in result
        assert "Summer Vibes" in result

    def test_fallback_summary_contains_board_name(self):
        profile = make_profile(items_count=10, dominant_styles={"casual": 0.6})
        result = self.explainer._fallback_summary(profile, "My Board")
        assert "My Board" in result
        assert "10" in result


# ── Tests: prompt builders ────────────────────────────────────────────────────

class TestPromptBuilders:
    def setup_method(self):
        self.explainer = MoodBoardExplainer.__new__(MoodBoardExplainer)
        self.explainer._llm = MagicMock()
        self.explainer._cache = {}

    def test_recommendation_prompt_includes_styles(self):
        profile = make_profile(dominant_styles={"minimalist": 0.8})
        outfit = {"dominant_colors": ["beige"], "dominant_styles": {"casual": 0.5}}
        prompt = self.explainer._build_recommendation_prompt(outfit, profile, ["id1"])
        assert "minimalist" in prompt
        assert "beige" in prompt

    def test_gap_prompt_includes_board_name(self):
        prompt = self.explainer._build_gap_prompt(
            60.0, ["red"], ["bohemian"], ["trench coat"], ["beige"], "Work Inspo"
        )
        assert "Work Inspo" in prompt
        assert "red" in prompt
        assert "trench coat" in prompt

    def test_summary_prompt_coherence_label(self):
        profile = make_profile(coherence_score=0.9)
        prompt = self.explainer._build_summary_prompt(profile, "Board A")
        assert "very cohesive" in prompt

        profile_low = make_profile(coherence_score=0.2)
        prompt_low = self.explainer._build_summary_prompt(profile_low, "Board B")
        assert "eclectic" in prompt_low


# ── Tests: async public methods (with mocked LLM) ────────────────────────────

class TestAsyncMethods:
    def setup_method(self):
        self.explainer = MoodBoardExplainer.__new__(MoodBoardExplainer)
        self.explainer._cache = {}

    @pytest.mark.asyncio
    async def test_explain_recommendation_uses_cache(self):
        profile = make_profile()
        outfit = {"dominant_colors": ["beige"]}

        call_count = 0

        async def fake_generate(prompt, fallback):
            nonlocal call_count
            call_count += 1
            return "LLM response"

        self.explainer._safe_generate = fake_generate

        # First call
        r1 = await self.explainer.explain_recommendation(outfit, profile)
        # Second call — should hit cache
        r2 = await self.explainer.explain_recommendation(outfit, profile)

        assert r1 == r2 == "LLM response"
        assert call_count == 1  # Only one real call

    @pytest.mark.asyncio
    async def test_explain_recommendation_fallback_on_error(self):
        profile = make_profile(dominant_styles={"minimalist": 0.8})
        outfit = {}

        async def failing_generate(prompt, fallback):
            return fallback  # simulate LLM unavailable → returns fallback

        self.explainer._safe_generate = failing_generate

        result = await self.explainer.explain_recommendation(outfit, profile)
        assert "minimalist" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_explain_gap_analysis_returns_string(self):
        async def fake_generate(prompt, fallback):
            return "Gap explanation"

        self.explainer._safe_generate = fake_generate

        result = await self.explainer.explain_gap_analysis(
            alignment_score=65.0,
            missing_colors=["red"],
            missing_styles=["bohemian"],
            missing_piece_types=["trench"],
            well_covered=["beige"],
            board_name="Summer Board",
        )
        assert result == "Gap explanation"

    @pytest.mark.asyncio
    async def test_generate_board_summary_returns_string(self):
        profile = make_profile()

        async def fake_generate(prompt, fallback):
            return "Style summary"

        self.explainer._safe_generate = fake_generate

        result = await self.explainer.generate_board_summary(profile, "My Board")
        assert result == "Style summary"

    @pytest.mark.asyncio
    async def test_safe_generate_returns_fallback_on_exception(self):
        self.explainer._llm = MagicMock()
        self.explainer._llm.generate = AsyncMock(side_effect=RuntimeError("API down"))

        result = await self.explainer._safe_generate("some prompt", "fallback text")
        assert result == "fallback text"


# ── Tests: _hash ──────────────────────────────────────────────────────────────

class TestHash:
    def test_same_input_same_hash(self):
        explainer = MoodBoardExplainer.__new__(MoodBoardExplainer)
        h1 = explainer._hash({"key": "value"})
        h2 = explainer._hash({"key": "value"})
        assert h1 == h2

    def test_different_input_different_hash(self):
        explainer = MoodBoardExplainer.__new__(MoodBoardExplainer)
        h1 = explainer._hash({"key": "a"})
        h2 = explainer._hash({"key": "b"})
        assert h1 != h2

    def test_returns_string(self):
        explainer = MoodBoardExplainer.__new__(MoodBoardExplainer)
        assert isinstance(explainer._hash({}), str)
