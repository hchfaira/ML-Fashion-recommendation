"""Unit tests for PromptSearchExplainer (Layer 4)."""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.layer4_llm.prompt_search_explainer import (
    PromptSearchExplainer,
    _CacheEntry,
)
from src.layer3_context.prompt_to_filters import WardrobeFilters


def _filters(**kwargs) -> WardrobeFilters:
    return WardrobeFilters(**kwargs)


# ---------------------------------------------------------------------------
# _CacheEntry
# ---------------------------------------------------------------------------

class TestCacheEntry:
    def test_fresh_entry_is_valid(self):
        e = _CacheEntry(text="hello", created_at=datetime.now(timezone.utc))
        assert e.is_valid()

    def test_expired_entry_is_invalid(self):
        e = _CacheEntry(
            text="hello",
            created_at=datetime.now(timezone.utc) - timedelta(days=31),
        )
        assert not e.is_valid()


# ---------------------------------------------------------------------------
# Fallback methods
# ---------------------------------------------------------------------------

class TestFallbacks:
    def setup_method(self):
        self.ex = PromptSearchExplainer()

    def test_fallback_result_contains_occasion(self):
        f = _filters(target_occasion="wedding")
        text = self.ex._fallback_result({"dominant_colors": ["blue"]}, f, "blue wedding")
        assert "wedding" in text.lower()

    def test_fallback_result_contains_color(self):
        f = _filters(target_colors=["blue"])
        text = self.ex._fallback_result({"dominant_colors": ["blue"]}, f, "blue dress")
        assert "blue" in text.lower()

    def test_fallback_compromise_contains_missing(self):
        text = self.ex._fallback_compromise("blue wedding", ["color: blue"])
        assert "color" in text.lower() or "blue" in text.lower()

    def test_fallback_missing_contains_occasion(self):
        f = _filters(target_occasion="gala", target_colors=["black"])
        text = self.ex._fallback_missing("black gala look", f)
        assert "gala" in text.lower() or "black" in text.lower()

    def test_fallback_missing_with_empty_filters(self):
        f = _filters()
        text = self.ex._fallback_missing("something nice", f)
        assert isinstance(text, str) and len(text) > 5


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

class TestPromptBuilders:
    def setup_method(self):
        self.ex = PromptSearchExplainer()

    def test_result_prompt_includes_user_request(self):
        f = _filters(target_occasion="wedding")
        p = self.ex._result_prompt(
            {"dominant_colors": ["navy"], "dominant_styles": {"elegant": 0.8}, "title": "Navy gown"},
            f, "blue wedding dress", 0.9,
        )
        assert "blue wedding dress" in p

    def test_compromise_prompt_includes_missing(self):
        p = self.ex._compromise_prompt("wedding", {}, ["color: blue", "occasion: wedding"])
        assert "color" in p.lower() or "wedding" in p.lower()

    def test_missing_prompt_includes_wardrobe_size(self):
        f = _filters(target_occasion="gala", target_colors=["gold"])
        p = self.ex._missing_prompt("gold gala look", f, wardrobe_size=42)
        assert "42" in p


# ---------------------------------------------------------------------------
# Async methods — cache & safe_generate
# ---------------------------------------------------------------------------

class TestAsyncMethods:
    def setup_method(self):
        self.ex = PromptSearchExplainer()

    def test_explain_result_cached(self):
        async def _run():
            f = _filters(target_occasion="wedding")
            outfit = {"dominant_colors": ["blue"], "title": "Blue gown"}
            # Patch LLM to return a known string
            with patch.object(self.ex._llm, "generate", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = "Great match!"
                r1 = await self.ex.explain_result(outfit, f, "blue wedding")
                r2 = await self.ex.explain_result(outfit, f, "blue wedding")
                # LLM called only once
                assert mock_gen.call_count == 1
                assert r1 == r2 == "Great match!"

        asyncio.get_event_loop().run_until_complete(_run())

    def test_explain_result_fallback_on_llm_error(self):
        async def _run():
            f = _filters(target_occasion="beach")
            outfit = {"dominant_colors": ["white"]}
            with patch.object(self.ex._llm, "generate", new_callable=AsyncMock) as mock_gen:
                mock_gen.side_effect = RuntimeError("LLM down")
                result = await self.ex.explain_result(outfit, f, "beach look")
                # Should not raise; fallback string returned
                assert isinstance(result, str) and len(result) > 0

        asyncio.get_event_loop().run_until_complete(_run())

    def test_explain_compromise_returns_string(self):
        async def _run():
            with patch.object(self.ex._llm, "generate", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = "Compromise explanation."
                result = await self.ex.explain_compromise(
                    "blue wedding", {"title": "White dress"}, ["color: blue"]
                )
                assert result == "Compromise explanation."

        asyncio.get_event_loop().run_until_complete(_run())

    def test_suggest_missing_returns_string(self):
        async def _run():
            f = _filters(target_occasion="gala", target_colors=["gold"])
            with patch.object(self.ex._llm, "generate", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = "Buy a gold gown."
                result = await self.ex.suggest_missing("gold gala look", f, wardrobe_size=10)
                assert result == "Buy a gold gown."

        asyncio.get_event_loop().run_until_complete(_run())

    def test_safe_generate_returns_fallback_on_exception(self):
        async def _run():
            with patch.object(self.ex._llm, "generate", new_callable=AsyncMock) as mock_gen:
                mock_gen.side_effect = Exception("network error")
                result = await self.ex._safe_generate("prompt", "fallback text")
                assert result == "fallback text"

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

class TestHash:
    def test_same_content_same_hash(self):
        h1 = PromptSearchExplainer._hash("hello world")
        h2 = PromptSearchExplainer._hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        assert PromptSearchExplainer._hash("a") != PromptSearchExplainer._hash("b")

    def test_returns_string(self):
        assert isinstance(PromptSearchExplainer._hash("x"), str)
