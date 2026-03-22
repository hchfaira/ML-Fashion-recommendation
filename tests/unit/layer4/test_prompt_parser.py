"""Unit tests for PromptParser (Layer 4)."""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.layer4_llm.prompt_parser import (
    ParsedPrompt,
    PromptParser,
    _CacheEntry,
    _OCCASION_MAP,
    _FORMALITY_BY_OCCASION,
)


# ---------------------------------------------------------------------------
# _CacheEntry
# ---------------------------------------------------------------------------

class TestCacheEntry:
    def test_fresh_entry_is_valid(self):
        e = _CacheEntry(ParsedPrompt(raw_prompt="x"))
        assert e.is_valid()

    def test_expired_entry_is_invalid(self):
        e = _CacheEntry(ParsedPrompt(raw_prompt="x"))
        e.created_at = datetime.now(timezone.utc) - timedelta(days=31)
        assert not e.is_valid()

    def test_entry_just_at_ttl_boundary(self):
        e = _CacheEntry(ParsedPrompt(raw_prompt="x"))
        e.created_at = datetime.now(timezone.utc) - timedelta(days=30, minutes=1)
        assert not e.is_valid()


# ---------------------------------------------------------------------------
# ParsedPrompt.to_dict
# ---------------------------------------------------------------------------

class TestParsedPromptToDict:
    def test_contains_all_keys(self):
        p = ParsedPrompt(raw_prompt="blue dress")
        d = p.to_dict()
        for key in ["raw_prompt", "colors", "occasion", "formality_range",
                    "styles", "season_hint", "anchor_pieces", "excluded_types",
                    "mood", "confidence", "is_off_topic", "clarification_needed"]:
            assert key in d

    def test_formality_range_is_list(self):
        p = ParsedPrompt(raw_prompt="x", formality_range=(0.3, 0.7))
        assert p.to_dict()["formality_range"] == [0.3, 0.7]

    def test_defaults(self):
        p = ParsedPrompt(raw_prompt="x")
        d = p.to_dict()
        assert d["colors"] == []
        assert d["occasion"] is None
        assert d["is_off_topic"] is False


# ---------------------------------------------------------------------------
# Off-topic detection
# ---------------------------------------------------------------------------

class TestOffTopicDetection:
    def setup_method(self):
        self.parser = PromptParser()

    def test_pizza_is_off_topic(self):
        assert self.parser._is_off_topic("pizza recipe")

    def test_clothing_prompt_is_not_off_topic(self):
        assert not self.parser._is_off_topic("blue dress for a wedding")

    def test_short_vague_prompt_flagged(self):
        assert self.parser._is_off_topic("hi")

    def test_outfit_keyword_overrides_off_topic_signal(self):
        # Has both "sport" (off-topic) and "outfit" (clothing signal)
        assert not self.parser._is_off_topic("sport outfit")

    def test_bitcoin_no_clothing_is_off_topic(self):
        assert self.parser._is_off_topic("bitcoin investment")


# ---------------------------------------------------------------------------
# Keyword parse — colors
# ---------------------------------------------------------------------------

class TestExtractColors:
    def setup_method(self):
        self.parser = PromptParser()

    def test_single_color(self):
        result = self.parser._extract_colors("i want something blue")
        assert "blue" in result

    def test_multiple_colors(self):
        result = self.parser._extract_colors("navy and white dress")
        assert "navy" in result
        assert "white" in result

    def test_no_colors(self):
        result = self.parser._extract_colors("something elegant for work")
        assert result == []

    def test_no_duplicates(self):
        result = self.parser._extract_colors("blue blue blue")
        assert result.count("blue") == 1

    def test_compound_color_navy(self):
        result = self.parser._extract_colors("a navy blazer")
        assert "navy" in result


# ---------------------------------------------------------------------------
# Keyword parse — occasion & formality
# ---------------------------------------------------------------------------

class TestKeywordOccasion:
    def setup_method(self):
        self.parser = PromptParser()

    def test_wedding_detected(self):
        r = self.parser._keyword_parse("blue dress for a wedding", "blue dress for a wedding")
        assert r.occasion == "wedding"

    def test_work_detected(self):
        r = self.parser._keyword_parse("smart outfit for office", "smart outfit for office")
        assert r.occasion == "work"

    def test_formality_derived_from_wedding(self):
        r = self.parser._keyword_parse("wedding outfit", "wedding outfit")
        lo, hi = r.formality_range
        assert lo >= 0.6
        assert hi == 1.0

    def test_no_occasion_default_formality(self):
        r = self.parser._keyword_parse("blue jeans", "blue jeans")
        assert r.formality_range == (0.0, 1.0)

    def test_formal_keyword_raises_floor(self):
        r = self.parser._keyword_parse("formal look", "formal look")
        assert r.formality_range[0] >= 0.5


# ---------------------------------------------------------------------------
# Keyword parse — styles & exclusions
# ---------------------------------------------------------------------------

class TestKeywordStyles:
    def setup_method(self):
        self.parser = PromptParser()

    def test_minimalist_detected(self):
        r = self.parser._keyword_parse("minimalist look", "minimalist look")
        assert "minimalist" in r.styles

    def test_bohemian_detected(self):
        r = self.parser._keyword_parse("boho vibes", "boho vibes")
        assert "bohemian" in r.styles

    def test_excluded_skirt(self):
        r = self.parser._keyword_parse("no skirts please", "no skirts please")
        assert "skirt" in r.excluded_types

    def test_excluded_heels(self):
        r = self.parser._keyword_parse("without heels", "without heels")
        assert "heels" in r.excluded_types


# ---------------------------------------------------------------------------
# Keyword parse — confidence & clarification
# ---------------------------------------------------------------------------

class TestKeywordConfidence:
    def setup_method(self):
        self.parser = PromptParser()

    def test_rich_prompt_has_high_confidence(self):
        r = self.parser._keyword_parse(
            "elegant navy dress for a formal wedding",
            "elegant navy dress for a formal wedding",
        )
        assert r.confidence >= 0.65

    def test_vague_prompt_needs_clarification(self):
        r = self.parser._keyword_parse("nice", "nice")
        assert r.clarification_needed is True

    def test_off_topic_result(self):
        r = ParsedPrompt(raw_prompt="pizza", is_off_topic=True)
        assert r.is_off_topic is True


# ---------------------------------------------------------------------------
# Async parse — cache & LLM fallback
# ---------------------------------------------------------------------------

class TestAsyncParse:
    def setup_method(self):
        self.parser = PromptParser()

    def test_cache_hit_returns_same_object(self):
        async def _run():
            # Fill cache with a first call (keyword fast-path — no LLM needed)
            r1 = await self.parser.parse("blue wedding dress")
            r2 = await self.parser.parse("blue wedding dress")
            # Same object from cache
            assert r1 is r2

        asyncio.get_event_loop().run_until_complete(_run())

    def test_off_topic_returns_early(self):
        async def _run():
            # No clothing signals at all
            r = await self.parser.parse("bitcoin investment strategy")
            assert r.is_off_topic

        asyncio.get_event_loop().run_until_complete(_run())

    def test_high_confidence_skips_llm(self):
        """Keyword fast-path should never call the LLM for a rich prompt."""
        async def _run():
            with patch.object(self.parser._llm, "generate", new_callable=AsyncMock) as mock_gen:
                await self.parser.parse("elegant navy dress for a formal wedding")
                mock_gen.assert_not_called()

        asyncio.get_event_loop().run_until_complete(_run())

    def test_llm_fallback_on_error(self):
        """If LLM raises, keyword result is returned without crashing."""
        async def _run():
            # Force low confidence so LLM is attempted
            with patch.object(self.parser, "_keyword_parse") as mock_kw:
                low_conf = ParsedPrompt(raw_prompt="x", confidence=0.3)
                mock_kw.return_value = low_conf
                with patch.object(self.parser, "_llm_parse", new_callable=AsyncMock) as mock_llm:
                    mock_llm.side_effect = RuntimeError("LLM down")
                    result = await self.parser.parse("x")
                    assert result.confidence >= 0.3  # keyword result returned

        asyncio.get_event_loop().run_until_complete(_run())

    def test_llm_result_used_when_available(self):
        async def _run():
            llm_result = ParsedPrompt(
                raw_prompt="test",
                colors=["emerald"],
                occasion="gala",
                confidence=0.9,
            )
            with patch.object(self.parser, "_keyword_parse") as mock_kw:
                # Low confidence AND not off-topic so LLM is tried
                low = ParsedPrompt(raw_prompt="test", confidence=0.3, is_off_topic=False)
                mock_kw.return_value = low
                with patch.object(self.parser, "_is_off_topic", return_value=False):
                    with patch.object(self.parser, "_llm_parse", new_callable=AsyncMock) as mock_llm:
                        mock_llm.return_value = llm_result
                        result = await self.parser._parse_uncached("test")
                        assert result.occasion == "gala"

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

class TestHash:
    def test_same_prompt_same_hash(self):
        h1 = PromptParser._hash("Blue wedding dress")
        h2 = PromptParser._hash("blue wedding dress")
        assert h1 == h2  # normalised to lowercase

    def test_different_prompts_different_hash(self):
        assert PromptParser._hash("blue") != PromptParser._hash("red")

    def test_returns_string(self):
        assert isinstance(PromptParser._hash("test"), str)
