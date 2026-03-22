"""
Prompt Search Explainer — Layer 4 LLM
========================================

Generates prompt-aware natural-language explanations for outfit recommendations
that were triggered by a free-text user query.

Features:
  • explain_result       — why this outfit matches the user's prompt
  • explain_compromise   — when wardrobe can't fully satisfy the prompt
  • suggest_missing      — what to buy to satisfy the prompt better
  • LRU cache (TTL 30 days) keyed on content hash
  • Graceful degradation with template fallbacks

Public API:
    explainer = PromptSearchExplainer()
    text = await explainer.explain_result(outfit, filters, prompt_score)
    text = await explainer.explain_compromise(prompt, outfit, missing_aspects)
    text = await explainer.suggest_missing(prompt, filters)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.core import get_logger
from src.layer3_context.prompt_to_filters import WardrobeFilters

logger = get_logger(__name__)

_CACHE_TTL_DAYS = 30
_MAX_TOKENS = 350


# ---------------------------------------------------------------------------
# Gemini backend (same singleton pattern as moodboard_explainer.py)
# ---------------------------------------------------------------------------

class _GeminiLLM:
    def __init__(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore
            from config import get_settings

            settings = get_settings()
            genai.configure(api_key=settings.google_api_key)
            model_name = getattr(settings, "llm_model", "gemini-2.0-flash")
            self._model = genai.GenerativeModel(model_name)
            self._available = True
        except Exception as exc:
            logger.warning(f"PromptSearchExplainer: Gemini unavailable — {exc}")
            self._available = False

    async def generate(self, prompt: str, max_tokens: int = _MAX_TOKENS) -> str:
        if not self._available:
            raise RuntimeError("Gemini LLM not available")
        import google.generativeai as genai  # type: ignore

        gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.7,
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_content(prompt, generation_config=gen_config),
        )
        return response.text.strip()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    text: str
    created_at: datetime

    def is_valid(self) -> bool:
        return datetime.now(timezone.utc) - self.created_at < timedelta(days=_CACHE_TTL_DAYS)


# ---------------------------------------------------------------------------
# PromptSearchExplainer
# ---------------------------------------------------------------------------

class PromptSearchExplainer:
    """
    Generates prompt-aware explanations for outfit search results.
    All public methods are async and cache their output.
    """

    def __init__(self) -> None:
        self._llm = _GeminiLLM()
        self._cache: Dict[str, _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def explain_result(
        self,
        outfit: Dict[str, Any],
        filters: WardrobeFilters,
        raw_prompt: str,
        prompt_score: float = 0.5,
    ) -> str:
        """
        Explain why *outfit* was recommended for *raw_prompt*.

        Returns a 2-3 sentence plain-text explanation.
        """
        key = self._hash(f"result|{raw_prompt}|{json.dumps(outfit, sort_keys=True, default=str)}")
        if cached := self._get_cached(key):
            return cached

        prompt = self._result_prompt(outfit, filters, raw_prompt, prompt_score)
        fallback = self._fallback_result(outfit, filters, raw_prompt)
        result = await self._safe_generate(prompt, fallback)
        self._cache[key] = _CacheEntry(text=result, created_at=datetime.now(timezone.utc))
        return result

    async def explain_compromise(
        self,
        raw_prompt: str,
        outfit: Dict[str, Any],
        missing_aspects: List[str],
    ) -> str:
        """
        Explain that the wardrobe couldn't fully match the prompt and describe the
        best available compromise.
        """
        key = self._hash(f"compromise|{raw_prompt}|{','.join(sorted(missing_aspects))}")
        if cached := self._get_cached(key):
            return cached

        prompt = self._compromise_prompt(raw_prompt, outfit, missing_aspects)
        fallback = self._fallback_compromise(raw_prompt, missing_aspects)
        result = await self._safe_generate(prompt, fallback)
        self._cache[key] = _CacheEntry(text=result, created_at=datetime.now(timezone.utc))
        return result

    async def suggest_missing(
        self,
        raw_prompt: str,
        filters: WardrobeFilters,
        wardrobe_size: int = 0,
    ) -> str:
        """
        When no outfit matches at all, suggest what piece(s) to buy.
        """
        key = self._hash(f"missing|{raw_prompt}|{wardrobe_size}")
        if cached := self._get_cached(key):
            return cached

        prompt = self._missing_prompt(raw_prompt, filters, wardrobe_size)
        fallback = self._fallback_missing(raw_prompt, filters)
        result = await self._safe_generate(prompt, fallback)
        self._cache[key] = _CacheEntry(text=result, created_at=datetime.now(timezone.utc))
        return result

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _result_prompt(
        self,
        outfit: Dict[str, Any],
        filters: WardrobeFilters,
        raw_prompt: str,
        prompt_score: float,
    ) -> str:
        colors = ", ".join(outfit.get("dominant_colors") or []) or "unknown"
        styles = ", ".join((outfit.get("dominant_styles") or {}).keys()) or "mixed"
        formality = outfit.get("formality_score")
        formality_str = f"{formality:.0%}" if formality is not None else "unknown"
        title = outfit.get("title") or "this outfit"
        score_pct = f"{prompt_score:.0%}"

        return (
            f"You are a friendly personal stylist. "
            f"The user asked: \"{raw_prompt}\". "
            f"You recommended {title} with dominant colours: {colors}, "
            f"styles: {styles}, formality: {formality_str}. "
            f"The match score is {score_pct}. "
            f"Write 2-3 sentences explaining specifically why this outfit answers the user's request. "
            f"Be direct, warm, and reference the user's words."
        )

    def _compromise_prompt(
        self,
        raw_prompt: str,
        outfit: Dict[str, Any],
        missing_aspects: List[str],
    ) -> str:
        title = outfit.get("title") or "this outfit"
        missing_str = ", ".join(missing_aspects)
        return (
            f"You are a friendly personal stylist. "
            f"The user asked: \"{raw_prompt}\". "
            f"Their wardrobe doesn't perfectly match; the closest outfit is {title}. "
            f"Missing aspects: {missing_str}. "
            f"Write 2-3 sentences acknowledging the gap and explaining the compromise positively. "
            f"End with one encouraging note."
        )

    def _missing_prompt(
        self,
        raw_prompt: str,
        filters: WardrobeFilters,
        wardrobe_size: int,
    ) -> str:
        colors = ", ".join(filters.target_colors) or "the requested colour"
        occasion = filters.target_occasion or "this occasion"
        return (
            f"You are a friendly personal stylist. "
            f"The user asked: \"{raw_prompt}\". "
            f"Their wardrobe of {wardrobe_size} items has no suitable outfit. "
            f"They need {colors} pieces suitable for {occasion}. "
            f"Suggest 1-2 specific wardrobe additions (with colour and garment type) "
            f"that would immediately unlock multiple outfits for their request. "
            f"Keep it under 3 sentences."
        )

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _fallback_result(self, outfit: Dict[str, Any], filters: WardrobeFilters, raw_prompt: str) -> str:
        colors = ", ".join(outfit.get("dominant_colors") or []) or "neutral tones"
        occasion = filters.target_occasion or "your occasion"
        return (
            f"This outfit in {colors} is a great match for {occasion}. "
            f"It aligns with your request: \"{raw_prompt[:60]}\"."
        )

    def _fallback_compromise(self, raw_prompt: str, missing_aspects: List[str]) -> str:
        missing_str = " and ".join(missing_aspects[:2]) if missing_aspects else "some aspects"
        return (
            f"Your wardrobe doesn't have a perfect match for \"{raw_prompt[:60]}\", "
            f"particularly regarding {missing_str}. "
            f"Here's the closest available option — it still captures the overall vibe!"
        )

    def _fallback_missing(self, raw_prompt: str, filters: WardrobeFilters) -> str:
        colors = ", ".join(filters.target_colors[:2]) if filters.target_colors else "the desired colour"
        occasion = filters.target_occasion or "this occasion"
        return (
            f"To fulfil your request for \"{raw_prompt[:60]}\", consider adding "
            f"a {colors} piece suitable for {occasion} to your wardrobe."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _safe_generate(self, prompt: str, fallback: str) -> str:
        try:
            return await self._llm.generate(prompt)
        except Exception as exc:
            logger.warning(f"PromptSearchExplainer LLM error — {exc}")
            return fallback

    def _get_cached(self, key: str) -> Optional[str]:
        entry = self._cache.get(key)
        if entry and entry.is_valid():
            return entry.text
        return None

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
