"""
Mood Board Explainer — Layer 4 LLM
====================================

Generates plain-language explanations linking outfit recommendations to the
user's mood board profile.

Features:
  • explain_recommendation  — why an outfit matches the mood board
  • explain_gap_analysis     — how to close the gap between board and wardrobe
  • generate_board_summary   — human-readable profile summary

Optimisations:
  • LRU-style in-memory cache keyed on content hash (TTL 30 days)
  • Graceful degradation: returns a templated message when the LLM is unavailable
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.core import get_logger
from src.database.models import MoodBoardStyleProfile

logger = get_logger(__name__)

_CACHE_TTL_DAYS = 30
_MAX_TOKENS = 300


# ---------------------------------------------------------------------------
# Shared Gemini backend (same pattern as capsule_explainer.py)
# ---------------------------------------------------------------------------

class _GeminiLLM:
    """Lightweight async wrapper over google.generativeai."""

    def __init__(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore
            from config import get_settings

            settings = get_settings()
            genai.configure(api_key=settings.google_api_key)
            model_name = getattr(settings, "llm_model", "gemini-2.5-flash")
            self._model = genai.GenerativeModel(model_name)
            self._available = True
        except Exception as exc:
            logger.warning(f"MoodBoardExplainer: Gemini unavailable — {exc}")
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
# Cache entry
# ---------------------------------------------------------------------------

class _CacheEntry:
    def __init__(self, text: str) -> None:
        self.text = text
        self.created_at = datetime.now(timezone.utc)

    def is_valid(self) -> bool:
        age = datetime.now(timezone.utc) - self.created_at
        return age < timedelta(days=_CACHE_TTL_DAYS)


# ---------------------------------------------------------------------------
# Public Explainer
# ---------------------------------------------------------------------------

class MoodBoardExplainer:
    """
    LLM-powered explainer for mood board features.

    Falls back to template strings when Gemini is unavailable so the rest of
    the pipeline never breaks.
    """

    def __init__(self) -> None:
        self._llm = _GeminiLLM()
        self._cache: Dict[str, _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def explain_recommendation(
        self,
        outfit: Dict[str, Any],
        profile: MoodBoardStyleProfile,
        matching_outfit_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Explain why a candidate outfit matches the user's mood board.

        Args:
            outfit:               dict with keys like dominant_colors, dominant_styles
            profile:              MoodBoardStyleProfile ORM row
            matching_outfit_ids:  IDs of the saved outfits that are most similar

        Returns:
            A short, friendly explanation string.
        """
        key = self._hash(
            {"outfit": outfit, "board": profile.board_id, "matches": matching_outfit_ids}
        )
        if cached := self._get_cache(key):
            return cached

        prompt = self._build_recommendation_prompt(outfit, profile, matching_outfit_ids)
        text = await self._safe_generate(prompt, self._fallback_recommendation(profile))
        self._set_cache(key, text)
        return text

    async def explain_gap_analysis(
        self,
        alignment_score: float,
        missing_colors: List[str],
        missing_styles: List[str],
        missing_piece_types: List[str],
        well_covered: List[str],
        board_name: str,
    ) -> str:
        """Explain the gap between the user's wardrobe and their mood board."""
        key = self._hash(
            {
                "alignment": alignment_score,
                "m_colors": missing_colors,
                "m_styles": missing_styles,
                "m_pieces": missing_piece_types,
                "board": board_name,
            }
        )
        if cached := self._get_cache(key):
            return cached

        prompt = self._build_gap_prompt(
            alignment_score, missing_colors, missing_styles, missing_piece_types,
            well_covered, board_name,
        )
        fallback = self._fallback_gap(alignment_score, board_name)
        text = await self._safe_generate(prompt, fallback)
        self._set_cache(key, text)
        return text

    async def generate_board_summary(
        self,
        profile: MoodBoardStyleProfile,
        board_name: str,
    ) -> str:
        """Generate a human-readable summary of the mood board's style profile."""
        key = self._hash({"board_id": profile.board_id, "items": profile.items_count})
        if cached := self._get_cache(key):
            return cached

        prompt = self._build_summary_prompt(profile, board_name)
        fallback = self._fallback_summary(profile, board_name)
        text = await self._safe_generate(prompt, fallback)
        self._set_cache(key, text)
        return text

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_recommendation_prompt(
        self,
        outfit: Dict[str, Any],
        profile: MoodBoardStyleProfile,
        matching_ids: Optional[List[str]],
    ) -> str:
        top_styles = list((profile.dominant_styles or {}).keys())[:3]
        top_colors_raw = profile.dominant_colors or []
        top_colors = [c["color"] if isinstance(c, dict) else c for c in top_colors_raw][:4]
        outfit_colors = outfit.get("dominant_colors", [])
        outfit_styles = list((outfit.get("dominant_styles") or {}).keys())[:3]
        n_matches = len(matching_ids) if matching_ids else 0

        return (
            "You are a friendly personal stylist. In 2-3 sentences explain why this outfit "
            "matches the user's mood board. Be specific, warm, and jargon-free.\n\n"
            f"Mood board top styles: {', '.join(top_styles) or 'not specified'}\n"
            f"Mood board top colors: {', '.join(top_colors) or 'not specified'}\n"
            f"Outfit colors: {', '.join(outfit_colors) or 'not specified'}\n"
            f"Outfit styles: {', '.join(outfit_styles) or 'not specified'}\n"
            f"Similar saved outfits found: {n_matches}\n\n"
            "Write the explanation now:"
        )

    def _build_gap_prompt(
        self,
        alignment_score: float,
        missing_colors: List[str],
        missing_styles: List[str],
        missing_piece_types: List[str],
        well_covered: List[str],
        board_name: str,
    ) -> str:
        return (
            "You are a friendly personal stylist. In 3-4 sentences, explain the gap between "
            f"the user's wardrobe and their mood board called '{board_name}'. "
            "Start with what's already working, then gently suggest what's missing. "
            "Be encouraging and actionable.\n\n"
            f"Alignment score: {alignment_score:.0f}/100\n"
            f"Well covered: {', '.join(well_covered) or 'nothing yet'}\n"
            f"Missing colors: {', '.join(missing_colors) or 'none'}\n"
            f"Missing styles: {', '.join(missing_styles) or 'none'}\n"
            f"Missing piece types: {', '.join(missing_piece_types) or 'none'}\n\n"
            "Write the explanation now:"
        )

    def _build_summary_prompt(
        self, profile: MoodBoardStyleProfile, board_name: str
    ) -> str:
        top_styles = list((profile.dominant_styles or {}).keys())[:3]
        top_colors_raw = profile.dominant_colors or []
        top_colors = [c["color"] if isinstance(c, dict) else c for c in top_colors_raw][:4]
        coherence = profile.coherence_score or 0.0
        coherence_label = (
            "very cohesive" if coherence > 0.75
            else "moderately cohesive" if coherence > 0.45
            else "quite eclectic"
        )

        return (
            "You are a friendly personal stylist. In 2-3 sentences, summarise the style "
            f"aesthetic of the mood board called '{board_name}'. "
            "Name the overall aesthetic, mention key colors and styles, and note the "
            "board's cohesiveness. Be encouraging.\n\n"
            f"Top styles: {', '.join(top_styles) or 'mixed'}\n"
            f"Top colors: {', '.join(top_colors) or 'varied'}\n"
            f"Coherence: {coherence_label} ({coherence:.0%})\n"
            f"Number of saved outfits: {profile.items_count}\n\n"
            "Write the summary now:"
        )

    # ------------------------------------------------------------------
    # Fallback templates (used when LLM is unavailable)
    # ------------------------------------------------------------------

    def _fallback_recommendation(self, profile: MoodBoardStyleProfile) -> str:
        top_styles_raw = profile.dominant_styles or {}
        top_style = next(iter(top_styles_raw), "your aesthetic")
        return (
            f"This outfit aligns nicely with your {top_style} mood board! "
            "The colors and silhouette capture the essence of the looks you've saved."
        )

    def _fallback_gap(self, alignment_score: float, board_name: str) -> str:
        return (
            f"Your wardrobe aligns at {alignment_score:.0f}% with your '{board_name}' board. "
            "Adding a few targeted pieces will help you live your aesthetic even more fully."
        )

    def _fallback_summary(
        self, profile: MoodBoardStyleProfile, board_name: str
    ) -> str:
        n = profile.items_count
        top_style = next(iter(profile.dominant_styles or {}), "eclectic")
        return (
            f"Your '{board_name}' board contains {n} outfit inspiration(s) with a "
            f"predominantly {top_style} aesthetic. Keep adding looks to refine your profile!"
        )

    # ------------------------------------------------------------------
    # Cache & utilities
    # ------------------------------------------------------------------

    async def _safe_generate(self, prompt: str, fallback: str) -> str:
        try:
            return await self._llm.generate(prompt)
        except Exception as exc:
            logger.warning(f"MoodBoardExplainer: LLM call failed — {exc}. Using fallback.")
            return fallback

    def _get_cache(self, key: str) -> Optional[str]:
        entry = self._cache.get(key)
        if entry and entry.is_valid():
            return entry.text
        if entry:
            del self._cache[key]
        return None

    def _set_cache(self, key: str, text: str) -> None:
        self._cache[key] = _CacheEntry(text)

    @staticmethod
    def _hash(data: Any) -> str:
        serialised = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()
