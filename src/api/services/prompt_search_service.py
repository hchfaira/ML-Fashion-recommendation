"""
Prompt Search Service — Orchestrator
======================================

Coordinates Layer 4 (PromptParser), Layer 3 (PromptToFilters),
Layer 2 (PromptAwareScorer), and Layer 4 (PromptSearchExplainer) to deliver
natural-language outfit search from a free-text prompt.

Public API:
    svc = PromptSearchService()

    # One-shot search
    result = await svc.search(user_id, prompt, candidate_outfits, ...)

    # Conversational search
    result = await svc.start_conversation(user_id, prompt, candidate_outfits, ...)
    result = await svc.refine_conversation(session_id, refinement, candidate_outfits, ...)

    # Debug
    parsed = await svc.parse_only(prompt)
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.core import get_logger
from src.layer4_llm.prompt_parser import ParsedPrompt, PromptParser
from src.layer3_context.prompt_to_filters import PromptToFilters, WardrobeFilters
from src.layer2_style.prompt_aware_scorer import PromptAwareScorer
from src.layer4_llm.prompt_search_explainer import PromptSearchExplainer
from src.api.services.outfit_conversation import OutfitConversationManager, SearchSession

logger = get_logger(__name__)

# Global singletons (stateless components re-used across requests)
_parser = PromptParser()
_translator = PromptToFilters()
_scorer = PromptAwareScorer()
_explainer = PromptSearchExplainer()
_conversation_mgr = OutfitConversationManager()

_DEFAULT_MAX_RESULTS = 10
_DEFAULT_PREFERENCE = 0.4
_MIN_SCORE_THRESHOLD = 0.20


def get_conversation_manager() -> OutfitConversationManager:
    """Expose the global conversation manager (for tests / DI)."""
    return _conversation_mgr


class PromptSearchResult:
    """Typed result of a prompt-based outfit search."""

    def __init__(
        self,
        outfits: List[Dict[str, Any]],
        parsed_prompt: ParsedPrompt,
        filters: WardrobeFilters,
        explanation: str = "",
        compromise_note: Optional[str] = None,
        missing_piece_suggestion: Optional[str] = None,
        session_id: Optional[str] = None,
        total_candidates: int = 0,
    ) -> None:
        self.outfits = outfits
        self.parsed_prompt = parsed_prompt
        self.filters = filters
        self.explanation = explanation
        self.compromise_note = compromise_note
        self.missing_piece_suggestion = missing_piece_suggestion
        self.session_id = session_id
        self.total_candidates = total_candidates

    def to_dict(self) -> dict:
        return {
            "outfits": self.outfits,
            "prompt_interpretation": self.parsed_prompt.to_dict(),
            "filters": self.filters.to_dict(),
            "explanation": self.explanation,
            "compromise_note": self.compromise_note,
            "missing_piece_suggestion": self.missing_piece_suggestion,
            "session_id": self.session_id,
            "total_candidates": self.total_candidates,
            "confidence": self.parsed_prompt.confidence,
            "is_off_topic": self.parsed_prompt.is_off_topic,
            "clarification_needed": self.parsed_prompt.clarification_needed,
        }


class PromptSearchService:
    """
    Orchestrates the full prompt → outfit pipeline.

    Stateless: all session state lives in ``OutfitConversationManager``.
    """

    # ------------------------------------------------------------------
    # One-shot search
    # ------------------------------------------------------------------

    async def search(
        self,
        user_id: str,
        prompt: str,
        candidate_outfits: List[Dict[str, Any]],
        base_scores: Optional[List[float]] = None,
        max_results: int = _DEFAULT_MAX_RESULTS,
        preference: float = _DEFAULT_PREFERENCE,
        explain: bool = True,
    ) -> PromptSearchResult:
        """
        Full pipeline: parse → filter → score → rank → explain.

        Args:
            user_id:           authenticated user
            prompt:            free-text request
            candidate_outfits: list of outfit dicts from the wardrobe
            base_scores:       optional pre-computed style scores (same length)
            max_results:       maximum outfits to return
            preference:        0-1, weight of prompt score vs base score
            explain:           whether to generate LLM explanation

        Returns:
            :class:`PromptSearchResult`
        """
        # 1. Parse
        parsed = await _parser.parse(prompt)

        if parsed.is_off_topic:
            return PromptSearchResult(
                outfits=[],
                parsed_prompt=parsed,
                filters=WardrobeFilters(),
                explanation="Your request doesn't seem to be about clothing. Please describe the outfit you're looking for.",
                total_candidates=len(candidate_outfits),
            )

        # 2. Translate to filters
        filters = _translator.translate(parsed)

        # 3. Apply hard filters
        passing = self._apply_hard_filters(candidate_outfits, filters)

        # 4. Align base_scores with filtered outfits
        if base_scores and len(base_scores) == len(candidate_outfits):
            idx_map = {id(o): s for o, s in zip(candidate_outfits, base_scores)}
            aligned_scores = [idx_map.get(id(o), 0.5) for o in passing]
        else:
            aligned_scores = [0.5] * len(passing)

        # 5. Score & rank
        ranked = _scorer.rank_outfits(passing, filters, aligned_scores, preference)

        # 6. Filter below threshold
        above_threshold = [o for o in ranked if o.get("final_score", 0) >= _MIN_SCORE_THRESHOLD]

        # 7. Paginate
        page_outfits = above_threshold[:max_results]

        # 8. Determine missing aspects for compromise note
        missing_aspects = self._identify_missing_aspects(parsed, page_outfits)
        compromise_note: Optional[str] = None
        missing_suggestion: Optional[str] = None

        # 9. Explanations (async)
        explanation = ""
        if explain:
            if not page_outfits:
                missing_suggestion = await _explainer.suggest_missing(
                    prompt, filters, wardrobe_size=len(candidate_outfits)
                )
            else:
                top = page_outfits[0]
                explanation = await _explainer.explain_result(
                    top, filters, prompt, top.get("prompt_score", 0.5)
                )
                if missing_aspects:
                    compromise_note = await _explainer.explain_compromise(prompt, top, missing_aspects)

        return PromptSearchResult(
            outfits=page_outfits,
            parsed_prompt=parsed,
            filters=filters,
            explanation=explanation,
            compromise_note=compromise_note,
            missing_piece_suggestion=missing_suggestion,
            total_candidates=len(candidate_outfits),
        )

    # ------------------------------------------------------------------
    # Conversational search
    # ------------------------------------------------------------------

    async def start_conversation(
        self,
        user_id: str,
        prompt: str,
        candidate_outfits: List[Dict[str, Any]],
        base_scores: Optional[List[float]] = None,
        max_results: int = _DEFAULT_MAX_RESULTS,
        preference: float = _DEFAULT_PREFERENCE,
    ) -> PromptSearchResult:
        """
        Run a search and open a conversation session for future refinements.

        Returns the search result with a ``session_id`` attached.
        """
        result = await self.search(
            user_id, prompt, candidate_outfits, base_scores,
            max_results, preference, explain=True,
        )
        session_id = _conversation_mgr.create_session(user_id, prompt, result.parsed_prompt)
        result.session_id = session_id
        return result

    async def refine_conversation(
        self,
        session_id: str,
        refinement: str,
        candidate_outfits: List[Dict[str, Any]],
        base_scores: Optional[List[float]] = None,
        max_results: int = _DEFAULT_MAX_RESULTS,
        preference: float = _DEFAULT_PREFERENCE,
    ) -> PromptSearchResult:
        """
        Apply a refinement to an existing conversation session and re-run
        the search with the updated cumulative prompt.

        Returns a fresh result with the same ``session_id``.
        """
        session = _conversation_mgr.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id!r} not found or expired")

        # Parse the refinement as a delta
        delta_parsed = await _parser.parse(refinement)

        # Apply delta to session
        session = _conversation_mgr.apply_refinement(session_id, refinement, delta_parsed)
        if session is None:
            raise ValueError(f"Session {session_id!r} disappeared during refinement")

        # Exclude styles the user rejected
        if session.excluded_styles:
            session.current_parsed.styles = [
                s for s in session.current_parsed.styles
                if s not in session.excluded_styles
            ]

        # Re-translate and re-run with the updated parsed prompt
        filters = _translator.translate(session.current_parsed)
        passing = self._apply_hard_filters(candidate_outfits, filters)

        if base_scores and len(base_scores) == len(candidate_outfits):
            idx_map = {id(o): s for o, s in zip(candidate_outfits, base_scores)}
            aligned_scores = [idx_map.get(id(o), 0.5) for o in passing]
        else:
            aligned_scores = [0.5] * len(passing)

        ranked = _scorer.rank_outfits(passing, filters, aligned_scores, preference)
        above_threshold = [o for o in ranked if o.get("final_score", 0) >= _MIN_SCORE_THRESHOLD]

        # Handle pagination
        page = session.current_page
        start = (page - 1) * max_results
        page_outfits = above_threshold[start: start + max_results]

        explanation = ""
        if page_outfits:
            top = page_outfits[0]
            explanation = await _explainer.explain_result(
                top, filters, session.original_prompt,
                top.get("prompt_score", 0.5),
            )

        return PromptSearchResult(
            outfits=page_outfits,
            parsed_prompt=session.current_parsed,
            filters=filters,
            explanation=explanation,
            session_id=session_id,
            total_candidates=len(candidate_outfits),
        )

    # ------------------------------------------------------------------
    # Debug / utility
    # ------------------------------------------------------------------

    async def parse_only(self, prompt: str) -> ParsedPrompt:
        """Return the ParsedPrompt without running the full search pipeline."""
        return await _parser.parse(prompt)

    def get_suggestions(self) -> List[str]:
        """Return a handful of example prompts."""
        return [
            "Something blue for a summer wedding",
            "A smart work outfit in navy or grey",
            "Casual chic for a weekend brunch",
            "All-black edgy look for a concert",
            "Romantic floral dress for a date night",
            "Comfortable travel outfit in neutral tones",
            "Formal gala look in burgundy or emerald",
            "Minimalist everyday outfit for the office",
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_hard_filters(
        self,
        outfits: List[Dict[str, Any]],
        filters: WardrobeFilters,
    ) -> List[Dict[str, Any]]:
        """Remove outfits that fail hard exclusion rules."""
        if not filters.excluded_types:
            return outfits
        return [o for o in outfits if _scorer.passes_hard_filters(o, filters)]

    def _identify_missing_aspects(
        self,
        parsed: ParsedPrompt,
        outfits: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Return a list of aspects from the prompt that none of the top outfits satisfy.
        """
        missing: List[str] = []

        if not outfits:
            if parsed.colors:
                missing.append(f"color: {', '.join(parsed.colors)}")
            if parsed.occasion:
                missing.append(f"occasion: {parsed.occasion}")
            return missing

        top = outfits[0]

        # Check color
        if parsed.colors:
            top_colors = [c.lower() for c in (top.get("dominant_colors") or [])]
            if not any(c in top_colors for c in parsed.colors):
                missing.append(f"color: {', '.join(parsed.colors)}")

        # Check occasion
        if parsed.occasion:
            tags = [t.lower() for t in (top.get("occasion_tags") or [])]
            if parsed.occasion.lower() not in tags:
                missing.append(f"occasion: {parsed.occasion}")

        return missing
