"""
Capsule Explainer — Layer 4 LLM narration for all capsule features
===================================================================

Provides plain-language narration for:
  F1 — wardrobe capsule diagnosis
  F2 — missing piece recommendations
  F3 — replacement plan verdicts
  F4 — capsule outfit descriptions
  F5 — evolution narrative

Optimisation-clé :
  • UN SEUL appel LLM batch par feature (pas un appel par pièce)
  • Cache dict[hash(input)] → narration, TTL 30 jours
  • Réutilise le _GeminiLLM pattern de outfit_improvement_explainer.py

Public API:
    explainer = CapsuleExplainer()
    analysis = await explainer.narrate_wardrobe_diagnosis(analysis_result)
    result   = await explainer.narrate_missing_pieces(missing_result)
    plan     = await explainer.narrate_replacement_plan(plan_result)
    outfits  = await explainer.narrate_capsule_outfits(outfits_result)
    evo      = await explainer.narrate_evolution(evolution_result)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.core import get_logger
from src.core.models import (
    CapsuleAnalysisResult,
    CapsuleEvolutionResult,
    CapsuleOutfitsResult,
    MissingPiecesResult,
    ReplacementPlanResult,
)

logger = get_logger(__name__)

_CACHE_TTL_DAYS = 30

# ---------------------------------------------------------------------------
# Shared Gemini backend (same pattern as outfit_improvement_explainer.py)
# ---------------------------------------------------------------------------

class _GeminiLLM:
    """Lightweight async text-completion wrapper over google.generativeai."""

    def __init__(self) -> None:
        import google.generativeai as genai  # type: ignore
        from config import get_settings
        settings = get_settings()
        genai.configure(api_key=settings.google_api_key)
        model_name = getattr(settings, "llm_model", "gemini-2.5-flash")
        self._model = genai.GenerativeModel(model_name)

    async def generate_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 400,
        temperature: float = 0.7,
    ) -> str:
        import google.generativeai as genai  # type: ignore
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
        gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_content(full_prompt, generation_config=gen_config),
        )
        try:
            return response.text.strip()
        except Exception:
            candidates = getattr(response, "candidates", [])
            if candidates:
                parts = getattr(candidates[0].content, "parts", [])
                if parts:
                    return parts[0].text.strip()
            return ""


# ---------------------------------------------------------------------------
# System messages
# ---------------------------------------------------------------------------

_SYSTEM_BASE = (
    "You are a professional personal stylist writing friendly, "
    "concise, actionable advice. Use plain language. "
    "No fashion jargon. No bullet points unless the prompt asks for them. "
    "Max 3 sentences per item unless otherwise specified."
)


class CapsuleExplainer:
    """
    Adds LLM narration to capsule analysis results.

    Each narrate_* method enriches the input result in-place and returns it.
    If the LLM is unavailable, the result is returned unchanged (graceful degradation).
    """

    def __init__(self) -> None:
        self._llm: Optional[_GeminiLLM] = None
        self._cache: Dict[str, Dict] = {}  # key → {text, ts}

    def _get_llm(self) -> _GeminiLLM:
        if self._llm is None:
            self._llm = _GeminiLLM()
        return self._llm

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> Optional[str]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        age_days = (datetime.now(timezone.utc) - entry["ts"]).days
        if age_days > _CACHE_TTL_DAYS:
            del self._cache[key]
            return None
        return entry["text"]

    def _cache_set(self, key: str, text: str) -> None:
        self._cache[key] = {"text": text, "ts": datetime.now(timezone.utc)}

    @staticmethod
    def _hash(data: str) -> str:
        return hashlib.md5(data.encode()).hexdigest()

    # ------------------------------------------------------------------
    # F1 — Wardrobe diagnosis narration
    # ------------------------------------------------------------------

    async def narrate_wardrobe_diagnosis(
        self, analysis: CapsuleAnalysisResult
    ) -> CapsuleAnalysisResult:
        """
        Adds a plain-language overview of the capsule score to
        analysis.recommendation if it is empty.
        """
        if analysis.recommendation:
            return analysis

        cache_key = self._hash(f"diagnosis:{analysis.cohesion_score}:{analysis.total_garments}")
        cached = self._cache_get(cache_key)
        if cached:
            analysis.recommendation = cached
            return analysis

        prompt = (
            f"Capsule wardrobe score: {analysis.cohesion_score:.0f}/100. "
            f"Total garments: {analysis.total_garments}. "
            f"Key pieces: {len(analysis.key_pieces)}. "
            f"Orphan pieces: {len(analysis.orphan_pieces)}. "
            f"Color cohesion: {analysis.color_cohesion_score:.0f}/100. "
            f"Profile: {analysis.capsule_profile}.\n\n"
            "In 2-3 sentences, explain what this score means for the user's wardrobe "
            "and the single most important thing they should do next."
        )
        try:
            text = await self._get_llm().generate_completion(prompt, _SYSTEM_BASE, max_tokens=120)
            analysis.recommendation = text
            self._cache_set(cache_key, text)
        except Exception as exc:
            logger.warning("CapsuleExplainer.narrate_wardrobe_diagnosis failed: %s", exc)

        return analysis

    # ------------------------------------------------------------------
    # F2 — Missing pieces narration (ONE batch call)
    # ------------------------------------------------------------------

    async def narrate_missing_pieces(
        self, result: MissingPiecesResult
    ) -> MissingPiecesResult:
        """
        Adds llm_narration to each MissingPieceRecommendation via a
        single batch LLM prompt.
        """
        if not result.recommendations:
            return result

        cache_key = self._hash(
            json.dumps([r.description for r in result.recommendations], sort_keys=True)
        )
        cached = self._cache_get(cache_key)
        if cached:
            narrations = json.loads(cached)
            for rec, narration in zip(result.recommendations, narrations):
                rec.llm_narration = narration
            return result

        # Build one numbered prompt
        pieces_text = "\n".join(
            f"{i+1}. {r.description} ({r.category}) — impact: {r.impact_outfits} new outfits. "
            f"Reason: {r.reason}. Profile note: {r.profile_note or 'none'}."
            for i, r in enumerate(result.recommendations)
        )
        prompt = (
            f"These are the top {len(result.recommendations)} pieces missing from this user's capsule wardrobe:\n"
            f"{pieces_text}\n\n"
            "For each numbered piece, write ONE sentence of warm, encouraging advice "
            "explaining why this piece is a smart addition. "
            "Return exactly a JSON array of strings, one per piece."
        )
        try:
            raw = await self._get_llm().generate_completion(prompt, _SYSTEM_BASE, max_tokens=500)
            # Parse JSON array from response
            narrations = self._parse_json_array(raw, len(result.recommendations))
            for rec, narration in zip(result.recommendations, narrations):
                rec.llm_narration = narration
            self._cache_set(cache_key, json.dumps(narrations))
        except Exception as exc:
            logger.warning("CapsuleExplainer.narrate_missing_pieces failed: %s", exc)

        return result

    # ------------------------------------------------------------------
    # F3 — Replacement plan narration (ONE batch call)
    # ------------------------------------------------------------------

    async def narrate_replacement_plan(
        self, plan: ReplacementPlanResult
    ) -> ReplacementPlanResult:
        """Adds llm_narration to each ReplacementVerdict."""
        if not plan.verdicts:
            return plan

        cache_key = self._hash(
            json.dumps(
                [{"keep": v.garment_keep_id, "remove": v.garment_remove_id} for v in plan.verdicts],
                sort_keys=True,
            )
        )
        cached = self._cache_get(cache_key)
        if cached:
            narrations = json.loads(cached)
            for verdict, narration in zip(plan.verdicts, narrations):
                verdict.llm_narration = narration
            return plan

        verdicts_text = "\n".join(
            f"{i+1}. Keep garment {v.garment_keep_id}, remove {v.garment_remove_id}. "
            f"Versatility gain: {v.versatility_gain:.2f}. Confidence: {v.confidence:.0%}. "
            f"Timing: {v.transition_timing}."
            for i, v in enumerate(plan.verdicts)
        )
        prompt = (
            f"Replacement plan for a capsule wardrobe:\n{verdicts_text}\n\n"
            "For each numbered verdict, write ONE sentence that explains the recommendation "
            "in a friendly, motivating way. "
            "Return exactly a JSON array of strings."
        )
        try:
            raw = await self._get_llm().generate_completion(prompt, _SYSTEM_BASE, max_tokens=400)
            narrations = self._parse_json_array(raw, len(plan.verdicts))
            for verdict, narration in zip(plan.verdicts, narrations):
                verdict.llm_narration = narration
            self._cache_set(cache_key, json.dumps(narrations))
        except Exception as exc:
            logger.warning("CapsuleExplainer.narrate_replacement_plan failed: %s", exc)

        return plan

    # ------------------------------------------------------------------
    # F4 — Capsule outfit narration (ONE batch call)
    # ------------------------------------------------------------------

    async def narrate_capsule_outfits(
        self, result: CapsuleOutfitsResult, top_n: int = 5
    ) -> CapsuleOutfitsResult:
        """Adds llm_narration to top-N CapsuleOutfits."""
        outfits = result.outfits[:top_n]
        if not outfits:
            return result

        cache_key = self._hash(
            json.dumps([o.garment_ids for o in outfits], sort_keys=True)
        )
        cached = self._cache_get(cache_key)
        if cached:
            narrations = json.loads(cached)
            for outfit, narration in zip(outfits, narrations):
                outfit.llm_narration = narration
            return result

        outfits_text = "\n".join(
            f"{i+1}. Tier: {o.tier}. Capsule score: {o.capsule_score:.0f}/100. "
            f"Occasions: {', '.join(o.occasions)}. Key pieces: {o.pct_key_pieces:.0%}."
            for i, o in enumerate(outfits)
        )
        prompt = (
            f"Here are {len(outfits)} capsule outfits from a user's wardrobe:\n{outfits_text}\n\n"
            "For each numbered outfit, write ONE sentence describing the vibe and when to wear it. "
            "Keep it inspiring and practical. "
            "Return exactly a JSON array of strings."
        )
        try:
            raw = await self._get_llm().generate_completion(prompt, _SYSTEM_BASE, max_tokens=400)
            narrations = self._parse_json_array(raw, len(outfits))
            for outfit, narration in zip(outfits, narrations):
                outfit.llm_narration = narration
            self._cache_set(cache_key, json.dumps(narrations))
        except Exception as exc:
            logger.warning("CapsuleExplainer.narrate_capsule_outfits failed: %s", exc)

        return result

    # ------------------------------------------------------------------
    # F5 — Evolution narration
    # ------------------------------------------------------------------

    async def narrate_evolution(
        self, evolution: CapsuleEvolutionResult
    ) -> CapsuleEvolutionResult:
        """Adds a summary narrative to evolution.latest_cohesion description (via trend_direction)."""
        cache_key = self._hash(
            f"evolution:{evolution.latest_cohesion}:{evolution.trend_direction}:"
            f"{evolution.delta_score}:{evolution.predicted_weeks_to_90}"
        )
        cached = self._cache_get(cache_key)
        if cached:
            # Store narration inside trend_direction as enriched string (safe — no side-effects)
            evolution.trend_direction = cached
            return evolution

        weeks_note = (
            f"Predicted weeks to reach 90/100: {evolution.predicted_weeks_to_90}."
            if evolution.predicted_weeks_to_90 is not None
            else "No prediction available yet (need more data)."
        )
        prompt = (
            f"Capsule wardrobe evolution summary:\n"
            f"Current score: {evolution.latest_cohesion}/100.\n"
            f"Score change over tracked period: {evolution.delta_score:+.1f} points.\n"
            f"New outfits unlocked: {evolution.delta_outfits:+d}.\n"
            f"Trend: {evolution.trend_direction}.\n"
            f"{weeks_note}\n\n"
            "Write 2-3 encouraging sentences summarising the user's progress and "
            "what they should focus on next. Be warm and specific."
        )
        try:
            text = await self._get_llm().generate_completion(prompt, _SYSTEM_BASE, max_tokens=150)
            evolution.trend_direction = text
            self._cache_set(cache_key, text)
        except Exception as exc:
            logger.warning("CapsuleExplainer.narrate_evolution failed: %s", exc)

        return evolution

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_array(raw: str, expected: int) -> List[str]:
        """Extract a JSON array from the LLM response; fall back to empty strings."""
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            arr = json.loads(raw[start:end])
            if isinstance(arr, list):
                # Pad or trim to match expected length
                while len(arr) < expected:
                    arr.append("")
                return [str(x) for x in arr[:expected]]
        except (ValueError, json.JSONDecodeError):
            pass
        return [""] * expected
