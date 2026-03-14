"""
Outfit Improvement Explainer
=============================

Bridges Layer 2 (OutfitImprover) and Layer 4 (LLM) to produce
natural-language improvement suggestions for an outfit.

Flow
----
  1. OutfitImprover (Layer 2) scores the outfit and produces:
       - OutfitDiagnosis  — weak / strong dimensions
       - AdditionSuggestions  — items from         # ── Step 3 : narrate each suggestion in parallel ─────────────
        max_tokens = _TOKEN_BUDGET.get(detail_level, 150)e wardrobe to add
       - ReplacementSuggestions — swaps that raise the score
       - PurchaseTargeted  — what to buy to fix weak dimensions
  2. OutfitImprovementExplainer (this module) takes that raw data
     plus the optional user profile (colour season / body shape)
     and asks the LLM to narrate each suggestion in plain language.
  3. The result is an ImprovementExplanation — a structured object
     ready to be rendered on the mobile UI.

Public API
----------
    explainer = OutfitImprovementExplainer()
    result = await explainer.explain_improvements(
        garments=garments,
        wardrobe=wardrobe,
        context=user_context,
        user_season=ColorSeason.AUTUMN,
        body_shape=BodyShape.PEAR,
        tone="friendly",
        detail_level="standard",
    )
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.models import Garment, UserContext
from src.core import get_logger

# ---------------------------------------------------------------------------
# Gemini-based LLM backend (text-only, no OpenAI key required)
# Uses the same GOOGLE_API_KEY + LLM_MODEL already configured in .env
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
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        import google.generativeai as genai  # type: ignore
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
        gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens or 300,
            temperature=temperature if temperature is not None else 0.7,
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_content(full_prompt, generation_config=gen_config),
        )
        # Guard against safety-filtered or empty responses
        try:
            return response.text.strip()
        except Exception:
            # Gemini may return no text when finish_reason is SAFETY (2) or OTHER
            candidates = getattr(response, "candidates", [])
            if candidates:
                parts = getattr(candidates[0].content, "parts", [])
                if parts:
                    return parts[0].text.strip()
            return ""


# Lazy imports to avoid circular dependencies at module load time.
# OutfitImprover lives in Layer 2; we import it inside methods so that
# Layer 4 → Layer 2 does not trigger the full Layer 3 initialisation
# chain during module import.
def _get_improver_class():
    from src.layer2_style.outfit_improver import OutfitImprover  # noqa: PLC0415
    return OutfitImprover

def _get_season_enum():
    from src.layer2_style.season_color_harmony import ColorSeason  # noqa: PLC0415
    return ColorSeason

def _get_shape_enum():
    from src.layer2_style.volume_balance_scorer import BodyShape  # noqa: PLC0415
    return BodyShape

# Type aliases (resolved at annotation time via TYPE_CHECKING guard)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.layer2_style.season_color_harmony import ColorSeason
    from src.layer2_style.volume_balance_scorer import BodyShape

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Technical → plain-language dimension name translation
# These are the internal scoring dimension names from Layer 2. They must
# NEVER appear verbatim in messages shown to the user.
# ---------------------------------------------------------------------------
_PLAIN_LANGUAGE: Dict[str, str] = {
    # 7-point rule
    "seven_point":             "outfit variety (mixing basics and statement pieces)",
    "7-Point Rule":            "outfit variety (mixing basics and statement pieces)",
    "Seven-Point Rule":        "outfit variety (mixing basics and statement pieces)",
    # Color
    "color_harmony":           "colour coordination",
    "Color Harmony":           "colour coordination",
    "three_color_rule":        "colour balance (not too many colours at once)",
    "Three-Color Rule":        "colour balance (not too many colours at once)",
    "season_color":            "colours that suit your complexion",
    "Season Color":            "colours that suit your complexion",
    # Proportions / volume
    "proportions":             "how well the outfit shapes your silhouette",
    "Proportions":             "how well the outfit shapes your silhouette",
    "volume_balance":          "balance between fitted and flowy pieces",
    "Volume Balance":          "balance between fitted and flowy pieces",
    # Patterns
    "pattern_mixing":          "mixing prints and textures without clashing",
    "Pattern Mixing":          "mixing prints and textures without clashing",
    # Style / design
    "design_principles":       "overall visual harmony of the look",
    "Design Principles":       "overall visual harmony of the look",
    "total_style":             "how put-together the overall look feels",
    "Total Style":             "how put-together the overall look feels",
    # Creativity
    "creativity":              "personal style and originality",
    "Creativity":              "personal style and originality",
    # Overall
    "overall":                 "overall outfit score",
    "Overall":                 "overall outfit score",
}

def _plain(technical_name: str) -> str:
    """Translate a technical scoring dimension name to plain language.

    If *technical_name* is a short key present in _PLAIN_LANGUAGE, return
    its plain-language equivalent.  Otherwise, scrub the full string of
    any residual technical jargon (score percentages, dimension names,
    rule references) so it is safe to show to a user or include in an
    LLM prompt.
    """
    import re as _re

    # 1. Direct lookup (for short dimension keys)
    if technical_name in _PLAIN_LANGUAGE:
        return _PLAIN_LANGUAGE[technical_name]

    # 2. Scrub a longer reason string
    text = technical_name

    # Replace known dimension names inside the string
    for tech, friendly in _PLAIN_LANGUAGE.items():
        text = text.replace(tech, friendly)

    # Remove parenthetical percentages like "(33%)" or "(0.33)"
    text = _re.sub(r"\s*\(\d+\.?\d*%?\)", "", text)

    # Remove stray "+XX%" or "-XX%" fragments
    text = _re.sub(r"\+?\-?\d+\.?\d*%", "", text)

    # Collapse double spaces
    text = _re.sub(r"  +", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Detail-level token budgets
# ---------------------------------------------------------------------------
_TOKEN_BUDGET: Dict[str, int] = {
    "brief":    600,
    "standard": 900,
    "detailed": 1200,
}

# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class SuggestedPiece:
    """
    One concrete piece (addition, replacement, or purchase) with a
    plain-language explanation produced by the LLM.
    """
    suggestion_type: str          # "addition" | "replacement" | "purchase"
    garment_id: Optional[str]     # None for purchase suggestions
    garment_description: str      # human-readable description
    expected_score_change: float  # ± delta on the style score
    llm_explanation: str          # natural-language reason (from LLM)
    priority: int = 0             # 1 = highest priority


@dataclass
class ImprovementExplanation:
    """
    Full improvement report for one outfit, ready for the mobile UI.

    Fields
    ------
    current_score        — outfit score before improvements
    grade                — letter grade (A/B/C/D/F)
    potential_score      — estimated score if all suggestions applied
    outfit_name          — two-word LLM-generated name (e.g., "Casual Chic")
    diagnosis_summary    — one-sentence LLM narrative of what is wrong
    weak_dimensions      — list of dimension names that are weak
    strong_dimensions    — list of dimension names that are strong
    suggested_pieces     — ranked list of concrete suggestions
    profile_note         — personalised sentence referencing user profile
    short_summary        — ultra-short text (≤ 20 words) for mobile cards
    """
    current_score: float
    grade: str
    potential_score: float
    outfit_name: str
    diagnosis_summary: str
    weak_dimensions: List[str] = field(default_factory=list)
    strong_dimensions: List[str] = field(default_factory=list)
    suggested_pieces: List[SuggestedPiece] = field(default_factory=list)
    profile_note: str = ""
    short_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_score": round(self.current_score, 3),
            "grade": self.grade,
            "potential_score": round(self.potential_score, 3),
            "outfit_name": self.outfit_name,
            "diagnosis_summary": self.diagnosis_summary,
            "weak_dimensions": self.weak_dimensions,
            "strong_dimensions": self.strong_dimensions,
            "suggested_pieces": [
                {
                    "type": p.suggestion_type,
                    "garment_id": p.garment_id,
                    "description": p.garment_description,
                    "expected_score_change": p.expected_score_change,
                    "explanation": p.llm_explanation,
                    "priority": p.priority,
                }
                for p in self.suggested_pieces
            ],
            "profile_note": self.profile_note,
            "short_summary": self.short_summary,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OutfitImprovementExplainer:
    """
    Orchestrates OutfitImprover (Layer 2) + LLMService (Layer 4) to
    produce human-friendly improvement suggestions.

    Parameters
    ----------
    max_suggestions : int
        Maximum total suggestions to include in the result (default 5).
    """

    def __init__(self, max_suggestions: int = 5) -> None:
        self._improver = _get_improver_class()()
        self._llm: Optional[_GeminiLLM] = None   # lazy — created on first use
        self._max_suggestions = max_suggestions

    def _get_llm(self) -> _GeminiLLM:
        """Return (and lazily create) the Gemini LLM backend."""
        if self._llm is None:
            self._llm = _GeminiLLM()
        return self._llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def explain_improvements(
        self,
        garments: List[Garment],
        wardrobe: List[Garment],
        context: Optional[UserContext] = None,
        user_season: Optional[ColorSeason] = None,
        body_shape: Optional[BodyShape] = None,
        tone: str = "friendly",
        detail_level: str = "standard",
    ) -> ImprovementExplanation:
        """
        Analyse an outfit, find improvements, and narrate them via LLM.

        Parameters
        ----------
        garments : list[Garment]
            The outfit to analyse.
        wardrobe : list[Garment]
            The full wardrobe to search for additions / replacements.
        context : UserContext | None
            Situational context (occasion, weather, …).
        user_season : ColorSeason | None
            User's colour season — used to personalise the narrative.
        body_shape : BodyShape | None
            User's body shape — used to personalise the narrative.
        tone : str
            LLM tone: "friendly" | "professional" | "expert" | "luxury".
        detail_level : str
            "brief" | "standard" | "detailed" — controls token budget.

        Returns
        -------
        ImprovementExplanation
        """
        if not garments:
            return ImprovementExplanation(
                current_score=0.0,
                grade="F",
                potential_score=0.0,
                outfit_name="Empty Outfit",
                diagnosis_summary="No garments provided.",
            )

        # ── Step 1 : run OutfitImprover (Layer 2) ─────────────────────
        logger.info(
            f"OutfitImprovementExplainer: analysing outfit "
            f"({len(garments)} garments, wardrobe={len(wardrobe)})"
        )
        improvement_result = self._improver.improve(garments, wardrobe, context)
        diagnosis = improvement_result.diagnosis

        # ── Step 2 : build ranked suggestion list ────────────────────
        raw_suggestions = self._collect_suggestions(improvement_result)
        raw_suggestions = raw_suggestions[: self._max_suggestions]

        # ── Step 3 : narrate each suggestion in parallel ─────────────
        max_tokens = _TOKEN_BUDGET.get(detail_level, 600)
        narration_tasks = [
            self._narrate_suggestion(s, user_season, body_shape, tone, max_tokens)
            for s in raw_suggestions
        ]
        narration_tasks.append(
            self._narrate_diagnosis(diagnosis, user_season, body_shape, tone)
        )
        narration_tasks.append(
            self._build_profile_note(user_season, body_shape, diagnosis)
        )
        narration_tasks.append(
            self._build_short_summary(diagnosis, improvement_result, tone)
        )
        narration_tasks.append(
            self._generate_outfit_name(garments, diagnosis, tone)
        )

        results = await asyncio.gather(*narration_tasks, return_exceptions=True)

        # Unpack results safely
        n = len(raw_suggestions)
        narrated_pieces: List[SuggestedPiece] = []
        for i, s in enumerate(raw_suggestions):
            explanation = results[i]
            if isinstance(explanation, Exception):
                logger.warning(f"LLM narration failed for suggestion {i}: {explanation}")
                explanation = s.get("fallback_explanation", "")
            narrated_pieces.append(SuggestedPiece(
                suggestion_type=s["type"],
                garment_id=s.get("garment_id"),
                garment_description=s["description"],
                expected_score_change=s["delta"],
                llm_explanation=str(explanation),
                priority=i + 1,
            ))

        diagnosis_summary = results[n]
        if isinstance(diagnosis_summary, Exception):
            diagnosis_summary = improvement_result.summary

        profile_note = results[n + 1]
        if isinstance(profile_note, Exception):
            profile_note = ""

        short_summary = results[n + 2]
        if isinstance(short_summary, Exception):
            short_summary = f"Score: {diagnosis.overall_score:.0%} ({diagnosis.grade})"

        outfit_name = results[n + 3]
        if isinstance(outfit_name, Exception):
            outfit_name = "Stylish Ensemble"

        # ── Step 4 : estimate potential score ─────────────────────────
        potential_score = self._estimate_potential(
            diagnosis.overall_score, narrated_pieces
        )

        return ImprovementExplanation(
            current_score=diagnosis.overall_score,
            grade=diagnosis.grade,
            potential_score=potential_score,
            outfit_name=str(outfit_name),
            diagnosis_summary=str(diagnosis_summary),
            weak_dimensions=[_plain(d["display_name"]) for d in diagnosis.weak_dimensions],
            strong_dimensions=[_plain(d["display_name"]) for d in diagnosis.strong_dimensions],
            suggested_pieces=narrated_pieces,
            profile_note=str(profile_note),
            short_summary=str(short_summary),
        )

    # ------------------------------------------------------------------
    # Private — suggestion collection
    # ------------------------------------------------------------------

    def _collect_suggestions(self, result) -> List[Dict[str, Any]]:
        """
        Flatten additions, replacements and purchases into a unified ranked list.
        Best delta first; purchases come after wardrobe suggestions.
        """
        suggestions: List[Dict[str, Any]] = []

        # 1. Additions from wardrobe
        for a in result.additions:
            suggestions.append({
                "type": "addition",
                "garment_id": a.garment_id,
                "description": a.garment_description,
                "delta": a.expected_score_change,
                "fallback_explanation": a.reason,
            })

        # 2. Replacements from wardrobe
        for r in result.replacements:
            suggestions.append({
                "type": "replacement",
                "garment_id": r.replacement_garment_id,
                "description": (
                    f"Replace '{r.original_description}' "
                    f"with '{r.replacement_description}'"
                ),
                "delta": r.expected_score_change,
                "fallback_explanation": r.reason,
            })

        # Sort wardrobe suggestions by delta (best first)
        suggestions.sort(key=lambda s: s["delta"], reverse=True)

        # 3. Purchase suggestions (append after wardrobe suggestions)
        for p in result.purchase_suggestions:
            suggestions.append({
                "type": "purchase",
                "garment_id": None,
                "description": p.description,
                "delta": p.expected_score_change,
                "fallback_explanation": p.reason,
            })

        return suggestions

    # ------------------------------------------------------------------
    # Private — LLM narrations
    # ------------------------------------------------------------------

    async def _narrate_suggestion(
        self,
        suggestion: Dict[str, Any],
        user_season: Optional[ColorSeason],
        body_shape: Optional[BodyShape],
        tone: str,
        max_tokens: int,
    ) -> str:
        """Ask the LLM to narrate one suggestion in everyday, jargon-free language."""
        # ── user profile in human-readable form ──────────────────────────
        season_labels = {
            "spring":  "warm and bright tones (spring palette)",
            "summer":  "soft and cool tones (summer palette)",
            "autumn":  "warm and earthy tones (autumn palette)",
            "winter":  "bold and high-contrast tones (winter palette)",
        }
        shape_labels = {
            "hourglass":          "hourglass figure (balanced shoulders and hips, defined waist)",
            "triangle":           "pear-shaped figure (hips wider than shoulders)",
            "inverted_triangle":  "inverted-triangle figure (broader shoulders, narrower hips)",
            "rectangle":          "rectangular figure (balanced, athletic build)",
            "oval":               "rounded figure (fuller midsection)",
            "athletic":           "athletic figure (strong, muscular proportions)",
        }
        season_val = user_season.value if user_season and hasattr(user_season, "value") else (str(user_season) if user_season else None)
        shape_val  = body_shape.value  if body_shape  and hasattr(body_shape,  "value") else (str(body_shape)  if body_shape  else None)
        season_desc = season_labels.get(season_val, season_val) if season_val else None
        shape_desc  = shape_labels.get(shape_val, shape_val)    if shape_val  else None

        profile_lines = []
        if season_desc:
            profile_lines.append(f"- Your best colours are {season_desc}.")
        if shape_desc:
            profile_lines.append(f"- Your body shape: {shape_desc}.")
        profile_block = "\n".join(profile_lines) if profile_lines else "- No specific profile provided."

        stype   = suggestion["type"]
        desc    = suggestion["description"]
        delta   = suggestion["delta"]
        fallback = suggestion.get("fallback_explanation", "")
        # Translate the fallback reason so it also stays jargon-free
        plain_reason = _plain(fallback)

        if stype == "addition":
            action_text = f"add this piece to the outfit: {desc}"
        elif stype == "replacement":
            action_text = f"make this swap: {desc}"
        else:
            action_text = f"consider buying: {desc}"

        prompt = (
            "You are a friendly personal stylist talking directly to a customer.\n"
            "CRITICAL RULES (follow strictly):\n"
            "- Use simple, everyday language a non-fashion person would use.\n"
            "- NEVER mention scores, percentages, rule names, point counts, "
            "  volume balance, seven-point rules, or any system internals.\n"
            "- Write a SHORT, PUNCHY reason: 1–2 sentences maximum (not more).\n"
            "- Each sentence MUST be complete and end with a period.\n"
            "- NO incomplete sentences, NO trailing thoughts.\n\n"
            f"### About this customer\n{profile_block}\n\n"
            f"### Suggested improvement\n"
            f"We recommend to {action_text}.\n"
            f"Why it helps: {plain_reason}\n\n"
            "In just 1–2 complete sentences, explain WHY this suits this customer personally. "
            "Reference their colours or body shape if possible. "
            "Be warm, encouraging, and practical. "
            "Keep it SHORT and punchy — don't over-explain."
        )

        try:
            return await self._get_llm().generate_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
            )
        except Exception as exc:
            logger.warning(f"LLM narration failed: {exc}")
            return plain_reason or desc

    async def _narrate_diagnosis(
        self,
        diagnosis,
        user_season: Optional[ColorSeason],
        body_shape: Optional[BodyShape],
        tone: str,
    ) -> str:
        """Generate a one-sentence plain-language diagnosis summary."""
        season_val = user_season.value if user_season and hasattr(user_season, "value") else (str(user_season) if user_season else None)
        shape_val  = body_shape.value  if body_shape  and hasattr(body_shape,  "value") else (str(body_shape)  if body_shape  else None)

        # Translate dimension names to plain language
        weak_plain   = [_plain(d["display_name"]) for d in diagnosis.weak_dimensions[:3]]
        strong_plain = [_plain(d["display_name"]) for d in diagnosis.strong_dimensions[:2]]

        profile_part = ""
        if season_val or shape_val:
            parts = []
            if season_val:
                parts.append(f"{season_val} colour palette")
            if shape_val:
                parts.append(f"{shape_val} body shape")
            profile_part = f" (keeping in mind their {' and '.join(parts)})"

        prompt = (
            "You are a friendly personal stylist. "
            "Use simple, everyday language — NO technical terms, NO score names, NO rule names.\n\n"
            f"This outfit looks pretty good overall{profile_part}.\n"
            f"Things that could be improved: {', '.join(weak_plain) if weak_plain else 'nothing major'}.\n"
            f"Things that are already working well: {', '.join(strong_plain) if strong_plain else 'the basics'}.\n\n"
            "In ONE warm, conversational sentence, summarise the state of the outfit and "
            "the single most important thing to improve. Be positive and encouraging."
        )

        try:
            return await self._get_llm().generate_completion(
                prompt=prompt,
                max_tokens=100,
                temperature=0.6,
            )
        except Exception as exc:
            logger.warning(f"LLM diagnosis narration failed: {exc}")
            if weak_plain:
                return f"Your outfit is off to a great start — focusing on {weak_plain[0]} will really elevate the look."
            return "Your outfit has a solid foundation — a few small tweaks could make it shine."

    async def _build_profile_note(
        self,
        user_season: Optional[ColorSeason],
        body_shape: Optional[BodyShape],
        diagnosis,
    ) -> str:
        """
        One personalised sentence connecting the suggestions to the user's profile.
        Returns empty string when no profile is available.
        """
        if not user_season and not body_shape:
            return ""

        season_val = user_season.value if user_season and hasattr(user_season, "value") else (str(user_season) if user_season else None)
        shape_val  = body_shape.value  if body_shape  and hasattr(body_shape,  "value") else (str(body_shape)  if body_shape  else None)

        weak_plain = [_plain(d["display_name"]) for d in diagnosis.weak_dimensions[:2]]

        prompt = (
            "You are a friendly personal stylist. "
            "Use simple, everyday language — NO technical terms or rule names.\n\n"
            f"The customer has a {season_val or 'unknown'} colour palette "
            f"and a {shape_val or 'unknown'} body shape.\n"
            f"The outfit could improve in: {', '.join(weak_plain) if weak_plain else 'a few areas'}.\n\n"
            "In ONE short sentence (max 20 words), explain WHY these improvements matter "
            "especially for this customer's colours and figure. Be warm and personal."
        )

        try:
            return await self._get_llm().generate_completion(
                prompt=prompt,
                max_tokens=60,
                temperature=0.6,
            )
        except Exception as exc:
            logger.warning(f"LLM profile note failed: {exc}")
            parts = []
            if season_val:
                parts.append(f"{season_val} colouring")
            if shape_val:
                parts.append(f"{shape_val} figure")
            return f"These tweaks are especially flattering for your {' and '.join(parts)}."

    async def _build_short_summary(
        self,
        diagnosis,
        improvement_result,
        tone: str,
    ) -> str:
        """Generate an ultra-short summary (≤ 20 words) for mobile cards — no jargon."""
        best = None
        all_suggestions = list(improvement_result.additions) + list(improvement_result.replacements)
        if all_suggestions:
            best = max(all_suggestions, key=lambda s: s.expected_score_change)

        best_hint = ""
        if best:
            item = getattr(best, "garment_description", None) or getattr(best, "replacement_description", "a new piece")
            best_hint = f" Try adding {item} to make it pop."

        prompt = (
            "You are a friendly stylist writing a one-line caption for a mobile app card. "
            "Use casual, jargon-free language. Max 20 words. No percentages, no rule names.\n\n"
            f"The outfit looks good but has room to grow.{best_hint} "
            "Write the caption now:"
        )

        try:
            return await self._get_llm().generate_completion(
                prompt=prompt,
                max_tokens=40,
                temperature=0.5,
            )
        except Exception as exc:
            logger.warning(f"LLM short summary failed: {exc}")
            return "Great start — a few easy tweaks will make this outfit really stand out!"

    async def _generate_outfit_name(
        self,
        garments: List[Garment],
        diagnosis,
        tone: str,
    ) -> str:
        """Generate a creative two-word outfit name (e.g., 'Casual Chic', 'Bold Statement')."""
        # Build a simple description of the garments
        categories = [g.attributes.category.value for g in garments]
        colors = [g.attributes.color.primary for g in garments if g.attributes.color and g.attributes.color.primary]
        
        garment_desc = ", ".join(categories[:3])
        color_desc = ", ".join(set(colors[:2]))
        
        prompt = (
            "You are a fashion stylist naming outfits for a mobile app. "
            "Generate a creative, catchy TWO-WORD outfit name only (nothing else).\n\n"
            f"Garments: {garment_desc}\n"
            f"Colours: {color_desc}\n"
            f"Vibe: {tone}\n\n"
            "Reply with ONLY the two-word name (e.g., 'Casual Chic', 'Bold Elegance', 'Street Cool'). "
            "No punctuation, no explanation:"
        )

        try:
            result = await self._get_llm().generate_completion(
                prompt=prompt,
                max_tokens=10,
                temperature=0.8,
            )
            # Clean up the result: extract only the two words
            words = result.strip().split()[:2]
            if len(words) >= 2:
                return " ".join(words)
            elif len(words) == 1:
                return f"{words[0]} Look"
            return "Stylish Ensemble"
        except Exception as exc:
            logger.warning(f"LLM outfit name generation failed: {exc}")
            return "Stylish Ensemble"

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_potential(
        current_score: float,
        pieces: List[SuggestedPiece],
    ) -> float:
        """
        Estimate the score after applying all suggestions.
        Uses diminishing returns: each successive gain is halved.
        """
        score = current_score
        for i, piece in enumerate(pieces):
            if piece.expected_score_change > 0:
                # Diminishing returns factor
                factor = 1.0 / (2 ** i)
                score += piece.expected_score_change * factor
        return round(min(1.0, score), 3)
