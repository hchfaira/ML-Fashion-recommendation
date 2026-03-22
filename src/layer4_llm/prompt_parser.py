"""
Prompt Parser — Layer 4 LLM
============================

Converts a free-text user prompt (e.g. "I want something blue for a wedding")
into a structured ParsedPrompt object that downstream layers can consume.

Features:
  • Keyword-first fast path  — covers ~70 % of common prompts for free
  • Gemini LLM fallback      — handles complex / ambiguous phrasing
  • Prompt-level validation  — rejects off-topic input
  • LRU-style cache          — keyed on normalised prompt hash (TTL 30 days)
  • Graceful degradation     — returns a broad ParsedPrompt on LLM failure

Public API:
    parser = PromptParser()
    result = await parser.parse("I want something blue for a wedding")
    # result.colors == ["blue"]
    # result.occasion == "wedding"
    # result.formality_range == (0.7, 1.0)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from src.core import get_logger

logger = get_logger(__name__)

_CACHE_TTL_DAYS = 30
_MAX_TOKENS = 400

# ---------------------------------------------------------------------------
# Known vocabulary maps (keyword fast-path)
# ---------------------------------------------------------------------------

_OCCASION_MAP: Dict[str, str] = {
    "wedding": "wedding",
    "bridal": "wedding",
    "gala": "gala",
    "ball": "gala",
    "work": "work",
    "office": "work",
    "business": "work",
    "meeting": "work",
    "interview": "work",
    "date": "date",
    "dinner": "dinner",
    "lunch": "lunch",
    "brunch": "brunch",
    "party": "party",
    "cocktail": "cocktail",
    "beach": "beach",
    "vacation": "vacation",
    "travel": "travel",
    "gym": "gym",
    "sport": "sport",
    "workout": "gym",
    "casual": "casual",
    "festival": "festival",
    "funeral": "funeral",
    "ceremony": "ceremony",
    "graduation": "graduation",
    "prom": "prom",
}

_FORMALITY_BY_OCCASION: Dict[str, Tuple[float, float]] = {
    "wedding": (0.7, 1.0),
    "gala": (0.85, 1.0),
    "prom": (0.75, 1.0),
    "graduation": (0.6, 0.9),
    "ceremony": (0.65, 0.95),
    "funeral": (0.65, 0.95),
    "work": (0.5, 0.8),
    "interview": (0.6, 0.85),
    "dinner": (0.45, 0.75),
    "cocktail": (0.55, 0.85),
    "date": (0.35, 0.65),
    "lunch": (0.3, 0.6),
    "brunch": (0.2, 0.55),
    "party": (0.3, 0.7),
    "beach": (0.0, 0.25),
    "vacation": (0.0, 0.35),
    "travel": (0.1, 0.45),
    "gym": (0.0, 0.2),
    "sport": (0.0, 0.2),
    "festival": (0.1, 0.4),
    "casual": (0.0, 0.4),
}

_STYLE_KEYWORDS: Dict[str, List[str]] = {
    "minimalist": ["minimalist", "minimal", "simple", "clean", "understated"],
    "romantic": ["romantic", "feminine", "flirty", "floral", "soft"],
    "classic": ["classic", "timeless", "traditional", "conservative"],
    "bohemian": ["boho", "bohemian", "hippie", "earthy", "free-spirited"],
    "streetwear": ["streetwear", "urban", "street", "casual cool", "hypebeast"],
    "sporty": ["sporty", "athletic", "activewear", "sport"],
    "edgy": ["edgy", "bold", "rock", "punk", "gothic", "dark"],
    "preppy": ["preppy", "collegiate", "ivy", "polo"],
    "luxury": ["luxury", "designer", "couture", "high-end", "elegant"],
    "elegant": ["elegant", "chic", "sophisticated", "refined", "polished"],
    "trendy": ["trendy", "fashion-forward", "on-trend", "modern"],
}

_SEASON_KEYWORDS: Dict[str, List[str]] = {
    "spring": ["spring", "spring/summer"],
    "summer": ["summer", "hot", "beach"],
    "autumn": ["autumn", "fall", "autumnal"],
    "winter": ["winter", "cold", "warm", "cozy", "festive"],
}

_EXCLUDED_TYPE_KEYWORDS: Dict[str, str] = {
    "no skirt": "skirt",
    "without skirt": "skirt",
    "no skirts": "skirt",
    "no dress": "dress",
    "without dress": "dress",
    "no heels": "heels",
    "without heels": "heels",
    "no tie": "tie",
    "without tie": "tie",
    "no suit": "suit",
    "without suit": "suit",
    "no jeans": "jeans",
    "without jeans": "jeans",
    "no shorts": "shorts",
    "without shorts": "shorts",
}

_OFF_TOPIC_SIGNALS = [
    "pizza", "food", "recipe", "car", "music", "movie", "game", "sport",
    "football", "bitcoin", "crypto", "invest", "weather",
]


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class ParsedPrompt:
    """Structured interpretation of a free-text outfit search prompt."""
    raw_prompt: str
    colors: List[str] = field(default_factory=list)
    occasion: Optional[str] = None
    formality_range: Tuple[float, float] = (0.0, 1.0)
    styles: List[str] = field(default_factory=list)
    season_hint: Optional[str] = None
    anchor_pieces: List[str] = field(default_factory=list)
    excluded_types: List[str] = field(default_factory=list)
    mood: Optional[str] = None
    confidence: float = 0.5
    is_off_topic: bool = False
    clarification_needed: bool = False

    def to_dict(self) -> dict:
        return {
            "raw_prompt": self.raw_prompt,
            "colors": self.colors,
            "occasion": self.occasion,
            "formality_range": list(self.formality_range),
            "styles": self.styles,
            "season_hint": self.season_hint,
            "anchor_pieces": self.anchor_pieces,
            "excluded_types": self.excluded_types,
            "mood": self.mood,
            "confidence": self.confidence,
            "is_off_topic": self.is_off_topic,
            "clarification_needed": self.clarification_needed,
        }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _CacheEntry:
    def __init__(self, result: ParsedPrompt) -> None:
        self.result = result
        self.created_at = datetime.now(timezone.utc)

    def is_valid(self) -> bool:
        return datetime.now(timezone.utc) - self.created_at < timedelta(days=_CACHE_TTL_DAYS)


# ---------------------------------------------------------------------------
# Gemini LLM backend (same singleton pattern as moodboard_explainer.py)
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
            logger.warning(f"PromptParser: Gemini unavailable — {exc}")
            self._available = False

    async def generate(self, prompt: str, max_tokens: int = _MAX_TOKENS) -> str:
        if not self._available:
            raise RuntimeError("Gemini LLM not available")
        import google.generativeai as genai  # type: ignore

        gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.2,  # Low temp for structured extraction
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_content(prompt, generation_config=gen_config),
        )
        return response.text.strip()


# ---------------------------------------------------------------------------
# PromptParser
# ---------------------------------------------------------------------------

class PromptParser:
    """
    Converts a free-text outfit search prompt into a ParsedPrompt.

    Fast path:  keyword matching  — O(n) scan, no API call
    Slow path:  Gemini LLM       — used when keyword pass confidence < 0.6
    Cache:      content-hash → ParsedPrompt, TTL 30 days
    """

    def __init__(self) -> None:
        self._llm = _GeminiLLM()
        self._cache: Dict[str, _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(self, prompt: str) -> ParsedPrompt:
        """
        Parse *prompt* and return a :class:`ParsedPrompt`.

        The result is cached by normalised-prompt hash so identical (or
        near-identical) prompts never hit the LLM twice.
        """
        key = self._hash(prompt)
        entry = self._cache.get(key)
        if entry and entry.is_valid():
            logger.debug("PromptParser cache hit")
            return entry.result

        result = await self._parse_uncached(prompt)
        self._cache[key] = _CacheEntry(result)
        return result

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _parse_uncached(self, prompt: str) -> ParsedPrompt:
        lower = prompt.lower()

        # 1. Off-topic guard
        if self._is_off_topic(lower):
            return ParsedPrompt(raw_prompt=prompt, is_off_topic=True, confidence=0.95)

        # 2. Keyword fast-path
        kw_result = self._keyword_parse(prompt, lower)

        # 3. If confidence is sufficient, skip LLM
        if kw_result.confidence >= 0.6:
            logger.debug("PromptParser: keyword fast-path (confidence=%.2f)", kw_result.confidence)
            return kw_result

        # 4. LLM enrichment
        try:
            llm_result = await self._llm_parse(prompt)
            # Merge: LLM overrides, but keyword exclusions are additive
            llm_result.excluded_types = list(
                set(llm_result.excluded_types) | set(kw_result.excluded_types)
            )
            logger.debug("PromptParser: LLM path (confidence=%.2f)", llm_result.confidence)
            return llm_result
        except Exception as exc:
            logger.warning(f"PromptParser LLM failed, using keyword result — {exc}")
            kw_result.confidence = max(kw_result.confidence, 0.4)
            return kw_result

    # ------------------------------------------------------------------
    # Off-topic detection
    # ------------------------------------------------------------------

    def _is_off_topic(self, lower: str) -> bool:
        """Return True when the prompt is clearly not about clothing."""
        clothing_signals = [
            "wear", "outfit", "dress", "shirt", "trouser", "skirt", "jacket",
            "coat", "jeans", "suit", "look", "style", "cloth", "garment",
            "shoe", "boot", "heel", "bag", "accessory", "color", "colour",
        ]
        # Occasion/event keywords also indicate a fashion context
        occasion_signals = list(_OCCASION_MAP.keys())
        has_clothing = any(s in lower for s in clothing_signals)
        has_occasion = any(s in lower for s in occasion_signals)
        has_off_topic = any(s in lower for s in _OFF_TOPIC_SIGNALS)
        if has_off_topic and not has_clothing and not has_occasion:
            return True
        # Very short prompts with no style/occasion content
        if len(lower.split()) <= 2 and not has_clothing and not has_occasion:
            return True
        return False

    # ------------------------------------------------------------------
    # Keyword fast-path
    # ------------------------------------------------------------------

    def _keyword_parse(self, prompt: str, lower: str) -> ParsedPrompt:
        result = ParsedPrompt(raw_prompt=prompt)

        # Colors — simple word scan
        result.colors = self._extract_colors(lower)

        # Occasion
        for kw, occ in _OCCASION_MAP.items():
            if kw in lower:
                result.occasion = occ
                break

        # Formality range from occasion
        if result.occasion and result.occasion in _FORMALITY_BY_OCCASION:
            result.formality_range = _FORMALITY_BY_OCCASION[result.occasion]
        elif "formal" in lower or "smart" in lower:
            result.formality_range = (0.6, 1.0)
        elif "casual" in lower or "relaxed" in lower:
            result.formality_range = (0.0, 0.45)

        # Styles
        for style, keywords in _STYLE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                result.styles.append(style)

        # Season
        for season, keywords in _SEASON_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                result.season_hint = season
                break

        # Excluded types
        for phrase, garment_type in _EXCLUDED_TYPE_KEYWORDS.items():
            if phrase in lower:
                result.excluded_types.append(garment_type)

        # Anchor pieces ("with my …")
        anchor_matches = re.findall(r"with (?:my |a |an )?([a-z ]+?)(?:\s*,|\s+and\b|\s*$)", lower)
        result.anchor_pieces = [m.strip() for m in anchor_matches if len(m.strip()) > 2]

        # Mood
        for mood_kw in ("romantic", "powerful", "playful", "comfortable", "confident", "bold"):
            if mood_kw in lower:
                result.mood = mood_kw
                break

        # Confidence heuristic: more signals → higher confidence
        signals = sum([
            bool(result.colors),
            bool(result.occasion),
            bool(result.styles),
            result.formality_range != (0.0, 1.0),
        ])
        result.confidence = min(0.5 + signals * 0.12, 0.85)

        # Flag vague prompts
        if signals <= 1 and len(lower.split()) < 4:
            result.clarification_needed = True

        return result

    def _extract_colors(self, lower: str) -> List[str]:
        """Extract color mentions from the lowercased prompt."""
        color_words = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "black", "white", "grey", "gray", "brown", "beige", "cream",
            "navy", "teal", "turquoise", "coral", "lavender", "burgundy",
            "ivory", "gold", "silver", "bronze", "nude", "rose", "lilac",
            "mint", "olive", "mustard", "cobalt", "indigo", "maroon",
            "charcoal", "tan", "camel", "blush", "peach", "emerald",
            "sage", "rust", "forest", "sky", "denim",
        ]
        found = [c for c in color_words if c in lower.split() or f" {c} " in f" {lower} "]
        # De-duplicate while preserving order
        seen: set = set()
        return [c for c in found if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _llm_parse(self, prompt: str) -> ParsedPrompt:
        system_prompt = (
            "You are a fashion AI assistant. "
            "Extract structured outfit search criteria from the user's request. "
            "Return ONLY valid JSON with these exact keys:\n"
            '  "colors": list of color strings (lowercase),\n'
            '  "occasion": string or null,\n'
            '  "formality_range": [min_float, max_float] both in [0,1],\n'
            '  "styles": list of style strings,\n'
            '  "season_hint": string or null,\n'
            '  "anchor_pieces": list of garment description strings,\n'
            '  "excluded_types": list of garment type strings,\n'
            '  "mood": string or null,\n'
            '  "confidence": float in [0,1],\n'
            '  "is_off_topic": boolean,\n'
            '  "clarification_needed": boolean\n'
            "If the prompt is not about clothing, set is_off_topic=true. "
            "Do not include markdown fences."
        )
        full_prompt = f"{system_prompt}\n\nUser request: {prompt}"
        raw = await self._llm.generate(full_prompt)

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        data = json.loads(raw)
        fr = data.get("formality_range", [0.0, 1.0])
        return ParsedPrompt(
            raw_prompt=prompt,
            colors=data.get("colors", []),
            occasion=data.get("occasion"),
            formality_range=(float(fr[0]), float(fr[1])),
            styles=data.get("styles", []),
            season_hint=data.get("season_hint"),
            anchor_pieces=data.get("anchor_pieces", []),
            excluded_types=data.get("excluded_types", []),
            mood=data.get("mood"),
            confidence=float(data.get("confidence", 0.5)),
            is_off_topic=bool(data.get("is_off_topic", False)),
            clarification_needed=bool(data.get("clarification_needed", False)),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(prompt: str) -> str:
        normalised = " ".join(prompt.lower().split())
        return hashlib.sha256(normalised.encode()).hexdigest()
