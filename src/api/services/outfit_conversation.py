"""
Outfit Search Conversation — in-memory session manager
=======================================================

Allows a user to iteratively refine an outfit search using natural language.

Each session stores:
  • The original prompt
  • All subsequent refinements (deltas)
  • The merged ParsedPrompt (cumulative state)
  • Special intents: "more formal", "show me more", "without heels", …

Sessions are ephemeral (in-memory only, no DB persistence).
A session expires after ``SESSION_TTL_MINUTES`` of inactivity.

Public API:
    mgr = OutfitConversationManager()
    session_id = mgr.create_session(user_id, original_prompt, parsed)
    session    = mgr.get_session(session_id)
    session    = mgr.apply_refinement(session_id, refinement_text, delta_parsed)
    mgr.close_session(session_id)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from src.core import get_logger
from src.layer4_llm.prompt_parser import ParsedPrompt

logger = get_logger(__name__)

SESSION_TTL_MINUTES = 60  # sessions expire after 1 h of inactivity

# Special refinement intents
_REFINEMENT_PATTERNS: Dict[str, Dict] = {
    "more formal":     {"formality_delta": +0.15},
    "less formal":     {"formality_delta": -0.15},
    "more casual":     {"formality_delta": -0.20},
    "show me more":    {"paginate": True},
    "next page":       {"paginate": True},
    "i don't like this style":  {"exclude_current_styles": True},
    "gap analysis":    {"switch_mode": "gap_analysis"},
    "what am i missing":        {"switch_mode": "gap_analysis"},
    "mix and match":   {"switch_mode": "mix_and_match"},
}


@dataclass
class ConversationTurn:
    """One turn in the conversation (user input + what changed)."""
    user_text: str
    applied_delta: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SearchSession:
    """Active outfit search conversation session."""
    session_id: str
    user_id: str
    original_prompt: str
    current_parsed: ParsedPrompt          # cumulative state
    turns: List[ConversationTurn] = field(default_factory=list)
    current_page: int = 1
    mode: str = "search"                  # "search" | "gap_analysis" | "mix_and_match"
    excluded_styles: List[str] = field(default_factory=list)
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_expired(self) -> bool:
        delta = datetime.now(timezone.utc) - self.last_active
        return delta > timedelta(minutes=SESSION_TTL_MINUTES)

    def touch(self) -> None:
        self.last_active = datetime.now(timezone.utc)


class OutfitConversationManager:
    """
    Manages ephemeral outfit search sessions.

    Responsibilities:
      • Create / retrieve / close sessions
      • Apply refinement deltas to the cumulative ParsedPrompt
      • Detect special intents (pagination, mode switch, exclusions)
      • Evict expired sessions lazily
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SearchSession] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self,
        user_id: str,
        original_prompt: str,
        parsed: ParsedPrompt,
    ) -> str:
        """Create a new session and return its ID."""
        self._evict_expired()
        session_id = str(uuid4())
        self._sessions[session_id] = SearchSession(
            session_id=session_id,
            user_id=user_id,
            original_prompt=original_prompt,
            current_parsed=parsed,
        )
        logger.debug("Created session %s for user %s", session_id, user_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[SearchSession]:
        """Return session by ID, or None if missing / expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired():
            del self._sessions[session_id]
            return None
        session.touch()
        return session

    def close_session(self, session_id: str) -> None:
        """Explicitly remove a session."""
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def apply_refinement(
        self,
        session_id: str,
        refinement_text: str,
        delta_parsed: Optional[ParsedPrompt] = None,
    ) -> Optional[SearchSession]:
        """
        Apply a user refinement to the existing session state.

        If *delta_parsed* is provided it is merged into the session's
        ``current_parsed``.  Either way, any detected special intent
        (pagination, mode switch, style exclusion) is applied first.

        Returns the updated session or None if the session is not found.
        """
        session = self.get_session(session_id)
        if session is None:
            return None

        lower = refinement_text.lower()
        delta: Dict = {}

        # Detect special intents
        for pattern, effect in _REFINEMENT_PATTERNS.items():
            if pattern in lower:
                delta.update(effect)
                break

        # Apply special effects
        if delta.get("paginate"):
            session.current_page += 1

        if "switch_mode" in delta:
            session.mode = delta["switch_mode"]

        if delta.get("exclude_current_styles"):
            session.excluded_styles += session.current_parsed.styles

        if "formality_delta" in delta:
            lo, hi = session.current_parsed.formality_range
            shift = delta["formality_delta"]
            new_lo = max(0.0, min(1.0, lo + shift))
            new_hi = max(0.0, min(1.0, hi + shift))
            # Ensure min < max
            if new_lo > new_hi:
                new_lo, new_hi = new_hi, new_lo
            session.current_parsed.formality_range = (new_lo, new_hi)

        # Merge delta_parsed if provided
        if delta_parsed:
            self._merge_parsed(session.current_parsed, delta_parsed)

        # Record this turn
        session.turns.append(ConversationTurn(
            user_text=refinement_text,
            applied_delta=delta,
        ))
        session.touch()
        return session

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_parsed(self, base: ParsedPrompt, delta: ParsedPrompt) -> None:
        """
        Merge *delta* into *base* in-place.

        Rules:
          • Non-empty lists in delta overwrite base
          • Non-None scalars in delta overwrite base
          • excluded_types and excluded_styles are additive
          • confidence takes the max
        """
        if delta.colors:
            base.colors = delta.colors
        if delta.occasion:
            base.occasion = delta.occasion
        if delta.formality_range != (0.0, 1.0):
            base.formality_range = delta.formality_range
        if delta.styles:
            base.styles = delta.styles
        if delta.season_hint:
            base.season_hint = delta.season_hint
        if delta.mood:
            base.mood = delta.mood
        # Additive lists
        base.excluded_types = list(set(base.excluded_types) | set(delta.excluded_types))
        base.anchor_pieces = list(set(base.anchor_pieces) | set(delta.anchor_pieces))
        base.confidence = max(base.confidence, delta.confidence)

    def _evict_expired(self) -> None:
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.debug("Evicted %d expired sessions", len(expired))

    # ------------------------------------------------------------------
    # Stats (for tests / monitoring)
    # ------------------------------------------------------------------

    @property
    def active_session_count(self) -> int:
        self._evict_expired()
        return len(self._sessions)
