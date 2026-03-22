"""Unit tests for OutfitConversationManager."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.api.services.outfit_conversation import (
    OutfitConversationManager,
    SearchSession,
    SESSION_TTL_MINUTES,
)
from src.layer4_llm.prompt_parser import ParsedPrompt


def _parsed(**kwargs) -> ParsedPrompt:
    defaults = dict(raw_prompt="test", confidence=0.8)
    defaults.update(kwargs)
    return ParsedPrompt(**defaults)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    def setup_method(self):
        self.mgr = OutfitConversationManager()

    def test_create_returns_id(self):
        sid = self.mgr.create_session("user1", "blue dress", _parsed())
        assert isinstance(sid, str) and len(sid) > 0

    def test_get_session_returns_session(self):
        sid = self.mgr.create_session("user1", "blue dress", _parsed())
        s = self.mgr.get_session(sid)
        assert s is not None
        assert s.session_id == sid

    def test_get_missing_session_returns_none(self):
        assert self.mgr.get_session("non-existent") is None

    def test_close_session_removes_it(self):
        sid = self.mgr.create_session("user1", "blue dress", _parsed())
        self.mgr.close_session(sid)
        assert self.mgr.get_session(sid) is None

    def test_expired_session_returns_none(self):
        sid = self.mgr.create_session("user1", "blue dress", _parsed())
        # Manually expire
        self.mgr._sessions[sid].last_active = (
            datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES + 1)
        )
        assert self.mgr.get_session(sid) is None

    def test_active_session_count(self):
        self.mgr.create_session("u1", "p1", _parsed())
        self.mgr.create_session("u2", "p2", _parsed())
        assert self.mgr.active_session_count == 2

    def test_expired_sessions_are_evicted(self):
        sid = self.mgr.create_session("u1", "p1", _parsed())
        self.mgr._sessions[sid].last_active = (
            datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES + 1)
        )
        assert self.mgr.active_session_count == 0  # triggers eviction


# ---------------------------------------------------------------------------
# Refinement — formality delta
# ---------------------------------------------------------------------------

class TestFormalityRefinement:
    def setup_method(self):
        self.mgr = OutfitConversationManager()

    def _make_session(self, **parsed_kwargs) -> str:
        return self.mgr.create_session("u1", "test", _parsed(**parsed_kwargs))

    def test_more_formal_raises_floor(self):
        sid = self._make_session(formality_range=(0.5, 0.75))
        self.mgr.apply_refinement(sid, "more formal")
        s = self.mgr.get_session(sid)
        assert s.current_parsed.formality_range[0] > 0.5

    def test_less_formal_lowers_floor(self):
        sid = self._make_session(formality_range=(0.5, 0.75))
        self.mgr.apply_refinement(sid, "less formal")
        s = self.mgr.get_session(sid)
        assert s.current_parsed.formality_range[0] < 0.5

    def test_formality_clamped_at_1(self):
        sid = self._make_session(formality_range=(0.9, 1.0))
        self.mgr.apply_refinement(sid, "more formal")
        s = self.mgr.get_session(sid)
        assert s.current_parsed.formality_range[1] <= 1.0

    def test_formality_clamped_at_0(self):
        sid = self._make_session(formality_range=(0.0, 0.1))
        self.mgr.apply_refinement(sid, "less formal")
        s = self.mgr.get_session(sid)
        assert s.current_parsed.formality_range[0] >= 0.0


# ---------------------------------------------------------------------------
# Refinement — pagination
# ---------------------------------------------------------------------------

class TestPaginationRefinement:
    def setup_method(self):
        self.mgr = OutfitConversationManager()

    def test_show_me_more_increments_page(self):
        sid = self.mgr.create_session("u1", "test", _parsed())
        assert self.mgr.get_session(sid).current_page == 1
        self.mgr.apply_refinement(sid, "show me more")
        assert self.mgr.get_session(sid).current_page == 2

    def test_next_page_increments_page(self):
        sid = self.mgr.create_session("u1", "test", _parsed())
        self.mgr.apply_refinement(sid, "next page")
        assert self.mgr.get_session(sid).current_page == 2


# ---------------------------------------------------------------------------
# Refinement — mode switch
# ---------------------------------------------------------------------------

class TestModeSwitching:
    def setup_method(self):
        self.mgr = OutfitConversationManager()

    def test_gap_analysis_switches_mode(self):
        sid = self.mgr.create_session("u1", "test", _parsed())
        self.mgr.apply_refinement(sid, "gap analysis")
        assert self.mgr.get_session(sid).mode == "gap_analysis"

    def test_what_am_i_missing_switches_mode(self):
        sid = self.mgr.create_session("u1", "test", _parsed())
        self.mgr.apply_refinement(sid, "what am i missing")
        assert self.mgr.get_session(sid).mode == "gap_analysis"

    def test_mix_and_match_switches_mode(self):
        sid = self.mgr.create_session("u1", "test", _parsed())
        self.mgr.apply_refinement(sid, "mix and match")
        assert self.mgr.get_session(sid).mode == "mix_and_match"


# ---------------------------------------------------------------------------
# Refinement — style exclusion
# ---------------------------------------------------------------------------

class TestStyleExclusion:
    def setup_method(self):
        self.mgr = OutfitConversationManager()

    def test_dislike_style_adds_to_excluded(self):
        sid = self.mgr.create_session(
            "u1", "test", _parsed(styles=["bohemian"])
        )
        self.mgr.apply_refinement(sid, "i don't like this style")
        s = self.mgr.get_session(sid)
        assert "bohemian" in s.excluded_styles


# ---------------------------------------------------------------------------
# Refinement — delta merge
# ---------------------------------------------------------------------------

class TestDeltaMerge:
    def setup_method(self):
        self.mgr = OutfitConversationManager()

    def test_new_color_replaces_old(self):
        sid = self.mgr.create_session(
            "u1", "test", _parsed(colors=["blue"])
        )
        delta = _parsed(colors=["red"])
        self.mgr.apply_refinement(sid, "actually red", delta)
        s = self.mgr.get_session(sid)
        assert "red" in s.current_parsed.colors

    def test_excluded_types_additive(self):
        sid = self.mgr.create_session(
            "u1", "test", _parsed(excluded_types=["skirt"])
        )
        delta = _parsed(excluded_types=["heels"])
        self.mgr.apply_refinement(sid, "without heels", delta)
        s = self.mgr.get_session(sid)
        assert "skirt" in s.current_parsed.excluded_types
        assert "heels" in s.current_parsed.excluded_types

    def test_turns_recorded(self):
        sid = self.mgr.create_session("u1", "test", _parsed())
        self.mgr.apply_refinement(sid, "more formal")
        self.mgr.apply_refinement(sid, "show me more")
        s = self.mgr.get_session(sid)
        assert len(s.turns) == 2

    def test_missing_session_returns_none(self):
        result = self.mgr.apply_refinement("missing-id", "more formal")
        assert result is None
