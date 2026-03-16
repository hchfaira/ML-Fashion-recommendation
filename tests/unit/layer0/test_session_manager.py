"""
Unit tests for Layer 0 — Import Session Manager
=================================================

Tests cover:
- Session creation and auto-acceptance of READY garments
- get_session / get_session_report
- accept_garment / bulk_accept_ready
- reject_garment
- correct_garment (attribute corrections)
- mark_for_resubmit
- finalise — returns only accepted/corrected records
- Session expiry
- cleanup_expired
- _get_record with invalid IDs
"""

import time
import uuid
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from src.layer0_segmentation.session_manager import (
    GarmentDecision,
    ImportSession,
    ImportSessionManager,
    SessionState,
)
from src.layer0_segmentation.quality_checker import (
    GarmentReport,
    GarmentStatus,
    QualityChecker,
    QualityThresholds,
)
from src.layer0_segmentation.models import ExtractedGarment, BoundingBox
from src.layer0_segmentation.taxonomy import GarmentCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_garment(
    category=GarmentCategory.TOPS,
    label="shirt",
    confidence=0.90,
    area=5000,
    mask_offset: int = 0,
) -> ExtractedGarment:
    """Build a minimal ExtractedGarment for testing.

    ``mask_offset`` tiles the mask horizontally so multiple garments
    created in the same test do not overlap each other.
    """
    img = Image.new("RGBA", (512, 512), (200, 180, 160, 255))
    mask = np.zeros((512, 512), dtype=np.uint8)
    col_start = (mask_offset * 170) % 512
    col_end = min(col_start + 80, 512)
    mask[20:100, col_start:col_end] = 255
    return ExtractedGarment(
        image=img,
        category=category,
        label=label,
        confidence=confidence,
        mask=mask,
        area=area,
    )


def _manager_with_session(n_garments=3, confidence=0.90):
    """Create a manager and a session with n good non-overlapping garments."""
    manager = ImportSessionManager()
    garments = [_make_garment(confidence=confidence, mask_offset=i) for i in range(n_garments)]
    session = manager.create_session(garments)
    return manager, session


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------

class TestSessionCreation:
    def test_session_id_is_uuid(self):
        manager = ImportSessionManager()
        session = manager.create_session([_make_garment()])
        assert len(session.session_id) == 36    # UUID4 string length

    def test_session_contains_correct_number_of_records(self):
        manager, session = _manager_with_session(n_garments=4)
        assert len(session.garment_records) == 4

    def test_good_garments_are_auto_accepted(self):
        manager, session = _manager_with_session(n_garments=3, confidence=0.92)
        accepted = session.accepted_records
        # All high-confidence garments should be auto-accepted
        assert len(accepted) == 3

    def test_failed_garments_are_pending(self):
        manager = ImportSessionManager()
        # Very small accessory → FAILED in quality checker → stays PENDING in session
        garments = [
            _make_garment(category=GarmentCategory.ACCESSORIES, area=10, label="bracelet")
        ]
        session = manager.create_session(garments)
        pending = session.pending_records
        # Failed / needs_review garments should not be auto-accepted
        assert len(pending) >= 1

    def test_session_state_is_pending(self):
        manager, session = _manager_with_session()
        assert session.state == SessionState.PENDING

    def test_session_stored_in_manager(self):
        manager, session = _manager_with_session()
        retrieved = manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id


# ---------------------------------------------------------------------------
# get_session / get_session_report
# ---------------------------------------------------------------------------

class TestGetSession:
    def test_get_session_returns_correct_session(self):
        manager, session = _manager_with_session()
        s = manager.get_session(session.session_id)
        assert s.session_id == session.session_id

    def test_get_session_nonexistent_returns_none(self):
        manager = ImportSessionManager()
        assert manager.get_session("does-not-exist") is None

    def test_get_session_report_returns_dict(self):
        manager, session = _manager_with_session(n_garments=2)
        report = manager.get_session_report(session.session_id)
        assert isinstance(report, dict)
        assert "session_id" in report
        assert "garments" in report

    def test_get_session_report_nonexistent_returns_none(self):
        manager = ImportSessionManager()
        assert manager.get_session_report("ghost") is None

    def test_get_session_report_garment_count_matches(self):
        manager, session = _manager_with_session(n_garments=3)
        report = manager.get_session_report(session.session_id)
        assert len(report["garments"]) == 3


# ---------------------------------------------------------------------------
# accept_garment
# ---------------------------------------------------------------------------

class TestAcceptGarment:
    def test_accept_specific_garment(self):
        manager = ImportSessionManager()
        garments = [
            _make_garment(category=GarmentCategory.ACCESSORIES, area=10, label="bracelet")
        ]
        session = manager.create_session(garments)
        sgid = list(session.garment_records.keys())[0]

        ok = manager.accept_garment(session.session_id, sgid)
        assert ok is True
        assert session.garment_records[sgid].decision == GarmentDecision.ACCEPTED

    def test_accept_nonexistent_garment_returns_false(self):
        manager, session = _manager_with_session()
        ok = manager.accept_garment(session.session_id, "nonexistent-id")
        assert ok is False

    def test_accept_on_expired_session_returns_false(self):
        manager = ImportSessionManager()
        session = manager.create_session([_make_garment()], ttl_minutes=-1)
        sgid = list(session.garment_records.keys())[0]
        ok = manager.accept_garment(session.session_id, sgid)
        assert ok is False

    def test_accept_updates_decided_at(self):
        manager = ImportSessionManager()
        garments = [_make_garment(category=GarmentCategory.ACCESSORIES, area=10)]
        session = manager.create_session(garments)
        sgid = list(session.garment_records.keys())[0]
        manager.accept_garment(session.session_id, sgid)
        assert session.garment_records[sgid].decided_at is not None


# ---------------------------------------------------------------------------
# bulk_accept_ready
# ---------------------------------------------------------------------------

class TestBulkAcceptReady:
    def test_bulk_accept_accepts_all_ready_pending(self):
        manager = ImportSessionManager()
        # Mix: 3 non-overlapping READY garments + 1 small FAILED accessory
        garments = [_make_garment(mask_offset=i) for i in range(3)]
        garments.append(_make_garment(category=GarmentCategory.ACCESSORIES, area=10, mask_offset=3))
        session = manager.create_session(garments)

        # Manually reset one READY to PENDING to test bulk accept
        for sgid, record in session.garment_records.items():
            if record.report.status == GarmentStatus.READY:
                record.decision = GarmentDecision.PENDING
                record.decided_at = None
                first_ready_id = sgid
                break

        count = manager.bulk_accept_ready(session.session_id)
        assert count >= 1
        assert session.garment_records[first_ready_id].decision == GarmentDecision.ACCEPTED

    def test_bulk_accept_nonexistent_session_returns_zero(self):
        manager = ImportSessionManager()
        assert manager.bulk_accept_ready("ghost") == 0


# ---------------------------------------------------------------------------
# reject_garment
# ---------------------------------------------------------------------------

class TestRejectGarment:
    def test_reject_garment(self):
        manager, session = _manager_with_session(n_garments=2)
        sgid = list(session.garment_records.keys())[0]
        ok = manager.reject_garment(session.session_id, sgid)
        assert ok is True
        assert session.garment_records[sgid].decision == GarmentDecision.REJECTED

    def test_reject_nonexistent_garment_returns_false(self):
        manager, session = _manager_with_session()
        ok = manager.reject_garment(session.session_id, "bad-id")
        assert ok is False

    def test_rejected_not_in_accepted_records(self):
        manager, session = _manager_with_session(n_garments=2)
        sgid = list(session.garment_records.keys())[0]
        manager.reject_garment(session.session_id, sgid)
        accepted_ids = [r.session_garment_id for r in session.accepted_records]
        assert sgid not in accepted_ids


# ---------------------------------------------------------------------------
# correct_garment
# ---------------------------------------------------------------------------

class TestCorrectGarment:
    def test_correct_applies_corrections(self):
        manager, session = _manager_with_session()
        sgid = list(session.garment_records.keys())[0]
        ok = manager.correct_garment(
            session.session_id, sgid, {"color": "ivory", "category": "blouse"}
        )
        assert ok is True
        record = session.garment_records[sgid]
        assert record.user_corrections["color"] == "ivory"
        assert record.user_corrections["category"] == "blouse"

    def test_correct_sets_decision_to_corrected(self):
        manager, session = _manager_with_session()
        sgid = list(session.garment_records.keys())[0]
        manager.correct_garment(session.session_id, sgid, {"color": "beige"})
        assert session.garment_records[sgid].decision == GarmentDecision.CORRECTED

    def test_multiple_corrections_are_merged(self):
        manager, session = _manager_with_session()
        sgid = list(session.garment_records.keys())[0]
        manager.correct_garment(session.session_id, sgid, {"color": "red"})
        manager.correct_garment(session.session_id, sgid, {"material": "silk"})
        record = session.garment_records[sgid]
        assert "color" in record.user_corrections
        assert "material" in record.user_corrections

    def test_correct_nonexistent_garment_returns_false(self):
        manager, session = _manager_with_session()
        ok = manager.correct_garment(session.session_id, "bad-id", {"color": "blue"})
        assert ok is False


# ---------------------------------------------------------------------------
# mark_for_resubmit
# ---------------------------------------------------------------------------

class TestMarkForResubmit:
    def test_mark_for_resubmit(self):
        manager, session = _manager_with_session()
        sgid = list(session.garment_records.keys())[0]
        ok = manager.mark_for_resubmit(session.session_id, sgid)
        assert ok is True
        assert session.garment_records[sgid].decision == GarmentDecision.RESUBMIT

    def test_resubmit_not_in_accepted(self):
        manager, session = _manager_with_session()
        sgid = list(session.garment_records.keys())[0]
        manager.mark_for_resubmit(session.session_id, sgid)
        accepted_ids = [r.session_garment_id for r in session.accepted_records]
        assert sgid not in accepted_ids


# ---------------------------------------------------------------------------
# finalise
# ---------------------------------------------------------------------------

class TestFinalise:
    def test_finalise_returns_accepted_records(self):
        manager, session = _manager_with_session(n_garments=3, confidence=0.92)
        records = manager.finalise(session.session_id)
        assert len(records) == 3

    def test_finalise_excludes_rejected(self):
        manager, session = _manager_with_session(n_garments=3, confidence=0.92)
        sgid = list(session.garment_records.keys())[0]
        manager.reject_garment(session.session_id, sgid)
        records = manager.finalise(session.session_id)
        result_ids = [r.session_garment_id for r in records]
        assert sgid not in result_ids

    def test_finalise_includes_corrected(self):
        manager, session = _manager_with_session(n_garments=2, confidence=0.92)
        sgid = list(session.garment_records.keys())[0]
        manager.correct_garment(session.session_id, sgid, {"color": "navy"})
        records = manager.finalise(session.session_id)
        result_ids = [r.session_garment_id for r in records]
        assert sgid in result_ids

    def test_finalise_excludes_resubmit(self):
        manager, session = _manager_with_session(n_garments=2, confidence=0.92)
        sgid = list(session.garment_records.keys())[0]
        manager.mark_for_resubmit(session.session_id, sgid)
        records = manager.finalise(session.session_id)
        result_ids = [r.session_garment_id for r in records]
        assert sgid not in result_ids

    def test_finalise_sets_session_state_to_finalised(self):
        manager, session = _manager_with_session()
        manager.finalise(session.session_id)
        assert session.state == SessionState.FINALISED

    def test_finalise_nonexistent_session_returns_empty(self):
        manager = ImportSessionManager()
        records = manager.finalise("ghost")
        assert records == []

    def test_user_corrections_preserved_after_finalise(self):
        manager, session = _manager_with_session(n_garments=1, confidence=0.92)
        sgid = list(session.garment_records.keys())[0]
        manager.correct_garment(session.session_id, sgid, {"color": "ivory"})
        records = manager.finalise(session.session_id)
        assert records[0].user_corrections["color"] == "ivory"


# ---------------------------------------------------------------------------
# Session expiry
# ---------------------------------------------------------------------------

class TestSessionExpiry:
    def test_session_is_not_expired_by_default(self):
        manager, session = _manager_with_session()
        assert session.is_expired is False

    def test_session_is_expired_when_ttl_negative(self):
        manager = ImportSessionManager()
        session = manager.create_session([_make_garment()], ttl_minutes=-1)
        assert session.is_expired is True

    def test_get_session_returns_none_for_expired(self):
        manager = ImportSessionManager()
        session = manager.create_session([_make_garment()], ttl_minutes=-1)
        result = manager.get_session(session.session_id)
        # Should mark as expired and return None or expired session
        # The manager returns the session but marks it expired
        if result is not None:
            assert result.state == SessionState.EXPIRED

    def test_cleanup_expired_removes_sessions(self):
        manager = ImportSessionManager()
        s1 = manager.create_session([_make_garment()], ttl_minutes=-1)
        s2 = manager.create_session([_make_garment()], ttl_minutes=60)
        removed = manager.cleanup_expired()
        assert removed >= 1
        assert manager.get_session(s2.session_id) is not None


# ---------------------------------------------------------------------------
# GarmentDecisionRecord.to_dict
# ---------------------------------------------------------------------------

class TestGarmentDecisionRecord:
    def test_to_dict_has_required_keys(self):
        manager, session = _manager_with_session(n_garments=1)
        record = list(session.garment_records.values())[0]
        d = record.to_dict()
        assert "session_garment_id" in d
        assert "decision" in d
        assert "user_corrections" in d
        assert "report" in d

    def test_apply_correction_updates_dict(self):
        manager, session = _manager_with_session(n_garments=1)
        record = list(session.garment_records.values())[0]
        record.apply_correction("color", "beige")
        d = record.to_dict()
        assert d["user_corrections"]["color"] == "beige"
        assert d["decision"] == "corrected"
