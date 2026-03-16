"""
Import Session Manager — Layer 0
==================================

Manages the lifecycle of a wardrobe-import session:

1. The pipeline extracts garments and the quality checker analyses them.
2. The session manager stores the resulting :class:`ImportSessionReport`
   in memory (or Redis in production) and exposes mutation methods that
   correspond directly to the API validation endpoints.
3. Once the user has reviewed all garments, ``finalise()`` returns only
   the accepted garments — ready to be handed to Layer 1 for attribute
   extraction and then stored in the Neo4j wardrobe.

Session lifecycle
-----------------
::

    PENDING  →  (user makes decisions)  →  FINALISED
                                        ↘  EXPIRED   (TTL exceeded)

Each garment inside the session moves through its own states::

    PENDING  →  ACCEPTED   (user approved)
             ↘  REJECTED   (user dismissed)
             ↘  CORRECTED  (user edited attributes manually)
             ↘  RESUBMIT   (user wants to re-photograph this piece)

"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from .models import ExtractedGarment
from .quality_checker import (
    GarmentReport,
    GarmentStatus,
    ImportSessionReport,
    QualityChecker,
    QualityThresholds,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SessionState(str, Enum):
    PENDING   = "pending"
    FINALISED = "finalised"
    EXPIRED   = "expired"


class GarmentDecision(str, Enum):
    PENDING   = "pending"
    ACCEPTED  = "accepted"
    REJECTED  = "rejected"
    CORRECTED = "corrected"
    RESUBMIT  = "resubmit"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GarmentDecisionRecord:
    """Tracks what the user decided for one garment in the session."""
    session_garment_id: str
    report: GarmentReport
    decision: GarmentDecision = GarmentDecision.PENDING
    user_corrections: Dict[str, str] = field(default_factory=dict)
    decided_at: Optional[datetime] = None

    def apply_correction(self, attribute: str, value: str) -> None:
        self.user_corrections[attribute] = value
        self.decision = GarmentDecision.CORRECTED
        self.decided_at = datetime.utcnow()
        logger.debug(
            f"[Session] garment {self.session_garment_id}: "
            f"corrected {attribute!r} → {value!r}"
        )

    def to_dict(self) -> dict:
        return {
            "session_garment_id": self.session_garment_id,
            "decision": self.decision.value,
            "user_corrections": self.user_corrections,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "report": self.report.to_dict(),
        }


@dataclass
class ImportSession:
    """
    Full state of one import session.

    Created when a user submits a photo, persists until finalised or expired.
    """
    session_id: str
    state: SessionState = SessionState.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    ttl_minutes: int = 30
    garment_records: Dict[str, GarmentDecisionRecord] = field(default_factory=dict)
    image_quality_warnings: list = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.created_at + timedelta(minutes=self.ttl_minutes)

    @property
    def pending_records(self) -> List[GarmentDecisionRecord]:
        return [r for r in self.garment_records.values() if r.decision == GarmentDecision.PENDING]

    @property
    def accepted_records(self) -> List[GarmentDecisionRecord]:
        return [
            r for r in self.garment_records.values()
            if r.decision in (GarmentDecision.ACCEPTED, GarmentDecision.CORRECTED)
        ]

    @property
    def rejected_records(self) -> List[GarmentDecisionRecord]:
        return [r for r in self.garment_records.values() if r.decision == GarmentDecision.REJECTED]

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": (self.created_at + timedelta(minutes=self.ttl_minutes)).isoformat(),
            "total_garments": len(self.garment_records),
            "pending": len(self.pending_records),
            "accepted": len(self.accepted_records),
            "rejected": len(self.rejected_records),
        }


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class ImportSessionManager:
    """
    In-memory session store for wardrobe import sessions.

    In a production deployment this would be backed by Redis; for now
    a plain dict with TTL checks is sufficient.

    Example::

        manager = ImportSessionManager()
        session = manager.create_session(garments, source_image)
        report  = session_to_report(session)          # send to client

        # User clicks "accept" on garment abc
        manager.accept_garment(session.session_id, "abc")

        # User corrects colour of garment xyz
        manager.correct_garment(session.session_id, "xyz", {"color": "navy blue"})

        # User finalises
        ready_garments = manager.finalise(session.session_id)
        # → pass ready_garments to Layer 1
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self._sessions: Dict[str, ImportSession] = {}
        self._checker = QualityChecker(thresholds)

    # ------------------------------------------------------------------
    # Session creation
    # ------------------------------------------------------------------

    def create_session(
        self,
        garments: List[ExtractedGarment],
        source_image=None,
        ttl_minutes: int = 30,
    ) -> ImportSession:
        """
        Run quality checks and create a new import session.

        Args:
            garments:     Extracted garments from Layer 0 pipeline.
            source_image: Original PIL image (for image-level checks).
            ttl_minutes:  How long the session is valid.

        Returns:
            A new :class:`ImportSession`.
        """
        session_id = str(uuid.uuid4())
        report = self._checker.check(
            garments=garments,
            source_image=source_image,
            session_id=session_id,
        )

        session = ImportSession(
            session_id=session_id,
            ttl_minutes=ttl_minutes,
            image_quality_warnings=report.image_quality_warnings,
        )

        for idx, garment_report in enumerate(report.garment_reports):
            sgid = f"{session_id}_{idx:04d}"
            garment_report.session_garment_id = sgid
            record = GarmentDecisionRecord(
                session_garment_id=sgid,
                report=garment_report,
                # Auto-accept READY garments; keep others PENDING
                decision=(
                    GarmentDecision.ACCEPTED
                    if garment_report.status == GarmentStatus.READY
                    else GarmentDecision.PENDING
                ),
                decided_at=(
                    datetime.utcnow()
                    if garment_report.status == GarmentStatus.READY
                    else None
                ),
            )
            session.garment_records[sgid] = record

        self._sessions[session_id] = session
        logger.info(
            f"[SessionManager] created session {session_id}: "
            f"{len(garments)} garments, "
            f"{len(session.accepted_records)} auto-accepted, "
            f"{len(session.pending_records)} pending review"
        )
        return session

    # ------------------------------------------------------------------
    # Session retrieval
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[ImportSession]:
        """Return session or None if not found / expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            session.state = SessionState.EXPIRED
            logger.info(f"[SessionManager] session {session_id} expired")
            return None
        return session

    def get_session_report(self, session_id: str) -> Optional[dict]:
        """Return a serialisable summary of the session."""
        session = self.get_session(session_id)
        if session is None:
            return None
        return {
            **session.summary(),
            "garments": [r.to_dict() for r in session.garment_records.values()],
            "image_quality_warnings": [
                w.to_dict() for w in session.image_quality_warnings
            ],
        }

    # ------------------------------------------------------------------
    # User decision endpoints
    # ------------------------------------------------------------------

    def accept_garment(self, session_id: str, session_garment_id: str) -> bool:
        """Mark a garment as accepted by the user."""
        record = self._get_record(session_id, session_garment_id)
        if record is None:
            return False
        record.decision = GarmentDecision.ACCEPTED
        record.decided_at = datetime.utcnow()
        logger.debug(f"[SessionManager] accepted {session_garment_id}")
        return True

    def reject_garment(self, session_id: str, session_garment_id: str) -> bool:
        """Mark a garment as rejected (will not be added to wardrobe)."""
        record = self._get_record(session_id, session_garment_id)
        if record is None:
            return False
        record.decision = GarmentDecision.REJECTED
        record.decided_at = datetime.utcnow()
        logger.debug(f"[SessionManager] rejected {session_garment_id}")
        return True

    def correct_garment(
        self,
        session_id: str,
        session_garment_id: str,
        corrections: Dict[str, str],
    ) -> bool:
        """
        Apply manual attribute corrections to a garment.

        Args:
            corrections: Mapping of attribute name → corrected value.
                         e.g. ``{"color": "ivory", "category": "blouse"}``
        """
        record = self._get_record(session_id, session_garment_id)
        if record is None:
            return False
        for attr, value in corrections.items():
            record.apply_correction(attr, value)
        return True

    def mark_for_resubmit(self, session_id: str, session_garment_id: str) -> bool:
        """
        Mark a garment as needing a better photo.

        The garment stays in the session but won't be added to the
        wardrobe until the user re-submits a new photo for it.
        """
        record = self._get_record(session_id, session_garment_id)
        if record is None:
            return False
        record.decision = GarmentDecision.RESUBMIT
        record.decided_at = datetime.utcnow()
        logger.debug(f"[SessionManager] {session_garment_id} marked for resubmit")
        return True

    def bulk_accept_ready(self, session_id: str) -> int:
        """
        Accept all READY garments in one call.

        Returns the number of garments accepted.
        """
        session = self.get_session(session_id)
        if session is None:
            return 0
        count = 0
        for record in session.garment_records.values():
            if (
                record.decision == GarmentDecision.PENDING
                and record.report.status == GarmentStatus.READY
            ):
                record.decision = GarmentDecision.ACCEPTED
                record.decided_at = datetime.utcnow()
                count += 1
        logger.info(f"[SessionManager] bulk_accept_ready: {count} garments in {session_id}")
        return count

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalise(
        self, session_id: str
    ) -> List[GarmentDecisionRecord]:
        """
        Close the session and return accepted / corrected garment records.

        Only garments with decision ACCEPTED or CORRECTED are returned.
        Garments still PENDING are treated as rejected.

        Returns:
            List of :class:`GarmentDecisionRecord` ready for Layer 1.
        """
        session = self.get_session(session_id)
        if session is None:
            logger.warning(f"[SessionManager] finalise: session {session_id} not found")
            return []

        session.state = SessionState.FINALISED
        accepted = session.accepted_records
        logger.info(
            f"[SessionManager] finalised session {session_id}: "
            f"{len(accepted)} garments accepted, "
            f"{len(session.rejected_records)} rejected"
        )
        return accepted

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        expired = [
            sid for sid, s in self._sessions.items()
            if s.is_expired or s.state == SessionState.EXPIRED
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"[SessionManager] cleaned up {len(expired)} expired sessions")
        return len(expired)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_record(
        self, session_id: str, session_garment_id: str
    ) -> Optional[GarmentDecisionRecord]:
        session = self.get_session(session_id)
        if session is None:
            logger.warning(f"[SessionManager] session {session_id} not found")
            return None
        record = session.garment_records.get(session_garment_id)
        if record is None:
            logger.warning(
                f"[SessionManager] garment {session_garment_id} "
                f"not found in session {session_id}"
            )
        return record
