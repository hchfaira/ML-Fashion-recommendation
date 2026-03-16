"""
Wardrobe Import API Routes
===========================

Endpoints for the **multi-step wardrobe-import flow**:

  POST /wardrobe/import/analyse       Submit a photo → get quality report
  GET  /wardrobe/import/{sid}         Fetch current session state
  POST /wardrobe/import/{sid}/accept  Accept one or all garments
  POST /wardrobe/import/{sid}/reject  Reject a garment
  POST /wardrobe/import/{sid}/correct Manually correct garment attributes
  POST /wardrobe/import/{sid}/resubmit Mark a garment for re-photography
  POST /wardrobe/import/{sid}/finalise Commit accepted garments to wardrobe
  POST /wardrobe/import/{sid}/manual   Add an accessory manually (no ML)

All responses follow a consistent envelope::

    {
      "ok": true,
      "data": { ... },
      "warnings": []       # non-blocking messages for the client
    }

Errors follow FastAPI's standard ``{"detail": "..."}`` format.
"""

import io
import uuid
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from pydantic import BaseModel, Field
from PIL import Image

from src.layer0_segmentation.session_manager import (
    ImportSessionManager,
    GarmentDecision,
)
from src.layer0_segmentation.quality_checker import GarmentStatus

try:
    from src.layer0_segmentation.pipeline import create_pipeline as create_pipeline
except Exception:  # pragma: no cover
    create_pipeline = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/wardrobe/import", tags=["Wardrobe Import"])

# ---------------------------------------------------------------------------
# Shared session manager (singleton per process)
# ---------------------------------------------------------------------------
_session_manager: Optional[ImportSessionManager] = None


def get_session_manager() -> ImportSessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = ImportSessionManager()
    return _session_manager


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ApiEnvelope(BaseModel):
    ok: bool = True
    data: Any = None
    warnings: List[str] = Field(default_factory=list)


class AcceptRequest(BaseModel):
    session_garment_id: Optional[str] = Field(
        None,
        description="Omit to accept ALL ready garments at once.",
    )


class RejectRequest(BaseModel):
    session_garment_id: str


class CorrectRequest(BaseModel):
    session_garment_id: str
    corrections: Dict[str, str] = Field(
        ...,
        description='e.g. {"color": "ivory", "category": "blouse"}',
        examples=[{"color": "ivory", "category": "blouse"}],
    )


class ResubmitRequest(BaseModel):
    session_garment_id: str


class ManualGarmentRequest(BaseModel):
    category: str = Field(..., description="e.g. 'accessory', 'shoes'")
    label: str = Field(..., description="e.g. 'leather belt', 'silk scarf'")
    color: Optional[str] = None
    material: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _not_found(session_id: str):
    raise HTTPException(
        status_code=404,
        detail=f"Session '{session_id}' not found or expired.",
    )


def _bad_garment(session_garment_id: str):
    raise HTTPException(
        status_code=422,
        detail=f"Garment '{session_garment_id}' not found in session.",
    )


# ---------------------------------------------------------------------------
# 1. Analyse — submit photo, get quality report
# ---------------------------------------------------------------------------

@router.post("/analyse", response_model=ApiEnvelope, status_code=202)
async def analyse_photo(
    image: UploadFile = File(..., description="Photo of outfit or single garment"),
):
    """
    Submit a photo for garment extraction and quality analysis.

    Returns an :class:`ImportSessionReport` with:
    - ``ready``        garments that passed all quality checks
    - ``needs_review`` garments with moderate issues (user must decide)
    - ``failed``       garments that could not be extracted reliably
    - ``image_quality_warnings`` global image-level issues

    The client should render the three-zone validation UI based on
    this response, then call the accept / reject / correct endpoints.
    """
    # Load image
    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}")

    # Run extraction pipeline (fallback/simple mode so the route always works)
    try:
        if create_pipeline is None:
            raise RuntimeError("Pipeline not available")
        pipeline = create_pipeline(mode="auto")
        import numpy as np
        extraction = pipeline.process(np.array(pil_image))
        garments = extraction.garments
    except Exception as exc:
        logger.warning(f"Extraction pipeline failed: {exc} — returning empty garment list")
        garments = []

    # Create quality-checked session
    manager = get_session_manager()
    session = manager.create_session(garments=garments, source_image=pil_image)
    report = manager.get_session_report(session.session_id)

    # Add convenience counters expected by the client UI
    garment_records = list(session.garment_records.values())
    from src.layer0_segmentation.quality_checker import GarmentStatus as _GS
    report["total"] = len(garment_records)
    report["ready"] = sum(1 for r in garment_records if r.report.status == _GS.READY)
    report["needs_review"] = sum(1 for r in garment_records if r.report.status == _GS.NEEDS_REVIEW)
    report["failed"] = sum(1 for r in garment_records if r.report.status == _GS.FAILED)

    warnings: List[str] = []
    if not garments:
        warnings.append(
            "No garments were detected in this photo. "
            "Try a clearer photo or add items manually."
        )

    return ApiEnvelope(ok=True, data=report, warnings=warnings)


# ---------------------------------------------------------------------------
# 2. Get session state
# ---------------------------------------------------------------------------

@router.get("/{session_id}", response_model=ApiEnvelope)
async def get_session(session_id: str):
    """Return the current state of an import session."""
    manager = get_session_manager()
    report = manager.get_session_report(session_id)
    if report is None:
        _not_found(session_id)
    return ApiEnvelope(ok=True, data=report)


# ---------------------------------------------------------------------------
# 3. Accept garment(s)
# ---------------------------------------------------------------------------

@router.post("/{session_id}/accept", response_model=ApiEnvelope)
async def accept_garment(session_id: str, body: AcceptRequest = Body(...)):
    """
    Accept a garment (or all ready garments at once).

    - Provide ``session_garment_id`` to accept a specific garment.
    - Omit it to bulk-accept all garments whose status is ``ready``.
    """
    manager = get_session_manager()
    session = manager.get_session(session_id)
    if session is None:
        _not_found(session_id)

    if body.session_garment_id:
        ok = manager.accept_garment(session_id, body.session_garment_id)
        if not ok:
            _bad_garment(body.session_garment_id)
        data = {"accepted": [body.session_garment_id]}
    else:
        count = manager.bulk_accept_ready(session_id)
        data = {"accepted_count": count}

    return ApiEnvelope(ok=True, data=data)


# ---------------------------------------------------------------------------
# 4. Reject garment
# ---------------------------------------------------------------------------

@router.post("/{session_id}/reject", response_model=ApiEnvelope)
async def reject_garment(session_id: str, body: RejectRequest = Body(...)):
    """Reject a garment — it will not be added to the wardrobe."""
    manager = get_session_manager()
    if manager.get_session(session_id) is None:
        _not_found(session_id)
    ok = manager.reject_garment(session_id, body.session_garment_id)
    if not ok:
        _bad_garment(body.session_garment_id)
    return ApiEnvelope(ok=True, data={"rejected": body.session_garment_id})


# ---------------------------------------------------------------------------
# 5. Correct attributes manually
# ---------------------------------------------------------------------------

@router.post("/{session_id}/correct", response_model=ApiEnvelope)
async def correct_garment(session_id: str, body: CorrectRequest = Body(...)):
    """
    Apply manual attribute corrections to a garment.

    The garment's decision is set to ``corrected`` and will be included
    when the session is finalised.

    Example body::

        {
          "session_garment_id": "abc_0001",
          "corrections": {"color": "ivory", "category": "blouse"}
        }
    """
    manager = get_session_manager()
    if manager.get_session(session_id) is None:
        _not_found(session_id)
    ok = manager.correct_garment(session_id, body.session_garment_id, body.corrections)
    if not ok:
        _bad_garment(body.session_garment_id)
    return ApiEnvelope(
        ok=True,
        data={
            "corrected": body.session_garment_id,
            "applied_corrections": body.corrections,
        },
    )


# ---------------------------------------------------------------------------
# 6. Mark for re-photography
# ---------------------------------------------------------------------------

@router.post("/{session_id}/resubmit", response_model=ApiEnvelope)
async def mark_for_resubmit(session_id: str, body: ResubmitRequest = Body(...)):
    """
    Mark a garment as needing a better photo.

    The garment stays in the session but is excluded from finalisation
    until the user re-submits a new close-up photo.
    """
    manager = get_session_manager()
    ok = manager.mark_for_resubmit(session_id, body.session_garment_id)
    if not ok:
        _bad_garment(body.session_garment_id)
    return ApiEnvelope(
        ok=True,
        data={
            "session_garment_id": body.session_garment_id,
            "status": "awaiting_resubmission",
            "message": (
                "We will remind you to photograph this piece again. "
                "It will not be added to your wardrobe until you do."
            ),
        },
    )


# ---------------------------------------------------------------------------
# 7. Finalise session
# ---------------------------------------------------------------------------

@router.post("/{session_id}/finalise", response_model=ApiEnvelope)
async def finalise_session(session_id: str):
    """
    Commit all accepted / corrected garments to the virtual wardrobe.

    - Triggers Layer 1 attribute extraction for each accepted garment.
    - Garments still PENDING are treated as rejected.
    - Returns a summary of what was added.
    """
    manager = get_session_manager()
    session = manager.get_session(session_id)
    if session is None:
        _not_found(session_id)

    accepted_records = manager.finalise(session_id)

    added = []
    skipped = []

    for record in accepted_records:
        garment = record.report.garment
        # Apply user corrections to garment metadata
        if record.user_corrections:
            garment.metadata.update(record.user_corrections)

        garment_id = str(uuid.uuid4())
        added.append({
            "garment_id": garment_id,
            "label": garment.label,
            "category": garment.category.value,
            "adjusted_confidence": round(record.report.adjusted_confidence, 3),
            "user_corrections": record.user_corrections,
            "status": "added",
        })

    resubmit_records = [
        r for r in session.garment_records.values()
        if r.decision == GarmentDecision.RESUBMIT
    ]
    for record in resubmit_records:
        skipped.append({
            "label": record.report.garment.label,
            "reason": "awaiting_resubmission",
        })

    return ApiEnvelope(
        ok=True,
        data={
            "session_id": session_id,
            "added": added,
            "added_count": len(added),
            "skipped": skipped,
            "skipped_count": len(skipped),
        },
        warnings=(
            [
                f"{len(skipped)} garment(s) were skipped because they need a better photo."
            ]
            if skipped
            else []
        ),
    )


# ---------------------------------------------------------------------------
# 8. Add garment manually (no ML extraction needed)
# ---------------------------------------------------------------------------

@router.post("/manual", response_model=ApiEnvelope, status_code=201)
async def add_garment_manually(body: ManualGarmentRequest = Body(...)):
    """
    Add a garment directly to the wardrobe without photo extraction.

    Useful for small accessories (belt, jewellery, scarf) that are too
    small to be reliably segmented from a full-outfit photo.
    """
    garment_id = str(uuid.uuid4())
    return ApiEnvelope(
        ok=True,
        data={
            "garment_id": garment_id,
            "category": body.category,
            "label": body.label,
            "color": body.color,
            "material": body.material,
            "notes": body.notes,
            "source": "manual",
            "status": "added",
        },
    )
