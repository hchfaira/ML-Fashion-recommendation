"""
Extraction Quality Checker — Layer 0
=====================================

Analyses each extracted mask and the source image to detect quality
problems **before** garments are passed to Layer 1 or stored in the
virtual wardrobe.

Problem codes
-------------
OVERLAP_CRITICAL       Two masks share > 50 % of surface area.
OVERLAP_MODERATE       Two masks share 20-50 % of surface area.
LOW_QUALITY_IMAGE      Source image is blurry or poorly lit.
SMALL_ACCESSORY        Garment area < minimum threshold for its category.
UNCERTAIN_CATEGORY     Detection confidence is below the uncertainty threshold.
PARTIAL_VISIBILITY     Mask covers < 40 % of its bounding box (truncated garment).
LOW_CONTRAST           Mean edge gradient near the mask border is too low.
INCOMPLETE_MASK        Mask has a high proportion of internal holes.

Each problem produces a structured :class:`QualityWarning` with:
- A human-readable ``user_message`` (never technical jargon).
- A concrete ``suggestion`` telling the user what to do.
- A ``confidence_penalty`` in [0, 1] that is subtracted from the
  garment's overall confidence score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .models import ExtractedGarment, FusedMask
from .taxonomy import GarmentCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProblemCode(str, Enum):
    """Standardised problem codes produced by the quality checker."""
    OVERLAP_CRITICAL    = "OVERLAP_CRITICAL"
    OVERLAP_MODERATE    = "OVERLAP_MODERATE"
    LOW_QUALITY_IMAGE   = "LOW_QUALITY_IMAGE"
    SMALL_ACCESSORY     = "SMALL_ACCESSORY"
    UNCERTAIN_CATEGORY  = "UNCERTAIN_CATEGORY"
    PARTIAL_VISIBILITY  = "PARTIAL_VISIBILITY"
    LOW_CONTRAST        = "LOW_CONTRAST"
    INCOMPLETE_MASK     = "INCOMPLETE_MASK"


class Severity(str, Enum):
    """How severe the quality problem is."""
    CRITICAL = "critical"   # Garment should NOT be added without user review.
    MODERATE = "moderate"   # Garment may be added but flagged for review.
    MINOR    = "minor"      # Informational — garment is added normally.


class GarmentStatus(str, Enum):
    """Final status assigned to each extracted garment."""
    READY        = "ready"        # High-confidence extraction — add automatically.
    NEEDS_REVIEW = "needs_review" # One or more MODERATE/CRITICAL warnings.
    FAILED       = "failed"       # At least one CRITICAL warning — block addition.


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QualityWarning:
    """A single quality problem detected for one garment."""

    code: ProblemCode
    severity: Severity
    confidence_penalty: float        # Subtracted from garment confidence [0, 1].
    user_message: str                # Plain-English message shown to the user.
    suggestion: str                  # Concrete action the user can take.
    affected_attributes: List[str] = field(default_factory=list)
    # Technical detail kept in logs, never sent to the client.
    _technical_detail: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "confidence_penalty": round(self.confidence_penalty, 3),
            "user_message": self.user_message,
            "suggestion": self.suggestion,
            "affected_attributes": self.affected_attributes,
        }


@dataclass
class GarmentReport:
    """
    Quality report for a single extracted garment.

    Consumed by the API layer to build the validation response
    sent to the client.
    """
    garment: ExtractedGarment
    warnings: List[QualityWarning] = field(default_factory=list)
    status: GarmentStatus = GarmentStatus.READY
    adjusted_confidence: float = 1.0

    # Temporary session identifier (set by SessionManager)
    session_garment_id: str = ""

    @property
    def critical_warnings(self) -> List[QualityWarning]:
        return [w for w in self.warnings if w.severity == Severity.CRITICAL]

    @property
    def moderate_warnings(self) -> List[QualityWarning]:
        return [w for w in self.warnings if w.severity == Severity.MODERATE]

    @property
    def minor_warnings(self) -> List[QualityWarning]:
        return [w for w in self.warnings if w.severity == Severity.MINOR]

    def to_dict(self) -> dict:
        return {
            "session_garment_id": self.session_garment_id,
            "category": self.garment.category.value,
            "label": self.garment.label,
            "confidence": round(self.garment.confidence, 3),
            "adjusted_confidence": round(self.adjusted_confidence, 3),
            "status": self.status.value,
            "warnings": [w.to_dict() for w in self.warnings],
        }


@dataclass
class ImportSessionReport:
    """
    Aggregated quality report for a complete photo-import session.

    Returned by the API after the initial photo submission so the
    client can render the three-zone validation UI.
    """
    session_id: str
    garment_reports: List[GarmentReport] = field(default_factory=list)
    image_quality_warnings: List[QualityWarning] = field(default_factory=list)

    @property
    def ready(self) -> List[GarmentReport]:
        return [r for r in self.garment_reports if r.status == GarmentStatus.READY]

    @property
    def needs_review(self) -> List[GarmentReport]:
        return [r for r in self.garment_reports if r.status == GarmentStatus.NEEDS_REVIEW]

    @property
    def failed(self) -> List[GarmentReport]:
        return [r for r in self.garment_reports if r.status == GarmentStatus.FAILED]

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "total": len(self.garment_reports),
            "ready": len(self.ready),
            "needs_review": len(self.needs_review),
            "failed": len(self.failed),
            "image_quality_warnings": [w.to_dict() for w in self.image_quality_warnings],
        }

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "garments": [r.to_dict() for r in self.garment_reports],
        }


# ---------------------------------------------------------------------------
# Thresholds (all tunable via constructor)
# ---------------------------------------------------------------------------

@dataclass
class QualityThresholds:
    """Configurable quality thresholds for the checker."""

    # Overlap
    overlap_critical_pct: float = 0.50   # IoM > this → CRITICAL
    overlap_moderate_pct: float = 0.20   # IoM > this → MODERATE

    # Mask completeness
    partial_visibility_ratio: float = 0.40  # mask_area / bbox_area < this
    incomplete_mask_hole_ratio: float = 0.15 # internal hole area / mask area > this

    # Area minimums per category (in pixels, at 512×512 canvas)
    min_area: dict = field(default_factory=lambda: {
        GarmentCategory.ACCESSORIES: 300,
        GarmentCategory.FOOTWEAR:    800,
        GarmentCategory.TOPS:       2000,
        GarmentCategory.BOTTOMS:    2000,
        GarmentCategory.FULL_BODY:  3000,
        GarmentCategory.OUTERWEAR:  3000,
    })

    # Detection confidence
    uncertain_confidence_threshold: float = 0.45

    # Image quality (Laplacian variance)
    blur_threshold: float = 80.0         # Below this → blurry image warning
    dark_threshold: float = 50.0         # Mean luminance below this → dark image
    bright_threshold: float = 220.0      # Mean luminance above this → overexposed

    # Edge contrast around mask border
    low_contrast_gradient_threshold: float = 15.0


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

class QualityChecker:
    """
    Analyses :class:`ExtractedGarment` objects and source image quality.

    Usage::

        checker = QualityChecker()
        session_report = checker.check(
            garments=pipeline_result.garments,
            source_image=pil_image,
            session_id="abc123",
        )
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        garments: List[ExtractedGarment],
        source_image: Optional[Image.Image] = None,
        session_id: str = "",
    ) -> ImportSessionReport:
        """
        Run all quality checks and return a full session report.

        Args:
            garments:     Extracted garments from the pipeline.
            source_image: Original input image (for image-level checks).
            session_id:   Identifier for the import session.

        Returns:
            :class:`ImportSessionReport` with per-garment and global warnings.
        """
        # ── Image-level checks ─────────────────────────────────────────
        image_warnings: List[QualityWarning] = []
        if source_image is not None:
            image_warnings = self._check_image_quality(source_image)
            for w in image_warnings:
                logger.info(f"[QualityChecker] image-level {w.severity.value}: {w.code.value}")

        # ── Per-garment checks ─────────────────────────────────────────
        reports: List[GarmentReport] = []
        for idx, garment in enumerate(garments):
            report = self._check_garment(garment, source_image, idx)
            reports.append(report)

        # ── Overlap checks (need all garments together) ────────────────
        self._check_overlaps(reports)

        # ── Propagate image-level penalties to all garments ────────────
        if image_warnings:
            severe_img = [w for w in image_warnings if w.severity != Severity.MINOR]
            for report in reports:
                for w in severe_img:
                    # Only add if not already present (same code)
                    existing_codes = {ew.code for ew in report.warnings}
                    if w.code not in existing_codes:
                        report.warnings.append(w)
                self._finalise_report(report)

        return ImportSessionReport(
            session_id=session_id,
            garment_reports=reports,
            image_quality_warnings=image_warnings,
        )

    # ------------------------------------------------------------------
    # Image-level checks
    # ------------------------------------------------------------------

    def _check_image_quality(
        self, image: Image.Image
    ) -> List[QualityWarning]:
        warnings: List[QualityWarning] = []
        img_np = np.array(image.convert("RGB"))

        # Blur detection — Laplacian variance
        gray = np.mean(img_np, axis=2).astype(np.float32)
        laplacian = self._laplacian_variance(gray)
        if laplacian < self.thresholds.blur_threshold:
            warnings.append(QualityWarning(
                code=ProblemCode.LOW_QUALITY_IMAGE,
                severity=Severity.MODERATE,
                confidence_penalty=0.15,
                user_message=(
                    "This photo appears blurry. "
                    "The extracted colours and details may not be accurate."
                ),
                suggestion=(
                    "Retake the photo in good light, keeping the camera steady. "
                    "You can also correct individual attributes manually after import."
                ),
                affected_attributes=["color", "pattern", "texture"],
                _technical_detail=f"Laplacian variance={laplacian:.1f} < {self.thresholds.blur_threshold}",
            ))

        # Luminance check
        luminance = float(np.mean(gray))
        if luminance < self.thresholds.dark_threshold:
            warnings.append(QualityWarning(
                code=ProblemCode.LOW_QUALITY_IMAGE,
                severity=Severity.MINOR,
                confidence_penalty=0.05,
                user_message=(
                    "This photo is quite dark. "
                    "Colour extraction may be less accurate."
                ),
                suggestion=(
                    "Try taking the photo near a window or in a well-lit room."
                ),
                affected_attributes=["color"],
                _technical_detail=f"Mean luminance={luminance:.1f} < {self.thresholds.dark_threshold}",
            ))
        elif luminance > self.thresholds.bright_threshold:
            warnings.append(QualityWarning(
                code=ProblemCode.LOW_QUALITY_IMAGE,
                severity=Severity.MINOR,
                confidence_penalty=0.05,
                user_message=(
                    "This photo is overexposed. "
                    "Some colour details may be washed out."
                ),
                suggestion="Avoid direct sunlight on the garment when photographing.",
                affected_attributes=["color"],
                _technical_detail=f"Mean luminance={luminance:.1f} > {self.thresholds.bright_threshold}",
            ))

        return warnings

    # ------------------------------------------------------------------
    # Per-garment checks
    # ------------------------------------------------------------------

    def _check_garment(
        self,
        garment: ExtractedGarment,
        source_image: Optional[Image.Image],
        index: int,
    ) -> GarmentReport:
        warnings: List[QualityWarning] = []

        # Uncertain category / low detection confidence
        if garment.confidence < self.thresholds.uncertain_confidence_threshold:
            warnings.append(QualityWarning(
                code=ProblemCode.UNCERTAIN_CATEGORY,
                severity=Severity.MODERATE,
                confidence_penalty=0.20,
                user_message=(
                    f"We are not sure whether this piece is a "
                    f"'{garment.label}'. The category may be incorrect."
                ),
                suggestion=(
                    "Please confirm or correct the category after import "
                    "so recommendations stay accurate."
                ),
                affected_attributes=["category", "subcategory"],
                _technical_detail=f"Detection confidence={garment.confidence:.2f}",
            ))

        # Area too small for category
        mask_area = garment.area
        min_area = self.thresholds.min_area.get(garment.category, 500)
        if mask_area < min_area:
            is_accessory = garment.category in (
                GarmentCategory.ACCESSORIES,
            )
            warnings.append(QualityWarning(
                code=ProblemCode.SMALL_ACCESSORY,
                severity=Severity.CRITICAL if is_accessory else Severity.MODERATE,
                confidence_penalty=0.30 if is_accessory else 0.15,
                user_message=(
                    f"The {garment.label} is too small in this photo to be "
                    f"analysed accurately."
                ),
                suggestion=(
                    "Take a close-up photo of this item on its own so we can "
                    "capture all its details. You can also add it manually."
                ),
                affected_attributes=["color", "pattern", "texture", "material"],
                _technical_detail=f"mask_area={mask_area} < min={min_area}",
            ))

        # Partial visibility — mask vs bounding box ratio
        if garment.bounding_box is not None and garment.bounding_box.area > 0:
            ratio = mask_area / garment.bounding_box.area
            if ratio < self.thresholds.partial_visibility_ratio:
                warnings.append(QualityWarning(
                    code=ProblemCode.PARTIAL_VISIBILITY,
                    severity=Severity.MODERATE,
                    confidence_penalty=0.20,
                    user_message=(
                        f"This {garment.label} appears to be partially hidden "
                        f"(perhaps under another garment)."
                    ),
                    suggestion=(
                        "For a complete analysis, photograph this piece on its own. "
                        "It will be added with partial attributes for now."
                    ),
                    affected_attributes=["color", "pattern", "length", "silhouette"],
                    _technical_detail=f"mask/bbox ratio={ratio:.2f} < {self.thresholds.partial_visibility_ratio}",
                ))

        # Incomplete mask — internal holes
        if garment.mask is not None:
            hole_ratio = self._compute_hole_ratio(garment.mask)
            if hole_ratio > self.thresholds.incomplete_mask_hole_ratio:
                warnings.append(QualityWarning(
                    code=ProblemCode.INCOMPLETE_MASK,
                    severity=Severity.MINOR,
                    confidence_penalty=0.10,
                    user_message=(
                        f"The outline of this {garment.label} has some gaps. "
                        f"Some edge details may be missed."
                    ),
                    suggestion=(
                        "A photo with a plain background gives much better results."
                    ),
                    affected_attributes=["silhouette", "pattern"],
                    _technical_detail=f"hole_ratio={hole_ratio:.2f}",
                ))

        # Low contrast at mask border
        if source_image is not None and garment.mask is not None:
            mean_grad = self._border_gradient(
                np.array(source_image.convert("L")), garment.mask
            )
            if mean_grad < self.thresholds.low_contrast_gradient_threshold:
                warnings.append(QualityWarning(
                    code=ProblemCode.LOW_CONTRAST,
                    severity=Severity.MINOR,
                    confidence_penalty=0.05,
                    user_message=(
                        f"This {garment.label} blends into the background. "
                        f"The cut-out edges may be slightly off."
                    ),
                    suggestion=(
                        "Wearing the item against a contrasting background "
                        "improves extraction quality."
                    ),
                    affected_attributes=["silhouette"],
                    _technical_detail=f"border_gradient={mean_grad:.1f}",
                ))

        report = GarmentReport(
            garment=garment,
            warnings=warnings,
            adjusted_confidence=garment.confidence,
        )
        self._finalise_report(report)
        return report

    # ------------------------------------------------------------------
    # Overlap checks (cross-garment)
    # ------------------------------------------------------------------

    def _check_overlaps(self, reports: List[GarmentReport]) -> None:
        """
        Detect masks that overlap significantly.

        Uses Intersection over Minimum (IoM) so that a small garment
        fully inside a large one always registers as critical.
        """
        masks_with_area = [
            (i, r, r.garment.mask)
            for i, r in enumerate(reports)
            if r.garment.mask is not None and r.garment.area > 0
        ]

        for i in range(len(masks_with_area)):
            idx_a, report_a, mask_a = masks_with_area[i]
            for j in range(i + 1, len(masks_with_area)):
                idx_b, report_b, mask_b = masks_with_area[j]

                # Skip garments that come from *different* source images —
                # overlap is only meaningful within the same photo.
                src_a = report_a.garment.source_path
                src_b = report_b.garment.source_path
                if src_a is not None and src_b is not None and src_a != src_b:
                    continue

                iom = self._intersection_over_minimum(mask_a, mask_b)
                if iom <= 0:
                    continue

                label_a = report_a.garment.label
                label_b = report_b.garment.label

                if iom > self.thresholds.overlap_critical_pct:
                    severity = Severity.CRITICAL
                    penalty  = 0.35
                    msg_a = (
                        f"Your {label_a} and {label_b} overlap heavily in this photo. "
                        f"The {label_a} extraction is likely incomplete."
                    )
                    msg_b = (
                        f"Your {label_b} and {label_a} overlap heavily in this photo. "
                        f"The {label_b} extraction is likely incomplete."
                    )
                    suggestion = (
                        "Photograph each piece separately for a complete wardrobe entry."
                    )
                elif iom > self.thresholds.overlap_moderate_pct:
                    severity = Severity.MODERATE
                    penalty  = 0.15
                    msg_a = (
                        f"Your {label_a} partially overlaps with the {label_b}. "
                        f"Some attributes of the {label_a} may be missing."
                    )
                    msg_b = (
                        f"Your {label_b} partially overlaps with the {label_a}. "
                        f"Some attributes of the {label_b} may be missing."
                    )
                    suggestion = (
                        "You can correct missing attributes manually after import, "
                        "or re-photograph the item alone for a complete analysis."
                    )
                else:
                    continue

                for report, msg in ((report_a, msg_a), (report_b, msg_b)):
                    report.warnings.append(QualityWarning(
                        code=(
                            ProblemCode.OVERLAP_CRITICAL
                            if severity == Severity.CRITICAL
                            else ProblemCode.OVERLAP_MODERATE
                        ),
                        severity=severity,
                        confidence_penalty=penalty,
                        user_message=msg,
                        suggestion=suggestion,
                        affected_attributes=["color", "pattern", "silhouette", "length"],
                        _technical_detail=f"IoM={iom:.2f} with garment index {idx_b if report is report_a else idx_a}",
                    ))
                    self._finalise_report(report)

                logger.info(
                    f"[QualityChecker] overlap {severity.value} "
                    f"({label_a} ↔ {label_b}, IoM={iom:.2f})"
                )

    # ------------------------------------------------------------------
    # Finalise a report (compute status + adjusted confidence)
    # ------------------------------------------------------------------

    def _finalise_report(self, report: GarmentReport) -> None:
        total_penalty = sum(w.confidence_penalty for w in report.warnings)
        report.adjusted_confidence = max(0.0, report.garment.confidence - total_penalty)

        if report.critical_warnings:
            report.status = GarmentStatus.FAILED
        elif report.moderate_warnings:
            report.status = GarmentStatus.NEEDS_REVIEW
        else:
            report.status = GarmentStatus.READY

    # ------------------------------------------------------------------
    # Numeric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """Estimate image sharpness via Laplacian variance.

        Applies a 3×3 Laplacian kernel via simple 2-D convolution and
        returns the variance of the result.  A perfectly uniform image
        (all pixels identical) produces a variance of exactly 0.
        """
        lap = (
            -4.0 * gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        )
        return float(np.var(lap))

    @staticmethod
    def _compute_hole_ratio(mask: np.ndarray) -> float:
        """
        Estimate the proportion of holes inside the mask.

        Floods the background from every border cell of the image.
        Any background pixel that is NOT reachable from the border
        is considered an interior hole.
        """
        binary = (mask > 127).astype(np.uint8)
        if binary.sum() == 0:
            return 0.0

        background = (binary == 0)  # True where there is no mask
        h, w = binary.shape

        # BFS from all border background pixels
        from collections import deque
        visited = np.zeros((h, w), dtype=bool)
        q: deque = deque()

        # Seed from every background pixel on the image border
        for r in range(h):
            for c in (0, w - 1):
                if background[r, c] and not visited[r, c]:
                    visited[r, c] = True
                    q.append((r, c))
        for c in range(w):
            for r in (0, h - 1):
                if background[r, c] and not visited[r, c]:
                    visited[r, c] = True
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not visited[nr, nc] and background[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))

        # Interior holes = background pixels not reachable from any border
        interior_holes = int(np.sum(background & ~visited))
        mask_area = float(binary.sum())
        return float(interior_holes) / mask_area if mask_area > 0 else 0.0

    @staticmethod
    def _border_gradient(gray: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute mean gradient magnitude along the mask boundary pixels.

        A low value means the garment blends into the background.
        """
        # Align mask to the image dimensions if they differ
        if gray.shape != mask.shape:
            from PIL import Image as _PIL
            mask_img = _PIL.fromarray(mask).resize(
                (gray.shape[1], gray.shape[0]), _PIL.NEAREST
            )
            mask = np.array(mask_img)

        binary = (mask > 127).astype(np.uint8)
        # Dilate mask by 2px to find border pixels
        border = np.zeros_like(binary)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                shifted = np.roll(np.roll(binary, dy, axis=0), dx, axis=1)
                border = np.maximum(border, shifted)
        border_ring = (border - binary).astype(bool)

        if not border_ring.any():
            return 100.0  # No border → no problem

        # Gradient via simple differences
        gy = np.abs(np.diff(gray.astype(np.float32), axis=0, prepend=gray[:1]))
        gx = np.abs(np.diff(gray.astype(np.float32), axis=1, prepend=gray[:, :1]))
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return float(magnitude[border_ring].mean())

    @staticmethod
    def _intersection_over_minimum(
        mask_a: np.ndarray, mask_b: np.ndarray
    ) -> float:
        """
        Intersection over Minimum (IoM).

        Unlike IoU, this measures how much of the *smaller* mask is
        covered by the other — ideal for detecting layered garments.
        """
        bin_a = (mask_a > 127).astype(np.uint8)
        bin_b = (mask_b > 127).astype(np.uint8)

        # Align shapes
        h = min(bin_a.shape[0], bin_b.shape[0])
        w = min(bin_a.shape[1], bin_b.shape[1])
        bin_a = bin_a[:h, :w]
        bin_b = bin_b[:h, :w]

        intersection = int(np.sum(bin_a & bin_b))
        min_area = min(int(bin_a.sum()), int(bin_b.sum()))
        return intersection / min_area if min_area > 0 else 0.0
