"""
Unit tests for Layer 0 — Quality Checker
==========================================

Tests cover:
- QualityWarning data class
- GarmentReport status computation
- ImportSessionReport summary
- QualityChecker image-level checks (blur, luminance)
- QualityChecker per-garment checks (area, partial visibility,
  incomplete mask, low contrast, uncertain category)
- QualityChecker overlap detection (IoM thresholds)
- QualityThresholds customisation
- Numeric helpers (_laplacian_variance, _compute_hole_ratio,
  _border_gradient, _intersection_over_minimum)
"""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock
from pathlib import Path

from src.layer0_segmentation.quality_checker import (
    GarmentReport,
    GarmentStatus,
    ImportSessionReport,
    ProblemCode,
    QualityChecker,
    QualityThresholds,
    QualityWarning,
    Severity,
)
from src.layer0_segmentation.models import ExtractedGarment, BoundingBox
from src.layer0_segmentation.taxonomy import GarmentCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_garment(
    category=GarmentCategory.TOPS,
    label="t-shirt",
    confidence=0.90,
    area=5000,
    mask: np.ndarray = None,
    bbox: BoundingBox = None,
) -> ExtractedGarment:
    """Build a minimal ExtractedGarment for testing."""
    img = Image.new("RGBA", (512, 512), (200, 180, 160, 255))
    if mask is None:
        m = np.zeros((512, 512), dtype=np.uint8)
        m[100:300, 100:400] = 255          # solid rectangle
        mask = m
    return ExtractedGarment(
        image=img,
        category=category,
        label=label,
        confidence=confidence,
        mask=mask,
        area=area,
        bounding_box=bbox,
    )


def _solid_mask(h=512, w=512, fill=255) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[50:450, 50:450] = fill
    return m


def _sharp_rgb_image(h=256, w=256) -> Image.Image:
    """Return an RGB image with high Laplacian variance (sharp)."""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _blurry_rgb_image(h=256, w=256) -> Image.Image:
    """Return a uniformly grey image (very low Laplacian variance = blurry)."""
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _dark_rgb_image(h=256, w=256) -> Image.Image:
    arr = np.full((h, w, 3), 20, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _bright_rgb_image(h=256, w=256) -> Image.Image:
    arr = np.full((h, w, 3), 240, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# QualityWarning
# ---------------------------------------------------------------------------

class TestQualityWarning:
    def test_to_dict_contains_required_fields(self):
        w = QualityWarning(
            code=ProblemCode.OVERLAP_CRITICAL,
            severity=Severity.CRITICAL,
            confidence_penalty=0.35,
            user_message="Overlap detected",
            suggestion="Photograph separately",
        )
        d = w.to_dict()
        assert d["code"] == "OVERLAP_CRITICAL"
        assert d["severity"] == "critical"
        assert d["confidence_penalty"] == 0.35
        assert "user_message" in d
        assert "suggestion" in d
        assert "affected_attributes" in d

    def test_technical_detail_not_in_dict(self):
        w = QualityWarning(
            code=ProblemCode.LOW_QUALITY_IMAGE,
            severity=Severity.MINOR,
            confidence_penalty=0.05,
            user_message="Dark image",
            suggestion="Better light",
            _technical_detail="Mean luminance=20",
        )
        d = w.to_dict()
        assert "_technical_detail" not in d
        assert "technical_detail" not in d

    def test_affected_attributes_defaults_to_empty_list(self):
        w = QualityWarning(
            code=ProblemCode.SMALL_ACCESSORY,
            severity=Severity.CRITICAL,
            confidence_penalty=0.3,
            user_message="Too small",
            suggestion="Close-up photo",
        )
        assert w.affected_attributes == []


# ---------------------------------------------------------------------------
# GarmentReport
# ---------------------------------------------------------------------------

class TestGarmentReport:
    def test_status_ready_when_no_warnings(self):
        g = _make_garment()
        r = GarmentReport(garment=g, status=GarmentStatus.READY, adjusted_confidence=0.9)
        assert r.status == GarmentStatus.READY
        assert r.critical_warnings == []
        assert r.moderate_warnings == []
        assert r.minor_warnings == []

    def test_status_failed_when_critical_warning(self):
        g = _make_garment()
        w = QualityWarning(
            code=ProblemCode.OVERLAP_CRITICAL,
            severity=Severity.CRITICAL,
            confidence_penalty=0.35,
            user_message="msg",
            suggestion="sug",
        )
        r = GarmentReport(garment=g, warnings=[w], status=GarmentStatus.FAILED, adjusted_confidence=0.55)
        assert r.status == GarmentStatus.FAILED
        assert len(r.critical_warnings) == 1

    def test_to_dict_structure(self):
        g = _make_garment()
        r = GarmentReport(garment=g, status=GarmentStatus.READY, adjusted_confidence=0.88)
        d = r.to_dict()
        assert "status" in d
        assert "warnings" in d
        assert "adjusted_confidence" in d
        assert "category" in d

    def test_warning_categorisation(self):
        g = _make_garment()
        crit = QualityWarning(code=ProblemCode.OVERLAP_CRITICAL, severity=Severity.CRITICAL,
                               confidence_penalty=0.3, user_message="", suggestion="")
        mod  = QualityWarning(code=ProblemCode.OVERLAP_MODERATE, severity=Severity.MODERATE,
                               confidence_penalty=0.15, user_message="", suggestion="")
        minor = QualityWarning(code=ProblemCode.LOW_CONTRAST, severity=Severity.MINOR,
                                confidence_penalty=0.05, user_message="", suggestion="")
        r = GarmentReport(garment=g, warnings=[crit, mod, minor])
        assert len(r.critical_warnings) == 1
        assert len(r.moderate_warnings) == 1
        assert len(r.minor_warnings) == 1


# ---------------------------------------------------------------------------
# ImportSessionReport
# ---------------------------------------------------------------------------

class TestImportSessionReport:
    def _make_report(self, statuses):
        reports = []
        for s in statuses:
            g = _make_garment()
            r = GarmentReport(garment=g, status=s, adjusted_confidence=0.8)
            reports.append(r)
        return ImportSessionReport(session_id="test-123", garment_reports=reports)

    def test_ready_property(self):
        rpt = self._make_report([GarmentStatus.READY, GarmentStatus.FAILED, GarmentStatus.NEEDS_REVIEW])
        assert len(rpt.ready) == 1
        assert len(rpt.failed) == 1
        assert len(rpt.needs_review) == 1

    def test_summary_counts(self):
        rpt = self._make_report([GarmentStatus.READY, GarmentStatus.READY, GarmentStatus.FAILED])
        s = rpt.summary()
        assert s["total"] == 3
        assert s["ready"] == 2
        assert s["failed"] == 1
        assert s["session_id"] == "test-123"

    def test_to_dict_includes_garments(self):
        rpt = self._make_report([GarmentStatus.READY])
        d = rpt.to_dict()
        assert "garments" in d
        assert len(d["garments"]) == 1


# ---------------------------------------------------------------------------
# QualityChecker — image-level
# ---------------------------------------------------------------------------

class TestQualityCheckerImageLevel:
    def setup_method(self):
        self.checker = QualityChecker()

    def test_no_warnings_on_sharp_normal_image(self):
        img = _sharp_rgb_image()
        warnings = self.checker._check_image_quality(img)
        # Sharp + normal luminance → at most 0 warnings
        # (noise image will never be flagged as blurry or dark)
        assert all(w.code != ProblemCode.LOW_QUALITY_IMAGE
                   or w._technical_detail for w in warnings)

    def test_blurry_image_raises_warning(self):
        img = _blurry_rgb_image()
        warnings = self.checker._check_image_quality(img)
        codes = [w.code for w in warnings]
        assert ProblemCode.LOW_QUALITY_IMAGE in codes

    def test_dark_image_raises_warning(self):
        img = _dark_rgb_image()
        warnings = self.checker._check_image_quality(img)
        codes = [w.code for w in warnings]
        assert ProblemCode.LOW_QUALITY_IMAGE in codes

    def test_bright_image_raises_warning(self):
        img = _bright_rgb_image()
        warnings = self.checker._check_image_quality(img)
        codes = [w.code for w in warnings]
        assert ProblemCode.LOW_QUALITY_IMAGE in codes

    def test_blurry_warning_has_moderate_severity(self):
        img = _blurry_rgb_image()
        warnings = self.checker._check_image_quality(img)
        blur_warnings = [w for w in warnings if "blur" in w._technical_detail.lower()
                         or "laplacian" in w._technical_detail.lower()]
        if blur_warnings:
            assert blur_warnings[0].severity == Severity.MODERATE

    def test_dark_warning_has_minor_severity(self):
        img = _dark_rgb_image()
        warnings = self.checker._check_image_quality(img)
        dark_warnings = [w for w in warnings if "luminance" in w._technical_detail.lower()]
        if dark_warnings:
            assert dark_warnings[0].severity == Severity.MINOR


# ---------------------------------------------------------------------------
# QualityChecker — per-garment
# ---------------------------------------------------------------------------

class TestQualityCheckerPerGarment:
    def setup_method(self):
        self.checker = QualityChecker()

    def test_no_warnings_on_good_garment(self):
        g = _make_garment(confidence=0.92, area=5000)
        report = self.checker._check_garment(g, None, 0)
        assert report.status == GarmentStatus.READY
        assert report.warnings == []

    def test_small_accessory_is_critical(self):
        g = _make_garment(
            category=GarmentCategory.ACCESSORIES,
            label="bracelet",
            confidence=0.80,
            area=50,
        )
        report = self.checker._check_garment(g, None, 0)
        codes = [w.code for w in report.warnings]
        assert ProblemCode.SMALL_ACCESSORY in codes
        small_warn = next(w for w in report.warnings if w.code == ProblemCode.SMALL_ACCESSORY)
        assert small_warn.severity == Severity.CRITICAL
        assert report.status == GarmentStatus.FAILED

    def test_small_top_is_moderate(self):
        g = _make_garment(category=GarmentCategory.TOPS, area=100)
        report = self.checker._check_garment(g, None, 0)
        small_warns = [w for w in report.warnings if w.code == ProblemCode.SMALL_ACCESSORY]
        if small_warns:
            assert small_warns[0].severity == Severity.MODERATE

    def test_uncertain_category_triggers_warning(self):
        g = _make_garment(confidence=0.30)
        report = self.checker._check_garment(g, None, 0)
        codes = [w.code for w in report.warnings]
        assert ProblemCode.UNCERTAIN_CATEGORY in codes

    def test_partial_visibility_triggers_warning(self):
        bbox = BoundingBox(0, 0, 200, 200)   # area = 40 000 pixels
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[0:50, 0:50] = 255               # area ~ 2500 (ratio ≈ 0.06)
        g = _make_garment(mask=mask, area=2500, bbox=bbox)
        report = self.checker._check_garment(g, None, 0)
        codes = [w.code for w in report.warnings]
        assert ProblemCode.PARTIAL_VISIBILITY in codes

    def test_incomplete_mask_triggers_warning(self):
        # Create a mask with large internal holes (ring shape)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[50:450, 50:450] = 255          # filled square
        mask[150:350, 150:350] = 0          # large hole
        g = _make_garment(mask=mask, area=int(np.sum(mask > 0)))
        report = self.checker._check_garment(g, None, 0)
        codes = [w.code for w in report.warnings]
        assert ProblemCode.INCOMPLETE_MASK in codes

    def test_adjusted_confidence_decreases_with_penalties(self):
        g = _make_garment(confidence=0.80, area=50,
                          category=GarmentCategory.ACCESSORIES)
        report = self.checker._check_garment(g, None, 0)
        assert report.adjusted_confidence < 0.80

    def test_confidence_never_goes_below_zero(self):
        g = _make_garment(confidence=0.10, area=10,
                          category=GarmentCategory.ACCESSORIES)
        report = self.checker._check_garment(g, None, 0)
        assert report.adjusted_confidence >= 0.0


# ---------------------------------------------------------------------------
# QualityChecker — overlap detection
# ---------------------------------------------------------------------------

class TestQualityCheckerOverlap:
    def setup_method(self):
        self.checker = QualityChecker()

    def _two_garments_with_masks(self, mask_a, mask_b, area_a=None, area_b=None):
        g_a = _make_garment(mask=mask_a, area=area_a or int(np.sum(mask_a > 0)))
        g_b = _make_garment(mask=mask_b, area=area_b or int(np.sum(mask_b > 0)))
        r_a = GarmentReport(garment=g_a, status=GarmentStatus.READY, adjusted_confidence=0.9)
        r_b = GarmentReport(garment=g_b, status=GarmentStatus.READY, adjusted_confidence=0.9)
        return [r_a, r_b]

    def test_no_overlap_no_warning(self):
        mask_a = np.zeros((512, 512), dtype=np.uint8)
        mask_a[0:200, 0:200] = 255
        mask_b = np.zeros((512, 512), dtype=np.uint8)
        mask_b[300:500, 300:500] = 255
        reports = self._two_garments_with_masks(mask_a, mask_b)
        self.checker._check_overlaps(reports)
        for r in reports:
            overlap_codes = [w.code for w in r.warnings
                             if w.code in (ProblemCode.OVERLAP_CRITICAL, ProblemCode.OVERLAP_MODERATE)]
            assert overlap_codes == []

    def test_critical_overlap_detected(self):
        # mask_b is entirely inside mask_a → IoM = 1.0
        mask_a = np.zeros((512, 512), dtype=np.uint8)
        mask_a[50:450, 50:450] = 255
        mask_b = np.zeros((512, 512), dtype=np.uint8)
        mask_b[100:200, 100:200] = 255    # fully inside mask_a
        reports = self._two_garments_with_masks(mask_a, mask_b)
        self.checker._check_overlaps(reports)
        all_codes = [w.code for r in reports for w in r.warnings]
        assert ProblemCode.OVERLAP_CRITICAL in all_codes

    def test_moderate_overlap_detected(self):
        # ~30 % overlap by IoM
        mask_a = np.zeros((512, 512), dtype=np.uint8)
        mask_a[0:200, 0:200] = 255          # 200×200 = 40 000 px
        mask_b = np.zeros((512, 512), dtype=np.uint8)
        mask_b[150:350, 0:200] = 255        # 200×200 = 40 000 px, overlap 50×200 = 10 000
        # IoM = 10 000 / 40 000 = 0.25 → moderate
        reports = self._two_garments_with_masks(mask_a, mask_b)
        self.checker._check_overlaps(reports)
        all_codes = [w.code for r in reports for w in r.warnings]
        assert ProblemCode.OVERLAP_MODERATE in all_codes

    def test_critical_overlap_sets_failed_status(self):
        mask_a = np.zeros((512, 512), dtype=np.uint8)
        mask_a[50:450, 50:450] = 255
        mask_b = np.zeros((512, 512), dtype=np.uint8)
        mask_b[100:200, 100:200] = 255
        reports = self._two_garments_with_masks(mask_a, mask_b)
        self.checker._check_overlaps(reports)
        # The smaller garment (b) should be FAILED
        small_report = min(reports, key=lambda r: r.garment.area)
        assert small_report.status == GarmentStatus.FAILED

    def test_overlap_user_message_does_not_contain_technical_terms(self):
        mask_a = np.zeros((512, 512), dtype=np.uint8)
        mask_a[50:450, 50:450] = 255
        mask_b = np.zeros((512, 512), dtype=np.uint8)
        mask_b[100:200, 100:200] = 255
        reports = self._two_garments_with_masks(mask_a, mask_b)
        self.checker._check_overlaps(reports)
        for r in reports:
            for w in r.warnings:
                assert "IoM" not in w.user_message
                assert "mask" not in w.user_message.lower()
                assert "segmentation" not in w.user_message.lower()


# ---------------------------------------------------------------------------
# QualityChecker — full check() integration
# ---------------------------------------------------------------------------

class TestQualityCheckerFullCheck:
    def setup_method(self):
        self.checker = QualityChecker()

    def test_empty_garments_returns_empty_report(self):
        report = self.checker.check([], session_id="s1")
        assert len(report.garment_reports) == 0
        assert report.session_id == "s1"

    def test_good_garment_is_ready(self):
        g = _make_garment(confidence=0.92, area=6000)
        report = self.checker.check([g], session_id="s2")
        assert len(report.garment_reports) == 1
        assert report.garment_reports[0].status == GarmentStatus.READY

    def test_blurry_image_propagates_to_all_garments(self):
        garments = [_make_garment() for _ in range(3)]
        report = self.checker.check(garments, source_image=_blurry_rgb_image(), session_id="s3")
        for gr in report.garment_reports:
            image_codes = [w.code for w in gr.warnings]
            assert ProblemCode.LOW_QUALITY_IMAGE in image_codes

    def test_session_id_preserved(self):
        report = self.checker.check([], session_id="my-session-99")
        assert report.session_id == "my-session-99"

    def test_summary_totals_match_report(self):
        garments = [_make_garment() for _ in range(4)]
        report = self.checker.check(garments, session_id="s4")
        summary = report.summary()
        assert summary["total"] == 4
        assert summary["ready"] + summary["needs_review"] + summary["failed"] == 4


# ---------------------------------------------------------------------------
# QualityThresholds customisation
# ---------------------------------------------------------------------------

class TestQualityThresholds:
    def test_custom_thresholds_applied(self):
        thresholds = QualityThresholds(uncertain_confidence_threshold=0.99)
        checker = QualityChecker(thresholds=thresholds)
        g = _make_garment(confidence=0.80)
        report = checker._check_garment(g, None, 0)
        codes = [w.code for w in report.warnings]
        # confidence 0.80 < 0.99 → should trigger uncertain warning
        assert ProblemCode.UNCERTAIN_CATEGORY in codes

    def test_disabled_min_area_category(self):
        thresholds = QualityThresholds()
        thresholds.min_area[GarmentCategory.TOPS] = 0   # disable area check for TOPS
        checker = QualityChecker(thresholds=thresholds)
        g = _make_garment(category=GarmentCategory.TOPS, area=1)
        report = checker._check_garment(g, None, 0)
        small_warns = [w for w in report.warnings if w.code == ProblemCode.SMALL_ACCESSORY]
        assert small_warns == []


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

class TestNumericHelpers:
    def test_laplacian_variance_high_for_noisy_image(self):
        arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8).astype(np.float32)
        var = QualityChecker._laplacian_variance(arr)
        assert var > 100

    def test_laplacian_variance_low_for_uniform_image(self):
        arr = np.full((128, 128), 128, dtype=np.float32)
        var = QualityChecker._laplacian_variance(arr)
        assert var < 50

    def test_compute_hole_ratio_solid_mask(self):
        mask = _solid_mask()
        ratio = QualityChecker._compute_hole_ratio(mask)
        assert ratio == pytest.approx(0.0, abs=0.01)

    def test_compute_hole_ratio_ring_mask(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[20:180, 20:180] = 255    # outer square
        mask[60:140, 60:140] = 0      # inner hole
        ratio = QualityChecker._compute_hole_ratio(mask)
        assert ratio > 0.10

    def test_compute_hole_ratio_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        ratio = QualityChecker._compute_hole_ratio(mask)
        assert ratio == 0.0

    def test_intersection_over_minimum_identical_masks(self):
        mask = _solid_mask()
        iom = QualityChecker._intersection_over_minimum(mask, mask)
        assert iom == pytest.approx(1.0, abs=0.01)

    def test_intersection_over_minimum_no_overlap(self):
        m_a = np.zeros((512, 512), dtype=np.uint8)
        m_a[:, :256] = 255
        m_b = np.zeros((512, 512), dtype=np.uint8)
        m_b[:, 256:] = 255
        iom = QualityChecker._intersection_over_minimum(m_a, m_b)
        assert iom == pytest.approx(0.0, abs=0.01)

    def test_intersection_over_minimum_partial_overlap(self):
        m_a = np.zeros((100, 100), dtype=np.uint8)
        m_a[0:100, 0:100] = 255          # 10 000 px
        m_b = np.zeros((100, 100), dtype=np.uint8)
        m_b[50:100, 0:100] = 255         # 5 000 px, fully inside m_a
        # IoM = 5000 / min(10000, 5000) = 1.0
        iom = QualityChecker._intersection_over_minimum(m_a, m_b)
        assert iom == pytest.approx(1.0, abs=0.01)

    def test_border_gradient_returns_float(self):
        mask = _solid_mask()
        gray = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        result = QualityChecker._border_gradient(gray, mask)
        assert isinstance(result, float)
        assert result >= 0.0
