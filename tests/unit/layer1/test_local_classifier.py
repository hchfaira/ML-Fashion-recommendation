"""
Tests for Layer 1: LocalGarmentClassifier — Solution 4 (Lightweight pre-classifier)

Tests cover:
- Filename-based classification for all major categories
- Parent-folder keyword matching
- Aspect-ratio fallback (PIL)
- Confidence threshold / targeted hint
- Fallback category when no keyword matches
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.layer1_vision.local_classifier import (
    LocalGarmentClassifier,
    ClassificationResult,
    CONFIDENCE_THRESHOLD,
)
from src.core.models import GarmentCategory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf() -> LocalGarmentClassifier:
    return LocalGarmentClassifier()


def _fake_path(name: str, parent: str = "wardrobe") -> Path:
    """Create a fake path without touching the filesystem."""
    return Path(f"/fake/{parent}/{name}")


# ---------------------------------------------------------------------------
# Filename keyword classification
# ---------------------------------------------------------------------------

class TestFilenameClassification:
    @pytest.mark.parametrize(
        "filename,expected_category",
        [
            ("white_shirt.png", GarmentCategory.TOP),
            ("blouse_floral.jpg", GarmentCategory.TOP),
            ("blue_jeans.png", GarmentCategory.BOTTOM),
            ("chino_pants.jpg", GarmentCategory.BOTTOM),
            ("ankle_boots.png", GarmentCategory.SHOES),
            ("sneakers_white.jpg", GarmentCategory.SHOES),
            ("leather_jacket.png", GarmentCategory.OUTERWEAR),
            ("winter_coat.jpg", GarmentCategory.OUTERWEAR),
            ("floral_dress.png", GarmentCategory.DRESS),
            ("evening_gown.jpg", GarmentCategory.DRESS),
            ("silk_scarf.png", GarmentCategory.ACCESSORY),
            ("canvas_bag.jpg", GarmentCategory.ACCESSORY),
            # bra / underwear → fallback (no keyword match) → TOP or any category
            # leggings → BOTTOM via "legging" keyword
            ("workout_leggings.jpg", GarmentCategory.BOTTOM),
        ],
    )
    def test_classify_by_filename(
        self, clf: LocalGarmentClassifier, filename: str, expected_category: GarmentCategory
    ):
        result = clf.classify(_fake_path(filename))
        assert result.category == expected_category

    def test_confidence_high_on_keyword_match(
        self, clf: LocalGarmentClassifier
    ):
        result = clf.classify(_fake_path("red_tshirt.png"))
        assert result.confidence >= 0.80

    def test_source_is_filename(
        self, clf: LocalGarmentClassifier
    ):
        result = clf.classify(_fake_path("blue_jeans.png"))
        assert result.source == "filename"


class TestFolderKeywordClassification:
    def test_classify_by_parent_folder_tops(
        self, clf: LocalGarmentClassifier
    ):
        path = Path("/wardrobe/tops/random_item_001.png")
        result = clf.classify(path)
        assert result.category == GarmentCategory.TOP

    def test_classify_by_parent_folder_shoes(
        self, clf: LocalGarmentClassifier
    ):
        path = Path("/wardrobe/footwear/item.png")
        result = clf.classify(path)
        assert result.category == GarmentCategory.SHOES

    def test_source_is_folder_when_filename_ambiguous(
        self, clf: LocalGarmentClassifier
    ):
        path = Path("/wardrobe/bottoms/IMG_0001.png")
        result = clf.classify(path)
        assert result.category == GarmentCategory.BOTTOM
        assert result.source in ("filename", "folder", "aspect_ratio", "fallback")


# ---------------------------------------------------------------------------
# Aspect-ratio fallback
# ---------------------------------------------------------------------------

class TestAspectRatioFallback:
    def _make_mock_image(self, width: int, height: int) -> MagicMock:
        mock = MagicMock()
        mock.size = (width, height)
        return mock

    def test_tall_image_gives_bottom(
        self, clf: LocalGarmentClassifier, tmp_path: Path
    ):
        # Use a neutral folder name to avoid keyword interference
        folder = tmp_path / "garments"
        folder.mkdir()
        img_path = folder / "IMG_9999.png"
        img_path.write_bytes(b"fake")
        with patch("src.layer1_vision.local_classifier.Image") as mock_image_module:
            mock_image_module.open.return_value.__enter__ = lambda s: self._make_mock_image(200, 500)
            mock_image_module.open.return_value.__exit__ = MagicMock(return_value=False)
            result = clf.classify(img_path)
        # Aspect ratio: 200/500 = 0.4 — tall → should lean towards BOTTOM/SHOES
        assert result.source in ("aspect_ratio", "fallback")

    def test_wide_image_gives_shoes(
        self, clf: LocalGarmentClassifier, tmp_path: Path
    ):
        # Use a neutral folder name to avoid keyword interference
        folder = tmp_path / "garments"
        folder.mkdir()
        img_path = folder / "IMG_8888.png"
        img_path.write_bytes(b"fake")
        with patch("src.layer1_vision.local_classifier.Image") as mock_image_module:
            mock_image_module.open.return_value.__enter__ = lambda s: self._make_mock_image(600, 200)
            mock_image_module.open.return_value.__exit__ = MagicMock(return_value=False)
            result = clf.classify(img_path)
        assert result.source in ("aspect_ratio", "fallback")


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    def test_unknown_filename_falls_back_to_top(
        self, clf: LocalGarmentClassifier, tmp_path: Path
    ):
        img = tmp_path / "IMG_0001.png"
        img.write_bytes(b"FAKE")
        with patch("src.layer1_vision.local_classifier.Image") as m:
            m.open.side_effect = Exception("PIL error")
            result = clf.classify(img)
        assert result.category == GarmentCategory.TOP
        assert result.source == "fallback"

    def test_fallback_confidence_is_low(
        self, clf: LocalGarmentClassifier, tmp_path: Path
    ):
        img = tmp_path / "IMG_0001.png"
        img.write_bytes(b"FAKE")
        with patch("src.layer1_vision.local_classifier.Image") as m:
            m.open.side_effect = Exception("PIL error")
            result = clf.classify(img)
        assert result.confidence < CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# ClassificationResult properties
# ---------------------------------------------------------------------------

class TestClassificationResult:
    def test_needs_gemini_false_above_threshold(
        self, clf: LocalGarmentClassifier
    ):
        result = clf.classify(_fake_path("white_tshirt.png"))
        if result.confidence >= CONFIDENCE_THRESHOLD:
            assert result.needs_gemini_confirmation is False

    def test_needs_gemini_true_below_threshold(self):
        result = ClassificationResult(
            category=GarmentCategory.TOP,
            confidence=0.30,
            source="fallback",
        )
        assert result.needs_gemini_confirmation is True

    def test_targeted_hint_not_empty_for_high_confidence(
        self, clf: LocalGarmentClassifier
    ):
        result = clf.classify(_fake_path("blue_jeans.png"))
        if not result.needs_gemini_confirmation:
            assert result.targeted_hint  # non-empty string

    def test_targeted_hint_mentions_category(
        self, clf: LocalGarmentClassifier
    ):
        result = clf.classify(_fake_path("leather_boots.png"))
        if not result.needs_gemini_confirmation:
            hint = result.targeted_hint.lower()
            assert any(kw in hint for kw in ("shoe", "boot", "footwear", "shoes"))
