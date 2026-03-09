"""
Tests for Analysis API Routes
================================

Covers:
- POST /analyze/image
- POST /analyze/outfit-image
- POST /analyze/color-harmony
- GET  /analyze/complementary-colors/{color}
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, UploadFile

from src.api.routes.analysis import (
    analyze_image,
    analyze_outfit_image,
    check_color_harmony,
    get_complementary_colors,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_vision_service():
    mock = MagicMock()
    mock.analyze_image = AsyncMock(return_value={"category": "top", "color": "blue"})
    with patch("src.api.routes.analysis.get_vision_service", return_value=mock):
        yield mock


@pytest.fixture
def mock_segmenter():
    mock = MagicMock()
    mock.segment_outfit = AsyncMock(return_value=[
        {"type": "top", "color": "white"},
        {"type": "bottom", "color": "navy"},
    ])
    with patch("src.api.routes.analysis.get_segmenter", return_value=mock):
        yield mock


@pytest.fixture
def mock_color_analyzer():
    mock = MagicMock()
    mock.analyze_outfit_colors = MagicMock(return_value=0.85)
    mock.get_complementary_colors = MagicMock(
        return_value=["orange", "teal", "coral"]
    )
    with patch("src.api.routes.analysis.get_color_analyzer", return_value=mock):
        yield mock


# ============================================================================
# POST /analyze/image
# ============================================================================

class TestAnalyzeImage:

    @pytest.mark.asyncio
    async def test_analyze_image_with_url(self, mock_vision_service):
        result = await analyze_image(image_url="https://example.com/img.jpg", image=None)
        assert result["status"] == "success"
        assert "analysis" in result
        mock_vision_service.analyze_image.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_analyze_image_with_upload(self, mock_vision_service):
        upload = MagicMock(spec=UploadFile)
        upload.read = AsyncMock(return_value=b"fake image bytes")

        result = await analyze_image(image=upload)
        assert result["status"] == "success"
        upload.read.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_analyze_image_no_input_raises_400(self, mock_vision_service):
        with pytest.raises(HTTPException) as exc_info:
            await analyze_image(image_url=None, image=None)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_analyze_image_service_error_raises_500(self, mock_vision_service):
        mock_vision_service.analyze_image = AsyncMock(
            side_effect=RuntimeError("model error")
        )
        with pytest.raises(HTTPException) as exc_info:
            await analyze_image(image_url="http://x.com/a.jpg", image=None)
        assert exc_info.value.status_code == 500


# ============================================================================
# POST /analyze/outfit-image
# ============================================================================

class TestAnalyzeOutfitImage:

    @pytest.mark.asyncio
    async def test_outfit_image_with_url(self, mock_segmenter):
        result = await analyze_outfit_image(image_url="https://x.com/outfit.jpg", image=None)
        assert result["status"] == "success"
        assert result["items_found"] == 2

    @pytest.mark.asyncio
    async def test_outfit_image_no_input_raises_400(self, mock_segmenter):
        with pytest.raises(HTTPException) as exc_info:
            await analyze_outfit_image(image_url=None, image=None)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_outfit_image_service_error_raises_500(self, mock_segmenter):
        mock_segmenter.segment_outfit = AsyncMock(
            side_effect=RuntimeError("segmentation error")
        )
        with pytest.raises(HTTPException) as exc_info:
            await analyze_outfit_image(image_url="http://x.com/o.jpg", image=None)
        assert exc_info.value.status_code == 500


# ============================================================================
# POST /analyze/color-harmony
# ============================================================================

class TestCheckColorHarmony:

    @pytest.mark.asyncio
    async def test_color_harmony_success(self, mock_color_analyzer):
        result = await check_color_harmony(["navy", "white", "beige"])
        assert "harmony_score" in result
        assert result["harmony_score"] == 0.85
        assert result["recommendation"] == "good"

    @pytest.mark.asyncio
    async def test_color_harmony_low_score(self, mock_color_analyzer):
        mock_color_analyzer.analyze_outfit_colors.return_value = 0.4
        result = await check_color_harmony(["red", "green"])
        assert result["recommendation"] == "could be improved"

    @pytest.mark.asyncio
    async def test_color_harmony_needs_at_least_two(self, mock_color_analyzer):
        with pytest.raises(HTTPException) as exc_info:
            await check_color_harmony(["red"])
        assert exc_info.value.status_code == 400


# ============================================================================
# GET /analyze/complementary-colors/{color}
# ============================================================================

class TestGetComplementaryColors:

    @pytest.mark.asyncio
    async def test_complementary_colors(self, mock_color_analyzer):
        result = await get_complementary_colors("blue")
        assert result["input_color"] == "blue"
        assert isinstance(result["complementary_colors"], list)
        assert len(result["complementary_colors"]) == 3
