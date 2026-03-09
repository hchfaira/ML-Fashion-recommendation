"""
Tests for Wardrobe API Routes
================================

Covers:
- POST   /wardrobe/items           (add garment)
- GET    /wardrobe/items            (get wardrobe)
- GET    /wardrobe/items/{id}       (get single garment)
- DELETE /wardrobe/items/{id}       (remove garment)
- POST   /wardrobe/bulk-upload
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, UploadFile

from src.core.models import Garment, GarmentAttributes, GarmentCategory, ColorProfile

from src.api.routes.wardrobe import (
    add_garment,
    get_wardrobe,
    get_garment,
    remove_garment,
    bulk_upload,
    _wardrobes,
)


# ============================================================================
# Fixtures
# ============================================================================

def _make_analysis():
    return {"category": "top", "color": "red", "pattern": "solid"}


def _make_attributes():
    return GarmentAttributes(
        category=GarmentCategory.TOP,
        color=ColorProfile(primary="red", hex_codes=[]),
    )


@pytest.fixture(autouse=True)
def clear_wardrobes():
    """Clear the in-memory wardrobe store before each test."""
    _wardrobes.clear()
    yield
    _wardrobes.clear()


@pytest.fixture
def mock_vision_service():
    mock = MagicMock()
    mock.analyze_image = AsyncMock(return_value=_make_analysis())
    with patch("src.api.routes.wardrobe.get_vision_service", return_value=mock):
        yield mock


@pytest.fixture
def mock_attribute_extractor():
    mock = MagicMock()
    mock.extract_attributes = AsyncMock(return_value=_make_attributes())
    with patch(
        "src.api.routes.wardrobe.get_attribute_extractor", return_value=mock
    ):
        yield mock


@pytest.fixture
def mock_embedding_generator():
    mock = MagicMock()
    mock.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    with patch(
        "src.api.routes.wardrobe.get_embedding_generator", return_value=mock
    ):
        yield mock


@pytest.fixture
def all_wardrobe_mocks(
    mock_vision_service, mock_attribute_extractor, mock_embedding_generator
):
    return {
        "vision": mock_vision_service,
        "extractor": mock_attribute_extractor,
        "embedding": mock_embedding_generator,
    }


# ============================================================================
# POST /wardrobe/items
# ============================================================================

class TestAddGarment:

    @pytest.mark.asyncio
    async def test_add_garment_with_url(self, all_wardrobe_mocks):
        result = await add_garment(
            user_id="user_1", image_url="https://example.com/top.jpg", image=None
        )
        assert isinstance(result, Garment)
        assert result.id.startswith("garment_")
        assert "user_1" in _wardrobes
        assert len(_wardrobes["user_1"]) == 1

    @pytest.mark.asyncio
    async def test_add_garment_with_upload(self, all_wardrobe_mocks):
        upload = MagicMock(spec=UploadFile)
        upload.read = AsyncMock(return_value=b"fake bytes")

        result = await add_garment(user_id="user_2", image=upload)
        assert isinstance(result, Garment)
        upload.read.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_garment_no_input_raises_400(self, all_wardrobe_mocks):
        with pytest.raises(HTTPException) as exc_info:
            await add_garment(user_id="u", image_url=None, image=None)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_add_garment_service_failure_raises_500(
        self, all_wardrobe_mocks
    ):
        all_wardrobe_mocks["vision"].analyze_image = AsyncMock(
            side_effect=RuntimeError("vision model crash")
        )
        with pytest.raises(HTTPException) as exc_info:
            await add_garment(
                user_id="u", image_url="https://x.com/a.jpg", image=None
            )
        assert exc_info.value.status_code == 500


# ============================================================================
# GET /wardrobe/items
# ============================================================================

class TestGetWardrobe:

    @pytest.mark.asyncio
    async def test_empty_wardrobe(self):
        result = await get_wardrobe(user_id="nobody")
        assert result == []

    @pytest.mark.asyncio
    async def test_wardrobe_with_items(self, all_wardrobe_mocks):
        await add_garment(user_id="u1", image_url="http://x.com/a.jpg", image=None)
        await add_garment(user_id="u1", image_url="http://x.com/b.jpg", image=None)

        result = await get_wardrobe(user_id="u1")
        assert len(result) == 2


# ============================================================================
# GET /wardrobe/items/{garment_id}
# ============================================================================

class TestGetGarment:

    @pytest.mark.asyncio
    async def test_get_existing_garment(self, all_wardrobe_mocks):
        garment = await add_garment(
            user_id="u1", image_url="http://x.com/a.jpg", image=None
        )
        result = await get_garment(user_id="u1", garment_id=garment.id)
        assert result.id == garment.id

    @pytest.mark.asyncio
    async def test_get_missing_garment_raises_404(self):
        with pytest.raises(HTTPException) as exc_info:
            await get_garment(user_id="u1", garment_id="nonexistent")
        assert exc_info.value.status_code == 404


# ============================================================================
# DELETE /wardrobe/items/{garment_id}
# ============================================================================

class TestRemoveGarment:

    @pytest.mark.asyncio
    async def test_remove_garment(self, all_wardrobe_mocks):
        garment = await add_garment(
            user_id="u1", image_url="http://x.com/a.jpg", image=None
        )
        result = await remove_garment(user_id="u1", garment_id=garment.id)
        assert result["status"] == "deleted"
        assert len(_wardrobes["u1"]) == 0

    @pytest.mark.asyncio
    async def test_remove_from_missing_wardrobe_raises_404(self):
        with pytest.raises(HTTPException) as exc_info:
            await remove_garment(user_id="nobody", garment_id="g1")
        assert exc_info.value.status_code == 404


# ============================================================================
# POST /wardrobe/bulk-upload
# ============================================================================

class TestBulkUpload:

    @pytest.mark.asyncio
    async def test_bulk_upload_success(self, all_wardrobe_mocks):
        files = []
        for i in range(3):
            f = MagicMock(spec=UploadFile)
            f.filename = f"image_{i}.jpg"
            f.read = AsyncMock(return_value=b"data")
            files.append(f)

        result = await bulk_upload(user_id="u1", images=files)
        assert len(result["successful"]) == 3
        assert len(result["failed"]) == 0

    @pytest.mark.asyncio
    async def test_bulk_upload_partial_failure(self, all_wardrobe_mocks):
        """Second image fails, first and third succeed."""
        call_count = 0

        async def flaky_analyze(source):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("corrupt image")
            return _make_analysis()

        all_wardrobe_mocks["vision"].analyze_image = AsyncMock(
            side_effect=flaky_analyze
        )

        files = []
        for i in range(3):
            f = MagicMock(spec=UploadFile)
            f.filename = f"image_{i}.jpg"
            f.read = AsyncMock(return_value=b"data")
            files.append(f)

        result = await bulk_upload(user_id="u1", images=files)
        assert len(result["successful"]) == 2
        assert len(result["failed"]) == 1
        assert result["failed"][0]["filename"] == "image_1.jpg"
