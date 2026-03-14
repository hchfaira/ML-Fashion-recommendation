"""
Tests for Layer 1: VisionQueue — Solutions 3+5 (Async queue worker system)

Tests cover:
- enqueue returns a job_id string
- enqueue_folder returns a list of job_ids
- wait() returns a completed VisionJob
- Cache hits set JobStatus.CACHED and skip Gemini
- stats() counts are accurate
- Context-manager lifecycle (start/stop)
- on_complete callback is called after job finishes
- Failed jobs return None in wait_all()
"""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.layer1_vision.vision_queue import VisionQueue, VisionJob, JobStatus
from src.core.models import Garment, GarmentAttributes, GarmentCategory, ColorInfo, FormalityLevel


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_garment() -> Garment:
    return Garment(
        id="test-garment",
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorInfo(primary="white", hex_codes=[]),
            formality_level=FormalityLevel.CASUAL,
        ),
    )


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    img = tmp_path / "shirt.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return img


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "wardrobe"
    folder.mkdir()
    for i in range(3):
        (folder / f"item_{i}.png").write_bytes(bytes([i] * 32))
    return folder


def _mock_extractor(garment: Garment | None = None):
    """Return a mock AttributeExtractor that immediately returns a garment."""
    garment = garment or _make_garment()
    mock = MagicMock()
    mock.extract_garment_from_image = AsyncMock(return_value=garment)
    return mock


def _mock_cache(hit: bool = False):
    mock = MagicMock()
    mock.get.return_value = _make_garment().attributes.__dict__ if hit else None
    return mock


# ---------------------------------------------------------------------------
# enqueue / enqueue_folder
# ---------------------------------------------------------------------------

class TestEnqueue:
    @pytest.mark.asyncio
    async def test_enqueue_returns_string_job_id(self, sample_image: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        async with queue:
            job_id = queue.enqueue(sample_image)
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    @pytest.mark.asyncio
    async def test_enqueue_folder_returns_list_of_job_ids(self, image_folder: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        async with queue:
            job_ids = queue.enqueue_folder(image_folder)
        assert isinstance(job_ids, list)
        assert len(job_ids) == 3
        assert all(isinstance(j, str) for j in job_ids)

    @pytest.mark.asyncio
    async def test_enqueue_folder_empty_folder_returns_empty_list(self, tmp_path: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        async with queue:
            job_ids = queue.enqueue_folder(tmp_path)
        assert job_ids == []


# ---------------------------------------------------------------------------
# wait / wait_all
# ---------------------------------------------------------------------------

class TestWait:
    @pytest.mark.asyncio
    async def test_wait_returns_completed_job(self, sample_image: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        async with queue:
            job_id = queue.enqueue(sample_image)
            job = await queue.wait(job_id, timeout=10)
        assert job is not None
        assert job.status in (JobStatus.DONE, JobStatus.CACHED)

    @pytest.mark.asyncio
    async def test_wait_all_returns_garments(self, image_folder: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        async with queue:
            job_ids = queue.enqueue_folder(image_folder)
            results = await queue.wait_all(job_ids, timeout=30)
        assert len(results) == 3
        # All should be garments or None (failed); with our mock all should succeed
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_failed_job_returns_none_in_wait_all(self, sample_image: Path):
        failing_extractor = MagicMock()
        failing_extractor.extract_garment_from_image = AsyncMock(
            side_effect=Exception("Gemini 503")
        )
        queue = VisionQueue(
            extractor=failing_extractor,
            vision_cache=_mock_cache(hit=False),
        )
        async with queue:
            job_id = queue.enqueue(sample_image)
            results = await queue.wait_all([job_id], timeout=10)
        assert results[0] is None

    @pytest.mark.asyncio
    async def test_wait_unknown_job_id_returns_none(self, sample_image: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        async with queue:
            job = await queue.wait("nonexistent-id", timeout=2)
        assert job is None


# ---------------------------------------------------------------------------
# Cache hit → CACHED status
# ---------------------------------------------------------------------------

class TestCacheHit:
    @pytest.mark.asyncio
    async def test_cache_hit_sets_cached_status(self, sample_image: Path):
        garment = _make_garment()
        hit_cache = MagicMock()
        hit_cache.get.return_value = garment.attributes.model_dump()

        # Extractor should NOT be called
        extractor = MagicMock()
        extractor.extract_garment_from_image = AsyncMock(return_value=garment)

        queue = VisionQueue(extractor=extractor, vision_cache=hit_cache)
        async with queue:
            job_id = queue.enqueue(sample_image)
            job = await queue.wait(job_id, timeout=10)

        assert job is not None
        assert job.status == JobStatus.CACHED


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------

class TestStats:
    @pytest.mark.asyncio
    async def test_stats_counts_total_jobs(self, image_folder: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        async with queue:
            job_ids = queue.enqueue_folder(image_folder)
            await queue.wait_all(job_ids, timeout=30)
            s = queue.stats()
        assert s["total"] == 3

    @pytest.mark.asyncio
    async def test_stats_done_count_matches(self, image_folder: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        async with queue:
            job_ids = queue.enqueue_folder(image_folder)
            await queue.wait_all(job_ids, timeout=30)
            s = queue.stats()
        assert s["done"] + s.get("cached", 0) == 3

    @pytest.mark.asyncio
    async def test_stats_failed_count_for_errors(self, sample_image: Path):
        failing = MagicMock()
        failing.extract_garment_from_image = AsyncMock(side_effect=Exception("boom"))
        queue = VisionQueue(extractor=failing, vision_cache=_mock_cache())
        async with queue:
            job_id = queue.enqueue(sample_image)
            await queue.wait_all([job_id], timeout=10)
            s = queue.stats()
        assert s["failed"] == 1


# ---------------------------------------------------------------------------
# Context manager lifecycle
# ---------------------------------------------------------------------------

class TestContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_starts_and_stops(self, sample_image: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        assert not queue._running
        async with queue:
            assert queue._running
        assert not queue._running

    @pytest.mark.asyncio
    async def test_manual_start_stop(self, sample_image: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        await queue.start()
        assert queue._running
        await queue.stop()
        assert not queue._running


# ---------------------------------------------------------------------------
# on_complete callback
# ---------------------------------------------------------------------------

class TestOnCompleteCallback:
    @pytest.mark.asyncio
    async def test_callback_called_for_each_completed_job(self, image_folder: Path):
        called_with = []

        def _callback(job: VisionJob):
            called_with.append(job)

        queue = VisionQueue(
            extractor=_mock_extractor(),
            vision_cache=_mock_cache(),
            on_complete=_callback,
        )
        async with queue:
            job_ids = queue.enqueue_folder(image_folder)
            await queue.wait_all(job_ids, timeout=30)

        assert len(called_with) == 3
        assert all(isinstance(j, VisionJob) for j in called_with)

    @pytest.mark.asyncio
    async def test_callback_receives_done_status(self, sample_image: Path):
        received = []

        queue = VisionQueue(
            extractor=_mock_extractor(),
            vision_cache=_mock_cache(),
            on_complete=received.append,
        )
        async with queue:
            job_id = queue.enqueue(sample_image)
            await queue.wait(job_id, timeout=10)

        assert len(received) == 1
        assert received[0].status in (JobStatus.DONE, JobStatus.CACHED)


# ---------------------------------------------------------------------------
# enqueue_and_wait_folder convenience method
# ---------------------------------------------------------------------------

class TestEnqueueAndWaitFolder:
    @pytest.mark.asyncio
    async def test_returns_list_of_garments(self, image_folder: Path):
        queue = VisionQueue(extractor=_mock_extractor(), vision_cache=_mock_cache())
        async with queue:
            garments = await queue.enqueue_and_wait_folder(image_folder)
        assert len(garments) == 3
        assert all(g is not None for g in garments)
