"""
Tests for CentralityRefreshJob (Phase 4)
==========================================

Covers:
- CentralityRefreshResult dataclass (fields, summary)
- CentralityRefreshJob initialisation (defaults + custom params)
- run() — success path, no garments, fetch_from_neo4j flag
- run() — fallback when Neo4j is unavailable
- run() — builder raises exception mid-run
- run() — on_complete sync + async callbacks
- run_for_all_users() — multi-user sequential run, partial failure isolation
- _get_client() — fallback_enabled=True vs False
- _get_builder() — caches builder, shares client
- _fetch_garments_from_neo4j() — happy path, empty rows, exception
- close() — resets _client and _builder
- Integration smoke: builder.update_centrality_scores called with correct args

All Neo4j I/O is mocked — no live Neo4j instance required.

Markers:
    @pytest.mark.unit    — fast, in-process
    @pytest.mark.asyncio — async tests (mode=AUTO)
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timezone

from src.jobs.centrality_refresh_job import (
    CentralityRefreshJob,
    CentralityRefreshResult,
)
from src.core.models import Garment, GarmentAttributes, GarmentCategory, ColorProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_CONFIG = "src.jobs.neo4j_graph_builder.get_neo4j_config"
_PATCH_CLIENT = "src.jobs.centrality_refresh_job.Neo4jClient"
_PATCH_BUILDER = "src.jobs.centrality_refresh_job.Neo4jGraphBuilder"


@pytest.fixture(autouse=True)
def _patch_neo4j_config(monkeypatch):
    """Prevent get_neo4j_config() from reading disk in every builder instantiation."""
    mock_cfg = MagicMock()
    mock_cfg.get_min_compatibility_weight.return_value = 0.5
    mock_cfg.get_batch_size.return_value = 50
    monkeypatch.setattr(_PATCH_CONFIG, lambda *a, **kw: mock_cfg)


def _g(gid: str = "g1") -> Garment:
    return Garment(
        id=gid,
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorProfile(primary="blue"),
        ),
    )


def _wardrobe(n: int = 3) -> list[Garment]:
    return [_g(f"g{i}") for i in range(n)]


def _mock_builder(updated: int = 3) -> MagicMock:
    builder = MagicMock()
    builder.update_centrality_scores = AsyncMock(return_value=updated)
    builder.close = AsyncMock()
    return builder


def _mock_client(garment_ids: list[str] | None = None) -> MagicMock:
    client = MagicMock()
    rows = [{"garment_id": gid} for gid in (garment_ids or [])]
    client.query = AsyncMock(return_value=rows)
    client.close = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# CentralityRefreshResult
# ---------------------------------------------------------------------------


class TestCentralityRefreshResult:
    """Tests for the CentralityRefreshResult dataclass."""

    def test_default_values(self):
        r = CentralityRefreshResult()
        assert r.success is False
        assert r.user_id == ""
        assert r.garments_updated == 0
        assert r.duration_seconds == 0.0
        assert r.fallback_used is False
        assert r.errors == []

    def test_ran_at_is_iso_string(self):
        r = CentralityRefreshResult()
        # Should parse without error
        dt = datetime.fromisoformat(r.ran_at)
        assert dt.tzinfo is not None

    def test_summary_success(self):
        r = CentralityRefreshResult(
            success=True, user_id="u1", garments_updated=10, duration_seconds=2.5
        )
        s = r.summary()
        assert "OK" in s
        assert "u1" in s
        assert "10" in s
        assert "2.50" in s

    def test_summary_failure(self):
        r = CentralityRefreshResult(success=False, user_id="u1")
        assert "FAILED" in r.summary()

    def test_summary_fallback_marker(self):
        r = CentralityRefreshResult(success=True, user_id="u1", fallback_used=True)
        assert "[fallback]" in r.summary()

    def test_summary_no_fallback_marker_when_false(self):
        r = CentralityRefreshResult(success=True, user_id="u1", fallback_used=False)
        assert "[fallback]" not in r.summary()

    def test_errors_list_independent(self):
        r1 = CentralityRefreshResult()
        r2 = CentralityRefreshResult()
        r1.errors.append("oops")
        assert r2.errors == []


# ---------------------------------------------------------------------------
# CentralityRefreshJob — init
# ---------------------------------------------------------------------------


class TestCentralityRefreshJobInit:
    """Tests for CentralityRefreshJob constructor."""

    def test_defaults(self):
        job = CentralityRefreshJob()
        assert job.fallback_enabled is True
        assert job.config_path is None
        assert job.on_complete is None
        assert job._client is None
        assert job._builder is None

    def test_custom_config_path(self):
        job = CentralityRefreshJob(config_path="/tmp/neo4j.json")
        assert job.config_path == "/tmp/neo4j.json"

    def test_fallback_disabled(self):
        job = CentralityRefreshJob(fallback_enabled=False)
        assert job.fallback_enabled is False

    def test_on_complete_stored(self):
        cb = lambda r: None
        job = CentralityRefreshJob(on_complete=cb)
        assert job.on_complete is cb


# ---------------------------------------------------------------------------
# CentralityRefreshJob.run() — happy path
# ---------------------------------------------------------------------------


class TestCentralityRefreshJobRun:
    """Tests for CentralityRefreshJob.run()."""

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Builder called with garments; returns success result."""
        garments = _wardrobe(3)
        mock_builder = _mock_builder(updated=3)
        mock_client = _mock_client()

        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = mock_client
        job._builder = mock_builder

        result = await job.run("u1", garments=garments)

        assert result.success is True
        assert result.garments_updated == 3
        assert result.user_id == "u1"
        assert result.fallback_used is False
        mock_builder.update_centrality_scores.assert_awaited_once_with(garments)

    @pytest.mark.asyncio
    async def test_run_no_garments_returns_success(self):
        """Empty garment list skips update_centrality_scores."""
        mock_builder = _mock_builder()
        mock_client = _mock_client()

        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = mock_client
        job._builder = mock_builder

        result = await job.run("u1", garments=[])

        assert result.success is True
        assert result.garments_updated == 0
        mock_builder.update_centrality_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_none_garments_not_fetch_returns_success(self):
        """garments=None without fetch_from_neo4j: no-op success."""
        mock_builder = _mock_builder()
        mock_client = _mock_client()

        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = mock_client
        job._builder = mock_builder

        result = await job.run("u1", garments=None, fetch_from_neo4j=False)

        assert result.success is True
        mock_builder.update_centrality_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_fetch_from_neo4j(self):
        """With fetch_from_neo4j=True, garments fetched from client."""
        client = _mock_client(garment_ids=["g1", "g2"])
        mock_builder = _mock_builder(updated=2)

        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = client
        job._builder = mock_builder

        result = await job.run("u1", garments=None, fetch_from_neo4j=True)

        assert result.success is True
        assert result.garments_updated == 2
        # Should have called query to fetch garments
        client.query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_duration_recorded(self):
        """duration_seconds is positive after a run."""
        mock_builder = _mock_builder(3)
        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = mock_builder

        result = await job.run("u1", garments=_wardrobe(1))
        assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# CentralityRefreshJob.run() — fallback / error paths
# ---------------------------------------------------------------------------


class TestCentralityRefreshJobFallback:
    """Fallback and error-path tests."""

    @pytest.mark.asyncio
    async def test_fallback_when_client_none(self):
        """No-op success when _get_client returns None."""
        job = CentralityRefreshJob(fallback_enabled=True)
        # Prevent any real client creation
        job._client = None

        with patch(_PATCH_CLIENT, side_effect=Exception("no neo4j")):
            result = await job.run("u1", garments=_wardrobe(2))

        assert result.success is True
        assert result.fallback_used is True
        assert result.garments_updated == 0

    @pytest.mark.asyncio
    async def test_builder_exception_sets_failure(self):
        """Exception from builder.update_centrality_scores → success=False."""
        mock_builder = MagicMock()
        mock_builder.update_centrality_scores = AsyncMock(
            side_effect=RuntimeError("db error")
        )
        mock_client = _mock_client()

        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = mock_client
        job._builder = mock_builder

        result = await job.run("u1", garments=_wardrobe(2))

        assert result.success is False
        assert len(result.errors) == 1
        assert "db error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_duration_recorded_on_failure(self):
        """Duration is still populated even when builder raises."""
        mock_builder = MagicMock()
        mock_builder.update_centrality_scores = AsyncMock(
            side_effect=RuntimeError("fail")
        )
        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = mock_builder

        result = await job.run("u1", garments=_wardrobe(1))
        assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# CentralityRefreshJob — on_complete callbacks
# ---------------------------------------------------------------------------


class TestCentralityRefreshJobCallbacks:
    """Tests for on_complete sync and async callback behaviour."""

    @pytest.mark.asyncio
    async def test_sync_callback_called_on_success(self):
        called_with = []

        def cb(r):
            called_with.append(r)

        job = CentralityRefreshJob(fallback_enabled=False, on_complete=cb)
        job._client = _mock_client()
        job._builder = _mock_builder(2)

        result = await job.run("u1", garments=_wardrobe(2))

        assert result.success is True
        assert len(called_with) == 1
        assert called_with[0] is result

    @pytest.mark.asyncio
    async def test_async_callback_called_on_success(self):
        called_with = []

        async def cb(r):
            called_with.append(r)

        job = CentralityRefreshJob(fallback_enabled=False, on_complete=cb)
        job._client = _mock_client()
        job._builder = _mock_builder(2)

        result = await job.run("u1", garments=_wardrobe(2))

        assert result.success is True
        assert len(called_with) == 1

    @pytest.mark.asyncio
    async def test_callback_not_called_on_failure(self):
        called = []

        def cb(r):
            called.append(r)

        mock_builder = MagicMock()
        mock_builder.update_centrality_scores = AsyncMock(
            side_effect=RuntimeError("fail")
        )
        job = CentralityRefreshJob(fallback_enabled=False, on_complete=cb)
        job._client = _mock_client()
        job._builder = mock_builder

        await job.run("u1", garments=_wardrobe(2))
        assert called == []

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_propagate(self):
        """A buggy callback should not crash the job."""

        def bad_cb(r):
            raise ValueError("callback bug")

        job = CentralityRefreshJob(fallback_enabled=False, on_complete=bad_cb)
        job._client = _mock_client()
        job._builder = _mock_builder(1)

        # Should not raise
        result = await job.run("u1", garments=_wardrobe(1))
        assert result.success is True


# ---------------------------------------------------------------------------
# CentralityRefreshJob.run_for_all_users()
# ---------------------------------------------------------------------------


class TestRunForAllUsers:
    """Tests for multi-user batch run."""

    @pytest.mark.asyncio
    async def test_all_users_run(self):
        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = _mock_builder(3)

        wardrobe_map = {
            "u1": _wardrobe(3),
            "u2": _wardrobe(2),
        }
        results = await job.run_for_all_users(wardrobe_map)

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_one_failure_does_not_abort_others(self):
        """Second user still runs even if first builder call fails."""
        call_count = 0

        async def _side_effect(garments):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first user fails")
            return len(garments)

        mock_builder = MagicMock()
        mock_builder.update_centrality_scores = AsyncMock(side_effect=_side_effect)

        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = mock_builder

        results = await job.run_for_all_users({"u1": _wardrobe(2), "u2": _wardrobe(2)})

        assert len(results) == 2
        assert results[0].success is False
        assert results[1].success is True

    @pytest.mark.asyncio
    async def test_empty_map_returns_empty_list(self):
        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = _mock_builder()
        results = await job.run_for_all_users({})
        assert results == []


# ---------------------------------------------------------------------------
# _get_client / _get_builder
# ---------------------------------------------------------------------------


class TestGetClientBuilder:
    """Tests for internal client/builder caching logic."""

    def test_get_client_caches(self):
        job = CentralityRefreshJob(fallback_enabled=False)
        mock_c = MagicMock()
        job._client = mock_c
        assert job._get_client() is mock_c

    def test_get_builder_caches(self):
        job = CentralityRefreshJob(fallback_enabled=False)
        mock_b = MagicMock()
        job._builder = mock_b
        assert job._get_builder() is mock_b

    def test_get_builder_creates_if_none(self):
        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = _mock_client()

        with patch(_PATCH_BUILDER) as mock_cls:
            builder = job._get_builder()
            # patch() auto-creates mock_cls.return_value
            assert builder is mock_cls.return_value

        assert job._builder is mock_cls.return_value

    def test_get_builder_shares_existing_client(self):
        """When a client already exists, it should be set on the builder."""
        existing_client = _mock_client()
        job = CentralityRefreshJob(fallback_enabled=False)
        job._client = existing_client

        with patch(_PATCH_BUILDER) as mock_cls:
            job._get_builder()
            built = mock_cls.return_value  # actual object returned by Neo4jGraphBuilder(...)

        assert built._client is existing_client

    def test_get_client_fallback_returns_none_on_exception(self):
        """With fallback_enabled=True, a creation exception → None."""
        job = CentralityRefreshJob(fallback_enabled=True)
        with patch(_PATCH_CLIENT, side_effect=Exception("no neo4j")):
            client = job._get_client()
        assert client is None


# ---------------------------------------------------------------------------
# _fetch_garments_from_neo4j
# ---------------------------------------------------------------------------


class TestFetchGarmentsFromNeo4j:
    """Tests for the static garment-fetching helper."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_stubs(self):
        client = _mock_client(garment_ids=["g1", "g2", "g3"])
        stubs = await CentralityRefreshJob._fetch_garments_from_neo4j(client, "u1")
        assert len(stubs) == 3
        assert {g.id for g in stubs} == {"g1", "g2", "g3"}

    @pytest.mark.asyncio
    async def test_empty_rows_returns_empty_list(self):
        client = _mock_client(garment_ids=[])
        stubs = await CentralityRefreshJob._fetch_garments_from_neo4j(client, "u1")
        assert stubs == []

    @pytest.mark.asyncio
    async def test_rows_with_no_garment_id_filtered(self):
        client = MagicMock()
        client.query = AsyncMock(
            return_value=[{"garment_id": "g1"}, {"garment_id": None}, {}]
        )
        stubs = await CentralityRefreshJob._fetch_garments_from_neo4j(client, "u1")
        assert len(stubs) == 1

    @pytest.mark.asyncio
    async def test_query_exception_returns_empty_list(self):
        from src.core.neo4j_client import Neo4jException

        client = MagicMock()
        client.query = AsyncMock(side_effect=Neo4jException("boom"))
        stubs = await CentralityRefreshJob._fetch_garments_from_neo4j(client, "u1")
        assert stubs == []


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestCentralityRefreshJobClose:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_resets_client_and_builder(self):
        job = CentralityRefreshJob()
        job._client = _mock_client()
        job._builder = _mock_builder()
        await job.close()
        assert job._client is None
        assert job._builder is None

    @pytest.mark.asyncio
    async def test_close_when_nothing_held(self):
        """close() with nothing initialised should not raise."""
        job = CentralityRefreshJob()
        await job.close()  # no error

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self):
        mock_c = _mock_client()
        job = CentralityRefreshJob()
        job._client = mock_c
        await job.close()
        mock_c.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_calls_builder_close(self):
        mock_b = _mock_builder()
        job = CentralityRefreshJob()
        job._builder = mock_b
        await job.close()
        mock_b.close.assert_awaited_once()
