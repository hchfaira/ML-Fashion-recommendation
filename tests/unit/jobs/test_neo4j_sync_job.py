"""
Tests for Neo4jSyncJob (Phase 5)
==================================

Covers:
- SyncResult dataclass (fields, __str__)
- ConsistencyReport dataclass (fields, __str__)
- Neo4jSyncJob initialisation (defaults + custom params)
- sync_garments(): success, empty wardrobe, fallback, exception
- sync_incremental(): new/modified/deleted, empty combinations, fallback
- delete_garments(): success, empty list, client unavailable, exception
- run_consistency_check(): consistent, orphaned, missing, isolated, client unavailable
- repair_consistency(): deletes orphans + upserts missing, combined counts
- close(): resets _client and _builder, calls close on each
- _get_client() / _get_builder(): caching, fallback_enabled toggle

All Neo4j I/O is mocked — no live instance required.

Markers:
    @pytest.mark.unit
    @pytest.mark.asyncio
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.jobs.neo4j_sync_job import (
    Neo4jSyncJob,
    SyncResult,
    ConsistencyReport,
)
from src.core.models import Garment, GarmentAttributes, GarmentCategory, ColorProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_CONFIG = "src.jobs.neo4j_graph_builder.get_neo4j_config"
_PATCH_CLIENT = "src.jobs.neo4j_sync_job.Neo4jClient"
_PATCH_BUILDER = "src.jobs.neo4j_sync_job.Neo4jGraphBuilder"


@pytest.fixture(autouse=True)
def _patch_neo4j_config(monkeypatch):
    """Prevent get_neo4j_config() from reading disk."""
    mock_cfg = MagicMock()
    mock_cfg.get_min_compatibility_weight.return_value = 0.5
    mock_cfg.get_batch_size.return_value = 50
    monkeypatch.setattr(_PATCH_CONFIG, lambda *a, **kw: mock_cfg)


def _g(gid: str = "g1") -> Garment:
    return Garment(
        id=gid,
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            color=ColorProfile(primary="black"),
        ),
    )


def _wardrobe(n: int = 3) -> list[Garment]:
    return [_g(f"g{i}") for i in range(n)]


def _mock_builder(upserted: int = 3) -> MagicMock:
    b = MagicMock()
    b.load_all_garments = AsyncMock(return_value=upserted)
    b.close = AsyncMock()
    return b


def _mock_client(query_rows: list | None = None) -> MagicMock:
    c = MagicMock()
    c.query = AsyncMock(return_value=query_rows or [])
    c.close = AsyncMock()
    return c


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------


class TestSyncResult:
    """Tests for the SyncResult dataclass."""

    def test_default_values(self):
        r = SyncResult()
        assert r.success is False
        assert r.upserted == 0
        assert r.deleted_count == 0
        assert r.errors_count == 0
        assert r.duration_seconds == 0.0
        assert r.consistency_issues == []
        assert r.fallback_used is False

    def test_synced_at_is_iso(self):
        r = SyncResult()
        dt = datetime.fromisoformat(r.synced_at)
        assert dt.tzinfo is not None

    def test_str_ok(self):
        r = SyncResult(success=True, user_id="u1", upserted=5, deleted_count=2)
        s = str(r)
        assert "OK" in s
        assert "u1" in s
        assert "5" in s

    def test_str_failed(self):
        r = SyncResult(success=False, user_id="u2")
        assert "FAILED" in str(r)

    def test_str_fallback_marker(self):
        r = SyncResult(success=True, user_id="u1", fallback_used=True)
        assert "[fallback]" in str(r)

    def test_consistency_issues_independent(self):
        r1 = SyncResult()
        r2 = SyncResult()
        r1.consistency_issues.append("issue")
        assert r2.consistency_issues == []


# ---------------------------------------------------------------------------
# ConsistencyReport
# ---------------------------------------------------------------------------


class TestConsistencyReport:
    """Tests for the ConsistencyReport dataclass."""

    def test_default_values(self):
        r = ConsistencyReport()
        assert r.is_consistent is True
        assert r.orphaned_node_ids == []
        assert r.missing_node_ids == []
        assert r.edge_anomalies == []
        assert r.recommendations == []

    def test_checked_at_is_iso(self):
        r = ConsistencyReport()
        dt = datetime.fromisoformat(r.checked_at)
        assert dt.tzinfo is not None

    def test_str_consistent(self):
        r = ConsistencyReport(is_consistent=True, user_id="u1")
        assert "CONSISTENT" in str(r)
        assert "u1" in str(r)

    def test_str_inconsistent(self):
        r = ConsistencyReport(
            is_consistent=False, user_id="u1",
            orphaned_node_ids=["x"], missing_node_ids=["y"]
        )
        assert "INCONSISTENT" in str(r)
        assert "1" in str(r)

    def test_lists_independent(self):
        r1 = ConsistencyReport()
        r2 = ConsistencyReport()
        r1.orphaned_node_ids.append("x")
        assert r2.orphaned_node_ids == []


# ---------------------------------------------------------------------------
# Neo4jSyncJob — init
# ---------------------------------------------------------------------------


class TestNeo4jSyncJobInit:
    """Tests for Neo4jSyncJob constructor."""

    def test_defaults(self):
        job = Neo4jSyncJob()
        assert job.batch_size == 50
        assert job.fallback_enabled is True
        assert job.config_path is None
        assert job._client is None
        assert job._builder is None

    def test_custom_batch_size(self):
        job = Neo4jSyncJob(batch_size=100)
        assert job.batch_size == 100

    def test_fallback_disabled(self):
        job = Neo4jSyncJob(fallback_enabled=False)
        assert job.fallback_enabled is False

    def test_custom_config_path(self):
        job = Neo4jSyncJob(config_path="/tmp/cfg.json")
        assert job.config_path == "/tmp/cfg.json"


# ---------------------------------------------------------------------------
# sync_garments
# ---------------------------------------------------------------------------


class TestSyncGarments:
    """Tests for Neo4jSyncJob.sync_garments()."""

    @pytest.mark.asyncio
    async def test_success(self):
        mock_builder = _mock_builder(5)
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = mock_builder

        result = await job.sync_garments(_wardrobe(5), "u1")

        assert result.success is True
        assert result.upserted == 5
        assert result.user_id == "u1"
        mock_builder.load_all_garments.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_wardrobe(self):
        mock_builder = _mock_builder(0)
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = mock_builder

        result = await job.sync_garments([], "u1")

        assert result.success is True
        assert result.upserted == 0

    @pytest.mark.asyncio
    async def test_fallback_when_builder_none(self):
        job = Neo4jSyncJob(fallback_enabled=True)
        with patch(_PATCH_BUILDER, side_effect=Exception("no builder")):
            with patch(_PATCH_CLIENT, side_effect=Exception("no client")):
                result = await job.sync_garments(_wardrobe(2), "u1")

        assert result.success is True
        assert result.fallback_used is True

    @pytest.mark.asyncio
    async def test_builder_exception_sets_failure(self):
        mock_builder = MagicMock()
        mock_builder.load_all_garments = AsyncMock(
            side_effect=RuntimeError("crash")
        )
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = mock_builder

        result = await job.sync_garments(_wardrobe(2), "u1")

        assert result.success is False
        assert result.errors_count == 1

    @pytest.mark.asyncio
    async def test_duration_recorded(self):
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = _mock_builder(1)
        result = await job.sync_garments(_wardrobe(1), "u1")
        assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# sync_incremental
# ---------------------------------------------------------------------------


class TestSyncIncremental:
    """Tests for Neo4jSyncJob.sync_incremental()."""

    @pytest.mark.asyncio
    async def test_new_garments_upserted(self):
        new = _wardrobe(2)
        mock_builder = _mock_builder(2)
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client([{"deleted": 0}])
        job._builder = mock_builder

        result = await job.sync_incremental(new, [], [], "u1")

        assert result.success is True
        assert result.upserted == 2

    @pytest.mark.asyncio
    async def test_modified_garments_re_synced(self):
        full = _wardrobe(4)
        modified_ids = [full[1].id, full[2].id]
        mock_builder = _mock_builder(2)
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client([{"deleted": 0}])
        job._builder = mock_builder

        result = await job.sync_incremental(
            [], modified_ids, [], "u1", full_wardrobe=full
        )

        assert result.success is True
        assert result.upserted == 2

    @pytest.mark.asyncio
    async def test_deleted_garments_removed(self):
        deleted_ids = ["g10", "g11"]
        mock_client = _mock_client([{"deleted": 2}])
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = mock_client
        job._builder = _mock_builder(0)

        result = await job.sync_incremental([], [], deleted_ids, "u1")

        assert result.success is True
        assert result.deleted_count == 2

    @pytest.mark.asyncio
    async def test_all_empty_returns_success(self):
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = _mock_builder(0)

        result = await job.sync_incremental([], [], [], "u1")

        assert result.success is True
        assert result.upserted == 0
        assert result.deleted_count == 0

    @pytest.mark.asyncio
    async def test_fallback_when_builder_none(self):
        job = Neo4jSyncJob(fallback_enabled=True)
        with patch(_PATCH_BUILDER, side_effect=Exception("fail")):
            with patch(_PATCH_CLIENT, side_effect=Exception("fail")):
                result = await job.sync_incremental(_wardrobe(1), [], [], "u1")

        assert result.fallback_used is True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_combined_new_plus_deleted(self):
        new = _wardrobe(3)
        deleted = ["old1", "old2"]
        mock_client = _mock_client([{"deleted": 2}])
        mock_builder = _mock_builder(3)
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = mock_client
        job._builder = mock_builder

        result = await job.sync_incremental(new, [], deleted, "u1")

        assert result.upserted == 3
        assert result.deleted_count == 2
        assert result.success is True

    @pytest.mark.asyncio
    async def test_modified_without_full_wardrobe_skipped(self):
        """modified_ids given but full_wardrobe=None → skip modified sync."""
        mock_builder = _mock_builder(0)
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = mock_builder

        result = await job.sync_incremental(
            [], ["g1", "g2"], [], "u1", full_wardrobe=None
        )

        assert result.success is True
        # load_all_garments should not be called for modified (no wardrobe ref)
        mock_builder.load_all_garments.assert_not_awaited()


# ---------------------------------------------------------------------------
# delete_garments
# ---------------------------------------------------------------------------


class TestDeleteGarments:
    """Tests for Neo4jSyncJob.delete_garments()."""

    @pytest.mark.asyncio
    async def test_deletes_and_returns_count(self):
        client = _mock_client([{"deleted": 3}])
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        deleted = await job.delete_garments(["g1", "g2", "g3"], "u1")

        assert deleted == 3
        client.query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero_no_query(self):
        client = _mock_client()
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        deleted = await job.delete_garments([], "u1")

        assert deleted == 0
        client.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_client_unavailable_returns_zero(self):
        job = Neo4jSyncJob(fallback_enabled=True)
        with patch(_PATCH_CLIENT, side_effect=Exception("no client")):
            deleted = await job.delete_garments(["g1"], "u1")
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_query_exception_returns_zero(self):
        client = MagicMock()
        client.query = AsyncMock(side_effect=RuntimeError("db fail"))
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        deleted = await job.delete_garments(["g1"], "u1")

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_user_id_passed_to_query(self):
        client = _mock_client([{"deleted": 1}])
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        await job.delete_garments(["g1"], "user_xyz")

        call_params = client.query.call_args[0][1]
        assert call_params["uid"] == "user_xyz"

    @pytest.mark.asyncio
    async def test_garment_ids_passed_to_query(self):
        client = _mock_client([{"deleted": 2}])
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        await job.delete_garments(["a", "b"], "u1")

        call_params = client.query.call_args[0][1]
        assert "a" in call_params["ids"]
        assert "b" in call_params["ids"]


# ---------------------------------------------------------------------------
# run_consistency_check
# ---------------------------------------------------------------------------


class TestRunConsistencyCheck:
    """Tests for Neo4jSyncJob.run_consistency_check()."""

    @pytest.mark.asyncio
    async def test_consistent_state(self):
        wardrobe = _wardrobe(3)
        ids = [g.id for g in wardrobe]
        # Neo4j returns same set; no isolated nodes
        client = MagicMock()
        client.query = AsyncMock(
            side_effect=[
                [{"gid": gid} for gid in ids],  # garment IDs query
                [],  # no isolated nodes
            ]
        )
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        report = await job.run_consistency_check(wardrobe, "u1")

        assert report.is_consistent is True
        assert report.orphaned_node_ids == []
        assert report.missing_node_ids == []

    @pytest.mark.asyncio
    async def test_orphaned_nodes_detected(self):
        wardrobe = _wardrobe(2)
        ids = [g.id for g in wardrobe]
        # Neo4j has extra node not in wardrobe
        neo4j_ids = ids + ["orphan1"]
        client = MagicMock()
        client.query = AsyncMock(
            side_effect=[
                [{"gid": gid} for gid in neo4j_ids],
                [],  # no isolated
            ]
        )
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        report = await job.run_consistency_check(wardrobe, "u1")

        assert report.is_consistent is False
        assert "orphan1" in report.orphaned_node_ids
        assert len(report.recommendations) >= 1

    @pytest.mark.asyncio
    async def test_missing_nodes_detected(self):
        wardrobe = _wardrobe(3)
        # Neo4j is missing one garment
        ids = [g.id for g in wardrobe]
        neo4j_ids = ids[:2]
        client = MagicMock()
        client.query = AsyncMock(
            side_effect=[
                [{"gid": gid} for gid in neo4j_ids],
                [],
            ]
        )
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        report = await job.run_consistency_check(wardrobe, "u1")

        assert report.is_consistent is False
        assert ids[2] in report.missing_node_ids

    @pytest.mark.asyncio
    async def test_isolated_nodes_detected(self):
        wardrobe = _wardrobe(2)
        ids = [g.id for g in wardrobe]
        client = MagicMock()
        client.query = AsyncMock(
            side_effect=[
                [{"gid": gid} for gid in ids],  # garment query
                [{"gid": ids[0]}],               # isolated node
            ]
        )
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        report = await job.run_consistency_check(wardrobe, "u1")

        assert report.is_consistent is False
        assert len(report.edge_anomalies) >= 1

    @pytest.mark.asyncio
    async def test_client_unavailable_returns_recommendation(self):
        job = Neo4jSyncJob(fallback_enabled=True)
        with patch(_PATCH_CLIENT, side_effect=Exception("no client")):
            report = await job.run_consistency_check(_wardrobe(2), "u1")

        assert len(report.recommendations) >= 1

    @pytest.mark.asyncio
    async def test_query_exception_marks_inconsistent(self):
        client = MagicMock()
        client.query = AsyncMock(side_effect=RuntimeError("db crash"))
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        report = await job.run_consistency_check(_wardrobe(2), "u1")

        assert report.is_consistent is False

    @pytest.mark.asyncio
    async def test_empty_wardrobe_orphaned_all_neo4j_nodes(self):
        """Empty wardrobe → every Neo4j node is orphaned."""
        client = MagicMock()
        client.query = AsyncMock(
            side_effect=[
                [{"gid": "g1"}, {"gid": "g2"}],
                [],
            ]
        )
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        report = await job.run_consistency_check([], "u1")

        assert "g1" in report.orphaned_node_ids
        assert "g2" in report.orphaned_node_ids

    @pytest.mark.asyncio
    async def test_report_user_id_set(self):
        client = MagicMock()
        client.query = AsyncMock(side_effect=[[], []])
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = client

        report = await job.run_consistency_check([], "alice")

        assert report.user_id == "alice"


# ---------------------------------------------------------------------------
# repair_consistency
# ---------------------------------------------------------------------------


class TestRepairConsistency:
    """Tests for Neo4jSyncJob.repair_consistency()."""

    @pytest.mark.asyncio
    async def test_deletes_orphans_and_upserts_missing(self):
        wardrobe = _wardrobe(4)
        report = ConsistencyReport(
            is_consistent=False,
            orphaned_node_ids=["orphan1"],
            missing_node_ids=[wardrobe[3].id],
            user_id="u1",
        )

        mock_client = _mock_client([{"deleted": 1}])
        mock_builder = _mock_builder(1)

        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = mock_client
        job._builder = mock_builder

        result = await job.repair_consistency(report, wardrobe, "u1")

        assert result.success is True
        assert result.deleted_count == 1
        assert result.upserted == 1

    @pytest.mark.asyncio
    async def test_no_orphans_no_deletes(self):
        wardrobe = _wardrobe(2)
        report = ConsistencyReport(
            orphaned_node_ids=[],
            missing_node_ids=[wardrobe[0].id],
            user_id="u1",
        )
        mock_builder = _mock_builder(1)
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = mock_builder

        result = await job.repair_consistency(report, wardrobe, "u1")

        assert result.deleted_count == 0
        assert result.upserted == 1

    @pytest.mark.asyncio
    async def test_no_missing_no_upserts(self):
        wardrobe = _wardrobe(2)
        report = ConsistencyReport(
            orphaned_node_ids=["ghost"],
            missing_node_ids=[],
            user_id="u1",
        )
        mock_client = _mock_client([{"deleted": 1}])
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = mock_client
        job._builder = _mock_builder(0)

        result = await job.repair_consistency(report, wardrobe, "u1")

        assert result.deleted_count == 1
        assert result.upserted == 0

    @pytest.mark.asyncio
    async def test_all_clean_report_is_noop(self):
        report = ConsistencyReport(is_consistent=True)
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = _mock_builder(0)

        result = await job.repair_consistency(report, _wardrobe(2), "u1")

        assert result.success is True
        assert result.upserted == 0
        assert result.deleted_count == 0

    @pytest.mark.asyncio
    async def test_exception_sets_failure(self):
        report = ConsistencyReport(missing_node_ids=["g1"])
        wardrobe = [_g("g1")]

        mock_builder = MagicMock()
        mock_builder.load_all_garments = AsyncMock(
            side_effect=RuntimeError("crash")
        )
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = _mock_client()
        job._builder = mock_builder

        result = await job.repair_consistency(report, wardrobe, "u1")

        assert result.success is False
        assert result.errors_count >= 1


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestNeo4jSyncJobClose:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_resets_client_and_builder(self):
        job = Neo4jSyncJob()
        job._client = _mock_client()
        job._builder = _mock_builder()
        await job.close()
        assert job._client is None
        assert job._builder is None

    @pytest.mark.asyncio
    async def test_close_with_nothing_held(self):
        job = Neo4jSyncJob()
        await job.close()  # no error

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self):
        c = _mock_client()
        job = Neo4jSyncJob()
        job._client = c
        await job.close()
        c.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_calls_builder_close(self):
        b = _mock_builder()
        job = Neo4jSyncJob()
        job._builder = b
        await job.close()
        b.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# _get_client / _get_builder
# ---------------------------------------------------------------------------


class TestGetClientBuilder:
    """Tests for internal caching and fallback logic."""

    def test_get_client_returns_cached(self):
        job = Neo4jSyncJob(fallback_enabled=False)
        c = _mock_client()
        job._client = c
        assert job._get_client() is c

    def test_get_builder_returns_cached(self):
        job = Neo4jSyncJob(fallback_enabled=False)
        b = _mock_builder()
        job._builder = b
        assert job._get_builder() is b

    def test_get_client_fallback_returns_none_on_exception(self):
        job = Neo4jSyncJob(fallback_enabled=True)
        with patch(_PATCH_CLIENT, side_effect=Exception("fail")):
            c = job._get_client()
        assert c is None

    def test_get_builder_fallback_returns_none_on_exception(self):
        job = Neo4jSyncJob(fallback_enabled=True)
        with patch(_PATCH_BUILDER, side_effect=Exception("fail")):
            b = job._get_builder()
        assert b is None

    def test_get_builder_shares_client(self):
        existing = _mock_client()
        job = Neo4jSyncJob(fallback_enabled=False)
        job._client = existing

        with patch(_PATCH_BUILDER) as mock_cls:
            # patch() auto-creates mock_cls.return_value — that's what _get_builder()
            # receives when it calls Neo4jGraphBuilder(...).
            job._get_builder()
            built = mock_cls.return_value  # the actual object _get_builder created

        assert job._builder is built
        # The implementation does: self._builder._client = self._client
        assert built._client is existing
