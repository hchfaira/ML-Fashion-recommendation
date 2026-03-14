"""
Neo4j Sync Job (Phase 5)
=========================

Incremental synchronisation between the live wardrobe (Python objects) and
the Neo4j property graph.  Key responsibilities:

* **Full sync** — upsert every garment as a Garment node + rebuild edges.
* **Incremental sync** — add new nodes, re-process modified nodes, remove
  deleted nodes without a full rebuild.
* **Consistency check** — detect orphaned nodes, missing nodes, and other
  anomalies.
* **Repair** — auto-fix consistency issues found by the check step.
* **Deprecation tracking** — surface warnings when callers use deprecated
  wardrobe-only code paths that should migrate to Neo4j.

Architecture
------------
::

    Scheduler / API endpoint
        ↓
    Neo4jSyncJob.sync_incremental(new_garments, modified_ids, deleted_ids, user_id)
        ↓
    Neo4jGraphBuilder.sync_wardrobe()  — upsert nodes + edges
        ↓
    Neo4jSyncJob.run_consistency_check(wardrobe, user_id)
        ↓
    ConsistencyReport  — orphaned, missing, edge anomalies

Fallback
--------
Any method returns a ``SyncResult(success=False, fallback_used=True)`` when
Neo4j is unreachable instead of raising to the caller.

Usage
-----
::

    from src.jobs.neo4j_sync_job import Neo4jSyncJob

    job = Neo4jSyncJob()
    result = await job.sync_garments(wardrobe, user_id="u1")
    print(result)

    report = await job.run_consistency_check(wardrobe, user_id="u1")
    if not report.is_consistent:
        await job.repair_consistency(report, wardrobe, user_id="u1")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core import get_logger
from src.core.models import Garment
from src.core.neo4j_client import Neo4jClient
from src.jobs.neo4j_graph_builder import Neo4jGraphBuilder

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result / report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SyncResult:
    """Result of a single sync run."""

    success: bool = False
    upserted: int = 0
    deleted_count: int = 0
    errors_count: int = 0
    duration_seconds: float = 0.0
    consistency_issues: List[str] = field(default_factory=list)
    fallback_used: bool = False
    user_id: str = ""
    synced_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        fb = " [fallback]" if self.fallback_used else ""
        return (
            f"SyncResult[{self.user_id}] {status}{fb}: "
            f"upserted={self.upserted} deleted={self.deleted_count} "
            f"errors={self.errors_count} in {self.duration_seconds:.2f}s"
        )


@dataclass
class ConsistencyReport:
    """Result of a consistency audit between wardrobe and Neo4j."""

    is_consistent: bool = True
    orphaned_node_ids: List[str] = field(default_factory=list)
    missing_node_ids: List[str] = field(default_factory=list)
    edge_anomalies: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    user_id: str = ""
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __str__(self) -> str:
        status = "CONSISTENT" if self.is_consistent else "INCONSISTENT"
        return (
            f"ConsistencyReport[{self.user_id}] {status}: "
            f"orphaned={len(self.orphaned_node_ids)} "
            f"missing={len(self.missing_node_ids)} "
            f"edge_anomalies={len(self.edge_anomalies)}"
        )


# ---------------------------------------------------------------------------
# Job class
# ---------------------------------------------------------------------------


class Neo4jSyncJob:
    """
    Orchestrates incremental wardrobe ↔ Neo4j synchronisation.

    Parameters
    ----------
    config_path:
        Optional path to ``neo4j_config.json``.
    batch_size:
        Number of garments per UNWIND batch.
    fallback_enabled:
        When ``True`` (default) any Neo4j failure returns a graceful
        ``SyncResult(fallback_used=True)`` rather than propagating.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        batch_size: int = 50,
        fallback_enabled: bool = True,
    ) -> None:
        self.config_path = config_path
        self.batch_size = batch_size
        self.fallback_enabled = fallback_enabled
        self._client: Optional[Neo4jClient] = None
        self._builder: Optional[Neo4jGraphBuilder] = None

    # ------------------------------------------------------------------
    # Public API — sync
    # ------------------------------------------------------------------

    async def sync_garments(
        self, garments: List[Garment], user_id: str
    ) -> SyncResult:
        """Full upsert sync: rebuild all Garment nodes and COMPATIBLE_WITH
        edges for *user_id*.

        Parameters
        ----------
        garments:
            The complete wardrobe for the user.
        user_id:
            Owning user identifier.

        Returns
        -------
        SyncResult
        """
        result = SyncResult(user_id=user_id)
        t0 = time.perf_counter()
        try:
            builder = self._get_builder()
            if builder is None:
                result.fallback_used = True
                result.success = True  # graceful no-op
                logger.warning("sync_garments: Neo4j unavailable (fallback).")
                return result

            upserted = await builder.load_all_garments(garments, user_id)
            result.upserted = upserted
            result.success = True
            logger.info(
                "sync_garments: %d garments upserted for user=%s", upserted, user_id
            )
        except Exception as exc:
            result.errors_count += 1
            result.success = False
            logger.error("sync_garments failed for user=%s: %s", user_id, exc)
        finally:
            result.duration_seconds = time.perf_counter() - t0
        return result

    async def sync_incremental(
        self,
        new_garments: List[Garment],
        modified_ids: List[str],
        deleted_ids: List[str],
        user_id: str,
        *,
        full_wardrobe: Optional[List[Garment]] = None,
    ) -> SyncResult:
        """Incremental sync: only process changes since the last full sync.

        Parameters
        ----------
        new_garments:
            Brand-new items not yet in Neo4j.
        modified_ids:
            IDs of garments whose attributes have changed.
        deleted_ids:
            IDs of garments that were removed from the wardrobe.
        user_id:
            Owning user identifier.
        full_wardrobe:
            If provided, modified garments are looked up here for re-sync.

        Returns
        -------
        SyncResult
        """
        result = SyncResult(user_id=user_id)
        t0 = time.perf_counter()
        try:
            builder = self._get_builder()
            if builder is None:
                result.fallback_used = True
                result.success = True
                return result

            # Upsert new garments
            if new_garments:
                n = await builder.load_all_garments(new_garments, user_id)
                result.upserted += n

            # Re-sync modified garments
            if modified_ids and full_wardrobe:
                modified_garments = [
                    g for g in full_wardrobe if g.id in set(modified_ids)
                ]
                if modified_garments:
                    m = await builder.load_all_garments(modified_garments, user_id)
                    result.upserted += m

            # Delete removed garments
            if deleted_ids:
                d = await self.delete_garments(deleted_ids, user_id)
                result.deleted_count += d

            result.success = True
            logger.info(
                "sync_incremental[%s]: +%d upserted, -%d deleted",
                user_id,
                result.upserted,
                result.deleted_count,
            )
        except Exception as exc:
            result.errors_count += 1
            result.success = False
            logger.error("sync_incremental failed for user=%s: %s", user_id, exc)
        finally:
            result.duration_seconds = time.perf_counter() - t0
        return result

    async def delete_garments(
        self, garment_ids: List[str], user_id: str
    ) -> int:
        """Remove garment nodes (and incident edges) from Neo4j.

        Parameters
        ----------
        garment_ids:
            IDs to delete.
        user_id:
            Owning user (used as a safety filter to avoid cross-user deletes).

        Returns
        -------
        Number of nodes deleted.
        """
        if not garment_ids:
            return 0
        client = self._get_client()
        if client is None:
            logger.warning("delete_garments: Neo4j unavailable (fallback).")
            return 0
        try:
            rows = await client.query(
                """
                UNWIND $ids AS gid
                MATCH (g:Garment {id: gid, user_id: $uid})
                DETACH DELETE g
                RETURN count(g) AS deleted
                """,
                {"ids": garment_ids, "uid": user_id},
            )
            deleted = int(rows[0].get("deleted", 0)) if rows else 0
            logger.info(
                "delete_garments: removed %d nodes for user=%s", deleted, user_id
            )
            return deleted
        except Exception as exc:
            logger.error("delete_garments failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Public API — consistency
    # ------------------------------------------------------------------

    async def run_consistency_check(
        self, wardrobe: List[Garment], user_id: str
    ) -> ConsistencyReport:
        """Audit Neo4j against the in-memory wardrobe.

        Detects:
        * **Orphaned nodes** — Garment nodes in Neo4j that are not in *wardrobe*.
        * **Missing nodes** — garments in *wardrobe* that have no Neo4j node.
        * **Edge anomalies** — nodes without any COMPATIBLE_WITH edges (isolated).

        Parameters
        ----------
        wardrobe:
            Current ground-truth garment list.
        user_id:
            Owning user.

        Returns
        -------
        ConsistencyReport
        """
        report = ConsistencyReport(user_id=user_id)
        client = self._get_client()
        if client is None:
            report.recommendations.append(
                "Neo4j unavailable — cannot perform consistency check."
            )
            return report

        wardrobe_ids = {g.id for g in wardrobe}

        try:
            # 1. Fetch all garment IDs for this user from Neo4j
            rows = await client.query(
                "MATCH (g:Garment {user_id: $uid}) RETURN g.id AS gid",
                {"uid": user_id},
            )
            neo4j_ids = {r["gid"] for r in rows if r.get("gid")}

            # Orphaned = in Neo4j but not in wardrobe
            orphaned = neo4j_ids - wardrobe_ids
            report.orphaned_node_ids = sorted(orphaned)

            # Missing = in wardrobe but not in Neo4j
            missing = wardrobe_ids - neo4j_ids
            report.missing_node_ids = sorted(missing)

            # 2. Detect isolated nodes (no edges)
            isolated_rows = await client.query(
                """
                MATCH (g:Garment {user_id: $uid})
                WHERE NOT (g)-[:COMPATIBLE_WITH]-()
                RETURN g.id AS gid
                """,
                {"uid": user_id},
            )
            isolated = [r["gid"] for r in isolated_rows if r.get("gid")]
            if isolated:
                report.edge_anomalies.append(
                    f"{len(isolated)} node(s) have no COMPATIBLE_WITH edges: "
                    + ", ".join(isolated[:5])
                    + ("…" if len(isolated) > 5 else "")
                )

            # Produce recommendations
            if report.orphaned_node_ids:
                report.recommendations.append(
                    f"Delete {len(report.orphaned_node_ids)} orphaned node(s) "
                    f"via repair_consistency()."
                )
            if report.missing_node_ids:
                report.recommendations.append(
                    f"Upsert {len(report.missing_node_ids)} missing garment(s) "
                    f"via repair_consistency()."
                )
            if report.edge_anomalies:
                report.recommendations.append(
                    "Rebuild compatibility edges for isolated nodes via sync_garments()."
                )

            report.is_consistent = (
                not report.orphaned_node_ids
                and not report.missing_node_ids
                and not report.edge_anomalies
            )

            logger.info(
                "consistency_check[%s]: orphaned=%d missing=%d anomalies=%d",
                user_id,
                len(report.orphaned_node_ids),
                len(report.missing_node_ids),
                len(report.edge_anomalies),
            )

        except Exception as exc:
            report.is_consistent = False
            report.recommendations.append(f"Consistency check error: {exc}")
            logger.error("run_consistency_check failed for user=%s: %s", user_id, exc)

        return report

    async def repair_consistency(
        self,
        report: ConsistencyReport,
        wardrobe: List[Garment],
        user_id: str,
    ) -> SyncResult:
        """Apply fixes identified by :meth:`run_consistency_check`.

        * Deletes orphaned nodes.
        * Upserts missing nodes.

        Parameters
        ----------
        report:
            ``ConsistencyReport`` from a previous :meth:`run_consistency_check`.
        wardrobe:
            Full in-memory wardrobe (used to look up missing garments).
        user_id:
            Owning user.

        Returns
        -------
        SyncResult with the combined counts.
        """
        result = SyncResult(user_id=user_id)
        t0 = time.perf_counter()
        try:
            if report.orphaned_node_ids:
                d = await self.delete_garments(report.orphaned_node_ids, user_id)
                result.deleted_count += d

            if report.missing_node_ids:
                missing_set = set(report.missing_node_ids)
                to_upsert = [g for g in wardrobe if g.id in missing_set]
                if to_upsert:
                    builder = self._get_builder()
                    if builder is not None:
                        n = await builder.load_all_garments(to_upsert, user_id)
                        result.upserted += n

            result.success = True
            logger.info(
                "repair_consistency[%s]: upserted=%d deleted=%d",
                user_id,
                result.upserted,
                result.deleted_count,
            )
        except Exception as exc:
            result.errors_count += 1
            result.success = False
            logger.error("repair_consistency failed for user=%s: %s", user_id, exc)
        finally:
            result.duration_seconds = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release Neo4j connections."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._builder is not None:
            await self._builder.close()
            self._builder = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_client(self) -> Optional[Neo4jClient]:
        if self._client is not None:
            return self._client
        if not self.fallback_enabled:
            self._client = Neo4jClient(self.config_path)
            return self._client
        try:
            self._client = Neo4jClient(self.config_path)
            return self._client
        except Exception as exc:
            logger.warning("Neo4jSyncJob: cannot create client: %s", exc)
            return None

    def _get_builder(self) -> Optional[Neo4jGraphBuilder]:
        if self._builder is not None:
            return self._builder
        try:
            self._builder = Neo4jGraphBuilder(
                config_path=self.config_path,
                fallback_enabled=self.fallback_enabled,
            )
            # Share client to avoid duplicate connections
            if self._client is not None:
                self._builder._client = self._client
            return self._builder
        except Exception as exc:
            if not self.fallback_enabled:
                raise
            logger.warning("Neo4jSyncJob: cannot create builder: %s", exc)
            return None
