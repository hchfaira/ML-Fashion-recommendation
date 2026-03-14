"""
Centrality Refresh Job (Phase 4)
==================================

Weekly background job that:

1. Fetches all Garment nodes from Neo4j for a given user.
2. Re-computes approximate degree / PageRank / betweenness centrality
   from the live COMPATIBLE_WITH edge graph.
3. Writes the scores back as properties on each Garment node.
4. Optionally broadcasts a completion event via a simple callback.

This decouples expensive graph analytics from real-time request handling:
the ``SmartRemovalAnalyzer`` reads **pre-computed** centrality scores from
Neo4j (property lookups) instead of traversing the graph at query time.

Architecture
------------
::

    Scheduler (cron / APScheduler / Celery beat)
        ↓  calls once per week
    CentralityRefreshJob.run(user_id)
        ↓
    Neo4jClient.query  — fetch garment_ids for user
        ↓
    Neo4jGraphBuilder.update_centrality_scores(garments)
        ↓
    Garment.degree_centrality / .pagerank / .betweenness_centrality
    written back to Neo4j

Fallback
--------
If Neo4j is unreachable the job logs a warning and returns a
``CentralityRefreshResult`` with ``success=False`` — no exception
propagates to the caller.

Usage
-----
::

    from src.jobs.centrality_refresh_job import CentralityRefreshJob

    job = CentralityRefreshJob()
    result = await job.run(user_id="u1", garments=wardrobe)
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.core import get_logger
from src.core.models import Garment
from src.core.neo4j_client import Neo4jClient, Neo4jException
from src.core.neo4j_config import get_neo4j_config
from src.jobs.neo4j_graph_builder import Neo4jGraphBuilder

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CentralityRefreshResult:
    """Result of a single centrality refresh run."""

    success: bool = False
    user_id: str = ""
    garments_updated: int = 0
    duration_seconds: float = 0.0
    fallback_used: bool = False
    errors: List[str] = field(default_factory=list)
    ran_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        status = "OK" if self.success else "FAILED"
        fb = " [fallback]" if self.fallback_used else ""
        return (
            f"CentralityRefresh[{self.user_id}] {status}{fb}: "
            f"{self.garments_updated} nodes updated in {self.duration_seconds:.2f}s"
        )


# ---------------------------------------------------------------------------
# Job class
# ---------------------------------------------------------------------------


class CentralityRefreshJob:
    """
    Weekly job that pre-computes and persists centrality scores.

    Parameters
    ----------
    config_path:
        Optional path to ``neo4j_config.json``.  When ``None`` the default
        config-loader path is used.
    fallback_enabled:
        When ``True`` (default) the job degrades gracefully if Neo4j is
        unreachable.
    on_complete:
        Optional async or sync callable invoked after each successful run
        with the ``CentralityRefreshResult`` as the sole argument.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        fallback_enabled: bool = True,
        on_complete: Optional[Callable[[CentralityRefreshResult], Any]] = None,
    ) -> None:
        self.config_path = config_path
        self.fallback_enabled = fallback_enabled
        self.on_complete = on_complete
        self._client: Optional[Neo4jClient] = None
        self._builder: Optional[Neo4jGraphBuilder] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        user_id: str,
        garments: Optional[List[Garment]] = None,
        *,
        fetch_from_neo4j: bool = False,
    ) -> CentralityRefreshResult:
        """
        Execute one centrality refresh cycle.

        Parameters
        ----------
        user_id:
            Identifier forwarded to Neo4j queries.
        garments:
            Pre-loaded wardrobe.  When ``None`` and ``fetch_from_neo4j``
            is ``True``, garment IDs are fetched directly from Neo4j and
            dummy ``Garment`` objects are built to carry the IDs through
            ``update_centrality_scores``.
        fetch_from_neo4j:
            If ``True`` **and** ``garments`` is ``None``, the job first
            queries Neo4j for all garment IDs belonging to *user_id*.

        Returns
        -------
        CentralityRefreshResult
        """
        result = CentralityRefreshResult(user_id=user_id)
        t0 = time.perf_counter()

        try:
            builder = self._get_builder()
            client = self._get_client()

            if client is None:
                result.fallback_used = True
                result.success = True  # graceful no-op
                logger.warning(
                    "CentralityRefreshJob: Neo4j unavailable, skipping (fallback)."
                )
                return result

            # Resolve garment list
            if garments is None and fetch_from_neo4j:
                garments = await self._fetch_garments_from_neo4j(client, user_id)

            if not garments:
                logger.info(
                    "CentralityRefreshJob: no garments for user=%s, nothing to do.",
                    user_id,
                )
                result.success = True
                return result

            updated = await builder.update_centrality_scores(garments)
            result.garments_updated = updated
            result.success = True
            logger.info(
                "CentralityRefreshJob: updated %d garments for user=%s",
                updated,
                user_id,
            )

        except Exception as exc:
            result.errors.append(str(exc))
            result.success = False
            logger.error(
                "CentralityRefreshJob failed for user=%s: %s", user_id, exc
            )
        finally:
            result.duration_seconds = time.perf_counter() - t0

        # Fire completion callback
        if result.success and self.on_complete is not None:
            try:
                if asyncio.iscoroutinefunction(self.on_complete):
                    await self.on_complete(result)
                else:
                    self.on_complete(result)
            except Exception as cb_exc:
                logger.warning("on_complete callback raised: %s", cb_exc)

        return result

    async def run_for_all_users(
        self,
        user_wardrobe_map: Dict[str, List[Garment]],
    ) -> List[CentralityRefreshResult]:
        """
        Run centrality refresh for multiple users sequentially.

        Errors in one user's run do not abort the others.

        Parameters
        ----------
        user_wardrobe_map:
            Mapping ``{user_id: list_of_garments}``.

        Returns
        -------
        List of ``CentralityRefreshResult``, one per user.
        """
        results: List[CentralityRefreshResult] = []
        for uid, garment_list in user_wardrobe_map.items():
            result = await self.run(uid, garments=garment_list)
            results.append(result)
        return results

    async def close(self) -> None:
        """Release Neo4j resources held by this job."""
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
        """Return (possibly cached) Neo4jClient, or None on failure."""
        if self._client is not None:
            return self._client
        if not self.fallback_enabled:
            self._client = Neo4jClient(self.config_path)
            return self._client
        try:
            self._client = Neo4jClient(self.config_path)
            return self._client
        except Exception as exc:
            logger.warning("CentralityRefreshJob: cannot create client: %s", exc)
            return None

    def _get_builder(self) -> Neo4jGraphBuilder:
        """Return (possibly cached) Neo4jGraphBuilder."""
        if self._builder is None:
            self._builder = Neo4jGraphBuilder(
                config_path=self.config_path,
                fallback_enabled=self.fallback_enabled,
            )
            # Share the same client to avoid duplicate connections
            if self._client is not None:
                self._builder._client = self._client
        return self._builder

    @staticmethod
    async def _fetch_garments_from_neo4j(
        client: Neo4jClient, user_id: str
    ) -> List[Garment]:
        """
        Fetch garment IDs from Neo4j and build minimal Garment stubs.

        The stubs carry only the ``id`` field — enough for
        ``update_centrality_scores`` which only needs the IDs.
        """
        from src.core.models import GarmentAttributes, ColorProfile, GarmentCategory

        try:
            rows = await client.query(
                "MATCH (g:Garment {user_id: $uid}) RETURN g.id AS garment_id",
                {"uid": user_id},
            )
        except Neo4jException as exc:
            logger.warning("Could not fetch garment IDs: %s", exc)
            return []

        garments: List[Garment] = []
        for row in rows:
            gid = row.get("garment_id")
            if gid:
                garments.append(
                    Garment(
                        id=gid,
                        attributes=GarmentAttributes(
                            category=GarmentCategory.TOP,
                            color=ColorProfile(primary="unknown"),
                        ),
                    )
                )
        return garments
