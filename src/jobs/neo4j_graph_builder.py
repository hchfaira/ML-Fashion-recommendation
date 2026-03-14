"""
Neo4j Graph Builder (Phase 2)
==============================

Responsible for **initialising and populating** the Neo4j graph from an
existing wardrobe (list of :class:`~src.core.models.Garment` objects).

Pipeline
--------
1. :meth:`initialize_schema`        — create indexes / constraints
2. :meth:`create_reference_nodes`   — Season / Occasion / Category /
                                      FormalityLevel / Color nodes
3. :meth:`load_garment`             — upsert a single Garment node
4. :meth:`load_all_garments`        — batch upsert the whole wardrobe
5. :meth:`link_garment_to_refs`     — SUITABLE_FOR / BELONGS_TO /
                                      HAS_COLOR edges for one garment
6. :meth:`build_compatibility_edges`— COMPATIBLE_WITH edges for all pairs
7. :meth:`build_color_harmony_edges`— Color-level COMPLEMENTS/ANALOGOUS/
                                      TRIADIC edges
8. :meth:`update_centrality_scores` — approximate PageRank / degree /
                                      betweenness stored on each Garment node
9. :meth:`validate_graph`           — sanity-check node/edge counts
10. :meth:`run_full_initialization` — orchestrate all steps above

Fallback
--------
Every public method is fully operational when Neo4j is **not** reachable
(``fallback_enabled=True``, the default).  In fallback mode the builder
logs a warning and returns gracefully — the rest of the system continues
using its in-memory scorers.

Usage
-----
::

    from src.jobs.neo4j_graph_builder import Neo4jGraphBuilder

    builder = Neo4jGraphBuilder()
    result = await builder.run_full_initialization(garments, user_id="u1")
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.models import (
    Garment,
    GarmentCategory,
    FormalityLevel,
    Season,
    Occasion,
)
from src.core.neo4j_client import Neo4jClient, Neo4jException
from src.core.neo4j_config import get_neo4j_config
from src.core.neo4j_models import (
    GarmentNode,
    GraphSchema,
    CompatibleWithRelationship,
    ColorHarmonizesRelationship,
    SuitableForRelationship,
    create_garment_node_from_dict,
    create_compatibility_from_scores,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GraphBuildResult:
    """Summary returned by :meth:`Neo4jGraphBuilder.run_full_initialization`."""

    success: bool = False
    fallback_used: bool = False

    # Counts
    garments_loaded: int = 0
    reference_nodes_created: int = 0
    garment_ref_edges_created: int = 0
    compatibility_edges_created: int = 0
    color_harmony_edges_created: int = 0

    # Timing (seconds)
    duration_seconds: float = 0.0

    # Validation
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)

    # Error info (populated on non-fatal failures)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:  # noqa: D401
        """Human-readable one-liner."""
        if self.fallback_used:
            return (
                "GraphBuilder ran in FALLBACK mode (Neo4j unavailable). "
                "In-memory scorers are unaffected."
            )
        status = "✓" if self.success else "✗"
        return (
            f"{status} Graph build complete in {self.duration_seconds:.1f}s | "
            f"garments={self.garments_loaded} "
            f"compat_edges={self.compatibility_edges_created} "
            f"ref_edges={self.garment_ref_edges_created} "
            f"color_edges={self.color_harmony_edges_created}"
        )


# ---------------------------------------------------------------------------
# Reference data — loaded from config/data/style_rules_config.json,
# with fallback to built-in defaults when the file is unavailable.
# ---------------------------------------------------------------------------

_STYLE_RULES_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "config" / "data" / "style_rules_config.json"
)


def _load_graph_config() -> dict:
    """Load the neo4j_graph section of style_rules_config.json."""
    try:
        with open(_STYLE_RULES_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("neo4j_graph", {})
    except Exception as exc:
        logger.warning(
            "Could not load style_rules_config.json: %s — using built-in defaults.", exc
        )
        return {}


def _build_defaults() -> dict:
    """Build the reference-data dicts from config (or built-in fallback)."""
    cfg = _load_graph_config()

    seasons = cfg.get("seasons", ["SPRING", "SUMMER", "FALL", "WINTER"])

    occasions = cfg.get("occasions", [
        "CASUAL", "BUSINESS", "FORMAL", "WEDDING", "ATHLETIC",
        "EVENING", "BEACH", "DATE", "TRAVEL",
    ])

    formality_levels = cfg.get("formality_levels", [
        {"name": "Very Casual",     "value": 0},
        {"name": "Casual",          "value": 1},
        {"name": "Smart Casual",    "value": 2},
        {"name": "Business Casual", "value": 3},
        {"name": "Business",        "value": 4},
        {"name": "Formal",          "value": 5},
        {"name": "Black Tie",       "value": 6},
    ])

    categories = cfg.get("categories", [
        {"name": "Top",       "value": "top",       "priority": 1},
        {"name": "Bottom",    "value": "bottom",    "priority": 2},
        {"name": "Dress",     "value": "dress",     "priority": 3},
        {"name": "Outerwear", "value": "outerwear", "priority": 4},
        {"name": "Shoes",     "value": "shoes",     "priority": 5},
        {"name": "Accessory", "value": "accessory", "priority": 6},
        {"name": "Bag",       "value": "bag",       "priority": 7},
    ])

    colors = cfg.get("colors", [
        {"name": "black",    "hex": "#000000", "role": "neutral"},
        {"name": "white",    "hex": "#FFFFFF", "role": "neutral"},
        {"name": "navy",     "hex": "#001F5B", "role": "cool"},
        {"name": "grey",     "hex": "#808080", "role": "neutral"},
        {"name": "beige",    "hex": "#F5F5DC", "role": "neutral"},
        {"name": "brown",    "hex": "#8B4513", "role": "warm"},
        {"name": "red",      "hex": "#FF0000", "role": "warm"},
        {"name": "blue",     "hex": "#0000FF", "role": "cool"},
        {"name": "green",    "hex": "#008000", "role": "cool"},
        {"name": "yellow",   "hex": "#FFFF00", "role": "warm"},
        {"name": "orange",   "hex": "#FFA500", "role": "warm"},
        {"name": "pink",     "hex": "#FFC0CB", "role": "warm"},
        {"name": "purple",   "hex": "#800080", "role": "cool"},
        {"name": "camel",    "hex": "#C19A6B", "role": "warm"},
        {"name": "cream",    "hex": "#FFFDD0", "role": "neutral"},
        {"name": "tan",      "hex": "#D2B48C", "role": "warm"},
        {"name": "burgundy", "hex": "#800020", "role": "warm"},
        {"name": "olive",    "hex": "#808000", "role": "neutral"},
        {"name": "teal",     "hex": "#008080", "role": "cool"},
        {"name": "coral",    "hex": "#FF7F50", "role": "warm"},
    ])

    # Convert list-of-dicts format → list-of-tuples (from, to, type, weight)
    raw_harmonies = cfg.get("color_harmonies", [
        {"from": "navy",     "to": "white",    "type": "COMPLEMENTS", "weight": 0.95},
        {"from": "navy",     "to": "beige",    "type": "ANALOGOUS",   "weight": 0.85},
        {"from": "navy",     "to": "camel",    "type": "COMPLEMENTS", "weight": 0.90},
        {"from": "navy",     "to": "grey",     "type": "ANALOGOUS",   "weight": 0.80},
        {"from": "black",    "to": "white",    "type": "COMPLEMENTS", "weight": 0.95},
        {"from": "black",    "to": "red",      "type": "COMPLEMENTS", "weight": 0.85},
        {"from": "black",    "to": "camel",    "type": "COMPLEMENTS", "weight": 0.88},
        {"from": "black",    "to": "beige",    "type": "COMPLEMENTS", "weight": 0.82},
        {"from": "white",    "to": "navy",     "type": "COMPLEMENTS", "weight": 0.95},
        {"from": "white",    "to": "black",    "type": "COMPLEMENTS", "weight": 0.95},
        {"from": "white",    "to": "beige",    "type": "ANALOGOUS",   "weight": 0.75},
        {"from": "grey",     "to": "navy",     "type": "ANALOGOUS",   "weight": 0.80},
        {"from": "grey",     "to": "black",    "type": "ANALOGOUS",   "weight": 0.85},
        {"from": "grey",     "to": "white",    "type": "ANALOGOUS",   "weight": 0.80},
        {"from": "beige",    "to": "brown",    "type": "ANALOGOUS",   "weight": 0.85},
        {"from": "beige",    "to": "camel",    "type": "ANALOGOUS",   "weight": 0.88},
        {"from": "beige",    "to": "navy",     "type": "COMPLEMENTS", "weight": 0.85},
        {"from": "beige",    "to": "white",    "type": "ANALOGOUS",   "weight": 0.80},
        {"from": "camel",    "to": "black",    "type": "COMPLEMENTS", "weight": 0.88},
        {"from": "camel",    "to": "white",    "type": "COMPLEMENTS", "weight": 0.85},
        {"from": "camel",    "to": "navy",     "type": "COMPLEMENTS", "weight": 0.90},
        {"from": "burgundy", "to": "camel",    "type": "COMPLEMENTS", "weight": 0.85},
        {"from": "burgundy", "to": "grey",     "type": "COMPLEMENTS", "weight": 0.80},
        {"from": "burgundy", "to": "navy",     "type": "ANALOGOUS",   "weight": 0.78},
        {"from": "red",      "to": "black",    "type": "COMPLEMENTS", "weight": 0.85},
        {"from": "red",      "to": "white",    "type": "COMPLEMENTS", "weight": 0.82},
        {"from": "blue",     "to": "orange",   "type": "COMPLEMENTS", "weight": 0.88},
        {"from": "green",    "to": "red",      "type": "COMPLEMENTS", "weight": 0.75},
        {"from": "purple",   "to": "yellow",   "type": "COMPLEMENTS", "weight": 0.78},
        {"from": "teal",     "to": "coral",    "type": "COMPLEMENTS", "weight": 0.82},
        {"from": "olive",    "to": "burgundy", "type": "COMPLEMENTS", "weight": 0.80},
    ])
    color_harmonies: List[Tuple[str, str, str, float]] = [
        (h["from"], h["to"], h["type"], h["weight"]) for h in raw_harmonies
    ]

    return {
        "seasons": seasons,
        "occasions": occasions,
        "formality_levels": formality_levels,
        "categories": categories,
        "colors": colors,
        "color_harmonies": color_harmonies,
    }


_GRAPH_DEFAULTS = _build_defaults()

_DEFAULT_SEASONS        = _GRAPH_DEFAULTS["seasons"]
_DEFAULT_OCCASIONS      = _GRAPH_DEFAULTS["occasions"]
_DEFAULT_FORMALITY_LEVELS = _GRAPH_DEFAULTS["formality_levels"]
_DEFAULT_CATEGORIES     = _GRAPH_DEFAULTS["categories"]
_DEFAULT_COLORS         = _GRAPH_DEFAULTS["colors"]
_COLOR_HARMONIES: List[Tuple[str, str, str, float]] = _GRAPH_DEFAULTS["color_harmonies"]


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

# Formality level mapping from FormalityLevel enum values to int
_FORMALITY_TO_INT: Dict[str, int] = {
    FormalityLevel.VERY_CASUAL.value: 0,
    FormalityLevel.CASUAL.value:      1,
    FormalityLevel.SMART_CASUAL.value: 2,
    FormalityLevel.BUSINESS_CASUAL.value: 3,
    FormalityLevel.BUSINESS.value:    4,
    FormalityLevel.FORMAL.value:      5,
    FormalityLevel.BLACK_TIE.value:   6,
}


def _formality_int(garment: Garment) -> int:
    """Extract integer formality level from a Garment."""
    fl = garment.attributes.formality_level
    if isinstance(fl, int):
        return fl
    val = fl.value if hasattr(fl, "value") else str(fl)
    return _FORMALITY_TO_INT.get(val, 2)


def _compute_compatibility_score(g1: Garment, g2: Garment) -> CompatibleWithRelationship:
    """
    Compute a :class:`CompatibleWithRelationship` for two garments.

    Scoring dimensions (each 0-1, then weighted average):

    * **formality_match** (weight 0.30) — penalises large formality gaps
    * **style_overlap**   (weight 0.25) — Jaccard similarity of style_tags
    * **color_harmony**   (weight 0.25) — basic warm/cool/neutral affinity
    * **category_base**   (weight 0.20) — reward complementary categories
    """
    attrs1 = g1.attributes
    attrs2 = g2.attributes

    # --- Formality match ---
    f1 = _formality_int(g1)
    f2 = _formality_int(g2)
    formality_delta = abs(f1 - f2)
    # Score 1.0 at delta=0, drops to 0.5 at delta=2, ~0 at delta=6
    formality_score = max(0.0, 1.0 - formality_delta * 0.15)

    # --- Style overlap (Jaccard) ---
    tags1 = set(t.lower() for t in attrs1.style_tags)
    tags2 = set(t.lower() for t in attrs2.style_tags)
    if tags1 or tags2:
        intersection = len(tags1 & tags2)
        union = len(tags1 | tags2)
        style_score = intersection / union if union > 0 else 0.5
        # Give a base of 0.4 so garments with no shared tags aren't penalised too hard
        style_score = 0.4 + style_score * 0.6
    else:
        style_score = 0.6  # no info → neutral

    # --- Colour harmony (warm/cool/neutral affinity) ---
    def _temp(a):
        ct = getattr(a.color, "color_temperature", None)
        return ct.value if ct else "neutral"

    t1, t2 = _temp(attrs1), _temp(attrs2)
    if t1 == t2 and t1 != "neutral":
        color_score = 0.75  # same temperature — analogous harmony
    elif "neutral" in (t1, t2):
        color_score = 0.85  # neutral pairs with everything
    else:
        color_score = 0.65  # warm+cool — contrasting, acceptable

    # --- Category base score ---
    cat_pairs_premium = {
        frozenset([GarmentCategory.TOP, GarmentCategory.BOTTOM]),
        frozenset([GarmentCategory.TOP, GarmentCategory.SHOES]),
        frozenset([GarmentCategory.BOTTOM, GarmentCategory.SHOES]),
        frozenset([GarmentCategory.DRESS, GarmentCategory.SHOES]),
        frozenset([GarmentCategory.DRESS, GarmentCategory.OUTERWEAR]),
        frozenset([GarmentCategory.TOP, GarmentCategory.OUTERWEAR]),
    }
    pair = frozenset([attrs1.category, attrs2.category])
    category_score = 0.9 if pair in cat_pairs_premium else 0.65

    # --- Weighted average ---
    weight = (
        0.30 * formality_score
        + 0.25 * style_score
        + 0.25 * color_score
        + 0.20 * category_score
    )
    weight = max(0.0, min(1.0, weight))

    # --- Build seasons / occasions lists ---
    seasons: List[str] = [
        s.value.upper() if hasattr(s, "value") else str(s).upper()
        for s in (attrs1.season_suitable or [])
        if s in (attrs2.season_suitable or [])
    ]
    occasions: List[str] = []
    op1 = attrs1.occasion_profile
    op2 = attrs2.occasion_profile
    if op1 and op2:
        occs1 = set(o.lower() for o in (op1.occasions or []))
        occs2 = set(o.lower() for o in (op2.occasions or []))
        occasions = list(occs1 & occs2)

    return CompatibleWithRelationship(
        weight=round(weight, 4),
        occasions=occasions,
        seasons=seasons,
        formality_delta=formality_delta,
        color_harmony=round(color_score, 4),
        pattern_compatible=True,
        volume_balance_score=0.5,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Neo4jGraphBuilder:
    """
    Builds the Neo4j garment compatibility graph.

    Parameters
    ----------
    config_path:
        Optional path to a custom ``neo4j_config.json``.  If *None* the
        default path ``config/data/neo4j_config.json`` is used.
    fallback_enabled:
        When *True* (default), every method silently succeeds when Neo4j
        is unreachable so the rest of the system is not blocked.
    min_compatibility_weight:
        Pairs with a computed weight below this threshold are **not**
        stored in the graph.  Defaults to the value in the config
        (typically 0.5).
    batch_size:
        Number of garments per Cypher UNWIND batch for load operations.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        fallback_enabled: bool = True,
        min_compatibility_weight: Optional[float] = None,
        batch_size: int = 50,
    ) -> None:
        self.fallback_enabled = fallback_enabled
        self._client: Optional[Neo4jClient] = None
        self._config_path = config_path

        # Resolve minimum weight, batch size, and PageRank params from config or parameter
        try:
            cfg = get_neo4j_config()
            self._min_weight = (
                min_compatibility_weight
                if min_compatibility_weight is not None
                else cfg.get_min_compatibility_weight()
            )
            self.batch_size = batch_size if batch_size != 50 else cfg.get_batch_query_size()
            self._pagerank_damping = cfg.get_pagerank_damping()
            self._pagerank_iterations = cfg.get_pagerank_power_iterations()
            self._reference_edge_weight = cfg.get_default_reference_edge_weight()
        except Exception:
            self._min_weight = min_compatibility_weight or 0.5
            self.batch_size = batch_size
            self._pagerank_damping = 0.85
            self._pagerank_iterations = 5
            self._reference_edge_weight = 0.8

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> Optional[Neo4jClient]:
        """Return a (possibly cached) Neo4jClient, or None in fallback."""
        if self._client is not None:
            return self._client
        try:
            self._client = Neo4jClient(config_path=self._config_path)
            return self._client
        except Neo4jException as exc:
            if self.fallback_enabled:
                logger.warning(
                    "Neo4j unavailable, running in fallback mode: %s", exc
                )
                return None
            raise

    async def close(self) -> None:
        """Close the underlying Neo4j connection if it was opened."""
        if self._client:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Step 1 — Schema initialisation
    # ------------------------------------------------------------------

    async def initialize_schema(self) -> bool:
        """
        Create Neo4j indexes and constraints defined in
        :class:`~src.core.neo4j_models.GraphSchema`.

        Returns
        -------
        bool
            *True* on success, *False* if Neo4j is unreachable and
            fallback is enabled.
        """
        client = self._get_client()
        if client is None:
            return False

        logger.info("Initializing Neo4j schema (indexes + constraints)…")
        errors: List[str] = []

        for name, cypher in GraphSchema.INDEXES:
            try:
                await client.query(cypher, ttl=0)
                logger.debug("Index created/verified: %s", name)
            except Neo4jException as exc:
                errors.append(f"Index '{name}': {exc}")
                logger.warning("Could not create index '%s': %s", name, exc)

        # Uniqueness constraints
        constraints = [
            (
                "garment_id_unique",
                "CREATE CONSTRAINT garment_id_unique IF NOT EXISTS "
                "FOR (g:Garment) REQUIRE g.id IS UNIQUE",
            ),
            (
                "user_id_unique",
                "CREATE CONSTRAINT user_id_unique IF NOT EXISTS "
                "FOR (u:User) REQUIRE u.id IS UNIQUE",
            ),
            (
                "season_name_unique",
                "CREATE CONSTRAINT season_name_unique IF NOT EXISTS "
                "FOR (s:Season) REQUIRE s.name IS UNIQUE",
            ),
            (
                "occasion_name_unique",
                "CREATE CONSTRAINT occasion_name_unique IF NOT EXISTS "
                "FOR (o:Occasion) REQUIRE o.name IS UNIQUE",
            ),
            (
                "color_name_unique",
                "CREATE CONSTRAINT color_name_unique IF NOT EXISTS "
                "FOR (c:Color) REQUIRE c.name IS UNIQUE",
            ),
        ]

        for name, cypher in constraints:
            try:
                await client.query(cypher, ttl=0)
                logger.debug("Constraint created/verified: %s", name)
            except Neo4jException as exc:
                errors.append(f"Constraint '{name}': {exc}")
                logger.warning("Could not create constraint '%s': %s", name, exc)

        if errors:
            logger.warning("%d schema errors (non-fatal): %s", len(errors), errors)
        else:
            logger.info("Schema initialisation complete (%d indexes, %d constraints)",
                        len(GraphSchema.INDEXES), len(constraints))

        return True

    # ------------------------------------------------------------------
    # Step 2 — Reference nodes
    # ------------------------------------------------------------------

    async def create_reference_nodes(self) -> int:
        """
        Upsert Season, Occasion, Category, FormalityLevel, and Color
        reference nodes.

        Returns
        -------
        int
            Total number of reference nodes created/merged, or 0 on
            fallback.
        """
        client = self._get_client()
        if client is None:
            return 0

        logger.info("Creating reference nodes…")
        total = 0

        # Seasons
        for season in _DEFAULT_SEASONS:
            await client.query(
                "MERGE (s:Season {name: $name}) ON CREATE SET s.value = $value",
                {"name": season, "value": season},
                ttl=0,
            )
            total += 1

        # Occasions
        for occ in _DEFAULT_OCCASIONS:
            await client.query(
                "MERGE (o:Occasion {name: $name}) ON CREATE SET o.value = $value, o.importance = 0.5",
                {"name": occ, "value": occ},
                ttl=0,
            )
            total += 1

        # Categories
        for cat in _DEFAULT_CATEGORIES:
            await client.query(
                "MERGE (c:Category {value: $value}) "
                "ON CREATE SET c.name = $name, c.priority = $priority",
                cat,
                ttl=0,
            )
            total += 1

        # Formality levels
        for fl in _DEFAULT_FORMALITY_LEVELS:
            await client.query(
                "MERGE (f:FormalityLevel {value: $value}) "
                "ON CREATE SET f.name = $name, f.order = $value",
                fl,
                ttl=0,
            )
            total += 1

        # Colors
        for col in _DEFAULT_COLORS:
            await client.query(
                "MERGE (c:Color {name: $name}) "
                "ON CREATE SET c.hex = $hex, c.role = $role",
                col,
                ttl=0,
            )
            total += 1

        logger.info("Reference nodes created/verified: %d", total)
        return total

    # ------------------------------------------------------------------
    # Step 3 — Load individual garment
    # ------------------------------------------------------------------

    async def load_garment(self, garment: Garment, user_id: str = "") -> bool:
        """
        Upsert a single Garment node into Neo4j.

        Parameters
        ----------
        garment:
            The :class:`~src.core.models.Garment` to load.
        user_id:
            The owner's user ID.  Written onto the node so season/occasion
            queries can be scoped per user.

        Returns
        -------
        bool
            *True* on success, *False* on fallback.
        """
        client = self._get_client()
        if client is None:
            return False

        attrs = garment.attributes
        node = GarmentNode(
            id=garment.id,
            title=attrs.subcategory or attrs.category.value,
            category=attrs.category.value,
            subcategory=attrs.subcategory,
            color_primary=attrs.color.primary if attrs.color else None,
            color_secondary=attrs.color.secondary if attrs.color else None,
            pattern_type=attrs.pattern.type if attrs.pattern else None,
            material=(
                attrs.material.primary
                if attrs.material and attrs.material.primary
                else None
            ),
            formality_level=_formality_int(garment),
            rating=float(garment.history.average_rating or 0.5),
            last_worn_at=(
                garment.history.last_worn_at.isoformat()
                if garment.history.last_worn_at
                else None
            ),
            total_wears=garment.history.total_wears,
            is_retired=garment.history.is_retired,
            user_id=user_id,
        )

        props = node.to_neo4j_properties()

        await client.query(
            """
            MERGE (g:Garment {id: $id})
            ON CREATE SET g += $props, g.created_at = timestamp()
            ON MATCH  SET g += $props
            """,
            {"id": garment.id, "props": props},
            ttl=0,
        )
        logger.debug("Garment upserted: %s (%s)", garment.id, node.title)
        return True

    # ------------------------------------------------------------------
    # Step 4 — Batch load all garments
    # ------------------------------------------------------------------

    async def load_all_garments(
        self, garments: List[Garment], user_id: str = ""
    ) -> int:
        """
        Batch-upsert a list of garments using UNWIND for efficiency.

        Parameters
        ----------
        garments:
            Wardrobe items to load.
        user_id:
            Owner user ID written onto every node.

        Returns
        -------
        int
            Number of garments successfully upserted (0 on fallback).
        """
        client = self._get_client()
        if client is None:
            return 0

        if not garments:
            return 0

        logger.info("Loading %d garments into Neo4j…", len(garments))
        loaded = 0

        # Process in batches
        for i in range(0, len(garments), self.batch_size):
            batch = garments[i: i + self.batch_size]
            rows = []
            for g in batch:
                attrs = g.attributes
                rows.append(
                    {
                        "id": g.id,
                        "title": attrs.subcategory or attrs.category.value,
                        "category": attrs.category.value,
                        "subcategory": attrs.subcategory or "",
                        "color_primary": attrs.color.primary if attrs.color else "",
                        "color_secondary": (
                            attrs.color.secondary if attrs.color else None
                        ),
                        "pattern_type": (
                            attrs.pattern.type if attrs.pattern else "solid"
                        ),
                        "formality_level": _formality_int(g),
                        "rating": float(g.history.average_rating or 0.5),
                        "total_wears": g.history.total_wears,
                        "is_retired": g.history.is_retired,
                        "user_id": user_id,
                    }
                )

            try:
                await client.query(
                    """
                    UNWIND $rows AS row
                    MERGE (g:Garment {id: row.id})
                    ON CREATE SET g += row, g.created_at = timestamp()
                    ON MATCH  SET g += row
                    """,
                    {"rows": rows},
                    ttl=0,
                )
                loaded += len(batch)
                logger.debug(
                    "Batch %d-%d loaded (%d garments)",
                    i + 1, min(i + self.batch_size, len(garments)), len(batch),
                )
            except Neo4jException as exc:
                logger.error("Batch load failed for rows %d-%d: %s", i, i + self.batch_size, exc)

        logger.info("Garments loaded: %d / %d", loaded, len(garments))
        return loaded

    # ------------------------------------------------------------------
    # Step 5 — Link garment → reference nodes
    # ------------------------------------------------------------------

    async def link_garment_to_refs(
        self, garment: Garment, user_id: str = ""
    ) -> int:
        """
        Create SUITABLE_FOR (season/occasion) and BELONGS_TO (category)
        edges for a single garment.

        Returns
        -------
        int
            Number of edges created (0 on fallback).
        """
        client = self._get_client()
        if client is None:
            return 0

        gid = garment.id
        attrs = garment.attributes
        edges = 0

        # --- BELONGS_TO Category ---
        try:
            await client.query(
                """
                MATCH (g:Garment {id: $gid})
                MATCH (c:Category {value: $cat})
                MERGE (g)-[r:BELONGS_TO]->(c)
                ON CREATE SET r.created_at = timestamp()
                """,
                {"gid": gid, "cat": attrs.category.value},
                ttl=0,
            )
            edges += 1
        except Neo4jException as exc:
            logger.warning("BELONGS_TO edge failed for %s: %s", gid, exc)

        # --- SUITABLE_FOR Season ---
        for season in (attrs.season_suitable or []):
            season_val = season.value.upper() if hasattr(season, "value") else str(season).upper()
            try:
                await client.query(
                    """
                    MATCH (g:Garment {id: $gid})
                    MATCH (s:Season {name: $season})
                    MERGE (g)-[r:SUITABLE_FOR]->(s)
                    ON CREATE SET r.weight = $weight, r.is_primary = false
                    """,
                    {"gid": gid, "season": season_val, "weight": self._reference_edge_weight},
                    ttl=0,
                )
                edges += 1
            except Neo4jException as exc:
                logger.warning("SUITABLE_FOR season edge failed for %s/%s: %s", gid, season_val, exc)

        # --- SUITABLE_FOR_OCCASION ---
        op = attrs.occasion_profile
        if op:
            for occ in (op.occasions or []):
                occ_val = occ.upper() if isinstance(occ, str) else occ.value.upper()
                try:
                    await client.query(
                        """
                        MATCH (g:Garment {id: $gid})
                        MATCH (o:Occasion {name: $occ})
                        MERGE (g)-[r:SUITABLE_FOR_OCCASION]->(o)
                        ON CREATE SET r.weight = $weight, r.is_primary = false
                        """,
                        {"gid": gid, "occ": occ_val, "weight": self._reference_edge_weight},
                        ttl=0,
                    )
                    edges += 1
                except Neo4jException as exc:
                    logger.warning("SUITABLE_FOR_OCCASION edge failed for %s/%s: %s", gid, occ_val, exc)

        # --- HAS_COLOR ---
        if attrs.color and attrs.color.primary:
            color_name = attrs.color.primary.lower().strip()
            try:
                await client.query(
                    """
                    MATCH (g:Garment {id: $gid})
                    MERGE (c:Color {name: $color})
                    MERGE (g)-[r:HAS_COLOR]->(c)
                    ON CREATE SET r.is_primary = true
                    """,
                    {"gid": gid, "color": color_name},
                    ttl=0,
                )
                edges += 1
            except Neo4jException as exc:
                logger.warning("HAS_COLOR edge failed for %s: %s", gid, exc)

        return edges

    async def link_all_garments_to_refs(
        self, garments: List[Garment], user_id: str = ""
    ) -> int:
        """
        Call :meth:`link_garment_to_refs` for every garment.

        Returns
        -------
        int
            Total edges created (0 on fallback).
        """
        total_edges = 0
        for garment in garments:
            total_edges += await self.link_garment_to_refs(garment, user_id)
        logger.info("Garment→reference edges created: %d", total_edges)
        return total_edges

    # ------------------------------------------------------------------
    # Step 6 — Compatibility edges
    # ------------------------------------------------------------------

    async def build_compatibility_edges(
        self,
        garments: List[Garment],
        min_weight: Optional[float] = None,
    ) -> int:
        """
        Compute pairwise compatibility and write COMPATIBLE_WITH edges.

        Only pairs with ``weight >= min_weight`` are stored to keep the
        graph sparse and meaningful.

        Parameters
        ----------
        garments:
            Full wardrobe list.
        min_weight:
            Override the instance-level threshold.

        Returns
        -------
        int
            Number of edges created (0 on fallback).
        """
        client = self._get_client()
        if client is None:
            return 0

        threshold = min_weight if min_weight is not None else self._min_weight
        if len(garments) < 2:
            return 0

        logger.info(
            "Building compatibility edges for %d garments (min_weight=%.2f)…",
            len(garments), threshold,
        )
        created = 0

        # Collect all edges first, then batch-write
        edges: List[Dict[str, Any]] = []
        for i, g1 in enumerate(garments):
            for g2 in garments[i + 1:]:
                rel = _compute_compatibility_score(g1, g2)
                if rel.weight < threshold:
                    continue
                props = rel.to_neo4j_properties()
                edges.append(
                    {
                        "from_id": g1.id,
                        "to_id":   g2.id,
                        "props":   props,
                    }
                )

        if not edges:
            logger.info("No compatibility edges met the threshold (%.2f)", threshold)
            return 0

        # Write in batches
        for i in range(0, len(edges), self.batch_size):
            batch = edges[i: i + self.batch_size]
            try:
                await client.query(
                    """
                    UNWIND $batch AS e
                    MATCH (a:Garment {id: e.from_id})
                    MATCH (b:Garment {id: e.to_id})
                    MERGE (a)-[r:COMPATIBLE_WITH]->(b)
                    ON CREATE SET r += e.props, r.created_at = timestamp()
                    ON MATCH  SET r += e.props
                    """,
                    {"batch": batch},
                    ttl=0,
                )
                created += len(batch)
            except Neo4jException as exc:
                logger.error(
                    "Compatibility edge batch %d-%d failed: %s",
                    i, i + self.batch_size, exc,
                )

        logger.info("Compatibility edges created: %d (of %d candidate pairs)", created, len(edges))
        return created

    # ------------------------------------------------------------------
    # Step 7 — Color harmony edges
    # ------------------------------------------------------------------

    async def build_color_harmony_edges(self) -> int:
        """
        Create COMPLEMENTS / ANALOGOUS / TRIADIC edges between Color nodes.

        These edges are derived from the built-in colour harmony table
        :data:`_COLOR_HARMONIES` and are upserted once per graph
        initialisation.

        Returns
        -------
        int
            Number of edges created (0 on fallback).
        """
        client = self._get_client()
        if client is None:
            return 0

        logger.info("Building colour harmony edges…")
        created = 0

        for from_color, to_color, harmony_type, weight in _COLOR_HARMONIES:
            try:
                await client.query(
                    f"""
                    MATCH (c1:Color {{name: $from_color}})
                    MATCH (c2:Color {{name: $to_color}})
                    MERGE (c1)-[r:{harmony_type}]->(c2)
                    ON CREATE SET r.weight = $weight, r.harmony_type = $harmony_type,
                                  r.created_at = timestamp()
                    ON MATCH  SET r.weight = $weight
                    """,
                    {
                        "from_color": from_color,
                        "to_color": to_color,
                        "weight": weight,
                        "harmony_type": harmony_type,
                    },
                    ttl=0,
                )
                created += 1
            except Neo4jException as exc:
                logger.warning(
                    "Color edge %s-%s failed: %s", from_color, to_color, exc
                )

        logger.info("Colour harmony edges created/verified: %d", created)
        return created

    # ------------------------------------------------------------------
    # Step 8 — Approximate centrality scores
    # ------------------------------------------------------------------

    async def update_centrality_scores(
        self, garments: List[Garment]
    ) -> int:
        """
        Compute approximate centrality scores from the COMPATIBLE_WITH
        graph and write them back onto each Garment node.

        This is a **lightweight approximation** that avoids the Neo4j GDS
        plugin requirement:

        * **degree_centrality** — normalised in-degree of COMPATIBLE_WITH edges
        * **pagerank** — simplified power-iteration (5 iterations)
        * **betweenness_centrality** — Brandes-style shortest-path estimate

        Full GDS-based centrality can replace this once Neo4j GDS is
        available.

        Returns
        -------
        int
            Number of garment nodes updated (0 on fallback).
        """
        client = self._get_client()
        if client is None:
            return 0

        if not garments:
            return 0

        logger.info("Updating centrality scores for %d garments…", len(garments))

        # --- Fetch edge list from Neo4j ---
        try:
            edge_records = await client.query(
                """
                MATCH (a:Garment)-[r:COMPATIBLE_WITH]->(b:Garment)
                RETURN a.id AS from_id, b.id AS to_id, r.weight AS weight
                """,
                ttl=0,
            )
        except Neo4jException as exc:
            logger.warning("Could not fetch edges for centrality: %s", exc)
            return 0

        # Build adjacency structure
        garment_ids = {g.id for g in garments}
        adjacency: Dict[str, List[Tuple[str, float]]] = {gid: [] for gid in garment_ids}

        for rec in edge_records:
            fid = rec.get("from_id", "")
            tid = rec.get("to_id", "")
            w   = float(rec.get("weight", 0.0))
            if fid in adjacency:
                adjacency[fid].append((tid, w))
            if tid in adjacency:  # undirected for centrality purposes
                adjacency[tid].append((fid, w))

        n = len(garment_ids)
        if n == 0:
            return 0

        # --- Degree centrality ---
        degree: Dict[str, float] = {
            gid: len(neighbours) / max(n - 1, 1)
            for gid, neighbours in adjacency.items()
        }

        # --- Simple PageRank (d=pagerank_damping, n=pagerank_iterations from config) ---
        damping = self._pagerank_damping
        pr: Dict[str, float] = {gid: 1.0 / n for gid in garment_ids}

        for _ in range(self._pagerank_iterations):
            new_pr: Dict[str, float] = {}
            for gid in garment_ids:
                incoming_sum = 0.0
                for other_id, neighbours in adjacency.items():
                    for nb_id, _ in neighbours:
                        if nb_id == gid and len(adjacency[other_id]) > 0:
                            incoming_sum += pr[other_id] / len(adjacency[other_id])
                new_pr[gid] = (1 - damping) / n + damping * incoming_sum
            pr = new_pr

        # Normalise PageRank to [0, 1]
        max_pr = max(pr.values()) if pr else 1.0
        pr = {gid: v / max_pr for gid, v in pr.items()}

        # --- Betweenness estimate (neighbour overlap proxy) ---
        betweenness: Dict[str, float] = {}
        for gid in garment_ids:
            nb_set = {nb for nb, _ in adjacency[gid]}
            if not nb_set:
                betweenness[gid] = 0.0
                continue
            # Fraction of neighbour-pairs that are NOT directly connected
            bridge_count = 0
            total_pairs = 0
            nb_list = list(nb_set)
            for i, nb1 in enumerate(nb_list):
                for nb2 in nb_list[i + 1:]:
                    total_pairs += 1
                    nb1_nbs = {nb for nb, _ in adjacency.get(nb1, [])}
                    if nb2 not in nb1_nbs:
                        bridge_count += 1
            betweenness[gid] = bridge_count / max(total_pairs, 1)

        # --- Write scores back to Neo4j in one batch ---
        rows = [
            {
                "id": gid,
                "degree": round(degree.get(gid, 0.0), 6),
                "pagerank": round(pr.get(gid, 0.0), 6),
                "betweenness": round(betweenness.get(gid, 0.0), 6),
                "eigenvector": round(degree.get(gid, 0.0) * 0.5, 6),  # proxy
            }
            for gid in garment_ids
        ]

        updated = 0
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i: i + self.batch_size]
            try:
                await client.query(
                    """
                    UNWIND $batch AS row
                    MATCH (g:Garment {id: row.id})
                    SET g.degree_centrality      = row.degree,
                        g.pagerank               = row.pagerank,
                        g.betweenness_centrality = row.betweenness,
                        g.eigenvector_centrality = row.eigenvector,
                        g.centrality_updated_at  = timestamp()
                    """,
                    {"batch": batch},
                    ttl=0,
                )
                updated += len(batch)
            except Neo4jException as exc:
                logger.error("Centrality batch update failed: %s", exc)

        logger.info("Centrality scores updated: %d garments", updated)
        return updated

    # ------------------------------------------------------------------
    # Step 9 — Validate graph
    # ------------------------------------------------------------------

    async def validate_graph(
        self,
        expected_garments: int = 0,
        expected_min_edges: int = 0,
    ) -> Tuple[bool, List[str]]:
        """
        Sanity-check the graph structure.

        Parameters
        ----------
        expected_garments:
            If > 0, fail if fewer nodes exist.
        expected_min_edges:
            If > 0, fail if fewer COMPATIBLE_WITH edges exist.

        Returns
        -------
        (passed, errors)
            *passed* is *True* when all checks succeed.
        """
        client = self._get_client()
        if client is None:
            return True, []  # fallback — no-op

        errors: List[str] = []

        # Count garments
        try:
            result = await client.query(
                "MATCH (g:Garment) RETURN count(g) AS cnt", ttl=0
            )
            garment_count = result[0]["cnt"] if result else 0
            if expected_garments > 0 and garment_count < expected_garments:
                errors.append(
                    f"Expected ≥{expected_garments} Garment nodes, found {garment_count}"
                )
            logger.info("Garment nodes: %d", garment_count)
        except Neo4jException as exc:
            errors.append(f"Could not count garments: {exc}")

        # Count compatibility edges
        try:
            result = await client.query(
                "MATCH ()-[r:COMPATIBLE_WITH]->() RETURN count(r) AS cnt", ttl=0
            )
            edge_count = result[0]["cnt"] if result else 0
            if expected_min_edges > 0 and edge_count < expected_min_edges:
                errors.append(
                    f"Expected ≥{expected_min_edges} COMPATIBLE_WITH edges, found {edge_count}"
                )
            logger.info("COMPATIBLE_WITH edges: %d", edge_count)
        except Neo4jException as exc:
            errors.append(f"Could not count edges: {exc}")

        # Check for isolated garments (no edges at all)
        try:
            result = await client.query(
                """
                MATCH (g:Garment)
                WHERE NOT (g)-[:COMPATIBLE_WITH]-()
                RETURN count(g) AS cnt
                """,
                ttl=0,
            )
            isolated = result[0]["cnt"] if result else 0
            if isolated > 0:
                logger.warning("%d garments have no compatibility edges", isolated)
        except Neo4jException:
            pass  # non-fatal

        passed = len(errors) == 0
        if passed:
            logger.info("Graph validation passed ✓")
        else:
            logger.warning("Graph validation found %d issue(s): %s", len(errors), errors)

        return passed, errors

    # ------------------------------------------------------------------
    # Full orchestration
    # ------------------------------------------------------------------

    async def run_full_initialization(
        self,
        garments: List[Garment],
        user_id: str = "",
        skip_schema: bool = False,
        skip_centrality: bool = False,
    ) -> GraphBuildResult:
        """
        Run the complete graph-build pipeline.

        Steps
        -----
        1. initialize_schema
        2. create_reference_nodes
        3. load_all_garments
        4. link_all_garments_to_refs
        5. build_compatibility_edges
        6. build_color_harmony_edges
        7. update_centrality_scores (unless *skip_centrality*)
        8. validate_graph

        Parameters
        ----------
        garments:
            Full wardrobe to load.
        user_id:
            Owner's user ID.
        skip_schema:
            Skip schema creation (use when indexes already exist).
        skip_centrality:
            Skip centrality computation (save time on large wardrobes).

        Returns
        -------
        :class:`GraphBuildResult`
        """
        result = GraphBuildResult()
        start = datetime.now(timezone.utc)

        client = self._get_client()
        if client is None:
            result.fallback_used = True
            result.success = True  # graceful degradation
            result.duration_seconds = 0.0
            logger.info("Graph builder: fallback mode (Neo4j unavailable)")
            return result

        try:
            # 1. Schema
            if not skip_schema:
                await self.initialize_schema()

            # 2. Reference nodes
            result.reference_nodes_created = await self.create_reference_nodes()

            # 3. Load garments
            result.garments_loaded = await self.load_all_garments(garments, user_id)

            # 4. Link garments to reference nodes
            result.garment_ref_edges_created = await self.link_all_garments_to_refs(
                garments, user_id
            )

            # 5. Compatibility edges
            result.compatibility_edges_created = await self.build_compatibility_edges(garments)

            # 6. Colour harmony edges
            result.color_harmony_edges_created = await self.build_color_harmony_edges()

            # 7. Centrality (optional)
            if not skip_centrality:
                await self.update_centrality_scores(garments)

            # 8. Validate
            min_edges = max(0, result.garments_loaded - 1)
            passed, v_errors = await self.validate_graph(
                expected_garments=result.garments_loaded,
                expected_min_edges=min_edges,
            )
            result.validation_passed = passed
            result.validation_errors = v_errors

            result.success = True

        except Exception as exc:
            result.errors.append(str(exc))
            logger.error("Graph build failed: %s", exc, exc_info=True)

        finally:
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            result.duration_seconds = elapsed
            logger.info("Graph build finished. %s", result.summary())

        return result
