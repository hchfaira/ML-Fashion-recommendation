"""
Neo4j Outfit Filter
===================

Pre-filtering and smart combination generation backed by Neo4j graph queries.

Pipeline (Phase 3):
    Wardrobe (up to ~150 items)
        ↓ [Step 1] Neo4j season/occasion/formality pre-filter
    Relevant items (~90)
        ↓ [Step 2] Smart hierarchical combination generator
    Valid combos (~100, not cartesian-product)
        ↓ [Step 3] Beam search with Neo4j heuristics  →  30-50 finalists
        ↓ [Step 4] Full OutfitScorecard on finalists only
    Top-K outfits
    Target: < 300 ms total

Public API
----------
- Neo4jOutfitFilter(neo4j_client, min_compatibility=0.5)
- await filter.filter_wardrobe(garments, context)          → List[Garment]
- filter.generate_smart_combinations(filtered, max_combos) → List[List[Garment]]
- await filter.get_compatibility_hint(gid_a, gid_b)        → float
- filter.score_partial_outfit_fast(garments)               → float
"""
from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.core import get_logger
from src.core.models import (
    FormalityLevel,
    Garment,
    GarmentCategory,
    Occasion,
    Season,
    UserContext,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# Map GarmentCategory values to our wardrobe-dict category keys
_CAT_TO_KEY: Dict[str, str] = {
    GarmentCategory.TOP.value: "tops",
    GarmentCategory.BOTTOM.value: "bottoms",
    GarmentCategory.DRESS.value: "full_body",
    GarmentCategory.OUTERWEAR.value: "outerwear",
    GarmentCategory.SHOES.value: "shoes",
    GarmentCategory.ACCESSORY.value: "accessories",
    GarmentCategory.BAG.value: "accessories",
}

# Wardrobe category priority order for anchor selection
_ANCHOR_PRIORITY: List[str] = [
    "full_body",
    "tops",
    "bottoms",
    "outerwear",
    "shoes",
    "accessories",
]

# Formality rank (int) for delta computation
_FORMALITY_RANK: Dict[str, int] = {
    FormalityLevel.VERY_CASUAL.value: 0,
    FormalityLevel.CASUAL.value: 1,
    FormalityLevel.SMART_CASUAL.value: 2,
    FormalityLevel.BUSINESS_CASUAL.value: 3,
    FormalityLevel.BUSINESS.value: 4,
    FormalityLevel.FORMAL.value: 5,
    FormalityLevel.BLACK_TIE.value: 6,
}

# Max formality delta still considered compatible
_MAX_FORMALITY_DELTA = 2


def _garment_category_key(g: Garment) -> str:
    """Map a garment's category to the wardrobe-dict key."""
    return _CAT_TO_KEY.get(g.attributes.category.value, g.attributes.category.value)


def _formality_rank(g: Garment) -> int:
    """Return integer formality rank for a garment."""
    # prefer occasion_profile over legacy field
    if g.attributes.occasion_profile is not None:
        fl = g.attributes.occasion_profile.formality_level
    else:
        fl = g.attributes.formality_level
    return _FORMALITY_RANK.get(fl.value, 1)


def _garment_seasons(g: Garment) -> Set[str]:
    """Return set of season values the garment is suitable for."""
    seasons: Set[str] = set()
    if g.attributes.seasonality and g.attributes.seasonality.seasons:
        for s in g.attributes.seasonality.seasons:
            seasons.add(s.value)
    elif g.attributes.season_suitable:
        for s in g.attributes.season_suitable:
            seasons.add(s.value)
    return seasons


def _garment_occasions(g: Garment) -> Set[str]:
    """Return set of occasion values the garment supports."""
    occasions: Set[str] = set()
    if g.attributes.occasion_profile and g.attributes.occasion_profile.occasions:
        for o in g.attributes.occasion_profile.occasions:
            occasions.add(o if isinstance(o, str) else o.value)
    return occasions


def _local_compatibility(g1: Garment, g2: Garment) -> float:
    """
    Fast local compatibility heuristic (no Neo4j).

    Weights:
      formality match  0.35
      season overlap   0.25
      occasion overlap 0.20
      style overlap    0.20
    """
    # Formality
    delta = abs(_formality_rank(g1) - _formality_rank(g2))
    formality_score = max(0.0, 1.0 - delta / _MAX_FORMALITY_DELTA)

    # Season overlap (Jaccard)
    s1, s2 = _garment_seasons(g1), _garment_seasons(g2)
    if s1 or s2:
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        season_score = intersection / union if union else 0.5
    else:
        season_score = 0.5  # unknown → neutral

    # Occasion overlap (Jaccard)
    o1, o2 = _garment_occasions(g1), _garment_occasions(g2)
    if o1 or o2:
        intersection = len(o1 & o2)
        union = len(o1 | o2)
        occasion_score = intersection / union if union else 0.5
    else:
        occasion_score = 0.5

    # Style overlap (Jaccard on style_tags)
    t1 = set(g1.attributes.style_tags or [])
    t2 = set(g2.attributes.style_tags or [])
    if t1 or t2:
        intersection = len(t1 & t2)
        union = len(t1 | t2)
        style_score = intersection / union if union else 0.5
    else:
        style_score = 0.5

    return (
        0.35 * formality_score
        + 0.25 * season_score
        + 0.20 * occasion_score
        + 0.20 * style_score
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Outcome of the pre-filter step."""

    kept: List[Garment] = field(default_factory=list)
    dropped: List[Garment] = field(default_factory=list)
    neo4j_used: bool = False
    filter_time_ms: float = 0.0

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    def summary(self) -> str:
        return (
            f"FilterResult: {self.kept_count} kept / {self.dropped_count} dropped "
            f"({'Neo4j' if self.neo4j_used else 'local'}, {self.filter_time_ms:.1f} ms)"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Neo4jOutfitFilter:
    """
    Pre-filters a wardrobe and generates smart outfit combinations.

    When a Neo4j client is available the filter augments local heuristics
    with graph-level scores (COMPATIBLE_WITH edge weights, centrality).
    When Neo4j is unavailable it falls back gracefully to pure-Python
    heuristics so the pipeline never fails.

    Parameters
    ----------
    neo4j_client:
        An initialised ``Neo4jClient`` instance (or ``None`` for fallback).
    min_compatibility:
        Edge-weight threshold below which a pair is considered incompatible.
    season_filter_weight:
        Minimum SUITABLE_FOR edge weight to keep a garment for a season.
    top_by_centrality:
        When Neo4j is available, keep at most this many items per category
        (by degree centrality).  Set to ``None`` to disable.
    """

    def __init__(
        self,
        neo4j_client=None,
        *,
        min_compatibility: float = 0.5,
        season_filter_weight: float = 0.5,
        top_by_centrality: Optional[int] = None,
    ) -> None:
        self._client = neo4j_client
        self.min_compatibility = min_compatibility
        self.season_filter_weight = season_filter_weight
        self.top_by_centrality = top_by_centrality

        # Cache: (gid_a, gid_b) → compatibility float
        self._compat_cache: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def filter_wardrobe(
        self,
        garments: List[Garment],
        context: Optional[UserContext] = None,
        *,
        user_id: str = "default",
        max_items: int = 90,
    ) -> FilterResult:
        """
        Step 1: Filter wardrobe to the most relevant items.

        Strategy (with Neo4j):
          1. Query garments matching context season via SUITABLE_FOR edges.
          2. Query garments matching context occasion via SUITABLE_FOR_OCCASION edges.
          3. Intersect + union → relevance-ranked set.
          4. Optionally prune per-category by degree centrality.

        Strategy (fallback / no Neo4j):
          1. Filter by season attribute match.
          2. Filter by occasion attribute match.
          3. Keep all if no season/occasion given in context.
          4. Sort by versatility_score desc.

        Returns
        -------
        FilterResult with `kept` (≤ max_items) and `dropped` lists.
        """
        import time

        t0 = time.perf_counter()

        # Build index: garment_id → Garment
        garment_index: Dict[str, Garment] = {g.id: g for g in garments}

        # Determine context season / occasion
        ctx_season: Optional[str] = None
        ctx_occasion: Optional[str] = None
        if context:
            if context.occasion:
                ctx_occasion = (
                    context.occasion.value
                    if hasattr(context.occasion, "value")
                    else str(context.occasion)
                )

        # Try Neo4j-assisted filtering
        if self._client is not None:
            try:
                kept_ids = await self._neo4j_filter_ids(
                    garment_index, ctx_season, ctx_occasion, user_id, max_items
                )
                kept = [garment_index[gid] for gid in kept_ids if gid in garment_index]
                dropped = [g for g in garments if g.id not in {k.id for k in kept}]
                elapsed = (time.perf_counter() - t0) * 1000
                result = FilterResult(
                    kept=kept, dropped=dropped, neo4j_used=True, filter_time_ms=elapsed
                )
                logger.debug(result.summary())
                return result
            except Exception as exc:  # Neo4j unavailable → graceful fallback
                logger.warning("Neo4j filter failed, using local heuristics: %s", exc)

        # ---- Local fallback ----
        kept = self._local_filter(
            garments,
            ctx_season=ctx_season,
            ctx_occasion=ctx_occasion,
            max_items=max_items,
        )
        dropped = [g for g in garments if g.id not in {k.id for k in kept}]
        elapsed = (time.perf_counter() - t0) * 1000
        result = FilterResult(
            kept=kept, dropped=dropped, neo4j_used=False, filter_time_ms=elapsed
        )
        logger.debug(result.summary())
        return result

    def generate_smart_combinations(
        self,
        filtered: List[Garment],
        *,
        max_combos: int = 100,
        require_top_or_full_body: bool = True,
        require_bottom: bool = True,
        include_shoes: bool = True,
        include_outerwear: bool = False,
        min_combo_score: float = 0.0,
    ) -> List[List[Garment]]:
        """
        Step 2: Hierarchical (non-cartesian) combination generator.

        Instead of computing the full cartesian product, this method:
          1. Buckets filtered garments by category.
          2. Picks an **anchor** item from the highest-priority category
             (full_body > tops > bottoms …).
          3. For each anchor, greedily extends the outfit by selecting the
             most compatible item from each subsequent category.
          4. Randomises anchor order + adds some diversity through shuffling
             to reach *max_combos* diverse combos instead of top-heavy lists.

        Guarantees
        ----------
        - No cartesian explosion: at most O(n × k) pairs computed where
          k = max categories, n = len(filtered).
        - Returns ≤ max_combos combinations.
        - Every combination is a valid minimum outfit (top+bottom OR full_body).

        Returns
        -------
        List of garment lists (each list is one combination).
        """
        buckets = self._bucket_by_category(filtered)

        # Check we have enough pieces for at least one valid outfit
        has_base = bool(buckets.get("full_body")) or (
            bool(buckets.get("tops")) and bool(buckets.get("bottoms"))
        )
        if not has_base:
            logger.warning(
                "generate_smart_combinations: no valid base items; "
                "need tops+bottoms or full_body."
            )
            return []

        combos: List[List[Garment]] = []

        # --- Full-body anchors ---
        for anchor in buckets.get("full_body", []):
            ext = self._extend_outfit(anchor, buckets, ["shoes", "accessories"], 2)
            combos.extend(ext)
            if len(combos) >= max_combos:
                break

        # --- Top+bottom anchors ---
        if len(combos) < max_combos:
            tops = buckets.get("tops", [])
            bottoms = buckets.get("bottoms", [])
            optional_cats = []
            if include_shoes:
                optional_cats.append("shoes")
            if include_outerwear:
                optional_cats.append("outerwear")
            optional_cats.append("accessories")

            # Shuffle for diversity
            tops_shuffled = list(tops)
            random.shuffle(tops_shuffled)

            for top in tops_shuffled:
                # Pick best bottom for this top
                best_bottoms = sorted(
                    bottoms,
                    key=lambda b: _local_compatibility(top, b),
                    reverse=True,
                )[:3]  # take top-3 bottoms per top anchor

                for bottom in best_bottoms:
                    outfit: List[Garment] = [top, bottom]
                    # Extend with optional categories
                    for cat in optional_cats:
                        items = buckets.get(cat, [])
                        if not items:
                            continue
                        best = max(
                            items,
                            key=lambda g: min(
                                _local_compatibility(existing, g)
                                for existing in outfit
                            ),
                        )
                        # Shoes are essential for a complete outfit → use
                        # a lower threshold so they are included more often.
                        # Other optional items use the standard threshold.
                        threshold = (
                            self.min_compatibility * 0.5
                            if cat == "shoes"
                            else self.min_compatibility
                        )
                        # Check compatibility against the *full outfit* (not
                        # just the top) to be consistent with the selection
                        # criterion above.
                        outfit_compat = min(
                            _local_compatibility(existing, best)
                            for existing in outfit
                        )
                        if outfit_compat >= threshold:
                            outfit.append(best)

                    if len(outfit) >= 2:  # top + bottom always valid
                        if min_combo_score <= 0.0 or self.score_partial_outfit_fast(outfit) >= min_combo_score:
                            combos.append(outfit)

                if len(combos) >= max_combos:
                    break

        # Deduplicate (by frozen id sets)
        seen: Set[frozenset] = set()
        unique: List[List[Garment]] = []
        for combo in combos:
            key = frozenset(g.id for g in combo)
            if key not in seen:
                seen.add(key)
                unique.append(combo)
            if len(unique) >= max_combos:
                break

        logger.debug(
            "generate_smart_combinations: %d unique combos from %d filtered items",
            len(unique),
            len(filtered),
        )
        return unique

    async def get_compatibility_hint(
        self,
        garment_id_a: str,
        garment_id_b: str,
    ) -> float:
        """
        Return Neo4j COMPATIBLE_WITH weight between two garments.

        Uses an in-memory cache to avoid repeated graph round-trips.
        Returns 0.5 (neutral) if Neo4j is unavailable or edge not found.
        """
        key = (garment_id_a, garment_id_b)
        key_rev = (garment_id_b, garment_id_a)

        if key in self._compat_cache:
            return self._compat_cache[key]
        if key_rev in self._compat_cache:
            return self._compat_cache[key_rev]

        if self._client is None:
            return 0.5

        try:
            edge = await self._client.get_edge_weight(
                garment_id_a, garment_id_b, "COMPATIBLE_WITH"
            )
            if edge is None:
                edge = await self._client.get_edge_weight(
                    garment_id_b, garment_id_a, "COMPATIBLE_WITH"
                )
            weight: float = float(edge["weight"]) if edge else 0.5
        except Exception:
            weight = 0.5

        self._compat_cache[key] = weight
        return weight

    async def batch_compatibility_hints(
        self,
        pairs: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], float]:
        """
        Fetch compatibility hints for a list of (gid_a, gid_b) pairs.

        All not-yet-cached pairs are fetched concurrently via
        ``asyncio.gather``.  Already-cached pairs are returned directly.

        Returns
        -------
        Dict mapping each input pair (or its reverse) → weight float.
        """
        results: Dict[Tuple[str, str], float] = {}
        missing: List[Tuple[str, str]] = []

        for pair in pairs:
            a, b = pair
            key = (a, b)
            rev = (b, a)
            if key in self._compat_cache:
                results[key] = self._compat_cache[key]
            elif rev in self._compat_cache:
                results[key] = self._compat_cache[rev]
            else:
                missing.append(pair)

        if missing and self._client is not None:
            fetched = await asyncio.gather(
                *[self.get_compatibility_hint(a, b) for a, b in missing],
                return_exceptions=True,
            )
            for (a, b), w in zip(missing, fetched):
                weight = float(w) if isinstance(w, (int, float)) else 0.5
                results[(a, b)] = weight
        else:
            # No client: all missing pairs get neutral score
            for a, b in missing:
                results[(a, b)] = 0.5

        return results

    def score_partial_outfit_fast(self, garments: List[Garment]) -> float:
        """
        Fast O(n²) pairwise compatibility score for a partial/full outfit.

        Uses local heuristics only (no async, no Neo4j).
        Suitable for beam search pruning where speed matters more than precision.

        Includes a small *completeness bonus* that rewards outfits covering
        more wardrobe categories (especially shoes), so fully-dressed outfits
        are preferred over incomplete ones when compatibility is similar.

        Returns
        -------
        Score in [0, 1].
        """
        if len(garments) < 2:
            return 0.5

        total = 0.0
        count = 0
        for i in range(len(garments)):
            for j in range(i + 1, len(garments)):
                total += _local_compatibility(garments[i], garments[j])
                count += 1

        mean_compat = total / count if count > 0 else 0.5

        # Completeness bonus: reward outfits that include shoes and more
        # categories.  Capped so it nudges ranking without dominating.
        cats = {_garment_category_key(g) for g in garments}
        completeness = 0.0
        if "shoes" in cats:
            completeness += 0.05
        if len(cats) >= 4:
            completeness += 0.02
        elif len(cats) >= 3:
            completeness += 0.01

        return min(1.0, mean_compat + completeness)

    # ------------------------------------------------------------------
    # Neo4j-backed internals
    # ------------------------------------------------------------------

    async def _neo4j_filter_ids(
        self,
        garment_index: Dict[str, Garment],
        ctx_season: Optional[str],
        ctx_occasion: Optional[str],
        user_id: str,
        max_items: int,
    ) -> List[str]:
        """
        Query Neo4j for contextually relevant garment IDs.

        Relevance score = season_weight * 0.5 + occasion_weight * 0.5.
        Falls back to degree_centrality if no context given.
        """
        relevance: Dict[str, float] = defaultdict(float)
        all_ids = set(garment_index.keys())

        async def _fetch_season() -> None:
            if ctx_season is None:
                return
            rows = await self._client.get_garments_by_season(  # type: ignore[union-attr]
                season=ctx_season, user_id=user_id
            )
            for row in rows:
                gid = row.get("garment_id")
                score = float(row.get("score", 0.8))
                if gid in all_ids:
                    relevance[gid] += score * 0.5

        async def _fetch_occasion() -> None:
            if ctx_occasion is None:
                return
            rows = await self._client.get_garments_by_occasion(  # type: ignore[union-attr]
                occasion=ctx_occasion, user_id=user_id
            )
            for row in rows:
                gid = row.get("garment_id")
                score = float(row.get("score", 0.8))
                if gid in all_ids:
                    relevance[gid] += score * 0.5

        await asyncio.gather(_fetch_season(), _fetch_occasion())

        if not relevance:
            # No context filters matched → use all items, sort by degree_centrality
            for gid in all_ids:
                relevance[gid] = 0.5

        # Sort by relevance descending, then limit
        ranked = sorted(relevance.items(), key=lambda x: x[1], reverse=True)
        return [gid for gid, _ in ranked[:max_items]]

    # ------------------------------------------------------------------
    # Local fallback internals
    # ------------------------------------------------------------------

    def _local_filter(
        self,
        garments: List[Garment],
        *,
        ctx_season: Optional[str],
        ctx_occasion: Optional[str],
        max_items: int,
    ) -> List[Garment]:
        """
        Pure-Python pre-filter.

        Scoring per garment:
          +0.5 if matches context season
          +0.5 if matches context occasion
          +versatility_score (0-1) as tie-breaker
        If no context is given, all garments score 1.0.
        """
        scored: List[Tuple[float, Garment]] = []

        for g in garments:
            score = 0.0

            if ctx_season is None and ctx_occasion is None:
                score = 1.0
            else:
                if ctx_season is not None:
                    seasons = _garment_seasons(g)
                    if not seasons or ctx_season in seasons:
                        score += 0.5  # unknown seasons pass through

                if ctx_occasion is not None:
                    occasions = _garment_occasions(g)
                    if not occasions or ctx_occasion in occasions:
                        score += 0.5

            # Tie-breaker: versatility
            if g.attributes.versatility:
                score += g.attributes.versatility.versatility_score * 0.1

            scored.append((score, g))

        # Sort by score desc, keep top max_items
        scored.sort(key=lambda x: x[0], reverse=True)
        # Remove items that scored 0 when context was given
        if ctx_season is not None or ctx_occasion is not None:
            scored = [(s, g) for s, g in scored if s > 0.0]

        return [g for _, g in scored[:max_items]]

    # ------------------------------------------------------------------
    # Combination helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bucket_by_category(
        garments: List[Garment],
    ) -> Dict[str, List[Garment]]:
        """Group garments by their wardrobe-dict category key."""
        buckets: Dict[str, List[Garment]] = defaultdict(list)
        for g in garments:
            key = _garment_category_key(g)
            buckets[key].append(g)
        return dict(buckets)

    def _extend_outfit(
        self,
        anchor: Garment,
        buckets: Dict[str, List[Garment]],
        optional_cats: List[str],
        max_variants: int = 3,
    ) -> List[List[Garment]]:
        """
        Extend *anchor* into up to *max_variants* outfits by adding optional items.

        For each optional category, selects the best-compatible item.
        Returns a list of complete outfits.
        """
        base: List[Garment] = [anchor]

        # Collect the best optional item per category
        extensions: List[Garment] = []
        for cat in optional_cats:
            items = buckets.get(cat, [])
            if not items:
                continue
            best = max(
                items,
                key=lambda g: _local_compatibility(anchor, g),
            )
            # Shoes use a lower threshold (essential for a complete outfit)
            threshold = (
                self.min_compatibility * 0.5
                if cat == "shoes"
                else self.min_compatibility
            )
            if _local_compatibility(anchor, best) >= threshold:
                extensions.append(best)

        if not extensions:
            return [base]

        # Build combos: anchor alone, anchor+each, anchor+all
        combos = [base]
        for ext in extensions:
            combos.append(base + [ext])
        if len(extensions) > 1:
            combos.append(base + extensions)

        return combos[:max_variants]
