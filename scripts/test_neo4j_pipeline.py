"""
Neo4j Full Pipeline Test Script
================================

Mirror of ``scripts/test_full_pipeline.py`` — same inputs, same CLI flags —
but every outfit-generation step is accelerated by Neo4j:

  Phase 1 — Schema + reference nodes created in Neo4j
  Phase 2 — Garments + compatibility/colour-harmony edges loaded into Neo4j
  Phase 3 — Outfit combinations found via Neo4j beam-search (<<300 ms)
  Phase 4 — Smart-removal verdicts enriched with graph-centrality scores
  Phase 5 — Incremental sync + consistency audit after the run

Neo4j must be running locally (docker-compose up -d) before you start.
The script connects using the credentials in  config/data/neo4j_config.json
(URI: bolt://localhost:7687, user: neo4j).

Usage
-----
  # Quickest demo — synthetic garments, no Vision API needed
  python scripts/test_neo4j_pipeline.py --demo

  # Real wardrobe folder (each image = one garment)
  python scripts/test_neo4j_pipeline.py --wardrobe data/sample_wardrobe/

  # With user profile photo for context-aware scoring
  python scripts/test_neo4j_pipeline.py --wardrobe data/sample_wardrobe/ --user-photo data/people/image.png

  # With measurements
  python scripts/test_neo4j_pipeline.py --wardrobe data/sample_wardrobe/ --user-photo data/people/image.png --height 170 --weight 65

  # Skip the Neo4j graph build (graph already built from a previous run)
  python scripts/test_neo4j_pipeline.py --wardrobe data/sample_wardrobe/ --skip-graph-build

  # Run Layer-0 quality check before the main pipeline
  python scripts/test_neo4j_pipeline.py --wardrobe data/sample_wardrobe/ --quality-check

  # Also run smart-removal analysis on the wardrobe
  python scripts/test_neo4j_pipeline.py --demo --removal-analysis

  # Save full results to disk
  python scripts/test_neo4j_pipeline.py --demo --save-results

Requirements
------------
  - docker-compose up -d          (Neo4j must be reachable)
  - GOOGLE_API_KEY set            (only when using --wardrobe / real images)
  - .venv/bin/python or python3   (project virtual environment)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Make sure the project root is on the path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import get_logger
from src.core.models import (
    FormalityLevel,
    Garment,
    GarmentAttributes,
    GarmentCategory,
    GarmentHistory,
    ColorProfile,
    PatternInfo,
    Occasion,
    UserContext,
)

# Jobs
from src.jobs import (
    Neo4jGraphBuilder,
    CentralityRefreshJob,
    Neo4jSyncJob,
    SyncResult,
    ConsistencyReport,
)

# Neo4j client
from src.core.neo4j_client import Neo4jClient

# Layer 2 — Neo4j outfit builder
from src.layer2_style.outfit_builder import OutfitBuilder, OutfitCandidate
from src.layer2_style.smart_removal_analyzer import SmartRemovalAnalyzer

# Layer 2 — hybrid recommender + colour/body helpers
from src.layer2_style.hybrid_recommender import HybridOutfitRecommender, RankedOutfit
from src.layer2_style.season_color_harmony import ColorSeason
from src.layer2_style.volume_balance_scorer import BodyShape as VolumeBodyShape
from src.layer2_style import OutfitScorecard

# Layer 3 — user profile pipeline
from src.layer3_context.user_profile import StyleProfilePipeline, PipelineConfig, StyleProfile

# Layer 4 — LLM outfit improvement explainer
from src.layer4_llm.outfit_improvement_explainer import OutfitImprovementExplainer

# Layer 3 — context engine components
from src.layer3_context import (
    ContextEngine,
    ContextCriteria,
    FitPredictor,
    ProportionHarmonizer,
    ColorHarmonyAdvisor,
)

# Layer 5 — visualization
from src.layer5_visualization.minimalist_visualizer import QuietLuxuryVisualizer

# Pipeline helpers (reused from the existing scripts/pipeline package)
from pipeline.report_generator import ReportGenerator
from pipeline.results_storage import ResultsStorage

# ---------------------------------------------------------------------------
# Layer 0 — Quality Check (optional, activated by --quality-check)
# ---------------------------------------------------------------------------
try:
    import numpy as _np
    from PIL import Image as _PIL_Image
    from uuid import uuid4 as _uuid4
    from src.layer0_segmentation.pipeline import create_pipeline as _create_l0_pipeline
    from src.layer0_segmentation.models import ExtractedGarment as _ExtractedGarment
    from src.layer0_segmentation.quality_checker import QualityChecker as _QualityChecker
    from src.layer0_segmentation.session_manager import ImportSessionManager as _ImportSessionManager
    from src.layer0_segmentation.taxonomy import GarmentCategory as _L0Category
    _LAYER0_AVAILABLE = True
except ImportError:
    _LAYER0_AVAILABLE = False

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTION = "=" * 70
SUB = "-" * 60
USER_ID = "local_test_user"


# ===========================================================================
# Helpers
# ===========================================================================


def _banner(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def _sub(title: str) -> None:
    print(f"\n{SUB}")
    print(f"  {title}")
    print(SUB)


def _ok(msg: str) -> None:
    print(f"  ✅  {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠️   {msg}")


def _info(msg: str) -> None:
    print(f"  ℹ️   {msg}")


# ---------------------------------------------------------------------------
# Demo garments (mirrors the demo in test_full_pipeline.py)
# ---------------------------------------------------------------------------


def build_demo_wardrobe() -> List[Garment]:
    """Return the same synthetic wardrobe as the original pipeline demo."""
    from uuid import uuid4

    def _g(cat, sub, color, formality=FormalityLevel.CASUAL):
        return Garment(
            id=str(uuid4()),
            attributes=GarmentAttributes(
                category=GarmentCategory(cat),
                subcategory=sub,
                color=ColorProfile(primary=color, hex_codes=[]),
                pattern=PatternInfo(type="solid"),
                formality_level=formality,
                season_suitable=["spring", "summer", "fall", "winter"],
            ),
            history=GarmentHistory(),
        )

    return [
        _g("top",       "white t-shirt",   "white"),
        _g("top",       "navy shirt",      "navy",   FormalityLevel.SMART_CASUAL),
        _g("top",       "gray sweater",    "gray",   FormalityLevel.SMART_CASUAL),
        _g("top",       "black blouse",    "black",  FormalityLevel.BUSINESS),
        _g("bottom",    "blue jeans",      "blue"),
        _g("bottom",    "black pants",     "black",  FormalityLevel.SMART_CASUAL),
        _g("bottom",    "beige chinos",    "beige",  FormalityLevel.SMART_CASUAL),
        _g("shoes",     "white sneakers",  "white"),
        _g("shoes",     "brown boots",     "brown",  FormalityLevel.SMART_CASUAL),
        _g("shoes",     "black loafers",   "black",  FormalityLevel.BUSINESS),
        _g("outerwear", "black jacket",    "black",  FormalityLevel.SMART_CASUAL),
        _g("outerwear", "beige coat",      "beige",  FormalityLevel.BUSINESS),
    ]


def garments_to_wardrobe_dict(garments: List[Garment]) -> Dict[str, List[Garment]]:
    """Convert a flat garment list to the category-keyed dict expected by OutfitBuilder."""
    wardrobe: Dict[str, List[Garment]] = {}
    cat_key_map = {
        GarmentCategory.TOP.value:       "tops",
        GarmentCategory.BOTTOM.value:    "bottoms",
        GarmentCategory.DRESS.value:     "full_body",
        GarmentCategory.OUTERWEAR.value: "outerwear",
        GarmentCategory.SHOES.value:     "shoes",
        GarmentCategory.ACCESSORY.value: "accessories",
        GarmentCategory.BAG.value:       "accessories",
    }
    for g in garments:
        key = cat_key_map.get(g.attributes.category.value, "tops")
        wardrobe.setdefault(key, []).append(g)
    return wardrobe


# ===========================================================================
# Phase 1 + 2 — Build / sync the Neo4j graph
# ===========================================================================


async def phase_graph_build(
    garments: List[Garment],
    client: Neo4jClient,
    *,
    skip: bool = False,
    skip_centrality: bool = False,
) -> bool:
    """
    Load the wardrobe into Neo4j (Phase 1 schema + Phase 2 graph build).

    Returns True on success (or True in fallback / skip mode).
    """
    _banner("Phase 2 — Neo4j Graph Build")

    if skip:
        _info("--skip-graph-build set, reusing existing graph.")
        return True

    builder = Neo4jGraphBuilder(fallback_enabled=True)
    builder._client = client

    print(f"\n  Loading {len(garments)} garments into Neo4j for user '{USER_ID}' …")
    t0 = time.perf_counter()

    result = await builder.run_full_initialization(
        garments,
        user_id=USER_ID,
        skip_centrality=skip_centrality,
    )

    elapsed = time.perf_counter() - t0

    if result.fallback_used:
        _warn("Neo4j is unreachable — graph build skipped (fallback mode).")
        _warn("Outfit generation will use local heuristics only.")
        return True

    if result.success:
        _ok(f"Graph built in {elapsed:.2f}s")
        print(f"      • Reference nodes   : {result.reference_nodes_created}")
        print(f"      • Garments loaded   : {result.garments_loaded}")
        print(f"      • Compat. edges     : {result.compatibility_edges_created}")
        print(f"      • Color edges       : {result.color_harmony_edges_created}")
        print(f"      • Validation        : {'PASSED' if result.validation_passed else 'FAILED'}")
        if result.validation_errors:
            for err in result.validation_errors:
                _warn(f"Validation: {err}")
    else:
        _warn(f"Graph build encountered errors: {result.errors}")

    return result.success or result.fallback_used


# ===========================================================================
# Phase 3 — Neo4j Beam-Search Outfit Generation
# ===========================================================================


async def phase_outfit_generation(
    garments: List[Garment],
    client: Optional[Neo4jClient],
    *,
    context: Optional[UserContext] = None,
    top_k: int = 5,
    profile: str = "default",
) -> List[OutfitCandidate]:
    """Run the 4-step Neo4j beam-search and return top-K outfit candidates."""
    _banner("Phase 3 — Neo4j Beam-Search Outfit Generation")

    wardrobe_dict = garments_to_wardrobe_dict(garments)
    builder = OutfitBuilder()

    categories_summary = {k: len(v) for k, v in wardrobe_dict.items()}
    print(f"\n  Wardrobe: {len(garments)} garments across {len(wardrobe_dict)} categories")
    for cat, cnt in sorted(categories_summary.items()):
        print(f"      {cat:<15} {cnt}")

    _sub("Running beam-search …")
    t0 = time.perf_counter()

    candidates = await builder.neo4j_beam_search_outfits(
        wardrobe_dict,
        neo4j_client=client,
        context=context,
        user_id=USER_ID,
        top_k=top_k,
        profile=profile,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _ok(f"Completed in {elapsed_ms:.0f} ms — {len(candidates)} outfit(s) ranked")

    if not candidates:
        _warn("No outfit candidates produced.")
        return []

    # ── Print ranking table ──────────────────────────────────────────────
    print()
    print(f"  {'Rank':<5} {'Outfit':<40} {'Score':>7}")
    print(f"  {'-'*5} {'-'*40} {'-'*7}")
    for i, c in enumerate(candidates, 1):
        names = " + ".join(
            (g.attributes.subcategory or g.attributes.category.value)[:14]
            for g in c.garments[:3]
        )
        if len(c.garments) > 3:
            names += f" (+{len(c.garments)-3})"
        marker = " 🥇" if i == 1 else ""
        print(f"  {i:<5} {names:<40} {c.overall_score:>6.1%}{marker}")

    # ── Detail on best outfit ────────────────────────────────────────────
    best = candidates[0]
    _sub("Best Outfit — Score Breakdown")
    full_names = " + ".join(
        g.attributes.subcategory or g.attributes.category.value
        for g in best.garments
    )
    print(f"  {full_names}")
    print(f"  Overall score : {best.overall_score:.1%}")
    if best.scorecard and best.scorecard.scores:
        for dim, val in sorted(best.scorecard.scores.items()):
            if dim != "overall" and isinstance(val, (int, float)):
                bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                print(f"    {dim:<28} {bar}  {val:.0%}")

    return candidates


# ===========================================================================
# Phase 4 — Smart Removal with Neo4j Centrality
# ===========================================================================


async def phase_smart_removal(
    garments: List[Garment],
    client: Optional[Neo4jClient],
    *,
    profile_name: str = "default",
    top_n: int = 5,
) -> None:
    """Run Neo4j-enriched smart removal analysis and print verdicts."""
    _banner("Phase 4 — Smart Removal (Neo4j Centrality Enrichment)")

    analyzer = SmartRemovalAnalyzer(profile_name=profile_name)

    print(f"\n  Analyzing {len(garments)} garments …")
    t0 = time.perf_counter()

    verdicts = await analyzer.analyze_wardrobe_with_neo4j(
        garments, client
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _ok(f"Analysis complete in {elapsed_ms:.0f} ms")

    if not verdicts:
        _warn("No verdicts produced.")
        return

    _sub(f"Top {top_n} Removal Candidates (lowest regret-risk first)")
    print(f"\n  {'Garment':<30} {'Verdict':<22} {'Risk':>6} {'Centrality':>11}")
    print(f"  {'-'*30} {'-'*22} {'-'*6} {'-'*11}")

    for v in verdicts[:top_n]:
        garment_name = v.garment_description or v.garment_id[:20]
        centrality = v.signals.get("centrality", 0.5)
        centrality_str = f"{centrality:.0%}" if centrality != 0.5 else "n/a"
        print(
            f"  {garment_name:<30} {v.verdict:<22} "
            f"{v.regret_risk:>5.0%}  {centrality_str:>10}"
        )

    print()
    verdict_counts: Dict[str, int] = {}
    for v in verdicts:
        verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1
    print("  Summary:", "  |  ".join(f"{k}: {n}" for k, n in sorted(verdict_counts.items())))


# ===========================================================================
# Phase 5 — Full Sync + Consistency Check
# ===========================================================================


async def phase_sync_and_consistency(
    garments: List[Garment],
    client: Optional[Neo4jClient],
) -> None:
    """Run incremental sync and consistency audit."""
    _banner("Phase 5 — Sync & Consistency Check")

    sync_job = Neo4jSyncJob(fallback_enabled=True)
    if client is not None:
        sync_job._client = client

    # Incremental sync (all garments treated as "new" for demo purposes)
    _sub("Incremental Sync")
    print(f"\n  Syncing {len(garments)} garments …")
    t0 = time.perf_counter()
    result: SyncResult = await sync_job.sync_garments(garments, USER_ID)
    elapsed = time.perf_counter() - t0

    if result.fallback_used:
        _warn("Neo4j unavailable — sync skipped.")
    elif result.success:
        _ok(f"Sync done in {elapsed:.2f}s  (upserted={result.upserted})")
    else:
        _warn(f"Sync failed (errors={result.errors_count})")

    # Consistency check
    _sub("Consistency Audit")
    print("\n  Running consistency check …")
    report: ConsistencyReport = await sync_job.run_consistency_check(garments, USER_ID)

    if report.is_consistent:
        _ok("Graph is fully consistent with the in-memory wardrobe.")
    else:
        _warn(f"Inconsistencies found:")
        if report.orphaned_node_ids:
            print(f"    • Orphaned nodes   : {len(report.orphaned_node_ids)}")
        if report.missing_node_ids:
            print(f"    • Missing nodes    : {len(report.missing_node_ids)}")
        if report.edge_anomalies:
            for anomaly in report.edge_anomalies[:3]:
                print(f"    • Edge anomaly: {anomaly}")
        if report.recommendations:
            print("\n  Recommendations:")
            for rec in report.recommendations:
                print(f"    → {rec}")

        # Auto-repair
        print("\n  Attempting auto-repair …")
        repair = await sync_job.repair_consistency(report, garments, USER_ID)
        if repair.success:
            _ok(
                f"Repair complete (upserted={repair.upserted}, "
                f"deleted={repair.deleted_count})"
            )
        else:
            _warn("Auto-repair failed.")

    await sync_job.close()


# ===========================================================================
# Phase 6 — Layer 4 LLM Outfit Improvement
# ===========================================================================


async def phase_outfit_improvement(
    outfit_candidate: OutfitCandidate,
    wardrobe: List[Garment],
    user_season: Optional[ColorSeason] = None,
    body_shape: Optional[VolumeBodyShape] = None,
) -> Optional[Dict[str, Any]]:
    """Generate LLM-based improvement suggestions for the best outfit."""
    _banner("Phase 6 — Layer 4 LLM Outfit Improvements")

    try:
        explainer = OutfitImprovementExplainer(max_suggestions=5)
    except Exception as exc:
        _warn(f"Could not initialize OutfitImprovementExplainer: {exc}")
        logger.warning(f"OutfitImprovementExplainer init failed: {exc}")
        return None

    if not outfit_candidate.garments:
        _warn("No garments in best outfit — skipping improvement suggestions.")
        return None

    print(f"\n  Analyzing outfit: {outfit_candidate.name}")
    print(f"  User season: {user_season.value if user_season else 'Not specified'}")
    print(f"  Body shape: {body_shape.value if body_shape else 'Not specified'}")

    try:
        improvement = await explainer.explain_improvements(
            garments=outfit_candidate.garments,
            wardrobe=wardrobe,
            context=None,
            user_season=user_season,
            body_shape=body_shape,
            tone="friendly",
            detail_level="standard",
        )

        if improvement:
            # Print outfit name and diagnosis
            print(f"\n  Outfit Name: {improvement.outfit_name}")
            if improvement.diagnosis_summary:
                import textwrap as _tw
                _diag_lines = _tw.wrap(improvement.diagnosis_summary, width=72)
                print(f"  Diagnosis: {_diag_lines[0]}")
                for _dl in _diag_lines[1:]:
                    print(f"             {_dl}")

            # Print suggested pieces
            if improvement.suggested_pieces:
                _sub(f"Suggested Improvements ({len(improvement.suggested_pieces)} pieces)")
                for i, piece in enumerate(improvement.suggested_pieces, 1):
                    print(f"\n  {i}. {piece.garment_description} ({piece.suggestion_type})")
                    print(f"     Expected score change: {piece.expected_score_change:+.1%}")
                    if piece.llm_explanation:
                        reason_lines = _tw.wrap(piece.llm_explanation.strip(), width=68)
                        print(f"     Reason: {reason_lines[0]}")
                        for rl in reason_lines[1:]:
                            print(f"             {rl}")

            # Convert to dict for JSON report
            return improvement.to_dict()
        else:
            _warn("OutfitImprovementExplainer returned no improvement.")
            return None

    except Exception as exc:
        logger.warning(f"OutfitImprovementExplainer failed: {exc}")
        _warn(f"Outfit improvement failed: {exc}")
        return None


# ===========================================================================
# Neo4j health check
# ===========================================================================


async def check_neo4j_connection() -> Optional[Neo4jClient]:
    """
    Try to connect to Neo4j.  Returns client on success, None on failure.
    Prints a clear error if Neo4j is not running.
    """
    print("\n  🔌 Connecting to Neo4j …")
    try:
        client = Neo4jClient()
        healthy = await client.health_check()
        if healthy:
            stats = client.get_statistics()  # not async
            _ok(
                f"Neo4j connected  "
                f"(queries={stats.get('queries_executed', 0)}, "
                f"cache={stats.get('cache_stats', {}).get('size', 0)} entries)"
            )
            return client
        else:
            _warn("Neo4j returned an unhealthy response.")
            return None
    except Exception as exc:
        _warn(f"Cannot connect to Neo4j: {exc}")
        print()
        print("  ─── Is Neo4j running? Try: ───────────────────────────────")
        print("    docker-compose up -d neo4j")
        print("  ──────────────────────────────────────────────────────────")
        print()
        print("  Continuing in LOCAL FALLBACK mode (no Neo4j acceleration).")
        return None


# ===========================================================================
# Load wardrobe from disk (reuses FullPipelineTester)
# ===========================================================================


async def load_wardrobe_from_folder(wardrobe_path: Path, profile: str) -> List[Garment]:
    """Extract garments from a flat folder of images using OutfitBuilder (Layer 1)."""
    _banner("Phase 1 — Wardrobe Loading (Vision AI)")
    print(f"\n  Loading images from: {wardrobe_path}")

    builder = OutfitBuilder()

    # For flat folders (all images in one dir) use load_outfit_from_folder
    # which extracts + categorises each image via the Vision API.
    garments: List[Garment] = await builder.load_outfit_from_folder(wardrobe_path)

    if garments:
        _ok(f"{len(garments)} garment(s) extracted")
        for g in garments:
            col = g.attributes.color.primary if g.attributes.color else "?"
            sub = g.attributes.subcategory or g.attributes.category.value
            cat = g.attributes.category.value
            print(f"    • {cat:<12} {sub:<30} {col}")
    else:
        _warn("No garments extracted — check that the folder contains images and GOOGLE_API_KEY is set.")

    return garments


# ===========================================================================
# Layer 0 — category conversion + quality-check wardrobe loader
# ===========================================================================

# Maps Layer 0 taxonomy values → core GarmentCategory
_L0_TO_CORE_CAT: Dict[str, "GarmentCategory"] = {}  # populated lazily below

def _build_cat_map() -> Dict[str, "GarmentCategory"]:
    return {
        "tops":        GarmentCategory.TOP,
        "bottoms":     GarmentCategory.BOTTOM,
        "full_body":   GarmentCategory.DRESS,
        "outerwear":   GarmentCategory.OUTERWEAR,
        "footwear":    GarmentCategory.SHOES,
        "accessories": GarmentCategory.ACCESSORY,
        "unknown":     GarmentCategory.TOP,  # safe fallback
    }


# ---------------------------------------------------------------------------
# Filename-based category heuristics for Layer-0 simple mode
# (GroundingDINO not available → every garment comes out as "unknown")
# ---------------------------------------------------------------------------

_FILENAME_CAT_HINTS: List[Tuple[List[str], "GarmentCategory", str]] = []
# Populated lazily after GarmentCategory is imported

def _build_filename_hints() -> List[Tuple[List[str], Any, str]]:
    return [
        # (keywords, category, subcategory_label)
        (["pants", "jeans", "trouser", "chino", "skirt", "shorts", "bottom", "leg"],
         GarmentCategory.BOTTOM, "pants"),
        (["dress", "gown", "jumpsuit", "romper", "overall"],
         GarmentCategory.DRESS, "dress"),
        (["jacket", "coat", "blazer", "outerwear", "parka", "trench"],
         GarmentCategory.OUTERWEAR, "jacket"),
        (["shoe", "boot", "sneaker", "heel", "sandal", "loafer", "footwear"],
         GarmentCategory.SHOES, "shoes"),
        (["bag", "purse", "scarf", "hat", "belt", "accessory", "jewelry", "watch"],
         GarmentCategory.ACCESSORY, "accessory"),
        (["shirt", "tshirt", "t-shirt", "blouse", "top", "sweater", "hoodie",
          "tank", "crop", "polo", "knit", "cardigan"],
         GarmentCategory.TOP, "shirt"),
    ]

_FILENAME_HINTS: List[Tuple[List[str], Any, str]] = []  # populated lazily


def _category_from_filename(filename: str) -> Tuple[Any, str]:
    """
    Guess GarmentCategory + subcategory label from the image filename.
    Falls back to (GarmentCategory.TOP, 'garment') when no keyword matches.
    """
    global _FILENAME_HINTS
    if not _FILENAME_HINTS:
        _FILENAME_HINTS = _build_filename_hints()

    name_lower = Path(filename).stem.lower().replace("-", " ").replace("_", " ")
    for keywords, cat, sub in _FILENAME_HINTS:
        if any(kw in name_lower for kw in keywords):
            return cat, sub
    return GarmentCategory.TOP, "garment"


def _extract_dominant_color(image_array: "_np.ndarray") -> str:
    """
    Extract the dominant color from an image.
    Returns a color name string (e.g. 'white', 'black', 'blue', 'red', 'green', 'yellow', 'brown', 'gray', 'purple', 'pink', 'orange').
    Falls back to 'unknown' if the image is too small.
    """
    try:
        if image_array is None or image_array.size == 0:
            return "unknown"
        
        # Reshape to (n_pixels, 3) and convert to RGB if needed
        if len(image_array.shape) != 3 or image_array.shape[2] < 3:
            return "unknown"
        
        pixels = image_array.reshape(-1, image_array.shape[2])[:, :3].astype(float)
        
        # Get dominant color (average of top-k brightest pixels for better results)
        brightness = pixels.sum(axis=1)
        top_k = max(10, len(pixels) // 100)  # top 1% of brightest pixels
        top_indices = _np.argsort(brightness)[-top_k:]
        dominant_rgb = pixels[top_indices].mean(axis=0).astype(int)
        
        r, g, b = dominant_rgb
        
        # Simple color classification based on RGB
        # (max channel defines the hue direction)
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c
        
        # Check for achromatic colors first
        if delta < 30:  # low saturation
            if max_c > 200:
                return "white"
            elif max_c < 50:
                return "black"
            else:
                return "gray"
        
        # Chromatic colors
        if max_c == r:
            if g > b:
                return "orange" if g > 150 else "brown"
            else:
                return "red" if r > 150 else "purple"
        elif max_c == g:
            return "green" if g > 150 else "olive"
        else:  # max_c == b
            return "blue" if b > 150 else "navy"
    except Exception:
        return "unknown"


def _detect_pattern(image_array: "_np.ndarray") -> str:
    """
    Detect pattern type from image texture analysis.
    Returns: 'solid', 'striped', 'checkered', 'floral', 'patterned', or 'textured'.
    Falls back to 'solid' if analysis fails.
    """
    try:
        if image_array is None or image_array.size == 0 or len(image_array.shape) != 3:
            return "solid"
        
        # Convert to grayscale for edge detection
        gray = _np.dot(image_array[..., :3], [0.299, 0.587, 0.114]).astype(_np.uint8)
        
        # Compute Laplacian (edge detection) as a texture measure
        # High variance in edges = textured/patterned
        if gray.size < 100:
            return "solid"
        
        gy, gx = _np.gradient(gray.astype(float))
        edge_strength = _np.sqrt(gx**2 + gy**2)
        edge_variance = edge_strength.var()
        
        if edge_variance < 50:
            return "solid"
        elif edge_variance < 150:
            return "textured"
        elif edge_variance < 300:
            return "patterned"
        else:
            return "striped"
    except Exception:
        return "solid"


def _extracted_to_garment(eg: "_ExtractedGarment", image_array: Optional["_np.ndarray"] = None) -> Garment:
    """
    Convert a Layer-0 ExtractedGarment into a core Garment.
    
    Args:
        eg: ExtractedGarment from Layer 0
        image_array: Optional original image for color/pattern extraction
    """
    global _L0_TO_CORE_CAT
    if not _L0_TO_CORE_CAT:
        _L0_TO_CORE_CAT = _build_cat_map()

    l0_cat_val = eg.category.value  # e.g. "tops", "bottoms", "unknown"
    if l0_cat_val != "unknown":
        # Layer 0 gave a real category — use it
        core_cat = _L0_TO_CORE_CAT.get(l0_cat_val, GarmentCategory.TOP)
        sub = eg.label
    else:
        # simple mode → try filename heuristic
        src = eg.source_path or ""
        core_cat, sub = _category_from_filename(src)

    # Extract color and pattern from image if available
    color_name = eg.metadata.get("color", None)
    pattern_name = eg.metadata.get("pattern", None)
    
    if color_name is None and image_array is not None:
        color_name = _extract_dominant_color(image_array)
    color_name = color_name or "unknown"
    
    if pattern_name is None and image_array is not None:
        pattern_name = _detect_pattern(image_array)
    pattern_name = pattern_name or "solid"

    from uuid import uuid4
    return Garment(
        id=str(uuid4()),
        attributes=GarmentAttributes(
            category=core_cat,
            subcategory=sub,
            color=ColorProfile(
                primary=color_name,
                hex_codes=[],
            ),
            pattern=PatternInfo(type=pattern_name),
            formality_level=FormalityLevel.CASUAL,
            season_suitable=["spring", "summer", "fall", "winter"],
        ),
        history=GarmentHistory(),
    )


async def load_wardrobe_with_quality_check(
    wardrobe_path: Path,
    profile: str,
    *,
    interactive: bool = False,
) -> List[Garment]:
    """
    Load wardrobe images through the Layer-0 extraction pipeline,
    then run QualityChecker + ImportSessionManager before converting
    accepted ExtractedGarments into core Garment objects.

    If Layer-0 is not available (missing optional dependencies) the function
    falls back to the standard vision-API loader automatically.
    """
    if not _LAYER0_AVAILABLE:
        _warn("Layer-0 dependencies not available — falling back to Vision AI loader.")
        return await load_wardrobe_from_folder(wardrobe_path, profile)

    _banner("Phase 1 — Wardrobe Loading + Quality Check (Layer 0)")
    print(f"\n  Loading images from: {wardrobe_path}")

    # ── 1. Collect image paths ────────────────────────────────────────────
    image_paths: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG"):
        image_paths.extend(wardrobe_path.glob(ext))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        _warn("No images found in the wardrobe folder.")
        return []

    print(f"  Found {len(image_paths)} image(s) to process.\n")

    # ── 2. Run Layer-0 extraction pipeline per image ──────────────────────
    all_extracted: List["_ExtractedGarment"] = []
    image_array_map: Dict[str, "_np.ndarray"] = {}  # map source_path → image array
    try:
        l0_pipeline = _create_l0_pipeline(mode="auto")
    except Exception as exc:
        _warn(f"Could not create Layer-0 pipeline: {exc} — falling back to Vision AI loader.")
        return await load_wardrobe_from_folder(wardrobe_path, profile)

    for img_path in image_paths:
        try:
            pil_img = _PIL_Image.open(img_path).convert("RGB")
            img_array = _np.array(pil_img)
            result = l0_pipeline.process(img_array)
            if result.garments:
                for eg in result.garments:
                    if eg.source_path is None:
                        eg.source_path = img_path
                    image_array_map[str(eg.source_path)] = img_array
                all_extracted.extend(result.garments)
                _ok(f"  {img_path.name}: {len(result.garments)} garment(s) extracted")
            else:
                _warn(f"  {img_path.name}: no garments detected")
        except Exception as exc:
            _warn(f"  {img_path.name}: Layer-0 failed ({exc}) — skipped")

    if not all_extracted:
        _warn("Layer-0 extracted no garments — falling back to Vision AI loader.")
        return await load_wardrobe_from_folder(wardrobe_path, profile)

    # ── 3 & 4. Quality check + session ───────────────────────────────────
    print(f"\n  Running quality check on {len(all_extracted)} extracted garment(s)…")
    manager = _ImportSessionManager()
    session = manager.create_session(all_extracted)

    ready_count        = sum(1 for r in session.garment_records.values() if r.report.status.value == "ready")
    needs_review_count = sum(1 for r in session.garment_records.values() if r.report.status.value == "needs_review")
    failed_count       = sum(1 for r in session.garment_records.values() if r.report.status.value == "failed")

    _ok(
        f"  Quality check: {ready_count} ready  |  "
        f"{needs_review_count} needs-review  |  "
        f"{failed_count} failed"
    )
    if needs_review_count or failed_count:
        print(
            f"  ℹ️   All {needs_review_count + failed_count} flagged garment(s) will still be imported "
            f"— warnings are informational only."
        )

    # Print per-garment warnings (one line per unique code, status-labelled)
    for garment_id, record in session.garment_records.items():
        status_val = record.report.status.value   # "ready" | "needs_review" | "failed"
        if status_val == "ready":
            continue  # no warnings to show for ready garments
        status_tag = "FAILED" if status_val == "failed" else "NEEDS REVIEW"
        seen_codes: set = set()
        for w in record.report.warnings:
            if w.code not in seen_codes:
                seen_codes.add(w.code)
                _warn(
                    f"  [{status_tag}] [{record.report.garment.label}] "
                    f"{w.user_message} → {w.suggestion}"
                )

    # Accept EVERY garment regardless of quality status —
    # quality issues are warnings, not blockers in non-interactive mode.
    for garment_id, record in list(session.garment_records.items()):
        if record.decision.value == "pending":
            try:
                manager.accept_garment(session.session_id, garment_id)
            except Exception:
                pass  # already decided (e.g. auto-accepted READY)

    # ── 5. Finalise and convert to core Garment objects ──────────────────
    accepted_records = manager.finalise(session.session_id)
    garments: List[Garment] = []

    if accepted_records:
        _ok(f"\n  {len(accepted_records)} garment(s) imported (quality warnings are non-blocking)")
        for rec in accepted_records:
            # Get the original image array for color/pattern extraction
            src_path_str = str(rec.report.garment.source_path) if rec.report.garment.source_path else None
            img_array = image_array_map.get(src_path_str, None) if src_path_str else None
            g = _extracted_to_garment(rec.report.garment, image_array=img_array)
            garments.append(g)
            status_val = rec.report.status.value
            badge = (
                " ⚠️  FAILED"      if status_val == "failed"
                else " ⚠️  REVIEW" if status_val == "needs_review"
                else ""
            )
            col = g.attributes.color.primary if g.attributes.color else "?"
            sub = g.attributes.subcategory or g.attributes.category.value
            cat = g.attributes.category.value
            print(f"    • {cat:<12} {sub:<30} {col}{badge}")

        # ── Round-robin category redistribution (simple-mode fallback) ────
        # When GroundingDINO/SAM are unavailable, all garments come out as
        # "unknown" and the filename heuristic couldn't help either.
        # To avoid 0 outfits, distribute the wardrobe across a realistic
        # category mix (tops → bottoms → shoes → outerwear → accessories).
        all_tops = all(g.attributes.category == GarmentCategory.TOP for g in garments)
        if all_tops and len(garments) >= 2:
            _info(
                "  ℹ️  All garments defaulted to 'top' (simple mode, no filename hints).\n"
                "     Redistributing across categories so outfit generation can proceed."
            )
            _DIST_CYCLE: List[Tuple[GarmentCategory, str]] = [
                (GarmentCategory.TOP,       "shirt"),
                (GarmentCategory.BOTTOM,    "pants"),
                (GarmentCategory.TOP,       "shirt"),
                (GarmentCategory.SHOES,     "shoes"),
                (GarmentCategory.BOTTOM,    "pants"),
                (GarmentCategory.OUTERWEAR, "jacket"),
                (GarmentCategory.TOP,       "shirt"),
                (GarmentCategory.ACCESSORY, "accessory"),
            ]
            for i, g in enumerate(garments):
                cat_enum, sub_label = _DIST_CYCLE[i % len(_DIST_CYCLE)]
                g.attributes.category = cat_enum
                g.attributes.subcategory = sub_label
            # Print updated list
            print()
            for g in garments:
                col = g.attributes.color.primary if g.attributes.color else "?"
                sub = g.attributes.subcategory or g.attributes.category.value
                cat = g.attributes.category.value
                print(f"    → {cat:<12} {sub:<30} {col}  (redistributed)")

    else:
        _warn("No garments imported — Layer-0 extraction produced no results.")

    return garments


# ===========================================================================
# User profile helpers (ported from test_full_pipeline.py)
# ===========================================================================


def extract_user_profile(
    photo_path: Path,
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    verbose: bool = True,
) -> Optional[StyleProfile]:
    """
    Extract user style profile from a photo.

    Args:
        photo_path: Path to user photo (selfie)
        height_cm: Optional height in cm
        weight_kg: Optional weight in kg
        verbose: Whether to print extraction progress

    Returns:
        StyleProfile or None if extraction failed
    """
    if verbose:
        print(f"\n👤 Extracting user profile from: {photo_path}")
        print("=" * 60)

    try:
        config = PipelineConfig(
            enable_body_analysis=True,
            enable_face_analysis=True,
            enable_color_analysis=True,
            enable_hair_analysis=True,
            enable_contrast_analysis=True,
            height_cm=height_cm,
            weight_kg=weight_kg,
        )

        if verbose:
            print("   🔧 Analyzing: body, face, color, hair, contrast")
            if height_cm:
                print(f"   📏 Height: {height_cm} cm")
            if weight_kg:
                print(f"   ⚖️  Weight: {weight_kg} kg")
            print()

        pipeline = StyleProfilePipeline(config)
        try:
            result = pipeline.analyze(
                str(photo_path),
                height_cm=height_cm,
                weight_kg=weight_kg,
            )
        finally:
            pipeline.close()  # Explicitly close MediaPipe landmarkers before GC

        profile = result.profile

        if verbose:
            print("   ✅ Profile extracted successfully!")
            print()

            if profile.body_metrics:
                bm = profile.body_metrics
                print("   👤 Body Analysis:")
                if bm.body_shape:
                    print(f"      Shape: {bm.body_shape.value}")
                if bm.shoulder_hip_ratio:
                    print(f"      Shoulder/Hip Ratio: {bm.shoulder_hip_ratio:.2f}")
                if bm.frame_size:
                    print(f"      Frame Size: {bm.frame_size}")
                if bm.estimated_top_size:
                    print(f"      Est. Top Size: {bm.estimated_top_size}")
                if bm.estimated_bottom_size:
                    print(f"      Est. Bottom Size: {bm.estimated_bottom_size}")

            if profile.skin_analysis:
                sa = profile.skin_analysis
                print("   🎨 Skin Analysis:")
                if sa.skin_tone:
                    print(f"      Skin Tone: {sa.skin_tone.value}")
                if sa.undertone:
                    print(f"      Undertone: {sa.undertone.value}")

            if profile.hair_analysis:
                ha = profile.hair_analysis
                print("   💇 Hair Analysis:")
                if ha.hair_color:
                    print(f"      Hair Color: {ha.hair_color.value}")

            if profile.contrast_level:
                print(f"   ⚡ Contrast Level: {profile.contrast_level.value}")

            print()

        return profile

    except Exception as e:
        logger.warning(f"Could not extract user profile: {e}")
        if verbose:
            print(f"   ⚠️  Warning: Could not extract profile: {e}")
        return None


def display_context_recommendations(
    style_profile: StyleProfile,
    color_advisor: ColorHarmonyAdvisor,
    proportion_harmonizer: ProportionHarmonizer,
) -> None:
    """Display context-aware recommendations based on user profile."""
    print("\n💡 Context-Aware Recommendations")
    print("=" * 60)

    # Color recommendations
    if style_profile.skin_analysis:
        print("\n🎨 Color Recommendations (based on your coloring):")

        skin = style_profile.skin_analysis
        hair = style_profile.hair_analysis

        color_profile = color_advisor.create_color_profile(
            skin_tone=str(skin.skin_tone.value) if skin.skin_tone else "MEDIUM",
            undertone=str(skin.undertone.value) if skin.undertone else "NEUTRAL",
            hair_color=str(hair.hair_color.value) if hair and hair.hair_color else "BROWN",
            contrast_level=str(style_profile.contrast_level.value) if style_profile.contrast_level else "MEDIUM",
        )

        recs = color_advisor.get_color_recommendations(color_profile)

        print(f"   Color Season: {color_profile.season}")
        print(f"   Best Colors: {', '.join(recs.best_colors[:5])}")
        print(f"   Neutral Colors: {', '.join(recs.neutral_colors[:4])}")
        if recs.colors_to_avoid:
            print(f"   Colors to Avoid: {', '.join(recs.colors_to_avoid[:4])}")

        if recs.tips:
            print("\n   💡 Color Tips:")
            for tip in recs.tips[:2]:
                print(f"      • {tip}")

    # Proportion recommendations
    if style_profile.body_metrics:
        bm = style_profile.body_metrics
        print("\n📐 Proportion Recommendations (based on your body):")

        analysis = proportion_harmonizer.analyze_proportions(
            torso_ratio=getattr(bm, "torso_leg_ratio", None),
            leg_ratio=None,
            frame_size=getattr(bm, "frame_size", None),
            shoulder_hip_ratio=getattr(bm, "shoulder_hip_ratio", None),
        )

        print(f"   Torso Proportion: {analysis.torso_proportion}")
        print(f"   Leg Proportion: {analysis.leg_proportion}")
        print(f"   Frame Size: {analysis.frame_size}")

        if analysis.recommended_silhouettes:
            print(f"   Recommended Silhouettes: {', '.join(analysis.recommended_silhouettes[:4])}")
        if analysis.avoid_silhouettes:
            print(f"   Silhouettes to Avoid: {', '.join(analysis.avoid_silhouettes[:3])}")

        tips = proportion_harmonizer.get_proportion_tips(
            analysis.torso_proportion,
            analysis.leg_proportion,
        )
        if tips:
            print("\n   💡 Styling Tips:")
            for tip in tips[:2]:
                print(f"      • {tip}")

    # Size recommendations
    if style_profile.body_metrics:
        bm = style_profile.body_metrics
        if bm.estimated_top_size or bm.estimated_bottom_size:
            print("\n👕 Size Recommendations:")
            if bm.estimated_top_size:
                print(f"   Estimated Top Size: {bm.estimated_top_size}")
            if bm.estimated_bottom_size:
                print(f"   Estimated Bottom Size: {bm.estimated_bottom_size}")

    print()


async def apply_context_scoring(
    result: Dict[str, Any],
    style_profile: StyleProfile,
    body_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply context-aware scoring to outfit results using user profile.

    Args:
        result: Original pipeline result with best_outfit
        style_profile: User's style profile
        body_type: Optional override for body type

    Returns:
        Enhanced result with context scores
    """
    print("\n🎯 Applying Context-Aware Scoring")
    print("=" * 60)

    fit_predictor = FitPredictor()
    proportion_harmonizer = ProportionHarmonizer()
    color_advisor = ColorHarmonyAdvisor()

    best_outfit = result.get("best_outfit", {})
    garments_data = best_outfit.get("garments", [])

    if not garments_data:
        print("   ⚠️  No garments in best outfit to score")
        return result

    from unittest.mock import MagicMock

    garments = []
    for g_data in garments_data:
        garment = MagicMock()
        garment.id = g_data.get("id", "unknown")
        garment.category = g_data.get("category", "top")
        garment.color = g_data.get("color", "unknown")
        garment.attributes = {
            "size": g_data.get("size"),
            "fit": g_data.get("fit"),
            "color": g_data.get("color"),
            "waist": g_data.get("waist_type"),
            "pattern": g_data.get("pattern"),
            "weight": g_data.get("weight"),
        }
        garments.append(garment)

    context_scores: Dict[str, Any] = {}

    # 1. Fit prediction
    if style_profile.body_metrics:
        bm = style_profile.body_metrics
        fit_result = fit_predictor.predict_outfit_fit(
            garments=garments,
            user_top_size=getattr(bm, "estimated_top_size", None),
            user_bottom_size=getattr(bm, "estimated_bottom_size", None),
            user_bmi_category=getattr(bm, "bmi_category", None),
        )
        context_scores["fit"] = {
            "score": fit_result.overall_score,
            "fit_type": fit_result.average_fit_type,
            "notes": fit_result.fit_notes,
        }
        print(f"   👕 Fit Score: {fit_result.overall_score:.2f} ({fit_result.average_fit_type})")

    # 2. Proportion harmony
    if style_profile.body_metrics:
        bm = style_profile.body_metrics
        analysis = proportion_harmonizer.analyze_proportions(
            torso_ratio=getattr(bm, "torso_leg_ratio", None),
            leg_ratio=None,
            frame_size=getattr(bm, "frame_size", None),
            shoulder_hip_ratio=getattr(bm, "shoulder_hip_ratio", None),
        )
        prop_score = proportion_harmonizer.score_outfit_harmony(garments, analysis)
        context_scores["proportions"] = {
            "score": prop_score.score,
            "harmony_level": prop_score.harmony_level,
            "positive_factors": prop_score.positive_factors,
            "negative_factors": prop_score.negative_factors,
            "tips": prop_score.styling_tips,
        }
        print(f"   📐 Proportion Score: {prop_score.score:.2f} ({prop_score.harmony_level})")

    # 3. Color harmony
    if style_profile.skin_analysis:
        skin = style_profile.skin_analysis
        hair = style_profile.hair_analysis

        color_profile = color_advisor.create_color_profile(
            skin_tone=str(skin.skin_tone.value) if skin.skin_tone else "MEDIUM",
            undertone=str(skin.undertone.value) if skin.undertone else "NEUTRAL",
            hair_color=str(hair.hair_color.value) if hair and hair.hair_color else "BROWN",
            contrast_level=str(style_profile.contrast_level.value) if style_profile.contrast_level else "MEDIUM",
        )

        color_score = color_advisor.score_outfit_colors(garments, color_profile)
        context_scores["color_harmony"] = {
            "score": color_score.score,
            "harmony_level": color_score.harmony_level,
            "matching_colors": color_score.matching_colors,
            "clashing_colors": color_score.clashing_colors,
            "notes": color_score.notes,
        }
        print(f"   🎨 Color Harmony Score: {color_score.score:.2f} ({color_score.harmony_level})")

    # Calculate overall context score
    if context_scores:
        scores = [s["score"] for s in context_scores.values() if "score" in s]
        overall_context_score = sum(scores) / len(scores) if scores else 0.7
        context_scores["overall_context_score"] = overall_context_score
        print(f"\n   ⭐ Overall Context Score: {overall_context_score:.2f}")

        original_score = best_outfit.get("overall_score", 0.7)
        blended_score = 0.6 * original_score + 0.4 * overall_context_score
        context_scores["blended_score"] = blended_score
        print(f"   🔄 Blended Score (60% style + 40% context): {blended_score:.2f}")

    result["context_analysis"] = context_scores
    result["best_outfit"]["context_adjusted_score"] = context_scores.get(
        "blended_score", best_outfit.get("overall_score", 0.7)
    )

    print()

    return result


async def run_hybrid_comparison(
    style_profile: Optional[StyleProfile],
    wardrobe_garments: Optional[List[Garment]] = None,
    top_k: int = 5,
) -> None:
    """
    Run HybridOutfitRecommender and print a side-by-side
    Style vs Context vs Combined comparison table.

    If ``wardrobe_garments`` is None, falls back to a mock demo wardrobe.
    If ``style_profile`` is provided, personalises the recommender.
    """
    print("\n" + "=" * 70)
    print("  🤖 Hybrid Comparison — StyleIntelligenceModel × ContextEngine")
    print("=" * 70)

    if not wardrobe_garments:
        print("\n⚠️  No wardrobe garments available — using mock wardrobe")
        from src.core.models import GarmentAttributes, GarmentCategory, ColorInfo, PatternInfo
        from uuid import uuid4

        def _mk(cat: str, sub: str, col: str) -> Garment:
            return Garment(
                id=str(uuid4()),
                attributes=GarmentAttributes(
                    category=GarmentCategory(cat),
                    subcategory=sub,
                    color=ColorInfo(primary=col, hex_codes=[]),
                    pattern=PatternInfo(type="solid"),
                    formality_level="casual",
                    season_suitable=["spring", "summer", "fall", "winter"],
                    fit="regular",
                ),
            )

        wardrobe_garments = [
            _mk("top", "white t-shirt", "white"),
            _mk("top", "navy shirt", "navy"),
            _mk("top", "gray sweater", "gray"),
            _mk("bottom", "blue jeans", "blue"),
            _mk("bottom", "black pants", "black"),
            _mk("bottom", "beige chinos", "beige"),
            _mk("shoes", "white sneakers", "white"),
            _mk("shoes", "brown boots", "brown"),
            _mk("outerwear", "black jacket", "black"),
        ]

    color_season: Optional[ColorSeason] = None
    body_shape_vol: Optional[VolumeBodyShape] = None

    if style_profile:
        if style_profile.skin_analysis and style_profile.skin_analysis.undertone:
            undertone = style_profile.skin_analysis.undertone.value.lower()
            season_map = {
                "warm": ColorSeason.AUTUMN,
                "cool": ColorSeason.WINTER,
                "neutral": ColorSeason.SPRING,
            }
            color_season = season_map.get(undertone, ColorSeason.SPRING)

        if style_profile.body_metrics and style_profile.body_metrics.body_shape:
            shape_val = style_profile.body_metrics.body_shape.value.upper().replace(" ", "_")
            try:
                body_shape_vol = VolumeBodyShape[shape_val]
            except KeyError:
                pass

        print(f"\n🎨 Colour season: {color_season.value if color_season else 'not set'}")
        print(f"👤 Body shape:    {body_shape_vol.value if body_shape_vol else 'not set'}")
    else:
        print("\n💡 No user profile — running with default (unpersonalised) weights")

    recommender = HybridOutfitRecommender(style_weight=0.40, context_weight=0.60)
    if color_season or body_shape_vol:
        recommender.set_user_profile(color_season=color_season, body_shape=body_shape_vol)

    user_context = UserContext(
        occasion=Occasion.CASUAL,
        formality_preference=FormalityLevel.CASUAL,
    )

    print(f"\n⏳ Scoring combinations (top {top_k}) …")
    import time as _time
    start = _time.time()
    ranked: List[RankedOutfit] = await recommender.recommend(
        wardrobe_garments, user_context, top_k=top_k
    )
    elapsed = (_time.time() - start) * 1000
    print(f"   Done in {elapsed:.0f} ms — {len(ranked)} outfits ranked\n")

    print(f"{'Rank':<5} {'Outfit':<35} {'Style':>7} {'Context':>9} {'Combined':>10} {'Grade':>6}")
    print("─" * 78)
    for i, ro in enumerate(ranked, 1):
        items = " + ".join(
            (g.attributes.subcategory or g.attributes.category.value)[:12]
            for g in ro.garments[:3]
        )
        if len(ro.garments) > 3:
            items += f" (+{len(ro.garments)-3})"
        marker = " ✅" if i == 1 else ""
        print(
            f"{i:<5} {items:<35} {ro.score.style_score:>6.0%}  "
            f"{ro.score.context_score:>8.0%}  {ro.score.combined_score:>9.0%}  "
            f"{ro.score.grade:>5}{marker}"
        )

    if ranked:
        top = ranked[0]
        s = top.score
        print("\n🥇 Best outfit detail:")
        items_full = " + ".join(
            g.attributes.subcategory or g.attributes.category.value
            for g in top.garments
        )
        print(f"   {items_full}")
        print(f"   Combined : {s.combined_score:.1%}  [{s.grade}]")
        print(f"   Style    : {s.style_score:.1%}   (weight {s.style_weight:.0%})")
        print(f"   Context  : {s.context_score:.1%}   (weight {s.context_weight:.0%})")
        if s.style_breakdown:
            print("   Style breakdown:")
            for k, v in s.style_breakdown.items():
                if isinstance(v, (int, float)):
                    print(f"      • {k}: {v:.0%}")
        if s.strengths:
            print(f"   ✅ {', '.join(s.strengths[:2])}")
        if s.improvements:
            print(f"   💡 {', '.join(s.improvements[:2])}")
    print()


# ===========================================================================
# Phase 7 — Capsule Wardrobe Analysis (F1-F5)
# ===========================================================================


async def phase_capsule_wardrobe(
    garments: List[Garment],
    candidates: List[OutfitCandidate],
    client: Optional["Neo4jClient"],
    *,
    user_id: str = USER_ID,
    user_season: Optional[str] = None,
    body_shape: Optional[str] = None,
    run_f1: bool = True,
    run_f2: bool = True,
    run_f3: bool = True,
    run_f4: bool = True,
    run_f5: bool = True,
    with_llm: bool = False,
) -> None:
    """
    Full capsule wardrobe analysis — Phases F1 through F5.

    Parameters
    ----------
    garments:       Wardrobe garments (from Phase 1).
    candidates:     OutfitCandidates from Phase 3 — reused by F4, zero extra cost.
    client:         Optional Neo4j client — centrality map fetched once if available.
    user_id:        Used by evolution tracker for snapshot persistence.
    user_season:    Optional colour season string (autumn/spring/summer/winter).
    body_shape:     Optional body shape string.
    run_f1-f5:      Per-feature gates (all True by default under --capsule-all).
    with_llm:       If True, add LLM narrations via CapsuleExplainer.
    """
    from src.layer2_style.capsule.wardrobe_capsule_analyzer import WardrobeCapsuleAnalyzer
    from src.layer2_style.capsule.missing_pieces_recommender import MissingPiecesRecommender
    from src.layer2_style.capsule.replacement_planner import ReplacementPlanner
    from src.layer2_style.capsule.capsule_outfit_generator import CapsuleOutfitGenerator
    from src.layer2_style.capsule.capsule_evolution_tracker import CapsuleEvolutionTracker

    _banner("Phase 7 — Capsule Wardrobe Analysis")

    if not garments:
        _warn("No garments — capsule analysis skipped.")
        return

    # ── Fetch centrality map once (if Neo4j available) ──────────────────
    centrality_map: Dict[str, float] = {}
    if client is not None:
        try:
            job = CentralityRefreshJob(client=client)
            centrality_map = await job.get_centrality_map(user_id=user_id)
            _ok(f"Centrality map loaded: {len(centrality_map)} garments")
        except Exception as exc:
            _warn(f"Could not load centrality map: {exc}")

    # Build outfit_count_map from Phase 3 candidates
    from collections import Counter as _Counter
    outfit_count_map: Dict[str, int] = {}
    for c in candidates:
        for g in c.garments:
            outfit_count_map[g.id] = outfit_count_map.get(g.id, 0) + 1

    # ── F1 — Capsule score ───────────────────────────────────────────────
    analysis = None
    if run_f1:
        _sub("F1 — Capsule Score & Garment Roles")
        t0 = time.perf_counter()
        analyzer = WardrobeCapsuleAnalyzer()
        analysis = analyzer.analyze(
            garments,
            centrality_map=centrality_map or None,
            outfit_count_map=outfit_count_map or None,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _ok(f"Capsule score: {analysis.cohesion_score:.0f}/100  (in {elapsed_ms:.0f} ms)")
        print(f"    Profile          : {analysis.capsule_profile}")
        print(f"    Key pieces       : {len(analysis.key_pieces)}")
        print(f"    Orphan pieces    : {len(analysis.orphan_pieces)}")
        print(f"    Redundant pairs  : {len(analysis.redundant_pairs)}")
        print(f"    Color cohesion   : {analysis.color_cohesion_score:.0%}")
        print(f"    Projected score  : {analysis.projected_score_after_cleanup:.0f}/100")
        if analysis.recommendation:
            print(f"\n  Diagnosis: {analysis.recommendation}")

    # ── F2 — Missing pieces ──────────────────────────────────────────────
    if run_f2 and analysis is not None:
        _sub("F2 — Missing Piece Recommendations")
        recommender = MissingPiecesRecommender(top_n=5)
        missing = recommender.recommend(analysis, user_season=user_season, body_shape=body_shape)

        if with_llm:
            from src.layer4_llm.capsule_explainer import CapsuleExplainer
            explainer = CapsuleExplainer()
            missing = await explainer.narrate_missing_pieces(missing)

        print(f"\n  Current cohesion : {missing.current_cohesion:.0f}/100")
        print(f"  Projected        : {missing.projected_cohesion:.0f}/100")
        print(f"  {missing.summary}\n")
        for rec in missing.recommendations:
            narration = f" — {rec.llm_narration}" if rec.llm_narration else ""
            colors_str = ", ".join(rec.suggested_colors[:3])
            print(f"  [{rec.priority}] {rec.description} ({rec.category}) → +{rec.impact_outfits} outfits")
            print(f"       Colors: {colors_str}{narration}")
            if rec.profile_note:
                print(f"       💡 {rec.profile_note}")

    # ── F3 — Replacement plan ────────────────────────────────────────────
    if run_f3 and analysis is not None and analysis.redundant_pairs:
        _sub("F3 — Replacement Plan")
        planner = ReplacementPlanner()
        plan = planner.plan(analysis.redundant_pairs, analysis.garment_scores)

        if with_llm:
            from src.layer4_llm.capsule_explainer import CapsuleExplainer
            explainer = CapsuleExplainer()
            plan = await explainer.narrate_replacement_plan(plan)

        print(f"\n  {plan.summary}")
        if plan.verdicts:
            print(f"\n  {'Keep':<20} {'Remove':<20} {'Gain':>6} {'Conf':>6} {'Timing'}")
            print(f"  {'-'*20} {'-'*20} {'-'*6} {'-'*6} {'-'*35}")
            for v in plan.verdicts:
                narration = f"\n       → {v.llm_narration}" if v.llm_narration else ""
                print(
                    f"  {v.garment_keep_id[:19]:<20} {v.garment_remove_id[:19]:<20} "
                    f"{v.versatility_gain:>5.2f}  {v.confidence:>5.0%}  {v.transition_timing}{narration}"
                )
    elif run_f3 and analysis is not None:
        _sub("F3 — Replacement Plan")
        _info("No redundant pairs detected — nothing to replace.")

    # ── F4 — Capsule outfits ─────────────────────────────────────────────
    if run_f4 and analysis is not None:
        _sub("F4 — Capsule Outfits (reusing Phase 3 candidates)")
        generator = CapsuleOutfitGenerator()
        capsule_outfits = generator.generate(candidates, analysis, top_n=10)

        if with_llm:
            from src.layer4_llm.capsule_explainer import CapsuleExplainer
            explainer = CapsuleExplainer()
            capsule_outfits = await explainer.narrate_capsule_outfits(capsule_outfits, top_n=5)

        print(f"\n  Basic     : {capsule_outfits.basic_count}")
        print(f"  Semi      : {capsule_outfits.semi_creative_count}")
        print(f"  Creative  : {capsule_outfits.creative_count}")

        if capsule_outfits.outfits:
            print(f"\n  {'Rank':<5} {'Tier':<14} {'Score':>7} {'Key%':>6} {'Occasions'}")
            print(f"  {'-'*5} {'-'*14} {'-'*7} {'-'*6} {'-'*28}")
            for co in capsule_outfits.outfits[:5]:
                occasions_str = ", ".join(co.occasions[:2])
                narration = f"\n       {co.llm_narration}" if co.llm_narration else ""
                print(
                    f"  {co.rank:<5} {co.tier:<14} {co.capsule_score:>6.0f}/100 "
                    f"{co.pct_key_pieces:>5.0%}  {occasions_str}{narration}"
                )

    # ── F5 — Evolution ───────────────────────────────────────────────────
    if run_f5 and analysis is not None:
        _sub("F5 — Capsule Evolution Tracker")
        tracker = CapsuleEvolutionTracker(user_id=user_id)
        wh = WardrobeCapsuleAnalyzer.wardrobe_hash(garments)
        evolution = tracker.record(analysis, wh, action="pipeline_run")

        if with_llm:
            from src.layer4_llm.capsule_explainer import CapsuleExplainer
            explainer = CapsuleExplainer()
            evolution = await explainer.narrate_evolution(evolution)

        print(f"\n  Snapshots recorded   : {len(evolution.snapshots)}")
        print(f"  Latest cohesion      : {evolution.latest_cohesion:.0f}/100")
        print(f"  Score change         : {evolution.delta_score:+.1f}")
        print(f"  New outfits unlocked : {evolution.delta_outfits:+d}")
        print(f"  Trend                : {evolution.trend_direction}")
        if evolution.predicted_weeks_to_90 is not None:
            print(f"  Weeks to 90/100      : ~{evolution.predicted_weeks_to_90}")
        else:
            print("  Weeks to 90/100      : (need more history)")


# ===========================================================================
# main
# ===========================================================================


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neo4j-accelerated full pipeline test (mirrors test_full_pipeline.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quickest demo (synthetic data, no API key needed)
  python scripts/test_neo4j_pipeline.py --demo

  # Real wardrobe images
  python scripts/test_neo4j_pipeline.py --wardrobe data/sample_wardrobe/

  # Single outfit image
  python scripts/test_neo4j_pipeline.py --outfit-image ./outfit.jpg

  # Multiple individual garment images
  python scripts/test_neo4j_pipeline.py --images top.jpg pants.jpg shoes.jpg

  # With user photo for context-aware scoring
  python scripts/test_neo4j_pipeline.py --wardrobe data/sample_wardrobe/ \\
      --user-photo data/people/image.png --height 170 --weight 65

  # Reuse an already-built Neo4j graph (faster re-runs)
  python scripts/test_neo4j_pipeline.py --demo --skip-graph-build

  # Also run smart-removal analysis
  python scripts/test_neo4j_pipeline.py --demo --removal-analysis

  # Save JSON report
  python scripts/test_neo4j_pipeline.py --demo --output report_neo4j.json

  # Save visualization image
  python scripts/test_neo4j_pipeline.py --demo --visualize best_outfit.png

  # Side-by-side hybrid comparison (Style vs Context vs Combined)
  python scripts/test_neo4j_pipeline.py --demo --hybrid-compare
  python scripts/test_neo4j_pipeline.py --wardrobe data/sample_wardrobe/ \\
      --user-photo data/people/image.png --hybrid-compare

  # Full run + save all results
  python scripts/test_neo4j_pipeline.py --wardrobe data/sample_wardrobe/ --save-results
        """,
    )

    # ── Input (same as test_full_pipeline.py) ───────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic demo garments (no Vision API needed)",
    )
    input_group.add_argument(
        "--wardrobe",
        type=Path,
        help="Path to a folder of garment images",
    )
    input_group.add_argument(
        "--outfit-image",
        type=Path,
        help="Path to a single outfit image to analyze",
    )
    input_group.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="List of individual garment image paths",
    )

    # ── User profile (same as test_full_pipeline.py) ─────────────────────
    pgroup = parser.add_argument_group("User profile")
    pgroup.add_argument("--user-photo", type=Path, help="Selfie for style-profile extraction")
    pgroup.add_argument("--height", type=float, help="Height in cm (used with --user-photo)")
    pgroup.add_argument("--weight", type=float, help="Weight in kg (used with --user-photo)")
    pgroup.add_argument(
        "--body-type",
        type=str,
        choices=["rectangle", "hourglass", "pear", "apple", "inverted_triangle", "athletic"],
        help="Body type (if known, instead of auto-detection)",
    )
    pgroup.add_argument(
        "--user-season",
        type=str,
        choices=["spring", "summer", "autumn", "winter"],
        help="Color season for personalised improvement suggestions",
    )
    pgroup.add_argument(
        "--body-shape",
        type=str,
        choices=["rectangle", "triangle", "inverted_triangle", "hourglass", "oval", "athletic"],
        help="Body shape for personalised improvement suggestions",
    )

    # ── Pipeline options ─────────────────────────────────────────────────
    parser.add_argument(
        "--profile",
        default="default",
        choices=["default", "minimalist", "creative", "business", "casual"],
        help="Scoring profile (default: default)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of outfit candidates to return (default: 5)",
    )
    parser.add_argument(
        "--occasion",
        choices=["casual", "business", "formal", "sport", "date", "travel"],
        default="casual",
        help="Target occasion for outfit filtering (default: casual)",
    )
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help=(
            "Pass wardrobe images through the Layer-0 extraction pipeline + "
            "QualityChecker before entering the main pipeline. "
            "Requires optional Layer-0 dependencies; falls back to Vision AI "
            "loader automatically if they are not available."
        ),
    )

    # ── Neo4j options ────────────────────────────────────────────────────
    ngroup = parser.add_argument_group("Neo4j options")
    ngroup.add_argument(
        "--skip-graph-build",
        action="store_true",
        help="Skip Phase 2 graph build (use when graph already exists from a previous run)",
    )
    ngroup.add_argument(
        "--skip-centrality",
        action="store_true",
        help="Skip centrality computation during graph build (faster)",
    )
    ngroup.add_argument(
        "--no-neo4j",
        action="store_true",
        help="Force local-fallback mode (don't try to connect to Neo4j)",
    )

    # ── Extra phases ─────────────────────────────────────────────────────
    parser.add_argument(
        "--removal-analysis",
        action="store_true",
        help="Run Phase 4 smart-removal analysis after outfit generation",
    )
    parser.add_argument(
        "--removal-profile",
        choices=["default", "minimalist_eco", "fashion_addict", "budget_conscious", "work_professional", "data_driven"],
        default=None,
        help=(
            "Smart-removal profile to use (default: auto-mapped from --profile). "
            "Available: default, minimalist_eco, fashion_addict, budget_conscious, work_professional, data_driven"
        ),
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip Phase 5 sync & consistency check",
    )

    # ── Output ───────────────────────────────────────────────────────────
    parser.add_argument("--output", type=Path, help="Save JSON report to this file")
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to tests/output/pipeline_runs/",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("tests/output/pipeline_runs"),
        help="Directory for saved results (default: tests/output/pipeline_runs)",
    )
    parser.add_argument(
        "--visualize",
        type=Path,
        help="Generate and save outfit visualization image to specified path (e.g., best_outfit.png)",
    )
    parser.add_argument(
        "--hybrid-compare",
        action="store_true",
        help=(
            "After the normal pipeline run, show a side-by-side "
            "Style vs Context vs Combined comparison table using HybridOutfitRecommender"
        ),
    )

    # ── Capsule wardrobe analysis (Phase 7) ──────────────────────────────
    cgroup = parser.add_argument_group("Capsule wardrobe (Phase 7)")
    cgroup.add_argument(
        "--capsule-all",
        action="store_true",
        help="Run all 5 capsule features (F1-F5) after the main pipeline",
    )
    cgroup.add_argument("--capsule-analysis",  action="store_true", help="F1 — capsule score & garment roles")
    cgroup.add_argument("--missing-pieces",    action="store_true", help="F2 — missing piece recommendations")
    cgroup.add_argument("--replacement-plan",  action="store_true", help="F3 — replacement plan for redundant pairs")
    cgroup.add_argument("--capsule-outfits",   action="store_true", help="F4 — capsule-scored outfits (reuses Phase 3)")
    cgroup.add_argument("--capsule-evolution", action="store_true", help="F5 — evolution tracker & trend prediction")
    cgroup.add_argument(
        "--capsule-llm",
        action="store_true",
        help="Add LLM narrations to capsule results (requires GOOGLE_API_KEY)",
    )

    args = parser.parse_args()

    # ── Validate paths ───────────────────────────────────────────────────
    if args.wardrobe and not args.wardrobe.exists():
        print(f"❌ Wardrobe folder not found: {args.wardrobe}")
        sys.exit(1)
    if args.user_photo and not args.user_photo.exists():
        print(f"❌ User photo not found: {args.user_photo}")
        sys.exit(1)

    t_total = time.perf_counter()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        Neo4j Full Pipeline Test                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Results storage ──────────────────────────────────────────────────
    storage: Optional[ResultsStorage] = None
    if args.save_results:
        storage = ResultsStorage(args.results_dir)

    # ── Phase 0 : Neo4j connection ───────────────────────────────────────
    _banner("Phase 0 — Neo4j Connection")
    client: Optional[Neo4jClient] = None
    if not args.no_neo4j:
        client = await check_neo4j_connection()
    else:
        _info("--no-neo4j flag set, running in local fallback mode.")

    # ── Phase 1 : Wardrobe loading ───────────────────────────────────────
    garments: List[Garment] = []
    if args.demo:
        _banner("Phase 1 — Demo Wardrobe")
        garments = build_demo_wardrobe()
        _ok(f"{len(garments)} synthetic garments created")
        for g in garments:
            cat = g.attributes.category.value
            sub = g.attributes.subcategory or ""
            col = g.attributes.color.primary if g.attributes.color else "?"
            print(f"    • {cat:<12} {sub:<20} {col}")
    elif args.wardrobe:
        if getattr(args, "quality_check", False):
            garments = await load_wardrobe_with_quality_check(args.wardrobe, args.profile)
        else:
            garments = await load_wardrobe_from_folder(args.wardrobe, args.profile)
    elif args.outfit_image:
        _banner("Phase 1 — Single Outfit Image")
        if not args.outfit_image.exists():
            print(f"❌ Image not found: {args.outfit_image}")
            sys.exit(1)
        builder = OutfitBuilder()
        garments = await builder.load_outfit_from_folder(args.outfit_image.parent)
        garments = [g for g in garments if g.image_path and Path(g.image_path).name == args.outfit_image.name]
        if not garments:
            garments = await builder.load_outfit_from_folder(args.outfit_image.parent)
    elif args.images:
        _banner("Phase 1 — Individual Garment Images")
        for img in args.images:
            if not img.exists():
                print(f"❌ Image not found: {img}")
                sys.exit(1)
        builder = OutfitBuilder()
        garments = []
        for img in args.images:
            loaded = await builder.load_outfit_from_folder(img.parent)
            garments.extend(loaded)

    if not garments:
        print("❌ No garments available — aborting.")
        sys.exit(1)

    # ── User profile (optional) ──────────────────────────────────────────
    style_profile: Optional[StyleProfile] = None
    if args.user_photo:
        _banner("Phase 1b — User Profile Extraction")
        style_profile = extract_user_profile(
            args.user_photo,
            height_cm=args.height,
            weight_kg=args.weight,
            verbose=True,
        )
        if style_profile:
            color_advisor = ColorHarmonyAdvisor()
            proportion_harmonizer = ProportionHarmonizer()
            display_context_recommendations(style_profile, color_advisor, proportion_harmonizer)
            if storage:
                profile_path = storage.run_dir / "user_profile.json"
                with open(profile_path, "w") as f:
                    json.dump(style_profile.to_dict(), f, indent=2, default=str)
                print(f"💾 User profile saved to: {profile_path}")

    # ── Resolve optional Layer 4 profile args to enums ───────────────────
    _user_season_enum: Optional[ColorSeason] = None
    _body_shape_enum = None
    if hasattr(args, "user_season") and args.user_season:
        try:
            _user_season_enum = ColorSeason(args.user_season)
        except Exception:
            pass
    if hasattr(args, "body_shape") and args.body_shape:
        try:
            from src.layer3_context.user_profile.models import BodyShape as ProfileBodyShape
            _body_shape_enum = ProfileBodyShape(args.body_shape)
        except Exception:
            pass

    # ── Build UserContext ────────────────────────────────────────────────
    _occasion_map = {
        "casual":   Occasion.CASUAL,
        "business": Occasion.WORK,
        "formal":   Occasion.FORMAL,
        "sport":    Occasion.SPORT,
        "date":     Occasion.DATE,
        "travel":   Occasion.TRAVEL,
    }
    context = UserContext(
        occasion=_occasion_map.get(args.occasion, Occasion.CASUAL),
    )

    # ── Phase 2 : Graph build ────────────────────────────────────────────
    if client is not None:
        await phase_graph_build(
            garments,
            client,
            skip=args.skip_graph_build,
            skip_centrality=args.skip_centrality,
        )
    else:
        _banner("Phase 2 — Neo4j Graph Build")
        _info("Skipped (no Neo4j connection).")

    # ── Phase 3 : Outfit generation ──────────────────────────────────────
    candidates = await phase_outfit_generation(
        garments,
        client,
        context=context,
        top_k=args.top_k,
        profile=args.profile,
    )

    # ── Phase 4 : Smart removal (optional) ──────────────────────────────
    if args.removal_analysis:
        # Smart-removal uses its own profile vocabulary — map scoring profile or
        # fall back to "default" when there is no direct match.
        _REMOVAL_PROFILE_MAP = {
            "minimalist": "minimalist_eco",
            "business":   "work_professional",
            "creative":   "fashion_addict",
            "casual":     "default",
            "default":    "default",
        }
        removal_profile = getattr(args, "removal_profile", None) or _REMOVAL_PROFILE_MAP.get(
            args.profile, "default"
        )
        await phase_smart_removal(garments, client, profile_name=removal_profile)

    # ── Phase 5 : Sync & consistency (optional) ──────────────────────────
    if not args.skip_sync and client is not None:
        await phase_sync_and_consistency(garments, client)
    elif client is None and not args.skip_sync:
        _info("Skipping Phase 5 (no Neo4j connection).")

    # ── Context scoring (optional — requires --user-photo) ───────────────
    report_best_outfit: Dict[str, Any] = {}
    if candidates:
        report_best_outfit = {
            "name": candidates[0].name,
            "overall_score": candidates[0].overall_score,
            "garments": [
                {
                    "id": g.id,
                    "category": g.attributes.category.value,
                    "subcategory": g.attributes.subcategory,
                    "color": g.attributes.color.primary if g.attributes.color else None,
                    "fit": getattr(g.attributes, "fit", None),
                    "pattern": g.attributes.pattern.type if g.attributes.pattern else None,
                    "formality_level": getattr(g.attributes, "formality_level", None),
                }
                for g in candidates[0].garments
            ],
        }

    if style_profile and report_best_outfit:
        context_result = {"best_outfit": report_best_outfit}
        body_type: Optional[str] = None
        if hasattr(args, "body_type") and args.body_type:
            body_type = args.body_type
        elif style_profile.body_metrics and style_profile.body_metrics.body_shape:
            body_type = style_profile.body_metrics.body_shape.value
        context_result = await apply_context_scoring(context_result, style_profile, body_type)
        report_best_outfit = context_result["best_outfit"]

    # ── Visualization (optional — requires --visualize) ───────────────────
    if args.visualize and candidates:
        print(f"\n🖼️  Generating outfit visualization …")
        try:
            # Use the QuietLuxuryVisualizer for editorial-style moodboard
            visualizer = QuietLuxuryVisualizer()
            viz = visualizer.create_visualization(
                garments=candidates[0].garments,
                segmented_images={},  # Pass empty dict as placeholder
                outfit_name=candidates[0].name,
                score=candidates[0].overall_score,
                season=args.user_season,
                occasion=args.occasion,
            )
            visualizer.save_visualization(viz, args.visualize)
            print(f"   ✅ Outfit visualization saved to: {args.visualize}")
        except Exception as e:
            logger.warning(f"Could not generate visualization: {e}")
            print(f"   ⚠️  Could not generate visualization: {e}")

    # ── Hybrid comparison (optional — requires --hybrid-compare) ─────────
    if args.hybrid_compare:
        wardrobe_for_hybrid: Optional[List[Garment]] = garments if garments else None
        await run_hybrid_comparison(
            style_profile=style_profile,
            wardrobe_garments=wardrobe_for_hybrid,
            top_k=args.top_k,
        )

    # ── Phase 6 : Layer 4 LLM Outfit Improvement (optional) ──────────────
    improvement_result: Optional[Dict[str, Any]] = None
    if candidates and (_user_season_enum or _body_shape_enum):
        # Convert ProfileBodyShape to VolumeBodyShape if needed
        body_shape_for_llm: Optional[VolumeBodyShape] = None
        if _body_shape_enum:
            try:
                # Map ProfileBodyShape to VolumeBodyShape
                shape_str = _body_shape_enum.value.upper().replace(" ", "_")
                body_shape_for_llm = VolumeBodyShape[shape_str]
            except (KeyError, ValueError):
                pass

        improvement_result = await phase_outfit_improvement(
            outfit_candidate=candidates[0],
            wardrobe=garments,
            user_season=_user_season_enum,
            body_shape=body_shape_for_llm,
        )

    # ── Phase 7 : Capsule wardrobe analysis (optional) ───────────────────
    _capsule_any = getattr(args, "capsule_all", False) or any([
        getattr(args, "capsule_analysis", False),
        getattr(args, "missing_pieces", False),
        getattr(args, "replacement_plan", False),
        getattr(args, "capsule_outfits", False),
        getattr(args, "capsule_evolution", False),
    ])
    if _capsule_any:
        _run_all = getattr(args, "capsule_all", False)
        await phase_capsule_wardrobe(
            garments=garments,
            candidates=candidates,
            client=client,
            user_id=USER_ID,
            user_season=getattr(args, "user_season", None),
            body_shape=getattr(args, "body_shape", None),
            run_f1=_run_all or getattr(args, "capsule_analysis", False),
            run_f2=_run_all or getattr(args, "missing_pieces", False),
            run_f3=_run_all or getattr(args, "replacement_plan", False),
            run_f4=_run_all or getattr(args, "capsule_outfits", False),
            run_f5=_run_all or getattr(args, "capsule_evolution", False),
            with_llm=getattr(args, "capsule_llm", False),
        )

    # ── Save report ──────────────────────────────────────────────────────
    report: Dict[str, Any] = {
        "neo4j_used": client is not None,
        "garments_count": len(garments),
        "outfits_generated": len(candidates),
        "best_outfit": report_best_outfit if report_best_outfit else None,
        "improvement_suggestions": improvement_result if improvement_result else None,
        "all_outfits": [
            {
                "rank": i + 1,
                "name": c.name,
                "score": c.overall_score,
                "score_breakdown": {
                    k: v
                    for k, v in (c.scorecard.scores if c.scorecard else {}).items()
                    if isinstance(v, (int, float))
                },
            }
            for i, c in enumerate(candidates)
        ],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        _ok(f"Report saved → {args.output}")

    if storage:
        from pipeline import graph_generator
        graphs_dir = storage.run_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        # Save garments
        storage.save_all_garments(garments)
        # Save outfit combinations
        if candidates:
            storage.save_combinations(candidates, args.profile)

        # ── Graph 1: overall score comparison (all candidates) ────────
        try:
            graph_generator.generate_score_graph(candidates, args.profile, graphs_dir)
        except Exception as _e:
            pass

        # ── Graph 2: criteria breakdown heatmap (top 10) ──────────────
        try:
            graph_generator.generate_criteria_breakdown_graph(
                candidates, args.profile, graphs_dir
            )
        except Exception as _e:
            pass

        # ── Graph 3: radar chart for the best outfit ──────────────────
        try:
            if candidates:
                graph_generator.generate_best_outfit_radar(
                    candidates[0], args.profile, graphs_dir
                )
        except Exception as _e:
            pass

        # ── Graph 4: full scoring dashboard (style + creativity + ctx) ─
        try:
            graph_generator.generate_scoring_dashboard(
                candidates, args.profile, graphs_dir
            )
        except Exception as _e:
            pass

        # ── Graph 5: wardrobe colour distribution ─────────────────────
        try:
            graph_generator.generate_color_distribution_graph(garments, graphs_dir)
        except Exception:
            pass

        # Save final report
        storage.save_final_report(
            report,
            candidates[0] if candidates else None,
            candidates,
        )
        _ok(f"Results saved → {storage.run_dir}")

    # ── Close client ─────────────────────────────────────────────────────
    if client is not None:
        await client.close()

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_total
    _banner("Done")
    print(f"  ⏱️   Total wall time : {elapsed_total:.2f}s")
    print(f"  🌐  Neo4j used       : {'Yes' if client else 'No (local fallback)'}")
    print(f"  👗  Garments loaded  : {len(garments)}")
    print(f"  🎯  Outfits ranked   : {len(candidates)}")
    if candidates:
        print(f"  🥇  Best score      : {candidates[0].overall_score:.1%}  ({candidates[0].name})")
    print()
    print("  ✅  Neo4j pipeline test complete!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
