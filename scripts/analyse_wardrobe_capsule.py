"""
Wardrobe Capsule Analyser
=========================
Analyses a wardrobe folder through the full capsule pipeline and stores
every result artefact in a timestamped output directory.

Pipeline:
  Layer 0  — Garment segmentation (SAM, optional)
  Layer 1  — Attribute extraction  (Vision AI / Gemini)
  Layer 2  — Style scoring + OutfitBuilder
  ─────────────────────────────────────────────────
  F1  WardrobeCapsuleAnalyzer   → cohesion score, roles, redundancy
  F2  MissingPiecesRecommender  → top-N missing pieces + impact
  F3  ReplacementPlanner        → which redundant to swap & when
  F4  CapsuleOutfitGenerator    → best capsule outfits (from F2 candidates)
  F5  CapsuleEvolutionTracker   → snapshot + trend
  ─────────────────────────────────────────────────
  Layer 4  — LLM narration (optional, requires GOOGLE_API_KEY)
  Layer 5  — Capsule charts (optional, requires matplotlib)

Output folder structure
  tests/output/capsule_runs/run_YYYYMMDD_HHMMSS/
  ├── garments/                 ← one JSON per extracted garment
  ├── capsule/
  │   ├── analysis.json         ← F1 full result
  │   ├── missing_pieces.json   ← F2 recommendations
  │   ├── replacement_plan.json ← F3 verdicts
  │   ├── capsule_outfits.json  ← F4 ranked outfits
  │   └── evolution.json        ← F5 snapshots + trend
  ├── charts/                   ← PNG charts (F1/F5 visualisations)
  ├── snapshots/                ← evolution profile (F5 persistence)
  └── final_report.json         ← merged summary

Usage examples
  # Analyse the default sample wardrobe
  python scripts/analyse_wardrobe_capsule.py

  # Custom wardrobe folder
  python scripts/analyse_wardrobe_capsule.py --wardrobe data/wardrobe_2

  # With LLM narration
  python scripts/analyse_wardrobe_capsule.py --llm

  # Custom user profile (personalises missing-piece colours)
  python scripts/analyse_wardrobe_capsule.py --season autumn --body-shape hourglass

  # Ask for 10 missing-piece recommendations
  python scripts/analyse_wardrobe_capsule.py --top-n 10

  # Skip heavy Vision AI extraction (use cached garments JSON)
  python scripts/analyse_wardrobe_capsule.py --from-cache tests/output/capsule_runs/run_xxx/garments

  # Store results in a custom directory
  python scripts/analyse_wardrobe_capsule.py --out my_results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import get_logger
from src.core.models import Garment, GarmentAttributes, GarmentCategory

logger = get_logger(__name__)

# ── Default paths ────────────────────────────────────────────────────────────
_DEFAULT_WARDROBE = Path(__file__).parent.parent / "data" / "sample_wardrobe"
_DEFAULT_OUTPUT   = Path(__file__).parent.parent / "tests" / "output" / "capsule_runs"
_PROFILES_ROOT    = Path(__file__).parent.parent / "tests" / "output" / "capsule_profiles"


# ══════════════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    width = 64
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _section(title: str) -> None:
    print(f"\n{'─' * 52}")
    print(f"  {title}")
    print(f"{'─' * 52}")


def _ok(msg: str)  -> None: print(f"  ✅  {msg}")
def _warn(msg: str) -> None: print(f"  ⚠️   {msg}")
def _info(msg: str) -> None: print(f"  ℹ️   {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# Run directory
# ══════════════════════════════════════════════════════════════════════════════

class RunDirectory:
    """Creates and owns the output directory tree for one analysis run."""

    def __init__(self, base: Path) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = base / f"run_{ts}"
        self.garments_dir  = self.root / "garments"
        self.capsule_dir   = self.root / "capsule"
        self.charts_dir    = self.root / "charts"
        self.snapshots_dir = self.root / "snapshots"
        for d in (self.garments_dir, self.capsule_dir,
                  self.charts_dir, self.snapshots_dir):
            d.mkdir(parents=True, exist_ok=True)
        print(f"\n📁  Results → {self.root}")

    # ── helpers ──────────────────────────────────────────────────────────────

    def write_json(self, rel_path: str, data: Any) -> Path:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        return path

    def read_json(self, path: Path) -> Any:
        with open(path) as fh:
            return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — garment extraction
# ══════════════════════════════════════════════════════════════════════════════

async def extract_garments(
    wardrobe_path: Path,
    run: RunDirectory,
    use_segmentation: bool = True,
) -> List[Garment]:
    """
    Run Layer 0 (segmentation) + Layer 1 (attribute extraction) on
    every image in *wardrobe_path*.

    Saves one JSON file per garment in run.garments_dir.
    Returns the list of extracted Garment objects.
    """
    _section("Layer 0+1 — Segmentation & Attribute Extraction")

    # ── Imports ───────────────────────────────────────────────────────────
    from src.layer1_vision import AttributeExtractor
    from src.layer2_style import OutfitBuilder

    builder   = OutfitBuilder()
    extractor = AttributeExtractor()

    # ── Segmentation (optional) ───────────────────────────────────────────
    seg_images: Dict[str, Any] = {}  # garment.id → PIL Image
    if use_segmentation:
        try:
            from src.layer0_segmentation import get_segmenter
            segmenter = get_segmenter(use_sam=False)
            exts = (".jpg", ".jpeg", ".png", ".webp")
            image_files = sorted(
                f for f in wardrobe_path.iterdir()
                if f.is_file() and f.suffix.lower() in exts
            )
            print(f"\n  Segmenting {len(image_files)} images …")
            for img_path in image_files:
                try:
                    result = segmenter.segment_garment(img_path)
                    seg_images[str(img_path.resolve())] = result.image
                    _ok(f"Segmented: {img_path.name}")
                except Exception as exc:
                    _warn(f"Could not segment {img_path.name}: {exc}")
        except ImportError:
            _warn("Layer 0 (segmentation) not available — skipping")

    # ── Layer 1: attribute extraction ─────────────────────────────────────
    print("\n  Extracting attributes (Layer 1) …")
    t0 = time.time()
    garments: List[Garment] = await builder.load_outfit_from_folder(wardrobe_path)
    elapsed = (time.time() - t0) * 1000
    print(f"\n  Extracted {len(garments)} garments in {elapsed:.0f} ms")

    # ── Persist garments ──────────────────────────────────────────────────
    for idx, g in enumerate(garments):
        attrs = g.attributes
        data: Dict[str, Any] = {
            "id":         g.id,
            "image_path": str(g.image_path) if g.image_path else None,
            "extracted_at": datetime.now().isoformat(),
            "attributes": {
                "category":       attrs.category.value,
                "subcategory":    attrs.subcategory,
                "color_primary":  attrs.color.primary if attrs.color else None,
                "color_secondary":attrs.color.secondary if attrs.color else None,
                "formality":      attrs.formality_level.value if attrs.formality_level else None,
                "seasons":        [s.value if hasattr(s, "value") else s
                                   for s in (attrs.season_suitable or [])],
                "pattern":        attrs.pattern.type if attrs.pattern else None,
                "material":       attrs.material.primary if attrs.material else None,
                "subcategory":    attrs.subcategory,
            },
        }
        run.write_json(f"garments/{idx:03d}_{g.id[:12]}.json", data)

    _ok(f"Saved {len(garments)} garment JSONs → {run.garments_dir.name}/")
    return garments


def load_garments_from_cache(cache_dir: Path) -> List[Garment]:
    """
    Reconstruct lightweight Garment stubs from previously saved JSON files.
    Used with --from-cache to skip re-extraction.
    """
    from src.core.models import (
        GarmentAttributes, ColorProfile, PatternInfo, MaterialProfile,
        FormalityLevel, Season, SeasonalityInfo,
    )

    garments: List[Garment] = []
    for jf in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
            attrs_raw = data.get("attributes", {})

            # Safely map category
            try:
                cat = GarmentCategory(attrs_raw.get("category", "top"))
            except ValueError:
                cat = GarmentCategory.TOP

            # Safely map formality
            try:
                formality = FormalityLevel(attrs_raw.get("formality", "casual"))
            except (ValueError, TypeError):
                formality = FormalityLevel.CASUAL

            # Safely map seasons
            season_list: List[Season] = []
            for sv in attrs_raw.get("seasons", []):
                try:
                    season_list.append(Season(sv))
                except ValueError:
                    pass
            if not season_list:
                season_list = [Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER]

            color_primary = attrs_raw.get("color_primary") or "unknown"
            pattern_type  = attrs_raw.get("pattern")  or "solid"
            material      = attrs_raw.get("material") or "unknown"

            garment = Garment(
                id=data.get("id", jf.stem),
                image_path=data.get("image_path"),
                attributes=GarmentAttributes(
                    category=cat,
                    subcategory=attrs_raw.get("subcategory"),
                    color=ColorProfile(primary=color_primary, hex_codes=[]),
                    formality_level=formality,
                    season_suitable=season_list,
                    seasonality=SeasonalityInfo(seasons=season_list),
                    pattern=PatternInfo(type=pattern_type),
                    material=MaterialProfile(primary=material),
                ),
            )
            garments.append(garment)
        except Exception as exc:
            _warn(f"Could not load {jf.name}: {exc}")

    _ok(f"Loaded {len(garments)} garments from cache ({cache_dir})")
    return garments


# ══════════════════════════════════════════════════════════════════════════════
# F1 — Capsule Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_capsule_analysis(
    garments: List[Garment],
    run: RunDirectory,
    centrality_map: Optional[Dict[str, float]] = None,
) -> Any:
    """F1: Analyse the wardrobe for capsule readiness."""
    _section("F1 — Wardrobe Capsule Analysis")

    from src.layer2_style.capsule import WardrobeCapsuleAnalyzer

    analyzer = WardrobeCapsuleAnalyzer()
    t0 = time.time()
    result = analyzer.analyze(garments, centrality_map=centrality_map)
    elapsed = (time.time() - t0) * 1000

    # ── Console summary ───────────────────────────────────────────────────
    score_bar = "█" * int(result.cohesion_score / 5) + "░" * (20 - int(result.cohesion_score / 5))
    print(f"\n  Cohesion score  : {result.cohesion_score:.1f}/100  [{score_bar}]")
    print(f"  Profile         : {result.capsule_profile}")
    print(f"  Total garments  : {result.total_garments}")
    print(f"  Key pieces      : {len(result.key_pieces)}")
    print(f"  Orphan pieces   : {len(result.orphan_pieces)}")
    print(f"  Redundant pairs : {len(result.redundant_pairs)}")
    print(f"  Dominant colors : {', '.join(result.dominant_colors[:5])}")
    print(f"  Color cohesion  : {result.color_cohesion_score:.2f}")
    print(f"  Versatility     : {result.versatility_ratio:.2f}")
    print(f"  Projected score : {result.projected_score_after_cleanup:.1f} (after cleanup)")
    print(f"  Computed in     : {elapsed:.0f} ms")

    if result.recommendation:
        print(f"\n  💬  {result.recommendation}")

    # ── Per-garment table ─────────────────────────────────────────────────
    print(f"\n  {'#':<3} {'Garment':<28} {'Role':<12} {'Versatility':>12}")
    print(f"  {'─'*3} {'─'*28} {'─'*12} {'─'*12}")
    for i, gs in enumerate(
        sorted(result.garment_scores, key=lambda x: x.versatility_score, reverse=True), 1
    ):
        role_icon = {"key_piece": "🟢", "acceptable": "🔵",
                     "orphan": "🔴", "redundant": "🟡"}.get(gs.capsule_role.value, "⚪")
        print(f"  {i:<3} {gs.garment_description[:28]:<28} "
              f"{role_icon} {gs.capsule_role.value:<10} {gs.versatility_score:>12.3f}")

    # ── Redundant pairs ───────────────────────────────────────────────────
    if result.redundant_pairs:
        print(f"\n  Redundant pairs ({len(result.redundant_pairs)}):")
        for rp in result.redundant_pairs[:5]:
            print(f"    • {rp.garment_a_description}  ≈  {rp.garment_b_description}"
                  f"  (sim={rp.similarity_score:.2f})")
        if len(result.redundant_pairs) > 5:
            print(f"    … and {len(result.redundant_pairs) - 5} more")

    # ── Persist ───────────────────────────────────────────────────────────
    run.write_json("capsule/analysis.json", result.model_dump())
    _ok(f"Saved → capsule/analysis.json  ({elapsed:.0f} ms)")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# F2 — Missing Pieces
# ══════════════════════════════════════════════════════════════════════════════

def run_missing_pieces(
    analysis: Any,
    run: RunDirectory,
    top_n: int = 5,
    user_season: Optional[str] = None,
    body_shape: Optional[str] = None,
) -> Any:
    """F2: Recommend the highest-impact missing pieces."""
    _section("F2 — Missing Pieces Recommender")

    from src.layer2_style.capsule import MissingPiecesRecommender

    recommender = MissingPiecesRecommender(top_n=top_n)
    t0 = time.time()
    result = recommender.recommend(analysis, user_season=user_season, body_shape=body_shape)
    elapsed = (time.time() - t0) * 1000

    print(f"\n  Current cohesion  : {result.current_cohesion:.1f}")
    print(f"  Projected cohesion: {result.projected_cohesion:.1f}"
          f"  (+{result.projected_cohesion - result.current_cohesion:.1f})")
    print(f"\n  Top-{len(result.recommendations)} missing pieces:")
    print(f"  {'#':<3} {'Category':<12} {'Description':<28} {'Impact outfits':>14}")
    print(f"  {'─'*3} {'─'*12} {'─'*28} {'─'*14}")
    for rec in result.recommendations:
        print(f"  {rec.priority:<3} {rec.category:<12} {rec.description[:28]:<28}"
              f" {rec.impact_outfits:>14}")
        if rec.profile_note:
            print(f"          ↳ {rec.profile_note}")

    if result.summary:
        print(f"\n  💬  {result.summary}")

    run.write_json("capsule/missing_pieces.json", result.model_dump())
    _ok(f"Saved → capsule/missing_pieces.json  ({elapsed:.0f} ms)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# F3 — Replacement Planner
# ══════════════════════════════════════════════════════════════════════════════

def run_replacement_planner(
    analysis: Any,
    run: RunDirectory,
) -> Any:
    """F3: For each redundant pair, decide which to keep and when to replace."""
    _section("F3 — Replacement Planner")

    from src.layer2_style.capsule import ReplacementPlanner

    planner = ReplacementPlanner()
    t0 = time.time()
    plan = planner.plan(analysis.redundant_pairs, analysis.garment_scores)
    elapsed = (time.time() - t0) * 1000

    if not plan.verdicts:
        _info("No replacement verdicts — wardrobe has no strongly redundant pairs.")
    else:
        print(f"\n  {len(plan.verdicts)} replacement verdict(s)  "
              f"(estimated +{plan.total_outfits_gained} outfit combinations)")
        print(f"\n  {'#':<3} {'Keep':<28} {'Remove':<28} {'Gain':>6} {'Conf':>6}")
        print(f"  {'─'*3} {'─'*28} {'─'*28} {'─'*6} {'─'*6}")
        for i, v in enumerate(plan.verdicts, 1):
            keep_label   = (v.garment_keep_description   or v.garment_keep_id)[:28]
            remove_label = (v.garment_remove_description or v.garment_remove_id)[:28]
            print(f"  {i:<3} {keep_label:<28} {remove_label:<28}"
                  f" {v.versatility_gain:>+6.2f} {v.confidence:>6.0%}")
            print(f"        ↳ {v.transition_timing}")

    if plan.summary:
        print(f"\n  💬  {plan.summary}")

    run.write_json("capsule/replacement_plan.json", plan.model_dump())
    _ok(f"Saved → capsule/replacement_plan.json  ({elapsed:.0f} ms)")
    return plan


# ══════════════════════════════════════════════════════════════════════════════
# F4 — Capsule Outfit Generator
# ══════════════════════════════════════════════════════════════════════════════

def run_capsule_outfit_generator(
    garments: List[Garment],
    analysis: Any,
    run: RunDirectory,
    top_n: int = 12,
) -> Any:
    """F4: Score existing outfit candidates for capsule readiness."""
    _section("F4 — Capsule Outfit Generator")

    from src.layer2_style import OutfitBuilder
    from src.layer2_style.capsule import CapsuleOutfitGenerator

    # Build outfit candidates (same as normal pipeline)
    print("  Building outfit combinations …")
    builder = OutfitBuilder()
    combinations = builder.generate_outfit_combinations(garments)

    t0 = time.time()
    candidates = builder.evaluate_all_combinations(combinations, profile="default")
    build_ms = (time.time() - t0) * 1000
    print(f"  Built {len(candidates)} outfit candidates in {build_ms:.0f} ms")

    if not candidates:
        _warn("No outfit candidates — wardrobe may be too small for combinations.")
        from src.core.models import CapsuleOutfitsResult
        empty = CapsuleOutfitsResult(outfits=[], basic_count=0,
                                    semi_creative_count=0, creative_count=0)
        run.write_json("capsule/capsule_outfits.json", empty.model_dump())
        return empty

    generator = CapsuleOutfitGenerator()
    t0 = time.time()
    result = generator.generate(candidates, analysis, top_n=top_n)
    gen_ms = (time.time() - t0) * 1000

    print(f"\n  {len(result.outfits)} capsule outfits selected (top {top_n})")
    print(f"  Tiers  →  Basic: {result.basic_count} "
          f"| Semi-creative: {result.semi_creative_count} "
          f"| Creative: {result.creative_count}")

    if result.outfits:
        print(f"\n  {'Rank':<5} {'Tier':<14} {'Score':>7} {'Key%':>6} {'Versatility':>12}")
        print(f"  {'─'*5} {'─'*14} {'─'*7} {'─'*6} {'─'*12}")
        for o in result.outfits[:10]:
            print(f"  {o.rank:<5} {o.tier:<14} {o.capsule_score:>7.1f}"
                  f" {o.pct_key_pieces:>6.0%} {o.avg_versatility:>12.3f}")
            if o.occasions:
                print(f"        occasions: {', '.join(o.occasions[:3])}")

    run.write_json("capsule/capsule_outfits.json", result.model_dump())
    _ok(f"Saved → capsule/capsule_outfits.json  ({gen_ms:.0f} ms)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# F5 — Evolution Tracker
# ══════════════════════════════════════════════════════════════════════════════

def run_evolution_tracker(
    garments: List[Garment],
    analysis: Any,
    run: RunDirectory,
    user_id: str = "default",
    action: str = "",
) -> Any:
    """F5: Record a snapshot and compute trend analytics."""
    _section("F5 — Capsule Evolution Tracker")

    from unittest.mock import patch
    from src.layer2_style.capsule import CapsuleEvolutionTracker, WardrobeCapsuleAnalyzer

    wardrobe_hash = WardrobeCapsuleAnalyzer.wardrobe_hash(garments)
    _info(f"Wardrobe hash: {wardrobe_hash}")

    # Store snapshots inside the run directory so they persist across runs
    snapshots_root = run.snapshots_dir
    with patch(
        "src.layer2_style.capsule.capsule_evolution_tracker._PROFILES_ROOT",
        snapshots_root,
    ):
        tracker = CapsuleEvolutionTracker(user_id=user_id)
        t0 = time.time()
        result = tracker.record(analysis, wardrobe_hash, action=action or "capsule_analysis_run")
        elapsed = (time.time() - t0) * 1000

    print(f"\n  Snapshots recorded : {len(result.snapshots)}")
    print(f"  Latest cohesion    : {result.latest_cohesion:.1f}")
    print(f"  Delta score        : {result.delta_score:+.1f}")
    print(f"  Delta outfits      : {result.delta_outfits:+d}")
    print(f"  Trend direction    : {result.trend_direction}")
    if result.predicted_weeks_to_90 is not None:
        if result.predicted_weeks_to_90 == 0:
            print(f"  Weeks to score 90  : 🎉  Already at 90+!")
        else:
            print(f"  Weeks to score 90  : ~{result.predicted_weeks_to_90} weeks")
    else:
        print(f"  Weeks to score 90  : not enough data to predict")

    run.write_json("capsule/evolution.json", result.model_dump())
    _ok(f"Saved → capsule/evolution.json  ({elapsed:.0f} ms)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LLM narration  (optional)
# ══════════════════════════════════════════════════════════════════════════════

async def run_llm_narration(
    analysis: Any,
    missing: Any,
    plan: Any,
    outfits: Any,
    evolution: Any,
    run: RunDirectory,
) -> None:
    """Add LLM (Gemini) narration to all capsule results in-place."""
    _section("Layer 4 — LLM Narration (Gemini)")

    try:
        from src.layer4_llm.capsule_explainer import CapsuleExplainer
    except ImportError as exc:
        _warn(f"CapsuleExplainer not available: {exc}")
        return

    explainer = CapsuleExplainer()

    async def _run(label: str, coro):
        t0 = time.time()
        try:
            await coro
            _ok(f"{label}  ({(time.time()-t0)*1000:.0f} ms)")
        except Exception as exc:
            _warn(f"{label} failed: {exc}")

    await _run("Wardrobe diagnosis", explainer.narrate_wardrobe_diagnosis(analysis))
    await _run("Missing pieces",     explainer.narrate_missing_pieces(missing))
    await _run("Replacement plan",   explainer.narrate_replacement_plan(plan))
    await _run("Capsule outfits",    explainer.narrate_capsule_outfits(outfits, top_n=5))
    await _run("Evolution trend",    explainer.narrate_evolution(evolution))

    # Re-persist with narrations
    run.write_json("capsule/analysis.json",          analysis.model_dump())
    run.write_json("capsule/missing_pieces.json",    missing.model_dump())
    run.write_json("capsule/replacement_plan.json",  plan.model_dump())
    run.write_json("capsule/capsule_outfits.json",   outfits.model_dump())
    run.write_json("capsule/evolution.json",         evolution.model_dump())
    _ok("All LLM narrations persisted")


# ══════════════════════════════════════════════════════════════════════════════
# Layer 5 — Charts
# ══════════════════════════════════════════════════════════════════════════════

async def run_charts(
    analysis: Any,
    evolution: Any,
    run: RunDirectory,
) -> None:
    """Generate all 4 capsule charts and save to charts/."""
    _section("Layer 5 — Capsule Charts")

    try:
        from src.layer5_visualization.capsule_visualizer import CapsuleVisualizer
    except ImportError as exc:
        _warn(f"CapsuleVisualizer not available: {exc}")
        return

    visualizer = CapsuleVisualizer(output_dir=run.charts_dir)
    t0 = time.time()
    paths = await visualizer.generate_all(analysis, evolution=evolution)
    elapsed = (time.time() - t0) * 1000

    if paths:
        for name, p in paths.items():
            _ok(f"Chart saved: {p.relative_to(run.root)}")
    else:
        _warn("No charts generated (matplotlib may not be installed)")

    print(f"  Total chart time: {elapsed:.0f} ms")


# ══════════════════════════════════════════════════════════════════════════════
# Final consolidated report
# ══════════════════════════════════════════════════════════════════════════════

def write_final_report(
    run: RunDirectory,
    garments: List[Garment],
    analysis: Any,
    missing: Any,
    plan: Any,
    outfits: Any,
    evolution: Any,
    elapsed_total_s: float,
) -> Path:
    """Merge all results into a single final_report.json."""
    _section("Saving Final Report")

    report: Dict[str, Any] = {
        "run_directory":    str(run.root),
        "generated_at":     datetime.now().isoformat(),
        "elapsed_seconds":  round(elapsed_total_s, 2),
        "wardrobe_size":    len(garments),
        "summary": {
            "cohesion_score":           analysis.cohesion_score,
            "capsule_profile":          analysis.capsule_profile,
            "key_pieces":               len(analysis.key_pieces),
            "orphan_pieces":            len(analysis.orphan_pieces),
            "redundant_pairs":          len(analysis.redundant_pairs),
            "projected_score_cleanup":  analysis.projected_score_after_cleanup,
            "missing_pieces_top1":      (
                missing.recommendations[0].description
                if missing.recommendations else None
            ),
            "missing_projected_gain":   round(
                missing.projected_cohesion - missing.current_cohesion, 2
            ),
            "replacement_verdicts":     len(plan.verdicts),
            "outfits_gained_by_replace":plan.total_outfits_gained,
            "capsule_outfits_count":    len(outfits.outfits),
            "evolution_trend":          evolution.trend_direction,
            "weeks_to_90":              evolution.predicted_weeks_to_90,
        },
        "capsule_analysis":    analysis.model_dump(),
        "missing_pieces":      missing.model_dump(),
        "replacement_plan":    plan.model_dump(),
        "capsule_outfits":     outfits.model_dump(),
        "evolution":           evolution.model_dump(),
    }

    path = run.write_json("final_report.json", report)
    _ok(f"Final report → {path.relative_to(run.root.parent.parent)}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_travel_mode(garments, args, run: RunDirectory) -> None:
    """Run the travel planning pipeline and write results."""
    from src.layer3_context.travel.constraint_parser import ConstraintParser
    from src.layer2_style.travel.travel_wardrobe_planner import TravelWardrobePlanner
    from src.layer2_style.travel.packing_score_calculator import PackingScoreCalculator

    _section("Travel Planning")
    parser_c = ConstraintParser()
    constraints = parser_c.parse(
        destination=args.destination or "generic",
        days=args.days,
        occasions=args.occasions,
        max_pieces=args.max_pieces,
        travel_date=args.travel_date,
        body_shape=args.body_shape,
        season=args.season,
    )
    _info(f"Destination : {constraints.destination} | {constraints.days} days | "
          f"{constraints.max_pieces} pieces max")
    _info(f"Occasions   : {constraints.occasions}")

    planner = TravelWardrobePlanner()
    t0 = time.time()
    plan = planner.plan(garments, constraints)
    elapsed = time.time() - t0

    _ok(f"Packing plan generated in {elapsed:.2f}s")
    _ok(f"Pieces packed    : {len(plan.packed_garments)} / {constraints.max_pieces}")
    _ok(f"Packing score    : {plan.packing_score:.1f}/100 ({plan.score_label})")
    _ok(f"Versatility ratio: {plan.versatility_ratio:.1f} outfits/piece")
    if plan.warnings:
        for w in plan.warnings:
            _warn(w)

    path = run.write_json("travel/packing_plan.json", plan.model_dump())
    _ok(f"Packing plan → {path.relative_to(run.root.parent.parent)}")


def run_season_mode(garments, args, run: RunDirectory) -> None:
    """Run the season planning pipeline and write results."""
    from src.layer2_style.travel.seasonal_audit_engine import SeasonalAuditEngine
    from src.layer2_style.travel.shopping_list_optimizer import ShoppingListOptimizer
    from src.layer3_context.travel.season_transition_advisor import SeasonTransitionAdvisor
    from src.layer3_context.travel.weekly_rotation_planner import WeeklyRotationPlanner

    target_season = args.target_season or args.season or "fall"

    # --- Seasonal audit ---
    _section("Season Audit")
    engine = SeasonalAuditEngine()
    audit = engine.audit(garments, target_season)
    _ok(f"Target season     : {target_season}")
    _ok(f"Ready             : {audit.ready_count} pieces")
    _ok(f"Adaptable         : {audit.adaptable_count} pieces")
    _ok(f"Store             : {audit.store_count} pieces")
    _ok(f"Overall readiness : {audit.overall_readiness_pct:.1f}%")
    if audit.top_gaps:
        for gap in audit.top_gaps:
            _warn(gap)

    run.write_json("season/audit.json", audit.model_dump())

    # --- Shopping list ---
    _section("Shopping List Optimiser")
    hard_gaps = [c.category for c in audit.coverage_by_category if c.gap > 0]
    optimizer = ShoppingListOptimizer()
    shopping = optimizer.optimize(
        garments,
        budget=args.budget,
        season=target_season,
        body_shape=args.body_shape,
        hard_gaps=hard_gaps,
    )
    _ok(f"Budget            : €{shopping.budget_eur:.0f}")
    _ok(f"Within budget     : {len(shopping.within_budget)} items (€{shopping.total_estimated_cost:.0f})")
    _ok(f"Projected new outfits: +{shopping.projected_new_outfits}")
    for item in shopping.within_budget[:3]:
        _info(f"  [{item.urgency.upper()}] {item.description} (ROI={item.roi:.1f}) — €{item.estimated_price_eur:.0f}")

    run.write_json("season/shopping_list.json", shopping.model_dump())

    # --- Season transition ---
    _section("Season Transition Plan")
    advisor = SeasonTransitionAdvisor()
    transition = advisor.advise(garments, args.season or "summer", target_season)
    _ok(f"Store : {len(transition.store_ids)} pieces")
    _ok(f"Keep  : {len(transition.keep_ids)} pieces")
    if transition.color_shift_notes:
        for note in transition.color_shift_notes[:3]:
            _info(f"  🎨  {note}")

    run.write_json("season/transition_plan.json", transition.model_dump())

    # --- Weekly rotation ---
    _section("Weekly Rotation Plan")
    rotation_planner = WeeklyRotationPlanner()
    weekly = rotation_planner.plan(
        garments,
        work_days=args.work_days,
        weekend_days=args.weekend_days,
        work_occasion="work",
        weekend_occasion="casual",
    )
    _ok(f"Slots planned     : {len(weekly.slots)}")
    _ok(f"Repeat rate       : {weekly.repeat_rate:.0%}")
    _ok(f"Coverage          : {weekly.coverage_pct:.0%}")
    if weekly.warnings:
        for w in weekly.warnings:
            _warn(w)

    run.write_json("season/weekly_rotation.json", weekly.model_dump())


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse a wardrobe folder through the full capsule pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", choices=["capsule", "travel", "season"], default="capsule",
        help="Analysis mode (default: capsule)",
    )

    # ── Input ──────────────────────────────────────────────────────────────
    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "--wardrobe", type=Path, default=_DEFAULT_WARDROBE,
        metavar="DIR",
        help=f"Wardrobe folder (default: {_DEFAULT_WARDROBE})",
    )
    io_group.add_argument(
        "--out", type=Path, default=_DEFAULT_OUTPUT,
        metavar="DIR",
        help=f"Base output directory (default: {_DEFAULT_OUTPUT})",
    )
    io_group.add_argument(
        "--from-cache", type=Path, default=None,
        metavar="GARMENTS_DIR",
        help="Load garments from a previous run's garments/ folder (skip Layer 1)",
    )

    # ── Profile ────────────────────────────────────────────────────────────
    profile_group = parser.add_argument_group("User Profile")
    profile_group.add_argument(
        "--season",
        choices=["spring", "summer", "autumn", "winter"],
        default=None,
        help="Colour season — personalises missing-piece colour suggestions",
    )
    profile_group.add_argument(
        "--body-shape",
        choices=["rectangle", "triangle", "inverted_triangle",
                 "hourglass", "oval", "athletic"],
        default=None,
        dest="body_shape",
        help="Body shape — personalises missing-piece notes",
    )
    profile_group.add_argument(
        "--user-id",
        default="default",
        dest="user_id",
        help="User ID used to persist evolution snapshots (default: 'default')",
    )

    # ── Analysis options ───────────────────────────────────────────────────
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--top-n", type=int, default=5,
        metavar="N",
        help="Number of missing-piece recommendations (default: 5)",
    )
    analysis_group.add_argument(
        "--top-outfits", type=int, default=12,
        metavar="N",
        help="Max capsule outfits to generate (default: 12)",
    )
    analysis_group.add_argument(
        "--action", default="",
        help="Label recorded in the evolution snapshot (e.g. 'removed_redundant_pair')",
    )
    analysis_group.add_argument(
        "--no-segmentation", action="store_true",
        help="Skip Layer 0 segmentation (faster, uses raw images for Layer 1)",
    )

    # ── Optional enrichments ───────────────────────────────────────────────
    extras_group = parser.add_argument_group("Optional Enrichments")
    extras_group.add_argument(
        "--llm", action="store_true",
        help="Add Gemini LLM narration to all capsule results (requires GOOGLE_API_KEY)",
    )
    extras_group.add_argument(
        "--charts", action="store_true",
        help="Generate matplotlib charts in charts/ (requires matplotlib)",
    )
    extras_group.add_argument(
        "--all", action="store_true",
        help="Enable all optional enrichments (--llm + --charts)",
    )

    # ── Travel mode args ───────────────────────────────────────────────────
    travel_group = parser.add_argument_group("Travel Mode (--mode travel)")
    travel_group.add_argument(
        "--destination", default=None,
        help="Trip destination city/country (e.g. 'Rome')",
    )
    travel_group.add_argument(
        "--days", type=int, default=7,
        help="Number of trip days (default: 7)",
    )
    travel_group.add_argument(
        "--occasions", default=None,
        help="Occasions CSV: 'casual:4,evening:2,tourism:1'",
    )
    travel_group.add_argument(
        "--max-pieces", type=int, default=12, dest="max_pieces",
        help="Max items to pack (default: 12)",
    )
    travel_group.add_argument(
        "--travel-date", default=None, dest="travel_date",
        help="Travel start date ISO format YYYY-MM-DD (infers season if not set)",
    )

    # ── Season mode args ───────────────────────────────────────────────────
    season_group = parser.add_argument_group("Season Mode (--mode season)")
    season_group.add_argument(
        "--target-season",
        choices=["spring", "summer", "fall", "autumn", "winter"],
        default=None, dest="target_season",
        help="Target season to prepare for",
    )
    season_group.add_argument(
        "--budget", type=float, default=300.0,
        help="Shopping budget in EUR (default: 300)",
    )
    season_group.add_argument(
        "--work-days", type=int, default=5, dest="work_days",
        help="Number of work days in weekly rotation (default: 5)",
    )
    season_group.add_argument(
        "--weekend-days", type=int, default=2, dest="weekend_days",
        help="Number of weekend days in weekly rotation (default: 2)",
    )

    args = parser.parse_args()

    if args.all:
        args.llm    = True
        args.charts = True

    # ── Validate ───────────────────────────────────────────────────────────
    if args.from_cache is None:
        if not args.wardrobe.exists():
            print(f"❌  Wardrobe folder not found: {args.wardrobe}")
            sys.exit(1)
        images = [
            f for f in args.wardrobe.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ]
        if not images:
            print(f"❌  No images found in: {args.wardrobe}")
            sys.exit(1)
    else:
        if not args.from_cache.exists():
            print(f"❌  Cache folder not found: {args.from_cache}")
            sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    _banner("Wardrobe Analyser")
    print(f"  Mode      : {args.mode}")
    print(f"  Wardrobe  : {args.wardrobe}")
    print(f"  Output    : {args.out}")
    print(f"  Season    : {args.season or 'not specified'}")
    print(f"  Body shape: {args.body_shape or 'not specified'}")
    if args.mode == "travel":
        print(f"  Destination : {args.destination or 'generic'}")
        print(f"  Days        : {args.days}")
        print(f"  Max pieces  : {args.max_pieces}")
    elif args.mode == "season":
        print(f"  Target season: {args.target_season or args.season or 'fall'}")
        print(f"  Budget       : €{args.budget:.0f}")
    else:
        print(f"  LLM       : {'yes' if args.llm else 'no'}")
        print(f"  Charts    : {'yes' if args.charts else 'no'}")

    run   = RunDirectory(args.out)
    t_start = time.time()

    # ── Layer 0+1 : garment extraction ────────────────────────────────────
    if args.from_cache:
        garments = load_garments_from_cache(args.from_cache)
    else:
        garments = await extract_garments(
            args.wardrobe,
            run,
            use_segmentation=not args.no_segmentation,
        )

    if not garments:
        print("❌  No garments could be extracted. Aborting.")
        sys.exit(1)

    # ── Route to mode ─────────────────────────────────────────────────────
    if args.mode == "travel":
        run_travel_mode(garments, args, run)
        elapsed_total = time.time() - t_start
        _banner("Travel Planning Complete ✅")
        print(f"  Total time : {elapsed_total:.1f} s")
        print(f"  Results    : {run.root}")
        return

    if args.mode == "season":
        run_season_mode(garments, args, run)
        elapsed_total = time.time() - t_start
        _banner("Season Planning Complete ✅")
        print(f"  Total time : {elapsed_total:.1f} s")
        print(f"  Results    : {run.root}")
        return

    # ── Capsule mode (default) ─────────────────────────────────────────────
    analysis = run_capsule_analysis(garments, run)

    # ── F2 ────────────────────────────────────────────────────────────────
    missing = run_missing_pieces(
        analysis, run,
        top_n=args.top_n,
        user_season=args.season,
        body_shape=args.body_shape,
    )

    # ── F3 ────────────────────────────────────────────────────────────────
    plan = run_replacement_planner(analysis, run)

    # ── F4 ────────────────────────────────────────────────────────────────
    outfits = run_capsule_outfit_generator(
        garments, analysis, run, top_n=args.top_outfits
    )

    # ── F5 ────────────────────────────────────────────────────────────────
    evolution = run_evolution_tracker(
        garments, analysis, run,
        user_id=args.user_id,
        action=args.action,
    )

    # ── LLM (optional) ────────────────────────────────────────────────────
    if args.llm:
        await run_llm_narration(analysis, missing, plan, outfits, evolution, run)

    # ── Charts (optional) ─────────────────────────────────────────────────
    if args.charts:
        await run_charts(analysis, evolution, run)

    # ── Final report ──────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    report_path = write_final_report(
        run, garments, analysis, missing, plan, outfits, evolution,
        elapsed_total_s=elapsed_total,
    )

    # ══════════════════════════════════════════════════════════════════════
    _banner("Analysis Complete ✅")
    print(f"  Total time   : {elapsed_total:.1f} s")
    print(f"  Results dir  : {run.root}")
    print()
    print("  Quick summary:")
    print(f"    Cohesion score : {analysis.cohesion_score:.1f}/100"
          f"  (projected {analysis.projected_score_after_cleanup:.1f} after cleanup)")
    print(f"    Key pieces     : {len(analysis.key_pieces)} / {len(garments)}")
    print(f"    Redundant pairs: {len(analysis.redundant_pairs)}")
    print(f"    Missing pieces : {len(missing.recommendations)} suggested"
          f"  (+{missing.projected_cohesion - missing.current_cohesion:.1f} pts)")
    print(f"    Replace verdicts: {len(plan.verdicts)}"
          f"  (+{plan.total_outfits_gained} outfit combos)")
    print(f"    Capsule outfits : {len(outfits.outfits)}"
          f"  (basic {outfits.basic_count}"
          f" | semi {outfits.semi_creative_count}"
          f" | creative {outfits.creative_count})")
    print(f"    Trend           : {evolution.trend_direction}")
    print()
    print(f"  📄  Full report → {report_path}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
