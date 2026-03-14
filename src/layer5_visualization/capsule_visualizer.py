"""
Layer 5: Capsule Wardrobe Visualizer
=====================================
Four matplotlib charts that illustrate the capsule analysis results.
Charts are generated in parallel via asyncio.gather() and saved to disk.
Hash-based skip logic avoids re-rendering unchanged data.

Charts:
  1. timeline_score     — Line plot of cohesion score over time (evolution snapshots)
  2. versatility_dist   — Histogram of versatility scores, coloured by role
                          (red = orphan, green = key_piece, grey = others)
  3. redundant_pairs    — Horizontal bar chart of top redundant pairs by similarity
  4. before_after       — Two bars: current cohesion vs projected cohesion after cleanup
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional matplotlib import (graceful degradation if not installed)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend — no display required
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — CapsuleVisualizer disabled")

from src.core.models import (
    CapsuleAnalysisResult,
    CapsuleEvolutionResult,
    GarmentCapsuleRole,
)

# ---------------------------------------------------------------------------
# Palette (Quiet Luxury)
# ---------------------------------------------------------------------------
_C_IVORY      = "#F5F2EB"
_C_CHARCOAL   = "#2C2C2C"
_C_GOLD       = "#C9A84C"
_C_KEY        = "#4CAF50"   # green  — key_piece
_C_ORPHAN     = "#E53935"   # red    — orphan
_C_ACCEPTABLE = "#90A4AE"   # grey   — acceptable / redundant
_C_BG         = "#FAFAF8"


# ============================================================================
# CapsuleVisualizer
# ============================================================================

class CapsuleVisualizer:
    """
    Generate and save all 4 capsule charts.

    Parameters
    ----------
    output_dir : Path
        Directory where PNG files are written.
    dpi : int
        Resolution (default 120).
    """

    def __init__(
        self,
        output_dir: Path | str = Path("tests/output/capsule_charts"),
        dpi: int = 120,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self._hash_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    async def generate_all(
        self,
        analysis: CapsuleAnalysisResult,
        evolution: Optional[CapsuleEvolutionResult] = None,
    ) -> dict[str, Path]:
        """
        Render all charts in parallel.

        Returns a dict mapping chart_name → saved Path (None if skipped).
        """
        if not _MPL_AVAILABLE:
            logger.warning("matplotlib unavailable — skipping capsule charts")
            return {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        tasks = [
            self._chart_versatility_dist(analysis),
            self._chart_redundant_pairs(analysis),
            self._chart_before_after(analysis),
        ]
        if evolution is not None:
            tasks.append(self._chart_timeline_score(evolution))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        paths: dict[str, Path] = {}
        chart_names = ["versatility_dist", "redundant_pairs", "before_after"]
        if evolution is not None:
            chart_names.append("timeline_score")

        for name, result in zip(chart_names, results):
            if isinstance(result, Exception):
                logger.warning("Chart '%s' failed: %s", name, result)
            elif result is not None:
                paths[name] = result

        return paths

    # ------------------------------------------------------------------ #
    #  Chart 1 — Timeline score
    # ------------------------------------------------------------------ #

    async def _chart_timeline_score(
        self, evolution: CapsuleEvolutionResult
    ) -> Optional[Path]:
        snapshots = evolution.snapshots
        if len(snapshots) < 2:
            logger.debug("Not enough snapshots for timeline chart (%d)", len(snapshots))
            return None

        data_hash = self._hash_data(
            [{"date": s.date, "score": s.cohesion_score} for s in snapshots]
        )
        out_path = self.output_dir / "timeline_score.png"
        if self._is_unchanged(out_path, data_hash):
            return out_path

        dates = [s.date[:10] for s in snapshots]   # ISO date → YYYY-MM-DD
        scores = [s.cohesion_score for s in snapshots]

        fig, ax = plt.subplots(figsize=(9, 4), facecolor=_C_BG)
        ax.set_facecolor(_C_BG)

        ax.plot(dates, scores, color=_C_GOLD, linewidth=2.5, marker="o",
                markersize=6, markerfacecolor=_C_CHARCOAL, zorder=3)
        ax.fill_between(dates, scores, alpha=0.08, color=_C_GOLD)

        ax.axhline(90, color=_C_KEY, linewidth=1.2, linestyle="--", alpha=0.7,
                   label="Target 90")
        ax.axhline(50, color=_C_ORPHAN, linewidth=1.0, linestyle=":", alpha=0.5,
                   label="Threshold 50")

        ax.set_title("Capsule Cohesion Over Time", fontsize=13,
                     color=_C_CHARCOAL, pad=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=9, color=_C_CHARCOAL)
        ax.set_ylabel("Cohesion Score", fontsize=9, color=_C_CHARCOAL)
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=30, labelsize=7, colors=_C_CHARCOAL)
        ax.tick_params(axis="y", labelsize=8, colors=_C_CHARCOAL)
        ax.legend(fontsize=8, framealpha=0.4)
        _style_ax(ax)

        # Annotate latest value
        ax.annotate(
            f"{scores[-1]:.0f}",
            xy=(dates[-1], scores[-1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=_C_GOLD,
            fontweight="bold",
        )

        fig.tight_layout()
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self._hash_cache[str(out_path)] = data_hash
        logger.info("Saved chart: %s", out_path)
        return out_path

    # ------------------------------------------------------------------ #
    #  Chart 2 — Versatility distribution
    # ------------------------------------------------------------------ #

    async def _chart_versatility_dist(
        self, analysis: CapsuleAnalysisResult
    ) -> Optional[Path]:
        scores = analysis.garment_scores
        if not scores:
            return None

        data_hash = self._hash_data(
            [{"id": g.garment_id, "v": g.versatility_score, "role": g.capsule_role.value}
             for g in scores]
        )
        out_path = self.output_dir / "versatility_dist.png"
        if self._is_unchanged(out_path, data_hash):
            return out_path

        _role_color = {
            GarmentCapsuleRole.KEY_PIECE:   _C_KEY,
            GarmentCapsuleRole.ACCEPTABLE:  _C_ACCEPTABLE,
            GarmentCapsuleRole.ORPHAN:      _C_ORPHAN,
            GarmentCapsuleRole.REDUNDANT:   "#FF8F00",
        }

        xs = [g.versatility_score for g in scores]
        colors = [_role_color.get(g.capsule_role, _C_ACCEPTABLE) for g in scores]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=_C_BG)
        ax.set_facecolor(_C_BG)

        ax.bar(range(len(xs)), xs, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0.15, color=_C_KEY, linewidth=1.2, linestyle="--", alpha=0.7,
                   label="Key piece min (0.15)")
        ax.axhline(0.05, color=_C_ORPHAN, linewidth=1.0, linestyle=":", alpha=0.7,
                   label="Orphan max (0.05)")

        ax.set_title("Versatility Score by Garment", fontsize=13,
                     color=_C_CHARCOAL, pad=12, fontweight="bold")
        ax.set_xlabel("Garment index", fontsize=9, color=_C_CHARCOAL)
        ax.set_ylabel("Versatility score", fontsize=9, color=_C_CHARCOAL)
        ax.tick_params(colors=_C_CHARCOAL, labelsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, framealpha=0.4)

        # Role legend patches
        patches = [
            mpatches.Patch(color=_C_KEY,       label="Key piece"),
            mpatches.Patch(color=_C_ACCEPTABLE, label="Acceptable"),
            mpatches.Patch(color=_C_ORPHAN,    label="Orphan"),
            mpatches.Patch(color="#FF8F00",     label="Redundant"),
        ]
        ax.legend(handles=patches, fontsize=8, framealpha=0.4, loc="upper right")
        _style_ax(ax)

        fig.tight_layout()
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self._hash_cache[str(out_path)] = data_hash
        logger.info("Saved chart: %s", out_path)
        return out_path

    # ------------------------------------------------------------------ #
    #  Chart 3 — Redundant pairs
    # ------------------------------------------------------------------ #

    async def _chart_redundant_pairs(
        self, analysis: CapsuleAnalysisResult
    ) -> Optional[Path]:
        pairs = analysis.redundant_pairs
        if not pairs:
            logger.debug("No redundant pairs — skipping redundant_pairs chart")
            return None

        data_hash = self._hash_data(
            [{"a": p.garment_a_id, "b": p.garment_b_id, "sim": p.similarity_score}
             for p in pairs]
        )
        out_path = self.output_dir / "redundant_pairs.png"
        if self._is_unchanged(out_path, data_hash):
            return out_path

        # Show top-10 pairs by similarity
        top_pairs = sorted(pairs, key=lambda p: p.similarity_score, reverse=True)[:10]
        labels = [
            f"{p.garment_a_description[:18]} / {p.garment_b_description[:18]}"
            for p in top_pairs
        ]
        sims = [p.similarity_score for p in top_pairs]

        fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 0.45 + 1)), facecolor=_C_BG)
        ax.set_facecolor(_C_BG)

        y_pos = range(len(labels))
        bar_colors = [
            _C_ORPHAN if s >= 0.9 else _C_GOLD if s >= 0.75 else _C_ACCEPTABLE
            for s in sims
        ]
        ax.barh(list(y_pos), sims, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=8, color=_C_CHARCOAL)
        ax.set_xlabel("Similarity score", fontsize=9, color=_C_CHARCOAL)
        ax.set_xlim(0, 1.05)
        ax.set_title("Top Redundant Pairs", fontsize=13,
                     color=_C_CHARCOAL, pad=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8, colors=_C_CHARCOAL)
        _style_ax(ax)

        # Value labels on bars
        for i, sim in enumerate(sims):
            ax.text(sim + 0.01, i, f"{sim:.2f}", va="center", fontsize=7,
                    color=_C_CHARCOAL)

        fig.tight_layout()
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self._hash_cache[str(out_path)] = data_hash
        logger.info("Saved chart: %s", out_path)
        return out_path

    # ------------------------------------------------------------------ #
    #  Chart 4 — Before / after cohesion
    # ------------------------------------------------------------------ #

    async def _chart_before_after(
        self, analysis: CapsuleAnalysisResult
    ) -> Optional[Path]:
        current = analysis.cohesion_score
        projected = analysis.projected_score_after_cleanup

        data_hash = self._hash_data({"current": current, "projected": projected})
        out_path = self.output_dir / "before_after.png"
        if self._is_unchanged(out_path, data_hash):
            return out_path

        fig, ax = plt.subplots(figsize=(5, 4), facecolor=_C_BG)
        ax.set_facecolor(_C_BG)

        bars = ax.bar(
            ["Current", "After cleanup"],
            [current, projected],
            color=[_C_ACCEPTABLE, _C_KEY],
            width=0.4,
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, val in zip(bars, [current, projected]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=_C_CHARCOAL,
            )

        ax.axhline(90, color=_C_GOLD, linewidth=1.2, linestyle="--",
                   alpha=0.7, label="Target 90")
        ax.set_ylim(0, 115)
        ax.set_ylabel("Cohesion score", fontsize=9, color=_C_CHARCOAL)
        ax.set_title("Cohesion: Current vs After Cleanup", fontsize=12,
                     color=_C_CHARCOAL, pad=12, fontweight="bold")
        ax.tick_params(colors=_C_CHARCOAL, labelsize=9)
        ax.legend(fontsize=8, framealpha=0.4)
        _style_ax(ax)

        # Delta annotation
        delta = projected - current
        if abs(delta) > 0.5:
            sign = "+" if delta > 0 else ""
            ax.annotate(
                f"{sign}{delta:.1f} pts",
                xy=(1, projected),
                xytext=(1.25, (current + projected) / 2),
                arrowprops=dict(arrowstyle="->", color=_C_GOLD, lw=1.2),
                fontsize=9,
                color=_C_GOLD,
                fontweight="bold",
            )

        fig.tight_layout()
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self._hash_cache[str(out_path)] = data_hash
        logger.info("Saved chart: %s", out_path)
        return out_path

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash_data(data: object) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def _is_unchanged(self, path: Path, data_hash: str) -> bool:
        """Return True if path exists and was already rendered from the same data."""
        if not path.exists():
            return False
        cached = self._hash_cache.get(str(path))
        return cached == data_hash


# ---------------------------------------------------------------------------
# Axis styling helper
# ---------------------------------------------------------------------------

def _style_ax(ax: "plt.Axes") -> None:
    """Apply shared quiet-luxury styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_C_CHARCOAL)
    ax.spines["bottom"].set_color(_C_CHARCOAL)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.yaxis.set_tick_params(width=0.5)
    ax.xaxis.set_tick_params(width=0.5)
