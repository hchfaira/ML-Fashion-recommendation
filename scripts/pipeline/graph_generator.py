"""
Graph Generator Module
======================

Generates visual reports (bar charts, pie charts, radar charts) for pipeline results.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

# Type hints
from src.core.models import Garment


# Color mapping for pie charts
COLOR_MAP = {
    'black': '#000000', 'white': '#FFFFFF', 'gray': '#808080', 'grey': '#808080',
    'red': '#FF0000', 'blue': '#0000FF', 'navy': '#000080', 'green': '#008000',
    'yellow': '#FFFF00', 'orange': '#FFA500', 'purple': '#800080', 'pink': '#FFC0CB',
    'brown': '#8B4513', 'beige': '#F5F5DC', 'cream': '#FFFDD0', 'tan': '#D2B48C',
    'lavender': '#E6E6FA', 'lilac': '#C8A2C8', 'mint': '#98FF98', 'coral': '#FF7F50',
    'turquoise': '#40E0D0', 'teal': '#008080', 'olive': '#808000', 'maroon': '#800000',
    'lightblue': '#ADD8E6', 'skyblue': '#87CEEB', 'salmon': '#FA8072', 'khaki': '#F0E68C',
}


def is_matplotlib_available() -> bool:
    """Check if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
        return True
    except ImportError:
        return False


def generate_score_graph(
    candidates: list, 
    profile: str, 
    output_dir: Path
) -> Optional[Path]:
    """
    Generate a bar chart comparing all outfit scores.
    
    Args:
        candidates: List of OutfitCandidate objects
        profile: Scoring profile name
        output_dir: Directory to save the graph
        
    Returns:
        Path to saved graph, or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("   ⚠️  matplotlib not installed, skipping graph generation")
        print("      Install with: pip install matplotlib")
        return None
    
    sorted_candidates = sorted(candidates, key=lambda c: c.overall_score, reverse=True)
    
    # Limit to top 20 for readability
    display_candidates = sorted_candidates[:20]
    
    # Prepare data
    names = []
    scores = []
    colors = []
    
    for i, candidate in enumerate(display_candidates):
        # Create short name from garment colors/types
        parts = []
        for g in candidate.garments:
            color = g.attributes.color.primary if g.attributes.color else "?"
            cat = g.attributes.subcategory or g.attributes.category.value
            parts.append(f"{color[:3]}-{cat[:4]}")
        name = " | ".join(parts[:3])
        names.append(f"#{i+1}: {name}")
        scores.append(candidate.overall_score * 100)
        
        # Color based on score
        if candidate.overall_score >= 0.8:
            colors.append('#2ecc71')  # Green
        elif candidate.overall_score >= 0.6:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(display_candidates) * 0.4)))
    
    # Horizontal bar chart
    y_pos = range(len(names))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add score labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
               f'{score:.1f}%', va='center', fontsize=9)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Overall Score (%)', fontsize=11)
    ax.set_title(
        f'Outfit Combinations Score Comparison\n'
        f'Profile: {profile.upper()} | Total: {len(candidates)} combinations', 
        fontsize=13, fontweight='bold'
    )
    ax.set_xlim(0, 110)
    
    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color='#2ecc71', label='Excellent (≥80%)'),
        mpatches.Patch(color='#f39c12', label='Good (60-79%)'),
        mpatches.Patch(color='#e74c3c', label='Needs work (<60%)')
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    filepath = output_dir / f"scores_comparison_{profile}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Saved score comparison graph to {filepath.name}")
    return filepath


def generate_criteria_breakdown_graph(
    candidates: list, 
    profile: str, 
    output_dir: Path
) -> Optional[Path]:
    """
    Generate a stacked bar chart showing criteria breakdown for top outfits.
    
    Args:
        candidates: List of OutfitCandidate objects
        profile: Scoring profile name
        output_dir: Directory to save the graph
        
    Returns:
        Path to saved graph, or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None
    
    sorted_candidates = sorted(candidates, key=lambda c: c.overall_score, reverse=True)
    top_candidates = sorted_candidates[:10]
    
    # Get all criteria
    all_criteria = set()
    for c in top_candidates:
        if hasattr(c, 'scorecard'):
            all_criteria.update(c.scorecard.get_filtered_scores().keys())
    
    # Remove 'overall' from criteria
    all_criteria.discard('overall')
    criteria_list = sorted(all_criteria)
    
    if not criteria_list:
        return None
    
    # Prepare data
    outfit_names = []
    criteria_scores = {crit: [] for crit in criteria_list}
    
    for i, candidate in enumerate(top_candidates):
        # Short name
        parts = []
        for g in candidate.garments[:2]:
            color = g.attributes.color.primary[:3] if g.attributes.color and g.attributes.color.primary else "?"
            parts.append(color)
        outfit_names.append(f"#{i+1} ({'+'.join(parts)})")
        
        scores = candidate.scorecard.get_filtered_scores() if hasattr(candidate, 'scorecard') else {}
        for crit in criteria_list:
            criteria_scores[crit].append(scores.get(crit, 0) * 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(outfit_names))
    width = 0.8 / len(criteria_list)
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(criteria_list)))
    
    for i, (crit, values) in enumerate(criteria_scores.items()):
        offset = (i - len(criteria_list)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=crit.replace('_', ' ').title(), color=colors[i])
    
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_xlabel('Outfit Combination', fontsize=11)
    ax.set_title(
        f'Criteria Breakdown - Top 10 Outfits\nProfile: {profile.upper()}', 
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(outfit_names, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save
    filepath = output_dir / f"criteria_breakdown_{profile}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Saved criteria breakdown graph to {filepath.name}")
    return filepath


def generate_color_distribution_graph(
    garments: List[Garment], 
    output_dir: Path
) -> Optional[Path]:
    """
    Generate a pie chart showing color distribution in wardrobe.
    
    Args:
        garments: List of Garment objects
        output_dir: Directory to save the graph
        
    Returns:
        Path to saved graph, or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    
    # Count colors
    color_counts = {}
    for g in garments:
        if g.attributes.color and g.attributes.color.primary:
            color = g.attributes.color.primary.lower()
            color_counts[color] = color_counts.get(color, 0) + 1
    
    if not color_counts:
        return None
    
    # Sort by count
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [c[0].title() for c in sorted_colors]
    sizes = [c[1] for c in sorted_colors]
    
    pie_colors = [COLOR_MAP.get(c[0].lower(), '#CCCCCC') for c in sorted_colors]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.0f%%',
        colors=pie_colors, startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Style
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title('Wardrobe Color Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    filepath = output_dir / "color_distribution.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   📊 Saved color distribution graph to {filepath.name}")
    return filepath


# ---------------------------------------------------------------------------
# Score radar / spider chart for the best outfit
# ---------------------------------------------------------------------------

# Human-readable labels for every scoring dimension
_SCORE_LABELS: Dict[str, str] = {
    "seven_point":        "7-Point Rule",
    "color_harmony":      "Color Harmony",
    "three_color":        "3-Color Rule",
    "proportion":         "Proportion",
    "volume_balance":     "Volume Balance",
    "pattern_mixing":     "Pattern Mixing",
    "design_principles":  "Design Principles",
    "total_style":        "Total Style",
    "creativity":         "Creativity",
    # context / occasion scores that may appear via context engine
    "occasion":           "Occasion Fit",
    "weather":            "Weather Match",
    "morphology":         "Morphology",
    "color_season":       "Color Season",
    "activity":           "Activity Fit",
}


def generate_best_outfit_radar(
    candidate: Any,
    profile: str,
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate a radar / spider chart showing every scoring dimension for the
    single best outfit candidate.

    Dimensions shown: style sub-scores + creativity + any context scores.

    Args:
        candidate: The top OutfitCandidate object (must have .scorecard).
        profile: Active scoring profile name.
        output_dir: Directory to save the chart.

    Returns:
        Path to the saved PNG, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        return None

    if not hasattr(candidate, "scorecard") or candidate.scorecard is None:
        return None

    scores: Dict[str, float] = candidate.scorecard.scores or {}
    # Drop the synthetic 'overall' key — it's the aggregate
    dim_scores = {k: v for k, v in scores.items() if k != "overall" and isinstance(v, float)}

    if not dim_scores:
        return None

    labels = [_SCORE_LABELS.get(k, k.replace("_", " ").title()) for k in dim_scores]
    values = list(dim_scores.values())
    N = len(values)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Close the polygon
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]
    labels_closed = labels  # labels stay N

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    # Fill area
    ax.fill(angles_closed[:-1], values_closed[:-1], alpha=0.25, color="#3498db")
    # Draw outline
    ax.plot(angles_closed, values_closed, color="#2980b9", linewidth=2)
    # Draw individual score points
    ax.scatter(angles, values, s=60, color="#e74c3c", zorder=5)

    # Annotate each point with its percentage
    for angle, val, lbl in zip(angles, values, labels):
        offset = 0.08
        ax.text(
            angle,
            min(val + offset, 1.05),
            f"{val:.0%}",
            ha="center",
            va="center",
            fontsize=8,
            color="#2c3e50",
            fontweight="bold",
        )

    # Axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7, color="grey")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)

    overall = candidate.overall_score
    garment_names = " + ".join(
        (g.attributes.subcategory or g.attributes.category.value).title()
        for g in candidate.garments[:3]
    )
    ax.set_title(
        f"Best Outfit — Score Breakdown\n"
        f"{garment_names}\n"
        f"Overall: {overall:.1%}  |  Profile: {profile.upper()}",
        fontsize=11,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    filepath = output_dir / f"best_outfit_radar_{profile}.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📊 Saved best-outfit radar chart to {filepath.name}")
    return filepath


# ---------------------------------------------------------------------------
# Full scoring dashboard: style + creativity + context per candidate
# ---------------------------------------------------------------------------

def generate_scoring_dashboard(
    candidates: list,
    profile: str,
    output_dir: Path,
    top_n: int = 8,
) -> Optional[Path]:
    """
    Generate a multi-panel dashboard with:
      - Top panel: overall score comparison (horizontal bars)
      - Middle panel: style sub-score heatmap (top N outfits × dimensions)
      - Bottom panel: creativity + context scores side-by-side

    Args:
        candidates: List of OutfitCandidate objects.
        profile: Active scoring profile name.
        output_dir: Directory to save the chart.
        top_n: How many candidates to show (default 8).

    Returns:
        Path to the saved PNG, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        return None

    sorted_cands = sorted(candidates, key=lambda c: c.overall_score, reverse=True)
    top = sorted_cands[:top_n]
    if not top:
        return None

    # ── Collect scores ──────────────────────────────────────────────────
    # Style dimensions (layout order)
    style_dims = [
        "seven_point", "color_harmony", "three_color",
        "proportion", "volume_balance", "pattern_mixing",
        "design_principles", "total_style",
    ]
    extra_dims = ["creativity", "occasion", "weather", "morphology",
                  "color_season", "activity"]

    def _outfit_label(i: int, c: Any) -> str:
        parts = [
            (g.attributes.color.primary[:3] if g.attributes.color else "?")
            for g in c.garments[:3]
        ]
        return f"#{i+1} {'+'.join(parts)} ({c.overall_score:.0%})"

    outfit_labels = [_outfit_label(i, c) for i, c in enumerate(top)]

    def _scores(c: Any) -> Dict[str, float]:
        if hasattr(c, "scorecard") and c.scorecard:
            return c.scorecard.scores or {}
        return {}

    # Build heatmap matrix: rows=outfits, cols=present style dims
    present_style = [d for d in style_dims
                     if any(d in _scores(c) for c in top)]
    present_extra = [d for d in extra_dims
                     if any(d in _scores(c) for c in top)]

    # ── Figure layout ───────────────────────────────────────────────────
    n_rows = 3 if present_extra else 2
    fig_height = 4 + len(top) * 0.55 + (3 if present_extra else 0)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, max(10, fig_height)),
        gridspec_kw={"height_ratios": ([2] + [len(top) * 0.55 + 1] * (n_rows - 1))},
    )
    if n_rows == 2:
        ax_bar, ax_heat = axes
        ax_extra = None
    else:
        ax_bar, ax_heat, ax_extra = axes

    fig.suptitle(
        f"Scoring Dashboard — Profile: {profile.upper()}  |  Top {len(top)} outfits",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # ── Panel 1: overall bar chart ──────────────────────────────────────
    overall_scores = [c.overall_score * 100 for c in top]
    bar_colors = [
        "#2ecc71" if s >= 80 else "#f39c12" if s >= 60 else "#e74c3c"
        for s in overall_scores
    ]
    y_pos = range(len(top))
    bars = ax_bar.barh(y_pos, overall_scores, color=bar_colors,
                        edgecolor="white", linewidth=0.5)
    for bar, score in zip(bars, overall_scores):
        ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%", va="center", fontsize=9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(outfit_labels, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Overall Score (%)")
    ax_bar.set_title("Overall Score Comparison", fontsize=11, fontweight="bold")
    ax_bar.set_xlim(0, 115)
    ax_bar.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax_bar.set_axisbelow(True)

    # ── Panel 2: style sub-score heatmap ────────────────────────────────
    if present_style:
        matrix = np.array([
            [_scores(c).get(d, np.nan) for d in present_style]
            for c in top
        ])
        col_labels = [_SCORE_LABELS.get(d, d.replace("_", " ").title())
                      for d in present_style]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rg", ["#e74c3c", "#f39c12", "#2ecc71"]
        )
        im = ax_heat.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax_heat.set_xticks(range(len(col_labels)))
        ax_heat.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
        ax_heat.set_yticks(range(len(top)))
        ax_heat.set_yticklabels(outfit_labels, fontsize=9)
        ax_heat.set_title("Style Sub-Score Heatmap", fontsize=11, fontweight="bold")
        # Annotate cells
        for ri in range(matrix.shape[0]):
            for ci in range(matrix.shape[1]):
                val = matrix[ri, ci]
                if not np.isnan(val):
                    ax_heat.text(ci, ri, f"{val:.0%}", ha="center", va="center",
                                 fontsize=8,
                                 color="white" if val < 0.45 or val > 0.85 else "black")
        plt.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02,
                     label="Score (0→1)")
    else:
        ax_heat.set_visible(False)

    # ── Panel 3: creativity + context scores ────────────────────────────
    if ax_extra is not None and present_extra:
        extra_matrix = np.array([
            [_scores(c).get(d, np.nan) for d in present_extra]
            for c in top
        ])
        extra_labels = [_SCORE_LABELS.get(d, d.replace("_", " ").title())
                        for d in present_extra]
        x = np.arange(len(top))
        width = 0.8 / len(present_extra)
        palette = plt.cm.Set2(np.linspace(0, 1, len(present_extra)))
        for j, (dim, lbl) in enumerate(zip(present_extra, extra_labels)):
            vals = extra_matrix[:, j]
            offset = (j - len(present_extra) / 2 + 0.5) * width
            bars_e = ax_extra.bar(x + offset, np.nan_to_num(vals) * 100,
                                   width, label=lbl, color=palette[j])
        ax_extra.set_xticks(x)
        ax_extra.set_xticklabels(outfit_labels, rotation=30, ha="right", fontsize=8)
        ax_extra.set_ylabel("Score (%)")
        ax_extra.set_ylim(0, 115)
        ax_extra.set_title("Creativity & Context Scores", fontsize=11, fontweight="bold")
        ax_extra.legend(loc="upper right", fontsize=8, ncol=2)
        ax_extra.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax_extra.set_axisbelow(True)
    elif ax_extra is not None:
        ax_extra.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    filepath = output_dir / f"scoring_dashboard_{profile}.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📊 Saved scoring dashboard to {filepath.name}")
    return filepath
