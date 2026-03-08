"""
Graph Generator Module
======================

Generates visual reports (bar charts, pie charts) for pipeline results.
"""

from pathlib import Path
from typing import List, Optional, Dict

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
