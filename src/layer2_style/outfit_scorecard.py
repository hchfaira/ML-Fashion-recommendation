"""
Outfit Scorecard
Comprehensive scoring system that aggregates all style scorers.

This module provides:
- OutfitScorecard: Complete scoring for an outfit with all metrics
- Visualization: Bar charts, radar charts, comparison charts
- Export: JSON export for scores and garments
- Configuration-driven scoring with multiple profiles
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from src.core.models import Garment, Outfit, UserContext
from src.core import get_logger

from .seven_point_rule import SevenPointRuleScorer
from .color_harmony import ColorHarmonyAnalyzer
from .proportion_scorer import ProportionScorer
from .volume_balance_scorer import VolumeBalanceScorer, BodyShape
from .three_color_scorer import ThreeColorScorer
from .pattern_mixing_scorer import PatternMixingScorer
from .design_principles_scorer import DesignPrinciplesScorer
from .total_style_scorer import TotalStyleScorer
from .creativity_scorer import CreativityScorer
from .season_color_harmony import ColorSeason
from .scoring_config_service import (
    ScoringConfigService, 
    ScoringProfile, 
    get_scoring_config_service
)

logger = get_logger(__name__)


class OutfitScorecard:
    """
    Complete scorecard for an outfit with all metrics.
    
    Aggregates 15+ style scorers into a single comprehensive score.
    Provides visualization and export capabilities.
    
    Supports configuration-driven scoring with multiple profiles:
    - default: Balanced scoring for everyday outfits
    - minimalist: Emphasizes color harmony and proportions
    - creative: Values rule-breaking and pattern mixing
    - business: Conservative professional scoring
    - casual: Relaxed everyday scoring
    
    Usage:
        # With default profile
        scorecard = OutfitScorecard(garments=my_garments)
        scorecard.calculate_all_scores(context)
        
        # With specific profile
        scorecard = OutfitScorecard(garments=my_garments, profile="creative")
        scorecard.calculate_all_scores(context)
        
        # Output
        print(scorecard.to_report())
        scorecard.plot_scores(save_path="outfit_scores.png")
        scorecard.save_json("outfit_result.json")
    """
    
    # Default score weights (fallback if no config loaded)
    DEFAULT_WEIGHTS = {
        "seven_point": 0.15,
        "color_harmony": 0.15,
        "three_color": 0.10,
        "proportion": 0.15,
        "volume_balance": 0.10,
        "pattern_mixing": 0.10,
        "design_principles": 0.15,
        "creativity": 0.10
    }
    
    def __init__(
        self, 
        garments: List[Garment],
        outfit: Optional[Outfit] = None,
        weights: Optional[Dict[str, float]] = None,
        profile: Optional[str] = None,
        config_service: Optional[ScoringConfigService] = None
    ):
        """
        Initialize the scorecard.
        
        Args:
            garments: List of Garment objects to score
            outfit: Optional Outfit object (for metadata)
            weights: Optional custom weights (overrides profile weights)
            profile: Scoring profile name ("default", "minimalist", "creative", etc.)
            config_service: Optional custom config service instance
        """
        self.outfit = outfit
        self.garments = garments
        self.scores: Dict[str, float] = {}
        self.details: Dict[str, Any] = {}
        
        # Load configuration
        self._config_service = config_service or get_scoring_config_service()
        self._profile_name = profile or "default"
        self._profile: ScoringProfile = self._config_service.get_profile(self._profile_name)
        
        # Use custom weights if provided, otherwise use profile weights
        if weights:
            self.weights = weights
        else:
            self.weights = self._profile.normalize_weights()
        
        # Initialize scorers
        self._seven_point_scorer = SevenPointRuleScorer()
        self._color_harmony = ColorHarmonyAnalyzer()
        self._proportion_scorer = ProportionScorer()
        self._volume_scorer = VolumeBalanceScorer()
        self._three_color_scorer = ThreeColorScorer()
        self._pattern_scorer = PatternMixingScorer()
        self._design_scorer = DesignPrinciplesScorer()
        self._total_scorer = TotalStyleScorer()
        self._creativity_scorer = CreativityScorer()
    
    def _is_criterion_enabled(self, criterion: str) -> bool:
        """Check if a criterion is enabled in the current profile."""
        if criterion in self._profile.criteria:
            return self._profile.criteria[criterion].enabled
        return True  # Default to enabled for unknown criteria
    
    def _should_show_in_summary(self, criterion: str) -> bool:
        """Check if a criterion should be shown in summary."""
        if criterion in self._profile.criteria:
            return self._profile.criteria[criterion].show_in_summary
        return True
    
    def _should_show_details(self, criterion: str) -> bool:
        """Check if a criterion's details should be shown."""
        if criterion in self._profile.criteria:
            return self._profile.criteria[criterion].show_details
        return True
    
    def _get_display_name(self, criterion: str) -> str:
        """Get the display name for a criterion."""
        if criterion in self._profile.criteria:
            return self._profile.criteria[criterion].display_name
        return criterion.replace("_", " ").title()
    
    def calculate_all_scores(self, context: Optional[UserContext] = None) -> "OutfitScorecard":
        """
        Calculate scores based on the configured profile.
        
        Only calculates enabled criteria from the active profile.
        Scores are weighted according to profile configuration.
        
        Args:
            context: Optional user context for personalized scoring
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Calculating scores for outfit with {len(self.garments)} items using profile '{self._profile_name}'")
        
        # Parse context values needed for multiple scorers
        body_shape = self._parse_body_shape(context)
        user_season = self._parse_color_season(context)
        season_sub = self._parse_season_sub(context)
        
        # Calculate 7-Point Rule (if enabled)
        if self._is_criterion_enabled("seven_point"):
            seven_point_result = self._seven_point_scorer.analyze_outfit(self.garments)
            self.scores["seven_point"] = seven_point_result.score
            self.details["seven_point"] = {
                "total_points": seven_point_result.total_points,
                "is_harmonious": seven_point_result.is_harmonious,
                "breakdown": seven_point_result.breakdown
            }
        
        # Calculate Color Harmony (if enabled)
        if self._is_criterion_enabled("color_harmony"):
            harmony_score = self._color_harmony.analyze_outfit_colors(self.garments)
            self.scores["color_harmony"] = harmony_score
            self.details["color_harmony"] = {
                "score": harmony_score
            }
        
        # Calculate Three Color Rule (if enabled)
        if self._is_criterion_enabled("three_color"):
            three_color_result = self._three_color_scorer.analyze_outfit(self.garments)
            self.scores["three_color"] = three_color_result.score
            self.details["three_color"] = {
                "color_count": three_color_result.unique_colors,
                "passes": three_color_result.within_limit,
                "colors": three_color_result.color_list
            }
        
        # Calculate Proportion Score (if enabled)
        if self._is_criterion_enabled("proportion"):
            proportion_result = self._proportion_scorer.analyze_outfit(self.garments)
            self.scores["proportion"] = proportion_result.score
            self.details["proportion"] = {
                "ratio": proportion_result.ratio.value if proportion_result.ratio else "unknown",
                "top_portion": proportion_result.top_portion,
                "bottom_portion": proportion_result.bottom_portion
            }
        
        # Calculate Volume Balance (if enabled)
        if self._is_criterion_enabled("volume_balance"):
            volume_result = self._volume_scorer.analyze_outfit(self.garments, body_shape)
            self.scores["volume_balance"] = volume_result.score
            self.details["volume_balance"] = {
                "top_volume": volume_result.top_volume.value if volume_result.top_volume else "unknown",
                "bottom_volume": volume_result.bottom_volume.value if volume_result.bottom_volume else "unknown",
                "balance_type": volume_result.balance_type
            }
        
        # Calculate Pattern Mixing (if enabled)
        if self._is_criterion_enabled("pattern_mixing"):
            pattern_result = self._pattern_scorer.analyze_outfit(self.garments)
            self.scores["pattern_mixing"] = pattern_result.overall_score
            self.details["pattern_mixing"] = {
                "pattern_count": pattern_result.pattern_count,
                "solid_count": pattern_result.solid_count,
                "has_scale_contrast": pattern_result.has_scale_contrast,
                "grade": pattern_result.grade
            }
        
        # Calculate Design Principles (if enabled)
        if self._is_criterion_enabled("design_principles"):
            design_result = self._design_scorer.analyze_outfit(self.garments, context)
            self.scores["design_principles"] = design_result.overall_score
            self.details["design_principles"] = {
                "grade": design_result.grade,
                "proportion": design_result.proportion.score,
                "balance": design_result.balance.score,
                "harmony": design_result.harmony.score,
                "emphasis": design_result.emphasis.score,
                "style": design_result.design_style.value
            }
        
        # Calculate Total Style Score (always calculated for internal use)
        total_result = self._total_scorer.analyze_outfit(self.garments, user_season, body_shape, season_sub=season_sub)
        self.scores["total_style"] = total_result.overall_score
        self.details["total_style"] = {
            "grade": total_result.grade,
            "seven_point": total_result.breakdown.seven_point_score,
            "color": total_result.breakdown.color_harmony_score,
            "proportion": total_result.breakdown.proportion_score,
            "volume": total_result.breakdown.volume_balance_score
        }
        
        # Calculate Creativity Score (if enabled)
        if self._is_criterion_enabled("creativity"):
            style_score = self.scores.get("total_style", 0.5)
            creativity_result = self._creativity_scorer.analyze_creativity(
                self.garments, 
                style_score=style_score,
                user_style_level="intermediate"
            )
            self.scores["creativity"] = creativity_result.creativity_score
            self.details["creativity"] = {
                "level": creativity_result.creativity_level.value,
                "rules_broken": len(creativity_result.rules_broken),
                "is_successful": creativity_result.is_successful_creativity,
                "fashion_forward": creativity_result.fashion_forward_score
            }
        
        # Calculate overall weighted score (only from enabled criteria)
        self.scores["overall"] = self._calculate_overall_score()
        
        logger.info(f"Scoring complete. Profile: {self._profile_name}, Overall: {self.scores['overall']:.1%}")
        
        return self
    
    def _parse_body_shape(self, context: Optional[UserContext]) -> Optional[BodyShape]:
        """Parse body shape from context."""
        if not context or not context.body_shape:
            return None
        try:
            return BodyShape(context.body_shape.lower())
        except (ValueError, AttributeError):
            return None
    
    def _parse_color_season(self, context: Optional[UserContext]) -> Optional[ColorSeason]:
        """Parse color season from context."""
        if not context or not context.color_season:
            return None
        try:
            return ColorSeason(context.color_season.upper())
        except (ValueError, AttributeError):
            return None
    
    def _parse_season_sub(self, context: Optional[UserContext]) -> Optional[str]:
        """Parse 12-sub-season from context (passthrough string)."""
        if not context:
            return None
        return getattr(context, "season_sub", None)
    
    def _calculate_overall_score(self) -> float:
        """Calculate weighted overall score."""
        total = sum(
            self.scores.get(key, 0) * weight 
            for key, weight in self.weights.items()
        )
        return min(1.0, max(0.0, total))
    
    def get_grade(self, score: Optional[float] = None) -> str:
        """
        Convert score to letter grade.
        
        Args:
            score: Score to grade (defaults to overall score)
            
        Returns:
            Letter grade (A+ to F)
        """
        if score is None:
            score = self.scores.get("overall", 0)
        return self._get_grade(score)
    
    def _get_grade(self, score: float) -> str:
        """Internal grade calculation."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        elif score >= 0.4:
            return "D"
        else:
            return "F"
    
    # ==================== Export Methods ====================
    
    def garment_to_dict(self, garment: Garment) -> Dict[str, Any]:
        """Convert a Garment object to a JSON-serializable dictionary."""
        attrs = garment.attributes
        
        return {
            "id": str(garment.id),
            "image_url": garment.image_url,
            "image_path": garment.image_path,
            "category": attrs.category.value if attrs.category else None,
            "subcategory": attrs.subcategory,
            "color": {
                "primary": attrs.color.primary if attrs.color else None,
                "secondary": attrs.color.secondary if attrs.color else None,
                "accent": attrs.color.accent if attrs.color else None,
                "hex_codes": attrs.color.hex_codes if attrs.color else [],
            } if attrs.color else None,
            "pattern": {
                "type": attrs.pattern.type if attrs.pattern else None,
                "scale": attrs.pattern.scale if attrs.pattern else None,
                "density": attrs.pattern.density if attrs.pattern else None,
                "pattern_colors": attrs.pattern.pattern_colors if attrs.pattern else [],
            } if attrs.pattern else None,
            "material": {
                "primary": attrs.material.primary if attrs.material else None,
                "secondary": attrs.material.secondary if attrs.material else None,
                "texture": attrs.material.texture if attrs.material else None,
            } if attrs.material else None,
            "formality_level": attrs.formality_level.value if attrs.formality_level else None,
            "season_suitable": [s.value for s in attrs.season_suitable] if attrs.season_suitable else [],
            "silhouette": attrs.silhouette,
            "fit": attrs.fit,
            "style_tags": attrs.style_tags or [],
            "neckline": attrs.neckline.value if attrs.neckline else None,
            "length_type": attrs.length_type.value if attrs.length_type else None,
            "waist_rise": attrs.waist_rise.value if attrs.waist_rise else None,
            "closure": attrs.closure.value if attrs.closure else None,
            "metadata": garment.metadata or {},
        }
    
    def get_filtered_scores(self, include_all: bool = False) -> Dict[str, Any]:
        """
        Get scores filtered by profile configuration.
        
        Args:
            include_all: If True, include all scores regardless of config
            
        Returns:
            Dictionary with filtered scores based on show_in_summary setting
        """
        if include_all:
            return {key: round(value, 4) for key, value in self.scores.items()}
        
        filtered = {}
        for key, value in self.scores.items():
            if key == "overall" or key == "total_style":
                filtered[key] = round(value, 4)
            elif self._should_show_in_summary(key):
                filtered[key] = round(value, 4)
        
        return filtered
    
    def get_filtered_details(self, include_all: bool = False) -> Dict[str, Any]:
        """
        Get details filtered by profile configuration.
        
        Args:
            include_all: If True, include all details regardless of config
            
        Returns:
            Dictionary with filtered details based on show_details setting
        """
        if include_all:
            return self.details
        
        filtered = {}
        for key, value in self.details.items():
            if key == "total_style":
                filtered[key] = value
            elif self._should_show_details(key):
                filtered[key] = value
        
        return filtered
    
    def to_json(self, include_all_scores: bool = False) -> Dict[str, Any]:
        """
        Convert the scorecard to a JSON-serializable dictionary.
        
        Scores and details are filtered based on the profile configuration
        unless include_all_scores is True.
        
        Args:
            include_all_scores: If True, include all scores regardless of config
            
        Returns:
            JSON-serializable dictionary
        """
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_items": len(self.garments),
                "profile": self._profile_name,
                "profile_name": self._profile.name,
            },
            "garments": [self.garment_to_dict(g) for g in self.garments],
            "scores": self.get_filtered_scores(include_all_scores),
            "details": self.get_filtered_details(include_all_scores),
            "summary": {
                "overall_score": round(self.scores.get("overall", 0), 2),
                "overall_percentage": f"{self.scores.get('overall', 0):.0%}",
                "grade": self._get_grade(self.scores.get("overall", 0)),
                "categories": [g.attributes.category.value for g in self.garments],
                "primary_colors": [g.attributes.color.primary for g in self.garments if g.attributes.color],
            },
            "criteria_config": {
                key: {
                    "display_name": self._get_display_name(key),
                    "weight": self.weights.get(key, 0),
                    "enabled": self._is_criterion_enabled(key)
                }
                for key in self.scores.keys() if key not in ["overall", "total_style"]
            }
        }
    
    def save_json(self, filepath: str, include_all_scores: bool = False) -> None:
        """Save the scorecard data to a JSON file."""
        data = self.to_json(include_all_scores)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON saved to: {filepath}")
    
    def to_report(self, include_all: bool = False) -> str:
        """
        Generate a human-readable report.
        
        Args:
            include_all: If True, include all scores regardless of config
        """
        lines = [
            "=" * 60,
            f"OUTFIT SCORECARD - Profile: {self._profile.name}",
            "=" * 60,
            "",
            f"Number of items: {len(self.garments)}",
            f"Categories: {', '.join(g.attributes.category.value for g in self.garments)}",
            "",
            "-" * 40,
            "SCORES (enabled criteria only)",
            "-" * 40,
        ]
        
        score_labels = {
            "seven_point": "7-Point Rule",
            "color_harmony": "Color Harmony",
            "three_color": "3-Color Rule",
            "proportion": "Proportions",
            "volume_balance": "Volume Balance",
            "pattern_mixing": "Pattern Mixing",
            "design_principles": "Design Principles",
            "creativity": "Creativity",
        }
        
        # Show only enabled scores (or all if include_all)
        for key, label in score_labels.items():
            if include_all or self._should_show_in_summary(key):
                if key in self.scores:
                    score = self.scores[key]
                    weight = self.weights.get(key, 0)
                    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                    lines.append(f"{label:20} [{bar}] {score:.1%} (weight: {weight:.0%})")
        
        # Always show overall
        overall = self.scores.get("overall", 0)
        bar = "█" * int(overall * 20) + "░" * (20 - int(overall * 20))
        lines.append(f"{'OVERALL SCORE':20} [{bar}] {overall:.1%}")
        
        lines.append("")
        lines.append("-" * 40)
        lines.append("DETAILS")
        lines.append("-" * 40)
        
        # Add key details (only for enabled criteria with show_details=True)
        if (include_all or self._should_show_details("seven_point")) and "seven_point" in self.details:
            sp = self.details["seven_point"]
            lines.append(f"7-Point Total: {sp['total_points']} pts (target: 7-10)")
            lines.append(f"  Harmonious: {'✓' if sp['is_harmonious'] else '✗'}")
        
        if (include_all or self._should_show_details("three_color")) and "three_color" in self.details:
            tc = self.details["three_color"]
            lines.append(f"Colors Used: {tc['color_count']} ({', '.join(tc['colors'][:5])})")
            lines.append(f"  Rule Passed: {'✓' if tc['passes'] else '✗'}")
        
        if (include_all or self._should_show_details("creativity")) and "creativity" in self.details:
            cr = self.details["creativity"]
            lines.append(f"Creativity Level: {cr['level']}")
            lines.append(f"  Rules Broken: {cr['rules_broken']}")
            lines.append(f"  Successful: {'✓' if cr['is_successful'] else '✗'}")
        
        if (include_all or self._should_show_details("pattern_mixing")) and "pattern_mixing" in self.details:
            pm = self.details["pattern_mixing"]
            lines.append(f"Patterns: {pm['pattern_count']} patterns, {pm['solid_count']} solids")
            lines.append(f"  Grade: {pm['grade']}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    # ==================== Visualization Methods ====================
    
    def plot_scores(
        self, 
        title: str = "Outfit Score Analysis", 
        save_path: Optional[str] = None, 
        show: bool = True,
        show_only_enabled: bool = True
    ) -> None:
        """
        Generate a visual bar chart of scores.
        
        Args:
            title: Chart title
            save_path: Optional path to save the chart image
            show: Whether to display the chart (set False for headless testing)
            show_only_enabled: If True, only show enabled criteria from profile
        """
        # Import here to avoid dependency issues
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Score labels and their display names
        score_config = {
            "seven_point": ("7-Point Rule", "#3498db"),
            "color_harmony": ("Color Harmony", "#9b59b6"),
            "three_color": ("3-Color Rule", "#1abc9c"),
            "proportion": ("Proportions", "#e74c3c"),
            "volume_balance": ("Volume Balance", "#f39c12"),
            "pattern_mixing": ("Pattern Mixing", "#2ecc71"),
            "design_principles": ("Design Principles", "#e91e63"),
            "creativity": ("Creativity", "#00bcd4"),
            "overall": ("OVERALL", "#2c3e50"),
        }
        
        # Filter only available scores
        labels = []
        values = []
        colors = []
        
        for key, (label, color) in score_config.items():
            if key in self.scores:
                labels.append(label)
                values.append(self.scores[key])
                colors.append(color)
        
        if not values:
            logger.warning("No scores to plot!")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor='white', linewidth=1)
        
        # Add score values on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            text_color = 'white' if value > 0.5 else 'black'
            x_pos = value + 0.02 if value <= 0.85 else value - 0.08
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{value:.0%}', va='center', ha='left' if value <= 0.85 else 'right',
                    fontsize=11, fontweight='bold', color=text_color if value > 0.85 else 'black')
        
        # Customize chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add gridlines
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add threshold lines
        ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Good (70%)')
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Average (50%)')
        ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Poor (30%)')
        
        # Add legend for threshold lines
        ax.legend(loc='lower right', fontsize=9)
        
        # Add color-coded background regions
        ax.axvspan(0, 0.3, alpha=0.1, color='red')
        ax.axvspan(0.3, 0.5, alpha=0.1, color='orange')
        ax.axvspan(0.5, 0.7, alpha=0.1, color='yellow')
        ax.axvspan(0.7, 1.1, alpha=0.1, color='green')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Chart saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_radar(
        self, 
        title: str = "Style Profile", 
        save_path: Optional[str] = None, 
        show: bool = True
    ) -> None:
        """
        Generate a radar/spider chart of scores for holistic view.
        
        Args:
            title: Chart title
            save_path: Optional path to save the chart image
            show: Whether to display the chart
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Score labels (excluding overall)
        score_config = {
            "seven_point": "7-Point",
            "color_harmony": "Color",
            "three_color": "3-Color",
            "proportion": "Proportion",
            "volume_balance": "Volume",
            "pattern_mixing": "Pattern",
            "design_principles": "Design",
            "creativity": "Creativity",
        }
        
        # Filter available scores
        labels = []
        values = []
        for key, label in score_config.items():
            if key in self.scores and key != "overall":
                labels.append(label)
                values.append(self.scores[key])
        
        if len(values) < 3:
            logger.warning("Need at least 3 scores for radar chart!")
            return
        
        # Number of variables
        num_vars = len(labels)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Complete the loop
        values += values[:1]
        angles += angles[:1]
        labels += labels[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Draw the outline
        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1], fontsize=11)
        
        # Set radial limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add overall score in center
        if "overall" in self.scores:
            overall = self.scores["overall"]
            ax.annotate(f'Overall\n{overall:.0%}', xy=(0, 0), ha='center', va='center',
                       fontsize=16, fontweight='bold', color='#2c3e50')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Radar chart saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(
        self, 
        other_scorecards: List['OutfitScorecard'], 
        names: List[str], 
        save_path: Optional[str] = None, 
        show: bool = True
    ) -> None:
        """
        Compare multiple outfits side by side.
        
        Args:
            other_scorecards: List of other OutfitScorecard instances to compare
            names: Names for each outfit (including this one)
            save_path: Optional path to save the chart image
            show: Whether to display the chart
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        all_scorecards = [self] + other_scorecards
        
        if len(names) != len(all_scorecards):
            logger.error(f"{len(names)} names provided but {len(all_scorecards)} scorecards")
            return
        
        # Score labels
        score_keys = ["seven_point", "color_harmony", "three_color", "proportion", 
                      "volume_balance", "pattern_mixing", "design_principles", 
                      "creativity", "overall"]
        score_labels = ["7-Point", "Color", "3-Color", "Proportion", "Volume", 
                        "Pattern", "Design", "Creativity", "OVERALL"]
        
        # Colors for each outfit
        outfit_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(score_labels))
        width = 0.8 / len(all_scorecards)
        
        # Plot each outfit's scores
        for i, (scorecard, name) in enumerate(zip(all_scorecards, names)):
            values = [scorecard.scores.get(key, 0) for key in score_keys]
            offset = (i - len(all_scorecards) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=name, 
                         color=outfit_colors[i % len(outfit_colors)], alpha=0.8)
        
        # Customize chart
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Outfit Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(score_labels, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 1.1)
        
        # Add threshold lines
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison chart saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
