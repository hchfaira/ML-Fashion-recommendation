"""
Proportion Harmonizer
=====================

Analyzes body proportions and recommends styling choices
that create visual harmony and balance.
"""

import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from config import get_config
from src.core.models import Garment

logger = logging.getLogger(__name__)


@dataclass
class ProportionAnalysis:
    """Analysis of body proportions."""
    torso_proportion: str  # "short", "average", "long"
    leg_proportion: str  # "short", "average", "long"
    frame_size: str  # "small", "medium", "large"
    balance_needed: List[str] = field(default_factory=list)
    recommended_silhouettes: List[str] = field(default_factory=list)
    avoid_silhouettes: List[str] = field(default_factory=list)


@dataclass
class ProportionScore:
    """Score for how well an outfit harmonizes with body proportions."""
    score: float  # 0.0 to 1.0
    harmony_level: str  # "excellent", "good", "fair", "poor"
    positive_factors: List[str] = field(default_factory=list)
    negative_factors: List[str] = field(default_factory=list)
    styling_tips: List[str] = field(default_factory=list)


class ProportionHarmonizer:
    """
    Analyzes body proportions and scores outfits based on
    how well they create visual balance and harmony.
    
    Uses torso/leg ratios, frame size, and body measurements
    to recommend flattering silhouettes.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize proportion harmonizer.
        
        Args:
            config_path: Optional path to body profile config
        """
        self.config = get_config()
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[Path] = None):
        """Load proportion harmonizer configuration."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                full_config = json.load(f)
        else:
            full_config = self.config.get_data("body_profile_config", default={})
        
        proportion_config = full_config.get("proportion_harmonizer", {})
        
        self.proportion_rules = {
            "short_torso": proportion_config.get("short_torso", {
                "recommended": ["high_waist", "cropped_tops", "v_neck", "vertical_stripes"],
                "avoid": ["long_tops", "low_rise_pants", "dropped_waist"],
                "tips": ["Wear high-waisted bottoms to elongate torso visually"]
            }),
            "short_legs": proportion_config.get("short_legs", {
                "recommended": ["high_waist", "vertical_lines", "monochrome", "pointed_shoes"],
                "avoid": ["cropped_pants", "ankle_straps", "horizontal_cuts"],
                "tips": ["Choose high-waisted pants in same color as shoes"]
            }),
            "long_torso": proportion_config.get("long_torso", {
                "recommended": ["low_rise", "belts_at_hip", "layered_tops", "horizontal_lines"],
                "avoid": ["high_waist", "tucked_in_tops"],
                "tips": ["Balance with longer tops and mid-rise bottoms"]
            }),
            "long_legs": proportion_config.get("long_legs", {
                "recommended": ["any_waist_height", "cropped_pants", "statement_shoes"],
                "avoid": [],  # Long legs are versatile
                "tips": ["You have flexibility with pant lengths and shoe styles"]
            })
        }
        
        self.frame_size_matching = proportion_config.get("frame_size_matching", {
            "small": {
                "preferred_weights": ["lightweight", "medium"],
                "avoid": ["heavy", "bulky"],
                "scale_recommendations": ["delicate_patterns", "fine_details"]
            },
            "medium": {
                "preferred_weights": ["lightweight", "medium", "medium_heavy"],
                "avoid": [],
                "scale_recommendations": ["medium_patterns", "balanced_proportions"]
            },
            "large": {
                "preferred_weights": ["medium", "heavy", "structured"],
                "avoid": ["too_delicate", "very_small_patterns"],
                "scale_recommendations": ["bold_patterns", "structured_pieces"]
            }
        })
        
        logger.info("ProportionHarmonizer configuration loaded")
    
    def analyze_proportions(
        self,
        torso_ratio: Optional[float] = None,
        leg_ratio: Optional[float] = None,
        frame_size: Optional[str] = None,
        shoulder_hip_ratio: Optional[float] = None,
    ) -> ProportionAnalysis:
        """
        Analyze body proportions and determine styling needs.
        
        Args:
            torso_ratio: Torso to height ratio (typical ~0.30)
            leg_ratio: Leg to height ratio (typical ~0.47)
            frame_size: "small", "medium", or "large"
            shoulder_hip_ratio: Shoulder to hip width ratio
            
        Returns:
            ProportionAnalysis with recommendations
        """
        # Classify torso proportion
        torso_prop = self._classify_torso(torso_ratio)
        
        # Classify leg proportion
        leg_prop = self._classify_legs(leg_ratio)
        
        # Default frame size if not provided
        frame = frame_size or "medium"
        
        # Determine what balance is needed
        balance_needed = []
        if torso_prop == "short":
            balance_needed.append("elongate_torso")
        elif torso_prop == "long":
            balance_needed.append("balance_torso")
        
        if leg_prop == "short":
            balance_needed.append("elongate_legs")
        elif leg_prop == "long":
            balance_needed.append("balance_legs")
        
        # Get recommendations based on proportions
        recommended = set()
        avoid = set()
        
        torso_key = f"{torso_prop}_torso"
        if torso_key in self.proportion_rules:
            rules = self.proportion_rules[torso_key]
            recommended.update(rules.get("recommended", []))
            avoid.update(rules.get("avoid", []))
        
        leg_key = f"{leg_prop}_legs"
        if leg_key in self.proportion_rules:
            rules = self.proportion_rules[leg_key]
            recommended.update(rules.get("recommended", []))
            avoid.update(rules.get("avoid", []))
        
        return ProportionAnalysis(
            torso_proportion=torso_prop,
            leg_proportion=leg_prop,
            frame_size=frame,
            balance_needed=balance_needed,
            recommended_silhouettes=list(recommended),
            avoid_silhouettes=list(avoid)
        )
    
    def score_outfit_harmony(
        self,
        garments: List[Garment],
        proportion_analysis: ProportionAnalysis,
    ) -> ProportionScore:
        """
        Score how well an outfit harmonizes with body proportions.
        
        Args:
            garments: List of garments in outfit
            proportion_analysis: User's proportion analysis
            
        Returns:
            ProportionScore with score and details
        """
        positive_factors = []
        negative_factors = []
        
        # Extract outfit features
        outfit_features = self._extract_outfit_features(garments)
        
        # Check recommended silhouettes
        recommended = set(proportion_analysis.recommended_silhouettes)
        avoid = set(proportion_analysis.avoid_silhouettes)
        
        score_adjustments = 0.0
        
        for feature in outfit_features:
            if feature in recommended:
                positive_factors.append(f"Good choice: {feature.replace('_', ' ')}")
                score_adjustments += 0.1
            elif feature in avoid:
                negative_factors.append(f"May not flatter: {feature.replace('_', ' ')}")
                score_adjustments -= 0.15
        
        # Check frame size matching
        frame_rules = self.frame_size_matching.get(proportion_analysis.frame_size, {})
        preferred_weights = frame_rules.get("preferred_weights", [])
        avoid_weights = frame_rules.get("avoid", [])
        
        for garment in garments:
            weight = self._get_garment_weight(garment)
            if weight:
                if weight in preferred_weights:
                    positive_factors.append(f"Good fabric weight: {weight}")
                    score_adjustments += 0.05
                elif weight in avoid_weights:
                    negative_factors.append(f"Fabric weight may overwhelm/underwhelm frame")
                    score_adjustments -= 0.1
        
        # Calculate final score (base of 0.7)
        base_score = 0.7
        final_score = max(0.0, min(1.0, base_score + score_adjustments))
        
        # Get styling tips
        tips = self._get_styling_tips(proportion_analysis)
        
        # Classify harmony level
        harmony_level = self._classify_harmony(final_score)
        
        return ProportionScore(
            score=final_score,
            harmony_level=harmony_level,
            positive_factors=positive_factors,
            negative_factors=negative_factors,
            styling_tips=tips
        )
    
    def get_proportion_tips(
        self,
        torso_proportion: str,
        leg_proportion: str
    ) -> List[str]:
        """
        Get styling tips based on proportions.
        
        Args:
            torso_proportion: "short", "average", or "long"
            leg_proportion: "short", "average", or "long"
            
        Returns:
            List of styling tips
        """
        tips = []
        
        torso_key = f"{torso_proportion}_torso"
        if torso_key in self.proportion_rules:
            tips.extend(self.proportion_rules[torso_key].get("tips", []))
        
        leg_key = f"{leg_proportion}_legs"
        if leg_key in self.proportion_rules:
            tips.extend(self.proportion_rules[leg_key].get("tips", []))
        
        if not tips:
            tips.append("Your proportions are well-balanced - most styles work well!")
        
        return tips
    
    def _classify_torso(self, ratio: Optional[float]) -> str:
        """Classify torso as short, average, or long."""
        if ratio is None:
            return "average"
        
        # Typical torso ratio is ~0.30 (30% of height)
        if ratio < 0.28:
            return "short"
        elif ratio > 0.32:
            return "long"
        return "average"
    
    def _classify_legs(self, ratio: Optional[float]) -> str:
        """Classify legs as short, average, or long."""
        if ratio is None:
            return "average"
        
        # Typical leg ratio is ~0.47 (47% of height)
        if ratio < 0.45:
            return "short"
        elif ratio > 0.49:
            return "long"
        return "average"
    
    def _extract_outfit_features(self, garments: List[Garment]) -> Set[str]:
        """Extract styling features from outfit."""
        features = set()
        
        for garment in garments:
            # Check attributes
            if hasattr(garment, 'attributes') and garment.attributes:
                attrs = garment.attributes
                if isinstance(attrs, dict):
                    # Check waist type
                    waist = attrs.get('waist') or ''
                    waist = str(waist).lower()
                    if 'high' in waist:
                        features.add('high_waist')
                    elif 'low' in waist:
                        features.add('low_rise')
                    
                    # Check neckline
                    neck = attrs.get('neckline') or ''
                    neck = str(neck).lower()
                    if 'v' in neck:
                        features.add('v_neck')
                    
                    # Check length
                    length = attrs.get('length') or ''
                    length = str(length).lower()
                    if 'crop' in length:
                        features.add('cropped_tops')
                    elif 'long' in length:
                        features.add('long_tops')
                    
                    # Check patterns
                    pattern = attrs.get('pattern') or ''
                    pattern = str(pattern).lower()
                    if 'vertical' in pattern or 'stripe' in pattern:
                        features.add('vertical_stripes')
                    elif 'horizontal' in pattern:
                        features.add('horizontal_lines')
            
            # Check category for certain features
            category = str(getattr(garment, 'category', '')).lower()
            if 'cropped' in category:
                features.add('cropped_pants')
        
        return features
    
    def _get_garment_weight(self, garment: Garment) -> Optional[str]:
        """Get fabric weight from garment."""
        if hasattr(garment, 'attributes') and garment.attributes:
            attrs = garment.attributes
            if isinstance(attrs, dict):
                weight = attrs.get('weight') or attrs.get('fabric_weight')
                if weight:
                    return str(weight).lower()
        return None
    
    def _get_styling_tips(self, analysis: ProportionAnalysis) -> List[str]:
        """Get styling tips based on proportion analysis."""
        tips = []
        
        torso_key = f"{analysis.torso_proportion}_torso"
        if torso_key in self.proportion_rules:
            tips.extend(self.proportion_rules[torso_key].get("tips", []))
        
        leg_key = f"{analysis.leg_proportion}_legs"
        if leg_key in self.proportion_rules:
            tips.extend(self.proportion_rules[leg_key].get("tips", []))
        
        return tips
    
    def _classify_harmony(self, score: float) -> str:
        """Classify harmony level based on score."""
        if score >= 0.85:
            return "excellent"
        elif score >= 0.70:
            return "good"
        elif score >= 0.50:
            return "fair"
        else:
            return "poor"
