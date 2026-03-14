"""
Seven Point Rule Scorer
Implements the 7-point styling rule for outfit harmony.

The 7-point rule states that a harmonious outfit should score between 7-10 points:
- Basic/staple items = 1 point (solid colors, minimal design)
- Statement/special items = 2 points (textures, prints, eye-catching designs)

This ensures balance between simple and special pieces.

Domain knowledge (statement patterns, textures, optimal range, etc.) is loaded
from config/data/style_rules_config.json so stylists can tune it without code changes.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.models import Garment, GarmentCategory, TransparencyLevel
from src.core import get_logger

logger = get_logger(__name__)

_STYLE_RULES_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "config" / "data" / "style_rules_config.json"
)


def _load_style_rules_config() -> dict:
    """Load style rules config, returning empty dict on failure."""
    try:
        with open(_STYLE_RULES_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not load style_rules_config.json: %s — using built-in defaults.", exc)
        return {}


class GarmentPointValue(Enum):
    """Point values for garments in the 7-point rule."""
    BASIC = 1  # Staple, simple pieces
    STATEMENT = 2  # Eye-catching, textured, printed pieces


@dataclass
class SevenPointResult:
    """Result of the 7-point rule analysis."""
    total_points: int
    is_harmonious: bool  # True if between 7-10 points
    breakdown: List[Dict]  # Point breakdown per item
    recommendation: str
    score: float  # Normalized score 0-1


class SevenPointRuleScorer:
    """
    Scores outfits based on the 7-point styling rule.
    
    The rule helps create balanced outfits that are interesting
    but not overwhelming. A harmonious outfit scores 7-10 points.
    
    Point assignment:
    - 1 point: Basic staples (solid colors, minimal design)
    - 2 points: Statement pieces (prints, textures, embellishments)
    """
    
    def __init__(self):
        cfg = _load_style_rules_config().get("seven_point_rule", {})

        # Patterns that make an item a statement piece
        self.statement_patterns: set = set(cfg.get("statement_patterns", [
            "stripes", "plaid", "floral", "geometric", "animal",
            "abstract", "paisley", "polka_dot", "checkered", "leopard",
            "zebra", "tropical", "tie_dye", "camo", "houndstooth",
        ]))

        # Textures that add visual interest
        self.statement_textures: set = set(cfg.get("statement_textures", [
            "velvet", "sequin", "leather", "lace", "fur", "faux_fur",
            "metallic", "satin", "silk", "tweed", "boucle", "crochet",
            "knit_cable", "quilted", "embossed", "pleated",
        ]))

        # Embellishments that make statement pieces
        self.statement_embellishments: set = set(cfg.get("statement_embellishments", [
            "sequins", "beading", "embroidery", "ruffles", "fringe",
            "studs", "crystals", "pearls", "patches", "applique",
            "cutouts", "lace_trim", "metallic_hardware",
        ]))

        # Categories that are typically statement by nature
        self.inherently_statement_subcategories: set = set(cfg.get(
            "inherently_statement_subcategories", [
                "blazer", "statement_jewelry", "cocktail_dress", "gown",
                "fur_coat", "leather_jacket", "sequin_top", "maxi_dress",
            ]
        ))

        # Bold colors (non-neutrals) that count as statement
        self.bold_colors: set = set(cfg.get("bold_colors", [
            "red", "orange", "yellow", "fuchsia", "magenta", "purple",
            "electric_blue", "lime", "coral", "hot_pink", "neon",
        ]))

        # Optimal range for harmonious outfits
        self.min_optimal_points: int = cfg.get("min_optimal_points", 7)
        self.max_optimal_points: int = cfg.get("max_optimal_points", 10)
        self._optimal_center: float = cfg.get("optimal_center", 8.5)
    
    def calculate_garment_points(self, garment: Garment) -> Tuple[int, List[str]]:
        """
        Calculate points for a single garment.
        
        Args:
            garment: The garment to evaluate
            
        Returns:
            Tuple of (points, reasons for the score)
        """
        attrs = garment.attributes
        reasons = []
        
        # Start with basic assumption
        is_statement = False
        
        # Check 1: Pattern
        pattern_type = attrs.pattern.type.lower() if attrs.pattern else "solid"
        if pattern_type != "solid" and pattern_type in self.statement_patterns:
            is_statement = True
            reasons.append(f"has {pattern_type} pattern")
        
        # Check 2: Material/Texture
        if attrs.material and attrs.material.texture:
            texture = attrs.material.texture.lower()
            if texture in self.statement_textures:
                is_statement = True
                reasons.append(f"has {texture} texture")
        
        if attrs.material and attrs.material.primary:
            material = attrs.material.primary.lower()
            if material in self.statement_textures:
                is_statement = True
                reasons.append(f"made of {material}")
        
        # Check 3: Embellishments
        if attrs.details.embellishments:
            for embellishment in attrs.details.embellishments:
                if embellishment.lower() in self.statement_embellishments:
                    is_statement = True
                    reasons.append(f"has {embellishment}")
                    break
        
        # Check 4: Transparency (sheer items are statement)
        if attrs.details.transparency != TransparencyLevel.OPAQUE:
            is_statement = True
            reasons.append(f"{attrs.details.transparency.value} fabric")
        
        # Check 5: Distressed details
        if attrs.details.distressed:
            is_statement = True
            reasons.append("distressed details")
        
        # Check 6: Bold colors (not neutrals)
        bold_colors = {
            "red", "orange", "yellow", "fuchsia", "magenta", "purple",
            "electric_blue", "lime", "coral", "hot_pink", "neon"
        }
        if attrs.color.primary.lower() in bold_colors:
            is_statement = True
            reasons.append(f"bold {attrs.color.primary} color")
        
        # Check 7: Subcategory inherently statement
        if attrs.subcategory and attrs.subcategory.lower() in self.inherently_statement_subcategories:
            is_statement = True
            reasons.append(f"statement {attrs.subcategory}")
        
        # Check 8: Pattern scale (large patterns are more statement)
        if attrs.pattern.scale and attrs.pattern.scale.lower() == "large":
            is_statement = True
            reasons.append("large-scale pattern")
        
        # Determine final points
        if is_statement:
            return GarmentPointValue.STATEMENT.value, reasons
        else:
            reasons.append("basic staple piece")
            return GarmentPointValue.BASIC.value, reasons
    
    def analyze_outfit(self, items: List[Garment]) -> SevenPointResult:
        """
        Analyze an outfit using the 7-point rule.
        
        Args:
            items: List of garments in the outfit
            
        Returns:
            SevenPointResult with complete analysis
        """
        if not items:
            return SevenPointResult(
                total_points=0,
                is_harmonious=False,
                breakdown=[],
                recommendation="Add items to your outfit",
                score=0.0
            )
        
        # Calculate points for each item
        breakdown = []
        total_points = 0
        
        for garment in items:
            points, reasons = self.calculate_garment_points(garment)
            total_points += points
            
            breakdown.append({
                "garment_id": garment.id,
                "category": garment.attributes.category.value,
                "subcategory": garment.attributes.subcategory,
                "color": garment.attributes.color.primary,
                "points": points,
                "point_type": "statement" if points == 2 else "basic",
                "reasons": reasons
            })
        
        # Determine if harmonious
        is_harmonious = self.min_optimal_points <= total_points <= self.max_optimal_points
        
        # Calculate normalized score
        score = self._calculate_normalized_score(total_points, len(items))
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            total_points, 
            len(items),
            breakdown
        )
        
        return SevenPointResult(
            total_points=total_points,
            is_harmonious=is_harmonious,
            breakdown=breakdown,
            recommendation=recommendation,
            score=score
        )
    
    def _calculate_normalized_score(self, total_points: int, num_items: int) -> float:
        """
        Calculate a normalized score between 0 and 1.
        
        Perfect score (1.0) is achieved when total is in 7-10 range.
        """
        if num_items == 0:
            return 0.0
        
        # Optimal center is mid-point of the harmonious range
        optimal_center = self._optimal_center
        
        # Calculate distance from optimal
        if self.min_optimal_points <= total_points <= self.max_optimal_points:
            # Within optimal range - high score
            # Closer to center = higher score
            distance_from_center = abs(total_points - optimal_center)
            return 1.0 - (distance_from_center / 10)  # Max distance is ~8.5
        
        elif total_points < self.min_optimal_points:
            # Too few points - outfit is too basic/boring
            deficit = self.min_optimal_points - total_points
            return max(0.3, 0.7 - (deficit * 0.1))
        
        else:
            # Too many points - outfit is too busy
            excess = total_points - self.max_optimal_points
            return max(0.2, 0.6 - (excess * 0.1))
    
    def _generate_recommendation(
        self,
        total_points: int,
        num_items: int,
        breakdown: List[Dict]
    ) -> str:
        """Generate styling recommendation based on analysis."""
        
        if self.min_optimal_points <= total_points <= self.max_optimal_points:
            return f"✅ Perfect balance! Your outfit scores {total_points} points - harmonious and stylish."
        
        basic_count = sum(1 for item in breakdown if item["points"] == 1)
        statement_count = sum(1 for item in breakdown if item["points"] == 2)
        
        if total_points < self.min_optimal_points:
            deficit = self.min_optimal_points - total_points
            if basic_count > statement_count:
                return (
                    f"⚠️ Your outfit scores {total_points} points (target: 7-10). "
                    f"Consider adding {deficit} statement piece(s) with texture, "
                    f"print, or interesting details to elevate the look."
                )
            else:
                return (
                    f"⚠️ Your outfit scores {total_points} points (target: 7-10). "
                    f"Add more pieces or swap a basic item for a statement piece."
                )
        
        else:  # total_points > self.max_optimal_points
            excess = total_points - self.max_optimal_points
            return (
                f"⚠️ Your outfit scores {total_points} points (target: 7-10). "
                f"It may look too busy. Consider replacing {excess} statement "
                f"piece(s) with simpler basics for better balance."
            )
    
    def suggest_additions(
        self,
        current_items: List[Garment],
        available_items: List[Garment],
        target_category: GarmentCategory = None
    ) -> List[Tuple[Garment, str]]:
        """
        Suggest items to add that would improve the 7-point balance.
        
        Args:
            current_items: Items already in the outfit
            available_items: Items available to add
            target_category: Optional category filter
            
        Returns:
            List of (garment, reason) tuples for suggested additions
        """
        current_result = self.analyze_outfit(current_items)
        suggestions = []
        
        # Filter by category if specified
        candidates = available_items
        if target_category:
            candidates = [g for g in available_items 
                         if g.attributes.category == target_category]
        
        for garment in candidates:
            points, _ = self.calculate_garment_points(garment)
            new_total = current_result.total_points + points
            
            # Check if this improves the score
            if not current_result.is_harmonious:
                if self.min_optimal_points <= new_total <= self.max_optimal_points:
                    point_type = "statement" if points == 2 else "basic"
                    reason = f"Adding this {point_type} piece brings total to {new_total} points (perfect range!)"
                    suggestions.append((garment, reason))
        
        # Sort by how close to optimal center (8.5)
        suggestions.sort(
            key=lambda x: abs((current_result.total_points + 
                              self.calculate_garment_points(x[0])[0]) - 8.5)
        )
        
        return suggestions[:5]  # Return top 5 suggestions


# Convenience function
def calculate_seven_point_score(items: List[Garment]) -> float:
    """
    Quick function to get the 7-point rule score for an outfit.
    
    Args:
        items: List of garments
        
    Returns:
        Score between 0 and 1
    """
    scorer = SevenPointRuleScorer()
    result = scorer.analyze_outfit(items)
    return result.score
