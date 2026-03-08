"""
Pattern Mixing Rules Scorer
Comprehensive scoring system for pattern combinations in outfits.

Rules implemented:
1. 60/40 Balance Rule - Patterns vs. Solids ratio
2. Scale Contrast Rule - Different pattern sizes
3. Pattern Family Hierarchy - Compatible pattern types
4. Pattern Sandwich Rule - Visual separation
5. Texture Harmony Rule - Tactile pattern compatibility

| Rule                    | Weight | Max Points |
|-------------------------|--------|------------|
| 60/40 Balance           | 25%    | 5 pts      |
| Scale Contrast          | 25%    | 5 pts      |
| Pattern Family          | 20%    | 5 pts      |
| Pattern Sandwich        | 15%    | 5 pts      |
| Texture Harmony         | 15%    | 5 pts      |
"""
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.core.models import Garment, GarmentCategory, PatternInfo, MaterialInfo
from src.core import get_logger

logger = get_logger(__name__)


# ============== Enums ==============

class PatternFamily(str, Enum):
    """Categories of pattern types."""
    SOLID = "solid"
    
    # Geometric patterns
    STRIPES = "stripes"
    CHECKS = "checks"  # Includes plaid, gingham, tartan
    DOTS = "dots"      # Polka dots, spots
    GEOMETRIC = "geometric"  # Abstract shapes, chevrons, zigzag
    
    # Organic patterns
    FLORAL = "floral"
    ANIMAL = "animal"  # Leopard, zebra, snake
    PAISLEY = "paisley"
    TROPICAL = "tropical"  # Leaves, palms
    
    # Classic patterns
    HOUNDSTOOTH = "houndstooth"
    HERRINGBONE = "herringbone"
    ARGYLE = "argyle"
    
    # Other
    ABSTRACT = "abstract"
    CAMO = "camo"
    TIE_DYE = "tie_dye"
    NOVELTY = "novelty"  # Graphics, logos, prints


class PatternScale(str, Enum):
    """Scale/size of patterns."""
    MICRO = "micro"      # Very small, almost texture-like
    SMALL = "small"      # Subtle, discrete
    MEDIUM = "medium"    # Standard size
    LARGE = "large"      # Bold, statement
    OVERSIZED = "oversized"  # Very large, dramatic


class PatternDensity(str, Enum):
    """How dense/busy the pattern is."""
    SPARSE = "sparse"    # Lots of negative space
    MODERATE = "moderate"
    DENSE = "dense"      # Busy, Liberty-style


class TextureType(str, Enum):
    """Texture categories for fabrics."""
    SMOOTH = "smooth"      # Silk, satin, fine cotton
    MATTE = "matte"        # Standard cotton, linen
    TEXTURED = "textured"  # Tweed, cable knit, corduroy
    SHINY = "shiny"        # Sequins, metallic, patent
    FUZZY = "fuzzy"        # Velvet, fleece, mohair


# ============== Data Classes ==============

@dataclass
class PatternAnalysis:
    """Analysis of a single garment's pattern."""
    garment_id: str
    category: GarmentCategory
    family: PatternFamily
    scale: PatternScale
    density: PatternDensity
    texture: TextureType
    colors: List[str]
    is_neutral_pattern: bool  # Breton stripes, gingham, etc.


@dataclass
class PatternMixingResult:
    """Complete result from pattern mixing analysis."""
    overall_score: float  # 0-1 normalized
    grade: str  # A, B, C, D, F
    
    # Individual rule scores (0-5 points each)
    balance_score: int
    scale_score: int
    family_score: int
    sandwich_score: int
    texture_score: int
    
    # Analysis details
    pattern_count: int
    solid_count: int
    pattern_ratio: float  # % of outfit that is patterned
    
    # Flags
    has_color_callback: bool  # Pattern color repeated in solid
    has_scale_contrast: bool
    has_visual_separator: bool
    
    # Recommendations
    recommendations: List[str]
    pattern_breakdown: List[PatternAnalysis]


# ============== Main Scorer ==============

class PatternMixingScorer:
    """
    Comprehensive pattern mixing scorer.
    
    Evaluates outfit pattern combinations based on classic styling rules
    used by fashion stylists and designers.
    """
    
    # Weights for each rule
    WEIGHTS = {
        "balance": 0.25,
        "scale": 0.25,
        "family": 0.20,
        "sandwich": 0.15,
        "texture": 0.15
    }
    
    # Pattern types considered "neutral" (go with everything)
    NEUTRAL_PATTERNS = {
        "breton stripes", "thin stripes", "pinstripes", "gingham",
        "small checks", "vichy", "micro dots", "subtle plaid"
    }
    
    # Pattern keywords for classification
    PATTERN_KEYWORDS = {
        PatternFamily.STRIPES: ["stripe", "stripes", "striped", "breton", "pinstripe", "nautical"],
        PatternFamily.CHECKS: ["check", "checked", "plaid", "tartan", "gingham", "vichy", "buffalo", "windowpane"],
        PatternFamily.DOTS: ["dot", "dots", "polka", "spotted", "spots"],
        PatternFamily.FLORAL: ["floral", "flower", "flowers", "botanical", "rose", "liberty"],
        PatternFamily.ANIMAL: ["leopard", "zebra", "snake", "snakeskin", "tiger", "cheetah", "animal"],
        PatternFamily.GEOMETRIC: ["geometric", "chevron", "zigzag", "triangle", "diamond", "abstract shape"],
        PatternFamily.HOUNDSTOOTH: ["houndstooth", "pied de poule"],
        PatternFamily.HERRINGBONE: ["herringbone"],
        PatternFamily.ARGYLE: ["argyle"],
        PatternFamily.PAISLEY: ["paisley"],
        PatternFamily.TROPICAL: ["tropical", "palm", "leaf", "leaves", "jungle"],
        PatternFamily.CAMO: ["camo", "camouflage", "military"],
        PatternFamily.TIE_DYE: ["tie-dye", "tie dye", "dyed"],
        PatternFamily.NOVELTY: ["print", "graphic", "logo", "cartoon", "novelty"],
    }
    
    # Compatible pattern family combinations
    COMPATIBLE_FAMILIES = {
        # Neutral patterns go with most things
        PatternFamily.STRIPES: {PatternFamily.DOTS, PatternFamily.FLORAL, PatternFamily.ANIMAL, PatternFamily.CHECKS},
        PatternFamily.CHECKS: {PatternFamily.STRIPES, PatternFamily.DOTS, PatternFamily.FLORAL},
        # Classic combos
        PatternFamily.DOTS: {PatternFamily.STRIPES, PatternFamily.FLORAL, PatternFamily.CHECKS},
        PatternFamily.FLORAL: {PatternFamily.STRIPES, PatternFamily.DOTS},
        # Animal prints are statement pieces
        PatternFamily.ANIMAL: {PatternFamily.STRIPES},  # Only stripes work well
        # Geometric works with organic
        PatternFamily.GEOMETRIC: {PatternFamily.FLORAL, PatternFamily.STRIPES},
    }
    
    # Texture compatibility
    TEXTURE_COMPATIBILITY = {
        (TextureType.SMOOTH, TextureType.TEXTURED): 1.0,  # Good contrast
        (TextureType.SMOOTH, TextureType.MATTE): 0.9,
        (TextureType.MATTE, TextureType.TEXTURED): 0.8,
        (TextureType.SMOOTH, TextureType.SMOOTH): 0.7,
        (TextureType.MATTE, TextureType.MATTE): 0.7,
        (TextureType.TEXTURED, TextureType.TEXTURED): 0.3,  # Avoid
        (TextureType.SHINY, TextureType.MATTE): 0.8,
        (TextureType.SHINY, TextureType.TEXTURED): 0.5,
        (TextureType.FUZZY, TextureType.SMOOTH): 0.8,
        (TextureType.FUZZY, TextureType.FUZZY): 0.4,  # Avoid
    }
    
    def __init__(self):
        self.max_points_per_rule = 5
    
    def analyze_outfit(self, items: List[Garment]) -> PatternMixingResult:
        """
        Analyze outfit for pattern mixing rules compliance.
        
        Args:
            items: Outfit garments
            
        Returns:
            PatternMixingResult with comprehensive analysis
        """
        if not items:
            return self._empty_result()
        
        # Analyze each item's pattern
        analyses = [self._analyze_garment(item) for item in items]
        
        # Separate patterned from solid
        patterned = [a for a in analyses if a.family != PatternFamily.SOLID]
        solids = [a for a in analyses if a.family == PatternFamily.SOLID]
        
        # Calculate each rule score
        balance_score, has_callback = self._score_balance_rule(patterned, solids, analyses)
        scale_score, has_scale_contrast = self._score_scale_rule(patterned)
        family_score = self._score_family_rule(patterned)
        sandwich_score, has_separator = self._score_sandwich_rule(analyses)
        texture_score = self._score_texture_rule(patterned)
        
        # Calculate overall score
        total_points = (
            self.WEIGHTS["balance"] * balance_score +
            self.WEIGHTS["scale"] * scale_score +
            self.WEIGHTS["family"] * family_score +
            self.WEIGHTS["sandwich"] * sandwich_score +
            self.WEIGHTS["texture"] * texture_score
        )
        overall_score = total_points / self.max_points_per_rule
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            balance_score, scale_score, family_score, sandwich_score, texture_score,
            patterned, solids, has_callback, has_scale_contrast
        )
        
        # Calculate pattern ratio
        pattern_ratio = len(patterned) / len(analyses) if analyses else 0
        
        return PatternMixingResult(
            overall_score=round(overall_score, 3),
            grade=self._calculate_grade(overall_score),
            
            balance_score=balance_score,
            scale_score=scale_score,
            family_score=family_score,
            sandwich_score=sandwich_score,
            texture_score=texture_score,
            
            pattern_count=len(patterned),
            solid_count=len(solids),
            pattern_ratio=round(pattern_ratio, 2),
            
            has_color_callback=has_callback,
            has_scale_contrast=has_scale_contrast,
            has_visual_separator=has_separator,
            
            recommendations=recommendations,
            pattern_breakdown=analyses
        )
    
    def _analyze_garment(self, garment: Garment) -> PatternAnalysis:
        """Analyze a single garment's pattern characteristics."""
        pattern = garment.attributes.pattern
        material = garment.attributes.material
        color = garment.attributes.color
        
        # Determine pattern family
        family = self._classify_pattern_family(pattern)
        
        # Determine scale
        scale = self._determine_scale(pattern)
        
        # Determine density
        density = self._determine_density(pattern, family)
        
        # Determine texture
        texture = self._determine_texture(material)
        
        # Extract colors
        colors = [color.primary.lower()]
        if color.secondary:
            colors.append(color.secondary.lower())
        if color.accent:
            colors.append(color.accent.lower())
        
        # Check if neutral pattern
        is_neutral = self._is_neutral_pattern(pattern, family, scale)
        
        return PatternAnalysis(
            garment_id=garment.id,
            category=garment.attributes.category,
            family=family,
            scale=scale,
            density=density,
            texture=texture,
            colors=colors,
            is_neutral_pattern=is_neutral
        )
    
    def _classify_pattern_family(self, pattern: PatternInfo) -> PatternFamily:
        """Classify pattern into a family."""
        if not pattern or pattern.type.lower() in ["solid", "plain", "none", ""]:
            return PatternFamily.SOLID
        
        pattern_str = f"{pattern.type} {pattern.description or ''}".lower()
        
        for family, keywords in self.PATTERN_KEYWORDS.items():
            if any(kw in pattern_str for kw in keywords):
                return family
        
        # Default to abstract if we can't classify
        if pattern.type.lower() != "solid":
            return PatternFamily.ABSTRACT
        
        return PatternFamily.SOLID
    
    def _determine_scale(self, pattern: PatternInfo) -> PatternScale:
        """Determine pattern scale."""
        if not pattern or not pattern.scale:
            return PatternScale.MEDIUM
        
        scale_str = pattern.scale.lower()
        
        if scale_str in ["micro", "tiny", "very small"]:
            return PatternScale.MICRO
        elif scale_str in ["small", "subtle", "discrete", "fine"]:
            return PatternScale.SMALL
        elif scale_str in ["medium", "standard", "regular"]:
            return PatternScale.MEDIUM
        elif scale_str in ["large", "bold", "big"]:
            return PatternScale.LARGE
        elif scale_str in ["oversized", "very large", "dramatic", "huge"]:
            return PatternScale.OVERSIZED
        
        return PatternScale.MEDIUM
    
    def _determine_density(self, pattern: PatternInfo, family: PatternFamily) -> PatternDensity:
        """Determine pattern density."""
        if family == PatternFamily.SOLID:
            return PatternDensity.SPARSE
        
        desc = (pattern.description or "").lower()
        
        # Keywords for density
        if any(w in desc for w in ["dense", "busy", "liberty", "packed", "tight"]):
            return PatternDensity.DENSE
        elif any(w in desc for w in ["sparse", "airy", "open", "spaced", "minimal"]):
            return PatternDensity.SPARSE
        
        # Default based on pattern type
        dense_patterns = {PatternFamily.FLORAL, PatternFamily.PAISLEY, PatternFamily.TROPICAL}
        if family in dense_patterns:
            return PatternDensity.DENSE
        
        return PatternDensity.MODERATE
    
    def _determine_texture(self, material: Optional[MaterialInfo]) -> TextureType:
        """Determine fabric texture type."""
        if not material:
            return TextureType.MATTE
        
        material_str = f"{material.primary} {material.texture or ''}".lower()
        
        smooth_keywords = ["silk", "satin", "charmeuse", "sateen", "smooth", "fine"]
        textured_keywords = ["tweed", "cable", "knit", "corduroy", "bouclé", "ribbed", "waffle"]
        shiny_keywords = ["sequin", "metallic", "patent", "lamé", "glitter", "lurex"]
        fuzzy_keywords = ["velvet", "fleece", "mohair", "sherpa", "fuzzy", "fur", "shearling"]
        
        if any(kw in material_str for kw in shiny_keywords):
            return TextureType.SHINY
        elif any(kw in material_str for kw in fuzzy_keywords):
            return TextureType.FUZZY
        elif any(kw in material_str for kw in textured_keywords):
            return TextureType.TEXTURED
        elif any(kw in material_str for kw in smooth_keywords):
            return TextureType.SMOOTH
        
        return TextureType.MATTE
    
    def _is_neutral_pattern(
        self, 
        pattern: PatternInfo, 
        family: PatternFamily,
        scale: PatternScale
    ) -> bool:
        """Check if pattern is considered a 'neutral' pattern."""
        if family == PatternFamily.SOLID:
            return False
        
        pattern_desc = f"{pattern.type} {pattern.description or ''}".lower()
        
        # Check against known neutral patterns
        if any(np in pattern_desc for np in self.NEUTRAL_PATTERNS):
            return True
        
        # Small/micro checks and stripes are generally neutral
        if family in [PatternFamily.STRIPES, PatternFamily.CHECKS]:
            if scale in [PatternScale.MICRO, PatternScale.SMALL]:
                return True
        
        return False
    
    # ============== Scoring Rules ==============
    
    def _score_balance_rule(
        self,
        patterned: List[PatternAnalysis],
        solids: List[PatternAnalysis],
        all_items: List[PatternAnalysis]
    ) -> Tuple[int, bool]:
        """
        Score the 60/40 balance rule.
        
        Best: 60-70% solid, 30-40% patterned (or vice versa with experience)
        Also checks for color callback (pattern color in solid piece)
        """
        total = len(all_items)
        if total == 0:
            return 0, False
        
        pattern_ratio = len(patterned) / total
        
        # Check for color callback
        has_callback = self._check_color_callback(patterned, solids)
        
        # Score based on ratio
        if len(patterned) == 0:
            # All solids - safe but not interesting
            return 3, False
        elif len(patterned) == 1 and len(solids) >= 1:
            # Ideal: one pattern piece anchored by solids
            base_score = 5 if has_callback else 4
            return base_score, has_callback
        elif len(patterned) == 2 and len(solids) >= 1:
            # Two patterns can work with solid separator
            return 4 if has_callback else 3, has_callback
        elif 0.3 <= pattern_ratio <= 0.5:
            # Good balance range
            return 4 if has_callback else 3, has_callback
        elif pattern_ratio > 0.7:
            # Too many patterns - risky
            return 1, has_callback
        else:
            return 3, has_callback
    
    def _check_color_callback(
        self,
        patterned: List[PatternAnalysis],
        solids: List[PatternAnalysis]
    ) -> bool:
        """Check if any color in patterns is repeated in solids."""
        if not patterned or not solids:
            return False
        
        # Get all colors from patterns
        pattern_colors = set()
        for p in patterned:
            pattern_colors.update(p.colors)
        
        # Check if any solid matches
        for s in solids:
            if any(c in pattern_colors for c in s.colors):
                return True
        
        return False
    
    def _score_scale_rule(self, patterned: List[PatternAnalysis]) -> Tuple[int, bool]:
        """
        Score the scale contrast rule.
        
        When mixing patterns, they should be different scales.
        Large + Small = Good
        Same scale = Visual conflict
        """
        if len(patterned) <= 1:
            return 5, True  # No conflict possible
        
        scales = [p.scale for p in patterned]
        scale_values = {
            PatternScale.MICRO: 1,
            PatternScale.SMALL: 2,
            PatternScale.MEDIUM: 3,
            PatternScale.LARGE: 4,
            PatternScale.OVERSIZED: 5
        }
        
        # Calculate scale differences
        values = [scale_values[s] for s in scales]
        max_diff = max(values) - min(values)
        
        # Also check density contrast
        densities = [p.density for p in patterned]
        has_density_contrast = len(set(densities)) > 1
        
        has_scale_contrast = max_diff >= 2
        
        if max_diff >= 3:
            return 5, True  # Great contrast
        elif max_diff == 2:
            return 4, True  # Good contrast
        elif max_diff == 1 and has_density_contrast:
            return 3, False  # Acceptable with density contrast
        elif max_diff == 0:
            return 1, False  # Same scale - visual conflict
        else:
            return 2, False
    
    def _score_family_rule(self, patterned: List[PatternAnalysis]) -> int:
        """
        Score pattern family compatibility.
        
        Some pattern combinations are more natural than others.
        """
        if len(patterned) <= 1:
            return 5  # No mixing = no conflict
        
        families = [p.family for p in patterned]
        
        # Check if any are neutral patterns (go with everything)
        neutral_count = sum(1 for p in patterned if p.is_neutral_pattern)
        if neutral_count == len(patterned) - 1:
            return 5  # All but one are neutral - safe
        
        # Check all pair combinations
        compatible_pairs = 0
        total_pairs = 0
        
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                total_pairs += 1
                f1, f2 = families[i], families[j]
                
                # Check compatibility
                is_compatible = (
                    f2 in self.COMPATIBLE_FAMILIES.get(f1, set()) or
                    f1 in self.COMPATIBLE_FAMILIES.get(f2, set())
                )
                if is_compatible:
                    compatible_pairs += 1
        
        if total_pairs == 0:
            return 5
        
        compatibility_ratio = compatible_pairs / total_pairs
        
        if compatibility_ratio == 1.0:
            return 5
        elif compatibility_ratio >= 0.7:
            return 4
        elif compatibility_ratio >= 0.5:
            return 3
        elif compatibility_ratio >= 0.3:
            return 2
        else:
            return 1
    
    def _score_sandwich_rule(self, all_items: List[PatternAnalysis]) -> Tuple[int, bool]:
        """
        Score the pattern sandwich rule.
        
        When wearing multiple patterns, separate them with a solid piece.
        Top pattern + Solid middle + Bottom pattern = Visual breathing room
        """
        if len(all_items) < 3:
            return 4, False  # Not enough items to evaluate
        
        # Sort by body position (top to bottom)
        position_order = {
            GarmentCategory.OUTERWEAR: 0,
            GarmentCategory.TOP: 1,
            GarmentCategory.DRESS: 1,
            GarmentCategory.BOTTOM: 2,
            GarmentCategory.SHOES: 3,
            GarmentCategory.ACCESSORY: 4,
            GarmentCategory.BAG: 4,
        }
        
        sorted_items = sorted(all_items, key=lambda x: position_order.get(x.category, 5))
        
        # Look for pattern-solid-pattern sequences
        patterned_indices = [i for i, item in enumerate(sorted_items) 
                           if item.family != PatternFamily.SOLID]
        
        if len(patterned_indices) <= 1:
            return 5, True  # Only one pattern or none
        
        # Check if patterns are separated by solids
        has_separator = True
        for i in range(len(patterned_indices) - 1):
            idx1 = patterned_indices[i]
            idx2 = patterned_indices[i + 1]
            
            # Check if there's a solid between them
            has_solid_between = any(
                sorted_items[j].family == PatternFamily.SOLID
                for j in range(idx1 + 1, idx2)
            )
            
            if not has_solid_between and idx2 - idx1 == 1:
                # Adjacent patterns without separator
                has_separator = False
                break
        
        if has_separator:
            return 5, True
        elif len(patterned_indices) == 2:
            return 3, False  # Two adjacent patterns - workable
        else:
            return 1, False  # Multiple adjacent patterns
    
    def _score_texture_rule(self, patterned: List[PatternAnalysis]) -> int:
        """
        Score texture harmony.
        
        Avoid mixing two heavily textured patterns.
        Prefer smooth printed + textured woven contrast.
        """
        if len(patterned) <= 1:
            return 5
        
        textures = [p.texture for p in patterned]
        
        # Calculate average compatibility
        total_compat = 0
        pairs = 0
        
        for i in range(len(textures)):
            for j in range(i + 1, len(textures)):
                pairs += 1
                t1, t2 = textures[i], textures[j]
                
                # Look up compatibility (check both orders)
                compat = self.TEXTURE_COMPATIBILITY.get(
                    (t1, t2),
                    self.TEXTURE_COMPATIBILITY.get((t2, t1), 0.6)
                )
                total_compat += compat
        
        if pairs == 0:
            return 5
        
        avg_compat = total_compat / pairs
        
        if avg_compat >= 0.8:
            return 5
        elif avg_compat >= 0.6:
            return 4
        elif avg_compat >= 0.5:
            return 3
        elif avg_compat >= 0.4:
            return 2
        else:
            return 1
    
    # ============== Helpers ==============
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade."""
        if score >= 0.85:
            return "A"
        elif score >= 0.70:
            return "B"
        elif score >= 0.55:
            return "C"
        elif score >= 0.40:
            return "D"
        return "F"
    
    def _generate_recommendations(
        self,
        balance: int,
        scale: int,
        family: int,
        sandwich: int,
        texture: int,
        patterned: List[PatternAnalysis],
        solids: List[PatternAnalysis],
        has_callback: bool,
        has_scale_contrast: bool
    ) -> List[str]:
        """Generate specific recommendations."""
        recommendations = []
        
        # Balance recommendations
        if balance < 4 and len(patterned) > 1 and len(solids) == 0:
            recommendations.append(
                "Add a solid piece to anchor your patterns. "
                "Try a neutral-colored item to give the eye a rest."
            )
        elif balance < 4 and not has_callback and patterned and solids:
            pattern_colors = set()
            for p in patterned:
                pattern_colors.update(p.colors)
            recommendations.append(
                f"Color callback tip: Your solid pieces could match one of your pattern colors "
                f"({', '.join(list(pattern_colors)[:3])}) for better cohesion."
            )
        
        # Scale recommendations
        if scale < 4 and len(patterned) >= 2:
            if not has_scale_contrast:
                recommendations.append(
                    "Your patterns are similar in scale, which can create visual confusion. "
                    "Try pairing a larger pattern with a smaller one."
                )
        
        # Family recommendations
        if family < 4 and len(patterned) >= 2:
            families = [p.family.value for p in patterned]
            recommendations.append(
                f"Mixing {' and '.join(families)} can be tricky. "
                "Consider swapping one for stripes or small checks, which work as 'neutral' patterns."
            )
        
        # Sandwich recommendations
        if sandwich < 4 and len(patterned) >= 2:
            recommendations.append(
                "Pattern sandwich tip: Separate your patterned pieces with a solid item "
                "(belt, plain pants, or solid jacket) to let each pattern breathe."
            )
        
        # Texture recommendations
        if texture < 4:
            textures = [p.texture.value for p in patterned]
            if textures.count("textured") >= 2:
                recommendations.append(
                    "Avoid pairing two heavily textured patterns (like tweed with corduroy). "
                    "Try a smooth printed fabric with a textured woven instead."
                )
        
        # Positive feedback
        if not recommendations:
            if balance >= 4 and scale >= 4:
                recommendations.append(
                    "Great pattern mixing! You've balanced prints well with good scale contrast."
                )
        
        return recommendations
    
    def _empty_result(self) -> PatternMixingResult:
        """Return empty result for no items."""
        return PatternMixingResult(
            overall_score=0.5,
            grade="C",
            balance_score=0,
            scale_score=0,
            family_score=0,
            sandwich_score=0,
            texture_score=0,
            pattern_count=0,
            solid_count=0,
            pattern_ratio=0,
            has_color_callback=False,
            has_scale_contrast=False,
            has_visual_separator=False,
            recommendations=["Add items to analyze"],
            pattern_breakdown=[]
        )


# ============== Convenience Functions ==============

def get_pattern_mixing_score(items: List[Garment]) -> float:
    """Get overall pattern mixing score (0-1)."""
    scorer = PatternMixingScorer()
    result = scorer.analyze_outfit(items)
    return result.overall_score


def check_pattern_compatibility(items: List[Garment]) -> bool:
    """Check if patterns in outfit are compatible."""
    scorer = PatternMixingScorer()
    result = scorer.analyze_outfit(items)
    return result.overall_score >= 0.6


def count_patterns(items: List[Garment]) -> int:
    """Count number of patterned items."""
    scorer = PatternMixingScorer()
    result = scorer.analyze_outfit(items)
    return result.pattern_count
