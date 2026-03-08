"""
Design Principles Scorer
Comprehensive scoring system based on fundamental fashion design principles.

Implements 11 aesthetic principles:
1. Proportion (Golden Ratio)
2. Balance (Symmetric/Asymmetric)
3. Rhythm (Visual movement)
4. Radiation (Lines from center)
5. Gradation (Progressive variation)
6. Emphasis (Focal point)
7. Contrast (Strong differences)
8. Harmony (Elements work together)
9. Unity (Cohesive whole)
10. Repetition (Repeated elements)
11. Scale (Size relationships)

Plus Line Direction analysis (Vertical/Horizontal/Diagonal effects)
"""
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.core.models import (
    Garment, GarmentCategory, GarmentAttributes,
    PatternInfo, ColorInfo, MaterialInfo,
    NecklineType, LengthType, WaistRise
)
from src.core import get_logger

logger = get_logger(__name__)


# ============== Enums ==============

class BalanceType(str, Enum):
    """Types of visual balance."""
    SYMMETRIC = "symmetric"      # Formal, classic, stable
    ASYMMETRIC = "asymmetric"    # Dynamic, modern, fashion-forward


class LineDirection(str, Enum):
    """Direction of visual lines in garment."""
    VERTICAL = "vertical"        # Elongating, slimming
    HORIZONTAL = "horizontal"    # Widening, shortening
    DIAGONAL = "diagonal"        # Dynamic, movement
    CURVED = "curved"            # Soft, feminine
    MIXED = "mixed"              # Combination


class DesignStyle(str, Enum):
    """Overall design style classification."""
    CLASSIC = "classic"          # Traditional, timeless
    MODERN = "modern"            # Contemporary, clean
    AVANT_GARDE = "avant_garde"  # Experimental, rule-breaking
    MINIMALIST = "minimalist"    # Simple, understated
    MAXIMALIST = "maximalist"    # Bold, layered, decorative


class EmphasisType(str, Enum):
    """Type of focal point."""
    COLOR = "color"              # Bright/contrasting color
    TEXTURE = "texture"          # Unique texture
    DETAIL = "detail"            # Embellishment, buttons, etc.
    SILHOUETTE = "silhouette"    # Unusual shape
    NECKLINE = "neckline"        # Statement neckline
    ACCESSORY = "accessory"      # Statement accessory
    NONE = "none"                # No clear focal point


# ============== Data Classes ==============

@dataclass
class ProportionAnalysis:
    """Analysis of proportions in outfit."""
    score: float  # 0-1
    ratio: str    # e.g., "1:2", "golden", "50:50"
    follows_golden_ratio: bool
    top_to_bottom_ratio: Optional[Tuple[int, int]] = None
    recommendation: str = ""


@dataclass
class BalanceAnalysis:
    """Analysis of visual balance."""
    score: float
    balance_type: BalanceType
    is_intentional: bool  # Intentional asymmetry vs. accidental imbalance
    recommendation: str = ""


@dataclass
class RhythmAnalysis:
    """Analysis of visual rhythm."""
    score: float
    has_rhythm: bool
    rhythm_sources: List[str]  # e.g., ["repeated buttons", "pattern flow"]
    recommendation: str = ""


@dataclass
class EmphasisAnalysis:
    """Analysis of focal point."""
    score: float
    focal_points: List[EmphasisType]
    has_single_focus: bool
    is_clear: bool
    recommendation: str = ""


@dataclass
class ContrastAnalysis:
    """Analysis of contrast elements."""
    score: float
    contrast_types: List[str]  # color, texture, shine, etc.
    contrast_level: str  # low, medium, high
    recommendation: str = ""


@dataclass
class HarmonyAnalysis:
    """Analysis of overall harmony."""
    score: float
    harmonious_elements: List[str]
    conflicting_elements: List[str]
    recommendation: str = ""


@dataclass
class LineAnalysis:
    """Analysis of line directions."""
    score: float
    dominant_direction: LineDirection
    visual_effect: str  # elongating, widening, dynamic
    flattering_for_body: bool
    recommendation: str = ""


@dataclass
class DesignPrinciplesResult:
    """Complete design principles analysis."""
    overall_score: float
    grade: str
    
    # Individual analyses
    proportion: ProportionAnalysis
    balance: BalanceAnalysis
    rhythm: RhythmAnalysis
    emphasis: EmphasisAnalysis
    contrast: ContrastAnalysis
    harmony: HarmonyAnalysis
    line_direction: LineAnalysis
    
    # Additional scores
    unity_score: float
    scale_score: float
    repetition_score: float
    gradation_score: float
    
    # Design classification
    design_style: DesignStyle
    
    # Summary
    strengths: List[str]
    improvements: List[str]
    creative_notes: List[str]  # For avant-garde/rule-breaking elements


# ============== Main Scorer ==============

class DesignPrinciplesScorer:
    """
    Comprehensive scorer based on fashion design principles.
    
    Evaluates outfits against fundamental aesthetic principles used
    by fashion designers and taught in fashion schools.
    """
    
    # Golden ratio approximation
    GOLDEN_RATIO = 1.618
    
    # Weights for overall score
    WEIGHTS = {
        "proportion": 0.12,
        "balance": 0.10,
        "rhythm": 0.08,
        "emphasis": 0.12,
        "contrast": 0.10,
        "harmony": 0.15,
        "unity": 0.10,
        "scale": 0.08,
        "repetition": 0.05,
        "gradation": 0.05,
        "line_direction": 0.05
    }
    
    # Line-creating elements
    VERTICAL_ELEMENTS = {
        "v_neck", "long cardigan", "vertical stripes", "long necklace",
        "single-breasted", "center seam", "straight leg", "maxi",
        "longline", "high waisted", "column dress"
    }
    
    HORIZONTAL_ELEMENTS = {
        "boat_neck", "horizontal stripes", "wide belt", "crop top",
        "peplum", "empire waist", "off_shoulder", "boxy",
        "cropped jacket", "wide leg"
    }
    
    DIAGONAL_ELEMENTS = {
        "asymmetric", "wrap dress", "diagonal zipper", "one shoulder",
        "chevron", "bias cut", "asymmetric hem", "draped"
    }
    
    # High contrast combinations
    HIGH_CONTRAST_COLORS = [
        ("black", "white"), ("navy", "cream"), ("red", "black"),
        ("black", "yellow"), ("white", "black")
    ]
    
    def __init__(self):
        pass
    
    def analyze_outfit(
        self,
        items: List[Garment],
        body_type: Optional[str] = None,
        intended_style: Optional[DesignStyle] = None
    ) -> DesignPrinciplesResult:
        """
        Analyze outfit against all design principles.
        
        Args:
            items: Outfit garments
            body_type: Optional body type for line analysis
            intended_style: Optional intended design style
            
        Returns:
            DesignPrinciplesResult with comprehensive analysis
        """
        if not items:
            return self._empty_result()
        
        # Run all analyses
        proportion = self._analyze_proportion(items)
        balance = self._analyze_balance(items, intended_style)
        rhythm = self._analyze_rhythm(items)
        emphasis = self._analyze_emphasis(items)
        contrast = self._analyze_contrast(items)
        harmony = self._analyze_harmony(items)
        line_direction = self._analyze_lines(items, body_type)
        
        unity_score = self._analyze_unity(items)
        scale_score = self._analyze_scale(items)
        repetition_score = self._analyze_repetition(items)
        gradation_score = self._analyze_gradation(items)
        
        # Calculate overall score
        overall_score = (
            self.WEIGHTS["proportion"] * proportion.score +
            self.WEIGHTS["balance"] * balance.score +
            self.WEIGHTS["rhythm"] * rhythm.score +
            self.WEIGHTS["emphasis"] * emphasis.score +
            self.WEIGHTS["contrast"] * contrast.score +
            self.WEIGHTS["harmony"] * harmony.score +
            self.WEIGHTS["unity"] * unity_score +
            self.WEIGHTS["scale"] * scale_score +
            self.WEIGHTS["repetition"] * repetition_score +
            self.WEIGHTS["gradation"] * gradation_score +
            self.WEIGHTS["line_direction"] * line_direction.score
        )
        
        # Determine design style
        design_style = self._classify_design_style(
            items, balance, emphasis, contrast, intended_style
        )
        
        # Generate summary
        strengths, improvements, creative_notes = self._generate_summary(
            proportion, balance, rhythm, emphasis, contrast, 
            harmony, line_direction, unity_score, scale_score,
            design_style
        )
        
        return DesignPrinciplesResult(
            overall_score=round(overall_score, 3),
            grade=self._calculate_grade(overall_score),
            
            proportion=proportion,
            balance=balance,
            rhythm=rhythm,
            emphasis=emphasis,
            contrast=contrast,
            harmony=harmony,
            line_direction=line_direction,
            
            unity_score=round(unity_score, 3),
            scale_score=round(scale_score, 3),
            repetition_score=round(repetition_score, 3),
            gradation_score=round(gradation_score, 3),
            
            design_style=design_style,
            
            strengths=strengths,
            improvements=improvements,
            creative_notes=creative_notes
        )
    
    # ============== Proportion Analysis ==============
    
    def _analyze_proportion(self, items: List[Garment]) -> ProportionAnalysis:
        """
        Analyze proportions - relationship between top and bottom.
        
        Golden Ratio (1:1.618) creates visually pleasing proportions.
        Common fashion ratios: 1/3-2/3, 2/3-1/3, 50/50
        """
        tops = [i for i in items if i.attributes.category in 
               [GarmentCategory.TOP, GarmentCategory.OUTERWEAR]]
        bottoms = [i for i in items if i.attributes.category == GarmentCategory.BOTTOM]
        dresses = [i for i in items if i.attributes.category == GarmentCategory.DRESS]
        
        if dresses:
            # Dress proportions are built-in
            dress = dresses[0]
            if dress.attributes.length_type == LengthType.MIDI:
                return ProportionAnalysis(
                    score=0.9,
                    ratio="golden",
                    follows_golden_ratio=True,
                    recommendation="Midi length creates pleasing proportions."
                )
            elif dress.attributes.length_type == LengthType.MAXI:
                return ProportionAnalysis(
                    score=0.7,
                    ratio="elongated",
                    follows_golden_ratio=False,
                    recommendation="Long proportions - add a belt to create waist definition."
                )
            else:
                return ProportionAnalysis(
                    score=0.8,
                    ratio="balanced",
                    follows_golden_ratio=False,
                    recommendation="Good dress proportions."
                )
        
        if not tops or not bottoms:
            return ProportionAnalysis(
                score=0.6,
                ratio="incomplete",
                follows_golden_ratio=False,
                recommendation="Need top and bottom for proportion analysis."
            )
        
        top = tops[0]
        bottom = bottoms[0]
        
        # Analyze length combinations
        top_length = top.attributes.length_type or LengthType.REGULAR
        bottom_waist = bottom.attributes.waist_rise or WaistRise.MID_RISE
        bottom_length = bottom.attributes.length_type or LengthType.REGULAR
        
        # Golden ratio approximations in fashion
        # Crop top + high rise = ~1:2 (close to golden)
        # Regular top + mid rise = ~1:1 (50/50)
        # Longline + low rise = unbalanced
        
        score = 0.6
        ratio = "50:50"
        follows_golden = False
        
        # Best proportions
        if top_length == LengthType.CROP and bottom_waist == WaistRise.HIGH_RISE:
            score = 0.95
            ratio = "1:2 (golden approximation)"
            follows_golden = True
        elif top_length == LengthType.REGULAR and bottom_waist == WaistRise.HIGH_RISE:
            score = 0.85
            ratio = "1:1.5"
            follows_golden = True
        elif top_length == LengthType.LONGLINE and bottom_waist == WaistRise.LOW_RISE:
            score = 0.4
            ratio = "inverted"
            follows_golden = False
        elif top_length == LengthType.REGULAR:
            score = 0.7
            ratio = "balanced"
            follows_golden = False
        
        # Tucked in bonus
        if "tucked" in " ".join(top.attributes.style_tags).lower():
            score = min(score + 0.1, 1.0)
            follows_golden = True
        
        recommendation = self._get_proportion_recommendation(
            top_length, bottom_waist, follows_golden
        )
        
        return ProportionAnalysis(
            score=score,
            ratio=ratio,
            follows_golden_ratio=follows_golden,
            top_to_bottom_ratio=(1, 2) if follows_golden else (1, 1),
            recommendation=recommendation
        )
    
    def _get_proportion_recommendation(
        self, 
        top_length: LengthType, 
        bottom_waist: WaistRise,
        follows_golden: bool
    ) -> str:
        """Generate proportion recommendation."""
        if follows_golden:
            return "Great proportions! The golden ratio creates visual harmony."
        
        if top_length == LengthType.LONGLINE:
            return "Try tucking or adding a belt to define the waist and improve proportions."
        elif top_length == LengthType.REGULAR and bottom_waist == WaistRise.LOW_RISE:
            return "Consider high-waisted bottoms to create better proportions."
        else:
            return "Proportions are balanced. For more interest, try a 1/3-2/3 split."
    
    # ============== Balance Analysis ==============
    
    def _analyze_balance(
        self, 
        items: List[Garment],
        intended_style: Optional[DesignStyle]
    ) -> BalanceAnalysis:
        """
        Analyze visual balance - symmetric vs asymmetric.
        
        Symmetric: Formal, classic, stable
        Asymmetric: Dynamic, modern, fashion-forward
        """
        asymmetric_count = 0
        symmetric_count = 0
        
        asymmetric_keywords = {
            "asymmetric", "one shoulder", "off shoulder", "wrap",
            "draped", "side slit", "diagonal", "uneven"
        }
        
        for item in items:
            item_text = self._get_item_text(item)
            
            if any(kw in item_text for kw in asymmetric_keywords):
                asymmetric_count += 1
            else:
                symmetric_count += 1
        
        # Determine balance type
        if asymmetric_count > symmetric_count:
            balance_type = BalanceType.ASYMMETRIC
        else:
            balance_type = BalanceType.SYMMETRIC
        
        # Score based on intention and consistency
        is_intentional = True
        if asymmetric_count > 0 and symmetric_count > 0:
            # Mixed can work if intentional
            if intended_style in [DesignStyle.AVANT_GARDE, DesignStyle.MODERN]:
                score = 0.8
            else:
                score = 0.6
                is_intentional = asymmetric_count <= 1
        else:
            score = 0.9  # Consistent balance
        
        recommendation = self._get_balance_recommendation(
            balance_type, is_intentional, intended_style
        )
        
        return BalanceAnalysis(
            score=score,
            balance_type=balance_type,
            is_intentional=is_intentional,
            recommendation=recommendation
        )
    
    def _get_balance_recommendation(
        self,
        balance_type: BalanceType,
        is_intentional: bool,
        intended_style: Optional[DesignStyle]
    ) -> str:
        """Generate balance recommendation."""
        if balance_type == BalanceType.SYMMETRIC:
            return "Classic symmetric balance creates a stable, polished look."
        elif is_intentional:
            return "Intentional asymmetry adds dynamic interest and modernity."
        else:
            return "Multiple asymmetric elements may compete. Consider reducing to one statement piece."
    
    # ============== Rhythm Analysis ==============
    
    def _analyze_rhythm(self, items: List[Garment]) -> RhythmAnalysis:
        """
        Analyze visual rhythm - movement created by patterns, lines, repetition.
        """
        rhythm_sources = []
        
        for item in items:
            pattern = item.attributes.pattern
            details = item.attributes.details
            
            # Pattern creates rhythm
            if pattern.type != "solid":
                if pattern.type in ["stripes", "checks", "dots"]:
                    rhythm_sources.append(f"{pattern.type} pattern")
            
            # Repeated elements create rhythm
            if details.embellishments:
                if "buttons" in str(details.embellishments).lower():
                    rhythm_sources.append("repeated buttons")
                if "pleats" in str(details.embellishments).lower():
                    rhythm_sources.append("pleated design")
                if "ruffles" in str(details.embellishments).lower():
                    rhythm_sources.append("ruffle details")
        
        has_rhythm = len(rhythm_sources) > 0
        
        if len(rhythm_sources) >= 2:
            score = 0.9
        elif len(rhythm_sources) == 1:
            score = 0.75
        else:
            score = 0.5  # No rhythm isn't bad, just neutral
        
        recommendation = ""
        if not has_rhythm:
            recommendation = "Add visual rhythm with a patterned piece or repeated details."
        elif len(rhythm_sources) >= 2:
            recommendation = "Good visual rhythm guides the eye through the outfit."
        
        return RhythmAnalysis(
            score=score,
            has_rhythm=has_rhythm,
            rhythm_sources=rhythm_sources,
            recommendation=recommendation
        )
    
    # ============== Emphasis Analysis ==============
    
    def _analyze_emphasis(self, items: List[Garment]) -> EmphasisAnalysis:
        """
        Analyze focal point - there should be ONE clear point of attention.
        """
        focal_points = []
        
        for item in items:
            # Color emphasis
            color = item.attributes.color.primary.lower()
            if color in ["red", "orange", "yellow", "fuchsia", "electric blue", "lime"]:
                focal_points.append(EmphasisType.COLOR)
            
            # Neckline emphasis
            if item.attributes.neckline in [
                NecklineType.HALTER, NecklineType.OFF_SHOULDER, 
                NecklineType.COWL, NecklineType.SQUARE
            ]:
                focal_points.append(EmphasisType.NECKLINE)
            
            # Detail emphasis
            if item.attributes.details.embellishments:
                if any(e in ["sequins", "beading", "embroidery", "rhinestones"] 
                       for e in item.attributes.details.embellishments):
                    focal_points.append(EmphasisType.DETAIL)
            
            # Texture emphasis
            if item.attributes.material and item.attributes.material.texture:
                if item.attributes.material.texture.lower() in [
                    "velvet", "sequin", "metallic", "fur", "feather"
                ]:
                    focal_points.append(EmphasisType.TEXTURE)
            
            # Silhouette emphasis
            if item.attributes.silhouette:
                if any(s in item.attributes.silhouette.lower() 
                       for s in ["dramatic", "sculptural", "exaggerated", "voluminous"]):
                    focal_points.append(EmphasisType.SILHOUETTE)
        
        # Count unique focal points
        unique_focal = list(set(focal_points))
        
        has_single_focus = len(unique_focal) == 1
        is_clear = len(unique_focal) <= 2
        
        if len(unique_focal) == 0:
            score = 0.5
            recommendation = "Add a focal point - a statement neckline, bold color, or interesting detail."
        elif len(unique_focal) == 1:
            score = 1.0
            recommendation = f"Perfect! Clear focal point through {unique_focal[0].value}."
        elif len(unique_focal) == 2:
            score = 0.7
            recommendation = "Two focal points can work if they're not competing."
        else:
            score = 0.4
            recommendation = "Too many focal points confuse the eye. Choose one statement element."
        
        return EmphasisAnalysis(
            score=score,
            focal_points=unique_focal,
            has_single_focus=has_single_focus,
            is_clear=is_clear,
            recommendation=recommendation
        )
    
    # ============== Contrast Analysis ==============
    
    def _analyze_contrast(self, items: List[Garment]) -> ContrastAnalysis:
        """
        Analyze contrast - strong differences between elements.
        """
        contrast_types = []
        
        colors = [i.attributes.color.primary.lower() for i in items]
        textures = [i.attributes.material.texture.lower() if i.attributes.material and i.attributes.material.texture else "matte" for i in items]
        
        # Color contrast
        for c1, c2 in self.HIGH_CONTRAST_COLORS:
            if c1 in colors and c2 in colors:
                contrast_types.append("color contrast (high)")
                break
        
        # Check for any color variety
        if len(set(colors)) >= 2 and "color contrast (high)" not in contrast_types:
            contrast_types.append("color variety")
        
        # Texture contrast
        smooth = any(t in ["silk", "satin", "smooth"] for t in textures)
        textured = any(t in ["tweed", "knit", "wool", "cable", "ribbed"] for t in textures)
        if smooth and textured:
            contrast_types.append("texture contrast")
        
        # Shine contrast
        shiny = any(t in ["metallic", "sequin", "patent", "satin"] for t in textures)
        matte = any(t in ["matte", "cotton", "linen", "wool"] for t in textures)
        if shiny and matte:
            contrast_types.append("shine contrast")
        
        # Determine contrast level
        if len(contrast_types) >= 2:
            contrast_level = "high"
            score = 0.9
        elif len(contrast_types) == 1:
            contrast_level = "medium"
            score = 0.7
        else:
            contrast_level = "low"
            score = 0.5
        
        recommendation = ""
        if contrast_level == "low":
            recommendation = "Add contrast through color, texture, or shine to create visual interest."
        elif contrast_level == "high":
            recommendation = "Great use of contrast to create visual impact."
        
        return ContrastAnalysis(
            score=score,
            contrast_types=contrast_types,
            contrast_level=contrast_level,
            recommendation=recommendation
        )
    
    # ============== Harmony Analysis ==============
    
    def _analyze_harmony(self, items: List[Garment]) -> HarmonyAnalysis:
        """
        Analyze harmony - all elements work well together.
        """
        harmonious = []
        conflicting = []
        
        # Check color harmony
        colors = [i.attributes.color.primary.lower() for i in items]
        if self._colors_are_harmonious(colors):
            harmonious.append("color palette")
        else:
            conflicting.append("colors may clash")
        
        # Check formality harmony
        formality_levels = [i.attributes.formality_level for i in items]
        if len(set(formality_levels)) <= 2:
            harmonious.append("consistent formality")
        else:
            conflicting.append("mixed formality levels")
        
        # Check style harmony
        all_tags = []
        for item in items:
            all_tags.extend(item.attributes.style_tags)
        
        # Style conflict detection
        casual_tags = {"casual", "sporty", "streetwear", "relaxed"}
        formal_tags = {"formal", "elegant", "dressy", "sophisticated"}
        
        has_casual = any(t.lower() in casual_tags for t in all_tags)
        has_formal = any(t.lower() in formal_tags for t in all_tags)
        
        if has_casual and has_formal:
            conflicting.append("mixing casual and formal styles")
        else:
            harmonious.append("consistent style")
        
        # Calculate score
        harmony_score = len(harmonious) / (len(harmonious) + len(conflicting)) if (harmonious or conflicting) else 0.5
        
        recommendation = ""
        if conflicting:
            recommendation = f"Consider addressing: {', '.join(conflicting)}"
        else:
            recommendation = "All elements work harmoniously together."
        
        return HarmonyAnalysis(
            score=harmony_score,
            harmonious_elements=harmonious,
            conflicting_elements=conflicting,
            recommendation=recommendation
        )
    
    def _colors_are_harmonious(self, colors: List[str]) -> bool:
        """Check if colors work together."""
        # Simplified harmony check
        neutral_colors = {"black", "white", "gray", "grey", "beige", "cream", "navy", "tan"}
        
        non_neutral = [c for c in colors if c not in neutral_colors]
        
        # All neutrals = harmonious
        if not non_neutral:
            return True
        
        # One color + neutrals = harmonious
        if len(set(non_neutral)) == 1:
            return True
        
        # Check for complementary pairs
        complementary = [
            {"blue", "orange"}, {"red", "green"}, {"yellow", "purple"},
            {"pink", "green"}, {"coral", "teal"}
        ]
        
        color_set = set(non_neutral)
        for pair in complementary:
            if len(color_set & pair) == 2:
                return True
        
        # Check for analogous (similar) colors
        warm = {"red", "orange", "yellow", "coral", "peach", "rust", "burgundy"}
        cool = {"blue", "green", "teal", "purple", "lavender", "mint"}
        
        if color_set.issubset(warm) or color_set.issubset(cool):
            return True
        
        return len(non_neutral) <= 2
    
    # ============== Line Direction Analysis ==============
    
    def _analyze_lines(
        self, 
        items: List[Garment],
        body_type: Optional[str]
    ) -> LineAnalysis:
        """
        Analyze line directions and their visual effects.
        
        Vertical: Elongating, slimming
        Horizontal: Widening
        Diagonal: Dynamic movement
        """
        vertical_count = 0
        horizontal_count = 0
        diagonal_count = 0
        
        for item in items:
            item_text = self._get_item_text(item)
            
            if any(v in item_text for v in self.VERTICAL_ELEMENTS):
                vertical_count += 1
            if any(h in item_text for h in self.HORIZONTAL_ELEMENTS):
                horizontal_count += 1
            if any(d in item_text for d in self.DIAGONAL_ELEMENTS):
                diagonal_count += 1
        
        # Determine dominant direction
        max_count = max(vertical_count, horizontal_count, diagonal_count)
        
        if max_count == 0:
            dominant = LineDirection.MIXED
            visual_effect = "neutral"
        elif vertical_count == max_count:
            dominant = LineDirection.VERTICAL
            visual_effect = "elongating and slimming"
        elif diagonal_count == max_count:
            dominant = LineDirection.DIAGONAL
            visual_effect = "dynamic and modern"
        else:
            dominant = LineDirection.HORIZONTAL
            visual_effect = "widening"
        
        # Score based on consistency and body type
        flattering = True
        score = 0.7
        
        if dominant == LineDirection.VERTICAL:
            score = 0.9
            flattering = True  # Generally flattering
        elif dominant == LineDirection.DIAGONAL:
            score = 0.85
            flattering = True
        elif dominant == LineDirection.HORIZONTAL:
            score = 0.6
            # Check if appropriate for body type
            if body_type and body_type.lower() in ["pear", "rectangle"]:
                flattering = True
                score = 0.8
            else:
                flattering = False
        
        recommendation = self._get_line_recommendation(dominant, visual_effect, flattering)
        
        return LineAnalysis(
            score=score,
            dominant_direction=dominant,
            visual_effect=visual_effect,
            flattering_for_body=flattering,
            recommendation=recommendation
        )
    
    def _get_line_recommendation(
        self,
        direction: LineDirection,
        effect: str,
        flattering: bool
    ) -> str:
        """Generate line direction recommendation."""
        if direction == LineDirection.VERTICAL:
            return f"Vertical lines create a {effect} effect - very flattering."
        elif direction == LineDirection.DIAGONAL:
            return f"Diagonal lines add {effect} - great for visual interest."
        elif direction == LineDirection.HORIZONTAL and not flattering:
            return "Horizontal lines can widen. Add vertical elements like a long necklace or V-neck."
        elif direction == LineDirection.MIXED:
            return "Consider adding more vertical lines for an elongating effect."
        return f"Lines create a {effect} effect."
    
    # ============== Additional Analyses ==============
    
    def _analyze_unity(self, items: List[Garment]) -> float:
        """Analyze if all pieces feel like they belong together."""
        # Check category appropriateness
        categories = [i.attributes.category for i in items]
        
        # Check for orphan pieces
        has_top = any(c in [GarmentCategory.TOP, GarmentCategory.DRESS] for c in categories)
        has_bottom = any(c in [GarmentCategory.BOTTOM, GarmentCategory.DRESS] for c in categories)
        
        if has_top and has_bottom:
            base_score = 0.7
        else:
            base_score = 0.4
        
        # Check style consistency
        all_tags = []
        for item in items:
            all_tags.extend([t.lower() for t in item.attributes.style_tags])
        
        if all_tags:
            # More repeated tags = more unity
            unique_ratio = len(set(all_tags)) / len(all_tags)
            unity_bonus = (1 - unique_ratio) * 0.3
        else:
            unity_bonus = 0
        
        return min(base_score + unity_bonus, 1.0)
    
    def _analyze_scale(self, items: List[Garment]) -> float:
        """Analyze if element sizes are appropriate for each other."""
        # Check pattern scales
        scales = []
        for item in items:
            if item.attributes.pattern.scale:
                scales.append(item.attributes.pattern.scale.lower())
        
        if not scales:
            return 0.7  # No patterns to check
        
        # Same scale patterns can clash
        if len(scales) >= 2 and len(set(scales)) == 1:
            return 0.5
        
        # Different scales work better
        return 0.85
    
    def _analyze_repetition(self, items: List[Garment]) -> float:
        """Analyze effective use of repetition."""
        repeated_elements = []
        
        # Check for repeated colors
        colors = [i.attributes.color.primary.lower() for i in items]
        for color in set(colors):
            if colors.count(color) >= 2:
                repeated_elements.append(f"color: {color}")
        
        # Check for repeated materials
        materials = [i.attributes.material.primary.lower() if i.attributes.material else "" for i in items]
        for mat in set(materials):
            if mat and materials.count(mat) >= 2:
                repeated_elements.append(f"material: {mat}")
        
        if len(repeated_elements) >= 2:
            return 0.9
        elif len(repeated_elements) == 1:
            return 0.7
        return 0.5
    
    def _analyze_gradation(self, items: List[Garment]) -> float:
        """Analyze use of gradation (progressive variation)."""
        # Ombré effect, graduated colors, etc.
        for item in items:
            desc = (item.attributes.pattern.description or "").lower()
            if any(g in desc for g in ["ombre", "gradient", "graduated", "fade"]):
                return 0.9
        
        # Check for tonal dressing (same color family, different shades)
        colors = [i.attributes.color.primary.lower() for i in items]
        
        # Simplified check for tonal
        color_bases = set()
        for color in colors:
            for base in ["blue", "red", "green", "pink", "purple", "brown"]:
                if base in color:
                    color_bases.add(base)
        
        if len(color_bases) == 1 and len(colors) >= 2:
            return 0.8  # Tonal dressing
        
        return 0.5
    
    # ============== Helpers ==============
    
    def _get_item_text(self, item: Garment) -> str:
        """Get searchable text representation of item."""
        attrs = item.attributes
        parts = [
            attrs.category.value,
            attrs.subcategory or "",
            attrs.silhouette or "",
            attrs.fit or "",
            attrs.neckline.value if attrs.neckline else "",
            attrs.pattern.type,
            " ".join(attrs.style_tags),
        ]
        return " ".join(parts).lower()
    
    def _classify_design_style(
        self,
        items: List[Garment],
        balance: BalanceAnalysis,
        emphasis: EmphasisAnalysis,
        contrast: ContrastAnalysis,
        intended: Optional[DesignStyle]
    ) -> DesignStyle:
        """Classify the overall design style."""
        if intended:
            return intended
        
        # Count style indicators
        all_tags = []
        for item in items:
            all_tags.extend([t.lower() for t in item.attributes.style_tags])
        
        if any(t in ["avant-garde", "experimental", "sculptural"] for t in all_tags):
            return DesignStyle.AVANT_GARDE
        
        if balance.balance_type == BalanceType.ASYMMETRIC:
            if len(emphasis.focal_points) >= 2:
                return DesignStyle.MAXIMALIST
            return DesignStyle.MODERN
        
        if len(emphasis.focal_points) == 0 and contrast.contrast_level == "low":
            return DesignStyle.MINIMALIST
        
        if any(t in ["classic", "timeless", "traditional"] for t in all_tags):
            return DesignStyle.CLASSIC
        
        return DesignStyle.MODERN
    
    def _generate_summary(
        self,
        proportion: ProportionAnalysis,
        balance: BalanceAnalysis,
        rhythm: RhythmAnalysis,
        emphasis: EmphasisAnalysis,
        contrast: ContrastAnalysis,
        harmony: HarmonyAnalysis,
        line: LineAnalysis,
        unity: float,
        scale: float,
        style: DesignStyle
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate strengths, improvements, and creative notes."""
        strengths = []
        improvements = []
        creative_notes = []
        
        # Proportion
        if proportion.follows_golden_ratio:
            strengths.append("Excellent proportions following golden ratio principles")
        elif proportion.score < 0.6:
            improvements.append(proportion.recommendation)
        
        # Emphasis
        if emphasis.has_single_focus:
            strengths.append(f"Clear focal point through {emphasis.focal_points[0].value}")
        elif len(emphasis.focal_points) > 2:
            improvements.append("Too many focal points - simplify")
        elif len(emphasis.focal_points) == 0:
            improvements.append("Add a statement element as focal point")
        
        # Harmony
        if harmony.score >= 0.8:
            strengths.append("All elements work harmoniously")
        elif harmony.conflicting_elements:
            improvements.append(f"Resolve conflicts: {', '.join(harmony.conflicting_elements)}")
        
        # Lines
        if line.dominant_direction == LineDirection.VERTICAL:
            strengths.append("Vertical lines create an elongating effect")
        
        # Contrast
        if contrast.contrast_level == "high":
            strengths.append("Strong use of contrast creates visual impact")
        
        # Style-specific notes
        if style == DesignStyle.AVANT_GARDE:
            creative_notes.append("Rule-breaking elements add creative interest")
        elif style == DesignStyle.MINIMALIST:
            creative_notes.append("Minimalist approach emphasizes quality and fit")
        
        # Unity
        if unity >= 0.8:
            strengths.append("Strong sense of unity - all pieces belong together")
        
        return strengths, improvements, creative_notes
    
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
    
    def _empty_result(self) -> DesignPrinciplesResult:
        """Return empty result."""
        empty_proportion = ProportionAnalysis(0.5, "unknown", False, None, "Add items")
        empty_balance = BalanceAnalysis(0.5, BalanceType.SYMMETRIC, True, "Add items")
        empty_rhythm = RhythmAnalysis(0.5, False, [], "Add items")
        empty_emphasis = EmphasisAnalysis(0.5, [], False, False, "Add items")
        empty_contrast = ContrastAnalysis(0.5, [], "low", "Add items")
        empty_harmony = HarmonyAnalysis(0.5, [], [], "Add items")
        empty_line = LineAnalysis(0.5, LineDirection.MIXED, "neutral", True, "Add items")
        
        return DesignPrinciplesResult(
            overall_score=0.5,
            grade="C",
            proportion=empty_proportion,
            balance=empty_balance,
            rhythm=empty_rhythm,
            emphasis=empty_emphasis,
            contrast=empty_contrast,
            harmony=empty_harmony,
            line_direction=empty_line,
            unity_score=0.5,
            scale_score=0.5,
            repetition_score=0.5,
            gradation_score=0.5,
            design_style=DesignStyle.CLASSIC,
            strengths=[],
            improvements=["Add items to analyze"],
            creative_notes=[]
        )


# ============== Convenience Functions ==============

def get_design_principles_score(items: List[Garment]) -> float:
    """Get overall design principles score (0-1)."""
    scorer = DesignPrinciplesScorer()
    result = scorer.analyze_outfit(items)
    return result.overall_score


def get_design_grade(items: List[Garment]) -> str:
    """Get design principles grade (A-F)."""
    scorer = DesignPrinciplesScorer()
    result = scorer.analyze_outfit(items)
    return result.grade


def check_golden_ratio(items: List[Garment]) -> bool:
    """Check if outfit follows golden ratio proportions."""
    scorer = DesignPrinciplesScorer()
    result = scorer.analyze_outfit(items)
    return result.proportion.follows_golden_ratio


def get_focal_point(items: List[Garment]) -> List[str]:
    """Get the focal point(s) of the outfit."""
    scorer = DesignPrinciplesScorer()
    result = scorer.analyze_outfit(items)
    return [fp.value for fp in result.emphasis.focal_points]
