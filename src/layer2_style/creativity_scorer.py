
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from src.core.models import (
    Garment, GarmentCategory, FormalityLevel,
    PatternInfo, ColorInfo, Occasion
)
from src.core import get_logger

logger = get_logger(__name__)

_STYLE_RULES_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "config" / "data" / "style_rules_config.json"
)


def _load_creativity_config() -> dict:
    """Load the creativity section of style_rules_config.json."""
    try:
        with open(_STYLE_RULES_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("creativity", {})
    except Exception as exc:
        logger.warning("Could not load style_rules_config.json: %s — using built-in defaults.", exc)
        return {}


class CreativityLevel(str, Enum):
    """Creativity classification."""
    SAFE = "safe"                    # Following all rules
    MODERATE = "moderate"            # Slight rule bending
    CREATIVE = "creative"            # Intentional rule breaking
    AVANT_GARDE = "avant_garde"      # Experimental, fashion-forward
    CHAOTIC = "chaotic"              # Breaking rules without purpose


class RuleBreakType(str, Enum):
    """Types of rules that can be broken."""
    COLOR_CLASH = "color_clash"                # Unexpected color combos
    FORMALITY_MIX = "formality_mix"            # Mixing formal + casual
    PATTERN_OVERLOAD = "pattern_overload"      # Multiple bold patterns
    PROPORTION_INVERSION = "proportion_inversion"  # Unusual proportions
    TEXTURE_CLASH = "texture_clash"            # Contrasting textures
    VOLUME_EXTREME = "volume_extreme"          # All oversized or extreme fit
    STYLE_FUSION = "style_fusion"              # Mixing style tribes
    UNEXPECTED_PAIRING = "unexpected_pairing"  # Unusual item combinations


@dataclass
class RuleBreak:
    """A detected rule break."""
    rule_type: RuleBreakType
    description: str
    severity: float  # 0-1, how much the rule is broken
    is_intentional: bool  # Does it seem purposeful?
    works_aesthetically: bool  # Does it still look good?


@dataclass
class CreativityResult:
    """Complete creativity analysis."""
    creativity_score: float  # 0-1
    creativity_level: CreativityLevel
    
    # Component scores
    rule_breaking_index: float  # How many rules broken (0-1)
    intentionality_score: float  # How purposeful (0-1)
    novelty_score: float  # How unexpected (0-1)
    risk_score: float  # How challenging (0-1)
    
    # Details
    rules_broken: List[RuleBreak]
    creative_elements: List[str]
    risk_elements: List[str]
    
    # Verdict
    is_successful_creativity: bool  # Creative AND works
    fashion_forward_score: float  # Creativity that advances style
    
    recommendation: str


class CreativityScorer:
    """
    Evaluates outfit creativity - the intentional breaking of rules
    to create something new and interesting.

    Domain knowledge (safe combos, creative combos, style tribes, risk pieces,
    scoring weights and thresholds) is loaded from
    config/data/style_rules_config.json so stylists can tune it without code changes.

    Formula:
    creativity = (rule_breaking * w_rb) + (intentionality * w_int) +
                 (novelty * w_nov) + (risk * w_risk)

    fashion_forward = creativity * style_success_rate
    """

    def __init__(self):
        cfg = _load_creativity_config()

        self.weights: Dict[str, float] = cfg.get("scoring_weights", {
            "rule_breaking": 0.30,
            "intentionality": 0.30,
            "novelty": 0.25,
            "risk": 0.15,
        })

        # Safe combinations (breaking these = creative)
        self.SAFE_COMBOS: dict = cfg.get("safe_combos", {
            "colors": [["black", "white", "gray"], ["navy", "white", "beige"], ["monochrome"]],
            "formality": "consistent",
            "patterns": "one_or_none",
            "proportions": "balanced",
        })

        # Unexpected but potentially successful combinations
        _cc = cfg.get("creative_combos", {})
        self.CREATIVE_COMBOS: dict = {
            "color_clashes_that_work": [tuple(p) for p in _cc.get("color_clashes_that_work", [
                ("pink", "red"), ("orange", "pink"), ("purple", "green"),
                ("brown", "black"), ("navy", "black"),
            ])],
            "formality_fusions": [tuple(p) for p in _cc.get("formality_fusions", [
                ("sneakers", "formal"), ("hoodie", "tailored"), ("t-shirt", "evening"),
            ])],
            "style_fusions": [tuple(p) for p in _cc.get("style_fusions", [
                ("sporty", "elegant"), ("punk", "preppy"),
                ("bohemian", "minimalist"), ("streetwear", "tailored"),
            ])],
        }

        # High-risk pieces that require skill to style
        self.RISK_PIECES: dict = cfg.get("risk_pieces", {
            "patterns": ["animal", "camo", "tie_dye", "abstract", "novelty"],
            "colors": ["neon", "lime", "hot pink", "orange", "yellow"],
            "styles": ["avant-garde", "deconstructed", "oversized", "sheer"],
            "combos": ["multiple patterns", "color blocking", "mixed metals"],
        })

        # Style tribes for style-fusion detection
        self._style_tribes: Dict[str, List[str]] = cfg.get("style_tribes", {
            "sporty":     ["athletic", "sporty", "activewear", "athleisure"],
            "elegant":    ["elegant", "sophisticated", "refined", "dressy"],
            "punk":       ["punk", "edgy", "grunge", "rock"],
            "preppy":     ["preppy", "classic", "collegiate", "traditional"],
            "bohemian":   ["boho", "bohemian", "hippie", "free-spirited"],
            "minimalist": ["minimal", "minimalist", "clean", "simple"],
            "streetwear": ["street", "urban", "hip-hop", "skate"],
            "romantic":   ["romantic", "feminine", "soft", "delicate"],
        })

        # Novelty & risk scoring granularity
        self._novel_materials: List[str]   = cfg.get("novel_materials", ["pvc", "latex", "neoprene", "mesh", "vinyl"])
        self._novel_patterns:  List[str]   = cfg.get("novel_patterns",  ["abstract", "tie_dye", "camo", "novelty"])
        self._novel_tags:      List[str]   = cfg.get("novel_style_tags", [
            "deconstructed", "avant-garde", "experimental", "architectural", "sculptural", "transformable",
        ])

        # Coherent rule-break pairs (same rule-types that reinforce each other)
        self._coherent_pairs: List[set] = [
            {rb[0], rb[1]} for rb in cfg.get("coherent_rule_break_pairs", [
                ["formality_mix", "style_fusion"],
                ["color_clash",   "pattern_overload"],
                ["volume_extreme","proportion_inversion"],
            ])
        ]

        # Numeric thresholds
        _t = cfg.get("thresholds", {})
        self._t_safe_max       = _t.get("safe_max_score",       0.20)
        self._t_moderate_max   = _t.get("moderate_max_score",   0.40)
        self._t_creative_max   = _t.get("creative_max_score",   0.70)
        self._t_int_creative   = _t.get("intentionality_min_creative",    0.60)
        self._t_int_avant      = _t.get("intentionality_min_avant_garde", 0.50)
        self._t_ff_min         = _t.get("fashion_forward_min",  0.50)
        self._t_success_creat  = _t.get("success_creativity_min", 0.30)
        self._t_success_style  = _t.get("success_style_min",    0.60)
        self._t_success_intent = _t.get("success_intentionality_min", 0.60)
        self._t_coherence_bonus = _t.get("max_intentional_coherence_bonus", 0.15)
        self._t_rb_divisor     = _t.get("max_rule_breaking_severity_divisor", 3.0)
        self._t_novelty_mat    = _t.get("novelty_material_score", 0.30)
        self._t_novelty_pat    = _t.get("novelty_pattern_score",  0.20)
        self._t_novelty_tag    = _t.get("novelty_tag_score",      0.30)
        self._t_risk_pat       = _t.get("risk_pattern_score",     0.25)
        self._t_risk_col       = _t.get("risk_color_score",       0.20)
        self._t_risk_sty       = _t.get("risk_style_score",       0.30)
        self._t_risk_sheer     = _t.get("risk_sheer_score",       0.20)
    
    def analyze_creativity(
        self,
        items: List[Garment],
        style_score: Optional[float] = None,  # From other scorers
        user_style_level: str = "intermediate"  # beginner, intermediate, advanced
    ) -> CreativityResult:
        """
        Analyze outfit creativity.
        
        Args:
            items: Outfit garments
            style_score: Overall style score from other scorers (0-1)
            user_style_level: User's styling experience level
            
        Returns:
            CreativityResult with full analysis
        """
        if not items:
            return self._empty_result()
        
        # Detect rule breaks
        rules_broken = self._detect_rule_breaks(items)
        
        # Calculate component scores
        rule_breaking_index = self._calculate_rule_breaking_index(rules_broken)
        intentionality_score = self._calculate_intentionality(rules_broken, items)
        novelty_score = self._calculate_novelty(items)
        risk_score = self._calculate_risk(items)
        
        # Calculate overall creativity
        creativity_score = (
            self.weights["rule_breaking"] * rule_breaking_index +
            self.weights["intentionality"] * intentionality_score +
            self.weights["novelty"] * novelty_score +
            self.weights["risk"] * risk_score
        )
        
        # Determine creativity level
        creativity_level = self._classify_creativity(
            creativity_score, intentionality_score, rules_broken
        )
        
        # Extract creative and risk elements
        creative_elements = self._extract_creative_elements(items, rules_broken)
        risk_elements = self._extract_risk_elements(items)
        
        # Determine if creativity is successful
        is_successful = self._evaluate_success(
            creativity_score, style_score, intentionality_score
        )
        
        # Fashion forward score (creativity that works)
        if style_score is not None:
            fashion_forward = creativity_score * style_score
        else:
            fashion_forward = creativity_score * intentionality_score
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            creativity_level, is_successful, user_style_level, rules_broken
        )
        
        return CreativityResult(
            creativity_score=round(creativity_score, 3),
            creativity_level=creativity_level,
            
            rule_breaking_index=round(rule_breaking_index, 3),
            intentionality_score=round(intentionality_score, 3),
            novelty_score=round(novelty_score, 3),
            risk_score=round(risk_score, 3),
            
            rules_broken=rules_broken,
            creative_elements=creative_elements,
            risk_elements=risk_elements,
            
            is_successful_creativity=is_successful,
            fashion_forward_score=round(fashion_forward, 3),
            
            recommendation=recommendation
        )
    
    def _detect_rule_breaks(self, items: List[Garment]) -> List[RuleBreak]:
        """Detect all rule breaks in the outfit."""
        breaks = []
        
        # Check color clashes
        color_break = self._check_color_rules(items)
        if color_break:
            breaks.append(color_break)
        
        # Check formality mixing
        formality_break = self._check_formality_rules(items)
        if formality_break:
            breaks.append(formality_break)
        
        # Check pattern overload
        pattern_break = self._check_pattern_rules(items)
        if pattern_break:
            breaks.append(pattern_break)
        
        # Check proportion inversion
        proportion_break = self._check_proportion_rules(items)
        if proportion_break:
            breaks.append(proportion_break)
        
        # Check texture clashes
        texture_break = self._check_texture_rules(items)
        if texture_break:
            breaks.append(texture_break)
        
        # Check style fusion
        style_break = self._check_style_fusion(items)
        if style_break:
            breaks.append(style_break)
        
        return breaks
    
    def _check_color_rules(self, items: List[Garment]) -> Optional[RuleBreak]:
        """Check for creative color rule breaking."""
        colors = [i.attributes.color.primary.lower() for i in items]
        
        # Check for known creative color clashes
        for c1, c2 in self.CREATIVE_COMBOS["color_clashes_that_work"]:
            if c1 in colors and c2 in colors:
                return RuleBreak(
                    rule_type=RuleBreakType.COLOR_CLASH,
                    description=f"Creative color pairing: {c1} + {c2}",
                    severity=0.6,
                    is_intentional=True,  # Known creative combo
                    works_aesthetically=True
                )
        
        # Check for unexpected color combinations
        warm = {"red", "orange", "yellow", "coral", "rust", "burgundy"}
        cool = {"blue", "green", "purple", "teal", "lavender"}
        
        has_warm = any(c in warm for c in colors)
        has_cool = any(c in cool for c in colors)
        
        if has_warm and has_cool and len(colors) >= 3:
            # Mixing warm and cool
            return RuleBreak(
                rule_type=RuleBreakType.COLOR_CLASH,
                description="Mixing warm and cool colors",
                severity=0.5,
                is_intentional=False,  # Need to check coherence
                works_aesthetically=False  # Unknown
            )
        
        return None
    
    def _check_formality_rules(self, items: List[Garment]) -> Optional[RuleBreak]:
        """Check for creative formality mixing."""
        formality_order = [
            FormalityLevel.VERY_CASUAL,
            FormalityLevel.CASUAL,
            FormalityLevel.SMART_CASUAL,
            FormalityLevel.BUSINESS_CASUAL,
            FormalityLevel.BUSINESS,
            FormalityLevel.FORMAL,
            FormalityLevel.BLACK_TIE,
        ]
        
        levels = [i.attributes.formality_level for i in items]
        indices = [formality_order.index(l) if l in formality_order else 1 for l in levels]
        
        spread = max(indices) - min(indices)
        
        if spread >= 3:
            # Major formality mixing
            categories = [i.attributes.category.value for i in items]
            
            # Check for known creative fusions
            is_intentional = any(
                any(f[0] in str(items).lower() and f[1] in str(levels).lower() 
                    for f in self.CREATIVE_COMBOS["formality_fusions"])
                for _ in [1]
            )
            
            return RuleBreak(
                rule_type=RuleBreakType.FORMALITY_MIX,
                description=f"Mixing {min(levels).value} with {max(levels).value}",
                severity=spread / 6,
                is_intentional=is_intentional,
                works_aesthetically=spread <= 4
            )
        
        return None
    
    def _check_pattern_rules(self, items: List[Garment]) -> Optional[RuleBreak]:
        """Check for creative pattern mixing."""
        patterns = [i.attributes.pattern for i in items 
                   if i.attributes.pattern.type.lower() != "solid"]
        
        if len(patterns) >= 2:
            pattern_types = [p.type for p in patterns]
            
            # Check scale variation (intentional if different scales)
            scales = [p.scale for p in patterns if p.scale]
            has_scale_contrast = len(set(scales)) > 1 if scales else False
            
            return RuleBreak(
                rule_type=RuleBreakType.PATTERN_OVERLOAD,
                description=f"Multiple patterns: {', '.join(pattern_types)}",
                severity=min(len(patterns) / 3, 1.0),
                is_intentional=has_scale_contrast,
                works_aesthetically=has_scale_contrast
            )
        
        return None
    
    def _check_proportion_rules(self, items: List[Garment]) -> Optional[RuleBreak]:
        """Check for creative proportion breaking."""
        tops = [i for i in items if i.attributes.category in 
               [GarmentCategory.TOP, GarmentCategory.OUTERWEAR]]
        bottoms = [i for i in items if i.attributes.category == GarmentCategory.BOTTOM]
        
        if not tops or not bottoms:
            return None
        
        top_fit = tops[0].attributes.fit or "regular"
        bottom_fit = bottoms[0].attributes.fit or "regular"
        
        # Extreme volume combinations
        if top_fit in ["oversized", "voluminous"] and bottom_fit in ["oversized", "wide"]:
            return RuleBreak(
                rule_type=RuleBreakType.VOLUME_EXTREME,
                description="All-oversized silhouette",
                severity=0.7,
                is_intentional=True,  # Deliberate choice
                works_aesthetically=True  # Can be very fashion-forward
            )
        
        return None
    
    def _check_texture_rules(self, items: List[Garment]) -> Optional[RuleBreak]:
        """Check for creative texture clashing."""
        textures = []
        for item in items:
            if item.attributes.material and item.attributes.material.texture:
                textures.append(item.attributes.material.texture.lower())
        
        if len(textures) < 2:
            return None
        
        # Contrasting texture pairs
        shiny = any(t in ["silk", "satin", "metallic", "sequin", "patent"] for t in textures)
        matte = any(t in ["cotton", "wool", "linen", "matte"] for t in textures)
        textured = any(t in ["tweed", "corduroy", "cable", "ribbed", "velvet"] for t in textures)
        
        if shiny and textured:
            return RuleBreak(
                rule_type=RuleBreakType.TEXTURE_CLASH,
                description="Mixing shiny and heavily textured fabrics",
                severity=0.5,
                is_intentional=True,  # Often intentional
                works_aesthetically=True
            )
        
        return None
    
    def _check_style_fusion(self, items: List[Garment]) -> Optional[RuleBreak]:
        """Check for creative style tribe fusion."""
        all_tags = []
        for item in items:
            all_tags.extend([t.lower() for t in item.attributes.style_tags])
        
        style_tribes = self._style_tribes
        
        found_tribes = set()
        for tribe, keywords in style_tribes.items():
            if any(kw in all_tags for kw in keywords):
                found_tribes.add(tribe)
        
        if len(found_tribes) >= 2:
            # Check for known creative fusions
            fusion_pair = tuple(sorted(found_tribes)[:2])
            known_fusions = [tuple(sorted(f)) for f in self.CREATIVE_COMBOS["style_fusions"]]
            is_known_fusion = fusion_pair in known_fusions
            
            return RuleBreak(
                rule_type=RuleBreakType.STYLE_FUSION,
                description=f"Style fusion: {' + '.join(found_tribes)}",
                severity=0.6,
                is_intentional=is_known_fusion,
                works_aesthetically=is_known_fusion or len(found_tribes) == 2
            )
        
        return None
    
    def _calculate_rule_breaking_index(self, rules_broken: List[RuleBreak]) -> float:
        """Calculate how much rules are being broken."""
        if not rules_broken:
            return 0.0
        
        total_severity = sum(rb.severity for rb in rules_broken)
        return min(total_severity / self._t_rb_divisor, 1.0)
    
    def _calculate_intentionality(
        self, 
        rules_broken: List[RuleBreak],
        items: List[Garment]
    ) -> float:
        """Calculate how intentional/purposeful the rule breaking seems."""
        if not rules_broken:
            return 0.5  # Neutral
        
        intentional_count = sum(1 for rb in rules_broken if rb.is_intentional)
        base_score = intentional_count / len(rules_broken)
        
        # Bonus for coherent rule-breaking
        # If multiple rules broken but they "go together"
        types = [rb.rule_type.value for rb in rules_broken]
        
        for pair in self._coherent_pairs:
            if pair.issubset(set(types)):
                base_score += self._t_coherence_bonus
        
        return min(base_score, 1.0)
    
    def _calculate_novelty(self, items: List[Garment]) -> float:
        """Calculate how novel/unexpected the outfit is."""
        novelty_points = 0
        
        for item in items:
            # Novel materials
            if item.attributes.material:
                material = item.attributes.material.primary.lower()
                if material in self._novel_materials:
                    novelty_points += self._t_novelty_mat
            
            # Novel patterns
            pattern = item.attributes.pattern.type.lower()
            if pattern in self._novel_patterns:
                novelty_points += self._t_novelty_pat
            
            # Novel style tags
            if any(nt in " ".join(item.attributes.style_tags).lower() for nt in self._novel_tags):
                novelty_points += self._t_novelty_tag
        
        return min(novelty_points, 1.0)
    
    def _calculate_risk(self, items: List[Garment]) -> float:
        """Calculate how much risk is being taken."""
        risk_points = 0
        
        for item in items:
            # Risky patterns
            pattern = item.attributes.pattern.type.lower()
            if pattern in self.RISK_PIECES["patterns"]:
                risk_points += self._t_risk_pat
            
            # Risky colors
            color = item.attributes.color.primary.lower()
            if any(rc in color for rc in self.RISK_PIECES["colors"]):
                risk_points += self._t_risk_col
            
            # Risky styles
            tags = " ".join(item.attributes.style_tags).lower()
            if any(rs in tags for rs in self.RISK_PIECES["styles"]):
                risk_points += self._t_risk_sty
            
            # Sheer/transparent
            if item.attributes.details.transparency.value != "opaque":
                risk_points += self._t_risk_sheer
        
        return min(risk_points, 1.0)
    
    def _classify_creativity(
        self,
        score: float,
        intentionality: float,
        rules_broken: List[RuleBreak]
    ) -> CreativityLevel:
        """Classify the creativity level."""
        if score < self._t_safe_max:
            return CreativityLevel.SAFE
        elif score < self._t_moderate_max:
            return CreativityLevel.MODERATE
        elif score < self._t_creative_max:
            if intentionality >= self._t_int_creative:
                return CreativityLevel.CREATIVE
            else:
                return CreativityLevel.MODERATE
        else:
            if intentionality >= self._t_int_avant:
                return CreativityLevel.AVANT_GARDE
            else:
                return CreativityLevel.CHAOTIC
    
    def _extract_creative_elements(
        self, 
        items: List[Garment],
        rules_broken: List[RuleBreak]
    ) -> List[str]:
        """Extract the creative elements in the outfit."""
        elements = []
        
        for rb in rules_broken:
            if rb.is_intentional and rb.works_aesthetically:
                elements.append(rb.description)
        
        # Add other creative observations
        for item in items:
            if "statement" in " ".join(item.attributes.style_tags).lower():
                elements.append(f"Statement {item.attributes.category.value}")
        
        return elements
    
    def _extract_risk_elements(self, items: List[Garment]) -> List[str]:
        """Extract risk-taking elements."""
        risks = []
        
        for item in items:
            pattern = item.attributes.pattern.type.lower()
            if pattern in self.RISK_PIECES["patterns"]:
                risks.append(f"{pattern.title()} pattern")
            
            color = item.attributes.color.primary.lower()
            if any(rc in color for rc in self.RISK_PIECES["colors"]):
                risks.append(f"Bold {color} color")
        
        return risks
    
    def _evaluate_success(
        self,
        creativity: float,
        style_score: Optional[float],
        intentionality: float
    ) -> bool:
        """Determine if the creativity is successful."""
        if style_score is not None:
            return creativity >= self._t_success_creat and style_score >= self._t_success_style
        else:
            return creativity >= self._t_success_creat and intentionality >= self._t_success_intent
    
    def _generate_recommendation(
        self,
        level: CreativityLevel,
        is_successful: bool,
        user_level: str,
        rules_broken: List[RuleBreak]
    ) -> str:
        """Generate creativity recommendation."""
        if level == CreativityLevel.SAFE:
            if user_level == "beginner":
                return "Safe choices! As you gain confidence, try introducing one bold element."
            else:
                return "Very safe outfit. Try breaking one rule intentionally - add an unexpected color or mix a pattern."
        
        elif level == CreativityLevel.MODERATE:
            return "Some creative touches! You're on the right track."
        
        elif level == CreativityLevel.CREATIVE:
            if is_successful:
                return "Excellent creative choices! Your intentional rule-breaking creates visual interest."
            else:
                return "Creative attempt! Try ensuring one unifying element ties the look together."
        
        elif level == CreativityLevel.AVANT_GARDE:
            if is_successful:
                return "Fashion-forward creativity! You're successfully pushing boundaries."
            else:
                return "Very experimental! Consider adding one grounding element for balance."
        
        else:  # CHAOTIC
            return "Bold choices, but the look feels random. Try breaking fewer rules, but more intentionally."
    
    def _empty_result(self) -> CreativityResult:
        """Return empty result."""
        return CreativityResult(
            creativity_score=0.0,
            creativity_level=CreativityLevel.SAFE,
            rule_breaking_index=0.0,
            intentionality_score=0.0,
            novelty_score=0.0,
            risk_score=0.0,
            rules_broken=[],
            creative_elements=[],
            risk_elements=[],
            is_successful_creativity=False,
            fashion_forward_score=0.0,
            recommendation="Add items to analyze"
        )


# ============== Convenience Functions ==============

def get_creativity_score(
    items: List[Garment],
    style_score: Optional[float] = None
) -> float:
    """Get overall creativity score (0-1)."""
    scorer = CreativityScorer()
    result = scorer.analyze_creativity(items, style_score)
    return result.creativity_score


def get_creativity_level(items: List[Garment]) -> str:
    """Get creativity level classification."""
    scorer = CreativityScorer()
    result = scorer.analyze_creativity(items)
    return result.creativity_level.value


def is_fashion_forward(
    items: List[Garment],
    style_score: float
) -> bool:
    """Check if outfit is fashion-forward (creative AND works)."""
    scorer = CreativityScorer()
    result = scorer.analyze_creativity(items, style_score)
    return result.fashion_forward_score >= 0.5


def get_fashion_forward_score(
    items: List[Garment],
    style_score: float
) -> float:
    """Get fashion-forward score (creativity * style success)."""
    scorer = CreativityScorer()
    result = scorer.analyze_creativity(items, style_score)
    return result.fashion_forward_score
