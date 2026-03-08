# Layer 2: Style Intelligence Model
from .compatibility_scorer import CompatibilityScorer
from .color_harmony import ColorHarmonyAnalyzer
from .silhouette_analyzer import SilhouetteAnalyzer
from .formality_matcher import FormalityMatcher
from .style_model import StyleIntelligenceModel

# Detailed scoring rules
from .seven_point_rule import SevenPointRuleScorer, SevenPointResult, calculate_seven_point_score
from .season_color_harmony import SeasonColorHarmonyScorer, ColorSeason, SkinUndertone, SeasonColorResult
from .proportion_scorer import ProportionScorer, ProportionRatio, ProportionResult
from .volume_balance_scorer import VolumeBalanceScorer, VolumeLevel, BodyShape, VolumeBalanceResult

# Additional styling rules
from .skin_contrast_scorer import SkinContrastScorer, ContrastType, SkinContrastResult, score_skin_contrast
from .sandwich_rule_scorer import SandwichRuleScorer, SandwichResult, check_sandwich_rule, get_sandwich_score
from .three_color_scorer import ThreeColorScorer, ThreeColorResult, StyleAesthetic, check_three_color_rule, count_outfit_colors, get_three_color_score
from .occasion_scorer import OccasionScorer, OccasionResult, check_occasion_fit, get_occasion_score, get_formality_gap

# Pattern mixing rules
from .pattern_mixing_scorer import (
    PatternMixingScorer,
    PatternMixingResult,
    PatternAnalysis,
    PatternFamily,
    PatternScale,
    PatternDensity,
    TextureType,
    get_pattern_mixing_score,
    check_pattern_compatibility,
    count_patterns
)

# Design Principles (Fashion School fundamentals)
from .design_principles_scorer import (
    DesignPrinciplesScorer,
    DesignPrinciplesResult,
    ProportionAnalysis,
    BalanceAnalysis,
    RhythmAnalysis,
    EmphasisAnalysis,
    ContrastAnalysis,
    HarmonyAnalysis,
    LineAnalysis,
    BalanceType,
    LineDirection,
    DesignStyle,
    EmphasisType,
    get_design_principles_score,
    get_design_grade,
    check_golden_ratio,
    get_focal_point
)

# Total Style Scorecard
from .total_style_scorer import (
    TotalStyleScorer,
    StyleReport,
    StyleScoreBreakdown,
    get_total_style_score,
    get_style_grade
)

# Creativity Scorer (Rule-Breaking Intelligence)
from .creativity_scorer import (
    CreativityScorer,
    CreativityResult,
    CreativityLevel,
    RuleBreak,
    RuleBreakType,
)

# Outfit Builder & Scorecard (High-level outfit construction)
from .outfit_scorecard import OutfitScorecard
from .outfit_builder import OutfitBuilder, OutfitCandidate

# Scoring Configuration Service
from .scoring_config_service import (
    ScoringConfigService,
    ScoringProfile,
    CriterionConfig,
    get_scoring_config_service,
    reload_scoring_config
)

__all__ = [
    # Core scorers
    "CompatibilityScorer",
    "ColorHarmonyAnalyzer",
    "SilhouetteAnalyzer",
    "FormalityMatcher",
    "StyleIntelligenceModel",
    
    # 7-Point Rule
    "SevenPointRuleScorer",
    "SevenPointResult",
    "calculate_seven_point_score",
    
    # Season Color Harmony
    "SeasonColorHarmonyScorer",
    "ColorSeason",
    "SkinUndertone",
    "SeasonColorResult",
    
    # Proportions (Rule of Thirds)
    "ProportionScorer",
    "ProportionRatio",
    "ProportionResult",
    
    # Volume Balance
    "VolumeBalanceScorer",
    "VolumeLevel",
    "BodyShape",
    "VolumeBalanceResult",
    
    # Skin Contrast Rule
    "SkinContrastScorer",
    "ContrastType",
    "SkinContrastResult",
    "score_skin_contrast",
    
    # Sandwich Rule (Color)
    "SandwichRuleScorer",
    "SandwichResult",
    "check_sandwich_rule",
    "get_sandwich_score",
    
    # Three Color Rule
    "ThreeColorScorer",
    "ThreeColorResult",
    "StyleAesthetic",
    "check_three_color_rule",
    "count_outfit_colors",
    "get_three_color_score",
    
    # Occasion Compatibility
    "OccasionScorer",
    "OccasionResult",
    "check_occasion_fit",
    "get_occasion_score",
    "get_formality_gap",
    
    # Pattern Mixing Rules
    "PatternMixingScorer",
    "PatternMixingResult",
    "PatternAnalysis",
    "PatternFamily",
    "PatternScale",
    "PatternDensity",
    "TextureType",
    "get_pattern_mixing_score",
    "check_pattern_compatibility",
    "count_patterns",
    
    # Design Principles (Fashion Fundamentals)
    "DesignPrinciplesScorer",
    "DesignPrinciplesResult",
    "ProportionAnalysis",
    "BalanceAnalysis",
    "RhythmAnalysis",
    "EmphasisAnalysis",
    "ContrastAnalysis",
    "HarmonyAnalysis",
    "LineAnalysis",
    "BalanceType",
    "LineDirection",
    "DesignStyle",
    "EmphasisType",
    "get_design_principles_score",
    "get_design_grade",
    "check_golden_ratio",
    "get_focal_point",
    
    # Total Style Scorecard
    "TotalStyleScorer",
    "StyleReport",
    "StyleScoreBreakdown",
    "get_total_style_score",
    "get_style_grade",
    
    # Creativity Scorer (Rule-Breaking Intelligence)
    "CreativityScorer",
    "CreativityResult",
    "CreativityLevel",
    "RuleBreak",
    "RuleBreakType",
    
    # Outfit Builder & Scorecard
    "OutfitScorecard",
    "OutfitBuilder",
    "OutfitCandidate",
    
    # Scoring Configuration
    "ScoringConfigService",
    "ScoringProfile",
    "CriterionConfig",
    "get_scoring_config_service",
    "reload_scoring_config",
]
