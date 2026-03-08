"""
Total Style Scorecard
Combines all styling rules into a comprehensive style report.

| Rule                | Weight | Source Attributes                    |
|---------------------|--------|--------------------------------------|
| 7-Point Rule        | 30%    | Pattern, Texture, Color Brightness   |
| Color Harmony       | 30%    | Primary Color + User Skin Tone       |
| Proportions         | 25%    | Length, Waist Rise, Silhouette       |
| Volume Balance      | 15%    | Fit (Oversized vs. Slim)             |
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from src.core.models import Garment, Outfit, OutfitItem
from src.core import get_logger

from .seven_point_rule import SevenPointRuleScorer, SevenPointResult
from .season_color_harmony import SeasonColorHarmonyScorer, ColorSeason, SeasonColorResult
from .proportion_scorer import ProportionScorer, ProportionResult
from .volume_balance_scorer import VolumeBalanceScorer, BodyShape, VolumeBalanceResult

logger = get_logger(__name__)


@dataclass
class StyleScoreBreakdown:
    """Breakdown of individual style scores."""
    seven_point_score: float
    seven_point_total: int
    seven_point_harmonious: bool
    
    color_harmony_score: float
    color_matches: List[str]
    color_clashes: List[str]
    
    proportion_score: float
    proportion_ratio: str
    
    volume_balance_score: float
    volume_balance_type: str


@dataclass
class StyleReport:
    """Complete style analysis report."""
    overall_score: float
    grade: str  # A, B, C, D, F
    breakdown: StyleScoreBreakdown
    
    # Individual recommendations
    seven_point_recommendation: str
    color_recommendation: str
    proportion_recommendation: str
    volume_recommendation: str
    
    # Summary
    strengths: List[str]
    improvements: List[str]
    
    # Metadata
    analyzed_items: int
    analysis_timestamp: str


class TotalStyleScorer:
    """
    Comprehensive style scorer combining all rules.
    
    Weights:
    - 7-Point Rule: 30%
    - Color Harmony: 30%
    - Proportions: 25%
    - Volume Balance: 15%
    """
    
    def __init__(self):
        self.seven_point_scorer = SevenPointRuleScorer()
        self.color_scorer = SeasonColorHarmonyScorer()
        self.proportion_scorer = ProportionScorer()
        self.volume_scorer = VolumeBalanceScorer()
        
        # Configurable weights
        self.weights = {
            "seven_point": 0.30,
            "color_harmony": 0.30,
            "proportions": 0.25,
            "volume_balance": 0.15
        }
        
        # Grade thresholds
        self.grade_thresholds = {
            "A": 0.85,
            "B": 0.70,
            "C": 0.55,
            "D": 0.40,
            "F": 0.0
        }
    
    def analyze_outfit(
        self,
        items: List[Garment],
        user_season: Optional[ColorSeason] = None,
        body_shape: Optional[BodyShape] = None
    ) -> StyleReport:
        """
        Perform complete style analysis.
        
        Args:
            items: Outfit garments
            user_season: Optional user color season for personalized color scoring
            body_shape: Optional body shape for personalized volume advice
            
        Returns:
            Complete StyleReport
        """
        if not items:
            return self._empty_report()
        
        # Run all scorers
        seven_point_result = self.seven_point_scorer.analyze_outfit(items)
        
        # Color harmony - use season if provided, otherwise basic harmony
        if user_season:
            color_result = self.color_scorer.analyze_outfit(items, user_season)
        else:
            # Use a default neutral analysis
            color_result = SeasonColorResult(
                score=0.7,  # Neutral score without season info
                points=0,
                matches=[],
                clashes=[],
                recommendation="Provide your color season for personalized color analysis"
            )
        
        proportion_result = self.proportion_scorer.analyze_outfit(items)
        volume_result = self.volume_scorer.analyze_outfit(items, body_shape)
        
        # Calculate overall score
        overall_score = (
            self.weights["seven_point"] * seven_point_result.score +
            self.weights["color_harmony"] * color_result.score +
            self.weights["proportions"] * proportion_result.score +
            self.weights["volume_balance"] * volume_result.score
        )
        
        # Determine grade
        grade = self._calculate_grade(overall_score)
        
        # Build breakdown
        breakdown = StyleScoreBreakdown(
            seven_point_score=round(seven_point_result.score, 3),
            seven_point_total=seven_point_result.total_points,
            seven_point_harmonious=seven_point_result.is_harmonious,
            
            color_harmony_score=round(color_result.score, 3),
            color_matches=color_result.matches,
            color_clashes=color_result.clashes,
            
            proportion_score=round(proportion_result.score, 3),
            proportion_ratio=proportion_result.ratio.value if hasattr(proportion_result.ratio, 'value') else str(proportion_result.ratio),
            
            volume_balance_score=round(volume_result.score, 3),
            volume_balance_type=volume_result.balance_type
        )
        
        # Identify strengths and improvements
        strengths, improvements = self._identify_strengths_and_improvements(
            seven_point_result, color_result, proportion_result, volume_result
        )
        
        return StyleReport(
            overall_score=round(overall_score, 3),
            grade=grade,
            breakdown=breakdown,
            
            seven_point_recommendation=seven_point_result.recommendation,
            color_recommendation=color_result.recommendation,
            proportion_recommendation=proportion_result.recommendation,
            volume_recommendation=volume_result.recommendation,
            
            strengths=strengths,
            improvements=improvements,
            
            analyzed_items=len(items),
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def analyze_and_create_outfit(
        self,
        items: List[Garment],
        user_season: Optional[ColorSeason] = None,
        body_shape: Optional[BodyShape] = None
    ) -> Outfit:
        """
        Analyze items and create a complete Outfit object with all scores.
        
        Args:
            items: Outfit garments
            user_season: Optional user color season
            body_shape: Optional body shape
            
        Returns:
            Outfit with all scores populated
        """
        report = self.analyze_outfit(items, user_season, body_shape)
        
        # Create outfit items
        outfit_items = []
        for i, garment in enumerate(items):
            role = self._determine_role(garment, i)
            outfit_items.append(OutfitItem(garment=garment, role=role))
        
        # Create outfit with scores
        outfit = Outfit(
            id=self._generate_id(),
            items=outfit_items,
            compatibility_score=report.breakdown.seven_point_score,
            style_coherence_score=report.breakdown.proportion_score,
            occasion_match_score=report.breakdown.color_harmony_score,
            overall_score=report.overall_score,
            seven_point_total=report.breakdown.seven_point_total,
            seven_point_harmonious=report.breakdown.seven_point_harmonious,
            explanation=self._generate_explanation(report)
        )
        
        return outfit
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        for grade, threshold in self.grade_thresholds.items():
            if score >= threshold:
                return grade
        return "F"
    
    def _identify_strengths_and_improvements(
        self,
        seven_point: SevenPointResult,
        color: SeasonColorResult,
        proportion: ProportionResult,
        volume: VolumeBalanceResult
    ) -> tuple:
        """Identify outfit strengths and areas for improvement."""
        strengths = []
        improvements = []
        
        # Seven-point analysis
        if seven_point.is_harmonious:
            strengths.append(f"Perfect balance of basic and statement pieces ({seven_point.total_points} points)")
        elif seven_point.total_points < 7:
            improvements.append("Add a statement piece with texture or print")
        else:
            improvements.append("Reduce statement pieces - outfit may look too busy")
        
        # Color analysis
        if color.score >= 0.8:
            strengths.append("Colors complement each other beautifully")
        elif color.clashes:
            improvements.append(f"Consider swapping {', '.join(color.clashes[:2])} for more flattering colors")
        
        # Proportion analysis
        if proportion.score >= 0.7:
            strengths.append(f"Good proportions with {proportion.ratio.value if hasattr(proportion.ratio, 'value') else proportion.ratio} balance")
        else:
            improvements.append("Adjust proportions - try high-waisted bottoms or cropped tops")
        
        # Volume analysis
        if volume.score >= 0.7:
            strengths.append(f"Balanced volumes ({volume.balance_type.replace('_', ' ')})")
        else:
            improvements.append("Balance volume - pair fitted with relaxed pieces")
        
        return strengths, improvements
    
    def _determine_role(self, garment: Garment, index: int) -> str:
        """Determine the role of a garment in the outfit."""
        category = garment.attributes.category.value
        
        role_map = {
            "top": "main_top",
            "bottom": "main_bottom",
            "dress": "main_dress",
            "outerwear": "layering",
            "shoes": "footwear",
            "accessory": "accessory",
            "bag": "accessory"
        }
        
        return role_map.get(category, f"item_{index}")
    
    def _generate_explanation(self, report: StyleReport) -> str:
        """Generate natural language explanation of the outfit."""
        grade_descriptions = {
            "A": "This is an excellent, well-balanced outfit!",
            "B": "This is a good outfit with strong styling.",
            "C": "This outfit works but has room for improvement.",
            "D": "This outfit needs some adjustments.",
            "F": "This outfit has styling issues to address."
        }
        
        intro = grade_descriptions.get(report.grade, "")
        
        # Add top strength
        strength_text = ""
        if report.strengths:
            strength_text = f" Strengths: {report.strengths[0]}."
        
        # Add main improvement
        improvement_text = ""
        if report.improvements:
            improvement_text = f" Suggestion: {report.improvements[0]}."
        
        return f"{intro}{strength_text}{improvement_text}"
    
    def _generate_id(self) -> str:
        """Generate unique outfit ID."""
        import uuid
        return f"outfit_{uuid.uuid4().hex[:12]}"
    
    def _empty_report(self) -> StyleReport:
        """Return empty report for no items."""
        return StyleReport(
            overall_score=0.0,
            grade="F",
            breakdown=StyleScoreBreakdown(
                seven_point_score=0,
                seven_point_total=0,
                seven_point_harmonious=False,
                color_harmony_score=0,
                color_matches=[],
                color_clashes=[],
                proportion_score=0,
                proportion_ratio="undefined",
                volume_balance_score=0,
                volume_balance_type="unknown"
            ),
            seven_point_recommendation="Add items to analyze",
            color_recommendation="Add items to analyze",
            proportion_recommendation="Add items to analyze",
            volume_recommendation="Add items to analyze",
            strengths=[],
            improvements=["Add items to your outfit"],
            analyzed_items=0,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def to_dict(self, report: StyleReport) -> Dict[str, Any]:
        """Convert StyleReport to dictionary."""
        return {
            "overall_score": report.overall_score,
            "grade": report.grade,
            "breakdown": asdict(report.breakdown),
            "recommendations": {
                "seven_point": report.seven_point_recommendation,
                "color": report.color_recommendation,
                "proportion": report.proportion_recommendation,
                "volume": report.volume_recommendation
            },
            "strengths": report.strengths,
            "improvements": report.improvements,
            "metadata": {
                "analyzed_items": report.analyzed_items,
                "timestamp": report.analysis_timestamp
            }
        }


# Convenience functions
def get_total_style_score(
    items: List[Garment],
    user_season: Optional[ColorSeason] = None,
    body_shape: Optional[BodyShape] = None
) -> float:
    """Quick function to get overall style score."""
    scorer = TotalStyleScorer()
    report = scorer.analyze_outfit(items, user_season, body_shape)
    return report.overall_score


def get_style_grade(
    items: List[Garment],
    user_season: Optional[ColorSeason] = None,
    body_shape: Optional[BodyShape] = None
) -> str:
    """Quick function to get style grade (A-F)."""
    scorer = TotalStyleScorer()
    report = scorer.analyze_outfit(items, user_season, body_shape)
    return report.grade
