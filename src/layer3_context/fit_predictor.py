"""
Fit Predictor
=============

Predicts how well garments will fit based on user body measurements
and estimated sizes.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from config import get_config
from src.core.models import Garment

logger = logging.getLogger(__name__)


@dataclass
class FitPrediction:
    """Result of fit prediction for a garment."""
    score: float  # 0.0 to 1.0
    fit_type: str  # "perfect", "good", "acceptable", "poor"
    size_difference: int  # 0 = exact, positive = too big, negative = too small
    notes: List[str] = field(default_factory=list)
    adjustments_needed: List[str] = field(default_factory=list)


@dataclass
class OutfitFitPrediction:
    """Aggregated fit prediction for entire outfit."""
    overall_score: float
    garment_scores: Dict[str, FitPrediction]
    average_fit_type: str
    fit_notes: List[str] = field(default_factory=list)


class FitPredictor:
    """
    Predicts garment fit based on user measurements and garment sizes.
    
    Uses body metrics from StyleProfile to compare against garment
    size labels and fit types.
    """
    
    # Size order for calculating distance
    SIZE_ORDER = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "3XL", "4XL"]
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize fit predictor.
        
        Args:
            config_path: Optional path to body profile config
        """
        self.config = get_config()
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[Path] = None):
        """Load fit prediction configuration."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                full_config = json.load(f)
        else:
            full_config = self.config.get_data("body_profile_config", default={})
        
        fit_config = full_config.get("fit_predictor", {})
        
        self.size_match_scores = fit_config.get("size_match_scores", {
            "exact": 1.0,
            "one_off": 0.7,
            "two_off": 0.4,
            "three_plus_off": 0.1
        })
        
        self.size_order = fit_config.get("size_order", self.SIZE_ORDER)
        
        self.fit_type_adjustments = fit_config.get("fit_type_adjustments", {
            "normal": {"oversized": 0.0, "regular": 0.0, "slim": 0.0}
        })
        
        logger.info("FitPredictor configuration loaded")
    
    def predict_fit(
        self,
        garment: Garment,
        user_top_size: Optional[str] = None,
        user_bottom_size: Optional[str] = None,
        user_bmi_category: Optional[str] = None,
    ) -> FitPrediction:
        """
        Predict how well a garment will fit the user.
        
        Args:
            garment: Garment to evaluate
            user_top_size: User's estimated top size (XS, S, M, etc.)
            user_bottom_size: User's estimated bottom size
            user_bmi_category: User's BMI category for fit type adjustments
            
        Returns:
            FitPrediction with score and details
        """
        # Get garment size
        garment_size = self._extract_garment_size(garment)
        
        if not garment_size:
            # No size info - return neutral score
            return FitPrediction(
                score=0.7,
                fit_type="unknown",
                size_difference=0,
                notes=["Garment size information not available"]
            )
        
        # Determine which user size to compare
        user_size = self._get_relevant_user_size(
            garment, user_top_size, user_bottom_size
        )
        
        if not user_size:
            return FitPrediction(
                score=0.7,
                fit_type="unknown",
                size_difference=0,
                notes=["User size information not available"]
            )
        
        # Calculate size match
        size_diff = self._calculate_size_difference(garment_size, user_size)
        base_score = self._get_size_match_score(abs(size_diff))
        
        # Apply fit type adjustments
        garment_fit_type = self._get_garment_fit_type(garment)
        fit_adjustment = self._get_fit_type_adjustment(
            user_bmi_category, garment_fit_type
        )
        
        final_score = max(0.0, min(1.0, base_score + fit_adjustment))
        
        # Determine fit type label
        fit_type = self._classify_fit(final_score)
        
        # Generate notes
        notes = self._generate_fit_notes(
            size_diff, garment_fit_type, user_bmi_category
        )
        
        return FitPrediction(
            score=final_score,
            fit_type=fit_type,
            size_difference=size_diff,
            notes=notes
        )
    
    def predict_outfit_fit(
        self,
        garments: List[Garment],
        user_top_size: Optional[str] = None,
        user_bottom_size: Optional[str] = None,
        user_bmi_category: Optional[str] = None,
    ) -> OutfitFitPrediction:
        """
        Predict fit for entire outfit.
        
        Args:
            garments: List of garments in outfit
            user_top_size: User's estimated top size
            user_bottom_size: User's estimated bottom size
            user_bmi_category: User's BMI category
            
        Returns:
            OutfitFitPrediction with overall and per-garment scores
        """
        garment_scores = {}
        scores = []
        
        for garment in garments:
            prediction = self.predict_fit(
                garment, user_top_size, user_bottom_size, user_bmi_category
            )
            garment_id = getattr(garment, 'id', str(id(garment)))
            garment_scores[garment_id] = prediction
            scores.append(prediction.score)
        
        # Calculate overall score (average with penalty for poor fits)
        if scores:
            overall_score = sum(scores) / len(scores)
            # Apply penalty if any garment has poor fit
            min_score = min(scores)
            if min_score < 0.5:
                overall_score *= 0.9  # 10% penalty for having poor-fitting items
        else:
            overall_score = 0.7
        
        # Determine average fit type
        avg_fit_type = self._classify_fit(overall_score)
        
        # Collect notes
        fit_notes = []
        for gid, pred in garment_scores.items():
            if pred.notes:
                fit_notes.extend(pred.notes)
        
        return OutfitFitPrediction(
            overall_score=overall_score,
            garment_scores=garment_scores,
            average_fit_type=avg_fit_type,
            fit_notes=list(set(fit_notes))  # Deduplicate
        )
    
    def _extract_garment_size(self, garment: Garment) -> Optional[str]:
        """Extract size from garment attributes."""
        # Check attributes for size
        if hasattr(garment, 'attributes') and garment.attributes:
            attrs = garment.attributes
            if isinstance(attrs, dict):
                size = attrs.get('size') or attrs.get('Size')
                if size:
                    return self._normalize_size(size)
        
        # Check other common fields
        for field in ['size', 'garment_size', 'clothing_size']:
            if hasattr(garment, field):
                val = getattr(garment, field)
                if val:
                    return self._normalize_size(val)
        
        return None
    
    def _normalize_size(self, size: str) -> str:
        """Normalize size string to standard format."""
        size = str(size).strip().upper()
        
        # Map common variations
        size_map = {
            "EXTRA SMALL": "XS",
            "SMALL": "S",
            "MEDIUM": "M",
            "LARGE": "L",
            "EXTRA LARGE": "XL",
            "2XL": "XXL",
            "XXXL": "3XL",
        }
        
        return size_map.get(size, size)
    
    def _get_relevant_user_size(
        self,
        garment: Garment,
        user_top_size: Optional[str],
        user_bottom_size: Optional[str]
    ) -> Optional[str]:
        """Determine which user size to compare based on garment category."""
        category = getattr(garment, 'category', None)
        
        if category:
            category_lower = str(category).lower()
            bottom_categories = ['pants', 'jeans', 'shorts', 'skirt', 'trouser', 'bottom']
            
            if any(cat in category_lower for cat in bottom_categories):
                return user_bottom_size
        
        # Default to top size
        return user_top_size
    
    def _calculate_size_difference(self, garment_size: str, user_size: str) -> int:
        """Calculate difference between sizes (positive = garment larger)."""
        try:
            garment_idx = self.size_order.index(garment_size)
            user_idx = self.size_order.index(user_size)
            return garment_idx - user_idx
        except ValueError:
            # Size not in standard list
            return 0
    
    def _get_size_match_score(self, size_diff: int) -> float:
        """Get score based on size difference."""
        if size_diff == 0:
            return self.size_match_scores.get("exact", 1.0)
        elif size_diff == 1:
            return self.size_match_scores.get("one_off", 0.7)
        elif size_diff == 2:
            return self.size_match_scores.get("two_off", 0.4)
        else:
            return self.size_match_scores.get("three_plus_off", 0.1)
    
    def _get_garment_fit_type(self, garment: Garment) -> str:
        """Determine garment fit type (slim, regular, oversized)."""
        if hasattr(garment, 'attributes') and garment.attributes:
            attrs = garment.attributes
            if isinstance(attrs, dict):
                fit = attrs.get('fit') or attrs.get('Fit') or attrs.get('fit_type')
                if fit:
                    fit_lower = str(fit).lower()
                    if 'slim' in fit_lower or 'tight' in fit_lower or 'skinny' in fit_lower:
                        return 'slim'
                    elif 'over' in fit_lower or 'loose' in fit_lower or 'relaxed' in fit_lower:
                        return 'oversized'
        
        return 'regular'
    
    def _get_fit_type_adjustment(
        self,
        bmi_category: Optional[str],
        fit_type: str
    ) -> float:
        """Get score adjustment based on BMI category and fit type."""
        if not bmi_category:
            return 0.0
        
        bmi_adjustments = self.fit_type_adjustments.get(bmi_category, {})
        return bmi_adjustments.get(fit_type, 0.0)
    
    def _classify_fit(self, score: float) -> str:
        """Classify fit based on score."""
        if score >= 0.9:
            return "perfect"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "acceptable"
        else:
            return "poor"
    
    def _generate_fit_notes(
        self,
        size_diff: int,
        fit_type: str,
        bmi_category: Optional[str]
    ) -> List[str]:
        """Generate fit notes based on analysis."""
        notes = []
        
        if size_diff > 0:
            notes.append(f"Garment runs {abs(size_diff)} size(s) larger")
        elif size_diff < 0:
            notes.append(f"Garment runs {abs(size_diff)} size(s) smaller")
        else:
            notes.append("Size matches well")
        
        if fit_type == 'slim' and bmi_category in ['overweight', 'obese']:
            notes.append("Slim fit may be uncomfortable; consider regular fit")
        elif fit_type == 'oversized' and bmi_category == 'underweight':
            notes.append("Oversized fit may look too loose; consider regular fit")
        
        return notes
