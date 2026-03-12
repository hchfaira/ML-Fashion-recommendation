"""
Smart Removal Analyzer Configuration Loader
=============================================

Loads and manages profiles from JSON config file.
Allows easy customization per user without code changes.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field

from src.core import get_logger

logger = get_logger(__name__)

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class RemovalAggressiveness:
    """Thresholds for verdict mapping."""
    donate_threshold: float = 0.15
    safe_to_remove_threshold: float = 0.30
    consider_threshold: float = 0.60


@dataclass
class ConfidenceStrictness:
    """How strictly to enforce confidence requirements."""
    penalty_per_data_gap: float = 0.15
    minimum_confidence_for_action: float = 0.4


@dataclass
class GoalWeights:
    """Weights for a single user goal."""
    recency: float
    frequency: float
    versatility: float
    outfit_impact: float
    replacement: float
    rating: float
    newness: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "GoalWeights":
        return cls(**d)


@dataclass
class NewnessGuard:
    """Newness protection parameters."""
    grace_period_days: int = 30
    half_life_days: float = 60.0


@dataclass
class FrequencySoftCap:
    """Frequency scoring soft cap."""
    max_recent_equivalent_wears: int = 10
    half_life_days: float = 60.0


@dataclass
class VersatilityWeights:
    """Weights for versatility subcomponents."""
    season_weight: float = 0.40
    occasion_weight: float = 0.35
    category_weight: float = 0.25


@dataclass
class ReplacementQualityWeights:
    """Weights for replacement quality evaluation."""
    season_coverage: float = 0.40
    formality_compatibility: float = 0.30
    color_role_match: float = 0.15
    occasion_coverage: float = 0.15


@dataclass
class SmartRemovalProfile:
    """Complete configuration for a smart removal profile."""
    name: str
    description: str
    recency_half_life_days: float
    formality_flexibility: int
    replacement_coverage_threshold: float
    removal_aggressiveness: RemovalAggressiveness
    confidence_strictness: ConfidenceStrictness
    goal_weights: Dict[str, GoalWeights]  # keyed by goal name (MAXIMIZE_OPTIONS, etc.)
    newness_guard: NewnessGuard
    frequency_soft_cap: FrequencySoftCap
    versatility: VersatilityWeights
    replacement_quality_weights: ReplacementQualityWeights
    neutral_colors: List[str]
    occasion_importance: Dict[str, float]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SmartRemovalProfile":
        """Convert dict (from JSON) to dataclass."""
        return cls(
            name=d["name"],
            description=d["description"],
            recency_half_life_days=d["recency_half_life_days"],
            formality_flexibility=d["formality_flexibility"],
            replacement_coverage_threshold=d["replacement_coverage_threshold"],
            removal_aggressiveness=RemovalAggressiveness(**d["removal_aggressiveness"]),
            confidence_strictness=ConfidenceStrictness(**d["confidence_strictness"]),
            goal_weights={
                goal_name: GoalWeights.from_dict(weights)
                for goal_name, weights in d["goal_weights"].items()
            },
            newness_guard=NewnessGuard(**d["newness_guard"]),
            frequency_soft_cap=FrequencySoftCap(**d["frequency_soft_cap"]),
            versatility=VersatilityWeights(**d["versatility"]),
            replacement_quality_weights=ReplacementQualityWeights(**d["replacement_quality_weights"]),
            neutral_colors=d["neutral_colors"],
            occasion_importance=d["occasion_importance"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "recency_half_life_days": self.recency_half_life_days,
            "formality_flexibility": self.formality_flexibility,
            "replacement_coverage_threshold": self.replacement_coverage_threshold,
            "removal_aggressiveness": asdict(self.removal_aggressiveness),
            "confidence_strictness": asdict(self.confidence_strictness),
            "goal_weights": {
                goal_name: weights.to_dict()
                for goal_name, weights in self.goal_weights.items()
            },
            "newness_guard": asdict(self.newness_guard),
            "frequency_soft_cap": asdict(self.frequency_soft_cap),
            "versatility": asdict(self.versatility),
            "replacement_quality_weights": asdict(self.replacement_quality_weights),
            "neutral_colors": self.neutral_colors,
            "occasion_importance": self.occasion_importance,
        }


# ============================================================================
# Configuration Loader
# ============================================================================


class SmartRemovalConfigLoader:
    """Loads and caches smart removal profiles from JSON config file."""

    _config_path = Path(__file__).parent.parent.parent / "config" / "data" / "smart_removal_config.json"
    _profiles_cache: Optional[Dict[str, SmartRemovalProfile]] = None

    @classmethod
    def load_profiles(cls, force_reload: bool = False) -> Dict[str, SmartRemovalProfile]:
        """Load all profiles from config file.

        Args:
            force_reload: Force reload from disk (ignores cache).

        Returns:
            Dict mapping profile name → SmartRemovalProfile.
        """
        if cls._profiles_cache is not None and not force_reload:
            return cls._profiles_cache

        if not cls._config_path.exists():
            logger.error(f"Config file not found: {cls._config_path}")
            raise FileNotFoundError(f"Smart removal config file not found: {cls._config_path}")

        try:
            with open(cls._config_path, "r") as f:
                data = json.load(f)

            profiles = {}
            for profile_name, profile_dict in data.get("profiles", {}).items():
                try:
                    profile = SmartRemovalProfile.from_dict(profile_dict)
                    profiles[profile_name] = profile
                    logger.debug(f"Loaded profile: {profile_name}")
                except Exception as e:
                    logger.error(f"Failed to load profile '{profile_name}': {e}")
                    raise

            cls._profiles_cache = profiles
            logger.info(f"Loaded {len(profiles)} smart removal profiles from {cls._config_path}")
            return profiles

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config JSON: {e}")
            raise

    @classmethod
    def get_profile(cls, profile_name: str) -> SmartRemovalProfile:
        """Get a specific profile by name.

        Args:
            profile_name: Name of the profile (e.g., 'default', 'minimalist_eco').

        Returns:
            SmartRemovalProfile instance.

        Raises:
            ValueError: If profile not found.
        """
        profiles = cls.load_profiles()
        if profile_name not in profiles:
            available = list(profiles.keys())
            raise ValueError(
                f"Profile '{profile_name}' not found. Available profiles: {available}"
            )
        return profiles[profile_name]

    @classmethod
    def list_profiles(cls) -> List[str]:
        """List all available profile names."""
        profiles = cls.load_profiles()
        return sorted(profiles.keys())

    @classmethod
    def get_profile_descriptions(cls) -> Dict[str, str]:
        """Get a dict of {profile_name: description} for all profiles."""
        profiles = cls.load_profiles()
        return {
            name: profile.description
            for name, profile in profiles.items()
        }

    @classmethod
    def validate_profile(cls, profile: SmartRemovalProfile) -> List[str]:
        """Validate a profile for consistency.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Check goal weights sum to ~1.0
        for goal_name, weights in profile.goal_weights.items():
            total = (
                weights.recency + weights.frequency + weights.versatility
                + weights.outfit_impact + weights.replacement + weights.rating
                + weights.newness
            )
            if not (0.95 <= total <= 1.05):
                errors.append(
                    f"Goal '{goal_name}' weights sum to {total:.2f}, expected ~1.0"
                )

        # Check verdict thresholds are in order
        if not (profile.removal_aggressiveness.donate_threshold
                < profile.removal_aggressiveness.safe_to_remove_threshold
                < profile.removal_aggressiveness.consider_threshold):
            errors.append(
                "Verdict thresholds not in ascending order: "
                f"{profile.removal_aggressiveness.donate_threshold} < "
                f"{profile.removal_aggressiveness.safe_to_remove_threshold} < "
                f"{profile.removal_aggressiveness.consider_threshold}"
            )

        # Check replacement quality weights sum to 1.0
        rq_weights = profile.replacement_quality_weights
        rq_total = (
            rq_weights.season_coverage + rq_weights.formality_compatibility
            + rq_weights.color_role_match + rq_weights.occasion_coverage
        )
        if not (0.95 <= rq_total <= 1.05):
            errors.append(f"Replacement quality weights sum to {rq_total:.2f}, expected ~1.0")

        # Check versatility weights sum to 1.0
        v_weights = profile.versatility
        v_total = v_weights.season_weight + v_weights.occasion_weight + v_weights.category_weight
        if not (0.95 <= v_total <= 1.05):
            errors.append(f"Versatility weights sum to {v_total:.2f}, expected ~1.0")

        # Check confidence strictness is in valid range
        if not (0 <= profile.confidence_strictness.penalty_per_data_gap <= 0.5):
            errors.append(
                f"penalty_per_data_gap should be 0-0.5, got {profile.confidence_strictness.penalty_per_data_gap}"
            )

        if not (0 <= profile.confidence_strictness.minimum_confidence_for_action <= 1.0):
            errors.append(
                f"minimum_confidence_for_action should be 0-1.0, got {profile.confidence_strictness.minimum_confidence_for_action}"
            )

        return errors

    @classmethod
    def validate_all_profiles(cls) -> Dict[str, List[str]]:
        """Validate all profiles. Returns dict of {profile_name: error_list}."""
        profiles = cls.load_profiles()
        results = {}
        for profile_name, profile in profiles.items():
            errors = cls.validate_profile(profile)
            if errors:
                results[profile_name] = errors
        return results

    @classmethod
    def save_profile(cls, profile_name: str, profile: SmartRemovalProfile) -> None:
        """Save a modified profile back to config file.

        Args:
            profile_name: Name to save under.
            profile: SmartRemovalProfile instance.
        """
        profiles = cls.load_profiles()
        profiles[profile_name] = profile

        with open(cls._config_path, "r") as f:
            data = json.load(f)

        data["profiles"][profile_name] = profile.to_dict()

        with open(cls._config_path, "w") as f:
            json.dump(data, f, indent=2)

        # Reset cache so next load reads the updated file
        cls._profiles_cache = None
        logger.info(f"Saved profile '{profile_name}' to {cls._config_path}")


# ============================================================================
# Convenience Functions
# ============================================================================


def load_smart_removal_profile(profile_name: str = "default") -> SmartRemovalProfile:
    """Convenience function to load a profile.

    Args:
        profile_name: Name of profile to load (default: 'default').

    Returns:
        SmartRemovalProfile instance.
    """
    return SmartRemovalConfigLoader.get_profile(profile_name)


def list_smart_removal_profiles() -> List[str]:
    """Convenience function to list available profiles."""
    return SmartRemovalConfigLoader.list_profiles()


def describe_smart_removal_profiles() -> Dict[str, str]:
    """Convenience function to get profile descriptions."""
    return SmartRemovalConfigLoader.get_profile_descriptions()
