"""
Scoring Configuration Service
Manages scoring profiles and criteria configuration from JSON.

This module provides:
- ScoringConfig: Data class for scoring configuration
- ScoringConfigService: Service to load and manage scoring profiles
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from functools import lru_cache

from src.core import get_logger

logger = get_logger(__name__)

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "scoring_config.json"


@dataclass
class CriterionConfig:
    """Configuration for a single scoring criterion."""
    name: str
    enabled: bool = True
    weight: float = 0.1
    display_name: str = ""
    description: str = ""
    show_in_summary: bool = True
    show_details: bool = True
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()


@dataclass
class ScoringProfile:
    """A complete scoring profile with all criteria configurations."""
    name: str
    description: str = ""
    criteria: Dict[str, CriterionConfig] = field(default_factory=dict)
    
    def get_enabled_criteria(self) -> Dict[str, CriterionConfig]:
        """Get only enabled criteria."""
        return {k: v for k, v in self.criteria.items() if v.enabled}
    
    def get_weights(self) -> Dict[str, float]:
        """Get weights for enabled criteria only."""
        return {k: v.weight for k, v in self.criteria.items() if v.enabled}
    
    def get_summary_criteria(self) -> List[str]:
        """Get criteria to show in summary."""
        return [k for k, v in self.criteria.items() if v.enabled and v.show_in_summary]
    
    def get_detail_criteria(self) -> List[str]:
        """Get criteria to show details for."""
        return [k for k, v in self.criteria.items() if v.enabled and v.show_details]
    
    def normalize_weights(self) -> Dict[str, float]:
        """Get normalized weights that sum to 1.0."""
        weights = self.get_weights()
        total = sum(weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}


class ScoringConfigService:
    """
    Service for managing scoring configuration.
    
    Loads scoring profiles from JSON and provides access to criteria settings.
    
    Usage:
        config_service = ScoringConfigService()
        profile = config_service.get_profile("default")
        weights = profile.get_weights()
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the scoring configuration service.
        
        Args:
            config_path: Path to the JSON configuration file.
                        Defaults to config/scoring_config.json
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: Dict[str, Any] = {}
        self._profiles: Dict[str, ScoringProfile] = {}
        self._default_profile: str = "default"
        self._loaded = False
        
    def load(self) -> "ScoringConfigService":
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self._load_defaults()
            return self
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            
            self._parse_profiles()
            self._default_profile = self._config.get("default_profile", "default")
            self._loaded = True
            
            logger.info(f"Loaded {len(self._profiles)} scoring profiles from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load scoring config: {e}")
            self._load_defaults()
            
        return self
    
    def _load_defaults(self) -> None:
        """Load default configuration."""
        self._profiles = {
            "default": ScoringProfile(
                name="Balanced Scoring",
                description="Standard balanced scoring",
                criteria={
                    "seven_point": CriterionConfig("seven_point", True, 0.15, "7-Point Rule"),
                    "color_harmony": CriterionConfig("color_harmony", True, 0.15, "Color Harmony"),
                    "three_color": CriterionConfig("three_color", True, 0.10, "3-Color Rule"),
                    "proportion": CriterionConfig("proportion", True, 0.15, "Proportions"),
                    "volume_balance": CriterionConfig("volume_balance", True, 0.10, "Volume Balance"),
                    "pattern_mixing": CriterionConfig("pattern_mixing", True, 0.10, "Pattern Mixing"),
                    "design_principles": CriterionConfig("design_principles", True, 0.15, "Design Principles"),
                    "creativity": CriterionConfig("creativity", True, 0.10, "Creativity"),
                }
            )
        }
        self._default_profile = "default"
        self._loaded = True
    
    def _parse_profiles(self) -> None:
        """Parse profiles from loaded config."""
        profiles_data = self._config.get("scoring_profiles", {})
        
        for profile_key, profile_data in profiles_data.items():
            criteria = {}
            for crit_key, crit_data in profile_data.get("criteria", {}).items():
                criteria[crit_key] = CriterionConfig(
                    name=crit_key,
                    enabled=crit_data.get("enabled", True),
                    weight=crit_data.get("weight", 0.1),
                    display_name=crit_data.get("display_name", crit_key.replace("_", " ").title()),
                    description=crit_data.get("description", ""),
                    show_in_summary=crit_data.get("show_in_summary", True),
                    show_details=crit_data.get("show_details", True),
                )
            
            self._profiles[profile_key] = ScoringProfile(
                name=profile_data.get("name", profile_key),
                description=profile_data.get("description", ""),
                criteria=criteria
            )
    
    def get_profile(self, profile_name: Optional[str] = None) -> ScoringProfile:
        """
        Get a scoring profile by name.
        
        Args:
            profile_name: Name of the profile. Uses default if not specified.
            
        Returns:
            ScoringProfile object
        """
        if not self._loaded:
            self.load()
        
        name = profile_name or self._default_profile
        
        if name not in self._profiles:
            logger.warning(f"Profile '{name}' not found, using default")
            name = self._default_profile
        
        return self._profiles.get(name, self._profiles.get("default"))
    
    def list_profiles(self) -> List[Dict[str, str]]:
        """List all available profiles."""
        if not self._loaded:
            self.load()
            
        return [
            {
                "key": key,
                "name": profile.name,
                "description": profile.description
            }
            for key, profile in self._profiles.items()
        ]
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output settings from config."""
        if not self._loaded:
            self.load()
        return self._config.get("output_settings", {})
    
    def get_grade_thresholds(self) -> Dict[str, float]:
        """Get grade thresholds."""
        settings = self.get_output_settings()
        return settings.get("grade_thresholds", {
            "A+": 0.90, "A": 0.85, "A-": 0.80,
            "B+": 0.75, "B": 0.70, "B-": 0.65,
            "C+": 0.60, "C": 0.55, "C-": 0.50,
            "D": 0.40, "F": 0.0
        })


# Global cached instance
@lru_cache(maxsize=1)
def get_scoring_config_service() -> ScoringConfigService:
    """Get the global scoring configuration service (cached)."""
    return ScoringConfigService().load()


def reload_scoring_config() -> ScoringConfigService:
    """Force reload of scoring configuration."""
    get_scoring_config_service.cache_clear()
    return get_scoring_config_service()
