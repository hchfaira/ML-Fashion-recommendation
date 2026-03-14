"""
Neo4j Configuration Loader

Loads and manages Neo4j connection configuration from neo4j_config.json.
Extends the main ConfigLoader system to provide Neo4j-specific configuration.

Usage:
    from src.core.neo4j_config import get_neo4j_config
    
    config = get_neo4j_config()
    print(config.get_uri())
    print(config.get_max_pool_size())
"""

from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache
import json
import logging
import os

logger = logging.getLogger(__name__)


class Neo4jConfigError(Exception):
    """Raised when Neo4j configuration is invalid."""
    pass


class Neo4jConfig:
    """
    Manages Neo4j configuration.
    
    Loads from config/data/neo4j_config.json and provides typed accessors
    for all configuration values with validation.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Neo4j configuration.
        
        Args:
            config_path: Path to neo4j_config.json. Defaults to config/data/neo4j_config.json
        """
        if config_path is None:
            # Default to config/data/neo4j_config.json
            config_path = Path(__file__).parent.parent.parent / "config" / "data" / "neo4j_config.json"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        
        self._load_config()
        self._apply_env_overrides()
        self._validate_config()
        
        logger.info(f"Neo4j configuration loaded from {self.config_path}")
    
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise Neo4jConfigError(
                f"Neo4j config file not found: {self.config_path}"
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        except json.JSONDecodeError as e:
            raise Neo4jConfigError(
                f"Invalid JSON in neo4j_config.json: {e}"
            )
    
    def _apply_env_overrides(self) -> None:
        """Override connection settings with environment variables if present.

        Supported variables (all optional — fall back to neo4j_config.json):
            NEO4J_URI       bolt://host:port
            NEO4J_USER      username
            NEO4J_PASSWORD  password
            NEO4J_DATABASE  database name
        """
        conn = self._config.setdefault("connection", {})
        env_map = {
            "NEO4J_URI":      "uri",
            "NEO4J_USER":     "username",
            "NEO4J_PASSWORD": "password",
            "NEO4J_DATABASE": "database",
        }
        for env_key, cfg_key in env_map.items():
            value = os.environ.get(env_key)
            if value:
                conn[cfg_key] = value
                logger.debug("Neo4j config: %s overridden from env", cfg_key)

        if os.environ.get("NEO4J_PASSWORD"):
            logger.debug("Neo4j password loaded from environment variable.")

    def _validate_config(self) -> None:
        """Validate configuration has required keys."""
        required_sections = ["connection", "query_cache", "retry_policy"]
        
        for section in required_sections:
            if section not in self._config:
                logger.warning(
                    f"Missing required config section: {section}. "
                    f"Using defaults."
                )
    
    # ==================== Connection Config ====================
    
    def get_uri(self) -> str:
        """Get Neo4j connection URI."""
        return self._config.get("connection", {}).get(
            "uri", "bolt://localhost:7687"
        )
    
    def get_username(self) -> str:
        """Get Neo4j username."""
        return self._config.get("connection", {}).get("username", "neo4j")
    
    def get_password(self) -> str:
        """Get Neo4j password."""
        password = self._config.get("connection", {}).get("password", "password")
        if password == "neo4j_password_123":
            logger.warning(
                "Using default Neo4j password. Please change this in production!"
            )
        return password
    
    def get_database(self) -> str:
        """Get Neo4j database name."""
        return self._config.get("connection", {}).get("database", "neo4j")
    
    def get_timeout(self) -> int:
        """Get connection timeout in seconds."""
        return self._config.get("connection", {}).get("timeout", 30)
    
    def get_max_pool_size(self) -> int:
        """Get maximum connection pool size."""
        return self._config.get("connection", {}).get("max_pool_size", 50)
    
    def get_min_pool_size(self) -> int:
        """Get minimum connection pool size."""
        return self._config.get("connection", {}).get("min_pool_size", 10)
    
    # ==================== Query Cache Config ====================
    
    def is_cache_enabled(self) -> bool:
        """Check if query cache is enabled."""
        return self._config.get("query_cache", {}).get("enabled", True)
    
    def get_cache_max_size(self) -> int:
        """Get maximum number of cached queries."""
        return self._config.get("query_cache", {}).get("max_size", 1000)
    
    def get_cache_default_ttl(self) -> int:
        """Get default cache TTL in seconds."""
        return self._config.get("query_cache", {}).get(
            "default_ttl_seconds", 3600
        )
    
    def get_cache_ttl_for_type(self, query_type: str) -> int:
        """
        Get cache TTL for a specific query type.
        
        Args:
            query_type: Type of query (compatible_items, centrality_scores, etc.)
            
        Returns:
            TTL in seconds, or default if not specified
        """
        ttl_map = self._config.get("query_cache", {}).get(
            "ttl_by_query_type", {}
        )
        return ttl_map.get(query_type, self.get_cache_default_ttl())
    
    # ==================== Retry Policy Config ====================
    
    def is_retry_enabled(self) -> bool:
        """Check if retry policy is enabled."""
        return self._config.get("retry_policy", {}).get("enabled", True)
    
    def get_max_retries(self) -> int:
        """Get maximum number of retries."""
        return self._config.get("retry_policy", {}).get("max_retries", 3)
    
    def get_initial_backoff_ms(self) -> int:
        """Get initial backoff in milliseconds."""
        return self._config.get("retry_policy", {}).get(
            "initial_backoff_ms", 100
        )
    
    def get_max_backoff_ms(self) -> int:
        """Get maximum backoff in milliseconds."""
        return self._config.get("retry_policy", {}).get(
            "max_backoff_ms", 5000
        )
    
    def get_backoff_multiplier(self) -> float:
        """Get exponential backoff multiplier."""
        return self._config.get("retry_policy", {}).get(
            "backoff_multiplier", 2.0
        )
    
    # ==================== Fallback Config ====================
    
    def is_fallback_enabled(self) -> bool:
        """Check if fallback to in-memory is enabled."""
        return self._config.get("fallback", {}).get("enabled", True)
    
    def use_in_memory_fallback(self) -> bool:
        """Check if should use in-memory fallback when Neo4j unavailable."""
        return self._config.get("fallback", {}).get("use_in_memory", True)
    
    def log_fallback_usage(self) -> bool:
        """Check if should log when using fallback."""
        return self._config.get("fallback", {}).get("log_fallback", True)
    
    # ==================== Graph Initialization ====================
    
    def should_clear_on_startup(self) -> bool:
        """Check if graph should be cleared on startup."""
        return self._config.get("graph_initialization", {}).get(
            "clear_on_startup", False
        )
    
    def should_auto_create_indexes(self) -> bool:
        """Check if indexes should be auto-created."""
        return self._config.get("graph_initialization", {}).get(
            "auto_create_indexes", True
        )
    
    def should_auto_create_reference_nodes(self) -> bool:
        """Check if reference nodes should be auto-created."""
        return self._config.get("graph_initialization", {}).get(
            "auto_create_reference_nodes", True
        )
    
    # ==================== Performance Config ====================
    
    def get_batch_query_size(self) -> int:
        """Get batch query size for bulk operations."""
        return self._config.get("performance", {}).get("batch_query_size", 100)
    
    def is_statistics_enabled(self) -> bool:
        """Check if statistics collection is enabled."""
        return self._config.get("performance", {}).get(
            "enable_statistics", True
        )
    
    def get_slow_query_threshold_ms(self) -> int:
        """Get threshold for logging slow queries in milliseconds."""
        return self._config.get("performance", {}).get(
            "log_slow_queries_ms", 1000
        )
    
    def is_pool_leak_detection_enabled(self) -> bool:
        """Check if connection pool leak detection is enabled."""
        return self._config.get("performance", {}).get(
            "connection_pool_leak_detection", True
        )
    
    # ==================== Centrality Scores Config ====================
    
    def get_centrality_compute_interval(self) -> int:
        """Get interval for computing centrality scores in hours."""
        return self._config.get("centrality_scores", {}).get(
            "compute_interval_hours", 24
        )
    
    def get_centrality_algorithms(self) -> list:
        """Get list of centrality algorithms to compute."""
        return self._config.get("centrality_scores", {}).get(
            "algorithms",
            ["pagerank", "betweenness_centrality", "eigenvector_centrality"]
        )
    
    def get_pagerank_iterations(self) -> int:
        """Get number of iterations for PageRank algorithm."""
        return self._config.get("centrality_scores", {}).get(
            "personalization_iterations", 40
        )

    def get_pagerank_damping(self) -> float:
        """Get PageRank damping factor (standard default: 0.85)."""
        return self._config.get("centrality_scores", {}).get(
            "pagerank_damping", 0.85
        )

    def get_pagerank_power_iterations(self) -> int:
        """Get number of power iterations for the simplified PageRank pass."""
        return self._config.get("centrality_scores", {}).get(
            "pagerank_iterations", 5
        )

    # ==================== Reference Nodes ====================
    
    def get_seasons(self) -> list:
        """Get list of seasons."""
        return self._config.get("reference_nodes", {}).get(
            "seasons", ["SPRING", "SUMMER", "FALL", "WINTER"]
        )
    
    def get_occasions(self) -> list:
        """Get list of occasions."""
        return self._config.get("reference_nodes", {}).get(
            "occasions",
            ["CASUAL", "BUSINESS", "FORMAL", "WEDDING", "ATHLETIC"]
        )
    
    def get_formality_levels(self) -> Dict[str, str]:
        """Get formality level mappings."""
        return self._config.get("reference_nodes", {}).get(
            "formality_levels",
            {
                "0": "Very Casual",
                "1": "Casual",
                "2": "Smart Casual",
                "3": "Business Casual",
                "4": "Business",
                "5": "Semi-Formal",
                "6": "Black Tie"
            }
        )
    
    def get_categories(self) -> list:
        """Get list of garment categories."""
        return self._config.get("reference_nodes", {}).get(
            "categories",
            ["TOP", "BOTTOM", "SHOES", "JACKET", "DRESS", "ACCESSORY", "OUTERWEAR"]
        )
    
    def get_color_palettes(self) -> list:
        """Get list of color palettes."""
        return self._config.get("reference_nodes", {}).get(
            "color_palettes",
            ["NEUTRAL", "EARTH_TONES", "PASTELS", "VIBRANT", "COOL_TONES", "WARM_TONES"]
        )
    
    # ==================== Constraints ====================
    
    def get_min_compatibility_weight(self) -> float:
        """Get minimum compatibility weight threshold."""
        return self._config.get("constraints", {}).get(
            "min_compatibility_weight", 0.5
        )
    
    def get_max_outfit_size(self) -> int:
        """Get maximum number of items in an outfit."""
        return self._config.get("constraints", {}).get("max_outfit_size", 4)
    
    def get_min_outfit_score(self) -> float:
        """Get minimum acceptable outfit score."""
        return self._config.get("constraints", {}).get("min_outfit_score", 0.6)
    
    def get_formality_tolerance(self) -> int:
        """Get formality level tolerance."""
        return self._config.get("constraints", {}).get("formality_tolerance", 1)
    
    def get_pattern_conflict_threshold(self) -> float:
        """Get pattern conflict threshold."""
        return self._config.get("constraints", {}).get(
            "pattern_conflict_threshold", 0.3
        )

    def get_default_reference_edge_weight(self) -> float:
        """Get default weight for reference node edges (SUITABLE_FOR, SUITABLE_FOR_OCCASION)."""
        return self._config.get("constraints", {}).get(
            "default_reference_edge_weight", 0.8
        )

    # ==================== Monitoring ====================
    
    def get_health_check_interval(self) -> int:
        """Get health check interval in seconds."""
        return self._config.get("monitoring", {}).get(
            "health_check_interval_seconds", 300
        )
    
    def track_query_performance(self) -> bool:
        """Check if query performance tracking is enabled."""
        return self._config.get("monitoring", {}).get(
            "track_query_performance", True
        )
    
    def track_cache_stats(self) -> bool:
        """Check if cache statistics tracking is enabled."""
        return self._config.get("monitoring", {}).get("track_cache_stats", True)
    
    def is_metrics_export_enabled(self) -> bool:
        """Check if metrics export is enabled."""
        return self._config.get("monitoring", {}).get(
            "export_metrics_enabled", False
        )
    
    def get_metrics_endpoint(self) -> str:
        """Get metrics export endpoint."""
        return self._config.get("monitoring", {}).get(
            "export_metrics_endpoint", "http://localhost:9090/metrics"
        )
    
    # ==================== Utility Methods ====================
    
    def get_connection_string(self) -> str:
        """Get full Neo4j connection string."""
        return f"{self.get_uri()}"
    
    def to_neo4j_driver_config(self) -> Dict[str, Any]:
        """
        Get configuration dict for neo4j.AsyncGraphDatabase.driver().

        Only keys accepted by the neo4j Python driver are included.
        Pool-size and timeout settings map to the correct driver kwarg names.

        Returns:
            Dict with driver configuration
        """
        return {
            "uri": self.get_uri(),
            "auth": (self.get_username(), self.get_password()),
            # connection_timeout is accepted by the neo4j driver directly
            "connection_timeout": self.get_timeout(),
            # max_connection_pool_size is the correct kwarg (not max_pool_size)
            "max_connection_pool_size": self.get_max_pool_size(),
        }
    
    def reload(self) -> None:
        """Reload configuration from disk."""
        self._load_config()
        self._validate_config()
        logger.info("Neo4j configuration reloaded")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config.copy()


# Singleton instance
_neo4j_config: Optional[Neo4jConfig] = None


@lru_cache()
def get_neo4j_config() -> Neo4jConfig:
    """
    Get the cached Neo4j configuration singleton.
    
    Returns:
        Neo4jConfig instance
        
    Raises:
        Neo4jConfigError: If configuration file is missing or invalid
    """
    global _neo4j_config
    if _neo4j_config is None:
        _neo4j_config = Neo4jConfig()
    return _neo4j_config


def reset_neo4j_config() -> None:
    """Reset the Neo4j configuration singleton (useful for testing)."""
    global _neo4j_config
    _neo4j_config = None
    get_neo4j_config.cache_clear()
