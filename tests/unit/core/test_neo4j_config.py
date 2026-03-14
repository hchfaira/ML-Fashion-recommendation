"""
Tests for Neo4j Configuration

Tests for Neo4j configuration loading and management:
- Configuration file loading
- Configuration validation
- Accessor methods
- Default values
- Singleton pattern

Markers:
    @pytest.mark.unit: Fast unit tests
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from src.core.neo4j_config import (
    Neo4jConfig,
    Neo4jConfigError,
    get_neo4j_config,
    reset_neo4j_config,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_neo4j_config():
    """Sample Neo4j configuration."""
    return {
        "enabled": True,
        "connection": {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "test_password",
            "database": "neo4j",
            "timeout": 30,
            "max_pool_size": 50,
            "min_pool_size": 10
        },
        "query_cache": {
            "enabled": True,
            "max_size": 1000,
            "default_ttl_seconds": 3600,
            "ttl_by_query_type": {
                "compatible_items": 7200,
                "centrality_scores": 86400
            }
        },
        "retry_policy": {
            "enabled": True,
            "max_retries": 3,
            "initial_backoff_ms": 100,
            "max_backoff_ms": 5000,
            "backoff_multiplier": 2.0
        },
        "fallback": {
            "enabled": True,
            "use_in_memory": True,
            "log_fallback": True
        },
        "constraints": {
            "min_compatibility_weight": 0.5,
            "max_outfit_size": 4,
            "min_outfit_score": 0.6
        }
    }


@pytest.fixture
def temp_config_file(sample_neo4j_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.json',
        delete=False
    ) as f:
        json.dump(sample_neo4j_config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


# ============================================================================
# CONFIGURATION LOADING TESTS
# ============================================================================


class TestConfigurationLoading:
    """Tests for configuration file loading."""
    
    def test_config_loads_from_file(self, temp_config_file):
        """Test configuration loads correctly from file."""
        config = Neo4jConfig(config_path=temp_config_file)
        
        assert config.get_uri() == "bolt://localhost:7687"
        assert config.get_username() == "neo4j"
        assert config.get_password() == "test_password"
    
    def test_config_file_not_found(self):
        """Test Neo4jConfigError raised when config file not found."""
        with pytest.raises(Neo4jConfigError, match="not found"):
            Neo4jConfig(config_path="/nonexistent/path/config.json")
    
    def test_invalid_json_config(self):
        """Test Neo4jConfigError raised for invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            f.write("{ invalid json")
            temp_path = f.name
        
        try:
            with pytest.raises(Neo4jConfigError, match="Invalid JSON"):
                Neo4jConfig(config_path=temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ============================================================================
# CONNECTION CONFIG ACCESSOR TESTS
# ============================================================================


class TestConnectionConfig:
    """Tests for connection configuration accessors."""
    
    def test_get_uri(self, temp_config_file):
        """Test get_uri returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_uri() == "bolt://localhost:7687"
    
    def test_get_username(self, temp_config_file):
        """Test get_username returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_username() == "neo4j"
    
    def test_get_password(self, temp_config_file):
        """Test get_password returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_password() == "test_password"
    
    def test_get_database(self, temp_config_file):
        """Test get_database returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_database() == "neo4j"
    
    def test_get_timeout(self, temp_config_file):
        """Test get_timeout returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_timeout() == 30
    
    def test_get_max_pool_size(self, temp_config_file):
        """Test get_max_pool_size returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_max_pool_size() == 50
    
    def test_get_min_pool_size(self, temp_config_file):
        """Test get_min_pool_size returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_min_pool_size() == 10
    
    def test_get_connection_string(self, temp_config_file):
        """Test get_connection_string returns full connection string."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_connection_string() == "bolt://localhost:7687"


# ============================================================================
# QUERY CACHE CONFIG TESTS
# ============================================================================


class TestQueryCacheConfig:
    """Tests for query cache configuration accessors."""
    
    def test_is_cache_enabled(self, temp_config_file):
        """Test is_cache_enabled returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.is_cache_enabled() is True
    
    def test_get_cache_max_size(self, temp_config_file):
        """Test get_cache_max_size returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_cache_max_size() == 1000
    
    def test_get_cache_default_ttl(self, temp_config_file):
        """Test get_cache_default_ttl returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_cache_default_ttl() == 3600
    
    def test_get_cache_ttl_for_type_specific(self, temp_config_file):
        """Test get_cache_ttl_for_type returns specific TTL."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_cache_ttl_for_type("compatible_items") == 7200
    
    def test_get_cache_ttl_for_type_default(self, temp_config_file):
        """Test get_cache_ttl_for_type returns default for unknown type."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_cache_ttl_for_type("unknown_type") == 3600


# ============================================================================
# RETRY POLICY CONFIG TESTS
# ============================================================================


class TestRetryPolicyConfig:
    """Tests for retry policy configuration accessors."""
    
    def test_is_retry_enabled(self, temp_config_file):
        """Test is_retry_enabled returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.is_retry_enabled() is True
    
    def test_get_max_retries(self, temp_config_file):
        """Test get_max_retries returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_max_retries() == 3
    
    def test_get_initial_backoff_ms(self, temp_config_file):
        """Test get_initial_backoff_ms returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_initial_backoff_ms() == 100
    
    def test_get_max_backoff_ms(self, temp_config_file):
        """Test get_max_backoff_ms returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_max_backoff_ms() == 5000
    
    def test_get_backoff_multiplier(self, temp_config_file):
        """Test get_backoff_multiplier returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_backoff_multiplier() == 2.0


# ============================================================================
# FALLBACK CONFIG TESTS
# ============================================================================


class TestFallbackConfig:
    """Tests for fallback configuration accessors."""
    
    def test_is_fallback_enabled(self, temp_config_file):
        """Test is_fallback_enabled returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.is_fallback_enabled() is True
    
    def test_use_in_memory_fallback(self, temp_config_file):
        """Test use_in_memory_fallback returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.use_in_memory_fallback() is True
    
    def test_log_fallback_usage(self, temp_config_file):
        """Test log_fallback_usage returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.log_fallback_usage() is True


# ============================================================================
# CONSTRAINTS CONFIG TESTS
# ============================================================================


class TestConstraintsConfig:
    """Tests for constraint configuration accessors."""
    
    def test_get_min_compatibility_weight(self, temp_config_file):
        """Test get_min_compatibility_weight returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_min_compatibility_weight() == 0.5
    
    def test_get_max_outfit_size(self, temp_config_file):
        """Test get_max_outfit_size returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_max_outfit_size() == 4
    
    def test_get_min_outfit_score(self, temp_config_file):
        """Test get_min_outfit_score returns correct value."""
        config = Neo4jConfig(config_path=temp_config_file)
        assert config.get_min_outfit_score() == 0.6


# ============================================================================
# REFERENCE NODES CONFIG TESTS
# ============================================================================


class TestReferenceNodesConfig:
    """Tests for reference nodes configuration accessors."""
    
    def test_get_seasons_default(self, sample_neo4j_config):
        """Test get_seasons returns default when not in config."""
        config_without_seasons = sample_neo4j_config.copy()
        config_without_seasons.pop("reference_nodes", None)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_without_seasons, f)
            temp_path = f.name
        
        try:
            config = Neo4jConfig(config_path=temp_path)
            seasons = config.get_seasons()
            assert "SPRING" in seasons
            assert "SUMMER" in seasons
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_get_occasions_default(self, sample_neo4j_config):
        """Test get_occasions returns default when not in config."""
        config_without_occasions = sample_neo4j_config.copy()
        config_without_occasions.pop("reference_nodes", None)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_without_occasions, f)
            temp_path = f.name
        
        try:
            config = Neo4jConfig(config_path=temp_path)
            occasions = config.get_occasions()
            assert "CASUAL" in occasions
            assert "BUSINESS" in occasions
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_get_categories_default(self, sample_neo4j_config):
        """Test get_categories returns default when not in config."""
        config_without_categories = sample_neo4j_config.copy()
        config_without_categories.pop("reference_nodes", None)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_without_categories, f)
            temp_path = f.name
        
        try:
            config = Neo4jConfig(config_path=temp_path)
            categories = config.get_categories()
            assert "TOP" in categories
            assert "BOTTOM" in categories
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ============================================================================
# UTILITY METHODS TESTS
# ============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""
    
    def test_to_neo4j_driver_config(self, temp_config_file):
        """Test to_neo4j_driver_config returns correct dictionary."""
        config = Neo4jConfig(config_path=temp_config_file)
        driver_config = config.to_neo4j_driver_config()

        assert driver_config["uri"] == "bolt://localhost:7687"
        assert driver_config["auth"] == ("neo4j", "test_password")
        # 'database' is a session-level key, NOT a driver-level key;
        # it must not be present in the driver config dict.
        assert "database" not in driver_config
        assert "connection_timeout" in driver_config
        assert "max_connection_pool_size" in driver_config
    
    def test_to_dict(self, temp_config_file):
        """Test to_dict returns full configuration."""
        config = Neo4jConfig(config_path=temp_config_file)
        config_dict = config.to_dict()
        
        assert "connection" in config_dict
        assert "query_cache" in config_dict
        assert "retry_policy" in config_dict
    
    def test_reload(self, sample_neo4j_config):
        """Test reload method reloads configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_neo4j_config, f)
            temp_path = f.name
        
        try:
            config = Neo4jConfig(config_path=temp_path)
            original_uri = config.get_uri()
            
            # Modify the config file
            sample_neo4j_config["connection"]["uri"] = "bolt://localhost:7688"
            with open(temp_path, 'w') as f:
                json.dump(sample_neo4j_config, f)
            
            # Reload
            config.reload()
            new_uri = config.get_uri()
            
            assert new_uri == "bolt://localhost:7688"
            assert original_uri != new_uri
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ============================================================================
# SINGLETON PATTERN TESTS
# ============================================================================


class TestSingletonPattern:
    """Tests for singleton pattern implementation."""
    
    def test_get_neo4j_config_singleton(self):
        """Test get_neo4j_config returns singleton."""
        reset_neo4j_config()
        
        with patch('src.core.neo4j_config.Neo4jConfig') as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            
            config1 = get_neo4j_config()
            config2 = get_neo4j_config()
            
            assert config1 is config2
            mock_config_class.assert_called_once()
    
    def test_reset_neo4j_config(self):
        """Test reset_neo4j_config clears singleton."""
        # Clear any existing config
        reset_neo4j_config()
        
        with patch('src.core.neo4j_config.Neo4jConfig') as mock_config_class:
            instances = [MagicMock(), MagicMock()]  # Create two different instances
            mock_config_class.side_effect = instances
            
            # First call
            config1 = get_neo4j_config()
            assert config1 is instances[0]
            
            # Reset
            reset_neo4j_config()
            
            # Second call (should create a new instance)
            config2 = get_neo4j_config()
            assert config2 is instances[1]
            
            # Should be called twice after reset
            assert mock_config_class.call_count == 2


# ============================================================================
# DEFAULT VALUES TESTS
# ============================================================================


class TestDefaultValues:
    """Tests for default values when config keys are missing."""
    
    def test_default_values_minimal_config(self):
        """Test default values are used for missing config keys."""
        minimal_config = {
            "connection": {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            temp_path = f.name
        
        try:
            config = Neo4jConfig(config_path=temp_path)
            
            # Test defaults
            assert config.get_database() == "neo4j"
            assert config.get_timeout() == 30
            assert config.get_max_pool_size() == 50
            assert config.is_cache_enabled() is True
            assert config.is_retry_enabled() is True
        finally:
            Path(temp_path).unlink(missing_ok=True)
