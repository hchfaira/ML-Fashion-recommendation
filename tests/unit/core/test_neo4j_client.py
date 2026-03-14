"""
Tests for Neo4j Client

Comprehensive unit tests for the Neo4j wrapper client including:
- Connection management
- Query execution with caching
- Query cache (LRU + TTL)
- Error handling and retries
- Batch operations
- Health checks
- Concurrent operations

Markers:
    @pytest.mark.unit: Fast unit tests
    @pytest.mark.asyncio: Async tests
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call
from datetime import datetime, timedelta
import time

from neo4j.exceptions import ServiceUnavailable, CypherSyntaxError, DatabaseError

from src.core.neo4j_client import (
    Neo4jClient,
    QueryCache,
    Neo4jException,
)


# ============================================================================
# HELPERS
# ============================================================================


def make_session_cm(mock_session):
    """
    Return a synchronous callable that, when called, returns an async
    context manager yielding *mock_session*.

    Neo4jClient uses::

        async with self.driver.session() as session:
            ...

    So ``driver.session`` must be a plain callable (not a coroutine function)
    that returns an object supporting ``__aenter__`` / ``__aexit__``.
    """
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=False)

    # driver.session() returns cm
    factory = MagicMock(return_value=cm)
    return factory


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j AsyncDriver."""
    driver = MagicMock()
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def neo4j_config():
    """Mock Neo4j configuration."""
    config = MagicMock()
    config.get_uri.return_value = "bolt://localhost:7687"
    config.get_username.return_value = "neo4j"
    config.get_password.return_value = "password"
    config.get_database.return_value = "neo4j"
    config.get_timeout.return_value = 30
    config.get_max_pool_size.return_value = 50
    config.get_cache_max_size.return_value = 1000
    config.get_cache_default_ttl.return_value = 3600
    config.is_retry_enabled.return_value = True
    config.get_max_retries.return_value = 3
    config.get_initial_backoff_ms.return_value = 100
    config.get_max_backoff_ms.return_value = 5000
    config.get_backoff_multiplier.return_value = 2.0
    config.to_neo4j_driver_config.return_value = {
        "uri": "bolt://localhost:7687",
        "auth": ("neo4j", "password"),
    }
    return config


@pytest.fixture
async def neo4j_client(neo4j_config):
    """Create a Neo4jClient with mocked driver."""
    with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver_factory:
        mock_driver = MagicMock()
        mock_driver.close = AsyncMock()
        mock_driver_factory.return_value = mock_driver

        with patch('src.core.neo4j_client.get_neo4j_config', return_value=neo4j_config):
            client = Neo4jClient()
            client.driver = mock_driver
            yield client
            await client.close()


# ============================================================================
# QUERY CACHE TESTS
# ============================================================================


class TestQueryCache:
    """Tests for QueryCache class."""
    
    def test_cache_initialization(self):
        """Test cache is initialized with correct parameters."""
        cache = QueryCache(max_size=100)
        assert cache.max_size == 100
        assert cache._cache == {}
        assert cache._timestamps == {}
    
    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        cache = QueryCache(max_size=100)
        
        cache.set("key1", "value1", ttl=3600)
        result = cache.get("key1")
        
        assert result == "value1"
    
    def test_cache_get_expired_value(self):
        """Test get returns None for expired values."""
        cache = QueryCache(max_size=100)
        
        cache.set("key1", "value1", ttl=1)
        time.sleep(1.1)
        
        result = cache.get("key1")
        assert result is None
    
    def test_cache_get_missing_key(self):
        """Test get returns None for missing keys."""
        cache = QueryCache(max_size=100)
        result = cache.get("nonexistent")
        assert result is None
    
    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = QueryCache(max_size=100)
        
        cache.set("key1", "value1", ttl=3600)
        cache.set("key2", "value2", ttl=3600)
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when max_size exceeded."""
        cache = QueryCache(max_size=3)
        
        cache.set("key1", "value1", ttl=3600)
        cache.set("key2", "value2", ttl=3600)
        cache.set("key3", "value3", ttl=3600)
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new key, should evict key2 (least recently used)
        cache.set("key4", "value4", ttl=3600)
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_cache_stats_hits_and_misses(self):
        """Test cache statistics tracking."""
        cache = QueryCache(max_size=100)
        
        cache.set("key1", "value1", ttl=3600)
        
        # Hit
        cache.get("key1")
        # Miss
        cache.get("key2")
        # Hit
        cache.get("key1")
        
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.667, rel=0.01)


# ============================================================================
# NEO4J CLIENT INITIALIZATION TESTS
# ============================================================================


class TestNeo4jClientInitialization:
    """Tests for Neo4jClient initialization."""

    @pytest.mark.asyncio
    async def test_client_initialization(self, neo4j_config):
        """Test client initializes correctly."""
        with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver_factory.return_value = mock_driver
            neo4j_config.to_neo4j_driver_config.return_value = {
                "uri": "bolt://localhost:7687",
                "auth": ("neo4j", "password"),
            }

            with patch('src.core.neo4j_client.get_neo4j_config', return_value=neo4j_config):
                client = Neo4jClient()

                assert client.driver is not None
                assert client.cache is not None
                await client.close()

    @pytest.mark.asyncio
    async def test_client_initialization_with_custom_config_path(self):
        """Test client can be initialized with custom config path."""
        mock_config = MagicMock()
        mock_config.get_uri.return_value = "bolt://localhost:7687"
        mock_config.get_username.return_value = "neo4j"
        mock_config.get_password.return_value = "password"
        mock_config.to_neo4j_driver_config.return_value = {
            "uri": "bolt://localhost:7687",
            "auth": ("neo4j", "password"),
        }
        mock_config.get_cache_max_size.return_value = 1000
        mock_config.get_cache_default_ttl.return_value = 3600
        mock_config.get_max_retries.return_value = 3
        mock_config.get_initial_backoff_ms.return_value = 100
        mock_config.get_max_backoff_ms.return_value = 5000
        mock_config.get_backoff_multiplier.return_value = 2.0

        with patch('neo4j.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver_factory.return_value = mock_driver

            with patch('src.core.neo4j_client.get_neo4j_config', return_value=mock_config):
                client = Neo4jClient(config_path="/custom/path")

                assert client.driver is not None
                await client.close()


# ============================================================================
# QUERY EXECUTION TESTS
# ============================================================================


class TestQueryExecution:
    """Tests for query execution methods."""
    
    @pytest.mark.asyncio
    async def test_query_execution(self, neo4j_client):
        """Test basic query execution."""
        # Mock session and result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"garment_id": "g1", "weight": 0.9}
        mock_result.data.return_value = [mock_record]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        result = await neo4j_client.query(
            "MATCH (g:Garment) RETURN g",
            params={}
        )
        
        assert result == [mock_record]
    
    @pytest.mark.asyncio
    async def test_query_with_caching(self, neo4j_client):
        """Test query results are cached."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"garment_id": "g1"}
        mock_result.data.return_value = [mock_record]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        # First query
        result1 = await neo4j_client.query(
            "MATCH (g:Garment) RETURN g",
            params={},
            ttl=3600
        )
        
        # Second query (should use cache)
        result2 = await neo4j_client.query(
            "MATCH (g:Garment) RETURN g",
            params={},
            ttl=3600
        )
        
        assert result1 == result2
        # Driver should only be called once (for the first query)
        assert mock_session.run.call_count == 1
    
    @pytest.mark.asyncio
    async def test_query_cache_ttl_expiration(self, neo4j_client):
        """Test cache expires after TTL."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"garment_id": "g1"}
        mock_result.data.return_value = [mock_record]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        # First query with short TTL
        result1 = await neo4j_client.query(
            "MATCH (g:Garment) RETURN g",
            params={},
            ttl=1
        )
        
        # Wait for TTL to expire
        await asyncio.sleep(1.1)
        
        # Second query (cache expired, should re-query)
        result2 = await neo4j_client.query(
            "MATCH (g:Garment) RETURN g",
            params={},
            ttl=1
        )
        
        assert result1 == result2
        # Driver should be called twice (cache expired)
        assert mock_session.run.call_count == 2
    
    @pytest.mark.asyncio
    async def test_query_with_parameters(self, neo4j_client):
        """Test query execution with parameters."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"garment_id": "g1", "weight": 0.9}
        mock_result.data.return_value = [mock_record]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        result = await neo4j_client.query(
            "MATCH (g:Garment {id: $id}) RETURN g",
            params={"id": "g1"}
        )
        
        assert result == [mock_record]


# ============================================================================
# SPECIALIZED QUERY TESTS
# ============================================================================


class TestSpecializedQueries:
    """Tests for specialized query methods."""
    
    @pytest.mark.asyncio
    async def test_get_compatible_items(self, neo4j_client):
        """Test get_compatible_items method."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_records = [
            {"garment_id": "g2", "weight": 0.95},
            {"garment_id": "g3", "weight": 0.85},
        ]
        mock_result.data.return_value = mock_records
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        result = await neo4j_client.get_compatible_items(
            garment_id="g1",
            top_k=5,
            min_weight=0.5
        )
        
        assert result == mock_records
    
    @pytest.mark.asyncio
    async def test_get_edge_weight(self, neo4j_client):
        """Test get_edge_weight method."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"weight": 0.85}
        mock_result.data.return_value = [mock_record]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        result = await neo4j_client.get_edge_weight(
            from_id="g1",
            to_id="g2",
            relationship="COMPATIBLE_WITH"
        )
        
        assert result == {"weight": 0.85}
    
    @pytest.mark.asyncio
    async def test_get_centrality_scores(self, neo4j_client):
        """Test get_centrality_scores method."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {
            "degree": 15,
            "betweenness": 0.42,
            "eigenvector": 0.78,
            "pagerank": 0.05
        }
        mock_result.data.return_value = [mock_record]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        result = await neo4j_client.get_centrality_scores(garment_id="g1")
        
        assert result == mock_record
    
    @pytest.mark.asyncio
    async def test_find_outfit_paths(self, neo4j_client):
        """Test find_outfit_paths method."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_paths = [
            ["g1", "g2", "g3", "g4"],
            ["g1", "g5", "g6", "g7"],
        ]
        mock_result.data.return_value = [{"path": p} for p in mock_paths]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        result = await neo4j_client.find_outfit_paths(
            anchor_id="g1",
            path_length=4,
            min_weight=0.6
        )
        
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_get_garments_by_season(self, neo4j_client):
        """Test get_garments_by_season method."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_records = [
            {"garment_id": "g1", "title": "Summer Dress"},
            {"garment_id": "g2", "title": "Shorts"},
        ]
        mock_result.data.return_value = mock_records
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        result = await neo4j_client.get_garments_by_season(
            season="SUMMER",
            user_id="user1"
        )
        
        assert result == mock_records
    
    @pytest.mark.asyncio
    async def test_get_garments_by_occasion(self, neo4j_client):
        """Test get_garments_by_occasion method."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_records = [
            {"garment_id": "g1", "title": "Suit Jacket"},
            {"garment_id": "g2", "title": "Dress Pants"},
        ]
        mock_result.data.return_value = mock_records
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        result = await neo4j_client.get_garments_by_occasion(
            occasion="BUSINESS",
            user_id="user1"
        )
        
        assert result == mock_records
    
    @pytest.mark.asyncio
    async def test_get_complementary_colors(self, neo4j_client):
        """Test get_complementary_colors method."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_records = [
            {"color": "orange", "harmony_type": "COMPLEMENTS"},
            {"color": "yellow", "harmony_type": "ANALOGOUS"},
        ]
        mock_result.data.return_value = mock_records
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        result = await neo4j_client.get_complementary_colors(
            color="blue",
            limit=5
        )
        
        assert result == mock_records


# ============================================================================
# BATCH OPERATIONS TESTS
# ============================================================================


class TestBatchOperations:
    """Tests for batch query operations."""
    
    @pytest.mark.asyncio
    async def test_batch_query_execution(self, neo4j_client):
        """Test batch_query executes multiple queries."""
        mock_session = AsyncMock()

        def _make_result(val):
            r = MagicMock()
            r.data = AsyncMock(return_value=[{"result": val}])
            return r

        mock_session.run.side_effect = [_make_result(1), _make_result(2), _make_result(3)]
        neo4j_client.driver.session = make_session_cm(mock_session)

        queries = [
            {"cypher": "MATCH (g:Garment) RETURN g", "params": {}},
            {"cypher": "MATCH (s:Season) RETURN s", "params": {}},
            {"cypher": "MATCH (c:Color) RETURN c", "params": {}},
        ]

        batch_result = await neo4j_client.batch_query(queries)

        assert mock_session.run.call_count == 3
        assert len(batch_result) == 3
    
    @pytest.mark.asyncio
    async def test_batch_query_with_empty_list(self, neo4j_client):
        """Test batch_query handles empty query list."""
        result = await neo4j_client.batch_query([])
        assert result == []


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and retries."""
    
    @pytest.mark.asyncio
    async def test_query_raises_neo4j_exception(self, neo4j_client):
        """Test query raises Neo4jException on driver error."""
        mock_session = AsyncMock()
        mock_session.run.side_effect = ServiceUnavailable("Connection failed")
        neo4j_client.driver.session = make_session_cm(mock_session)

        with pytest.raises(Neo4jException):
            await neo4j_client.query("INVALID CYPHER")

    @pytest.mark.asyncio
    async def test_query_retry_logic(self, neo4j_client):
        """Test query retry on transient failures."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_record = {"result": "success"}
        mock_result.data = AsyncMock(return_value=[mock_record])

        # Fail twice, then succeed
        mock_session.run.side_effect = [
            ServiceUnavailable("Transient error"),
            ServiceUnavailable("Transient error"),
            mock_result,
        ]

        neo4j_client.driver.session = make_session_cm(mock_session)

        result = await neo4j_client.query("MATCH (g:Garment) RETURN g")

        assert result == [mock_record]
        assert mock_session.run.call_count == 3


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, neo4j_client):
        """Test successful health check."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [{"result": "OK"}]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        is_healthy = await neo4j_client.health_check()
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, neo4j_client):
        """Test failed health check."""
        mock_session = AsyncMock()
        mock_session.run.side_effect = Exception("Connection failed")
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        is_healthy = await neo4j_client.health_check()
        
        assert is_healthy is False


# ============================================================================
# STATISTICS TESTS
# ============================================================================


class TestStatistics:
    """Tests for statistics collection."""
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, neo4j_client):
        """Test that statistics are tracked."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [{"result": 1}]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        # Execute a query
        await neo4j_client.query("MATCH (g:Garment) RETURN g")
        
        stats = neo4j_client.get_statistics()
        
        assert stats["queries_executed"] >= 1


# ============================================================================
# CONCURRENT OPERATIONS TESTS
# ============================================================================


class TestConcurrentOperations:
    """Tests for concurrent query execution."""
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, neo4j_client):
        """Test multiple concurrent queries."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [{"result": 1}]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        tasks = [
            neo4j_client.query("MATCH (g:Garment) RETURN g"),
            neo4j_client.query("MATCH (s:Season) RETURN s"),
            neo4j_client.query("MATCH (c:Color) RETURN c"),
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(r == [{"result": 1}] for r in results)


# ============================================================================
# CLEANUP TESTS
# ============================================================================


class TestCleanup:
    """Tests for client cleanup."""
    
    @pytest.mark.asyncio
    async def test_close_client(self, neo4j_client):
        """Test closing client closes driver."""
        await neo4j_client.close()
        neo4j_client.driver.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, neo4j_client):
        """Test clearing cache."""
        # Add something to cache
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [{"result": 1}]
        mock_session.run.return_value = mock_result
        
        neo4j_client.driver.session = make_session_cm(mock_session)
        
        await neo4j_client.query("MATCH (g:Garment) RETURN g", ttl=3600)
        
        # Clear cache
        neo4j_client.clear_cache()
        
        # Verify cache is empty by checking statistics
        stats = neo4j_client.cache.stats()
        assert stats["size"] == 0
