"""
Neo4j Client
============

Lightweight async wrapper around Neo4j driver with caching support.
Handles connection pooling, query execution, and result caching.

Features:
- Async driver for non-blocking operations
- LRU in-memory caching with TTL
- Automatic retry logic with exponential backoff
- Built-in query statistics and health checks
- Specialized query methods (compatible items, paths, centrality, etc.)

Usage:
    from src.core.neo4j_config import get_neo4j_config
    from src.core.neo4j_client import Neo4jClient
    
    config = get_neo4j_config()
    client = Neo4jClient(config_path="path/to/neo4j_config.json")
    
    # Execute query
    result = await client.query("MATCH (g:Garment) RETURN g LIMIT 10")
    
    # Get specialized data
    items = await client.get_compatible_items("garment_id", top_k=5)
    scores = await client.get_centrality_scores("garment_id")
    
    await client.close()
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
import hashlib
import time

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    from neo4j.exceptions import ServiceUnavailable, CypherSyntaxError, DatabaseError
except ImportError:
    raise ImportError("neo4j package required: pip install neo4j")

from src.core.neo4j_config import get_neo4j_config

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class Neo4jException(Exception):
    """Base exception for Neo4j client errors."""
    pass


# ============================================================================
# QUERY CACHE
# ============================================================================


class QueryCache:
    """LRU cache with TTL support for query results."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._access_order: List[str] = []
        self._stats = {"hits": 0, "misses": 0}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if expired/missing
        """
        if key not in self._cache:
            self._stats["misses"] += 1
            return None
        
        cached_item = self._cache[key]
        ttl = cached_item.get("ttl", float('inf'))
        
        # Check if expired
        if time.time() - self._timestamps[key] > ttl:
            del self._cache[key]
            del self._timestamps[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._stats["misses"] += 1
            return None
        
        # Update access order (for LRU)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        self._stats["hits"] += 1
        return cached_item["value"]
    
    def set(self, key: str, value: Any, ttl: int) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        # Evict least recently used if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                if lru_key in self._cache:
                    del self._cache[lru_key]
                    del self._timestamps[lru_key]
        
        self._cache[key] = {"value": value, "ttl": ttl}
        self._timestamps[key] = time.time()
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._timestamps.clear()
        self._access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }


# ============================================================================
# NEO4J CLIENT
# ============================================================================


class Neo4jClient:
    """
    Async Neo4j client with caching and connection pooling.
    
    Provides high-level interface for Neo4j operations with automatic
    caching, retry logic, and query statistics.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Neo4j client.
        
        Args:
            config_path: Path to neo4j_config.json file
        """
        self.config = get_neo4j_config() if config_path is None else get_neo4j_config()
        self.driver: Optional[AsyncDriver] = None
        self.cache = QueryCache(max_size=self.config.get_cache_max_size())
        
        # Statistics
        self._stats = {
            "queries_executed": 0,
            "queries_cached": 0,
            "total_execution_time_ms": 0.0,
            "last_health_check": None,
            "health_status": "unknown"
        }
        
        self._initialize_driver()
    
    def _initialize_driver(self) -> None:
        """Initialize Neo4j driver from configuration."""
        try:
            driver_config = self.config.to_neo4j_driver_config()
            self.driver = AsyncGraphDatabase.driver(**driver_config)
            logger.info(f"Neo4j driver initialized for {self.config.get_uri()}")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise Neo4jException(f"Driver initialization failed: {e}")
    
    async def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")
    
    def _make_cache_key(self, cypher: str, params: Dict[str, Any]) -> str:
        """Generate cache key from query and parameters."""
        params_str = str(sorted(params.items()))
        key_data = f"{cypher}:{params_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        ttl: int = 3600
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query with caching.
        
        Args:
            cypher: Cypher query string
            params: Query parameters dict
            ttl: Cache TTL in seconds
            
        Returns:
            List of result records as dicts
            
        Raises:
            Neo4jException: On query execution error
        """
        if not self.driver:
            raise Neo4jException("Neo4j driver not initialized")
        
        params = params or {}
        cache_key = self._make_cache_key(cypher, params)
        
        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._stats["queries_cached"] += 1
            return cached_result
        
        # Execute query with retry logic
        max_retries = self.config.get_max_retries()
        backoff_ms = self.config.get_initial_backoff_ms()
        max_backoff_ms = self.config.get_max_backoff_ms()
        multiplier = self.config.get_backoff_multiplier()
        
        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                async with self.driver.session() as session:
                    result = await session.run(cypher, params)
                    records = [dict(r) for r in await result.data()]
                
                execution_time_ms = (time.time() - start_time) * 1000
                self._stats["queries_executed"] += 1
                self._stats["total_execution_time_ms"] += execution_time_ms
                
                # Cache result
                self.cache.set(cache_key, records, ttl)
                
                logger.debug(f"Query executed in {execution_time_ms:.2f}ms")
                return records
            
            except (ServiceUnavailable, DatabaseError, CypherSyntaxError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(backoff_ms * (multiplier ** attempt), max_backoff_ms) / 1000
                    logger.warning(f"Query failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.2f}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Query failed after {max_retries} attempts: {e}")
                    raise Neo4jException(f"Query execution failed: {e}")
        
        raise Neo4jException(f"Query failed: {last_error}")
    
    async def get_compatible_items(
        self,
        garment_id: str,
        top_k: int = 5,
        min_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get compatible items for a garment.
        
        Args:
            garment_id: ID of the garment
            top_k: Number of results
            min_weight: Minimum compatibility weight
            
        Returns:
            List of compatible garment records
        """
        query = """
        MATCH (g:Garment {id: $garment_id})
              -[r:COMPATIBLE_WITH]-> (compatible:Garment)
        WHERE r.weight >= $min_weight
        RETURN compatible.id as garment_id, r.weight as weight
        ORDER BY r.weight DESC
        LIMIT $top_k
        """
        
        return await self.query(query, {
            'garment_id': garment_id,
            'top_k': top_k,
            'min_weight': min_weight
        })
    
    async def get_edge_weight(
        self,
        from_id: str,
        to_id: str,
        relationship: str = "COMPATIBLE_WITH"
    ) -> Optional[Dict[str, Any]]:
        """Get edge weight and properties between two nodes."""
        query = f"""
        MATCH (a:Garment {{id: $from_id}})
              -[r:{relationship}]->(b:Garment {{id: $to_id}})
        RETURN r.weight as weight, properties(r) as properties
        """
        
        result = await self.query(query, {
            'from_id': from_id,
            'to_id': to_id
        })
        
        return result[0] if result else None
    
    async def get_centrality_scores(
        self,
        garment_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get pre-computed centrality scores for a garment.
        
        Args:
            garment_id: ID of garment
            
        Returns:
            Dict with centrality scores
        """
        query = """
        MATCH (g:Garment {id: $garment_id})
        RETURN g.degree_centrality as degree,
               g.betweenness_centrality as betweenness,
               g.eigenvector_centrality as eigenvector,
               g.pagerank as pagerank
        """
        
        result = await self.query(query, {'garment_id': garment_id})
        return result[0] if result else None
    
    async def find_outfit_paths(
        self,
        anchor_id: str,
        path_length: int = 4,
        min_weight: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Find valid outfit paths starting from anchor item."""
        query = f"""
        MATCH path = (anchor:Garment {{id: $anchor_id}})
              -[:COMPATIBLE_WITH*{path_length-1}]->(end:Garment)
        WHERE all(rel in relationships(path) WHERE rel.weight > $min_weight)
        RETURN [node in nodes(path) | node.id] as path
        LIMIT 20
        """
        
        return await self.query(query, {
            'anchor_id': anchor_id,
            'min_weight': min_weight
        })
    
    async def get_garments_by_season(
        self,
        season: str,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get garments suitable for a season."""
        query = """
        MATCH (g:Garment {user_id: $user_id})
              -[r:SUITABLE_FOR]-> (s:Season {value: $season})
        RETURN g.id as garment_id, g.title as title, r.weight as score
        ORDER BY r.weight DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        return await self.query(query, {
            'season': season,
            'user_id': user_id
        })
    
    async def get_garments_by_occasion(
        self,
        occasion: str,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get garments suitable for an occasion."""
        query = """
        MATCH (g:Garment {user_id: $user_id})
              -[r:SUITABLE_FOR_OCCASION]-> (o:Occasion {value: $occasion})
        RETURN g.id as garment_id, g.title as title, r.weight as score
        ORDER BY r.weight DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        return await self.query(query, {
            'occasion': occasion,
            'user_id': user_id
        })
    
    async def get_complementary_colors(
        self,
        color: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get complementary colors for a color."""
        query = """
        MATCH (c1:Color {name: $color})
              -[r]-> (c2:Color)
        RETURN c2.name as color, type(r) as harmony_type, r.weight as score
        ORDER BY r.weight DESC
        LIMIT $limit
        """
        
        return await self.query(query, {
            'color': color,
            'limit': limit
        })
    
    async def batch_query(
        self,
        queries_list: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Execute multiple queries efficiently.
        
        Args:
            queries_list: List of dicts with 'cypher' and 'params' keys
            
        Returns:
            List of result lists (one per query)
        """
        results = []
        for query_dict in queries_list:
            result = await self.query(
                query_dict.get("cypher", ""),
                query_dict.get("params", {})
            )
            results.append(result)
        
        return results
    
    async def health_check(self) -> bool:
        """
        Check if Neo4j is accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            result = await self.query("RETURN 1 as result")
            self._stats["health_status"] = "healthy"
            self._stats["last_health_check"] = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self._stats["health_status"] = "unhealthy"
            self._stats["last_health_check"] = datetime.now().isoformat()
            return False
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.cache.clear()
        logger.info("Query cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self._stats,
            "cache_stats": self.cache.stats()
        }
