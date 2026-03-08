"""
Outfit Storage System
Hybrid storage for garments, outfits, and compatibility relationships.

Supports:
- JSON Files: Local tests and development (`data/outfits/*.json`)
- Neo4j Graph Database: Local and production with relationship queries

The graph database is the "king" for fashion because it stores RELATIONSHIPS:
- Node A (Blue Jeans) COMPATIBLE_WITH Node B (Sneakers)
- Query: "Find all paths (outfits) connecting a Top, Bottom, and Shoes with COMPATIBLE relationship"
"""
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
from enum import Enum

from src.core.models import (
    Garment, GarmentAttributes, Outfit, OutfitItem,
    GarmentCategory, ColorProfile, MaterialProfile, PatternInfo
)
from src.core import get_logger

logger = get_logger(__name__)


# ============== Relationship Types ==============

class RelationType(str, Enum):
    """Types of relationships between garments."""
    COMPATIBLE = "COMPATIBLE"           # General compatibility
    PAIRS_WELL = "PAIRS_WELL"           # Strong pairing
    COLOR_HARMONY = "COLOR_HARMONY"     # Colors work together
    STYLE_MATCH = "STYLE_MATCH"         # Same style aesthetic
    FORMALITY_MATCH = "FORMALITY_MATCH" # Same formality level
    SEASON_MATCH = "SEASON_MATCH"       # Same season suitability
    LAYERABLE = "LAYERABLE"             # Can layer together
    CONTRASTS_WITH = "CONTRASTS_WITH"   # Creates interesting contrast
    AVOIDS = "AVOIDS"                   # Should not be paired


class CompatibilityScore:
    """Compatibility relationship with score and metadata."""
    
    def __init__(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        score: float,
        reasons: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.score = score
        self.reasons = reasons or []
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "score": self.score,
            "reasons": self.reasons,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompatibilityScore":
        instance = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            score=data["score"],
            reasons=data.get("reasons", []),
            metadata=data.get("metadata", {})
        )
        if "created_at" in data:
            instance.created_at = datetime.fromisoformat(data["created_at"])
        return instance


# ============== Abstract Storage Interface ==============

class OutfitStorageInterface(ABC):
    """
    Abstract interface for outfit storage.
    
    Implementations must provide methods for:
    - Garment CRUD operations
    - Outfit CRUD operations
    - Compatibility relationship management
    - Graph queries for outfit building
    """
    
    # ==================== Garment Operations ====================
    
    @abstractmethod
    async def save_garment(self, garment: Garment) -> str:
        """Save a garment and return its ID."""
        pass
    
    @abstractmethod
    async def get_garment(self, garment_id: str) -> Optional[Garment]:
        """Get a garment by ID."""
        pass
    
    @abstractmethod
    async def get_all_garments(self) -> List[Garment]:
        """Get all garments."""
        pass
    
    @abstractmethod
    async def get_garments_by_category(self, category: GarmentCategory) -> List[Garment]:
        """Get all garments of a specific category."""
        pass
    
    @abstractmethod
    async def delete_garment(self, garment_id: str) -> bool:
        """Delete a garment by ID."""
        pass
    
    # ==================== Outfit Operations ====================
    
    @abstractmethod
    async def save_outfit(self, outfit: Outfit) -> str:
        """Save an outfit and return its ID."""
        pass
    
    @abstractmethod
    async def get_outfit(self, outfit_id: str) -> Optional[Outfit]:
        """Get an outfit by ID."""
        pass
    
    @abstractmethod
    async def get_all_outfits(self) -> List[Outfit]:
        """Get all saved outfits."""
        pass
    
    @abstractmethod
    async def delete_outfit(self, outfit_id: str) -> bool:
        """Delete an outfit by ID."""
        pass
    
    # ==================== Compatibility Operations ====================
    
    @abstractmethod
    async def save_compatibility(
        self,
        garment_id_1: str,
        garment_id_2: str,
        relation_type: RelationType,
        score: float,
        reasons: Optional[List[str]] = None
    ) -> None:
        """
        Save a compatibility relationship between two garments.
        
        Args:
            garment_id_1: First garment ID
            garment_id_2: Second garment ID
            relation_type: Type of relationship
            score: Compatibility score (0-1)
            reasons: List of reasons for this compatibility
        """
        pass
    
    @abstractmethod
    async def get_compatible_garments(
        self,
        garment_id: str,
        relation_type: Optional[RelationType] = None,
        min_score: float = 0.0
    ) -> List[Tuple[Garment, CompatibilityScore]]:
        """
        Get all garments compatible with a given garment.
        
        Args:
            garment_id: The garment to find matches for
            relation_type: Optional filter by relation type
            min_score: Minimum compatibility score
            
        Returns:
            List of (garment, compatibility_score) tuples
        """
        pass
    
    @abstractmethod
    async def find_outfit_paths(
        self,
        categories: List[GarmentCategory],
        relation_type: RelationType = RelationType.COMPATIBLE,
        min_score: float = 0.7
    ) -> List[List[Garment]]:
        """
        Find all outfit paths connecting garments from specified categories.
        
        This is the "graph query" power:
        "Find all paths connecting a Top, Bottom, and Shoes with COMPATIBLE relationship"
        
        Args:
            categories: List of categories to include (e.g., [TOP, BOTTOM, SHOES])
            relation_type: Required relationship type between items
            min_score: Minimum compatibility score for each relationship
            
        Returns:
            List of garment lists, each representing a valid outfit
        """
        pass
    
    # ==================== Statistics ====================
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass


# ============== JSON File Storage ==============

class JSONOutfitStorage(OutfitStorageInterface):
    """
    JSON file-based storage for local testing and development.
    
    Structure:
        data/outfits/
            garments.json          - All garments
            outfits.json           - All saved outfits
            compatibility.json     - Compatibility relationships
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize JSON storage.
        
        Args:
            base_path: Base path for JSON files (default: data/outfits/)
        """
        self.base_path = base_path or Path("data/outfits")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.garments_file = self.base_path / "garments.json"
        self.outfits_file = self.base_path / "outfits.json"
        self.compatibility_file = self.base_path / "compatibility.json"
        
        # In-memory cache (loaded from files)
        self._garments: Dict[str, Garment] = {}
        self._outfits: Dict[str, Outfit] = {}
        self._compatibility: Dict[str, List[CompatibilityScore]] = {}
        
        # Load existing data
        self._load_all()
        
        logger.info(f"JSONOutfitStorage initialized at {self.base_path}")
    
    def _load_all(self) -> None:
        """Load all data from JSON files."""
        self._load_garments()
        self._load_outfits()
        self._load_compatibility()
    
    def _load_garments(self) -> None:
        """Load garments from JSON file."""
        if self.garments_file.exists():
            try:
                with open(self.garments_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("garments", []):
                        garment = Garment.model_validate(item)
                        self._garments[garment.id] = garment
                logger.info(f"Loaded {len(self._garments)} garments from JSON")
            except Exception as e:
                logger.error(f"Error loading garments: {e}")
    
    def _load_outfits(self) -> None:
        """Load outfits from JSON file."""
        if self.outfits_file.exists():
            try:
                with open(self.outfits_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("outfits", []):
                        outfit = Outfit.model_validate(item)
                        self._outfits[outfit.id] = outfit
                logger.info(f"Loaded {len(self._outfits)} outfits from JSON")
            except Exception as e:
                logger.error(f"Error loading outfits: {e}")
    
    def _load_compatibility(self) -> None:
        """Load compatibility relationships from JSON file."""
        if self.compatibility_file.exists():
            try:
                with open(self.compatibility_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for garment_id, relations in data.get("compatibility", {}).items():
                        self._compatibility[garment_id] = [
                            CompatibilityScore.from_dict(r) for r in relations
                        ]
                total = sum(len(r) for r in self._compatibility.values())
                logger.info(f"Loaded {total} compatibility relationships from JSON")
            except Exception as e:
                logger.error(f"Error loading compatibility: {e}")
    
    def _save_garments(self) -> None:
        """Save garments to JSON file."""
        data = {
            "garments": [g.model_dump(mode="json") for g in self._garments.values()],
            "updated_at": datetime.utcnow().isoformat()
        }
        with open(self.garments_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_outfits(self) -> None:
        """Save outfits to JSON file."""
        data = {
            "outfits": [o.model_dump(mode="json") for o in self._outfits.values()],
            "updated_at": datetime.utcnow().isoformat()
        }
        with open(self.outfits_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_compatibility(self) -> None:
        """Save compatibility relationships to JSON file."""
        data = {
            "compatibility": {
                gid: [c.to_dict() for c in compat]
                for gid, compat in self._compatibility.items()
            },
            "updated_at": datetime.utcnow().isoformat()
        }
        with open(self.compatibility_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # ==================== Garment Operations ====================
    
    async def save_garment(self, garment: Garment) -> str:
        """Save a garment and return its ID."""
        if not garment.id:
            garment.id = str(uuid4())
        
        self._garments[garment.id] = garment
        self._save_garments()
        logger.debug(f"Saved garment {garment.id}")
        return garment.id
    
    async def get_garment(self, garment_id: str) -> Optional[Garment]:
        """Get a garment by ID."""
        return self._garments.get(garment_id)
    
    async def get_all_garments(self) -> List[Garment]:
        """Get all garments."""
        return list(self._garments.values())
    
    async def get_garments_by_category(self, category: GarmentCategory) -> List[Garment]:
        """Get all garments of a specific category."""
        return [g for g in self._garments.values() if g.attributes.category == category]
    
    async def delete_garment(self, garment_id: str) -> bool:
        """Delete a garment by ID."""
        if garment_id in self._garments:
            del self._garments[garment_id]
            # Also remove compatibility relationships
            if garment_id in self._compatibility:
                del self._compatibility[garment_id]
            # Remove references from other garments
            for gid in self._compatibility:
                self._compatibility[gid] = [
                    c for c in self._compatibility[gid]
                    if c.target_id != garment_id
                ]
            self._save_garments()
            self._save_compatibility()
            logger.info(f"Deleted garment {garment_id}")
            return True
        return False
    
    # ==================== Outfit Operations ====================
    
    async def save_outfit(self, outfit: Outfit) -> str:
        """Save an outfit and return its ID."""
        if not outfit.id:
            outfit.id = str(uuid4())
        
        self._outfits[outfit.id] = outfit
        self._save_outfits()
        logger.debug(f"Saved outfit {outfit.id}")
        return outfit.id
    
    async def get_outfit(self, outfit_id: str) -> Optional[Outfit]:
        """Get an outfit by ID."""
        return self._outfits.get(outfit_id)
    
    async def get_all_outfits(self) -> List[Outfit]:
        """Get all saved outfits."""
        return list(self._outfits.values())
    
    async def delete_outfit(self, outfit_id: str) -> bool:
        """Delete an outfit by ID."""
        if outfit_id in self._outfits:
            del self._outfits[outfit_id]
            self._save_outfits()
            logger.info(f"Deleted outfit {outfit_id}")
            return True
        return False
    
    # ==================== Compatibility Operations ====================
    
    async def save_compatibility(
        self,
        garment_id_1: str,
        garment_id_2: str,
        relation_type: RelationType,
        score: float,
        reasons: Optional[List[str]] = None
    ) -> None:
        """Save a compatibility relationship between two garments."""
        compat = CompatibilityScore(
            source_id=garment_id_1,
            target_id=garment_id_2,
            relation_type=relation_type,
            score=score,
            reasons=reasons
        )
        
        # Add to source garment's relationships
        if garment_id_1 not in self._compatibility:
            self._compatibility[garment_id_1] = []
        
        # Check if relationship already exists
        existing = [c for c in self._compatibility[garment_id_1] 
                    if c.target_id == garment_id_2 and c.relation_type == relation_type]
        if existing:
            # Update existing
            self._compatibility[garment_id_1].remove(existing[0])
        
        self._compatibility[garment_id_1].append(compat)
        
        # Also add reverse relationship for bidirectional queries
        reverse = CompatibilityScore(
            source_id=garment_id_2,
            target_id=garment_id_1,
            relation_type=relation_type,
            score=score,
            reasons=reasons
        )
        
        if garment_id_2 not in self._compatibility:
            self._compatibility[garment_id_2] = []
        
        existing_rev = [c for c in self._compatibility[garment_id_2]
                        if c.target_id == garment_id_1 and c.relation_type == relation_type]
        if existing_rev:
            self._compatibility[garment_id_2].remove(existing_rev[0])
        
        self._compatibility[garment_id_2].append(reverse)
        
        self._save_compatibility()
        logger.debug(f"Saved compatibility: {garment_id_1} <-{relation_type.value}-> {garment_id_2} (score: {score:.2f})")
    
    async def get_compatible_garments(
        self,
        garment_id: str,
        relation_type: Optional[RelationType] = None,
        min_score: float = 0.0
    ) -> List[Tuple[Garment, CompatibilityScore]]:
        """Get all garments compatible with a given garment."""
        if garment_id not in self._compatibility:
            return []
        
        results = []
        for compat in self._compatibility[garment_id]:
            # Filter by relation type
            if relation_type and compat.relation_type != relation_type:
                continue
            
            # Filter by score
            if compat.score < min_score:
                continue
            
            # Get the target garment
            target = self._garments.get(compat.target_id)
            if target:
                results.append((target, compat))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1].score, reverse=True)
        return results
    
    async def find_outfit_paths(
        self,
        categories: List[GarmentCategory],
        relation_type: RelationType = RelationType.COMPATIBLE,
        min_score: float = 0.7
    ) -> List[List[Garment]]:
        """
        Find all outfit paths connecting garments from specified categories.
        
        Uses recursive path finding through the compatibility graph.
        """
        if len(categories) < 2:
            # Return all garments from the single category
            if categories:
                garments = await self.get_garments_by_category(categories[0])
                return [[g] for g in garments]
            return []
        
        # Get all garments from first category as starting points
        first_category = categories[0]
        remaining_categories = categories[1:]
        
        starting_garments = await self.get_garments_by_category(first_category)
        
        all_paths = []
        
        for start_garment in starting_garments:
            # Find paths from this garment
            paths = await self._find_paths_from(
                current_path=[start_garment],
                remaining_categories=remaining_categories,
                relation_type=relation_type,
                min_score=min_score
            )
            all_paths.extend(paths)
        
        return all_paths
    
    async def _find_paths_from(
        self,
        current_path: List[Garment],
        remaining_categories: List[GarmentCategory],
        relation_type: RelationType,
        min_score: float
    ) -> List[List[Garment]]:
        """Recursive helper to find paths through the graph."""
        if not remaining_categories:
            return [current_path]
        
        last_garment = current_path[-1]
        next_category = remaining_categories[0]
        remaining = remaining_categories[1:]
        
        # Find compatible garments in the next category
        compatible = await self.get_compatible_garments(
            last_garment.id,
            relation_type=relation_type,
            min_score=min_score
        )
        
        # Filter to only the next category
        next_garments = [
            (g, c) for g, c in compatible
            if g.attributes.category == next_category
        ]
        
        paths = []
        for garment, _ in next_garments:
            # Recursively find paths
            new_paths = await self._find_paths_from(
                current_path=current_path + [garment],
                remaining_categories=remaining,
                relation_type=relation_type,
                min_score=min_score
            )
            paths.extend(new_paths)
        
        return paths
    
    # ==================== Statistics ====================
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_compat = sum(len(c) for c in self._compatibility.values())
        
        # Count by category
        category_counts = {}
        for garment in self._garments.values():
            cat = garment.attributes.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "storage_type": "json",
            "base_path": str(self.base_path),
            "total_garments": len(self._garments),
            "total_outfits": len(self._outfits),
            "total_compatibility_relations": total_compat // 2,  # Divided by 2 for bidirectional
            "garments_by_category": category_counts
        }


# ============== Neo4j Graph Storage ==============

class Neo4jOutfitStorage(OutfitStorageInterface):
    """
    Neo4j graph database storage for local and production.
    
    The graph database is the "king" for fashion because it stores RELATIONSHIPS:
    
    Nodes:
        - (:Garment {id, category, color, material, style_tags, ...})
        - (:Outfit {id, score, occasion, created_at})
    
    Relationships:
        - (:Garment)-[:COMPATIBLE {score, reasons}]->(:Garment)
        - (:Garment)-[:PAIRS_WELL {score}]->(:Garment)
        - (:Garment)-[:COLOR_HARMONY {score}]->(:Garment)
        - (:Garment)-[:PART_OF]->(:Outfit)
    
    Example Cypher queries:
        // Find all compatible outfits with Top + Bottom + Shoes
        MATCH path = (top:Garment {category: 'top'})-[:COMPATIBLE]->(bottom:Garment {category: 'bottom'})-[:COMPATIBLE]->(shoes:Garment {category: 'shoes'})
        WHERE ALL(r in relationships(path) WHERE r.score >= 0.7)
        RETURN top, bottom, shoes
        
        // Find best matches for a specific garment
        MATCH (g:Garment {id: $garmentId})-[r:COMPATIBLE]->(match:Garment)
        WHERE r.score >= 0.7
        RETURN match, r.score ORDER BY r.score DESC LIMIT 10
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j"
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (default: from NEO4J_URI env var or bolt://localhost:7687)
            username: Neo4j username (default: from NEO4J_USERNAME env var)
            password: Neo4j password (default: from NEO4J_PASSWORD env var)
            database: Database name (default: neo4j)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database
        
        self._driver = None
        self._initialized = False
        
        logger.info(f"Neo4jOutfitStorage configured for {self.uri}")
    
    async def _ensure_connection(self) -> None:
        """Ensure Neo4j driver is connected."""
        if self._driver is None:
            try:
                from neo4j import AsyncGraphDatabase
                self._driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
                await self._verify_connection()
                if not self._initialized:
                    await self._create_indexes()
                    self._initialized = True
                logger.info("Neo4j connection established")
            except ImportError:
                raise ImportError(
                    "neo4j package not installed. "
                    "Install with: pip install neo4j"
                )
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise
    
    async def _verify_connection(self) -> None:
        """Verify Neo4j connection is working."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("RETURN 1 as n")
            await result.single()
    
    async def _create_indexes(self) -> None:
        """Create necessary indexes for optimal performance."""
        async with self._driver.session(database=self.database) as session:
            # Index on garment ID
            await session.run(
                "CREATE INDEX garment_id IF NOT EXISTS FOR (g:Garment) ON (g.id)"
            )
            # Index on garment category
            await session.run(
                "CREATE INDEX garment_category IF NOT EXISTS FOR (g:Garment) ON (g.category)"
            )
            # Index on outfit ID
            await session.run(
                "CREATE INDEX outfit_id IF NOT EXISTS FOR (o:Outfit) ON (o.id)"
            )
            logger.info("Neo4j indexes created")
    
    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
    
    def _garment_to_node_props(self, garment: Garment) -> Dict[str, Any]:
        """Convert Garment to Neo4j node properties."""
        attrs = garment.attributes
        return {
            "id": garment.id,
            "image_url": garment.image_url,
            "image_path": garment.image_path,
            "category": attrs.category.value,
            "subcategory": attrs.subcategory,
            "color_primary": attrs.color.primary,
            "color_secondary": attrs.color.secondary,
            "color_temperature": attrs.color.color_temperature.value if attrs.color.color_temperature else None,
            "material_primary": attrs.material.primary if attrs.material else None,
            "pattern_type": attrs.pattern.type if attrs.pattern else "solid",
            "formality": attrs.formality_level.value if attrs.formality_level else "casual",
            "style_tags": attrs.style_tags,
            "seasons": [s.value for s in attrs.season_suitable] if attrs.season_suitable else [],
            "created_at": garment.created_at.isoformat(),
            # Store full JSON for reconstruction
            "attributes_json": garment.attributes.model_dump_json()
        }
    
    def _node_to_garment(self, node_props: Dict[str, Any]) -> Garment:
        """Convert Neo4j node properties to Garment."""
        attrs = GarmentAttributes.model_validate_json(node_props["attributes_json"])
        return Garment(
            id=node_props["id"],
            image_url=node_props.get("image_url"),
            image_path=node_props.get("image_path"),
            attributes=attrs,
            created_at=datetime.fromisoformat(node_props["created_at"])
        )
    
    # ==================== Garment Operations ====================
    
    async def save_garment(self, garment: Garment) -> str:
        """Save a garment to Neo4j."""
        await self._ensure_connection()
        
        if not garment.id:
            garment.id = str(uuid4())
        
        props = self._garment_to_node_props(garment)
        
        async with self._driver.session(database=self.database) as session:
            await session.run(
                """
                MERGE (g:Garment {id: $id})
                SET g = $props
                """,
                id=garment.id,
                props=props
            )
        
        logger.debug(f"Saved garment {garment.id} to Neo4j")
        return garment.id
    
    async def get_garment(self, garment_id: str) -> Optional[Garment]:
        """Get a garment from Neo4j by ID."""
        await self._ensure_connection()
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                "MATCH (g:Garment {id: $id}) RETURN g",
                id=garment_id
            )
            record = await result.single()
            if record:
                return self._node_to_garment(dict(record["g"]))
        return None
    
    async def get_all_garments(self) -> List[Garment]:
        """Get all garments from Neo4j."""
        await self._ensure_connection()
        
        garments = []
        async with self._driver.session(database=self.database) as session:
            result = await session.run("MATCH (g:Garment) RETURN g")
            async for record in result:
                garments.append(self._node_to_garment(dict(record["g"])))
        return garments
    
    async def get_garments_by_category(self, category: GarmentCategory) -> List[Garment]:
        """Get all garments of a specific category."""
        await self._ensure_connection()
        
        garments = []
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                "MATCH (g:Garment {category: $category}) RETURN g",
                category=category.value
            )
            async for record in result:
                garments.append(self._node_to_garment(dict(record["g"])))
        return garments
    
    async def delete_garment(self, garment_id: str) -> bool:
        """Delete a garment and its relationships from Neo4j."""
        await self._ensure_connection()
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (g:Garment {id: $id})
                DETACH DELETE g
                RETURN count(*) as deleted
                """,
                id=garment_id
            )
            record = await result.single()
            deleted = record["deleted"] > 0 if record else False
        
        if deleted:
            logger.info(f"Deleted garment {garment_id} from Neo4j")
        return deleted
    
    # ==================== Outfit Operations ====================
    
    async def save_outfit(self, outfit: Outfit) -> str:
        """Save an outfit to Neo4j."""
        await self._ensure_connection()
        
        if not outfit.id:
            outfit.id = str(uuid4())
        
        async with self._driver.session(database=self.database) as session:
            # Create/update outfit node
            await session.run(
                """
                MERGE (o:Outfit {id: $id})
                SET o.compatibility_score = $compat_score,
                    o.style_coherence_score = $style_score,
                    o.occasion_match_score = $occasion_score,
                    o.overall_score = $overall_score,
                    o.explanation = $explanation,
                    o.created_at = $created_at
                """,
                id=outfit.id,
                compat_score=outfit.compatibility_score,
                style_score=outfit.style_coherence_score,
                occasion_score=outfit.occasion_match_score,
                overall_score=outfit.overall_score,
                explanation=outfit.explanation,
                created_at=outfit.created_at.isoformat()
            )
            
            # Remove existing PART_OF relationships
            await session.run(
                "MATCH (g:Garment)-[r:PART_OF]->(o:Outfit {id: $id}) DELETE r",
                id=outfit.id
            )
            
            # Create PART_OF relationships
            for item in outfit.items:
                await session.run(
                    """
                    MATCH (g:Garment {id: $garment_id})
                    MATCH (o:Outfit {id: $outfit_id})
                    MERGE (g)-[:PART_OF {role: $role}]->(o)
                    """,
                    garment_id=item.garment.id,
                    outfit_id=outfit.id,
                    role=item.role
                )
        
        logger.debug(f"Saved outfit {outfit.id} to Neo4j")
        return outfit.id
    
    async def get_outfit(self, outfit_id: str) -> Optional[Outfit]:
        """Get an outfit from Neo4j by ID."""
        await self._ensure_connection()
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (o:Outfit {id: $id})
                OPTIONAL MATCH (g:Garment)-[r:PART_OF]->(o)
                RETURN o, collect({garment: g, role: r.role}) as items
                """,
                id=outfit_id
            )
            record = await result.single()
            if record:
                outfit_props = dict(record["o"])
                items = []
                for item_data in record["items"]:
                    if item_data["garment"]:
                        garment = self._node_to_garment(dict(item_data["garment"]))
                        items.append(OutfitItem(
                            garment=garment,
                            role=item_data["role"] or "item"
                        ))
                
                return Outfit(
                    id=outfit_props["id"],
                    items=items,
                    compatibility_score=outfit_props.get("compatibility_score", 0),
                    style_coherence_score=outfit_props.get("style_coherence_score", 0),
                    occasion_match_score=outfit_props.get("occasion_match_score", 0),
                    overall_score=outfit_props.get("overall_score", 0),
                    explanation=outfit_props.get("explanation"),
                    created_at=datetime.fromisoformat(outfit_props["created_at"])
                )
        return None
    
    async def get_all_outfits(self) -> List[Outfit]:
        """Get all outfits from Neo4j."""
        await self._ensure_connection()
        
        outfits = []
        async with self._driver.session(database=self.database) as session:
            result = await session.run("MATCH (o:Outfit) RETURN o.id as id")
            async for record in result:
                outfit = await self.get_outfit(record["id"])
                if outfit:
                    outfits.append(outfit)
        return outfits
    
    async def delete_outfit(self, outfit_id: str) -> bool:
        """Delete an outfit from Neo4j."""
        await self._ensure_connection()
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (o:Outfit {id: $id})
                DETACH DELETE o
                RETURN count(*) as deleted
                """,
                id=outfit_id
            )
            record = await result.single()
            deleted = record["deleted"] > 0 if record else False
        
        if deleted:
            logger.info(f"Deleted outfit {outfit_id} from Neo4j")
        return deleted
    
    # ==================== Compatibility Operations ====================
    
    async def save_compatibility(
        self,
        garment_id_1: str,
        garment_id_2: str,
        relation_type: RelationType,
        score: float,
        reasons: Optional[List[str]] = None
    ) -> None:
        """Save a compatibility relationship between two garments."""
        await self._ensure_connection()
        
        rel_type = relation_type.value
        
        async with self._driver.session(database=self.database) as session:
            # Create bidirectional relationship
            # Using MERGE to update existing or create new
            query = f"""
            MATCH (a:Garment {{id: $id1}})
            MATCH (b:Garment {{id: $id2}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.score = $score, r.reasons = $reasons, r.created_at = $created_at
            MERGE (b)-[r2:{rel_type}]->(a)
            SET r2.score = $score, r2.reasons = $reasons, r2.created_at = $created_at
            """
            await session.run(
                query,
                id1=garment_id_1,
                id2=garment_id_2,
                score=score,
                reasons=reasons or [],
                created_at=datetime.utcnow().isoformat()
            )
        
        logger.debug(f"Saved {rel_type} relationship: {garment_id_1} <-> {garment_id_2} (score: {score:.2f})")
    
    async def get_compatible_garments(
        self,
        garment_id: str,
        relation_type: Optional[RelationType] = None,
        min_score: float = 0.0
    ) -> List[Tuple[Garment, CompatibilityScore]]:
        """Get all garments compatible with a given garment."""
        await self._ensure_connection()
        
        # Build relationship type filter
        if relation_type:
            rel_filter = f":{relation_type.value}"
        else:
            rel_filter = ""
        
        results = []
        async with self._driver.session(database=self.database) as session:
            query = f"""
            MATCH (a:Garment {{id: $id}})-[r{rel_filter}]->(b:Garment)
            WHERE r.score >= $min_score
            RETURN b, type(r) as rel_type, r.score as score, r.reasons as reasons
            ORDER BY r.score DESC
            """
            result = await session.run(
                query,
                id=garment_id,
                min_score=min_score
            )
            
            async for record in result:
                garment = self._node_to_garment(dict(record["b"]))
                compat = CompatibilityScore(
                    source_id=garment_id,
                    target_id=garment.id,
                    relation_type=RelationType(record["rel_type"]),
                    score=record["score"],
                    reasons=record["reasons"] or []
                )
                results.append((garment, compat))
        
        return results
    
    async def find_outfit_paths(
        self,
        categories: List[GarmentCategory],
        relation_type: RelationType = RelationType.COMPATIBLE,
        min_score: float = 0.7
    ) -> List[List[Garment]]:
        """
        Find all outfit paths connecting garments from specified categories.
        
        This is where Neo4j SHINES! Complex graph queries like:
        "Find all paths connecting Top -> Bottom -> Shoes with COMPATIBLE relationships"
        """
        await self._ensure_connection()
        
        if len(categories) < 2:
            if categories:
                return [[g] for g in await self.get_garments_by_category(categories[0])]
            return []
        
        # Build dynamic Cypher query for path finding
        rel_type = relation_type.value
        
        # Build MATCH pattern: (a:Garment)-[:REL]->(b:Garment)-[:REL]->(c:Garment)...
        node_vars = [chr(ord('a') + i) for i in range(len(categories))]
        
        match_parts = []
        where_parts = []
        
        for i, (var, cat) in enumerate(zip(node_vars, categories)):
            match_parts.append(f"({var}:Garment)")
            where_parts.append(f"{var}.category = '{cat.value}'")
        
        # Build relationship pattern
        path_pattern = match_parts[0]
        for i in range(1, len(node_vars)):
            path_pattern += f"-[r{i}:{rel_type}]->" + match_parts[i]
        
        # Build score filter for all relationships
        score_filters = [f"r{i}.score >= {min_score}" for i in range(1, len(node_vars))]
        where_parts.extend(score_filters)
        
        query = f"""
        MATCH path = {path_pattern}
        WHERE {' AND '.join(where_parts)}
        RETURN {', '.join(node_vars)}
        LIMIT 100
        """
        
        paths = []
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query)
            async for record in result:
                path = []
                for var in node_vars:
                    garment = self._node_to_garment(dict(record[var]))
                    path.append(garment)
                paths.append(path)
        
        logger.info(f"Found {len(paths)} outfit paths for categories {[c.value for c in categories]}")
        return paths
    
    # ==================== Advanced Graph Queries ====================
    
    async def find_best_matches_for_garment(
        self,
        garment_id: str,
        target_category: GarmentCategory,
        limit: int = 5
    ) -> List[Tuple[Garment, float]]:
        """
        Find the best matching garments for a specific garment.
        
        Example: "Find the best pants to match this shirt"
        """
        await self._ensure_connection()
        
        results = []
        async with self._driver.session(database=self.database) as session:
            query = """
            MATCH (a:Garment {id: $id})-[r:COMPATIBLE|PAIRS_WELL|COLOR_HARMONY]->(b:Garment {category: $category})
            RETURN b, max(r.score) as best_score
            ORDER BY best_score DESC
            LIMIT $limit
            """
            result = await session.run(
                query,
                id=garment_id,
                category=target_category.value,
                limit=limit
            )
            
            async for record in result:
                garment = self._node_to_garment(dict(record["b"]))
                results.append((garment, record["best_score"]))
        
        return results
    
    async def get_outfit_graph_visualization(self) -> Dict[str, Any]:
        """
        Get data for visualizing the outfit compatibility graph.
        
        Returns nodes and edges for visualization tools like D3.js or Cytoscape.
        """
        await self._ensure_connection()
        
        nodes = []
        edges = []
        
        async with self._driver.session(database=self.database) as session:
            # Get all garments as nodes
            result = await session.run(
                "MATCH (g:Garment) RETURN g.id as id, g.category as category, g.color_primary as color"
            )
            async for record in result:
                nodes.append({
                    "id": record["id"],
                    "category": record["category"],
                    "color": record["color"]
                })
            
            # Get all relationships as edges
            result = await session.run(
                """
                MATCH (a:Garment)-[r]->(b:Garment)
                WHERE a.id < b.id
                RETURN a.id as source, b.id as target, type(r) as type, r.score as score
                """
            )
            async for record in result:
                edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["type"],
                    "score": record["score"]
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    
    # ==================== Statistics ====================
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics from Neo4j."""
        await self._ensure_connection()
        
        stats = {"storage_type": "neo4j", "uri": self.uri}
        
        async with self._driver.session(database=self.database) as session:
            # Count garments
            result = await session.run("MATCH (g:Garment) RETURN count(g) as count")
            record = await result.single()
            stats["total_garments"] = record["count"] if record else 0
            
            # Count outfits
            result = await session.run("MATCH (o:Outfit) RETURN count(o) as count")
            record = await result.single()
            stats["total_outfits"] = record["count"] if record else 0
            
            # Count relationships
            result = await session.run(
                """
                MATCH (a:Garment)-[r]->(b:Garment)
                WHERE a.id < b.id
                RETURN count(r) as count
                """
            )
            record = await result.single()
            stats["total_compatibility_relations"] = record["count"] if record else 0
            
            # Count by category
            result = await session.run(
                "MATCH (g:Garment) RETURN g.category as category, count(*) as count"
            )
            category_counts = {}
            async for record in result:
                category_counts[record["category"]] = record["count"]
            stats["garments_by_category"] = category_counts
        
        return stats


# ============== Factory Function ==============

def get_outfit_storage(
    storage_type: str = "auto",
    **kwargs
) -> OutfitStorageInterface:
    """
    Factory function to get the appropriate storage backend.
    
    Args:
        storage_type: "json", "neo4j", or "auto"
            - "json": Always use JSON file storage
            - "neo4j": Always use Neo4j (will fail if not available)
            - "auto": Use Neo4j if available, fallback to JSON
        **kwargs: Additional arguments for the storage backend
        
    Returns:
        OutfitStorageInterface implementation
        
    Examples:
        # For local testing
        storage = get_outfit_storage("json")
        
        # For production with Neo4j
        storage = get_outfit_storage("neo4j", uri="bolt://neo4j:7687")
        
        # Auto-detect
        storage = get_outfit_storage("auto")
    """
    if storage_type == "json":
        return JSONOutfitStorage(**kwargs)
    
    elif storage_type == "neo4j":
        return Neo4jOutfitStorage(**kwargs)
    
    elif storage_type == "auto":
        # Try Neo4j first, fallback to JSON
        try:
            import neo4j
            neo4j_uri = os.getenv("NEO4J_URI")
            if neo4j_uri:
                logger.info("Using Neo4j storage (NEO4J_URI found)")
                return Neo4jOutfitStorage(**kwargs)
        except ImportError:
            pass
        
        logger.info("Using JSON file storage (Neo4j not available)")
        return JSONOutfitStorage(**kwargs)
    
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
