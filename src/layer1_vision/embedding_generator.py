"""
Embedding Generator
Generates vector embeddings for clothing items for similarity matching.

Supports both Google Gemini and OpenAI embedding models:
- Gemini: text-embedding-004 (free with Google API key)
- OpenAI: text-embedding-3-small, text-embedding-3-large

Solution 6 — Pre-computed persistent embedding cache:
    Embeddings are stored on disk (data/processed/embeddings_cache/) keyed
    by an MD5 hash of the garment's serialised attributes.  On subsequent
    calls the vector is read from disk instantly — zero API calls.
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib
import json

from config import get_settings
from src.core.models import Garment, GarmentAttributes
from src.core.exceptions import EmbeddingError
from src.core import get_logger

logger = get_logger(__name__)

_DEFAULT_EMBED_CACHE_DIR = Path("data/processed/embeddings_cache")


class EmbeddingGenerator:
    """
    Generates embeddings for clothing items.
    
    Supports multiple embedding providers:
    - Google Gemini (text-embedding-004) - default, uses GOOGLE_API_KEY
    - OpenAI (text-embedding-3-small/large) - requires OPENAI_API_KEY
    
    These embeddings are used for:
    - Similarity search
    - Style matching
    - Outfit compatibility
    """
    
    # Gemini embedding models
    GEMINI_MODELS = {"gemini-embedding-001", "text-embedding-004", "embedding-001"}
    # OpenAI embedding models
    OPENAI_MODELS = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}
    
    def __init__(
        self,
        provider: Optional[str] = None,
        embed_cache_dir: Path = _DEFAULT_EMBED_CACHE_DIR,
    ):
        """
        Initialize the embedding generator.

        Args:
            provider: Force a specific provider ('gemini' or 'openai').
                     If None, auto-detects based on embedding_model setting.
            embed_cache_dir: Directory for persistent on-disk embedding cache
                            (Solution 6).  Created automatically if missing.
        """
        self.settings = get_settings()
        self.model = self.settings.embedding_model
        # In-memory LRU-style cache (fast path)
        self._cache: Dict[str, List[float]] = {}
        # Solution 6 — persistent on-disk embedding cache
        self._embed_cache_dir = Path(embed_cache_dir)
        self._embed_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine provider
        if provider:
            self.provider = provider.lower()
        elif self.model in self.GEMINI_MODELS:
            self.provider = "gemini"
        elif self.model in self.OPENAI_MODELS:
            self.provider = "openai"
        else:
            # Default to Gemini if Google API key is available
            if self.settings.google_api_key:
                self.provider = "gemini"
                self.model = "gemini-embedding-001"
            else:
                self.provider = "openai"
        
        # Normalize Gemini model name (always use full format)
        if self.provider == "gemini" and not self.model.startswith("gemini-"):
            self.model = "gemini-embedding-001"
        
        # Initialize the appropriate client
        if self.provider == "gemini":
            self._init_gemini_client()
        else:
            self._init_openai_client()
        
        # Warm in-memory cache from disk on startup (Solution 6)
        self._load_disk_cache()
        logger.info(
            f"EmbeddingGenerator initialized with {self.provider} ({self.model}), "
            f"disk cache: {self._embed_cache_dir} ({len(self._cache)} entries warmed)"
        )
    
    def _init_gemini_client(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.settings.google_api_key)
            self._genai = genai
        except ImportError:
            raise EmbeddingError("google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Gemini client: {e}")
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI
            if not self.settings.openai_api_key:
                raise EmbeddingError("OpenAI API key not configured. Set OPENAI_API_KEY in .env")
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        except ImportError:
            raise EmbeddingError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI client: {e}")
    
    async def generate_embedding(
        self,
        garment: Garment,
        use_cache: bool = True
    ) -> List[float]:
        """
        Generate embedding for a single garment.
        
        Args:
            garment: The garment to generate embedding for
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of floats representing the embedding vector
        """
        # Create cache key
        cache_key = self._get_cache_key(garment)

        if use_cache and cache_key in self._cache:
            logger.debug(f"Using in-memory cached embedding for garment {garment.id}")
            return self._cache[cache_key]

        # Solution 6 — check disk before calling the API
        if use_cache:
            disk_val = self._load_single_from_disk(cache_key)
            if disk_val is not None:
                self._cache[cache_key] = disk_val
                logger.debug(f"Loaded embedding from disk cache for garment {garment.id}")
                return disk_val

        try:
            # Generate text description for embedding
            text = self._garment_to_text(garment)

            # Generate embedding based on provider
            if self.provider == "gemini":
                embedding = await self._generate_gemini_embedding(text)
            else:
                embedding = await self._generate_openai_embedding(text)

            # Persist to in-memory + disk cache
            if use_cache:
                self._cache[cache_key] = embedding
                self._save_embedding_to_disk(cache_key, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
    
    async def _generate_gemini_embedding(self, text: str) -> List[float]:
        """Generate embedding using Google Gemini."""
        import asyncio
        
        # Gemini's embed_content is synchronous, run in executor
        def _embed():
            result = self._genai.embed_content(
                model=f"models/{self.model}",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _embed)
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    async def generate_batch_embeddings(
        self,
        garments: List[Garment],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple garments efficiently.
        
        Args:
            garments: List of garments to generate embeddings for
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of embedding vectors
        """
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check in-memory then disk cache first
        for i, garment in enumerate(garments):
            cache_key = self._get_cache_key(garment)
            if use_cache and cache_key in self._cache:
                results.append(self._cache[cache_key])
            elif use_cache:
                disk_val = self._load_single_from_disk(cache_key)
                if disk_val is not None:
                    self._cache[cache_key] = disk_val
                    results.append(disk_val)
                else:
                    texts_to_embed.append(self._garment_to_text(garment))
                    indices_to_embed.append(i)
                    results.append(None)  # Placeholder
            else:
                texts_to_embed.append(self._garment_to_text(garment))
                indices_to_embed.append(i)
                results.append(None)  # Placeholder
        
        # Generate embeddings for non-cached items
        if texts_to_embed:
            try:
                if self.provider == "gemini":
                    embeddings = await self._generate_gemini_batch_embeddings(texts_to_embed)
                else:
                    embeddings = await self._generate_openai_batch_embeddings(texts_to_embed)
                
                for j, embedding in enumerate(embeddings):
                    original_index = indices_to_embed[j]
                    results[original_index] = embedding
                    
                    # Cache the result (memory + disk)
                    if use_cache:
                        cache_key = self._get_cache_key(garments[original_index])
                        self._cache[cache_key] = embedding
                        self._save_embedding_to_disk(cache_key, embedding)
                        
            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                raise EmbeddingError(f"Failed to generate batch embeddings: {str(e)}")
        
        return results
    
    async def _generate_gemini_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using Google Gemini."""
        import asyncio
        
        def _embed_batch():
            # Gemini supports batch embedding
            result = self._genai.embed_content(
                model=f"models/{self.model}",
                content=texts,
                task_type="retrieval_document"
            )
            return result['embedding']
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _embed_batch)
    
    async def _generate_openai_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using OpenAI."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        return [data.embedding for data in response.data]
    
    async def generate_style_embedding(self, style_description: str) -> List[float]:
        """
        Generate embedding for a style description.
        
        Useful for searching garments by style concepts.
        
        Args:
            style_description: Natural language style description
            
        Returns:
            Embedding vector for the style
        """
        try:
            if self.provider == "gemini":
                return await self._generate_gemini_embedding(style_description)
            else:
                return await self._generate_openai_embedding(style_description)
            
        except Exception as e:
            logger.error(f"Style embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate style embedding: {str(e)}")
    
    def _garment_to_text(self, garment: Garment) -> str:
        """
        Convert garment attributes to a rich text description for embedding.
        
        Args:
            garment: The garment to convert
            
        Returns:
            Text representation of the garment
        """
        attrs = garment.attributes
        
        parts = [
            f"Category: {attrs.category.value}",
            f"Subcategory: {attrs.subcategory}" if attrs.subcategory else "",
            f"Primary color: {attrs.color.primary}",
            f"Secondary color: {attrs.color.secondary}" if attrs.color.secondary else "",
            f"Pattern: {attrs.pattern.type}",
            f"Formality: {attrs.formality_level.value}",
            f"Silhouette: {attrs.silhouette}" if attrs.silhouette else "",
            f"Fit: {attrs.fit}" if attrs.fit else "",
            f"Style tags: {', '.join(attrs.style_tags)}" if attrs.style_tags else "",
            f"Seasons: {', '.join(s.value for s in attrs.season_suitable)}" if attrs.season_suitable else "",
            # New extended fields
            f"Neckline: {attrs.neckline.value}" if attrs.neckline else "",
            f"Sleeve length: {attrs.sleeves.length}" if attrs.sleeves else "",
            f"Sleeve style: {attrs.sleeves.style}" if attrs.sleeves and attrs.sleeves.style else "",
            f"Length: {attrs.length_type.value}" if attrs.length_type else "",
            f"Waist rise: {attrs.waist_rise.value}" if attrs.waist_rise else "",
            f"Closure: {attrs.closure.value}" if attrs.closure else "",
            f"Has pockets: {'yes' if attrs.details.has_pockets else 'no'}",
            f"Distressed: {'yes' if attrs.details.distressed else 'no'}",
            f"Embellishments: {', '.join(attrs.details.embellishments)}" if attrs.details.embellishments else "",
            f"Transparency: {attrs.details.transparency.value}" if attrs.details.transparency else "",
        ]
        
        return " | ".join(filter(None, parts))
    
    def _get_cache_key(self, garment: Garment) -> str:
        """Generate a unique cache key for a garment."""
        # Create hash from garment attributes
        attrs_dict = garment.attributes.model_dump()
        attrs_str = json.dumps(attrs_dict, sort_keys=True)
        return hashlib.md5(attrs_str.encode()).hexdigest()
    
    def clear_cache(self) -> int:
        """Clear both the in-memory and on-disk embedding caches.

        Returns:
            Number of disk entries removed.
        """
        self._cache.clear()
        removed = 0
        cache_dir = getattr(self, "_embed_cache_dir", None)
        if cache_dir is not None:
            for p in cache_dir.glob("*.json"):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        logger.info(f"Embedding cache cleared ({removed} disk entries removed)")
        return removed

    # ------------------------------------------------------------------
    # Solution 6 — Persistent disk-cache helpers
    # ------------------------------------------------------------------

    def _disk_cache_path(self, key: str) -> Path:
        """Return the JSON file path for a given cache key."""
        return self._embed_cache_dir / f"{key}.json"

    def _load_disk_cache(self) -> None:
        """Warm the in-memory cache from all persisted JSON files on startup."""
        loaded = 0
        for p in self._embed_cache_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                key = p.stem
                if isinstance(data, list):
                    self._cache[key] = data
                    loaded += 1
            except Exception as exc:
                logger.warning(f"Could not load disk embedding {p.name}: {exc}")
        if loaded:
            logger.debug(f"Warmed embedding cache with {loaded} entries from disk")

    def _save_embedding_to_disk(self, key: str, embedding: List[float]) -> None:
        """Persist a single embedding vector to disk as JSON."""
        try:
            self._disk_cache_path(key).write_text(json.dumps(embedding))
        except OSError as exc:
            logger.warning(f"Could not persist embedding to disk: {exc}")

    def _load_single_from_disk(self, key: str) -> Optional[List[float]]:
        """Load a single embedding from disk; returns None on miss / error."""
        path = self._disk_cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, list) else None
        except Exception as exc:
            logger.warning(f"Could not read disk embedding {path.name}: {exc}")
            return None


    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
