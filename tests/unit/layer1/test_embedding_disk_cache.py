"""
Tests for Layer 1: EmbeddingGenerator — Solution 6 (Persistent disk cache)

Tests cover:
- Disk cache persists across EmbeddingGenerator instances
- In-memory cache is warmed from disk on __init__
- clear_cache() removes disk files
- generate_embedding uses disk cache on second call (zero extra API calls)
- generate_batch_embeddings also writes / reads from disk
"""
import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import uuid4

from src.core.models import (
    Garment, GarmentAttributes, GarmentCategory,
    ColorInfo, PatternInfo, FormalityLevel, Season,
)
from src.layer1_vision.embedding_generator import EmbeddingGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_EMBEDDING = [0.1, 0.2, 0.3]


def _make_garment(garment_id: str | None = None) -> Garment:
    attrs = GarmentAttributes(
        category=GarmentCategory.TOP,
        color=ColorInfo(primary="white", hex_codes=[]),
        formality_level=FormalityLevel.CASUAL,
    )
    return Garment(id=garment_id or f"g_{uuid4().hex[:6]}", attributes=attrs)


def _make_eg(cache_dir: Path) -> EmbeddingGenerator:
    """Build an EmbeddingGenerator with a patched Gemini client."""
    with patch.object(EmbeddingGenerator, "_init_gemini_client"):
        eg = EmbeddingGenerator.__new__(EmbeddingGenerator)
        # Manually call __init__ with patched internals
        eg._embed_cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        eg._cache = {}
        eg._load_disk_cache()
        return eg

def _make_full_eg(cache_dir: Path) -> EmbeddingGenerator:
    """Build a fully initialised EmbeddingGenerator that skips API init."""
    with (
        patch("src.layer1_vision.embedding_generator.get_settings") as mock_settings,
        patch.object(EmbeddingGenerator, "_init_gemini_client"),
    ):
        settings = MagicMock()
        settings.embedding_model = "gemini-embedding-001"
        settings.google_api_key = "fake-key"
        settings.openai_api_key = None
        mock_settings.return_value = settings
        eg = EmbeddingGenerator(embed_cache_dir=cache_dir)
        eg.provider = "gemini"
    return eg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embed_dir(tmp_path: Path) -> Path:
    return tmp_path / "embeddings_cache"


@pytest.fixture
def eg(embed_dir: Path) -> EmbeddingGenerator:
    return _make_full_eg(embed_dir)


@pytest.fixture
def garment() -> Garment:
    return _make_garment()


# ---------------------------------------------------------------------------
# Disk cache persists across instances
# ---------------------------------------------------------------------------

class TestDiskCachePersistence:
    def test_disk_file_created_after_generate(
        self, embed_dir: Path, garment: Garment
    ):
        eg = _make_full_eg(embed_dir)
        eg._generate_gemini_embedding = AsyncMock(return_value=MOCK_EMBEDDING)
        asyncio.run(eg.generate_embedding(garment))
        assert any(embed_dir.glob("*.json"))

    def test_disk_file_contains_embedding(
        self, embed_dir: Path, garment: Garment
    ):
        eg = _make_full_eg(embed_dir)
        eg._generate_gemini_embedding = AsyncMock(return_value=MOCK_EMBEDDING)
        asyncio.run(eg.generate_embedding(garment))
        json_files = list(embed_dir.glob("*.json"))
        data = json.loads(json_files[0].read_text())
        assert data == MOCK_EMBEDDING

    def test_second_instance_reads_from_disk_no_api_call(
        self, embed_dir: Path, garment: Garment
    ):
        # First instance: persist to disk
        eg1 = _make_full_eg(embed_dir)
        eg1._generate_gemini_embedding = AsyncMock(return_value=MOCK_EMBEDDING)
        asyncio.run(eg1.generate_embedding(garment))

        # Second instance: should load from disk, never call API
        eg2 = _make_full_eg(embed_dir)
        eg2._generate_gemini_embedding = AsyncMock(return_value=[9.9])
        result = asyncio.run(eg2.generate_embedding(garment))
        assert result == MOCK_EMBEDDING
        eg2._generate_gemini_embedding.assert_not_called()


# ---------------------------------------------------------------------------
# In-memory cache warmed from disk on init
# ---------------------------------------------------------------------------

class TestDiskWarmup:
    def test_cache_warmed_on_init(self, embed_dir: Path, garment: Garment):
        eg1 = _make_full_eg(embed_dir)
        eg1._generate_gemini_embedding = AsyncMock(return_value=MOCK_EMBEDDING)
        asyncio.run(eg1.generate_embedding(garment))

        eg2 = _make_full_eg(embed_dir)
        # Memory cache should already contain the key
        assert len(eg2._cache) == 1

    def test_warmed_cache_used_before_api(self, embed_dir: Path, garment: Garment):
        eg1 = _make_full_eg(embed_dir)
        eg1._generate_gemini_embedding = AsyncMock(return_value=MOCK_EMBEDDING)
        asyncio.run(eg1.generate_embedding(garment))

        eg2 = _make_full_eg(embed_dir)
        eg2._generate_gemini_embedding = AsyncMock(return_value=[0.0])
        result = asyncio.run(eg2.generate_embedding(garment))
        assert result == MOCK_EMBEDDING
        eg2._generate_gemini_embedding.assert_not_called()


# ---------------------------------------------------------------------------
# clear_cache()
# ---------------------------------------------------------------------------

class TestClearCache:
    def test_clear_removes_disk_files(self, embed_dir: Path, garment: Garment):
        eg = _make_full_eg(embed_dir)
        eg._generate_gemini_embedding = AsyncMock(return_value=MOCK_EMBEDDING)
        asyncio.run(eg.generate_embedding(garment))
        assert any(embed_dir.glob("*.json"))
        eg.clear_cache()
        assert not any(embed_dir.glob("*.json"))

    def test_clear_returns_count(self, embed_dir: Path):
        eg = _make_full_eg(embed_dir)
        # Manually inject 3 disk files
        for i in range(3):
            (embed_dir / f"abc{i}.json").write_text(json.dumps([float(i)]))
        eg._cache = {"abc0": [0.0], "abc1": [1.0], "abc2": [2.0]}
        removed = eg.clear_cache()
        assert removed == 3

    def test_clear_empties_in_memory_cache(self, embed_dir: Path, garment: Garment):
        eg = _make_full_eg(embed_dir)
        eg._generate_gemini_embedding = AsyncMock(return_value=MOCK_EMBEDDING)
        asyncio.run(eg.generate_embedding(garment))
        eg.clear_cache()
        assert eg._cache == {}

    def test_clear_on_empty_cache_returns_zero(self, embed_dir: Path):
        eg = _make_full_eg(embed_dir)
        assert eg.clear_cache() == 0


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

class TestDiskCacheHelpers:
    def test_disk_cache_path_uses_key_as_stem(self, embed_dir: Path):
        eg = _make_full_eg(embed_dir)
        path = eg._disk_cache_path("abc123")
        assert path == embed_dir / "abc123.json"

    def test_load_single_from_disk_returns_none_on_miss(self, embed_dir: Path):
        eg = _make_full_eg(embed_dir)
        assert eg._load_single_from_disk("nonexistent") is None

    def test_save_then_load_single(self, embed_dir: Path):
        eg = _make_full_eg(embed_dir)
        eg._save_embedding_to_disk("mykey", MOCK_EMBEDDING)
        loaded = eg._load_single_from_disk("mykey")
        assert loaded == MOCK_EMBEDDING

    def test_load_single_returns_none_for_corrupt_file(self, embed_dir: Path):
        eg = _make_full_eg(embed_dir)
        (embed_dir / "badkey.json").write_text("NOT JSON {{{")
        assert eg._load_single_from_disk("badkey") is None


# ---------------------------------------------------------------------------
# Batch embeddings also use disk cache
# ---------------------------------------------------------------------------

class TestBatchEmbeddingDiskCache:
    def test_batch_writes_disk_files(self, embed_dir: Path):
        eg = _make_full_eg(embed_dir)
        eg._generate_gemini_batch_embeddings = AsyncMock(
            return_value=[MOCK_EMBEDDING, [0.4, 0.5, 0.6]]
        )
        # Use garments with different attributes so they get different cache keys
        g1 = _make_garment()
        g2 = Garment(
            id=f"g_{uuid4().hex[:6]}",
            attributes=GarmentAttributes(
                category=GarmentCategory.BOTTOM,
                color=ColorInfo(primary="blue", hex_codes=[]),
                formality_level=FormalityLevel.CASUAL,
            ),
        )
        asyncio.run(eg.generate_batch_embeddings([g1, g2]))
        assert len(list(embed_dir.glob("*.json"))) == 2

    def test_batch_reads_disk_on_second_call(self, embed_dir: Path):
        eg = _make_full_eg(embed_dir)
        g = _make_garment()
        eg._generate_gemini_batch_embeddings = AsyncMock(return_value=[MOCK_EMBEDDING])
        asyncio.run(eg.generate_batch_embeddings([g]))

        # Second instance — should load from disk
        eg2 = _make_full_eg(embed_dir)
        eg2._generate_gemini_batch_embeddings = AsyncMock(return_value=[[9.9]])
        result = asyncio.run(eg2.generate_batch_embeddings([g]))
        assert result[0] == MOCK_EMBEDDING
        eg2._generate_gemini_batch_embeddings.assert_not_called()


# ---------------------------------------------------------------------------
# use_cache=False bypasses disk
# ---------------------------------------------------------------------------

class TestUseCacheFalse:
    def test_use_cache_false_always_calls_api(self, embed_dir: Path, garment: Garment):
        eg = _make_full_eg(embed_dir)
        eg._generate_gemini_embedding = AsyncMock(return_value=MOCK_EMBEDDING)
        asyncio.run(eg.generate_embedding(garment, use_cache=False))
        asyncio.run(eg.generate_embedding(garment, use_cache=False))
        assert eg._generate_gemini_embedding.call_count == 2

    def test_use_cache_false_does_not_write_disk(self, embed_dir: Path, garment: Garment):
        eg = _make_full_eg(embed_dir)
        eg._generate_gemini_embedding = AsyncMock(return_value=MOCK_EMBEDDING)
        asyncio.run(eg.generate_embedding(garment, use_cache=False))
        assert not any(embed_dir.glob("*.json"))
