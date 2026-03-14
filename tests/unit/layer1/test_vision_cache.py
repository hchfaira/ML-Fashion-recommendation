"""
Tests for Layer 1: VisionCache — Solution 1 (Persistent SHA-256 cache)

Tests cover:
- Cache miss on cold start
- Cache set + get round-trip
- Hash changes when file contents change
- Invalidation and full clear
- Expiry (max_age_days)
- stats() hit-rate reporting
"""
import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.layer1_vision.vision_cache import VisionCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "vision_cache"


@pytest.fixture
def cache(cache_dir: Path) -> VisionCache:
    return VisionCache(cache_dir=cache_dir)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a small fake PNG file."""
    img = tmp_path / "shirt.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return img


@pytest.fixture
def sample_attrs() -> dict:
    return {
        "category": "TOP",
        "color": {"primary": "white"},
        "style_tags": ["casual"],
    }


# ---------------------------------------------------------------------------
# Basic cache operations
# ---------------------------------------------------------------------------

class TestCacheMiss:
    def test_miss_on_new_image(self, cache: VisionCache, sample_image: Path):
        assert cache.get(sample_image) is None

    def test_miss_on_nonexistent_path(self, cache: VisionCache, tmp_path: Path):
        assert cache.get(tmp_path / "ghost.png") is None


class TestCacheSetGet:
    def test_set_then_get_returns_same_dict(
        self, cache: VisionCache, sample_image: Path, sample_attrs: dict
    ):
        cache.set(sample_image, sample_attrs)
        result = cache.get(sample_image)
        assert result == sample_attrs

    def test_json_file_created_on_set(
        self, cache: VisionCache, cache_dir: Path, sample_image: Path, sample_attrs: dict
    ):
        cache.set(sample_image, sample_attrs)
        json_files = list(cache_dir.glob("*.json"))
        assert len(json_files) == 1

    def test_json_file_contains_attrs(
        self, cache: VisionCache, cache_dir: Path, sample_image: Path, sample_attrs: dict
    ):
        cache.set(sample_image, sample_attrs)
        json_files = list(cache_dir.glob("*.json"))
        stored = json.loads(json_files[0].read_text())
        assert stored.get("attributes") == sample_attrs


class TestCacheHashSensitivity:
    def test_different_content_gives_different_entry(
        self, cache: VisionCache, tmp_path: Path, sample_attrs: dict
    ):
        img_a = tmp_path / "a.png"
        img_b = tmp_path / "b.png"
        img_a.write_bytes(b"AAAA")
        img_b.write_bytes(b"BBBB")

        cache.set(img_a, sample_attrs)
        cache.set(img_b, {**sample_attrs, "category": "BOTTOM"})

        assert cache.get(img_a)["category"] == "TOP"
        assert cache.get(img_b)["category"] == "BOTTOM"

    def test_modified_file_misses_cache(
        self, cache: VisionCache, sample_image: Path, sample_attrs: dict
    ):
        cache.set(sample_image, sample_attrs)
        # Overwrite bytes → different hash
        sample_image.write_bytes(b"COMPLETELY DIFFERENT CONTENT")
        assert cache.get(sample_image) is None


class TestCacheInvalidate:
    def test_invalidate_removes_entry(
        self, cache: VisionCache, sample_image: Path, sample_attrs: dict
    ):
        cache.set(sample_image, sample_attrs)
        cache.invalidate(sample_image)
        assert cache.get(sample_image) is None

    def test_invalidate_nonexistent_no_error(
        self, cache: VisionCache, sample_image: Path
    ):
        cache.invalidate(sample_image)  # should not raise

    def test_invalidate_removes_json_file(
        self, cache: VisionCache, cache_dir: Path, sample_image: Path, sample_attrs: dict
    ):
        cache.set(sample_image, sample_attrs)
        cache.invalidate(sample_image)
        assert list(cache_dir.glob("*.json")) == []


class TestCacheClear:
    def test_clear_removes_all_entries(
        self, cache: VisionCache, tmp_path: Path, sample_attrs: dict
    ):
        for i in range(3):
            img = tmp_path / f"item_{i}.png"
            img.write_bytes(bytes([i] * 32))
            cache.set(img, sample_attrs)

        removed = cache.clear()
        assert removed == 3

    def test_clear_returns_count(
        self, cache: VisionCache, tmp_path: Path, sample_attrs: dict
    ):
        img = tmp_path / "x.png"
        img.write_bytes(b"X")
        cache.set(img, sample_attrs)
        assert cache.clear() == 1

    def test_clear_empties_directory(
        self, cache: VisionCache, cache_dir: Path, tmp_path: Path, sample_attrs: dict
    ):
        img = tmp_path / "x.png"
        img.write_bytes(b"X")
        cache.set(img, sample_attrs)
        cache.clear()
        assert list(cache_dir.glob("*.json")) == []


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

class TestCacheExpiry:
    def test_fresh_entry_not_expired(
        self, cache_dir: Path, sample_image: Path, sample_attrs: dict
    ):
        cache = VisionCache(cache_dir=cache_dir, max_age_days=1)
        cache.set(sample_image, sample_attrs)
        assert cache.get(sample_image) is not None

    def test_old_entry_expired(
        self, cache_dir: Path, sample_image: Path, sample_attrs: dict
    ):
        cache = VisionCache(cache_dir=cache_dir, max_age_days=1)
        cache.set(sample_image, sample_attrs)

        # Patch: move the stored timestamp back by more than 1 day
        json_files = list(cache_dir.glob("*.json"))
        data = json.loads(json_files[0].read_text())
        data["cached_at"] = time.time() - (2 * 86_400)  # 2 days ago
        json_files[0].write_text(json.dumps(data))

        assert cache.get(sample_image) is None

    def test_no_max_age_never_expires(
        self, cache_dir: Path, sample_image: Path, sample_attrs: dict
    ):
        cache = VisionCache(cache_dir=cache_dir, max_age_days=None)
        cache.set(sample_image, sample_attrs)

        json_files = list(cache_dir.glob("*.json"))
        data = json.loads(json_files[0].read_text())
        data["cached_at"] = 0  # epoch — very old
        json_files[0].write_text(json.dumps(data))

        assert cache.get(sample_image) is not None


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_initial_stats_zero(self, cache: VisionCache):
        s = cache.stats()
        assert s["hits"] == 0
        assert s["misses"] == 0

    def test_hit_increments_counter(
        self, cache: VisionCache, sample_image: Path, sample_attrs: dict
    ):
        cache.set(sample_image, sample_attrs)
        cache.get(sample_image)
        assert cache.stats()["hits"] == 1

    def test_miss_increments_counter(
        self, cache: VisionCache, sample_image: Path
    ):
        cache.get(sample_image)
        assert cache.stats()["misses"] == 1

    def test_hit_rate_calculation(
        self, cache: VisionCache, sample_image: Path, sample_attrs: dict
    ):
        cache.set(sample_image, sample_attrs)
        cache.get(sample_image)   # hit
        cache.get(sample_image)   # hit
        cache.get(sample_image)   # hit
        # cold miss on a non-existent path — use a different Path object
        from pathlib import Path as _P
        cache.get(_P("/nonexistent/ghost.png"))  # miss
        s = cache.stats()
        assert s["hit_rate"] == pytest.approx(0.75)

    def test_stats_includes_cache_size(
        self, cache: VisionCache, tmp_path: Path, sample_attrs: dict
    ):
        for i in range(2):
            img = tmp_path / f"item_{i}.png"
            img.write_bytes(bytes([i] * 32))
            cache.set(img, sample_attrs)
        assert cache.stats()["cache_size"] == 2
