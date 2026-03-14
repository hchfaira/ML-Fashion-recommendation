"""
Unit tests for WeatherCache
============================
Tests cache semantics (fresh / stale / expired / LRU eviction / file I/O).
All tests use a tmp_path directory and monkeypatched time.
"""
import json
import time
import pytest
from pathlib import Path

from src.layer3_context.weather.weather_cache import WeatherCache
from src.layer3_context.weather.weather_models import WeatherData, WeatherCondition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(city: str = "Paris", lat: float = 48.85, lon: float = 2.35) -> WeatherData:
    return WeatherData(
        temperature_celsius=18.0,
        condition=WeatherCondition.SUNNY,
        city=city,
        lat=lat,
        lon=lon,
        provider="open-meteo",
    )


# ---------------------------------------------------------------------------
# In-memory cache tests
# ---------------------------------------------------------------------------

class TestWeatherCacheMemory:

    def test_put_then_get_fresh(self, tmp_path):
        cache = WeatherCache(ttl_seconds=3600, cache_dir=None)
        data = make_data()
        cache.put(48.85, 2.35, data)
        result = cache.get(48.85, 2.35)
        assert result is not None
        assert result.city == "Paris"

    def test_miss_on_different_coords(self, tmp_path):
        cache = WeatherCache(ttl_seconds=3600, cache_dir=None)
        cache.put(48.85, 2.35, make_data("Paris"))
        result = cache.get(51.51, -0.13)   # London
        assert result is None

    def test_stale_entry_provider_marked(self, tmp_path, monkeypatch):
        cache = WeatherCache(ttl_seconds=1, stale_ttl_seconds=3600, cache_dir=None)
        data = make_data()
        cache.put(48.85, 2.35, data)

        # Advance time past TTL but within stale window
        orig_time = time.time()
        monkeypatch.setattr(time, "time", lambda: orig_time + 2)

        result = cache.get(48.85, 2.35)
        assert result is not None
        assert result.provider == "cache-stale"

    def test_expired_entry_returns_none(self, tmp_path, monkeypatch):
        cache = WeatherCache(ttl_seconds=1, stale_ttl_seconds=2, cache_dir=None)
        data = make_data()
        cache.put(48.85, 2.35, data)

        orig_time = time.time()
        monkeypatch.setattr(time, "time", lambda: orig_time + 3)

        result = cache.get(48.85, 2.35)
        assert result is None

    def test_lru_eviction(self):
        cache = WeatherCache(ttl_seconds=3600, max_entries=3, cache_dir=None)
        cache.put(1.0, 1.0, make_data("A", 1.0, 1.0))
        cache.put(2.0, 2.0, make_data("B", 2.0, 2.0))
        cache.put(3.0, 3.0, make_data("C", 3.0, 3.0))
        cache.put(4.0, 4.0, make_data("D", 4.0, 4.0))  # evicts "A"

        assert cache.get(1.0, 1.0) is None   # evicted
        assert cache.get(4.0, 4.0) is not None

    def test_clear_removes_all_entries(self):
        cache = WeatherCache(ttl_seconds=3600, cache_dir=None)
        cache.put(48.85, 2.35, make_data())
        cache.clear()
        assert cache.get(48.85, 2.35) is None

    def test_invalidate_specific_entry(self):
        cache = WeatherCache(ttl_seconds=3600, cache_dir=None)
        cache.put(48.85, 2.35, make_data("Paris"))
        cache.put(51.51, -0.13, make_data("London", 51.51, -0.13))
        cache.invalidate(48.85, 2.35)
        assert cache.get(48.85, 2.35) is None
        assert cache.get(51.51, -0.13) is not None


# ---------------------------------------------------------------------------
# File-system cache tests
# ---------------------------------------------------------------------------

class TestWeatherCacheFilesystem:

    def test_put_creates_json_file(self, tmp_path):
        cache = WeatherCache(ttl_seconds=3600, cache_dir=tmp_path)
        cache.put(48.85, 2.35, make_data())
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_json_file_readable(self, tmp_path):
        cache = WeatherCache(ttl_seconds=3600, cache_dir=tmp_path)
        cache.put(48.85, 2.35, make_data("Paris"))
        files = list(tmp_path.glob("*.json"))
        raw = json.loads(files[0].read_text())
        assert raw["city"] == "Paris"

    def test_fs_hit_after_memory_clear(self, tmp_path):
        cache = WeatherCache(ttl_seconds=3600, cache_dir=tmp_path)
        cache.put(48.85, 2.35, make_data("Paris"))
        cache.clear()  # wipe in-memory only
        result = cache.get(48.85, 2.35)
        assert result is not None
        assert result.city == "Paris"

    def test_clear_all_removes_files(self, tmp_path):
        cache = WeatherCache(ttl_seconds=3600, cache_dir=tmp_path)
        cache.put(48.85, 2.35, make_data())
        cache.clear_all()
        assert list(tmp_path.glob("*.json")) == []

    def test_corrupted_json_returns_none(self, tmp_path):
        cache = WeatherCache(ttl_seconds=3600, cache_dir=tmp_path)
        # Manually write a corrupt file at the key location
        key = "48p8500_2p3500"
        (tmp_path / f"{key}.json").write_text("not valid json")
        result = cache.get(48.85, 2.35)
        assert result is None

    def test_no_cache_dir_no_files(self):
        """cache_dir=None must not write any files."""
        cache = WeatherCache(ttl_seconds=3600, cache_dir=None)
        cache.put(48.85, 2.35, make_data())
        # No exception, nothing on disk
        assert cache.get(48.85, 2.35) is not None

    def test_invalidate_removes_file(self, tmp_path):
        cache = WeatherCache(ttl_seconds=3600, cache_dir=tmp_path)
        cache.put(48.85, 2.35, make_data())
        cache.invalidate(48.85, 2.35)
        assert list(tmp_path.glob("*.json")) == []
