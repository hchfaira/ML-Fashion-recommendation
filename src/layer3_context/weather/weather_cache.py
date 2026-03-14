"""
WeatherCache
============
Two-level caching for weather data:

  1. **In-memory** LRU dict (fast, process-lifetime)
  2. **File-system** JSON cache (survives restarts, configurable TTL)

Cache key: ``"{lat:.4f},{lon:.4f}"`` (4 decimal places ≈ 11 m precision).

TTL semantics
-------------
- Fresh    (age < TTL)         → use without hitting API
- Stale    (TTL ≤ age < stale_TTL) → use but trigger background refresh
- Expired  (age ≥ stale_TTL)  → do **not** use; treat as cache miss
                                  (but keep file for emergency fallback)
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .weather_models import WeatherData
from src.core import get_logger

logger = get_logger(__name__)

_DEFAULT_TTL = 3600          # 1 hour
_DEFAULT_STALE_TTL = 86_400  # 24 hours
_DEFAULT_MAX = 100
_DEFAULT_DIR = ".weather_cache"


class WeatherCache:
    """
    Two-level weather cache.

    Parameters
    ----------
    ttl_seconds : int
        How long (seconds) a cached entry is considered *fresh*.
    stale_ttl_seconds : int
        How long after TTL an entry can still be used as a stale fallback.
    max_entries : int
        Maximum in-memory entries (LRU eviction).
    cache_dir : str | Path | None
        Directory for file-system JSON cache.  Pass ``None`` to disable.
    """

    def __init__(
        self,
        ttl_seconds: int = _DEFAULT_TTL,
        stale_ttl_seconds: int = _DEFAULT_STALE_TTL,
        max_entries: int = _DEFAULT_MAX,
        cache_dir: str | Path | None = _DEFAULT_DIR,
    ) -> None:
        self._ttl = ttl_seconds
        self._stale_ttl = stale_ttl_seconds
        self._max = max_entries
        self._mem: OrderedDict[str, WeatherData] = OrderedDict()

        if cache_dir is not None:
            self._dir: Path | None = Path(cache_dir)
            self._dir.mkdir(parents=True, exist_ok=True)
        else:
            self._dir = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, lat: float, lon: float) -> Optional[WeatherData]:
        """
        Return cached ``WeatherData`` for the given coordinates, or ``None``.

        A *stale* entry is returned with ``provider`` set to ``"cache-stale"``
        so callers can decide whether to refresh in the background.
        """
        key = self._key(lat, lon)

        # 1. memory hit
        if key in self._mem:
            data = self._mem[key]
            age = self._age(data)
            if age < self._ttl:
                self._mem.move_to_end(key)
                logger.debug("cache HIT (fresh) %s age=%ds", key, int(age))
                return data
            if age < self._stale_ttl:
                logger.debug("cache HIT (stale) %s age=%ds", key, int(age))
                return self._mark_stale(data)
            # expired in memory — evict
            del self._mem[key]

        # 2. file-system hit
        if self._dir is not None:
            fs_data = self._fs_get(key)
            if fs_data is not None:
                age = self._age(fs_data)
                if age < self._stale_ttl:
                    self._mem_put(key, fs_data)
                    if age < self._ttl:
                        return fs_data
                    return self._mark_stale(fs_data)
                # fully expired on disk — leave file but return None
                logger.debug("cache MISS (expired) %s age=%ds", key, int(age))

        return None

    def put(self, lat: float, lon: float, data: WeatherData) -> None:
        """Store a freshly-fetched ``WeatherData`` entry."""
        key = self._key(lat, lon)
        self._mem_put(key, data)
        if self._dir is not None:
            self._fs_put(key, data)

    def invalidate(self, lat: float, lon: float) -> None:
        """Remove a specific cache entry."""
        key = self._key(lat, lon)
        self._mem.pop(key, None)
        if self._dir is not None:
            self._fp(key).unlink(missing_ok=True)

    def clear(self) -> None:
        """Clear all in-memory entries (does NOT remove files)."""
        self._mem.clear()

    def clear_all(self) -> None:
        """Clear memory **and** delete all cache files."""
        self._mem.clear()
        if self._dir is not None:
            for fp in self._dir.glob("*.json"):
                fp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(lat: float, lon: float) -> str:
        return f"{lat:.4f},{lon:.4f}"

    @staticmethod
    def _age(data: WeatherData) -> float:
        """Seconds since the entry was fetched."""
        from datetime import timezone
        return time.time() - data.fetched_at.astimezone(timezone.utc).timestamp()

    @staticmethod
    def _mark_stale(data: WeatherData) -> WeatherData:
        return data.model_copy(update={"provider": "cache-stale"})

    def _mem_put(self, key: str, data: WeatherData) -> None:
        if key in self._mem:
            self._mem.move_to_end(key)
        self._mem[key] = data
        if len(self._mem) > self._max:
            self._mem.popitem(last=False)   # evict LRU

    # ---- File I/O --------------------------------------------------------

    def _fp(self, key: str) -> Path:
        safe = key.replace(",", "_").replace(".", "p")
        return self._dir / f"{safe}.json"  # type: ignore[operator]

    def _fs_put(self, key: str, data: WeatherData) -> None:
        try:
            self._fp(key).write_text(
                json.dumps(data.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("weather cache write failed: %s", exc)

    def _fs_get(self, key: str) -> Optional[WeatherData]:
        fp = self._fp(key)
        if not fp.exists():
            return None
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
            # Re-hydrate condition enum
            raw["condition"] = raw.get("condition", "unknown")
            return WeatherData.model_validate(raw)
        except Exception as exc:
            logger.warning("weather cache read failed for %s: %s", fp, exc)
            return None
