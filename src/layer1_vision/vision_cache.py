"""
Vision Cache  (Solution 1 — Persistent Cache)
=============================================

Persistent, SHA-256-keyed cache for garment attribute extraction results.

A garment image is **analysed only once in its lifetime**.  On subsequent
calls the cached JSON attributes are returned instantly — zero API calls.

Cache location  : ``data/processed/vision_cache/``  (configurable)
Cache format    : One JSON file per image, named ``<sha256>.json``
Key             : SHA-256 hash of the raw image bytes
Invalidation    : Automatic when the image file changes (new hash)

Usage
-----
    cache = VisionCache()

    # Check before calling Gemini
    cached = cache.get(image_path)
    if cached is None:
        result = await gemini_extract(image_path)
        cache.set(image_path, result)
    else:
        result = cached
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.core import get_logger

logger = get_logger(__name__)

# Default cache directory  (relative to project root)
_DEFAULT_CACHE_DIR = Path("data/processed/vision_cache")


class VisionCache:
    """
    Persistent file-based cache for vision extraction results.

    Parameters
    ----------
    cache_dir:
        Directory where JSON cache files are stored.
        Created automatically if it does not exist.
    max_age_days:
        Maximum age of a cache entry in days.  Entries older than this
        are treated as expired (re-analysed on next access).
        ``None`` means entries never expire.
    """

    def __init__(
        self,
        cache_dir: Path = _DEFAULT_CACHE_DIR,
        max_age_days: Optional[float] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_days * 86_400 if max_age_days else None
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Return cached attributes for *image_path*, or ``None`` on miss.

        Parameters
        ----------
        image_path:
            Absolute or relative path to the garment image file.

        Returns
        -------
        dict | None
            The previously stored attribute dict, or ``None`` if the
            image is not cached / cache is stale.
        """
        cache_file = self._cache_file(image_path)
        if cache_file is None:
            # image file doesn't exist — nothing to look up
            self._misses += 1
            return None
        if not cache_file.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"VisionCache: corrupt entry {cache_file.name} — {exc}")
            cache_file.unlink(missing_ok=True)
            self._misses += 1
            return None

        # Expiry check
        if self.max_age_seconds is not None:
            age = time.time() - data.get("cached_at", 0)
            if age > self.max_age_seconds:
                logger.debug(f"VisionCache: expired entry for {image_path.name}")
                cache_file.unlink(missing_ok=True)
                self._misses += 1
                return None

        self._hits += 1
        logger.debug(f"VisionCache HIT  — {image_path.name}")
        return data.get("attributes")

    def set(self, image_path: Path, attributes: Dict[str, Any]) -> None:
        """
        Persist *attributes* for *image_path*.

        Parameters
        ----------
        image_path:
            Path to the garment image that was analysed.
        attributes:
            Raw attribute dict returned by the extractor.
        """
        cache_file = self._cache_file(image_path)
        if cache_file is None:
            logger.warning(f"VisionCache: cannot cache — image not found: {image_path}")
            return
        payload = {
            "_image_name": image_path.name,
            "cached_at": time.time(),
            "attributes": attributes,
        }
        try:
            cache_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            logger.debug(f"VisionCache SET  — {image_path.name}")
        except OSError as exc:
            logger.warning(f"VisionCache: could not write {cache_file.name} — {exc}")

    def invalidate(self, image_path: Path) -> bool:
        """
        Remove the cache entry for *image_path*.

        Returns ``True`` if an entry was deleted, ``False`` otherwise.
        """
        cache_file = self._cache_file(image_path)
        if cache_file is None:
            return False
        if cache_file.exists():
            cache_file.unlink()
            logger.debug(f"VisionCache INVALIDATE — {image_path.name}")
            return True
        return False

    def clear(self) -> int:
        """Remove ALL cache entries.  Returns count of deleted files."""
        deleted = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            deleted += 1
        logger.info(f"VisionCache: cleared {deleted} entries")
        self._hits = 0
        self._misses = 0
        return deleted

    def stats(self) -> Dict[str, Any]:
        """Return hit/miss statistics and cache size."""
        total = self._hits + self._misses
        files = list(self.cache_dir.glob("*.json"))
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "cache_size": len(files),
            "cache_dir": str(self.cache_dir),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _image_hash(self, image_path: Path) -> Optional[str]:
        """Compute SHA-256 hash of the image file bytes.  Returns None if file missing."""
        try:
            sha = hashlib.sha256()
            with open(image_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65_536), b""):
                    sha.update(chunk)
            return sha.hexdigest()
        except OSError:
            return None

    def _cache_file(self, image_path: Path) -> Optional[Path]:
        """Return the cache file Path for the given image, or None if image missing."""
        image_hash = self._image_hash(Path(image_path))
        if image_hash is None:
            return None
        return self.cache_dir / f"{image_hash}.json"


# ---------------------------------------------------------------------------
# Module-level singleton (shared across the process)
# ---------------------------------------------------------------------------
_default_cache: Optional[VisionCache] = None


def get_vision_cache(cache_dir: Path = _DEFAULT_CACHE_DIR) -> VisionCache:
    """Return the process-level VisionCache singleton."""
    global _default_cache
    if _default_cache is None:
        _default_cache = VisionCache(cache_dir=cache_dir)
    return _default_cache
