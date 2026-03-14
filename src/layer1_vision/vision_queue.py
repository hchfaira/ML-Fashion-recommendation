"""
Vision Queue  (Solution 3 + 5 — Background Pre-processing & Queue System)
==========================================================================

Async queue-based vision extraction worker system.

Concepts
--------
* **Pre-processing at upload time** — callers submit image paths to the
  queue immediately (fire-and-forget).  The worker analyses them in the
  background.  By the time the user requests their wardrobe, everything
  is already in the VisionCache.

* **Worker pool** — multiple ``asyncio`` workers consume the queue
  concurrently (default: 3).  This limits API concurrency to avoid rate
  limits while still being much faster than sequential extraction.

* **Progressive results** — callers can subscribe to ``on_complete``
  callbacks and receive each garment as soon as it is ready, instead of
  waiting for the entire batch.

Architecture
------------

    ┌─────────────┐     enqueue()      ┌──────────────────┐
    │   Caller    │ ──────────────────▶│  VisionQueue     │
    │  (upload /  │                    │  asyncio.Queue   │
    │   folder)   │◀── on_complete ────│  N workers       │
    └─────────────┘     callback       │  VisionCache     │
                                       └──────────────────┘
                                              │
                                              ▼ (cache miss only)
                                       ┌──────────────────┐
                                       │  AttributeExtractor│
                                       │  (Gemini API)    │
                                       └──────────────────┘

Usage
-----
    queue = VisionQueue(max_workers=3)
    await queue.start()

    # Submit a whole folder
    job_ids = await queue.enqueue_folder(Path("data/sample_wardrobe/"))

    # Wait for all to finish and get garments
    garments = await queue.wait_all(job_ids)

    await queue.stop()

    # Or use the async context manager
    async with VisionQueue(max_workers=3) as q:
        garments = await q.enqueue_and_wait_folder(Path("data/sample_wardrobe/"))
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from src.core import get_logger
from src.core.models import Garment, GarmentAttributes
from src.layer1_vision.vision_cache import VisionCache, get_vision_cache

logger = get_logger(__name__)

_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class VisionJob:
    """A single vision extraction job."""
    job_id: str
    image_path: Path
    status: JobStatus = JobStatus.PENDING
    garment: Optional[Garment] = None
    error: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    @property
    def elapsed_ms(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.submitted_at) * 1000
        return None


# Callback type: called with the completed VisionJob
OnCompleteCallback = Callable[[VisionJob], None]


class VisionQueue:
    """
    Async queue-based vision extraction worker pool.

    Parameters
    ----------
    max_workers : int
        Number of concurrent extraction workers (default 3).
    cache : VisionCache | None
        Cache instance to use.  Defaults to the process singleton.
    on_complete : OnCompleteCallback | None
        Optional callback called for each completed job.
    """

    def __init__(
        self,
        max_workers: int = 3,
        cache: Optional[VisionCache] = None,
        on_complete: Optional[OnCompleteCallback] = None,
        # Test-injection aliases
        vision_cache: Optional[VisionCache] = None,
        extractor: Any = None,
    ) -> None:
        self.max_workers = max_workers
        self.cache = vision_cache or cache or get_vision_cache()
        self.on_complete = on_complete

        self._queue: asyncio.Queue[VisionJob] = asyncio.Queue()
        self._jobs: Dict[str, VisionJob] = {}
        self._workers: List[asyncio.Task] = []
        self._running = False
        # Allow injection of a custom extractor (e.g. a mock in tests)
        self._extractor = extractor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return
        self._running = True
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(i), name=f"vision-worker-{i}")
            self._workers.append(task)
        logger.info(f"VisionQueue started with {self.max_workers} workers")

    async def stop(self) -> None:
        """Gracefully stop the worker pool (finish queued jobs first)."""
        self._running = False
        # Send sentinel None for each worker
        for _ in self._workers:
            await self._queue.put(None)  # type: ignore[arg-type]
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("VisionQueue stopped")

    async def __aenter__(self) -> "VisionQueue":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Submission API
    # ------------------------------------------------------------------

    def enqueue(self, image_path: Path) -> str:
        """
        Submit a single image for extraction.

        Returns
        -------
        str
            Job ID that can be used with :meth:`wait`.
        """
        job_id = str(uuid.uuid4())
        job = VisionJob(job_id=job_id, image_path=image_path)
        self._jobs[job_id] = job
        self._queue.put_nowait(job)
        logger.debug(f"VisionQueue: enqueued {image_path.name} (job={job_id[:8]})")
        return job_id

    def enqueue_folder(self, folder_path: Path) -> List[str]:
        """
        Submit all images in *folder_path* for extraction.

        Returns
        -------
        list[str]
            List of job IDs (one per image).
        """
        if not folder_path.exists():
            return []

        images = sorted(
            p for p in folder_path.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        job_ids = [self.enqueue(img) for img in images]
        logger.info(f"VisionQueue: enqueued {len(job_ids)} images from {folder_path.name}/")
        return job_ids

    # ------------------------------------------------------------------
    # Wait / result API
    # ------------------------------------------------------------------

    async def wait(self, job_id: str, timeout: float = 120.0) -> Optional[VisionJob]:
        """
        Wait for a single job to finish.

        Parameters
        ----------
        job_id : str
        timeout : float
            Maximum seconds to wait.  Returns ``None`` on timeout or unknown ID.
        """
        deadline = time.monotonic() + timeout
        while True:
            job = self._jobs.get(job_id)
            if job is None:
                # Unknown job — poll briefly in case it's being registered
                if time.monotonic() >= deadline:
                    return None
            elif job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CACHED):
                return job
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            await asyncio.sleep(0.05)

    async def wait_all(
        self,
        job_ids: List[str],
        timeout: float = 300.0,
    ) -> List[Optional[Garment]]:
        """
        Wait for all jobs and return garments in submission order.

        Garments for failed jobs are returned as ``None``.
        """
        tasks = [self.wait(jid, timeout=timeout) for jid in job_ids]
        jobs = await asyncio.gather(*tasks, return_exceptions=True)
        garments: List[Optional[Garment]] = []
        for job in jobs:
            if isinstance(job, Exception):
                logger.warning(f"VisionQueue wait_all: job failed — {job}")
                garments.append(None)
            elif isinstance(job, VisionJob) and job.garment is not None:
                garments.append(job.garment)
            else:
                garments.append(None)
        return garments

    async def enqueue_and_wait_folder(
        self,
        folder_path: Path,
        timeout: float = 300.0,
    ) -> List[Garment]:
        """
        Convenience: enqueue folder, wait, return only successful garments.
        """
        job_ids = self.enqueue_folder(folder_path)
        all_garments = await self.wait_all(job_ids, timeout=timeout)
        return [g for g in all_garments if g is not None]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return queue statistics."""
        statuses = [j.status for j in self._jobs.values()]
        return {
            "total": len(self._jobs),
            "total_jobs": len(self._jobs),  # backward-compat alias
            "pending": statuses.count(JobStatus.PENDING),
            "processing": statuses.count(JobStatus.PROCESSING),
            "done": statuses.count(JobStatus.DONE),
            "cached": statuses.count(JobStatus.CACHED),
            "failed": statuses.count(JobStatus.FAILED),
            "queue_size": self._queue.qsize(),
            "workers": self.max_workers,
        }

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------

    async def _worker(self, worker_id: int) -> None:
        """Consume jobs from the queue until stopped."""
        logger.debug(f"VisionQueue worker-{worker_id} started")
        while True:
            item = await self._queue.get()
            if item is None:
                # Sentinel — stop worker
                self._queue.task_done()
                break

            job: VisionJob = item
            job.status = JobStatus.PROCESSING
            try:
                garment = await self._process_job(job)
                job.garment = garment
                # Preserve CACHED status set by _process_job; only override if still PROCESSING
                if job.status == JobStatus.PROCESSING:
                    job.status = JobStatus.DONE if garment else JobStatus.FAILED
            except Exception as exc:
                job.error = str(exc)
                job.status = JobStatus.FAILED
                logger.warning(f"VisionQueue worker-{worker_id}: job {job.job_id[:8]} failed — {exc}")
            finally:
                job.completed_at = time.time()
                self._queue.task_done()
                if self.on_complete:
                    try:
                        self.on_complete(job)
                    except Exception:
                        pass

        logger.debug(f"VisionQueue worker-{worker_id} stopped")

    async def _process_job(self, job: VisionJob) -> Optional[Garment]:
        """Extract garment from image, using cache when available."""
        from uuid import uuid4 as _uuid4

        image_path = job.image_path

        # --- Cache check ---
        cached_attrs = self.cache.get(image_path)
        if cached_attrs is not None:
            job.status = JobStatus.CACHED
            logger.debug(f"VisionQueue: cache hit for {image_path.name}")
            garment = self._attrs_dict_to_garment(cached_attrs, image_path)
            return garment

        # --- Extraction ---
        extractor = self._get_extractor()

        # Support both OutfitBuilder-style (extract_garment_from_image → Garment)
        # and AttributeExtractor-style (_extract_single_garment → GarmentAttributes)
        if hasattr(extractor, "extract_garment_from_image"):
            garment = await extractor.extract_garment_from_image(image_path)
            if garment is None:
                return None
            # Persist attrs to cache
            try:
                self.cache.set(image_path, garment.attributes.model_dump())
            except Exception as exc:
                logger.warning(f"VisionQueue: could not cache {image_path.name} — {exc}")
            return garment
        else:
            attrs = await extractor._extract_single_garment(str(image_path))
            try:
                self.cache.set(image_path, attrs.model_dump())
            except Exception as exc:
                logger.warning(f"VisionQueue: could not cache {image_path.name} — {exc}")
            return Garment(
                id=str(_uuid4()),
                image_path=str(image_path),
                attributes=attrs,
            )

    def _get_extractor(self):
        """Lazy-init AttributeExtractor (shared across workers)."""
        if self._extractor is None:
            from src.layer1_vision.attribute_extractor import AttributeExtractor
            self._extractor = AttributeExtractor()
        return self._extractor

    @staticmethod
    def _attrs_dict_to_garment(attrs_dict: Dict[str, Any], image_path: Path) -> Garment:
        """Reconstruct a Garment from a cached attribute dict."""
        from uuid import uuid4 as _uuid4
        from src.core.models import GarmentAttributes

        try:
            attrs = GarmentAttributes.model_validate(attrs_dict)
        except Exception as exc:
            logger.warning(f"VisionQueue: could not reconstruct GarmentAttributes from cache — {exc}")
            raise

        return Garment(
            id=str(_uuid4()),
            image_path=str(image_path),
            attributes=attrs,
        )
