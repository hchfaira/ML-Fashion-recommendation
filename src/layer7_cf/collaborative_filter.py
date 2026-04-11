"""
Collaborative Filter — ALS / BPR via ``implicit``.
====================================================

Wraps `implicit.als.AlternatingLeastSquares` **and**
`implicit.bpr.BayesianPersonalizedRanking` and exposes high-level
methods for outfit recommendation, user-based / item-based CF,
similarity search and boost scoring.

Key implementation notes
------------------------
* ``implicit`` expects the training matrix in **items × users** format
  (CSR sparse).  We store the transpose (``user_items_csr``) for
  recommendation calls.
* Cold-start users (not in the training set) receive popularity-based
  recommendations with ``confidence=0.3``.
* The model refuses to train when there are fewer than 10 unique users
  (too little signal for matrix factorisation).
* An optional **Redis** cache stores user latent vectors for fast
  retrieval in real-time scoring (graceful fallback if unavailable).
* An **item co-occurrence matrix** is built alongside the latent model
  to power ``find_item_pairs()``.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core import get_logger
from .models import CFScore, InteractionMatrix, UserCFNeighbour, ItemPair

logger = get_logger(__name__)

# Minimum users required before we train
_MIN_USERS = 10

# ALS hyper-parameters (sensible defaults)
_DEFAULT_FACTORS = 64
_DEFAULT_ITERATIONS = 20
_DEFAULT_REGULARIZATION = 0.1

# Redis key prefix / TTL
_REDIS_PREFIX = "cf:user_vec:"
_REDIS_TTL = 3600 * 6  # 6 hours


class CollaborativeFilter:
    """ALS / BPR collaborative filter for implicit feedback.

    Parameters
    ----------
    factors : int
        Dimensionality of the latent embeddings.
    iterations : int
        Number of ALS / BPR iterations.
    regularization : float
        L2 regularisation strength.
    use_gpu : bool
        Whether to use GPU acceleration (requires ``implicit[gpu]``).
    model_type : str
        ``"als"`` (default) or ``"bpr"``.
    redis_url : str or None
        Redis connection URL.  ``None`` disables caching.
    """

    def __init__(
        self,
        factors: int = _DEFAULT_FACTORS,
        iterations: int = _DEFAULT_ITERATIONS,
        regularization: float = _DEFAULT_REGULARIZATION,
        use_gpu: bool = False,
        model_type: str = "als",
        redis_url: Optional[str] = None,
    ) -> None:
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.use_gpu = use_gpu
        self.model_type = model_type.lower()

        # State
        self._model: Any = None
        self._user_items_csr: Any = None  # scipy CSR  (users × items)
        self._item_users_csr: Any = None  # scipy CSR  (items × users)
        self._cooccurrence_csr: Any = None  # scipy CSR  (items × items)
        self._user_to_idx: Dict[str, int] = {}
        self._garment_to_idx: Dict[str, int] = {}
        self._idx_to_user: Dict[int, str] = {}
        self._idx_to_garment: Dict[int, str] = {}
        self._popularity: Optional[np.ndarray] = None  # column-sum vector
        self.is_trained: bool = False

        # Redis (optional)
        self._redis: Any = None
        if redis_url:
            try:
                import redis as _redis_lib
                self._redis = _redis_lib.Redis.from_url(
                    redis_url, decode_responses=False, socket_timeout=2,
                )
                self._redis.ping()
                logger.info("Redis cache connected at %s", redis_url)
            except Exception as exc:
                logger.warning("Redis unavailable (%s) — caching disabled", exc)
                self._redis = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, matrix: InteractionMatrix) -> bool:
        """Fit the ALS or BPR model on *matrix*.

        Returns ``True`` on success, ``False`` if training was skipped
        (e.g. too few users).
        """
        from scipy.sparse import csr_matrix
        from implicit.als import AlternatingLeastSquares
        from implicit.bpr import BayesianPersonalizedRanking

        n_users = len(matrix.user_ids)
        n_garments = len(matrix.garment_ids)

        if n_users < _MIN_USERS:
            logger.warning(
                "CF training skipped: only %d users (need ≥ %d)",
                n_users,
                _MIN_USERS,
            )
            self.is_trained = False
            return False

        # Build dense numpy array → sparse
        dense = np.array(matrix.data, dtype=np.float32)  # (users, garments)
        user_items = csr_matrix(dense)
        item_users = user_items.T.tocsr()

        # Index maps
        self._user_to_idx = dict(matrix.user_to_idx)
        self._garment_to_idx = dict(matrix.garment_to_idx)
        self._idx_to_user = {v: k for k, v in self._user_to_idx.items()}
        self._idx_to_garment = {v: k for k, v in self._garment_to_idx.items()}

        # Popularity vector (sum of each garment column)
        self._popularity = np.asarray(user_items.sum(axis=0)).flatten()

        # Store sparse matrices
        self._user_items_csr = user_items
        self._item_users_csr = item_users

        # Build item co-occurrence matrix (items × items)
        # C_ij = number of users who interacted with both item i and item j
        binary = (user_items > 0).astype(np.float32)
        self._cooccurrence_csr = (binary.T @ binary).tocsr()
        # Zero-out diagonal
        self._cooccurrence_csr.setdiag(0)
        self._cooccurrence_csr.eliminate_zeros()

        # Fit model
        if self.model_type == "bpr":
            model = BayesianPersonalizedRanking(
                factors=self.factors,
                iterations=self.iterations,
                regularization=self.regularization,
                use_gpu=self.use_gpu,
            )
        else:
            model = AlternatingLeastSquares(
                factors=self.factors,
                iterations=self.iterations,
                regularization=self.regularization,
                use_gpu=self.use_gpu,
            )

        model.fit(item_users)
        self._model = model
        self.is_trained = True

        # Cache user vectors in Redis
        self._cache_user_vectors()

        logger.info(
            "CF model (%s) trained — %d users × %d garments, %d factors",
            self.model_type.upper(), n_users, n_garments, self.factors,
        )
        return True

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        filter_owned: bool = True,
    ) -> List[CFScore]:
        """Return top-*n* item recommendations for *user_id*.

        If the user is unknown (cold start), returns popularity-based
        recommendations with ``confidence=0.3``.
        """
        if not self.is_trained:
            return []

        if user_id not in self._user_to_idx:
            return self._cold_start_recommend(n)

        user_idx = self._user_to_idx[user_id]
        ids, scores = self._model.recommend(
            user_idx,
            self._user_items_csr[user_idx],
            N=n,
            filter_already_liked_items=filter_owned,
        )
        results: List[CFScore] = []
        for item_idx, score in zip(ids, scores):
            gid = self._idx_to_garment.get(int(item_idx), "?")
            # Normalise ALS score to [0, 1]
            norm = float(np.clip(score, 0.0, 1.0))
            results.append(CFScore(garment_id=gid, score=norm, confidence=0.8))
        return results

    # ------------------------------------------------------------------
    # Boost score (for hybrid integration)
    # ------------------------------------------------------------------

    def get_boost_score(self, user_id: str, garment_id: str) -> CFScore:
        """Compute a CF affinity score between *user_id* and *garment_id*.

        Uses cosine similarity between the user-factor and item-factor
        vectors, normalised to ``[0, 1]`` via ``(cos + 1) / 2``.

        Returns a neutral ``CFScore(score=0.5, confidence=0.0)`` if
        either ID is unknown.
        """
        if not self.is_trained:
            return CFScore(garment_id=garment_id, score=0.5, confidence=0.0)

        if user_id not in self._user_to_idx or garment_id not in self._garment_to_idx:
            return CFScore(garment_id=garment_id, score=0.5, confidence=0.0)

        user_idx = self._user_to_idx[user_id]
        garment_idx = self._garment_to_idx[garment_id]

        u_vec = np.array(self._model.user_factors[user_idx], dtype=np.float64)
        i_vec = np.array(self._model.item_factors[garment_idx], dtype=np.float64)

        denom = np.linalg.norm(u_vec) * np.linalg.norm(i_vec)
        if denom == 0:
            return CFScore(garment_id=garment_id, score=0.5, confidence=0.0)

        cosine = float(np.dot(u_vec, i_vec) / denom)
        score = (cosine + 1.0) / 2.0  # map [-1, 1] → [0, 1]
        return CFScore(garment_id=garment_id, score=score, confidence=0.7)

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_similar_garments(
        self,
        garment_id: str,
        n: int = 5,
    ) -> List[CFScore]:
        """Find *n* garments most similar to *garment_id* in latent space."""
        if not self.is_trained:
            return []
        if garment_id not in self._garment_to_idx:
            return []

        garment_idx = self._garment_to_idx[garment_id]
        ids, scores = self._model.similar_items(garment_idx, N=n + 1)
        results: List[CFScore] = []
        for idx, score in zip(ids, scores):
            idx = int(idx)
            if idx == garment_idx:
                continue
            gid = self._idx_to_garment.get(idx, "?")
            results.append(
                CFScore(garment_id=gid, score=float(np.clip(score, 0.0, 1.0)), confidence=0.7)
            )
        return results[:n]

    def find_similar_users(
        self,
        user_id: str,
        n: int = 5,
    ) -> List[UserCFNeighbour]:
        """Find *n* users most similar to *user_id* in latent space.

        Returns a list of :class:`UserCFNeighbour` with similarity and
        the number of shared items.
        """
        if not self.is_trained:
            return []
        if user_id not in self._user_to_idx:
            return []

        user_idx = self._user_to_idx[user_id]
        ids, scores = self._model.similar_users(user_idx, N=n + 1)

        # Precompute the query user's non-zero item set
        query_items = set(self._user_items_csr[user_idx].indices)

        results: List[UserCFNeighbour] = []
        for idx, score in zip(ids, scores):
            idx = int(idx)
            if idx == user_idx:
                continue
            uid = self._idx_to_user.get(idx, "?")
            # Guard against out-of-range indices
            if idx >= self._user_items_csr.shape[0]:
                results.append(
                    UserCFNeighbour(user_id=uid, similarity=float(score), shared_items=0)
                )
                continue
            neighbour_items = set(self._user_items_csr[idx].indices)
            shared = len(query_items & neighbour_items)
            results.append(
                UserCFNeighbour(
                    user_id=uid,
                    similarity=float(score),
                    shared_items=shared,
                )
            )
        return results[:n]

    # ------------------------------------------------------------------
    # Item-based co-occurrence
    # ------------------------------------------------------------------

    def find_item_pairs(
        self,
        garment_id: str,
        n: int = 5,
    ) -> List[ItemPair]:
        """Find *n* garments most commonly co-used with *garment_id*.

        Uses the item co-occurrence matrix (binary user overlap count).
        Returns a list of :class:`ItemPair` sorted by score descending.
        """
        if not self.is_trained or self._cooccurrence_csr is None:
            return []
        if garment_id not in self._garment_to_idx:
            return []

        garment_idx = self._garment_to_idx[garment_id]
        row = self._cooccurrence_csr[garment_idx]
        if row.nnz == 0:
            return []

        # Normalise co-occurrence counts to [0, 1]
        max_count = float(row.max())
        if max_count == 0:
            return []

        # Get top-n indices
        indices = row.indices
        data = row.data
        top_n = min(n, len(indices))
        top_idx = np.argsort(data)[::-1][:top_n]

        results: List[ItemPair] = []
        for i in top_idx:
            paired_idx = int(indices[i])
            co_count = int(data[i])
            paired_id = self._idx_to_garment.get(paired_idx, "?")
            results.append(
                ItemPair(
                    source_id=garment_id,
                    paired_id=paired_id,
                    score=float(data[i]) / max_count,
                    co_users=co_count,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Redis caching helpers
    # ------------------------------------------------------------------

    def _cache_user_vectors(self) -> None:
        """Push all user latent vectors to Redis (fire-and-forget)."""
        if self._redis is None or self._model is None:
            return
        try:
            pipe = self._redis.pipeline(transaction=False)
            for uid, idx in self._user_to_idx.items():
                vec = np.array(self._model.user_factors[idx], dtype=np.float32)
                key = f"{_REDIS_PREFIX}{uid}"
                pipe.setex(key, _REDIS_TTL, vec.tobytes())
            pipe.execute()
            logger.info("Cached %d user vectors in Redis", len(self._user_to_idx))
        except Exception as exc:
            logger.warning("Redis cache write failed: %s", exc)

    def get_cached_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        """Retrieve a user latent vector from Redis (or None)."""
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(f"{_REDIS_PREFIX}{user_id}")
            if raw:
                return np.frombuffer(raw, dtype=np.float32).copy()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Cold start
    # ------------------------------------------------------------------

    def _cold_start_recommend(self, n: int) -> List[CFScore]:
        """Popularity-based fallback for unknown users."""
        if self._popularity is None:
            return []
        top_indices = np.argsort(self._popularity)[::-1][:n]
        total = float(self._popularity.sum()) or 1.0
        results: List[CFScore] = []
        for idx in top_indices:
            gid = self._idx_to_garment.get(int(idx), "?")
            score = float(self._popularity[idx]) / total
            results.append(CFScore(garment_id=gid, score=score, confidence=0.3))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Persist the trained model and index maps to *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self._model,
            "model_type": self.model_type,
            "user_to_idx": self._user_to_idx,
            "garment_to_idx": self._garment_to_idx,
            "idx_to_user": self._idx_to_user,
            "idx_to_garment": self._idx_to_garment,
            "popularity": self._popularity,
            "is_trained": self.is_trained,
            "factors": self.factors,
            "iterations": self.iterations,
            "regularization": self.regularization,
        }
        with open(directory / "cf_model.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Also save sparse matrices
        if self._user_items_csr is not None:
            from scipy.sparse import save_npz
            save_npz(directory / "user_items.npz", self._user_items_csr)
            save_npz(directory / "item_users.npz", self._item_users_csr)
        if self._cooccurrence_csr is not None:
            from scipy.sparse import save_npz
            save_npz(directory / "cooccurrence.npz", self._cooccurrence_csr)

        logger.info("CF model saved to %s", directory)

    def load(self, directory: str | Path) -> bool:
        """Load a previously saved model from *directory*.

        Returns ``True`` on success, ``False`` on any error.
        """
        directory = Path(directory)
        pkl_path = directory / "cf_model.pkl"
        if not pkl_path.exists():
            logger.warning("CF model file not found at %s", pkl_path)
            return False

        try:
            with open(pkl_path, "rb") as f:
                state = pickle.load(f)  # noqa: S301

            self._model = state["model"]
            self.model_type = state.get("model_type", "als")
            self._user_to_idx = state["user_to_idx"]
            self._garment_to_idx = state["garment_to_idx"]
            self._idx_to_user = state["idx_to_user"]
            self._idx_to_garment = state["idx_to_garment"]
            self._popularity = state["popularity"]
            self.is_trained = state["is_trained"]
            self.factors = state["factors"]
            self.iterations = state["iterations"]
            self.regularization = state["regularization"]

            # Sparse matrices (optional, for recommend calls)
            from scipy.sparse import load_npz
            ui_path = directory / "user_items.npz"
            iu_path = directory / "item_users.npz"
            co_path = directory / "cooccurrence.npz"
            if ui_path.exists() and iu_path.exists():
                self._user_items_csr = load_npz(ui_path)
                self._item_users_csr = load_npz(iu_path)
            if co_path.exists():
                self._cooccurrence_csr = load_npz(co_path)

            logger.info("CF model loaded from %s (trained=%s, type=%s)",
                        directory, self.is_trained, self.model_type)
            return True
        except Exception as exc:
            logger.error("Failed to load CF model: %s", exc)
            self.is_trained = False
            return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return a JSON-serialisable status dict."""
        return {
            "trained": self.is_trained,
            "model_type": self.model_type,
            "factors": self.factors,
            "iterations": self.iterations,
            "regularization": self.regularization,
            "n_users": len(self._user_to_idx),
            "n_garments": len(self._garment_to_idx),
            "redis_connected": self._redis is not None,
        }
