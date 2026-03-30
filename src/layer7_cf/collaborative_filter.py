"""
Collaborative Filter — ALS via ``implicit``.
==============================================

Wraps `implicit.als.AlternatingLeastSquares` and exposes high-level
methods for outfit recommendation, similarity search and boost scoring.

Key implementation notes
------------------------
* ``implicit`` expects the training matrix in **items × users** format
  (CSR sparse).  We store the transpose (``user_items_csr``) for
  recommendation calls.
* Cold-start users (not in the training set) receive popularity-based
  recommendations with ``confidence=0.3``.
* The model refuses to train when there are fewer than 10 unique users
  (too little signal for matrix factorisation).
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core import get_logger
from .models import CFScore, InteractionMatrix

logger = get_logger(__name__)

# Minimum users required before we train
_MIN_USERS = 10

# ALS hyper-parameters (sensible defaults)
_DEFAULT_FACTORS = 64
_DEFAULT_ITERATIONS = 20
_DEFAULT_REGULARIZATION = 0.1


class CollaborativeFilter:
    """ALS-based collaborative filter for implicit feedback.

    Parameters
    ----------
    factors : int
        Dimensionality of the latent embeddings.
    iterations : int
        Number of ALS iterations.
    regularization : float
        L2 regularisation strength.
    use_gpu : bool
        Whether to use GPU acceleration (requires ``implicit[gpu]``).
    """

    def __init__(
        self,
        factors: int = _DEFAULT_FACTORS,
        iterations: int = _DEFAULT_ITERATIONS,
        regularization: float = _DEFAULT_REGULARIZATION,
        use_gpu: bool = False,
    ) -> None:
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.use_gpu = use_gpu

        # State
        self._model: Any = None
        self._user_items_csr: Any = None  # scipy CSR  (users × items)
        self._item_users_csr: Any = None  # scipy CSR  (items × users)
        self._user_to_idx: Dict[str, int] = {}
        self._garment_to_idx: Dict[str, int] = {}
        self._idx_to_user: Dict[int, str] = {}
        self._idx_to_garment: Dict[int, str] = {}
        self._popularity: Optional[np.ndarray] = None  # column-sum vector
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, matrix: InteractionMatrix) -> bool:
        """Fit the ALS model on *matrix*.

        Returns ``True`` on success, ``False`` if training was skipped
        (e.g. too few users).
        """
        from scipy.sparse import csr_matrix
        from implicit.als import AlternatingLeastSquares

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

        # Fit ALS
        model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            use_gpu=self.use_gpu,
        )
        model.fit(item_users)
        self._model = model
        self.is_trained = True

        logger.info(
            "CF model trained — %d users × %d garments, %d factors",
            n_users, n_garments, self.factors,
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
    ) -> List[Tuple[str, float]]:
        """Find *n* users most similar to *user_id* in latent space.

        Returns a list of ``(user_id, similarity_score)`` tuples.
        """
        if not self.is_trained:
            return []
        if user_id not in self._user_to_idx:
            return []

        user_idx = self._user_to_idx[user_id]
        ids, scores = self._model.similar_users(user_idx, N=n + 1)
        results: List[Tuple[str, float]] = []
        for idx, score in zip(ids, scores):
            idx = int(idx)
            if idx == user_idx:
                continue
            uid = self._idx_to_user.get(idx, "?")
            results.append((uid, float(score)))
        return results[:n]

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
            if ui_path.exists() and iu_path.exists():
                self._user_items_csr = load_npz(ui_path)
                self._item_users_csr = load_npz(iu_path)

            logger.info("CF model loaded from %s (trained=%s)", directory, self.is_trained)
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
            "factors": self.factors,
            "iterations": self.iterations,
            "regularization": self.regularization,
            "n_users": len(self._user_to_idx),
            "n_garments": len(self._garment_to_idx),
        }
