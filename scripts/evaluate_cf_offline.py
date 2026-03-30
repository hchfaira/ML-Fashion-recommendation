#!/usr/bin/env python3
"""
CF Offline Evaluation
=====================
Measures Precision@K, Recall@K, NDCG@K and Coverage of the
Collaborative Filter using a **temporal train/test split** on synthetic
(or real) interaction data — no real users required.

Pipeline
--------
1. Generate (or load) interaction data
2. Split by time cutoff  →  train set  |  test set
3. Train CF on train set
4. For each user in test set: get top-K recommendations
5. Compute Precision@K, Recall@K, NDCG@K, Coverage
6. Repeat for style-only baseline (popularity)
7. Print comparison table + save JSON report

Usage
-----
    # Quick synthetic evaluation (default: 30 users, 20 garments)
    python scripts/evaluate_cf_offline.py

    # Larger synthetic dataset
    python scripts/evaluate_cf_offline.py --users 100 --garments 50

    # Load real interactions from a JSON file
    python scripts/evaluate_cf_offline.py --data path/to/interactions.json

    # Change top-K
    python scripts/evaluate_cf_offline.py --k 5

    # Save report to custom path
    python scripts/evaluate_cf_offline.py --out results/cf_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ── project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Inject implicit mock if not installed (CI / dev without C++ build tools) ──
import importlib.util as _ilu
if _ilu.find_spec("implicit") is None:
    from unittest.mock import MagicMock
    import numpy as _np

    _mock_implicit     = MagicMock()
    _mock_implicit_als = MagicMock()

    def _als_recommend(user_idx, user_items_csr=None, N=10, **kw):
        n_items = min(N, 30)
        return _np.arange(n_items, dtype=_np.int32), _np.linspace(0.9, 0.1, n_items, dtype=_np.float32)

    _als_instance = MagicMock()
    _als_instance.recommend     = _als_recommend
    _als_instance.user_factors  = _np.zeros((1, 32), dtype=_np.float32)
    _als_instance.item_factors  = _np.zeros((1, 32), dtype=_np.float32)
    _mock_implicit_als.AlternatingLeastSquares.return_value = _als_instance
    sys.modules["implicit"]     = _mock_implicit
    sys.modules["implicit.als"] = _mock_implicit_als

from src.layer7_cf.interaction_builder import InteractionBuilder, SIGNAL_WEIGHTS
from src.layer7_cf.collaborative_filter import CollaborativeFilter
from src.layer7_cf.models import InteractionMatrix


# ── constants ────────────────────────────────────────────────────────────────
CUTOFF_RATIO = 0.8          # 80 % train, 20 % test (temporal split)
DEFAULT_K    = 10
DEFAULT_USERS    = 30
DEFAULT_GARMENTS = 20
REPORT_DIR = Path("tests/output/cf_eval")


# ────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ────────────────────────────────────────────────────────────────────────────

def _generate_interactions(
    n_users: int,
    n_garments: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate realistic-looking interactions with a timestamp field.

    Each user has a random favourite cluster of garments they interact
    with most (to make CF actually useful vs. pure popularity).
    """
    rng = random.Random(seed)
    garments = [f"g{i:03d}" for i in range(n_garments)]
    interactions: List[Dict[str, Any]] = []

    for u in range(n_users):
        user_id = f"user_{u:03d}"
        # Each user strongly prefers a personal cluster of garments
        fav_count = max(2, n_garments // 4)
        favourites: Set[str] = set(rng.sample(garments, fav_count))

        for t, g in enumerate(garments):
            # Base probability — higher for favourites
            p = 0.70 if g in favourites else 0.15
            if rng.random() > p:
                continue

            entry: Dict[str, Any] = {
                "user_id":    user_id,
                "garment_id": g,
                "timestamp":  t + u * n_garments,   # deterministic ordering
            }

            # Add random signals
            if g in favourites:
                entry["times_worn"]   = rng.randint(3, 12)
                entry["is_favorite"]  = 1
                entry["outfit_liked"] = 1
            else:
                entry["times_worn"]   = rng.randint(1, 3)
                entry["outfit_saved"] = 1

            interactions.append(entry)

    return interactions


# ────────────────────────────────────────────────────────────────────────────
# Temporal split
# ────────────────────────────────────────────────────────────────────────────

def _temporal_split(
    interactions: List[Dict[str, Any]],
    cutoff_ratio: float = CUTOFF_RATIO,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split by timestamp.  Earlier interactions → train, later → test."""
    sorted_data = sorted(interactions, key=lambda x: x.get("timestamp", 0))
    cut = int(len(sorted_data) * cutoff_ratio)
    return sorted_data[:cut], sorted_data[cut:]


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def _precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    if not recommended:
        return 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / min(k, len(recommended))


def _recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant items found in top-K recommendations."""
    if not relevant:
        return 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / len(relevant)


def _ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K."""
    def _dcg(items: List[str], rel: Set[str], k: int) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i, item in enumerate(items[:k])
            if item in rel
        )

    dcg  = _dcg(recommended, relevant, k)
    # Ideal DCG: all relevant items at the top
    ideal_items = list(relevant)[:k]
    idcg = _dcg(ideal_items, relevant, k)
    return dcg / idcg if idcg > 0 else 0.0


def _coverage(all_recommendations: List[List[str]], all_garments: List[str]) -> float:
    """Fraction of the catalogue ever recommended to any user."""
    recommended_set = {g for recs in all_recommendations for g in recs}
    return len(recommended_set) / len(all_garments) if all_garments else 0.0


# ────────────────────────────────────────────────────────────────────────────
# Popularity baseline
# ────────────────────────────────────────────────────────────────────────────

def _popularity_baseline(
    train: List[Dict[str, Any]],
    k: int,
) -> List[str]:
    """Return the globally most-interacted garments as popularity baseline."""
    counts: Dict[str, float] = {}
    for entry in train:
        gid = entry.get("garment_id", "")
        counts[gid] = counts.get(gid, 0.0) + 1.0
    return sorted(counts, key=counts.__getitem__, reverse=True)[:k]


# ────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate(
    interactions: List[Dict[str, Any]],
    k: int = DEFAULT_K,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full offline evaluation and return a results dict."""
    print(f"\n{'═'*60}")
    print(" CF Offline Evaluation")
    print(f"{'═'*60}")
    print(f"  Total interactions : {len(interactions)}")

    # ── split ────────────────────────────────────────────────────────────────
    train, test = _temporal_split(interactions)
    print(f"  Train interactions : {len(train)}")
    print(f"  Test  interactions : {len(test)}")

    all_garments = list({e["garment_id"] for e in interactions})

    # ── build test ground-truth ──────────────────────────────────────────────
    # For each user in test set: which garments did they interact with?
    test_user_items: Dict[str, Set[str]] = {}
    for entry in test:
        uid = entry["user_id"]
        gid = entry["garment_id"]
        test_user_items.setdefault(uid, set()).add(gid)

    # ── train CF ─────────────────────────────────────────────────────────────
    builder = InteractionBuilder()
    matrix  = builder.build(train)
    cf      = CollaborativeFilter(factors=32, iterations=15)

    print(f"\n  Training CF on {len(matrix.user_ids)} users × "
          f"{len(matrix.garment_ids)} garments …")
    t0 = time.time()
    trained = cf.train(matrix)
    train_time = time.time() - t0

    if not trained:
        print("  ⚠️  CF not trained (too few users — falling back to cold-start only)")

    print(f"  Training time      : {train_time:.2f}s  |  trained={trained}")

    # ── popularity baseline ──────────────────────────────────────────────────
    pop_recs = _popularity_baseline(train, k)

    # ── per-user metrics ─────────────────────────────────────────────────────
    cf_p, cf_r, cf_ndcg       = [], [], []
    pop_p, pop_r, pop_ndcg    = [], [], []
    cf_all_recs:  List[List[str]] = []
    pop_all_recs: List[List[str]] = []

    cold_start_users = 0

    for uid, relevant in test_user_items.items():
        # CF recommendations
        cf_scores  = cf.recommend(uid, n=k, filter_owned=False)
        cf_rec_ids = [s.garment_id for s in cf_scores]

        if not cf_scores:
            cold_start_users += 1

        cf_all_recs.append(cf_rec_ids)
        cf_p.append(_precision_at_k(cf_rec_ids, relevant, k))
        cf_r.append(_recall_at_k(cf_rec_ids, relevant, k))
        cf_ndcg.append(_ndcg_at_k(cf_rec_ids, relevant, k))

        # Popularity baseline
        pop_all_recs.append(pop_recs)
        pop_p.append(_precision_at_k(pop_recs, relevant, k))
        pop_r.append(_recall_at_k(pop_recs, relevant, k))
        pop_ndcg.append(_ndcg_at_k(pop_recs, relevant, k))

    n_eval = len(test_user_items)
    if n_eval == 0:
        print("  ⚠️  No test users — increase dataset size.")
        return {}

    def _avg(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    results = {
        "k": k,
        "n_train": len(train),
        "n_test":  len(test),
        "n_eval_users": n_eval,
        "cold_start_users": cold_start_users,
        "cold_start_rate": cold_start_users / n_eval,
        "cf_trained": trained,
        "train_time_s": round(train_time, 3),
        "cf": {
            f"precision@{k}":  round(_avg(cf_p),    4),
            f"recall@{k}":     round(_avg(cf_r),    4),
            f"ndcg@{k}":       round(_avg(cf_ndcg), 4),
            "coverage":        round(_coverage(cf_all_recs, all_garments), 4),
        },
        "popularity_baseline": {
            f"precision@{k}":  round(_avg(pop_p),    4),
            f"recall@{k}":     round(_avg(pop_r),    4),
            f"ndcg@{k}":       round(_avg(pop_ndcg), 4),
            "coverage":        round(_coverage(pop_all_recs, all_garments), 4),
        },
    }

    # ── compute lift ─────────────────────────────────────────────────────────
    def _lift(cf_val: float, base_val: float) -> str:
        if base_val == 0:
            return "—"
        return f"{(cf_val - base_val) / base_val * 100:+.1f}%"

    cf_m   = results["cf"]
    pop_m  = results["popularity_baseline"]

    # ── print table ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Metric':<20} {'Popularity':>12} {'CF':>12} {'Lift':>10}")
    print(f"{'─'*60}")
    for metric in [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage"]:
        cf_v   = cf_m[metric]
        pop_v  = pop_m[metric]
        lift   = _lift(cf_v, pop_v)
        print(f"  {metric:<20} {pop_v:>12.4f} {cf_v:>12.4f} {lift:>10}")
    print(f"{'─'*60}")
    print(f"  Cold-start rate    : {results['cold_start_rate']:.1%}  "
          f"({cold_start_users}/{n_eval} users)")

    # ── interpretation ───────────────────────────────────────────────────────
    ndcg_val = cf_m[f"ndcg@{k}"]
    if ndcg_val >= 0.30:
        verdict = "✅  CF is performing WELL (NDCG ≥ 0.30)"
    elif ndcg_val >= 0.20:
        verdict = "⚠️  CF needs tuning (NDCG 0.20–0.30)"
    else:
        verdict = "❌  CF is underperforming (NDCG < 0.20)"
    print(f"\n  Verdict: {verdict}")
    print(f"{'═'*60}\n")

    return results


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CF offline evaluation")
    parser.add_argument("--users",    type=int,  default=DEFAULT_USERS,    help="Synthetic user count")
    parser.add_argument("--garments", type=int,  default=DEFAULT_GARMENTS, help="Synthetic garment count")
    parser.add_argument("--k",        type=int,  default=DEFAULT_K,        help="Top-K cut-off")
    parser.add_argument("--data",     type=Path, default=None,             help="Path to real interactions JSON")
    parser.add_argument("--out",      type=Path, default=None,             help="Save JSON report to this path")
    parser.add_argument("--seed",     type=int,  default=42,               help="Random seed for synthetic data")
    args = parser.parse_args()

    if args.data:
        print(f"Loading interactions from {args.data} …")
        with open(args.data) as f:
            interactions = json.load(f)
    else:
        print(f"Generating synthetic data: {args.users} users × {args.garments} garments …")
        interactions = _generate_interactions(args.users, args.garments, seed=args.seed)

    results = evaluate(interactions, k=args.k)

    # ── save report ──────────────────────────────────────────────────────────
    out_path = args.out or REPORT_DIR / f"offline_eval_k{args.k}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved → {out_path}\n")


if __name__ == "__main__":
    main()
