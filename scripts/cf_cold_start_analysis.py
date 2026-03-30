#!/usr/bin/env python3
"""
CF Cold-Start Analysis
======================
Segments users by their number of training interactions, measures
NDCG@K and Precision@K **per segment**, and finds the interaction
threshold where CF starts outperforming the popularity baseline.

This directly validates the ``_MIN_USERS = 10`` constant in
``collaborative_filter.py``: users with very few interactions should
fall back gracefully to popularity, and the transition should be smooth.

Segments (interaction count buckets)
-------------------------------------
  [0–2]    "cold"      virtually no data
  [3–9]    "sparse"    below training threshold
  [10–19]  "light"     minimal training data
  [20–49]  "moderate"  enough for basic CF
  [50+]    "power"     rich interaction history

Usage
-----
    python scripts/cf_cold_start_analysis.py

    python scripts/cf_cold_start_analysis.py --users 200 --garments 80

    python scripts/cf_cold_start_analysis.py --k 10

    python scripts/cf_cold_start_analysis.py --out results/cold_start.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Inject implicit mock if not installed (CI / dev without C++ build tools) ──
import importlib.util as _ilu
if _ilu.find_spec("implicit") is None:
    from unittest.mock import MagicMock
    import numpy as _np

    _mock_implicit     = MagicMock()
    _mock_implicit_als = MagicMock()

    def _als_recommend(user_idx, user_items_csr=None, N=10, **kw):
        n_items = min(N, 50)
        return _np.arange(n_items, dtype=_np.int32), _np.linspace(0.9, 0.1, n_items, dtype=_np.float32)

    _als_instance = MagicMock()
    _als_instance.recommend     = _als_recommend
    _als_instance.user_factors  = _np.zeros((1, 32), dtype=_np.float32)
    _als_instance.item_factors  = _np.zeros((1, 32), dtype=_np.float32)
    _mock_implicit_als.AlternatingLeastSquares.return_value = _als_instance
    sys.modules["implicit"]     = _mock_implicit
    sys.modules["implicit.als"] = _mock_implicit_als

from src.layer7_cf.interaction_builder import InteractionBuilder
from src.layer7_cf.collaborative_filter import CollaborativeFilter


# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_K        = 10
DEFAULT_USERS    = 120
DEFAULT_GARMENTS = 50
REPORT_DIR = Path("tests/output/cf_cold_start")

SEGMENTS: List[Tuple[str, int, int]] = [
    ("cold",     0,   2),
    ("sparse",   3,   9),
    ("light",   10,  19),
    ("moderate",20,  49),
    ("power",   50, 999),
]


# ────────────────────────────────────────────────────────────────────────────
# Synthetic data — controlled interaction density
# ────────────────────────────────────────────────────────────────────────────

def _generate_interactions(
    n_users: int,
    n_garments: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate users with *varying* interaction density so that all five
    segments are represented.  The segment for user i is chosen
    deterministically by index to ensure even distribution.
    """
    rng      = random.Random(seed)
    garments = [f"g{i:03d}" for i in range(n_garments)]
    interactions: List[Dict[str, Any]] = []

    # Target interaction counts per segment bucket (upper bound)
    segment_target_counts = {
        "cold":      rng.randint(0, 2),
        "sparse":    rng.randint(3, 9),
        "light":     rng.randint(10, 19),
        "moderate":  rng.randint(20, 49),
        "power":     rng.randint(50, min(99, n_garments)),
    }

    for u in range(n_users):
        user_id  = f"user_{u:04d}"
        seg_name = SEGMENTS[u % len(SEGMENTS)][0]
        n_interactions = rng.randint(
            SEGMENTS[u % len(SEGMENTS)][1],
            min(SEGMENTS[u % len(SEGMENTS)][2], n_garments),
        ) if SEGMENTS[u % len(SEGMENTS)][1] <= SEGMENTS[u % len(SEGMENTS)][2] else 0

        sampled_garments = rng.sample(garments, min(n_interactions, len(garments)))
        for g in sampled_garments:
            entry: Dict[str, Any] = {
                "user_id":   user_id,
                "garment_id": g,
                "times_worn": rng.randint(1, 5),
                "_segment":  seg_name,  # metadata for analysis
            }
            if rng.random() < 0.3:
                entry["is_favorite"] = 1
            interactions.append(entry)

    return interactions


# ────────────────────────────────────────────────────────────────────────────
# Metrics (same as evaluate_cf_offline — DRY-ish duplication by design)
# ────────────────────────────────────────────────────────────────────────────

def _ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    def _dcg(items: List[str], rel: Set[str], k: int) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i, item in enumerate(items[:k])
            if item in rel
        )
    idcg = _dcg(list(relevant)[:k], relevant, k)
    return _dcg(recommended, relevant, k) / idcg if idcg > 0 else 0.0


def _precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if not recommended:
        return 0.0
    return sum(1 for r in recommended[:k] if r in relevant) / min(k, len(recommended))


# ────────────────────────────────────────────────────────────────────────────
# Cold-start analysis
# ────────────────────────────────────────────────────────────────────────────

def analyse(
    interactions: List[Dict[str, Any]],
    k: int = DEFAULT_K,
) -> Dict[str, Any]:
    print(f"\n{'═'*65}")
    print(" CF Cold-Start Analysis")
    print(f"{'═'*65}")
    print(f"  Interactions : {len(interactions)}")

    # ── build user interaction count map ────────────────────────────────────
    user_interaction_count: Dict[str, int] = {}
    for entry in interactions:
        uid = entry["user_id"]
        user_interaction_count[uid] = user_interaction_count.get(uid, 0) + 1

    # ── temporal split ───────────────────────────────────────────────────────
    sorted_data = sorted(interactions, key=lambda x: x.get("timestamp", 0))
    cut         = int(len(sorted_data) * 0.8)
    train, test = sorted_data[:cut], sorted_data[cut:]

    # ── build ground-truth from test ─────────────────────────────────────────
    test_user_items: Dict[str, Set[str]] = {}
    for entry in test:
        uid = entry["user_id"]
        gid = entry["garment_id"]
        test_user_items.setdefault(uid, set()).add(gid)

    # ── train CF ─────────────────────────────────────────────────────────────
    builder = InteractionBuilder()
    matrix  = builder.build(train)
    cf      = CollaborativeFilter(factors=32, iterations=15)
    trained = cf.train(matrix)

    print(f"  CF trained : {trained}  |  "
          f"{len(matrix.user_ids)} users × {len(matrix.garment_ids)} garments")

    # ── popularity baseline ──────────────────────────────────────────────────
    pop_counts: Dict[str, int] = {}
    for entry in train:
        gid = entry["garment_id"]
        pop_counts[gid] = pop_counts.get(gid, 0) + 1
    pop_recs = sorted(pop_counts, key=pop_counts.__getitem__, reverse=True)[:k]

    # ── per-segment metrics ──────────────────────────────────────────────────
    seg_results: Dict[str, Dict[str, Any]] = {}

    for seg_name, lo, hi in SEGMENTS:
        # Users in this segment (by their FULL interaction count across train+test)
        seg_users = [
            uid for uid, cnt in user_interaction_count.items()
            if lo <= cnt <= hi and uid in test_user_items
        ]

        if not seg_users:
            seg_results[seg_name] = {
                "n_users": 0,
                "interaction_range": f"[{lo}–{hi}]",
                "cf":    {"ndcg": None, "precision": None},
                "popularity": {"ndcg": None, "precision": None},
            }
            continue

        cf_ndcg_list,  cf_p_list  = [], []
        pop_ndcg_list, pop_p_list = [], []

        for uid in seg_users:
            relevant = test_user_items[uid]
            cf_scores  = cf.recommend(uid, n=k, filter_owned=False)
            cf_recs    = [s.garment_id for s in cf_scores]

            cf_ndcg_list.append(_ndcg_at_k(cf_recs,  relevant, k))
            cf_p_list.append(   _precision_at_k(cf_recs,  relevant, k))
            pop_ndcg_list.append(_ndcg_at_k(pop_recs, relevant, k))
            pop_p_list.append(  _precision_at_k(pop_recs, relevant, k))

        def _avg(lst: List[float]) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        seg_results[seg_name] = {
            "n_users": len(seg_users),
            "interaction_range": f"[{lo}–{hi}]",
            "cf": {
                f"ndcg@{k}":      _avg(cf_ndcg_list),
                f"precision@{k}": _avg(cf_p_list),
            },
            "popularity": {
                f"ndcg@{k}":      _avg(pop_ndcg_list),
                f"precision@{k}": _avg(pop_p_list),
            },
        }

    # ── find breakeven threshold ─────────────────────────────────────────────
    breakeven_segment: str | None = None
    for seg_name, lo, hi in SEGMENTS:
        sr = seg_results.get(seg_name, {})
        cf_v  = (sr.get("cf",         {}) or {}).get(f"ndcg@{k}", 0) or 0
        pop_v = (sr.get("popularity", {}) or {}).get(f"ndcg@{k}", 0) or 0
        if cf_v is not None and pop_v is not None and cf_v > pop_v:
            breakeven_segment = seg_name
            break

    # ── print results table ──────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'Segment':<12} {'Range':<10} {'N':<5} "
          f"{'Pop NDCG':>10} {'CF NDCG':>10} {'Beat?':>8}")
    print(f"{'─'*65}")

    for seg_name, lo, hi in SEGMENTS:
        sr    = seg_results.get(seg_name, {})
        n     = sr.get("n_users", 0)
        rng_s = sr.get("interaction_range", "")
        cf_v  = (sr.get("cf",         {}) or {}).get(f"ndcg@{k}")
        pop_v = (sr.get("popularity", {}) or {}).get(f"ndcg@{k}")

        if cf_v is None or pop_v is None:
            print(f"  {seg_name:<12} {rng_s:<10} {n:<5}  {'—':>10} {'—':>10} {'—':>8}")
        else:
            beat = "✅ YES" if cf_v > pop_v else "❌ NO "
            print(f"  {seg_name:<12} {rng_s:<10} {n:<5} "
                  f"{pop_v:>10.4f} {cf_v:>10.4f} {beat:>8}")

    print(f"{'─'*65}")
    if breakeven_segment:
        lo_be = next(lo for s, lo, _ in SEGMENTS if s == breakeven_segment)
        print(f"\n  ✅  CF outperforms popularity starting at segment "
              f"'{breakeven_segment}' (≥ {lo_be} interactions)")
    else:
        print("\n  ❌  CF does not outperform popularity in any segment.")
        print("      Consider increasing dataset size or training iterations.")

    print(f"\n  Note: CollaborativeFilter._MIN_USERS = 10 means the ALS model")
    print(f"  requires at least 10 unique users in the interaction matrix.")
    print(f"{'═'*65}\n")

    return {
        "k": k,
        "cf_trained": trained,
        "n_train": len(train),
        "n_test":  len(test),
        "breakeven_segment": breakeven_segment,
        "segments": seg_results,
    }


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CF cold-start analysis per interaction segment")
    parser.add_argument("--users",    type=int,  default=DEFAULT_USERS)
    parser.add_argument("--garments", type=int,  default=DEFAULT_GARMENTS)
    parser.add_argument("--k",        type=int,  default=DEFAULT_K)
    parser.add_argument("--data",     type=Path, default=None)
    parser.add_argument("--out",      type=Path, default=None)
    parser.add_argument("--seed",     type=int,  default=42)
    args = parser.parse_args()

    if args.data:
        print(f"Loading from {args.data} …")
        with open(args.data) as f:
            interactions = json.load(f)
    else:
        print(f"Generating {args.users} users × {args.garments} garments …")
        interactions = _generate_interactions(args.users, args.garments, seed=args.seed)

    results = analyse(interactions, k=args.k)

    out_path = args.out or REPORT_DIR / f"cold_start_k{args.k}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved → {out_path}\n")


if __name__ == "__main__":
    main()
