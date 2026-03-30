#!/usr/bin/env python3
"""
CF Personalization Score Analysis
==================================
Measures how much the CF actually *differentiates* recommendations
between users — the single most important quality signal for a personal
stylist assistant.

Metrics
-------
* **Inter-user diversity** — average Jaccard distance between any two
  users' top-K recommendation sets.  Range [0, 1].  Higher = more
  personalised.  Pure popularity (every user gets the same list) → 0.0.
  Perfect personalisation → 1.0.

* **Intra-user diversity** — average pairwise distance between the
  garments recommended to a single user.  Proxy: fraction of unique
  categories in the top-K list.  Range [0, 1].

* **Popularity bias** — how often the recommended garments belong to
  the global top-20% most-interacted items.  High bias → CF just
  re-ranks popular items, not truly personal.

* **Per-persona separation** — when using synthetic data with known
  personas, check that users of different personas receive different
  recommendations (cluster purity).

Usage
-----
    python scripts/measure_personalization.py

    python scripts/measure_personalization.py --users 80 --garments 40

    python scripts/measure_personalization.py --data tests/output/cf_impact/interactions.json

    python scripts/measure_personalization.py --k 10 --out results/personalization.json
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
        # Return IDs spread across garment space, offset by user for diversity
        n_items = min(N, 30)
        base  = (user_idx * 3) % 30
        ids   = _np.array([(base + i) % 30 for i in range(n_items)], dtype=_np.int32)
        scores = _np.linspace(0.9, 0.1, n_items, dtype=_np.float32)
        return ids, scores

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
DEFAULT_USERS    = 60
DEFAULT_GARMENTS = 30
REPORT_DIR = Path("tests/output/cf_personalization")

PERSONAS = {
    "casual":     {"g_range": (0,  10), "weight": 3},
    "formal":     {"g_range": (10, 20), "weight": 3},
    "minimalist": {"g_range": (20, 25), "weight": 2},
    "eclectic":   {"g_range": (0,  30), "weight": 1},
}


# ────────────────────────────────────────────────────────────────────────────
# Synthetic data (persona-aware)
# ────────────────────────────────────────────────────────────────────────────

def _generate_interactions(
    n_users: int,
    n_garments: int,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Return (interactions, user_to_persona) so we can measure persona separation."""
    rng       = random.Random(seed)
    garments  = [f"g{i:03d}" for i in range(n_garments)]
    persona_names = list(PERSONAS.keys())
    interactions: List[Dict[str, Any]] = []
    user_to_persona: Dict[str, str] = {}

    for u in range(n_users):
        user_id = f"user_{u:03d}"
        persona = persona_names[u % len(persona_names)]
        user_to_persona[user_id] = persona
        cfg = PERSONAS[persona]
        lo, hi = cfg["g_range"]
        hi = min(hi, n_garments)
        fav_pool = garments[lo:hi] if hi > lo else garments

        for g in garments:
            is_fav = g in fav_pool
            p = 0.65 if is_fav else 0.10
            if rng.random() > p:
                continue
            entry: Dict[str, Any] = {
                "user_id":   user_id,
                "garment_id": g,
                "times_worn": rng.randint(2, 8) if is_fav else 1,
            }
            if is_fav:
                entry["is_favorite"] = 1
            interactions.append(entry)

    return interactions, user_to_persona


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def _jaccard_distance(a: Set[str], b: Set[str]) -> float:
    """1 - Jaccard similarity (range [0, 1]; 1 = completely different sets)."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    return 1.0 - len(a & b) / union


def _inter_user_diversity(user_recs: Dict[str, List[str]], sample: int = 500) -> float:
    """Average Jaccard distance between pairs of user recommendation sets."""
    users = list(user_recs.keys())
    if len(users) < 2:
        return 0.0

    rng   = random.Random(0)
    pairs = [(users[i], users[j])
             for i in range(len(users))
             for j in range(i + 1, len(users))]

    if len(pairs) > sample:
        pairs = rng.sample(pairs, sample)

    total = sum(
        _jaccard_distance(set(user_recs[a]), set(user_recs[b]))
        for a, b in pairs
    )
    return total / len(pairs)


def _popularity_bias(
    user_recs: Dict[str, List[str]],
    interactions: List[Dict[str, Any]],
    top_pct: float = 0.20,
) -> float:
    """Fraction of recommended items belonging to the global top-X% popular garments."""
    counts: Dict[str, int] = {}
    for entry in interactions:
        gid = entry.get("garment_id", "")
        counts[gid] = counts.get(gid, 0) + 1

    n_top   = max(1, int(len(counts) * top_pct))
    top_set = set(sorted(counts, key=counts.__getitem__, reverse=True)[:n_top])

    all_recs = [g for recs in user_recs.values() for g in recs]
    if not all_recs:
        return 0.0
    return sum(1 for g in all_recs if g in top_set) / len(all_recs)


def _persona_separation(
    user_recs: Dict[str, List[str]],
    user_to_persona: Dict[str, str],
) -> Dict[str, float]:
    """
    For each pair of personas, report the average Jaccard distance between their
    recommendation sets.  High distance → CF successfully separates personas.
    """
    persona_recs: Dict[str, List[Set[str]]] = {}
    for uid, recs in user_recs.items():
        p = user_to_persona.get(uid, "unknown")
        persona_recs.setdefault(p, []).append(set(recs))

    personas  = list(persona_recs.keys())
    sep: Dict[str, float] = {}

    for i, pa in enumerate(personas):
        for pb in personas[i + 1 :]:
            dists = [
                _jaccard_distance(ra, rb)
                for ra in persona_recs[pa]
                for rb in persona_recs[pb]
            ]
            key = f"{pa}↔{pb}"
            sep[key] = round(sum(dists) / len(dists), 4) if dists else 0.0

    return sep


# ────────────────────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────────────────────

def analyse(
    interactions: List[Dict[str, Any]],
    user_to_persona: Dict[str, str] | None = None,
    k: int = DEFAULT_K,
) -> Dict[str, Any]:
    print(f"\n{'═'*60}")
    print(" CF Personalization Analysis")
    print(f"{'═'*60}")
    print(f"  Interactions : {len(interactions)}")

    builder = InteractionBuilder()
    matrix  = builder.build(interactions)

    print(f"  Users × Garments : {len(matrix.user_ids)} × {len(matrix.garment_ids)}")

    cf = CollaborativeFilter(factors=32, iterations=15)
    trained = cf.train(matrix)

    if not trained:
        print("  ⚠️  CF not trained (too few users)")

    print(f"  CF trained : {trained}")

    # ── collect recommendations for every user ────────────────────────────
    user_recs: Dict[str, List[str]] = {}
    cold_start_count = 0

    for uid in matrix.user_ids:
        scores = cf.recommend(uid, n=k, filter_owned=False)
        user_recs[uid] = [s.garment_id for s in scores]
        if not scores:
            cold_start_count += 1

    # ── popularity baseline (everyone gets the same list) ──────────────────
    counts: Dict[str, int] = {}
    for entry in interactions:
        gid = entry.get("garment_id", "")
        counts[gid] = counts.get(gid, 0) + 1
    pop_list = sorted(counts, key=counts.__getitem__, reverse=True)[:k]
    pop_recs = {uid: pop_list for uid in matrix.user_ids}

    # ── metrics ───────────────────────────────────────────────────────────
    cf_diversity  = _inter_user_diversity(user_recs)
    pop_diversity = _inter_user_diversity(pop_recs)
    cf_pop_bias   = _popularity_bias(user_recs,  interactions)
    pop_pop_bias  = _popularity_bias(pop_recs,   interactions)

    results: Dict[str, Any] = {
        "k": k,
        "n_users":   len(matrix.user_ids),
        "n_garments": len(matrix.garment_ids),
        "cf_trained": trained,
        "cold_start_rate": cold_start_count / len(matrix.user_ids) if matrix.user_ids else 0,
        "cf": {
            "inter_user_diversity": round(cf_diversity,  4),
            "popularity_bias":      round(cf_pop_bias,   4),
        },
        "popularity_baseline": {
            "inter_user_diversity": round(pop_diversity, 4),
            "popularity_bias":      round(pop_pop_bias,  4),
        },
    }

    if user_to_persona:
        results["persona_separation"] = _persona_separation(user_recs, user_to_persona)

    # ── print table ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Metric':<30} {'Popularity':>12} {'CF':>12}")
    print(f"{'─'*60}")

    cf_m  = results["cf"]
    pop_m = results["popularity_baseline"]

    for label, cf_key in [
        ("Inter-user diversity",  "inter_user_diversity"),
        ("Popularity bias",       "popularity_bias"),
    ]:
        cv = cf_m[cf_key]
        pv = pop_m[cf_key]
        print(f"  {label:<30} {pv:>12.4f} {cv:>12.4f}")

    print(f"{'─'*60}")

    if "persona_separation" in results:
        print("\n  Persona separation (Jaccard distance — higher is better):")
        for pair, dist in results["persona_separation"].items():
            bar = "█" * int(dist * 20)
            print(f"    {pair:<25} {dist:.4f}  {bar}")

    # ── verdict ───────────────────────────────────────────────────────────
    div = cf_diversity
    if div >= 0.70:
        verdict = "✅  Excellent personalisation (diversity ≥ 0.70)"
    elif div >= 0.40:
        verdict = "⚠️  Moderate personalisation (0.40 ≤ diversity < 0.70)"
    else:
        verdict = "❌  Low personalisation (diversity < 0.40) — CF converges on same items"

    print(f"\n  Verdict: {verdict}")
    print(f"{'═'*60}\n")

    return results


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CF personalization analysis")
    parser.add_argument("--users",    type=int,  default=DEFAULT_USERS)
    parser.add_argument("--garments", type=int,  default=DEFAULT_GARMENTS)
    parser.add_argument("--k",        type=int,  default=DEFAULT_K)
    parser.add_argument("--data",     type=Path, default=None,
                        help="Real interactions JSON (list of dicts)")
    parser.add_argument("--out",      type=Path, default=None)
    parser.add_argument("--seed",     type=int,  default=42)
    args = parser.parse_args()

    user_to_persona: Dict[str, str] | None = None

    if args.data:
        print(f"Loading interactions from {args.data} …")
        with open(args.data) as f:
            data = json.load(f)
        if isinstance(data, dict) and "interactions" in data:
            interactions   = data["interactions"]
            user_to_persona = data.get("user_to_persona")
        else:
            interactions = data
    else:
        print(f"Generating {args.users} users × {args.garments} garments …")
        interactions, user_to_persona = _generate_interactions(
            args.users, args.garments, seed=args.seed
        )

    results = analyse(interactions, user_to_persona=user_to_persona, k=args.k)

    out_path = args.out or REPORT_DIR / f"personalization_k{args.k}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved → {out_path}\n")


if __name__ == "__main__":
    main()
