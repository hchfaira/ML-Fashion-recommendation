#!/usr/bin/env python3
"""
CF Impact Simulation
====================
Generates a rich synthetic dataset and runs a **full end-to-end
simulation** of CF impact: builds interaction data, trains the model,
computes offline metrics, measures personalization, and produces a
single JSON report that can feed the other analysis scripts.

This is the **data generation hub** — run this first, then pass its
output to the other CF analysis scripts.

Outputs
-------
``tests/output/cf_impact/``
├── interactions.json          Raw interaction list (feed to evaluate_cf_offline.py etc.)
├── ab_log.json                Simulated A/B engagement log (feed to run_ab_test_analysis.py)
└── simulation_report.json     Full combined report

Usage
-----
    # Quick 100-user simulation
    python scripts/simulate_cf_impact.py

    # Larger run
    python scripts/simulate_cf_impact.py --users 300 --garments 100

    # Custom output directory
    python scripts/simulate_cf_impact.py --out-dir results/simulation/

    # Change K
    python scripts/simulate_cf_impact.py --k 10
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

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Inject implicit mock if not installed (CI / dev without C++ build tools) ──
import importlib.util as _ilu
if _ilu.find_spec("implicit") is None:
    from unittest.mock import MagicMock
    import numpy as _np

    _mock_implicit = MagicMock()
    _mock_implicit_als = MagicMock()

    def _als_recommend(user_idx, user_items_csr=None, N=10, **kw):
        n_items = min(N, 30)
        return _np.arange(n_items, dtype=_np.int32), _np.linspace(0.9, 0.1, n_items, dtype=_np.float32)

    def _als_similar_items(item_idx, N=6, **kw):
        ids    = _np.arange(min(N, 30), dtype=_np.int32)
        ids    = ids[ids != item_idx][:N - 1]
        return ids, _np.linspace(0.85, 0.2, len(ids), dtype=_np.float32)

    _als_instance = MagicMock()
    _als_instance.recommend     = _als_recommend
    _als_instance.similar_items = _als_similar_items
    _als_instance.user_factors  = _np.zeros((1, 32), dtype=_np.float32)
    _als_instance.item_factors  = _np.zeros((1, 32), dtype=_np.float32)
    _mock_implicit_als.AlternatingLeastSquares.return_value = _als_instance
    sys.modules["implicit"]     = _mock_implicit
    sys.modules["implicit.als"] = _mock_implicit_als

from src.layer7_cf.interaction_builder import InteractionBuilder, SIGNAL_WEIGHTS
from src.layer7_cf.collaborative_filter import CollaborativeFilter
from src.layer7_cf.hybrid_recommender import CFHybridRecommender
from src.layer7_cf.models import InteractionMatrix


# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_USERS    = 100
DEFAULT_GARMENTS = 50
DEFAULT_K        = 10
OUTPUT_DIR = Path("tests/output/cf_impact")

# User personas: each defines which garment indices they prefer
PERSONAS: Dict[str, Dict[str, Any]] = {
    "casual": {
        "description": "Everyday comfortable clothing",
        "preferred_range": (0, 15),
        "interaction_density": 0.70,
        "wear_frequency": (3, 12),
    },
    "formal": {
        "description": "Professional and business wear",
        "preferred_range": (15, 30),
        "interaction_density": 0.60,
        "wear_frequency": (2, 8),
    },
    "minimalist": {
        "description": "Clean lines, neutral palette",
        "preferred_range": (30, 40),
        "interaction_density": 0.50,
        "wear_frequency": (5, 15),
    },
    "eclectic": {
        "description": "Bold, experimental fashion",
        "preferred_range": (0, 50),   # spread across all
        "interaction_density": 0.30,
        "wear_frequency": (1, 6),
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Interaction generator
# ────────────────────────────────────────────────────────────────────────────

def generate_interactions(
    n_users:    int = DEFAULT_USERS,
    n_garments: int = DEFAULT_GARMENTS,
    seed:       int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Returns (interactions, user_to_persona).

    Each user is assigned a persona that shapes their garment preference
    distribution.  Interactions contain multiple CF signals so the
    InteractionBuilder can weight them appropriately.
    """
    rng      = random.Random(seed)
    garments = [f"g{i:03d}" for i in range(n_garments)]
    persona_list = list(PERSONAS.keys())
    interactions: List[Dict[str, Any]] = []
    user_to_persona: Dict[str, str] = {}

    timestamp = 0
    for u in range(n_users):
        user_id = f"user_{u:04d}"
        persona = persona_list[u % len(persona_list)]
        user_to_persona[user_id] = persona
        cfg = PERSONAS[persona]

        lo, hi = cfg["preferred_range"]
        hi     = min(hi, n_garments)
        pref_pool = garments[lo:hi] if hi > lo else garments
        other_pool = [g for g in garments if g not in set(pref_pool)]

        density = cfg["interaction_density"]
        wear_lo, wear_hi = cfg["wear_frequency"]

        for g in garments:
            is_preferred = g in set(pref_pool)
            p = density if is_preferred else 0.10
            if rng.random() > p:
                continue

            entry: Dict[str, Any] = {
                "user_id":    user_id,
                "garment_id": g,
                "timestamp":  timestamp,
            }
            timestamp += 1

            # Signal richness proportional to preference
            if is_preferred:
                entry["times_worn"]   = rng.randint(wear_lo, wear_hi)
                if rng.random() < 0.60:
                    entry["is_favorite"] = 1
                if rng.random() < 0.50:
                    entry["outfit_created"] = 1
                if rng.random() < 0.40:
                    entry["outfit_liked"] = 1
                if rng.random() < 0.30:
                    entry["outfit_saved"] = 1
            else:
                entry["times_worn"] = rng.randint(1, 2)
                if rng.random() < 0.10:
                    entry["outfit_saved"] = 1

            interactions.append(entry)

    return interactions, user_to_persona


# ────────────────────────────────────────────────────────────────────────────
# A/B log generator (uses CF recommendations vs. popularity)
# ────────────────────────────────────────────────────────────────────────────

def generate_ab_log(
    interactions:    List[Dict[str, Any]],
    user_to_persona: Dict[str, str],
    k:               int = DEFAULT_K,
    seed:            int = 42,
) -> List[Dict[str, Any]]:
    """
    Simulate A/B test engagement.  Treatment users get CF-personalised
    recommendations; control users get popularity-only.  Treatment
    engagement is higher when the recommendation aligns with the user's
    preferred garment cluster.
    """
    rng = random.Random(seed)

    # ── train CF ─────────────────────────────────────────────────────────────
    builder = InteractionBuilder()
    matrix  = builder.build(interactions)
    cf      = CollaborativeFilter(factors=32, iterations=15)
    cf.train(matrix)

    # ── popularity baseline ──────────────────────────────────────────────────
    counts: Dict[str, int] = {}
    for entry in interactions:
        gid = entry["garment_id"]
        counts[gid] = counts.get(gid, 0) + 1
    pop_list = sorted(counts, key=counts.__getitem__, reverse=True)[:k]
    pop_set  = set(pop_list)

    # ── per-user preferred garments (ground-truth) ────────────────────────────
    user_preferred: Dict[str, Set[str]] = {}
    for entry in interactions:
        uid = entry["user_id"]
        if entry.get("is_favorite") or (entry.get("times_worn", 0) or 0) >= 4:
            user_preferred.setdefault(uid, set()).add(entry["garment_id"])

    # ── generate log ──────────────────────────────────────────────────────────
    log: List[Dict[str, Any]] = []
    users = list(matrix.user_ids)

    for i, uid in enumerate(users):
        group = "control" if i % 2 == 0 else "treatment"
        pref  = user_preferred.get(uid, set())

        if group == "treatment":
            scores = cf.recommend(uid, n=k, filter_owned=False)
            recs   = [s.garment_id for s in scores] if scores else pop_list
        else:
            recs = pop_list

        # Engagement: higher when recommendation overlaps with preferences
        if pref:
            overlap = len(set(recs) & pref)
            base_engagement = 0.40 + 0.40 * (overlap / min(k, len(pref) + 1))
        else:
            base_engagement = 0.40

        engagement = max(0.0, min(1.0, rng.gauss(base_engagement, 0.12)))

        log.append({
            "user_id":    uid,
            "group":      group,
            "engagement": round(engagement, 4),
            "clicked":    1 if engagement > 0.55 else 0,
            "saved":      1 if engagement > 0.75 else 0,
            "persona":    user_to_persona.get(uid, "unknown"),
            "recs":       recs[:5],  # preview (not needed for stats)
        })

    return log


# ────────────────────────────────────────────────────────────────────────────
# Offline evaluation (inline — reuses the same logic as evaluate_cf_offline)
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


def quick_offline_eval(
    interactions: List[Dict[str, Any]],
    k: int = DEFAULT_K,
) -> Dict[str, Any]:
    sorted_data = sorted(interactions, key=lambda x: x.get("timestamp", 0))
    cut         = int(len(sorted_data) * 0.8)
    train, test = sorted_data[:cut], sorted_data[cut:]

    test_gt: Dict[str, Set[str]] = {}
    for e in test:
        test_gt.setdefault(e["user_id"], set()).add(e["garment_id"])

    builder = InteractionBuilder()
    matrix  = builder.build(train)
    cf      = CollaborativeFilter(factors=32, iterations=15)
    t0      = time.time()
    trained = cf.train(matrix)
    train_t = time.time() - t0

    pop_counts: Dict[str, int] = {}
    for e in train:
        gid = e["garment_id"]
        pop_counts[gid] = pop_counts.get(gid, 0) + 1
    pop_recs = sorted(pop_counts, key=pop_counts.__getitem__, reverse=True)[:k]

    cf_ndcg_list, pop_ndcg_list = [], []
    for uid, rel in test_gt.items():
        cf_recs = [s.garment_id for s in cf.recommend(uid, n=k, filter_owned=False)]
        cf_ndcg_list.append(_ndcg_at_k(cf_recs,  rel, k))
        pop_ndcg_list.append(_ndcg_at_k(pop_recs, rel, k))

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    return {
        "cf_trained":      trained,
        "train_time_s":    round(train_t, 3),
        f"cf_ndcg@{k}":   round(avg(cf_ndcg_list),  4),
        f"pop_ndcg@{k}":  round(avg(pop_ndcg_list), 4),
        "n_eval_users":    len(test_gt),
    }


# ────────────────────────────────────────────────────────────────────────────
# Main simulation
# ────────────────────────────────────────────────────────────────────────────

def simulate(
    n_users:    int = DEFAULT_USERS,
    n_garments: int = DEFAULT_GARMENTS,
    k:          int = DEFAULT_K,
    seed:       int = 42,
    out_dir:    Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    print(f"\n{'═'*65}")
    print(" CF Impact Simulation")
    print(f"{'═'*65}")
    print(f"  Users    : {n_users}")
    print(f"  Garments : {n_garments}")
    print(f"  K        : {k}")
    print(f"  Seed     : {seed}")
    print(f"  Output   : {out_dir}")
    print()

    # ── step 1: generate interactions ────────────────────────────────────────
    print("  [1/4] Generating interactions …")
    t0 = time.time()
    interactions, user_to_persona = generate_interactions(n_users, n_garments, seed)
    gen_time = time.time() - t0

    print(f"        {len(interactions)} interactions generated  ({gen_time:.2f}s)")
    persona_counts = {}
    for p in user_to_persona.values():
        persona_counts[p] = persona_counts.get(p, 0) + 1
    for p, cnt in persona_counts.items():
        print(f"        Persona '{p}': {cnt} users")

    # ── step 2: quick offline eval ────────────────────────────────────────────
    print("\n  [2/4] Quick offline evaluation …")
    eval_results = quick_offline_eval(interactions, k=k)
    cf_ndcg_key  = f"cf_ndcg@{k}"
    pop_ndcg_key = f"pop_ndcg@{k}"
    print(f"        CF trained     : {eval_results['cf_trained']}")
    print(f"        CF NDCG@{k:<2}    : {eval_results[cf_ndcg_key]:.4f}")
    print(f"        Pop NDCG@{k:<2}   : {eval_results[pop_ndcg_key]:.4f}")
    ndcg_lift = (
        (eval_results[cf_ndcg_key] - eval_results[pop_ndcg_key])
        / eval_results[pop_ndcg_key] * 100
        if eval_results[pop_ndcg_key] else 0
    )
    print(f"        NDCG lift      : {ndcg_lift:+.1f}%")

    # ── step 3: generate A/B log ──────────────────────────────────────────────
    print("\n  [3/4] Generating A/B engagement log …")
    ab_log = generate_ab_log(interactions, user_to_persona, k=k, seed=seed)
    ctrl_e = [e["engagement"] for e in ab_log if e["group"] == "control"]
    trt_e  = [e["engagement"] for e in ab_log if e["group"] == "treatment"]
    ctrl_m = sum(ctrl_e) / len(ctrl_e) if ctrl_e else 0
    trt_m  = sum(trt_e)  / len(trt_e)  if trt_e  else 0
    print(f"        Control   mean engagement : {ctrl_m:.4f}  (n={len(ctrl_e)})")
    print(f"        Treatment mean engagement : {trt_m:.4f}  (n={len(trt_e)})")
    eng_lift = (trt_m - ctrl_m) / ctrl_m * 100 if ctrl_m else 0
    print(f"        Engagement lift           : {eng_lift:+.1f}%")

    # ── step 4: save files ────────────────────────────────────────────────────
    print(f"\n  [4/4] Saving files to {out_dir} …")
    out_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = out_dir / "interactions.json"
    with open(interactions_path, "w") as f:
        json.dump(interactions, f, indent=2)
    print(f"        Saved → {interactions_path}")

    ab_log_path = out_dir / "ab_log.json"
    with open(ab_log_path, "w") as f:
        json.dump(ab_log, f, indent=2)
    print(f"        Saved → {ab_log_path}")

    # Build combined report
    builder = InteractionBuilder()
    matrix  = builder.build(interactions)
    stats   = builder.stats(matrix)

    report: Dict[str, Any] = {
        "config": {
            "n_users":    n_users,
            "n_garments": n_garments,
            "k":          k,
            "seed":       seed,
        },
        "interaction_stats":  stats,
        "persona_distribution": persona_counts,
        "offline_eval":       eval_results,
        "ab_simulation": {
            "n_control":          len(ctrl_e),
            "n_treatment":        len(trt_e),
            "control_mean_eng":   round(ctrl_m, 4),
            "treatment_mean_eng": round(trt_m, 4),
            "engagement_lift_pct": round(eng_lift, 2),
        },
        "files": {
            "interactions": str(interactions_path),
            "ab_log":       str(ab_log_path),
        },
    }

    report_path = out_dir / "simulation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"        Saved → {report_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(" Summary")
    print(f"{'─'*65}")
    print(f"  Interactions generated : {len(interactions)}")
    print(f"  Matrix sparsity        : {stats.get('sparsity', 0):.1%}")
    print(f"  CF vs Popularity NDCG  : {eval_results[cf_ndcg_key]:.4f} vs "
          f"{eval_results[pop_ndcg_key]:.4f}  ({ndcg_lift:+.1f}%)")
    print(f"  A/B engagement lift    : {eng_lift:+.1f}%")
    print(f"\n  Next steps:")
    print(f"    python scripts/evaluate_cf_offline.py --data {interactions_path}")
    print(f"    python scripts/measure_personalization.py --data {interactions_path}")
    print(f"    python scripts/cf_cold_start_analysis.py --data {interactions_path}")
    print(f"    python scripts/run_ab_test_analysis.py --log {ab_log_path}")
    print(f"{'═'*65}\n")

    return report


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CF impact simulation data"
    )
    parser.add_argument("--users",    type=int,  default=DEFAULT_USERS)
    parser.add_argument("--garments", type=int,  default=DEFAULT_GARMENTS)
    parser.add_argument("--k",        type=int,  default=DEFAULT_K)
    parser.add_argument("--out-dir",  type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed",     type=int,  default=42)
    args = parser.parse_args()

    simulate(
        n_users=args.users,
        n_garments=args.garments,
        k=args.k,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
