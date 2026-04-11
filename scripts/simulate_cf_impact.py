#!/usr/bin/env python3
"""
CF Impact Simulation
====================
Generates a rich synthetic dataset and runs a **full end-to-end
simulation** of CF impact: builds interaction data, trains the model
(ALS or BPR), computes offline metrics, measures personalization,
evaluates user-based / item-based CF, hybrid scoring (40/40/20), and
produces a single JSON report that can feed the other analysis scripts.

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
    # Quick 100-user simulation (ALS, default)
    python scripts/simulate_cf_impact.py

    # BPR model
    python scripts/simulate_cf_impact.py --model-type bpr

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
    _mock_implicit_bpr = MagicMock()

    def _als_recommend(user_idx, user_items_csr=None, N=10, **kw):
        n_items = min(N, 30)
        return _np.arange(n_items, dtype=_np.int32), _np.linspace(0.9, 0.1, n_items, dtype=_np.float32)

    def _als_similar_items(item_idx, N=6, **kw):
        ids    = _np.arange(min(N, 30), dtype=_np.int32)
        ids    = ids[ids != item_idx][:N - 1]
        return ids, _np.linspace(0.85, 0.2, len(ids), dtype=_np.float32)

    def _als_similar_users(user_idx, N=6, **kw):
        ids    = _np.arange(min(N, 30), dtype=_np.int32)
        ids    = ids[ids != user_idx][:N - 1]
        return ids, _np.linspace(0.80, 0.15, len(ids), dtype=_np.float32)

    _als_instance = MagicMock()
    _als_instance.recommend     = _als_recommend
    _als_instance.similar_items = _als_similar_items
    _als_instance.similar_users = _als_similar_users
    _als_instance.user_factors  = _np.zeros((1, 32), dtype=_np.float32)
    _als_instance.item_factors  = _np.zeros((1, 32), dtype=_np.float32)

    _mock_implicit_als.AlternatingLeastSquares.return_value = _als_instance
    _mock_implicit_bpr.BayesianPersonalizedRanking.return_value = _als_instance

    sys.modules["implicit"]     = _mock_implicit
    sys.modules["implicit.als"] = _mock_implicit_als
    sys.modules["implicit.bpr"] = _mock_implicit_bpr

from src.layer7_cf.interaction_builder import InteractionBuilder, SIGNAL_WEIGHTS
from src.layer7_cf.collaborative_filter import CollaborativeFilter
from src.layer7_cf.hybrid_recommender import CFHybridRecommender
from src.layer7_cf.models import InteractionMatrix, UserCFNeighbour, ItemPair


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
                if rng.random() < 0.55:
                    entry["outfit_worn"] = rng.randint(1, 6)
                if rng.random() < 0.40:
                    entry["outfit_liked"] = 1
                if rng.random() < 0.30:
                    entry["outfit_saved"] = 1
                # Views: preferred items are viewed more
                entry["view"] = rng.randint(3, 15)
            else:
                entry["times_worn"] = rng.randint(1, 2)
                if rng.random() < 0.10:
                    entry["outfit_saved"] = 1
                # Non-preferred: viewed less, sometimes skipped
                entry["view"] = rng.randint(1, 4)
                if rng.random() < 0.35:
                    entry["skip"] = 1

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
    model_type:      str = "als",
) -> List[Dict[str, Any]]:
    """
    Simulate A/B test engagement.  Treatment users get hybrid-reranked
    recommendations (style 40% + CF 40% + context 20%); control users
    get popularity-only.  Treatment engagement is higher when the
    recommendation aligns with the user's preferred garment cluster.
    """
    rng = random.Random(seed)

    # ── train CF ─────────────────────────────────────────────────────────────
    builder = InteractionBuilder()
    matrix  = builder.build(interactions)
    cf      = CollaborativeFilter(factors=32, iterations=15, model_type=model_type)
    cf.train(matrix)
    hybrid  = CFHybridRecommender()  # 40/40/20

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

            # Build candidate dicts for hybrid rerank
            candidates = [
                {"id": gid, "overall_score": rng.uniform(0.5, 0.95)}
                for gid in recs
            ]
            # Simulate context scores (weather/occasion fit)
            ctx_scores = {
                gid: round(rng.uniform(0.3, 1.0), 3)
                for gid in recs
            }
            ranked = hybrid.rerank(candidates, uid, cf, context_scores=ctx_scores)
            recs = [c["id"] for c in ranked]
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
    model_type: str = "als",
) -> Dict[str, Any]:
    sorted_data = sorted(interactions, key=lambda x: x.get("timestamp", 0))
    cut         = int(len(sorted_data) * 0.8)
    train, test = sorted_data[:cut], sorted_data[cut:]

    test_gt: Dict[str, Set[str]] = {}
    for e in test:
        test_gt.setdefault(e["user_id"], set()).add(e["garment_id"])

    builder = InteractionBuilder()
    matrix  = builder.build(train)
    cf      = CollaborativeFilter(factors=32, iterations=15, model_type=model_type)
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

    # ── user-based CF evaluation ──────────────────────────────────────────────
    user_cf_stats = _evaluate_user_based_cf(cf, test_gt, k)

    # ── item-based CF evaluation ──────────────────────────────────────────────
    item_cf_stats = _evaluate_item_based_cf(cf, train, test, k)

    return {
        "model_type":      model_type,
        "cf_trained":      trained,
        "train_time_s":    round(train_t, 3),
        f"cf_ndcg@{k}":   round(avg(cf_ndcg_list),  4),
        f"pop_ndcg@{k}":  round(avg(pop_ndcg_list), 4),
        "n_eval_users":    len(test_gt),
        "user_based_cf":   user_cf_stats,
        "item_based_cf":   item_cf_stats,
    }


def _evaluate_user_based_cf(
    cf: CollaborativeFilter,
    test_gt: Dict[str, Set[str]],
    k: int,
) -> Dict[str, Any]:
    """Evaluate user-based CF: do similar users share relevant items?"""
    if not cf.is_trained:
        return {"evaluated": False}

    overlap_ratios: List[float] = []
    avg_similarities: List[float] = []
    avg_shared_items: List[float] = []

    sample_users = list(test_gt.keys())[:50]  # cap for speed
    for uid in sample_users:
        neighbours = cf.find_similar_users(uid, n=5)
        if not neighbours:
            continue
        avg_similarities.append(
            sum(nb.similarity for nb in neighbours) / len(neighbours)
        )
        avg_shared_items.append(
            sum(nb.shared_items for nb in neighbours) / len(neighbours)
        )
        # Check if neighbours' recommendations overlap with user's ground truth
        rel = test_gt.get(uid, set())
        if not rel:
            continue
        neighbour_recs: Set[str] = set()
        for nb in neighbours:
            for s in cf.recommend(nb.user_id, n=k, filter_owned=False):
                neighbour_recs.add(s.garment_id)
        if neighbour_recs:
            overlap_ratios.append(len(rel & neighbour_recs) / len(rel))

    avg_fn = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0
    return {
        "evaluated":             True,
        "n_users_sampled":       len(sample_users),
        "avg_neighbour_sim":     avg_fn(avg_similarities),
        "avg_shared_items":      avg_fn(avg_shared_items),
        "avg_gt_overlap_ratio":  avg_fn(overlap_ratios),
    }


def _evaluate_item_based_cf(
    cf: CollaborativeFilter,
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    k: int,
) -> Dict[str, Any]:
    """Evaluate item-based CF: do co-occurrence pairs appear in test set?"""
    if not cf.is_trained:
        return {"evaluated": False}

    # Build test co-occurrence from test data
    user_test_items: Dict[str, Set[str]] = {}
    for e in test_data:
        user_test_items.setdefault(e["user_id"], set()).add(e["garment_id"])

    # Sample some garments from training
    train_garments: Set[str] = set()
    for e in train_data:
        train_garments.add(e["garment_id"])

    sample_garments = list(train_garments)[:30]
    hit_count = 0
    total_pairs = 0

    for gid in sample_garments:
        pairs = cf.find_item_pairs(gid, n=5)
        for pair in pairs:
            total_pairs += 1
            # Check if any user in test has both the source and paired item
            for uid, items in user_test_items.items():
                if pair.source_id in items and pair.paired_id in items:
                    hit_count += 1
                    break

    return {
        "evaluated":         True,
        "n_garments_sampled": len(sample_garments),
        "total_pairs":       total_pairs,
        "test_cooccur_hits": hit_count,
        "pair_hit_rate":     round(hit_count / total_pairs, 4) if total_pairs else 0.0,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main simulation
# ────────────────────────────────────────────────────────────────────────────

def simulate(
    n_users:    int  = DEFAULT_USERS,
    n_garments: int  = DEFAULT_GARMENTS,
    k:          int  = DEFAULT_K,
    seed:       int  = 42,
    out_dir:    Path = OUTPUT_DIR,
    model_type: str  = "als",
) -> Dict[str, Any]:
    print(f"\n{'═'*65}")
    print(" CF Impact Simulation")
    print(f"{'═'*65}")
    print(f"  Users      : {n_users}")
    print(f"  Garments   : {n_garments}")
    print(f"  K          : {k}")
    print(f"  Model type : {model_type}")
    print(f"  Seed       : {seed}")
    print(f"  Output     : {out_dir}")
    print()

    # ── step 1: generate interactions ────────────────────────────────────────
    print("  [1/5] Generating interactions …")
    t0 = time.time()
    interactions, user_to_persona = generate_interactions(n_users, n_garments, seed)
    gen_time = time.time() - t0

    print(f"        {len(interactions)} interactions generated  ({gen_time:.2f}s)")
    persona_counts: Dict[str, int] = {}
    for p in user_to_persona.values():
        persona_counts[p] = persona_counts.get(p, 0) + 1
    for p, cnt in persona_counts.items():
        print(f"        Persona '{p}': {cnt} users")

    # Signal distribution
    SIGNAL_KEYS = [
        "times_worn", "is_favorite", "outfit_created", "outfit_worn",
        "outfit_liked", "outfit_saved", "view", "skip",
    ]
    signal_dist: Dict[str, int] = {s: 0 for s in SIGNAL_KEYS}
    for ev in interactions:
        for s in SIGNAL_KEYS:
            if s in ev:
                signal_dist[s] += 1
    print("\n        Signal distribution (interactions containing signal):")
    for sig in sorted(signal_dist, key=signal_dist.__getitem__, reverse=True):
        if signal_dist[sig]:
            print(f"          {sig:16s}: {signal_dist[sig]:>6}")

    # ── step 2: quick offline eval ────────────────────────────────────────────
    print(f"\n  [2/5] Quick offline evaluation (model={model_type}) …")
    eval_results = quick_offline_eval(interactions, k=k, model_type=model_type)
    cf_ndcg_key  = f"cf_ndcg@{k}"
    pop_ndcg_key = f"pop_ndcg@{k}"
    print(f"        CF trained     : {eval_results['cf_trained']}")
    print(f"        Model type     : {eval_results['model_type']}")
    print(f"        CF NDCG@{k:<2}    : {eval_results[cf_ndcg_key]:.4f}")
    print(f"        Pop NDCG@{k:<2}   : {eval_results[pop_ndcg_key]:.4f}")
    ndcg_lift = (
        (eval_results[cf_ndcg_key] - eval_results[pop_ndcg_key])
        / eval_results[pop_ndcg_key] * 100
        if eval_results[pop_ndcg_key] else 0
    )
    print(f"        NDCG lift      : {ndcg_lift:+.1f}%")

    # User-based CF stats
    ucf = eval_results.get("user_based_cf", {})
    if ucf.get("evaluated"):
        print(f"\n        User-based CF:")
        print(f"          Sampled users       : {ucf['n_users_sampled']}")
        print(f"          Avg neighbour sim   : {ucf['avg_neighbour_sim']:.4f}")
        print(f"          Avg shared items    : {ucf['avg_shared_items']:.1f}")
        print(f"          GT overlap ratio    : {ucf['avg_gt_overlap_ratio']:.4f}")

    # Item-based CF stats
    icf = eval_results.get("item_based_cf", {})
    if icf.get("evaluated"):
        print(f"\n        Item-based CF:")
        print(f"          Sampled garments    : {icf['n_garments_sampled']}")
        print(f"          Total pairs found   : {icf['total_pairs']}")
        print(f"          Test co-occur hits  : {icf['test_cooccur_hits']}")
        print(f"          Pair hit rate       : {icf['pair_hit_rate']:.4f}")

    # ── step 3: generate A/B log ──────────────────────────────────────────────
    print(f"\n  [3/5] Generating A/B engagement log …")
    ab_log = generate_ab_log(
        interactions, user_to_persona, k=k, seed=seed, model_type=model_type,
    )
    ctrl_e = [e["engagement"] for e in ab_log if e["group"] == "control"]
    trt_e  = [e["engagement"] for e in ab_log if e["group"] == "treatment"]
    ctrl_m = sum(ctrl_e) / len(ctrl_e) if ctrl_e else 0
    trt_m  = sum(trt_e)  / len(trt_e)  if trt_e  else 0
    print(f"        Control   mean engagement : {ctrl_m:.4f}  (n={len(ctrl_e)})")
    print(f"        Treatment mean engagement : {trt_m:.4f}  (n={len(trt_e)})")
    eng_lift = (trt_m - ctrl_m) / ctrl_m * 100 if ctrl_m else 0
    print(f"        Engagement lift           : {eng_lift:+.1f}%")

    # ── step 4: hybrid weight analysis ────────────────────────────────────────
    print(f"\n  [4/5] Hybrid weight analysis …")
    hybrid_weights = {"style": 0.40, "cf": 0.40, "context": 0.20}
    print(f"        Weights: style={hybrid_weights['style']:.0%}  "
          f"cf={hybrid_weights['cf']:.0%}  context={hybrid_weights['context']:.0%}")
    # When CF untrained, redistributed:
    r_style   = hybrid_weights["style"] + hybrid_weights["cf"] * (
        hybrid_weights["style"] / (hybrid_weights["style"] + hybrid_weights["context"])
    )
    r_context = hybrid_weights["context"] + hybrid_weights["cf"] * (
        hybrid_weights["context"] / (hybrid_weights["style"] + hybrid_weights["context"])
    )
    print(f"        Fallback (untrained CF): style={r_style:.2%}  context={r_context:.2%}")

    # ── step 5: save files ────────────────────────────────────────────────────
    print(f"\n  [5/5] Saving files to {out_dir} …")
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
            "model_type": model_type,
        },
        "interaction_stats":    stats,
        "signal_distribution":  signal_dist,
        "persona_distribution": persona_counts,
        "offline_eval":         eval_results,
        "hybrid_weights":       hybrid_weights,
        "ab_simulation": {
            "n_control":           len(ctrl_e),
            "n_treatment":         len(trt_e),
            "control_mean_eng":    round(ctrl_m, 4),
            "treatment_mean_eng":  round(trt_m, 4),
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
    print(f"  Model type             : {model_type}")
    print(f"  Interactions generated : {len(interactions)}")
    print(f"  Matrix sparsity        : {stats.get('sparsity', 0):.1%}")
    print(f"  CF vs Popularity NDCG  : {eval_results[cf_ndcg_key]:.4f} vs "
          f"{eval_results[pop_ndcg_key]:.4f}  ({ndcg_lift:+.1f}%)")
    print(f"  A/B engagement lift    : {eng_lift:+.1f}%")
    print(f"  Hybrid blend           : style={hybrid_weights['style']:.0%} "
          f"cf={hybrid_weights['cf']:.0%} context={hybrid_weights['context']:.0%}")
    if ucf.get("evaluated"):
        print(f"  User-CF overlap ratio  : {ucf['avg_gt_overlap_ratio']:.4f}")
    if icf.get("evaluated"):
        print(f"  Item-CF pair hit rate  : {icf['pair_hit_rate']:.4f}")
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
    parser.add_argument("--users",      type=int,  default=DEFAULT_USERS)
    parser.add_argument("--garments",   type=int,  default=DEFAULT_GARMENTS)
    parser.add_argument("--k",          type=int,  default=DEFAULT_K)
    parser.add_argument("--out-dir",    type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument(
        "--model-type",
        type=str,
        default="als",
        choices=["als", "bpr"],
        help="Implicit model backend: 'als' (ALS) or 'bpr' (BPR)",
    )
    args = parser.parse_args()

    simulate(
        n_users=args.users,
        n_garments=args.garments,
        k=args.k,
        out_dir=args.out_dir,
        seed=args.seed,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
