#!/usr/bin/env python3
"""
CF A/B Test Analysis
====================
Reads simulated (or real) A/B test log data, splits results into
**control** (style-only) vs **treatment** (CF-boosted) groups, then
runs statistical significance tests to determine whether CF produces
a measurable improvement.

Statistical Tests
-----------------
* **Welch's t-test** (`scipy.stats.ttest_ind(equal_var=False)`)
  — Tests whether mean engagement differs between groups.  Does NOT
  assume equal variance (appropriate for unequal group sizes).

* **Mann-Whitney U test** (`scipy.stats.mannwhitneyu`)
  — Non-parametric alternative when engagement scores are not normally
  distributed (ordinal ratings, click counts, etc.)

* **Effect size (Cohen's d)** — Practical significance beyond p-value.
  |d| < 0.2 → negligible, 0.2–0.5 → small, 0.5–0.8 → medium, > 0.8 → large.

A/B Log Schema
--------------
Each entry in the log is a dict with at minimum:

    {
        "user_id":    "user_042",
        "group":      "treatment",          # "control" or "treatment"
        "engagement": 0.73,                 # float in [0, 1]
        "clicked":    1,                    # optional binary
        "saved":      0                     # optional binary
    }

Usage
-----
    # Generate synthetic A/B data and analyse
    python scripts/run_ab_test_analysis.py

    # Load a real log file
    python scripts/run_ab_test_analysis.py --log tests/output/ab_log.json

    # Save report
    python scripts/run_ab_test_analysis.py --out results/ab_analysis.json

    # Required sample size estimator
    python scripts/run_ab_test_analysis.py --power-calc --effect 0.2 --alpha 0.05
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import scipy.stats as stats  # noqa: E402  (scipy is a project dep)

# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_N_USERS   = 200
ALPHA             = 0.05        # significance threshold
REPORT_DIR = Path("tests/output/cf_ab_test")


# ────────────────────────────────────────────────────────────────────────────
# Synthetic A/B data generator
# ────────────────────────────────────────────────────────────────────────────

def _generate_ab_log(
    n_users:            int   = DEFAULT_N_USERS,
    treatment_lift:     float = 0.08,   # expected engagement lift for treatment
    base_engagement:    float = 0.50,
    noise:              float = 0.20,
    seed:               int   = 42,
) -> List[Dict[str, Any]]:
    """
    Simulate A/B logs.  Half of users go to control, half to treatment.
    Treatment group has a slightly higher mean engagement.
    """
    rng  = random.Random(seed)
    log: List[Dict[str, Any]] = []

    for u in range(n_users):
        user_id = f"user_{u:04d}"
        group   = "control" if u % 2 == 0 else "treatment"
        mean_e  = base_engagement + (treatment_lift if group == "treatment" else 0.0)
        # Gaussian engagement score clipped to [0, 1]
        engagement = max(0.0, min(1.0, rng.gauss(mean_e, noise)))
        clicked    = 1 if engagement > 0.55 else 0
        saved      = 1 if engagement > 0.75 else 0

        log.append({
            "user_id":    user_id,
            "group":      group,
            "engagement": round(engagement, 4),
            "clicked":    clicked,
            "saved":      saved,
        })

    return log


# ────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ────────────────────────────────────────────────────────────────────────────

def _cohens_d(a: List[float], b: List[float]) -> float:
    """Effect size: difference in means / pooled std dev."""
    n_a, n_b   = len(a), len(b)
    mean_a, mean_b = sum(a) / n_a, sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1) if n_a > 1 else 0
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1) if n_b > 1 else 0
    pooled_std = math.sqrt((var_a + var_b) / 2)
    return (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0


def _confidence_interval(
    values: List[float],
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """95% CI using t-distribution."""
    n    = len(values)
    mean = sum(values) / n
    se   = math.sqrt(sum((x - mean) ** 2 for x in values) / (n * (n - 1))) if n > 1 else 0.0
    t    = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return (mean - t * se, mean + t * se)


def _power_calc(
    effect_size: float = 0.2,
    alpha:       float = 0.05,
    power:       float = 0.80,
) -> int:
    """Estimate required sample size per group (two-sample t-test)."""
    # Approximation via z-scores
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    n = ((z_alpha + z_power) / effect_size) ** 2
    return math.ceil(n)


# ────────────────────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────────────────────

def analyse(
    log: List[Dict[str, Any]],
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    print(f"\n{'═'*65}")
    print(" CF A/B Test Analysis")
    print(f"{'═'*65}")
    print(f"  Total log entries : {len(log)}")

    # ── split groups ─────────────────────────────────────────────────────────
    control_eng:   List[float] = []
    treatment_eng: List[float] = []
    control_click: List[float] = []
    treatment_click: List[float] = []

    for entry in log:
        group = entry.get("group", "")
        eng   = float(entry.get("engagement", 0))
        click = float(entry.get("clicked",   0))

        if group == "control":
            control_eng.append(eng)
            control_click.append(click)
        elif group == "treatment":
            treatment_eng.append(eng)
            treatment_click.append(click)

    n_ctrl = len(control_eng)
    n_trt  = len(treatment_eng)

    if n_ctrl < 2 or n_trt < 2:
        print("  ⚠️  Not enough entries in each group for statistical testing.")
        return {}

    print(f"  Control group    : {n_ctrl} users")
    print(f"  Treatment group  : {n_trt} users")

    # ── engagement metrics ───────────────────────────────────────────────────
    mean_ctrl = sum(control_eng) / n_ctrl
    mean_trt  = sum(treatment_eng) / n_trt
    lift_pct  = (mean_trt - mean_ctrl) / mean_ctrl * 100 if mean_ctrl else 0.0

    ci_ctrl = _confidence_interval(control_eng, alpha=alpha)
    ci_trt  = _confidence_interval(treatment_eng, alpha=alpha)

    # ── significance tests ───────────────────────────────────────────────────
    t_stat, p_value_t = stats.ttest_ind(treatment_eng, control_eng, equal_var=False)
    u_stat, p_value_u = stats.mannwhitneyu(
        treatment_eng, control_eng, alternative="greater"
    )
    d = _cohens_d(treatment_eng, control_eng)

    # ── CTR (click-through rate) ─────────────────────────────────────────────
    ctr_ctrl = sum(control_click)   / n_ctrl if n_ctrl else 0
    ctr_trt  = sum(treatment_click) / n_trt  if n_trt  else 0
    ctr_lift = (ctr_trt - ctr_ctrl) / ctr_ctrl * 100 if ctr_ctrl else 0.0

    # ── build results ─────────────────────────────────────────────────────────
    sig_t = bool(p_value_t < alpha)
    sig_u = bool(p_value_u < alpha)

    def _d_magnitude(d: float) -> str:
        ad = abs(d)
        if ad < 0.2: return "negligible"
        if ad < 0.5: return "small"
        if ad < 0.8: return "medium"
        return "large"

    results: Dict[str, Any] = {
        "alpha": alpha,
        "n_control": n_ctrl,
        "n_treatment": n_trt,
        "engagement": {
            "control_mean":   round(mean_ctrl, 4),
            "treatment_mean": round(mean_trt, 4),
            "lift_pct":       round(lift_pct, 2),
            "control_ci_95":  [round(ci_ctrl[0], 4), round(ci_ctrl[1], 4)],
            "treatment_ci_95":[round(ci_trt[0], 4),  round(ci_trt[1], 4)],
        },
        "ctr": {
            "control":   round(ctr_ctrl, 4),
            "treatment": round(ctr_trt,  4),
            "lift_pct":  round(ctr_lift, 2),
        },
        "welch_t_test": {
            "t_stat":    round(t_stat,   4),
            "p_value":   round(p_value_t, 6),
            "significant": sig_t,
        },
        "mann_whitney_u_test": {
            "u_stat":    round(u_stat,   4),
            "p_value":   round(p_value_u, 6),
            "significant": sig_u,
        },
        "cohens_d": round(d, 4),
        "effect_magnitude": _d_magnitude(d),
        "recommendation": "",
    }

    # ── print table ───────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'Metric':<30} {'Control':>12} {'Treatment':>12} {'Lift':>8}")
    print(f"{'─'*65}")
    print(f"  {'Engagement (mean)':<30} {mean_ctrl:>12.4f} {mean_trt:>12.4f} {lift_pct:>+7.1f}%")
    print(f"  {'CTR':<30} {ctr_ctrl:>12.4f} {ctr_trt:>12.4f} {ctr_lift:>+7.1f}%")
    print(f"{'─'*65}")
    print(f"\n  Statistical Tests:")
    print(f"  Welch's t-test  : t={t_stat:.4f},  p={p_value_t:.6f}  "
          f"{'✅ significant' if sig_t else '❌ not significant'} (α={alpha})")
    print(f"  Mann-Whitney U  : U={u_stat:.0f},    p={p_value_u:.6f}  "
          f"{'✅ significant' if sig_u else '❌ not significant'} (α={alpha})")
    print(f"  Cohen's d       : {d:.4f}  ({_d_magnitude(d)} effect)")

    # ── confidence intervals ──────────────────────────────────────────────────
    print(f"\n  95% CI (engagement):")
    print(f"    Control   : [{ci_ctrl[0]:.4f}, {ci_ctrl[1]:.4f}]")
    print(f"    Treatment : [{ci_trt[0]:.4f},  {ci_trt[1]:.4f}]")
    ci_overlap = ci_trt[0] < ci_ctrl[1]
    print(f"    Overlap   : {'Yes (inconclusive)' if ci_overlap else 'No (separated intervals — strong signal)'}")

    # ── recommendation ────────────────────────────────────────────────────────
    both_sig = sig_t and sig_u
    if both_sig and d > 0 and lift_pct > 0:
        rec = "✅  DEPLOY CF — both tests significant, positive effect."
    elif both_sig and d > 0 and lift_pct <= 0:
        rec = "⚠️  Statistically significant but lift is negative — INVESTIGATE."
    elif not both_sig and abs(lift_pct) < 3.0:
        rec = "⚠️  Inconclusive — increase sample size or run longer."
    else:
        rec = "❌  No significant improvement — CF may not be ready."

    results["recommendation"] = rec
    print(f"\n  Recommendation: {rec}")
    print(f"{'═'*65}\n")

    return results


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CF A/B test statistical analysis")
    parser.add_argument("--log",     type=Path, default=None,
                        help="Path to A/B log JSON (list of entries with 'group', 'engagement')")
    parser.add_argument("--users",   type=int,  default=DEFAULT_N_USERS,
                        help="Synthetic user count (if no --log)")
    parser.add_argument("--lift",    type=float, default=0.08,
                        help="Expected treatment lift for synthetic data")
    parser.add_argument("--out",     type=Path, default=None)
    parser.add_argument("--alpha",   type=float, default=ALPHA)
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--power-calc", action="store_true",
                        help="Print required sample size and exit")
    parser.add_argument("--effect",  type=float, default=0.2,
                        help="Expected Cohen's d for --power-calc")
    args = parser.parse_args()

    if args.power_calc:
        n = _power_calc(effect_size=args.effect, alpha=args.alpha)
        print(f"\n  Required sample size per group: {n}")
        print(f"  (α={args.alpha}, power=0.80, effect={args.effect})\n")
        return

    if args.log:
        print(f"Loading A/B log from {args.log} …")
        with open(args.log) as f:
            log = json.load(f)
    else:
        print(f"Generating synthetic A/B log ({args.users} users, lift={args.lift:.0%}) …")
        log = _generate_ab_log(
            n_users=args.users,
            treatment_lift=args.lift,
            seed=args.seed,
        )

    results = analyse(log, alpha=args.alpha)

    out_path = args.out or REPORT_DIR / "ab_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved → {out_path}\n")


if __name__ == "__main__":
    main()
