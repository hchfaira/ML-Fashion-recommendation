"""ShoppingListOptimizer — turn wardrobe gaps into a budget-sorted shopping list.

The optimizer wraps ``MissingPiecesRecommender`` and re-ranks its output by
ROI (outfits unlocked per euro), then filters items to fit inside the user's
budget.

Optimisation notes
------------------
* Re-ranking is O(n log n) sort — negligible.
* Price lookup is a simple dict — O(1) per item.
* Urgency is driven by gap count and ROI: high-urgency items are those that
  fill a hard gap (required_coverage) **and** have ROI above the median.
"""
from __future__ import annotations

import logging
from statistics import median
from typing import Dict, List, Optional

from config import get_config
from src.core.models import Garment
from src.core.travel_models import ShoppingItem, ShoppingListResult
from src.layer2_style.capsule.missing_pieces_recommender import MissingPiecesRecommender
from src.layer2_style.capsule.wardrobe_capsule_analyzer import WardrobeCapsuleAnalyzer

logger = logging.getLogger(__name__)

_DEFAULT_PRICE_MID = 80.0


class ShoppingListOptimizer:
    """Produce a budget-sorted shopping list from wardrobe gaps.

    Parameters
    ----------
    config_path:
        Optional override for ``travel_data.json``.

    Usage
    -----
    ::

        optimizer = ShoppingListOptimizer()
        result = optimizer.optimize(garments, budget=300, season="fall", body_shape="pear")
    """

    def __init__(self, config_path=None):
        cfg = get_config()
        travel_data = cfg.get_data("travel_data", default={})
        self._price_estimates: Dict = travel_data.get("price_estimates_eur", {})
        self._capsule_analyzer = WardrobeCapsuleAnalyzer()
        self._missing_recommender = MissingPiecesRecommender()
        logger.debug("ShoppingListOptimizer initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        garments: List[Garment],
        budget: float,
        season: str = "fall",
        body_shape: str = "rectangle",
        price_tier: str = "mid",
        hard_gaps: Optional[List[str]] = None,
    ) -> ShoppingListResult:
        """Generate a shopping list ranked by ROI.

        Parameters
        ----------
        garments:
            Current wardrobe.
        budget:
            Total budget in EUR.
        season:
            Target season (``spring`` | ``summer`` | ``fall`` | ``winter``).
        body_shape:
            User body shape for MissingPiecesRecommender.
        price_tier:
            ``budget`` | ``mid`` | ``luxury`` — selects price point from config.
        hard_gaps:
            Category names that are absolutely required (e.g. from SeasonCoverage gaps).
            These get an urgency boost.

        Returns
        -------
        ShoppingListResult
        """
        hard_gaps = hard_gaps or []

        # 1. Analyse current wardrobe
        analysis = self._capsule_analyzer.analyze(garments)

        # 2. Get missing piece recommendations
        missing = self._missing_recommender.recommend(
            analysis=analysis,
            user_season=season,
            body_shape=body_shape,
        )
        recommendations = missing.recommendations if missing and missing.recommendations else []

        # 3. Build shopping items with price and ROI
        items = self._build_items(recommendations, price_tier, hard_gaps, len(garments))

        # 4. Sort by ROI descending, then by urgency
        items.sort(key=lambda i: (-_urgency_rank(i.urgency), -i.roi))

        # 5. Re-rank
        for rank, item in enumerate(items, start=1):
            item.rank = rank

        # 6. Split into within / over budget
        within_budget: List[ShoppingItem] = []
        over_budget: List[ShoppingItem] = []
        running = 0.0
        for item in items:
            if running + item.estimated_price_eur <= budget:
                within_budget.append(item)
                running += item.estimated_price_eur
            else:
                over_budget.append(item)

        total_projected = sum(i.outfits_unlocked for i in within_budget)

        return ShoppingListResult(
            budget_eur=budget,
            items=items,
            total_estimated_cost=round(running, 2),
            projected_new_outfits=total_projected,
            within_budget=within_budget,
            over_budget=over_budget,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_items(
        self,
        recommendations,
        price_tier: str,
        hard_gaps: List[str],
        wardrobe_size: int,
    ) -> List[ShoppingItem]:
        items: List[ShoppingItem] = []

        # Compute ROI median for urgency thresholding (done after building raw list)
        raw: List[Dict] = []
        for rec in recommendations:
            cat = (rec.category or "top").lower()
            price = self._get_price(cat, price_tier)
            if price <= 0:
                continue

            outfits_unlocked = getattr(rec, "impact_outfits", 0) or 0
            roi = (outfits_unlocked / price * 100) if price else 0.0
            colors = getattr(rec, "suggested_colors", []) or []

            raw.append(
                dict(
                    category=cat,
                    description=getattr(rec, "description", cat),
                    estimated_price_eur=price,
                    outfits_unlocked=outfits_unlocked,
                    roi=roi,
                    color_suggestions=colors[:4],
                    reason=getattr(rec, "llm_narration", "") or "",
                    hard_gap=(cat in hard_gaps),
                )
            )

        # Median ROI for urgency thresholding
        roi_values = [r["roi"] for r in raw]
        med_roi = median(roi_values) if roi_values else 0.0

        for rank, r in enumerate(raw, start=1):
            is_hard = r["hard_gap"]
            above_median = r["roi"] >= med_roi

            if is_hard and above_median:
                urgency = "high"
            elif is_hard or above_median:
                urgency = "medium"
            else:
                urgency = "low"

            items.append(
                ShoppingItem(
                    rank=rank,
                    category=r["category"],
                    description=r["description"],
                    estimated_price_eur=r["estimated_price_eur"],
                    outfits_unlocked=r["outfits_unlocked"],
                    roi=round(r["roi"], 2),
                    urgency=urgency,
                    color_suggestions=r["color_suggestions"],
                    reason=r["reason"],
                )
            )

        return items

    def _get_price(self, category: str, tier: str) -> float:
        cat_prices = self._price_estimates.get(category, {})
        if not cat_prices:
            # fallback: try without plural 's'
            cat_prices = self._price_estimates.get(category.rstrip("s"), {})
        price = cat_prices.get(tier, cat_prices.get("mid", _DEFAULT_PRICE_MID))
        return float(price)


def _urgency_rank(urgency: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(urgency, 0)
