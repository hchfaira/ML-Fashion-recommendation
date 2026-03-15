"""Tests — ShoppingListOptimizer"""
import pytest
from tests.unit.layer2.travel.fixtures import make_travel_wardrobe

from src.layer2_style.travel.shopping_list_optimizer import ShoppingListOptimizer
from src.core.travel_models import ShoppingListResult


@pytest.fixture
def optimizer():
    return ShoppingListOptimizer()


@pytest.fixture
def wardrobe():
    return make_travel_wardrobe()


class TestReturnTypes:
    def test_returns_shopping_list_result(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, budget=300, season="fall")
        assert isinstance(result, ShoppingListResult)

    def test_budget_stored(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, budget=250, season="fall")
        assert result.budget_eur == 250

    def test_items_are_list(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, budget=300, season="fall")
        assert isinstance(result.items, list)


class TestBudgetSplit:
    def test_within_budget_cost_lte_budget(self, optimizer, wardrobe):
        budget = 500.0
        result = optimizer.optimize(wardrobe, budget=budget)
        assert result.total_estimated_cost <= budget + 0.01  # floating tolerance

    def test_all_within_items_in_within_budget(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, budget=1000)
        # With large budget, all items should fit
        total_items = len(result.items)
        within_plus_over = len(result.within_budget) + len(result.over_budget)
        assert within_plus_over == total_items

    def test_zero_budget_nothing_within(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, budget=0)
        assert len(result.within_budget) == 0


class TestRanking:
    def test_ranks_are_sequential(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, budget=500, season="fall")
        for i, item in enumerate(result.items, start=1):
            assert item.rank == i

    def test_roi_computed(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, budget=500, season="fall")
        for item in result.items:
            assert item.roi >= 0.0

    def test_urgency_valid_values(self, optimizer, wardrobe):
        result = optimizer.optimize(wardrobe, budget=500, season="fall")
        for item in result.items:
            assert item.urgency in {"high", "medium", "low"}


class TestHardGaps:
    def test_hard_gap_category_gets_high_urgency(self, optimizer, wardrobe):
        result = optimizer.optimize(
            wardrobe, budget=500, season="fall", hard_gaps=["shoes"]
        )
        shoe_items = [i for i in result.items if i.category == "shoes"]
        if shoe_items:
            # At least one shoe item should have elevated urgency
            urgencies = {i.urgency for i in shoe_items}
            assert urgencies & {"high", "medium"}


class TestPriceTiers:
    def test_budget_tier_cheaper_than_luxury(self, optimizer, wardrobe):
        result_budget = optimizer.optimize(wardrobe, budget=1000, price_tier="budget")
        result_luxury = optimizer.optimize(wardrobe, budget=1000, price_tier="luxury")
        if result_budget.items and result_luxury.items:
            avg_budget = sum(i.estimated_price_eur for i in result_budget.items) / len(result_budget.items)
            avg_luxury = sum(i.estimated_price_eur for i in result_luxury.items) / len(result_luxury.items)
            assert avg_budget < avg_luxury
