"""
Integration tests — CapsuleOptimizer end-to-end
=================================================
Full pipeline validation with realistic wardrobe data, verifying that
the optimizer, service layer, and scoring work together correctly.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List
from uuid import uuid4

from src.layer2_style.capsule.capsule_optimizer import (
    CapsuleOptimizer,
    CapsuleOptimizerResult,
    score_capsule,
    _garment_cat,
)
from src.api.services.capsule_optimizer_service import (
    OptimizeCapsuleRequest,
    OptimizedCapsuleResponse,
    optimize_capsule,
)


# ============================================================================
# Helpers
# ============================================================================

def _g(
    garment_id: str | None = None,
    category: str = "top",
    subcategory: str = "t-shirt",
    color_hex: str = "#000000",
    formality: str = "casual",
    seasons: list[str] | None = None,
    material: str = "cotton",
    confidence: float = 0.95,
) -> Dict[str, Any]:
    return {
        "id": garment_id or f"g_{uuid4().hex[:8]}",
        "attributes": {
            "category": category,
            "subcategory": subcategory,
            "color_hex": color_hex,
            "formality": formality,
            "seasons": seasons or ["spring", "summer", "autumn", "winter"],
            "material": material,
            "confidence": confidence,
        },
    }


def _realistic_wardrobe() -> List[Dict[str, Any]]:
    """20-piece realistic wardrobe for integration tests."""
    return [
        # Tops
        _g("t1", "top", "t-shirt", "#FFFFFF", "casual"),
        _g("t2", "top", "polo", "#1C3A5F", "smart_casual"),
        _g("t3", "top", "blouse", "#000000", "business"),
        _g("t4", "top", "sweater", "#8B0000", "casual",
           seasons=["autumn", "winter"]),
        _g("t5", "top", "button-down", "#87CEEB", "business"),
        # Bottoms
        _g("b1", "bottom", "jeans", "#00008B", "casual"),
        _g("b2", "bottom", "chinos", "#D2B48C", "smart_casual"),
        _g("b3", "bottom", "trousers", "#808080", "business"),
        _g("b4", "bottom", "shorts", "#228B22", "casual",
           seasons=["spring", "summer"]),
        # Dresses
        _g("d1", "dress", "midi-dress", "#FF69B4", "smart_casual"),
        _g("d2", "dress", "cocktail-dress", "#000000", "formal"),
        # Shoes
        _g("s1", "shoes", "sneakers", "#FFFFFF", "casual"),
        _g("s2", "shoes", "loafers", "#8B4513", "smart_casual"),
        _g("s3", "shoes", "heels", "#000000", "formal"),
        # Outerwear
        _g("o1", "outerwear", "blazer", "#2F4F4F", "business",
           seasons=["autumn", "winter", "spring"]),
        _g("o2", "outerwear", "parka", "#556B2F", "casual",
           seasons=["autumn", "winter"]),
        # Accessories
        _g("a1", "accessory", "watch", "#C0C0C0", "casual"),
        _g("a2", "accessory", "scarf", "#800000", "smart_casual",
           seasons=["autumn", "winter"]),
        _g("a3", "accessory", "belt", "#000000", "business"),
        _g("a4", "accessory", "sunglasses", "#000000", "casual",
           seasons=["spring", "summer"]),
    ]


# ============================================================================
# Core optimizer integration
# ============================================================================

@pytest.mark.integration
class TestOptimizerPipeline:
    """End-to-end optimizer with realistic wardrobe."""

    def test_selects_exact_n_pieces(self):
        opt = CapsuleOptimizer()
        wb = _realistic_wardrobe()
        for n in (5, 8, 12, 15):
            result = opt.optimize(wb, n_pieces=n)
            assert result.n_pieces == n, f"Expected {n}, got {result.n_pieces}"

    def test_total_wardrobe_count(self):
        opt = CapsuleOptimizer()
        wb = _realistic_wardrobe()
        result = opt.optimize(wb, n_pieces=8)
        assert result.total_wardrobe == 20

    def test_score_is_positive(self):
        opt = CapsuleOptimizer()
        result = opt.optimize(_realistic_wardrobe(), n_pieces=8)
        assert result.total_score > 0.0

    def test_generates_valid_outfits(self):
        opt = CapsuleOptimizer()
        result = opt.optimize(_realistic_wardrobe(), n_pieces=8,
                              occasion_types=["casual", "smart_casual"])
        assert result.valid_combinations > 0

    def test_occasion_coverage_keys(self):
        opt = CapsuleOptimizer()
        occasions = ["casual", "business"]
        result = opt.optimize(_realistic_wardrobe(), n_pieces=8,
                              occasion_types=occasions)
        for occ in occasions:
            assert occ in result.occasion_coverage

    def test_anchors_in_result(self):
        opt = CapsuleOptimizer()
        result = opt.optimize(
            _realistic_wardrobe(), n_pieces=8,
            anchor_ids=["t1", "b1", "s1"],
        )
        selected_ids = {g["id"] for g in result.selected_garments}
        assert {"t1", "b1", "s1"}.issubset(selected_ids)

    def test_2opt_does_not_degrade(self):
        wb = _realistic_wardrobe()
        r_greedy = CapsuleOptimizer(max_2opt_iterations=0).optimize(wb, n_pieces=8)
        r_2opt = CapsuleOptimizer(max_2opt_iterations=5).optimize(wb, n_pieces=8)
        assert r_2opt.total_score >= r_greedy.total_score

    def test_warm_climate_avoids_winter(self):
        opt = CapsuleOptimizer()
        result = opt.optimize(_realistic_wardrobe(), n_pieces=6, climate="warm")
        for g in result.selected_garments:
            seasons = (g.get("attributes") or {}).get("seasons", [])
            # If the garment has seasons, at least one should be warm-compatible
            if seasons:
                assert any(s in ["spring", "summer"] for s in seasons) or len(seasons) == 0

    def test_cold_climate_includes_layers(self):
        opt = CapsuleOptimizer()
        result = opt.optimize(_realistic_wardrobe(), n_pieces=10, climate="cold")
        cats = {_garment_cat(g) for g in result.selected_garments}
        # Should have outerwear or at least mention missing layers
        has_outerwear = "outerwear" in cats
        mentions_layer = any("warm" in m.lower() or "coat" in m.lower()
                             for m in result.missing_pieces)
        assert has_outerwear or mentions_layer

    def test_piece_contributions_match_n(self):
        opt = CapsuleOptimizer()
        result = opt.optimize(_realistic_wardrobe(), n_pieces=8)
        assert len(result.piece_contributions) == result.n_pieces


# ============================================================================
# Service layer integration
# ============================================================================

@pytest.mark.integration
class TestOptimizeServicePipeline:
    """Integration tests for the service layer wrapping the optimizer."""

    def test_service_returns_response_model(self):
        request = OptimizeCapsuleRequest(
            user_id="test-user",
            n_pieces=8,
            occasion_types=["casual", "smart_casual"],
            climate="mixed",
        )
        response = optimize_capsule(request, _realistic_wardrobe())
        assert isinstance(response, OptimizedCapsuleResponse)

    def test_service_selected_pieces_have_garment_data(self):
        request = OptimizeCapsuleRequest(
            user_id="test-user",
            n_pieces=6,
            occasion_types=["casual"],
        )
        response = optimize_capsule(request, _realistic_wardrobe())
        for piece in response.selected_garments:
            assert piece.garment_id
            assert piece.category
            assert piece.role

    def test_service_summary_text(self):
        request = OptimizeCapsuleRequest(
            user_id="test-user",
            n_pieces=8,
            occasion_types=["casual", "business"],
        )
        response = optimize_capsule(request, _realistic_wardrobe())
        assert "8 pieces selected" in response.summary
        assert "casual & business" in response.summary

    def test_service_with_anchors(self):
        request = OptimizeCapsuleRequest(
            user_id="test-user",
            n_pieces=6,
            anchor_ids=["t1", "s1"],
        )
        response = optimize_capsule(request, _realistic_wardrobe())
        selected_ids = {p.garment_id for p in response.selected_garments}
        assert "t1" in selected_ids
        assert "s1" in selected_ids

    def test_service_with_exclusions(self):
        request = OptimizeCapsuleRequest(
            user_id="test-user",
            n_pieces=6,
            excluded_ids=["t1", "b1"],
        )
        response = optimize_capsule(request, _realistic_wardrobe())
        selected_ids = {p.garment_id for p in response.selected_garments}
        assert "t1" not in selected_ids
        assert "b1" not in selected_ids

    def test_service_source_field(self):
        request = OptimizeCapsuleRequest(
            user_id="test-user",
            n_pieces=5,
        )
        response = optimize_capsule(request, _realistic_wardrobe())
        assert response.source == "greedy_2opt"

    def test_service_alternatives(self):
        request = OptimizeCapsuleRequest(
            user_id="test-user",
            n_pieces=8,
            occasion_types=["casual"],
        )
        response = optimize_capsule(request, _realistic_wardrobe())
        for alt in response.alternatives:
            assert alt.total_score >= 0.0
            assert alt.label

    def test_service_score_range(self):
        request = OptimizeCapsuleRequest(
            user_id="test-user",
            n_pieces=10,
            occasion_types=["casual", "smart_casual", "business"],
        )
        response = optimize_capsule(request, _realistic_wardrobe())
        assert 0.0 <= response.total_score <= 100.0
        assert response.valid_combinations >= 0
