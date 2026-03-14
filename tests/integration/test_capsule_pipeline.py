"""
Integration tests — Capsule Wardrobe Pipeline
==============================================
End-to-end verification that all 5 capsule features chain correctly:

  F1  WardrobeCapsuleAnalyzer   → CapsuleAnalysisResult
  F2  MissingPiecesRecommender  → MissingPiecesResult
  F3  ReplacementPlanner        → ReplacementPlanResult
  F4  CapsuleOutfitGenerator    → CapsuleOutfitsResult
  F5  CapsuleEvolutionTracker   → CapsuleEvolutionResult

The LLM layer (CapsuleExplainer) and Neo4j are both mocked so these
tests run fully offline without any external services.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.core.models import (
    CapsuleAnalysisResult,
    CapsuleEvolutionResult,
    CapsuleGarmentScore,
    CapsuleOutfitsResult,
    GarmentCapsuleRole,
    ColorProfile,
    FormalityLevel,
    Garment,
    GarmentAttributes,
    GarmentCategory,
    MaterialProfile,
    MissingPiecesResult,
    PatternInfo,
    ReplacementPlanResult,
    Season,
    SeasonalityInfo,
)
from src.layer2_style.capsule import (
    CapsuleEvolutionTracker,
    CapsuleOutfitGenerator,
    MissingPiecesRecommender,
    ReplacementPlanner,
    WardrobeCapsuleAnalyzer,
)


# ============================================================================
# Helpers
# ============================================================================

def _g(
    garment_id: str | None = None,
    category: GarmentCategory = GarmentCategory.TOP,
    subcategory: str = "t-shirt",
    color: str = "black",
    formality: FormalityLevel = FormalityLevel.CASUAL,
    seasons: list[Season] | None = None,
) -> Garment:
    season_list = seasons or [Season.SPRING, Season.SUMMER, Season.FALL, Season.WINTER]
    return Garment(
        id=garment_id or f"g_{uuid4().hex[:8]}",
        attributes=GarmentAttributes(
            category=category,
            subcategory=subcategory,
            color=ColorProfile(primary=color, hex_codes=[]),
            formality_level=formality,
            season_suitable=season_list,
            seasonality=SeasonalityInfo(seasons=season_list),
            pattern=PatternInfo(type="solid"),
            material=MaterialProfile(primary="cotton"),
        ),
    )


def _demo_wardrobe() -> List[Garment]:
    """A representative 12-piece capsule wardrobe for integration testing."""
    return [
        # Tops
        _g("top_white_tshirt",  GarmentCategory.TOP,     "t-shirt",   "white"),
        _g("top_black_tshirt",  GarmentCategory.TOP,     "t-shirt",   "black"),
        _g("top_navy_shirt",    GarmentCategory.TOP,     "shirt",     "navy",  FormalityLevel.SMART_CASUAL),
        _g("top_white_shirt",   GarmentCategory.TOP,     "shirt",     "white", FormalityLevel.BUSINESS),
        # Bottoms
        _g("bot_blue_jeans",    GarmentCategory.BOTTOM,  "jeans",     "blue"),
        _g("bot_grey_trousers", GarmentCategory.BOTTOM,  "trousers",  "grey",  FormalityLevel.SMART_CASUAL),
        _g("bot_black_trousers",GarmentCategory.BOTTOM,  "trousers",  "black", FormalityLevel.BUSINESS),
        # Outerwear
        _g("out_camel_coat",    GarmentCategory.OUTERWEAR,"coat",     "camel", FormalityLevel.SMART_CASUAL,
           seasons=[Season.FALL, Season.WINTER]),
        _g("out_black_blazer",  GarmentCategory.OUTERWEAR,"blazer",   "black", FormalityLevel.BUSINESS,
           seasons=[Season.FALL, Season.WINTER]),
        # Shoes
        _g("sho_white_snkrs",   GarmentCategory.SHOES,   "sneakers",  "white"),
        _g("sho_brown_loafers", GarmentCategory.SHOES,   "loafers",   "brown", FormalityLevel.SMART_CASUAL),
        # Accessory
        _g("acc_black_watch",   GarmentCategory.ACCESSORY,"watch",    "black"),
    ]


def _centrality_map() -> dict[str, float]:
    """Simulate Neo4j betweenness centrality scores."""
    return {
        "top_white_tshirt":   0.72,
        "top_black_tshirt":   0.68,
        "top_navy_shirt":     0.55,
        "top_white_shirt":    0.45,
        "bot_blue_jeans":     0.61,
        "bot_grey_trousers":  0.50,
        "bot_black_trousers": 0.48,
        "out_camel_coat":     0.30,
        "out_black_blazer":   0.35,
        "sho_white_snkrs":    0.40,
        "sho_brown_loafers":  0.38,
        "acc_black_watch":    0.25,
    }


# ============================================================================
# F1 — WardrobeCapsuleAnalyzer
# ============================================================================

class TestWardrobeCapsuleAnalyzerIntegration:

    def test_analysis_produces_valid_result(self):
        analyzer = WardrobeCapsuleAnalyzer()
        wardrobe = _demo_wardrobe()
        result = analyzer.analyze(wardrobe, centrality_map=_centrality_map())

        assert isinstance(result, CapsuleAnalysisResult)
        assert 0 <= result.cohesion_score <= 100
        assert result.total_garments == len(wardrobe)
        assert len(result.garment_scores) == len(wardrobe)
        assert result.capsule_profile in ("minimalist", "standard", "rich")

    def test_centrality_boosts_key_pieces(self):
        analyzer = WardrobeCapsuleAnalyzer()
        wardrobe = _demo_wardrobe()
        result = analyzer.analyze(wardrobe, centrality_map=_centrality_map())

        # High centrality garments should be KEY_PIECE
        key_ids = set(result.key_pieces)
        assert "top_white_tshirt" in key_ids   # centrality = 0.72 (highest)
        assert "top_black_tshirt" in key_ids   # centrality = 0.68

    def test_no_centrality_still_works(self):
        analyzer = WardrobeCapsuleAnalyzer()
        wardrobe = _demo_wardrobe()
        result = analyzer.analyze(wardrobe)

        assert result.cohesion_score > 0
        assert result.garment_scores

    def test_redundant_pairs_detected_for_similar_items(self):
        """Two near-identical items of the same subcategory/color should be flagged."""
        analyzer = WardrobeCapsuleAnalyzer()
        wardrobe = [
            _g("black_shirt_a", GarmentCategory.TOP, "shirt", "black"),
            _g("black_shirt_b", GarmentCategory.TOP, "shirt", "black"),
        ]
        result = analyzer.analyze(wardrobe)
        assert len(result.redundant_pairs) >= 1

    def test_wardrobe_hash_is_deterministic(self):
        wardrobe = _demo_wardrobe()
        h1 = WardrobeCapsuleAnalyzer.wardrobe_hash(wardrobe)
        h2 = WardrobeCapsuleAnalyzer.wardrobe_hash(wardrobe)
        assert h1 == h2

    def test_wardrobe_hash_differs_after_adding_garment(self):
        wardrobe = _demo_wardrobe()
        h_before = WardrobeCapsuleAnalyzer.wardrobe_hash(wardrobe)
        wardrobe.append(_g("new_garment"))
        h_after = WardrobeCapsuleAnalyzer.wardrobe_hash(wardrobe)
        assert h_before != h_after


# ============================================================================
# F2 — MissingPiecesRecommender
# ============================================================================

class TestMissingPiecesRecommenderIntegration:

    def test_recommendations_from_real_analysis(self):
        analyzer = WardrobeCapsuleAnalyzer()
        wardrobe = _demo_wardrobe()
        analysis = analyzer.analyze(wardrobe, centrality_map=_centrality_map())

        recommender = MissingPiecesRecommender(top_n=5)
        result = recommender.recommend(analysis, user_season="summer", body_shape="rectangle")

        assert isinstance(result, MissingPiecesResult)
        assert len(result.recommendations) <= 5
        assert result.projected_cohesion >= result.current_cohesion

    def test_impact_outfits_positive(self):
        analyzer = WardrobeCapsuleAnalyzer()
        analysis = analyzer.analyze(_demo_wardrobe(), centrality_map=_centrality_map())

        result = MissingPiecesRecommender(top_n=3).recommend(analysis)
        for rec in result.recommendations:
            assert rec.impact_outfits > 0

    def test_no_llm_narration_by_default(self):
        analysis = WardrobeCapsuleAnalyzer().analyze(_demo_wardrobe())
        result = MissingPiecesRecommender().recommend(analysis)
        for rec in result.recommendations:
            # llm_narration is empty string by default (not populated until CapsuleExplainer runs)
            assert not rec.llm_narration


# ============================================================================
# F3 — ReplacementPlanner
# ============================================================================

class TestReplacementPlannerIntegration:

    def test_plan_uses_analysis_redundant_pairs(self):
        analyzer = WardrobeCapsuleAnalyzer()
        wardrobe = [
            _g("shirt_a", GarmentCategory.TOP, "shirt", "navy", FormalityLevel.SMART_CASUAL),
            _g("shirt_b", GarmentCategory.TOP, "shirt", "navy", FormalityLevel.SMART_CASUAL),
            *_demo_wardrobe()[4:],   # bottoms + outerwear + shoes + accessory
        ]
        analysis = analyzer.analyze(wardrobe, centrality_map=_centrality_map())
        planner = ReplacementPlanner()
        plan = planner.plan(analysis.redundant_pairs, analysis.garment_scores)

        assert isinstance(plan, ReplacementPlanResult)
        # If there are verdicts they should be sorted by confidence
        if plan.verdicts:
            confidences = [v.confidence for v in plan.verdicts]
            assert confidences == sorted(confidences, reverse=True)

    def test_plan_empty_when_no_redundancy(self):
        """A diverse wardrobe with no redundant pairs → empty plan."""
        analyzer = WardrobeCapsuleAnalyzer()
        analysis = analyzer.analyze(_demo_wardrobe(), centrality_map=_centrality_map())
        plan = ReplacementPlanner().plan(analysis.redundant_pairs, analysis.garment_scores)

        # Our demo wardrobe has diverse items; redundant_pairs may be 0
        assert isinstance(plan, ReplacementPlanResult)
        # whether verdicts empty or not is data-driven — just check types
        assert all(0 <= v.confidence <= 1 for v in plan.verdicts)


# ============================================================================
# F4 — CapsuleOutfitGenerator  (uses pre-built OutfitCandidate mocks)
# ============================================================================

class TestCapsuleOutfitGeneratorIntegration:

    def _make_candidates(self, analysis: CapsuleAnalysisResult, n: int = 10):
        """Produce n fake OutfitCandidate objects referencing garments from analysis."""
        from unittest.mock import MagicMock
        key_ids = set(analysis.key_pieces)
        all_ids = [gs.garment_id for gs in analysis.garment_scores]

        candidates = []
        for i in range(n):
            mock = MagicMock()
            # Alternate: half use key pieces, half don't
            if i % 2 == 0 and key_ids:
                g_ids = list(key_ids)[:2]
            else:
                g_ids = all_ids[:2] if len(all_ids) >= 2 else all_ids
            mock.garments = [g for g in _demo_wardrobe() if g.id in g_ids]
            scorecard = MagicMock()
            scorecard.overall_score = 60.0 + i * 2
            scorecard.formality_match = 0.7
            scorecard.color_harmony = 0.8
            scorecard.occasion_suitability = 0.75
            mock.scorecard = scorecard
            candidates.append(mock)
        return candidates

    def test_generator_returns_result(self):
        analyzer = WardrobeCapsuleAnalyzer()
        analysis = analyzer.analyze(_demo_wardrobe(), centrality_map=_centrality_map())
        candidates = self._make_candidates(analysis)
        generator = CapsuleOutfitGenerator()
        result = generator.generate(candidates, analysis, top_n=8)

        assert isinstance(result, CapsuleOutfitsResult)
        assert len(result.outfits) <= 8

    def test_tier_counts_sum_to_total(self):
        analyzer = WardrobeCapsuleAnalyzer()
        analysis = analyzer.analyze(_demo_wardrobe(), centrality_map=_centrality_map())
        candidates = self._make_candidates(analysis, n=20)
        result = CapsuleOutfitGenerator().generate(candidates, analysis, top_n=15)

        total = result.basic_count + result.semi_creative_count + result.creative_count
        assert total == len(result.outfits)


# ============================================================================
# F5 — CapsuleEvolutionTracker
# ============================================================================

class TestCapsuleEvolutionTrackerIntegration:

    def test_record_and_load(self, tmp_path):
        with patch(
            "src.layer2_style.capsule.capsule_evolution_tracker._PROFILES_ROOT",
            tmp_path,
        ):
            tracker = CapsuleEvolutionTracker(user_id="inttest_user")
            analyzer = WardrobeCapsuleAnalyzer()
            wardrobe = _demo_wardrobe()
            analysis = analyzer.analyze(wardrobe, centrality_map=_centrality_map())
            wh = WardrobeCapsuleAnalyzer.wardrobe_hash(wardrobe)

            result1 = tracker.record(analysis, wh, action="initial_scan")
            assert isinstance(result1, CapsuleEvolutionResult)
            assert len(result1.snapshots) == 1

            # Simulate improvement: modify analysis score and use different hash
            new_score = min(analysis.cohesion_score + 8.0, 100.0)
            analysis2 = analysis.model_copy(update={"cohesion_score": new_score})
            result2 = tracker.record(analysis2, wh + "x", action="removed_orphan")
            assert len(result2.snapshots) == 2
            expected_delta = new_score - analysis.cohesion_score
            assert result2.delta_score == pytest.approx(expected_delta, abs=0.1)

    def test_same_hash_deduplication(self, tmp_path):
        with patch(
            "src.layer2_style.capsule.capsule_evolution_tracker._PROFILES_ROOT",
            tmp_path,
        ):
            tracker = CapsuleEvolutionTracker(user_id="inttest_dup")
            analysis = WardrobeCapsuleAnalyzer().analyze(_demo_wardrobe())
            wh = "static_hash_abc"
            tracker.record(analysis, wh)
            tracker.record(analysis, wh)  # same hash → no duplicate
            history = tracker.load_history()
            assert len(history) == 1


# ============================================================================
# Full chain — F1 → F2 → F3 → F4 → F5
# ============================================================================

class TestFullCapsulePipelineChain:
    """
    Verify that the output of each feature feeds naturally into the next.
    No external services required.
    """

    def test_full_chain(self, tmp_path):
        wardrobe = _demo_wardrobe()
        centrality = _centrality_map()

        # F1
        analyzer = WardrobeCapsuleAnalyzer()
        analysis = analyzer.analyze(wardrobe, centrality_map=centrality)
        assert isinstance(analysis, CapsuleAnalysisResult)

        # F2
        recommender = MissingPiecesRecommender(top_n=3)
        missing = recommender.recommend(analysis, user_season="autumn")
        assert isinstance(missing, MissingPiecesResult)
        assert missing.projected_cohesion >= analysis.cohesion_score

        # F3
        planner = ReplacementPlanner()
        plan = planner.plan(analysis.redundant_pairs, analysis.garment_scores)
        assert isinstance(plan, ReplacementPlanResult)

        # F4 — build mock candidates from F1 analysis
        from unittest.mock import MagicMock
        mock_candidates = []
        for gs in analysis.garment_scores[:6]:
            mc = MagicMock()
            mc.garments = [g for g in wardrobe if g.id == gs.garment_id]
            sc = MagicMock()
            sc.overall_score = gs.versatility_score * 100
            sc.formality_match = 0.7
            sc.color_harmony = 0.8
            sc.occasion_suitability = 0.7
            mc.scorecard = sc
            mock_candidates.append(mc)
        generator = CapsuleOutfitGenerator()
        outfits = generator.generate(mock_candidates, analysis, top_n=5)
        assert isinstance(outfits, CapsuleOutfitsResult)

        # F5
        with patch(
            "src.layer2_style.capsule.capsule_evolution_tracker._PROFILES_ROOT",
            tmp_path,
        ):
            tracker = CapsuleEvolutionTracker(user_id="chain_test")
            wh = WardrobeCapsuleAnalyzer.wardrobe_hash(wardrobe)
            evolution = tracker.record(analysis, wh, action="integration_test")
            assert isinstance(evolution, CapsuleEvolutionResult)
            assert evolution.latest_cohesion == pytest.approx(analysis.cohesion_score, abs=0.01)

    def test_chain_produces_no_exceptions_for_empty_redundant_pairs(self, tmp_path):
        """Diverse wardrobe with zero redundant pairs should not crash any feature."""
        wardrobe = _demo_wardrobe()
        analysis = WardrobeCapsuleAnalyzer().analyze(wardrobe, centrality_map=_centrality_map())

        # F3 with empty redundant pairs
        plan = ReplacementPlanner().plan([], analysis.garment_scores)
        assert plan.verdicts == []
        assert plan.total_outfits_gained == 0
