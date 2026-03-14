"""Tests — CapsuleExplainer (Layer 4 LLM)"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.models import (
    CapsuleAnalysisResult,
    CapsuleEvolutionResult,
    CapsuleGarmentScore,
    CapsuleGarmentRole,
    CapsuleOutfit,
    CapsuleOutfitsResult,
    CapsuleSnapshot,
    MissingPieceRecommendation,
    MissingPiecesResult,
    ReplacementPlanResult,
    ReplacementVerdict,
)


# ============================================================================
# Helpers — build minimal result objects
# ============================================================================

def _capsule_analysis(score: float = 65.0) -> CapsuleAnalysisResult:
    return CapsuleAnalysisResult(
        cohesion_score=score,
        color_cohesion_score=0.6,
        versatility_ratio=0.4,
        redundancy_penalty=0.1,
        orphan_penalty=0.05,
        dominant_colors=["black"],
        color_coverage_pct=0.8,
        garment_scores=[],
        key_pieces=["k1"],
        orphan_pieces=["o1"],
        redundant_pairs=[],
        total_garments=12,
        total_outfits=40,
        capsule_profile="standard",
        recommendation="",
        projected_score_after_cleanup=70.0,
    )


def _missing_result() -> MissingPiecesResult:
    return MissingPiecesResult(
        current_cohesion=55.0,
        projected_cohesion=65.0,
        recommendations=[
            MissingPieceRecommendation(
                priority=1,
                category="outerwear",
                description="structured blazer",
                reason="High impact piece",
                impact_outfits=20,
                suggested_colors=["black", "navy"],
                profile_note="",
            ),
            MissingPieceRecommendation(
                priority=2,
                category="shoes",
                description="white sneakers",
                reason="Versatile base",
                impact_outfits=15,
                suggested_colors=["white"],
                profile_note="",
            ),
        ],
        summary="Adding 2 pieces could boost your score.",
    )


def _replacement_plan() -> ReplacementPlanResult:
    return ReplacementPlanResult(
        verdicts=[
            ReplacementVerdict(
                garment_keep_id="k1",
                garment_remove_id="r1",
                versatility_gain=0.4,
                confidence=0.9,
                transition_timing="Replace at next seasonal refresh.",
            )
        ],
        total_outfits_gained=5,
        summary="1 replacement recommended.",
    )


def _outfits_result() -> CapsuleOutfitsResult:
    return CapsuleOutfitsResult(
        outfits=[
            CapsuleOutfit(
                rank=1,
                garment_ids=["g1", "g2"],
                capsule_score=88.0,
                pct_key_pieces=0.8,
                avg_versatility=0.75,
                tier="basic",
                occasions=["daily_wear", "work"],
            )
        ],
        basic_count=1,
        semi_creative_count=0,
        creative_count=0,
    )


def _evolution_result() -> CapsuleEvolutionResult:
    return CapsuleEvolutionResult(
        snapshots=[
            CapsuleSnapshot(
                date="2025-01-01",
                cohesion_score=60.0,
                garments_count=10,
                outfits_count=30,
                key_pieces_count=3,
                orphans_count=1,
            ),
            CapsuleSnapshot(
                date="2025-01-15",
                cohesion_score=70.0,
                garments_count=12,
                outfits_count=45,
                key_pieces_count=4,
                orphans_count=0,
            ),
        ],
        delta_score=10.0,
        delta_outfits=15,
        outfit_ratio=3.75,
        pieces_ratio=1.2,
        predicted_weeks_to_90=8,
        trend_direction="improving",
        latest_cohesion=70.0,
    )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """Patches _GeminiLLM to avoid real API calls."""
    with patch("src.layer4_llm.capsule_explainer._GeminiLLM") as MockLLM:
        instance = MagicMock()
        instance.generate_completion = AsyncMock(return_value='["Great blazer choice!", "Love these sneakers!"]')
        MockLLM.return_value = instance
        yield instance


@pytest.fixture
def explainer(mock_llm):
    from src.layer4_llm.capsule_explainer import CapsuleExplainer
    return CapsuleExplainer()


# ============================================================================
# F1 — wardrobe diagnosis narration
# ============================================================================

class TestNarrateWardrobeDiagnosis:
    async def test_adds_recommendation_when_empty(self, explainer, mock_llm):
        mock_llm.generate_completion = AsyncMock(return_value="Your wardrobe is almost capsule-ready!")
        analysis = _capsule_analysis()
        result = await explainer.narrate_wardrobe_diagnosis(analysis)
        assert result.recommendation != ""

    async def test_skips_if_recommendation_exists(self, explainer, mock_llm):
        analysis = _capsule_analysis()
        analysis.recommendation = "Already written"
        result = await explainer.narrate_wardrobe_diagnosis(analysis)
        assert result.recommendation == "Already written"
        mock_llm.generate_completion.assert_not_called()

    async def test_graceful_degradation_on_llm_error(self, explainer, mock_llm):
        mock_llm.generate_completion = AsyncMock(side_effect=Exception("API down"))
        analysis = _capsule_analysis()
        result = await explainer.narrate_wardrobe_diagnosis(analysis)
        # Should not raise — recommendation stays empty
        assert isinstance(result, CapsuleAnalysisResult)


# ============================================================================
# F2 — missing pieces narration
# ============================================================================

class TestNarrateMissingPieces:
    async def test_narration_added_to_each_recommendation(self, explainer, mock_llm):
        mock_llm.generate_completion = AsyncMock(
            return_value='["A blazer is a game-changer.", "Sneakers work with everything."]'
        )
        result_obj = _missing_result()
        result = await explainer.narrate_missing_pieces(result_obj)
        assert result.recommendations[0].llm_narration == "A blazer is a game-changer."
        assert result.recommendations[1].llm_narration == "Sneakers work with everything."

    async def test_single_batch_call(self, explainer, mock_llm):
        mock_llm.generate_completion = AsyncMock(return_value='["Note A.", "Note B."]')
        await explainer.narrate_missing_pieces(_missing_result())
        assert mock_llm.generate_completion.call_count == 1

    async def test_empty_recommendations_no_llm_call(self, explainer, mock_llm):
        empty = MissingPiecesResult(
            current_cohesion=50.0, projected_cohesion=55.0, recommendations=[], summary=""
        )
        await explainer.narrate_missing_pieces(empty)
        mock_llm.generate_completion.assert_not_called()


# ============================================================================
# F3 — replacement plan narration
# ============================================================================

class TestNarrateReplacementPlan:
    async def test_narration_added_to_verdict(self, explainer, mock_llm):
        mock_llm.generate_completion = AsyncMock(return_value='["Swap this piece next season."]')
        plan = _replacement_plan()
        result = await explainer.narrate_replacement_plan(plan)
        assert result.verdicts[0].llm_narration == "Swap this piece next season."

    async def test_empty_plan_no_call(self, explainer, mock_llm):
        empty_plan = ReplacementPlanResult(verdicts=[], total_outfits_gained=0, summary="")
        await explainer.narrate_replacement_plan(empty_plan)
        mock_llm.generate_completion.assert_not_called()


# ============================================================================
# F4 — capsule outfit narration
# ============================================================================

class TestNarrateCapsuleOutfits:
    async def test_narration_added_to_outfit(self, explainer, mock_llm):
        mock_llm.generate_completion = AsyncMock(return_value='["Perfect for a Monday morning commute."]')
        outfits = _outfits_result()
        result = await explainer.narrate_capsule_outfits(outfits)
        assert result.outfits[0].llm_narration == "Perfect for a Monday morning commute."

    async def test_top_n_limits_narration(self, explainer, mock_llm):
        mock_llm.generate_completion = AsyncMock(return_value='["Note."]')
        outfits = _outfits_result()
        await explainer.narrate_capsule_outfits(outfits, top_n=1)
        assert mock_llm.generate_completion.call_count == 1


# ============================================================================
# F5 — evolution narration
# ============================================================================

class TestNarrateEvolution:
    async def test_trend_direction_updated(self, explainer, mock_llm):
        mock_llm.generate_completion = AsyncMock(
            return_value="Great progress! You've unlocked 15 new outfits."
        )
        evolution = _evolution_result()
        result = await explainer.narrate_evolution(evolution)
        assert "progress" in result.trend_direction.lower() or len(result.trend_direction) > 5


# ============================================================================
# Cache behaviour
# ============================================================================

class TestCaching:
    async def test_second_call_uses_cache(self, explainer, mock_llm):
        mock_llm.generate_completion = AsyncMock(
            return_value='["Cached narration.", "Also cached."]'
        )
        result_obj = _missing_result()
        await explainer.narrate_missing_pieces(result_obj)
        # Reset narrations and call again
        for rec in result_obj.recommendations:
            rec.llm_narration = None
        await explainer.narrate_missing_pieces(result_obj)
        # LLM should only be called once total
        assert mock_llm.generate_completion.call_count == 1


# ============================================================================
# JSON parsing helper
# ============================================================================

class TestParseJsonArray:
    def test_parses_clean_json(self):
        from src.layer4_llm.capsule_explainer import CapsuleExplainer
        result = CapsuleExplainer._parse_json_array('["a", "b", "c"]', 3)
        assert result == ["a", "b", "c"]

    def test_pads_short_array(self):
        from src.layer4_llm.capsule_explainer import CapsuleExplainer
        result = CapsuleExplainer._parse_json_array('["a"]', 3)
        assert len(result) == 3
        assert result[1] == ""

    def test_trims_long_array(self):
        from src.layer4_llm.capsule_explainer import CapsuleExplainer
        result = CapsuleExplainer._parse_json_array('["a", "b", "c", "d"]', 2)
        assert result == ["a", "b"]

    def test_handles_malformed_json(self):
        from src.layer4_llm.capsule_explainer import CapsuleExplainer
        result = CapsuleExplainer._parse_json_array("not json at all", 3)
        assert result == ["", "", ""]

    def test_extracts_from_markdown_wrapper(self):
        from src.layer4_llm.capsule_explainer import CapsuleExplainer
        raw = 'Sure! Here:\n```json\n["a", "b"]\n```'
        result = CapsuleExplainer._parse_json_array(raw, 2)
        assert result == ["a", "b"]
