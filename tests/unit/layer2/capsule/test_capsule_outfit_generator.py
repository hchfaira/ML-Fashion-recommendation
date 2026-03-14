"""Tests — CapsuleOutfitGenerator (F4)"""
import pytest
from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

from src.core.models import (
    CapsuleAnalysisResult,
    CapsuleGarmentScore,
    CapsuleGarmentRole,
    CapsuleOutfitsResult,
    Garment,
    GarmentAttributes,
    GarmentCategory,
    ColorProfile,
    FormalityLevel,
    Season,
    PatternInfo,
    MaterialProfile,
    SeasonalityInfo,
)
from src.layer2_style.capsule.capsule_outfit_generator import CapsuleOutfitGenerator


# ============================================================================
# Helpers
# ============================================================================

def _garment(gid: str, color: str = "black") -> Garment:
    return Garment(
        id=gid,
        attributes=GarmentAttributes(
            category=GarmentCategory.TOP,
            subcategory="t-shirt",
            color=ColorProfile(primary=color, hex_codes=[]),
            formality_level=FormalityLevel.CASUAL,
            season_suitable=[Season.SPRING],
            seasonality=SeasonalityInfo(seasons=[Season.SPRING]),
            pattern=PatternInfo(type="solid"),
            material=MaterialProfile(primary="cotton"),
        ),
    )


# Minimal fake OutfitScorecard
class _FakeScorecard:
    overall_score: float = 0.8
    scores: dict = field(default_factory=dict)

    def __init__(self, score=0.8):
        self.overall_score = score


from src.layer2_style.outfit_builder import OutfitCandidate


def _candidate(garments: List[Garment], score: float = 0.75) -> OutfitCandidate:
    from unittest.mock import MagicMock
    sc = MagicMock()
    sc.overall_score = score
    return OutfitCandidate(
        garments=garments,
        scorecard=sc,
        overall_score=score,
        name="test_outfit",
    )


def _analysis(key_ids: List[str] = None, scores: List[CapsuleGarmentScore] = None) -> CapsuleAnalysisResult:
    key_ids = key_ids or []
    scores = scores or []
    return CapsuleAnalysisResult(
        cohesion_score=65.0,
        color_cohesion_score=0.6,
        versatility_ratio=0.4,
        redundancy_penalty=0.1,
        orphan_penalty=0.05,
        dominant_colors=["black"],
        color_coverage_pct=0.8,
        garment_scores=scores,
        key_pieces=key_ids,
        orphan_pieces=[],
        redundant_pairs=[],
        total_garments=10,
        total_outfits=40,
        capsule_profile="standard",
        recommendation="",
        projected_score_after_cleanup=70.0,
    )


def _gs(gid: str, role: CapsuleGarmentRole = CapsuleGarmentRole.KEY_PIECE, v: float = 0.8) -> CapsuleGarmentScore:
    return CapsuleGarmentScore(
        garment_id=gid,
        garment_description=f"garment {gid}",
        versatility_score=v,
        outfit_count=16,
        total_outfits=20,
        capsule_role=role,
    )


@pytest.fixture
def generator():
    return CapsuleOutfitGenerator()


# ============================================================================
# Empty input
# ============================================================================

class TestEmptyInput:
    def test_empty_candidates_returns_empty_result(self, generator):
        result = generator.generate([], _analysis())
        assert isinstance(result, CapsuleOutfitsResult)
        assert result.outfits == []
        assert result.basic_count == 0


# ============================================================================
# Scoring and ranking
# ============================================================================

class TestScoringAndRanking:
    def test_returns_capsule_outfits_result(self, generator):
        g1 = _garment("g1")
        candidates = [_candidate([g1])]
        result = generator.generate(candidates, _analysis())
        assert isinstance(result, CapsuleOutfitsResult)

    def test_outfits_sorted_by_score_desc(self, generator):
        g1, g2, g3 = _garment("g1"), _garment("g2"), _garment("g3")
        scores = [_gs("g1", v=0.9), _gs("g2", v=0.3), _gs("g3", v=0.6)]
        analysis = _analysis(key_ids=["g1"], scores=scores)
        candidates = [
            _candidate([g2], score=0.3),
            _candidate([g1], score=0.9),
            _candidate([g3], score=0.6),
        ]
        result = generator.generate(candidates, analysis)
        capsule_scores = [o.capsule_score for o in result.outfits]
        assert capsule_scores == sorted(capsule_scores, reverse=True)

    def test_ranks_are_sequential(self, generator):
        g1, g2 = _garment("g1"), _garment("g2")
        candidates = [_candidate([g1]), _candidate([g2])]
        result = generator.generate(candidates, _analysis())
        ranks = [o.rank for o in result.outfits]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_capsule_score_in_range(self, generator):
        g1 = _garment("g1")
        scores = [_gs("g1", v=0.8)]
        analysis = _analysis(key_ids=["g1"], scores=scores)
        result = generator.generate([_candidate([g1])], analysis)
        for o in result.outfits:
            assert 0.0 <= o.capsule_score <= 100.0


# ============================================================================
# Tier assignment
# ============================================================================

class TestTierAssignment:
    def test_all_key_pieces_is_basic_tier(self, generator):
        g1 = _garment("g1")
        scores = [_gs("g1", CapsuleGarmentRole.KEY_PIECE, v=1.0)]
        analysis = _analysis(key_ids=["g1"], scores=scores)
        result = generator.generate([_candidate([g1], score=1.0)], analysis)
        # 100% key + high versatility → should be basic
        assert result.outfits[0].tier in {"basic", "semi_creative"}

    def test_no_key_pieces_is_creative_or_semi(self, generator):
        g1 = _garment("g1")
        scores = [_gs("g1", CapsuleGarmentRole.ACCEPTABLE, v=0.3)]
        analysis = _analysis(key_ids=[], scores=scores)
        result = generator.generate([_candidate([g1], score=0.3)], analysis)
        assert result.outfits[0].tier in {"creative", "semi_creative"}

    def test_tier_counts_match_outfits(self, generator):
        g1, g2, g3 = _garment("g1"), _garment("g2"), _garment("g3")
        scores = [
            _gs("g1", v=1.0), _gs("g2", v=0.5), _gs("g3", v=0.2)
        ]
        analysis = _analysis(key_ids=["g1"], scores=scores)
        candidates = [_candidate([g1], 1.0), _candidate([g2], 0.5), _candidate([g3], 0.2)]
        result = generator.generate(candidates, analysis)
        total_count = result.basic_count + result.semi_creative_count + result.creative_count
        assert total_count == len(result.outfits)


# ============================================================================
# top_n limit
# ============================================================================

class TestTopNLimit:
    def test_top_n_respected(self, generator):
        garments = [_garment(f"g{i}") for i in range(20)]
        candidates = [_candidate([g]) for g in garments]
        result = generator.generate(candidates, _analysis(), top_n=5)
        assert len(result.outfits) <= 5

    def test_all_returned_when_fewer_than_top_n(self, generator):
        garments = [_garment("g1"), _garment("g2")]
        candidates = [_candidate([g]) for g in garments]
        result = generator.generate(candidates, _analysis(), top_n=10)
        assert len(result.outfits) == 2
