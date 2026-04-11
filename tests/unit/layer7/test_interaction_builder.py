"""Unit tests for InteractionBuilder — Layer 7 CF."""
from __future__ import annotations

import math

import pytest

from src.layer7_cf.interaction_builder import (
    SIGNAL_WEIGHTS,
    InteractionBuilder,
    _COUNTER_SIGNALS,
    _NEGATIVE_SIGNALS,
)
from src.layer7_cf.models import InteractionMatrix, InteractionRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder() -> InteractionBuilder:
    return InteractionBuilder()


def _entry(
    user: str = "u1",
    garment: str = "g1",
    **signals,
) -> dict:
    return {"user_id": user, "garment_id": garment, **signals}


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSignalExtraction:
    def test_single_signal_creates_record(self, builder):
        records = builder.extract_records([_entry(times_worn=3)])
        assert len(records) == 1
        assert records[0].signal_type == "times_worn"

    def test_multiple_signals_create_multiple_records(self, builder):
        records = builder.extract_records(
            [_entry(times_worn=2, is_favorite=1)]
        )
        types = {r.signal_type for r in records}
        assert "times_worn" in types
        assert "is_favorite" in types

    def test_unknown_signal_is_ignored(self, builder):
        records = builder.extract_records(
            [_entry(unknown_field=99)]
        )
        assert records == []

    def test_zero_value_is_ignored(self, builder):
        records = builder.extract_records(
            [_entry(times_worn=0)]
        )
        assert records == []

    def test_negative_value_is_ignored(self, builder):
        records = builder.extract_records(
            [_entry(outfit_liked=-1)]
        )
        assert records == []

    def test_missing_user_id_skips_entry(self, builder):
        records = builder.extract_records(
            [{"garment_id": "g1", "times_worn": 5}]
        )
        assert records == []

    def test_missing_garment_id_skips_entry(self, builder):
        records = builder.extract_records(
            [{"user_id": "u1", "times_worn": 5}]
        )
        assert records == []


# ---------------------------------------------------------------------------
# Log1p compression
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLog1pCompression:
    def test_counter_signal_is_compressed(self, builder):
        records = builder.extract_records([_entry(times_worn=10)])
        expected_raw = math.log1p(10)
        assert records[0].raw_value == pytest.approx(expected_raw)

    def test_worn_count_is_compressed(self, builder):
        records = builder.extract_records([_entry(worn_count=5)])
        expected = math.log1p(5)
        assert records[0].raw_value == pytest.approx(expected)

    def test_non_counter_signal_is_not_compressed(self, builder):
        records = builder.extract_records([_entry(is_favorite=1)])
        assert records[0].raw_value == 1.0

    def test_outfit_liked_is_not_compressed(self, builder):
        records = builder.extract_records([_entry(outfit_liked=3)])
        assert records[0].raw_value == 3.0

    def test_counter_signals_set_is_correct(self):
        assert _COUNTER_SIGNALS == {"times_worn", "worn_count", "view"}


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWeights:
    def test_default_weights(self, builder):
        assert builder.weights == SIGNAL_WEIGHTS

    def test_custom_weight_overrides_default(self):
        b = InteractionBuilder(weights={"times_worn": 99.0})
        assert b.weights["times_worn"] == 99.0
        # Other weights unchanged
        assert b.weights["is_favorite"] == SIGNAL_WEIGHTS["is_favorite"]

    def test_record_has_correct_weight(self, builder):
        records = builder.extract_records([_entry(outfit_saved=1)])
        assert records[0].weight == SIGNAL_WEIGHTS["outfit_saved"]


# ---------------------------------------------------------------------------
# Cumulation / aggregation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAggregation:
    def test_single_user_single_garment(self, builder):
        matrix = builder.build([_entry(times_worn=1)])
        assert len(matrix.user_ids) == 1
        assert len(matrix.garment_ids) == 1
        assert matrix.data[0][0] > 0

    def test_multiple_signals_accumulate(self, builder):
        raw = [_entry(times_worn=5, outfit_liked=2)]
        matrix = builder.build(raw)
        tw_contrib = math.log1p(5) * SIGNAL_WEIGHTS["times_worn"]
        ol_contrib = 2.0 * SIGNAL_WEIGHTS["outfit_liked"]
        expected = tw_contrib + ol_contrib
        assert matrix.data[0][0] == pytest.approx(expected)

    def test_multiple_entries_same_pair_accumulate(self, builder):
        raw = [_entry(times_worn=1), _entry(times_worn=2)]
        matrix = builder.build(raw)
        tw1 = math.log1p(1) * SIGNAL_WEIGHTS["times_worn"]
        tw2 = math.log1p(2) * SIGNAL_WEIGHTS["times_worn"]
        assert matrix.data[0][0] == pytest.approx(tw1 + tw2)

    def test_two_users_two_garments(self, builder):
        raw = [
            _entry(user="u1", garment="g1", times_worn=1),
            _entry(user="u2", garment="g2", outfit_liked=1),
        ]
        matrix = builder.build(raw)
        assert len(matrix.user_ids) == 2
        assert len(matrix.garment_ids) == 2
        # Cross-entries are zero
        u1_idx = matrix.user_to_idx["u1"]
        g2_idx = matrix.garment_to_idx["g2"]
        assert matrix.data[u1_idx][g2_idx] == 0.0

    def test_index_maps_are_consistent(self, builder):
        raw = [
            _entry(user="alice", garment="hat", times_worn=1),
            _entry(user="bob", garment="scarf", times_worn=1),
        ]
        matrix = builder.build(raw)
        for uid in matrix.user_ids:
            assert uid in matrix.user_to_idx
        for gid in matrix.garment_ids:
            assert gid in matrix.garment_to_idx


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStats:
    def test_sparsity_empty(self, builder):
        matrix = builder.build([])
        stats = InteractionBuilder.stats(matrix)
        assert stats["n_users"] == 0
        assert stats["sparsity"] == 1.0

    def test_sparsity_full(self, builder):
        raw = [_entry(user="u1", garment="g1", times_worn=1)]
        matrix = builder.build(raw)
        stats = InteractionBuilder.stats(matrix)
        assert stats["sparsity"] == 0.0  # 1 cell, 1 non-zero

    def test_stats_keys(self, builder):
        raw = [_entry(times_worn=1)]
        stats = InteractionBuilder.stats(builder.build(raw))
        for key in ["n_users", "n_garments", "total_cells", "non_zero", "sparsity"]:
            assert key in stats
