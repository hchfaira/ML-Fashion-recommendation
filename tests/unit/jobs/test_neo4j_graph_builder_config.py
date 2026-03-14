"""
Tests for Neo4jGraphBuilder — config-driven domain knowledge
============================================================

Verifies that the module-level reference data (_DEFAULT_SEASONS,
_DEFAULT_OCCASIONS, _DEFAULT_COLORS, _COLOR_HARMONIES, etc.) is
loaded from config/data/style_rules_config.json, not hardcoded.

No live Neo4j instance is required.

Markers:
    @pytest.mark.unit
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.jobs.neo4j_graph_builder import (
    _DEFAULT_SEASONS,
    _DEFAULT_OCCASIONS,
    _DEFAULT_FORMALITY_LEVELS,
    _DEFAULT_CATEGORIES,
    _DEFAULT_COLORS,
    _COLOR_HARMONIES,
    Neo4jGraphBuilder,
)

_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "config" / "data" / "style_rules_config.json"
)

_PATCH_NEO4J_CONFIG = "src.jobs.neo4j_graph_builder.get_neo4j_config"


@pytest.fixture(autouse=True)
def _patch_neo4j_config(monkeypatch):
    """Prevent get_neo4j_config() from touching disk / Neo4j."""
    mock_cfg = MagicMock()
    mock_cfg.get_min_compatibility_weight.return_value = 0.5
    mock_cfg.get_batch_query_size.return_value = 100
    mock_cfg.get_pagerank_damping.return_value = 0.85
    mock_cfg.get_pagerank_power_iterations.return_value = 5
    mock_cfg.get_default_reference_edge_weight.return_value = 0.8
    monkeypatch.setattr("src.jobs.neo4j_graph_builder.get_neo4j_config", lambda: mock_cfg)


@pytest.fixture
def _cfg():
    with open(_CONFIG_PATH) as f:
        return json.load(f)["neo4j_graph"]


# ============================================================
# Module-level reference data comes from style_rules_config.json
# ============================================================

@pytest.mark.unit
class TestGraphBuilderReferenceDataFromConfig:
    """All module-level reference data must match style_rules_config.json."""

    def test_seasons_match_config(self, _cfg):
        assert _DEFAULT_SEASONS == _cfg["seasons"]

    def test_occasions_match_config(self, _cfg):
        assert _DEFAULT_OCCASIONS == _cfg["occasions"]

    def test_formality_levels_match_config(self, _cfg):
        assert _DEFAULT_FORMALITY_LEVELS == _cfg["formality_levels"]

    def test_categories_match_config(self, _cfg):
        assert _DEFAULT_CATEGORIES == _cfg["categories"]

    def test_colors_match_config(self, _cfg):
        assert _DEFAULT_COLORS == _cfg["colors"]

    def test_color_harmonies_count_matches_config(self, _cfg):
        """Number of color harmony tuples matches config entries."""
        assert len(_COLOR_HARMONIES) == len(_cfg["color_harmonies"])

    def test_color_harmonies_first_entry(self, _cfg):
        """First color harmony entry is correctly parsed from config."""
        first_cfg = _cfg["color_harmonies"][0]
        first_code = _COLOR_HARMONIES[0]
        assert first_code == (
            first_cfg["from"], first_cfg["to"], first_cfg["type"], first_cfg["weight"]
        )

    def test_color_harmonies_are_tuples(self):
        """Color harmonies are stored as 4-tuples (from, to, type, weight)."""
        for entry in _COLOR_HARMONIES:
            assert isinstance(entry, tuple)
            assert len(entry) == 4
            assert isinstance(entry[3], float)


# ============================================================
# Builder initialization reads operational config
# ============================================================

@pytest.mark.unit
class TestGraphBuilderInit:
    """Neo4jGraphBuilder reads batch_size, damping, etc. from config."""

    def test_default_batch_size_from_config(self):
        """batch_size defaults to config value (100) when constructor gets 50."""
        builder = Neo4jGraphBuilder()
        assert builder.batch_size == 100   # from mock cfg.get_batch_query_size()

    def test_explicit_batch_size_overrides_config(self):
        """Explicitly passing batch_size overrides the config value."""
        builder = Neo4jGraphBuilder(batch_size=25)
        assert builder.batch_size == 25

    def test_pagerank_damping_from_config(self):
        builder = Neo4jGraphBuilder()
        assert builder._pagerank_damping == 0.85

    def test_pagerank_iterations_from_config(self):
        builder = Neo4jGraphBuilder()
        assert builder._pagerank_iterations == 5

    def test_reference_edge_weight_from_config(self):
        builder = Neo4jGraphBuilder()
        assert builder._reference_edge_weight == 0.8
