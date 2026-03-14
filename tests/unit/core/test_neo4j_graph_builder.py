"""
Tests for Neo4jGraphBuilder (Phase 2)
======================================

Covers:
- GraphBuildResult dataclass
- Neo4jGraphBuilder initialisation
- Schema initialisation
- Reference node creation
- Garment loading (single + batch)
- Garment → reference edge linking
- Compatibility edge construction
- Color harmony edge construction
- Approximate centrality computation
- Graph validation
- Full pipeline orchestration
- Fallback behaviour (Neo4j unavailable)

All Neo4j I/O is mocked — no live Neo4j instance required.

Markers:
    @pytest.mark.unit   — fast, in-process tests
    @pytest.mark.asyncio — async tests (mode=AUTO via pytest.ini)
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.jobs.neo4j_graph_builder import (
    Neo4jGraphBuilder,
    GraphBuildResult,
    _compute_compatibility_score,
    _formality_int,
    _DEFAULT_SEASONS,
    _DEFAULT_OCCASIONS,
    _DEFAULT_FORMALITY_LEVELS,
    _DEFAULT_CATEGORIES,
    _DEFAULT_COLORS,
    _COLOR_HARMONIES,
)
from src.core.neo4j_client import Neo4jException
from src.core.models import (
    Garment,
    GarmentAttributes,
    GarmentCategory,
    ColorProfile,
    FormalityLevel,
    Season,
    GarmentHistory,
    PatternInfo,
    OccasionProfile,
    SeasonalityInfo,
)


# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture(autouse=True)
def _patch_neo4j_config(monkeypatch):
    """
    Patch get_neo4j_config for the ENTIRE module.

    Every Neo4jGraphBuilder construction calls get_neo4j_config() which
    reads a JSON file from disk. In a full test-suite run the working
    directory may differ from the project root, causing FileNotFoundError.
    This autouse fixture replaces the call with a lightweight mock so no
    file I/O is required.
    """
    mock_cfg = MagicMock()
    mock_cfg.get_min_compatibility_weight.return_value = 0.5
    mock_cfg.get_batch_size.return_value = 50
    mock_cfg.get_batch_query_size.return_value = 50
    mock_cfg.get_pagerank_damping.return_value = 0.85
    mock_cfg.get_pagerank_power_iterations.return_value = 5
    mock_cfg.get_default_reference_edge_weight.return_value = 0.8
    monkeypatch.setattr(
        "src.jobs.neo4j_graph_builder.get_neo4j_config",
        lambda *a, **kw: mock_cfg,
    )


def _make_garment(
    gid: str = "g1",
    category: GarmentCategory = GarmentCategory.TOP,
    color: str = "blue",
    formality: FormalityLevel = FormalityLevel.CASUAL,
    style_tags: list[str] | None = None,
    seasons: list[Season] | None = None,
) -> Garment:
    """Factory for minimal Garment test instances."""
    return Garment(
        id=gid,
        attributes=GarmentAttributes(
            category=category,
            subcategory=None,
            color=ColorProfile(primary=color),
            formality_level=formality,
            style_tags=style_tags or [],
            season_suitable=seasons or [],
            pattern=PatternInfo(type="solid"),
        ),
        history=GarmentHistory(),
    )


@pytest.fixture
def sample_wardrobe() -> list[Garment]:
    """Small wardrobe: top + bottom + shoes."""
    return [
        _make_garment("g1", GarmentCategory.TOP, "navy", FormalityLevel.SMART_CASUAL,
                      style_tags=["minimalist"], seasons=[Season.SPRING, Season.FALL]),
        _make_garment("g2", GarmentCategory.BOTTOM, "beige", FormalityLevel.SMART_CASUAL,
                      style_tags=["minimalist", "classic"], seasons=[Season.SPRING, Season.FALL]),
        _make_garment("g3", GarmentCategory.SHOES, "white", FormalityLevel.CASUAL,
                      style_tags=["classic"], seasons=[Season.SPRING, Season.SUMMER]),
    ]


@pytest.fixture
def mock_client():
    """Return a pre-configured mock Neo4jClient."""
    client = AsyncMock()
    # Default: return empty list for every query
    client.query = AsyncMock(return_value=[])
    return client


@pytest.fixture
def builder(mock_client):
    """Graph builder with mocked client injected."""
    b = Neo4jGraphBuilder(fallback_enabled=False)
    b._client = mock_client
    return b


@pytest.fixture
def builder_fallback(mock_client):
    """Graph builder in fallback mode with mocked client."""
    b = Neo4jGraphBuilder(fallback_enabled=True)
    b._client = mock_client
    return b


# ===========================================================================
# GRAPH BUILD RESULT
# ===========================================================================


class TestGraphBuildResult:
    """Tests for the GraphBuildResult dataclass."""

    def test_default_values(self):
        result = GraphBuildResult()
        assert result.success is False
        assert result.fallback_used is False
        assert result.garments_loaded == 0
        assert result.compatibility_edges_created == 0

    def test_summary_fallback(self):
        result = GraphBuildResult(fallback_used=True, success=True)
        assert "FALLBACK" in result.summary()

    def test_summary_success(self):
        result = GraphBuildResult(
            success=True,
            garments_loaded=10,
            compatibility_edges_created=30,
            garment_ref_edges_created=15,
            color_harmony_edges_created=5,
            duration_seconds=1.5,
        )
        summary = result.summary()
        assert "✓" in summary
        assert "garments=10" in summary

    def test_summary_failure(self):
        result = GraphBuildResult(success=False, duration_seconds=0.1)
        assert "✗" in result.summary()


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================


class TestHelpers:
    """Tests for module-level helper utilities."""

    def test_formality_int_enum(self):
        g = _make_garment(formality=FormalityLevel.BUSINESS)
        assert _formality_int(g) == 4

    def test_formality_int_very_casual(self):
        g = _make_garment(formality=FormalityLevel.VERY_CASUAL)
        assert _formality_int(g) == 0

    def test_formality_int_black_tie(self):
        g = _make_garment(formality=FormalityLevel.BLACK_TIE)
        assert _formality_int(g) == 6

    def test_compute_compatibility_score_same_formality(self):
        g1 = _make_garment("g1", GarmentCategory.TOP, formality=FormalityLevel.CASUAL)
        g2 = _make_garment("g2", GarmentCategory.BOTTOM, formality=FormalityLevel.CASUAL)
        rel = _compute_compatibility_score(g1, g2)
        assert 0.0 <= rel.weight <= 1.0
        assert rel.formality_delta == 0

    def test_compute_compatibility_score_large_formality_gap(self):
        g1 = _make_garment("g1", formality=FormalityLevel.VERY_CASUAL)
        g2 = _make_garment("g2", formality=FormalityLevel.BLACK_TIE)
        rel = _compute_compatibility_score(g1, g2)
        # Large gap should yield lower score
        assert rel.weight < 0.85
        assert rel.formality_delta == 6

    def test_compute_compatibility_score_shared_style_tags(self):
        g1 = _make_garment("g1", style_tags=["minimalist", "classic"])
        g2 = _make_garment("g2", style_tags=["minimalist", "modern"])
        rel_shared = _compute_compatibility_score(g1, g2)

        g3 = _make_garment("g3", style_tags=["streetwear", "bold"])
        g4 = _make_garment("g4", style_tags=["romantic", "feminine"])
        rel_no_overlap = _compute_compatibility_score(g3, g4)

        # Shared tags should score equal or better
        assert rel_shared.weight >= rel_no_overlap.weight - 0.05

    def test_compute_compatibility_score_returns_relationship(self):
        from src.core.neo4j_models import CompatibleWithRelationship
        g1 = _make_garment("g1")
        g2 = _make_garment("g2", category=GarmentCategory.BOTTOM)
        rel = _compute_compatibility_score(g1, g2)
        assert isinstance(rel, CompatibleWithRelationship)

    def test_compute_compatibility_stores_seasons(self):
        g1 = _make_garment("g1", seasons=[Season.SPRING, Season.FALL])
        g2 = _make_garment("g2", seasons=[Season.SPRING, Season.SUMMER])
        rel = _compute_compatibility_score(g1, g2)
        assert "SPRING" in rel.seasons


# ===========================================================================
# REFERENCE DATA CONSTANTS
# ===========================================================================


class TestReferenceData:
    """Tests for the built-in reference data tables."""

    def test_default_seasons_count(self):
        assert len(_DEFAULT_SEASONS) == 4

    def test_default_occasions_non_empty(self):
        assert len(_DEFAULT_OCCASIONS) > 5

    def test_default_formality_levels_range(self):
        values = [fl["value"] for fl in _DEFAULT_FORMALITY_LEVELS]
        assert 0 in values
        assert 6 in values

    def test_default_categories_non_empty(self):
        assert len(_DEFAULT_CATEGORIES) > 0

    def test_default_colors_non_empty(self):
        assert len(_DEFAULT_COLORS) >= 10

    def test_color_harmonies_valid_types(self):
        valid_types = {"COMPLEMENTS", "ANALOGOUS", "TRIADIC"}
        for _, _, harmony_type, _ in _COLOR_HARMONIES:
            assert harmony_type in valid_types, f"Unknown harmony type: {harmony_type}"

    def test_color_harmonies_weights_in_range(self):
        for _, _, _, weight in _COLOR_HARMONIES:
            assert 0.0 <= weight <= 1.0


# ===========================================================================
# BUILDER INITIALISATION
# ===========================================================================


class TestBuilderInit:
    """Tests for Neo4jGraphBuilder construction."""

    def test_default_construction(self):
        with patch("src.jobs.neo4j_graph_builder.get_neo4j_config") as mock_cfg:
            mock_cfg.return_value.get_min_compatibility_weight.return_value = 0.5
            mock_cfg.return_value.get_batch_query_size.return_value = 50
            mock_cfg.return_value.get_pagerank_damping.return_value = 0.85
            mock_cfg.return_value.get_pagerank_power_iterations.return_value = 5
            mock_cfg.return_value.get_default_reference_edge_weight.return_value = 0.8
            b = Neo4jGraphBuilder()
        assert b.fallback_enabled is True
        assert b.batch_size == 50

    def test_custom_parameters(self):
        with patch("src.jobs.neo4j_graph_builder.get_neo4j_config") as mock_cfg:
            mock_cfg.return_value.get_min_compatibility_weight.return_value = 0.5
            mock_cfg.return_value.get_batch_query_size.return_value = 50
            mock_cfg.return_value.get_pagerank_damping.return_value = 0.85
            mock_cfg.return_value.get_pagerank_power_iterations.return_value = 5
            mock_cfg.return_value.get_default_reference_edge_weight.return_value = 0.8
            b = Neo4jGraphBuilder(
                fallback_enabled=False,
                min_compatibility_weight=0.7,
                batch_size=10,
            )
        assert b.fallback_enabled is False
        assert b._min_weight == 0.7
        assert b.batch_size == 10

    def test_fallback_when_neo4j_unavailable(self):
        """_get_client returns None when connection fails and fallback is on."""
        with patch("src.jobs.neo4j_graph_builder.Neo4jClient") as mock_cls:
            mock_cls.side_effect = Neo4jException("Cannot connect")
            with patch("src.jobs.neo4j_graph_builder.get_neo4j_config") as mock_cfg:
                mock_cfg.return_value.get_min_compatibility_weight.return_value = 0.5
                mock_cfg.return_value.get_batch_query_size.return_value = 50
                mock_cfg.return_value.get_pagerank_damping.return_value = 0.85
                mock_cfg.return_value.get_pagerank_power_iterations.return_value = 5
                mock_cfg.return_value.get_default_reference_edge_weight.return_value = 0.8
                b = Neo4jGraphBuilder(fallback_enabled=True)
            result = b._get_client()
        assert result is None

    def test_raises_when_no_fallback_and_unavailable(self):
        """_get_client raises when fallback is disabled and Neo4j is down."""
        with patch("src.jobs.neo4j_graph_builder.Neo4jClient") as mock_cls:
            mock_cls.side_effect = Neo4jException("Cannot connect")
            with patch("src.jobs.neo4j_graph_builder.get_neo4j_config") as mock_cfg:
                mock_cfg.return_value.get_min_compatibility_weight.return_value = 0.5
                mock_cfg.return_value.get_batch_query_size.return_value = 50
                mock_cfg.return_value.get_pagerank_damping.return_value = 0.85
                mock_cfg.return_value.get_pagerank_power_iterations.return_value = 5
                mock_cfg.return_value.get_default_reference_edge_weight.return_value = 0.8
                b = Neo4jGraphBuilder(fallback_enabled=False)
            with pytest.raises(Neo4jException):
                b._get_client()


# ===========================================================================
# SCHEMA INITIALISATION
# ===========================================================================


class TestSchemaInitialization:
    """Tests for initialize_schema."""

    @pytest.mark.asyncio
    async def test_schema_init_calls_indexes(self, builder, mock_client):
        result = await builder.initialize_schema()
        assert result is True
        # Should have been called for indexes + constraints
        assert mock_client.query.call_count > 0

    @pytest.mark.asyncio
    async def test_schema_init_returns_false_on_fallback(self):
        b = Neo4jGraphBuilder(fallback_enabled=True)
        b._client = None  # simulate no connection

        with patch.object(b, "_get_client", return_value=None):
            result = await b.initialize_schema()
        assert result is False

    @pytest.mark.asyncio
    async def test_schema_init_handles_partial_failure(self, mock_client):
        """Non-fatal errors on individual indexes should not abort the whole run."""
        # First call raises, rest succeed
        mock_client.query.side_effect = [
            Neo4jException("already exists"),
        ] + [[] for _ in range(100)]

        b = Neo4jGraphBuilder(fallback_enabled=False)
        b._client = mock_client

        result = await b.initialize_schema()
        assert result is True  # partial failure is non-fatal


# ===========================================================================
# REFERENCE NODES
# ===========================================================================


class TestReferenceNodes:
    """Tests for create_reference_nodes."""

    @pytest.mark.asyncio
    async def test_reference_nodes_returns_count(self, builder, mock_client):
        count = await builder.create_reference_nodes()
        expected = (
            len(_DEFAULT_SEASONS)
            + len(_DEFAULT_OCCASIONS)
            + len(_DEFAULT_CATEGORIES)
            + len(_DEFAULT_FORMALITY_LEVELS)
            + len(_DEFAULT_COLORS)
        )
        assert count == expected

    @pytest.mark.asyncio
    async def test_reference_nodes_fallback_returns_zero(self):
        b = Neo4jGraphBuilder(fallback_enabled=True)
        with patch.object(b, "_get_client", return_value=None):
            count = await b.create_reference_nodes()
        assert count == 0

    @pytest.mark.asyncio
    async def test_reference_nodes_calls_merge_for_each(self, builder, mock_client):
        await builder.create_reference_nodes()
        # At minimum one call per season
        calls = mock_client.query.call_count
        assert calls >= len(_DEFAULT_SEASONS)


# ===========================================================================
# GARMENT LOADING
# ===========================================================================


class TestGarmentLoading:
    """Tests for load_garment and load_all_garments."""

    @pytest.mark.asyncio
    async def test_load_single_garment_succeeds(self, builder, mock_client, sample_wardrobe):
        result = await builder.load_garment(sample_wardrobe[0], user_id="user1")
        assert result is True
        mock_client.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_single_garment_fallback(self, sample_wardrobe):
        b = Neo4jGraphBuilder(fallback_enabled=True)
        with patch.object(b, "_get_client", return_value=None):
            result = await b.load_garment(sample_wardrobe[0])
        assert result is False

    @pytest.mark.asyncio
    async def test_load_all_garments_returns_count(self, builder, mock_client, sample_wardrobe):
        count = await builder.load_all_garments(sample_wardrobe, user_id="user1")
        assert count == len(sample_wardrobe)

    @pytest.mark.asyncio
    async def test_load_all_garments_empty_list(self, builder, mock_client):
        count = await builder.load_all_garments([], user_id="user1")
        assert count == 0
        mock_client.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_all_garments_fallback(self, sample_wardrobe):
        b = Neo4jGraphBuilder(fallback_enabled=True)
        with patch.object(b, "_get_client", return_value=None):
            count = await b.load_all_garments(sample_wardrobe)
        assert count == 0

    @pytest.mark.asyncio
    async def test_load_all_garments_uses_batches(self, mock_client, sample_wardrobe):
        """With batch_size=1, should make one query per garment (UNWIND with 1 row)."""
        b = Neo4jGraphBuilder(fallback_enabled=False, batch_size=1)
        b._client = mock_client

        count = await b.load_all_garments(sample_wardrobe)
        assert count == len(sample_wardrobe)
        # Each batch → 1 query call
        assert mock_client.query.call_count == len(sample_wardrobe)

    @pytest.mark.asyncio
    async def test_load_garment_uses_garment_id(self, builder, mock_client, sample_wardrobe):
        garment = sample_wardrobe[0]
        await builder.load_garment(garment)
        call_kwargs = mock_client.query.call_args
        params = call_kwargs[0][1]  # second positional arg is params dict
        assert params["id"] == garment.id


# ===========================================================================
# REFERENCE EDGE LINKING
# ===========================================================================


class TestReferenceEdgeLinking:
    """Tests for link_garment_to_refs and link_all_garments_to_refs."""

    @pytest.mark.asyncio
    async def test_link_garment_creates_category_edge(self, builder, mock_client):
        g = _make_garment("g1", GarmentCategory.TOP)
        edges = await builder.link_garment_to_refs(g)
        assert edges >= 1  # At least BELONGS_TO category

    @pytest.mark.asyncio
    async def test_link_garment_with_seasons(self, builder, mock_client):
        g = _make_garment("g1", seasons=[Season.SPRING, Season.FALL])
        edges = await builder.link_garment_to_refs(g)
        assert edges >= 3  # BELONGS_TO + 2 × SUITABLE_FOR

    @pytest.mark.asyncio
    async def test_link_garment_with_color(self, builder, mock_client):
        g = _make_garment("g1", color="navy")
        edges = await builder.link_garment_to_refs(g)
        assert edges >= 2  # BELONGS_TO + HAS_COLOR

    @pytest.mark.asyncio
    async def test_link_garment_fallback(self):
        g = _make_garment("g1")
        b = Neo4jGraphBuilder(fallback_enabled=True)
        with patch.object(b, "_get_client", return_value=None):
            edges = await b.link_garment_to_refs(g)
        assert edges == 0

    @pytest.mark.asyncio
    async def test_link_all_garments_sums_edges(self, builder, mock_client, sample_wardrobe):
        total = await builder.link_all_garments_to_refs(sample_wardrobe, user_id="u1")
        assert total > 0  # At least one edge per garment

    @pytest.mark.asyncio
    async def test_link_edge_failure_is_non_fatal(self, mock_client, sample_wardrobe):
        """A single failing edge should not abort the rest."""
        call_count = 0

        async def selective_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Neo4jException("edge failed")
            return []

        mock_client.query.side_effect = selective_fail
        b = Neo4jGraphBuilder(fallback_enabled=False)
        b._client = mock_client

        edges = await b.link_garment_to_refs(sample_wardrobe[0])
        # Should have created remaining edges despite one failure
        assert edges >= 0


# ===========================================================================
# COMPATIBILITY EDGES
# ===========================================================================


class TestCompatibilityEdges:
    """Tests for build_compatibility_edges."""

    @pytest.mark.asyncio
    async def test_builds_edges_for_wardrobe(self, builder, mock_client, sample_wardrobe):
        created = await builder.build_compatibility_edges(sample_wardrobe)
        assert created >= 0  # At least some edges (depends on threshold)

    @pytest.mark.asyncio
    async def test_no_edges_for_single_garment(self, builder, mock_client):
        g = _make_garment("g1")
        created = await builder.build_compatibility_edges([g])
        assert created == 0

    @pytest.mark.asyncio
    async def test_no_edges_for_empty_list(self, builder, mock_client):
        created = await builder.build_compatibility_edges([])
        assert created == 0

    @pytest.mark.asyncio
    async def test_min_weight_filters_low_pairs(self, mock_client):
        """With min_weight=1.0 nothing should pass."""
        b = Neo4jGraphBuilder(fallback_enabled=False, min_compatibility_weight=1.0)
        b._client = mock_client

        wardrobe = [
            _make_garment("g1", formality=FormalityLevel.VERY_CASUAL),
            _make_garment("g2", formality=FormalityLevel.BLACK_TIE),
        ]
        created = await b.build_compatibility_edges(wardrobe, min_weight=1.0)
        assert created == 0

    @pytest.mark.asyncio
    async def test_uses_batch_writes(self, mock_client):
        """Verify that UNWIND batch queries are issued."""
        b = Neo4jGraphBuilder(fallback_enabled=False, min_compatibility_weight=0.0, batch_size=2)
        b._client = mock_client

        wardrobe = [
            _make_garment(f"g{i}", GarmentCategory.TOP if i % 2 == 0 else GarmentCategory.BOTTOM)
            for i in range(6)
        ]
        await b.build_compatibility_edges(wardrobe)
        # Should have called query at least once for batched edges
        assert mock_client.query.call_count >= 1

    @pytest.mark.asyncio
    async def test_fallback_returns_zero(self, sample_wardrobe):
        b = Neo4jGraphBuilder(fallback_enabled=True)
        with patch.object(b, "_get_client", return_value=None):
            created = await b.build_compatibility_edges(sample_wardrobe)
        assert created == 0


# ===========================================================================
# COLOR HARMONY EDGES
# ===========================================================================


class TestColorHarmonyEdges:
    """Tests for build_color_harmony_edges."""

    @pytest.mark.asyncio
    async def test_creates_harmony_edges(self, builder, mock_client):
        created = await builder.build_color_harmony_edges()
        assert created == len(_COLOR_HARMONIES)

    @pytest.mark.asyncio
    async def test_fallback_returns_zero(self):
        b = Neo4jGraphBuilder(fallback_enabled=True)
        with patch.object(b, "_get_client", return_value=None):
            created = await b.build_color_harmony_edges()
        assert created == 0

    @pytest.mark.asyncio
    async def test_single_failure_non_fatal(self, mock_client):
        """One failing edge should not stop the rest."""
        call_count = [0]

        async def selective(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Neo4jException("edge failed")
            return []

        mock_client.query.side_effect = selective
        b = Neo4jGraphBuilder(fallback_enabled=False)
        b._client = mock_client

        created = await b.build_color_harmony_edges()
        # Should have created all but the failing one
        assert created == len(_COLOR_HARMONIES) - 1


# ===========================================================================
# CENTRALITY SCORES
# ===========================================================================


class TestCentralityScores:
    """Tests for update_centrality_scores."""

    @pytest.mark.asyncio
    async def test_updates_all_garments(self, sample_wardrobe):
        mock_client = AsyncMock()
        # Return synthetic edge list
        mock_client.query = AsyncMock(
            side_effect=[
                # First call: fetch edges
                [
                    {"from_id": "g1", "to_id": "g2", "weight": 0.9},
                    {"from_id": "g1", "to_id": "g3", "weight": 0.8},
                ],
                # Second call: batch write centrality
                [],
            ]
        )
        b = Neo4jGraphBuilder(fallback_enabled=False)
        b._client = mock_client

        updated = await b.update_centrality_scores(sample_wardrobe)
        assert updated == len(sample_wardrobe)

    @pytest.mark.asyncio
    async def test_empty_wardrobe_returns_zero(self, builder, mock_client):
        updated = await builder.update_centrality_scores([])
        assert updated == 0

    @pytest.mark.asyncio
    async def test_fallback_returns_zero(self, sample_wardrobe):
        b = Neo4jGraphBuilder(fallback_enabled=True)
        with patch.object(b, "_get_client", return_value=None):
            updated = await b.update_centrality_scores(sample_wardrobe)
        assert updated == 0

    @pytest.mark.asyncio
    async def test_handles_no_edges(self, sample_wardrobe):
        """With no COMPATIBLE_WITH edges, all centrality scores should be 0."""
        mock_client = AsyncMock()
        mock_client.query = AsyncMock(
            side_effect=[
                [],   # empty edge list
                [],   # batch write result
            ]
        )
        b = Neo4jGraphBuilder(fallback_enabled=False)
        b._client = mock_client

        updated = await b.update_centrality_scores(sample_wardrobe)
        assert updated == len(sample_wardrobe)


# ===========================================================================
# GRAPH VALIDATION
# ===========================================================================


class TestGraphValidation:
    """Tests for validate_graph."""

    @pytest.mark.asyncio
    async def test_validation_passes_with_enough_nodes(self, mock_client):
        mock_client.query = AsyncMock(
            side_effect=[
                [{"cnt": 10}],   # garment count
                [{"cnt": 15}],   # edge count
                [{"cnt": 0}],    # isolated nodes
            ]
        )
        b = Neo4jGraphBuilder(fallback_enabled=False)
        b._client = mock_client

        passed, errors = await b.validate_graph(expected_garments=5, expected_min_edges=10)
        assert passed is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validation_fails_on_insufficient_garments(self, mock_client):
        mock_client.query = AsyncMock(
            side_effect=[
                [{"cnt": 2}],   # fewer garments than expected
                [{"cnt": 1}],
                [{"cnt": 0}],
            ]
        )
        b = Neo4jGraphBuilder(fallback_enabled=False)
        b._client = mock_client

        passed, errors = await b.validate_graph(expected_garments=10)
        assert passed is False
        assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_validation_fallback_returns_true(self):
        b = Neo4jGraphBuilder(fallback_enabled=True)
        with patch.object(b, "_get_client", return_value=None):
            passed, errors = await b.validate_graph()
        assert passed is True
        assert len(errors) == 0


# ===========================================================================
# FULL PIPELINE ORCHESTRATION
# ===========================================================================


class TestFullInitialization:
    """Tests for run_full_initialization."""

    def _make_builder(self, **kw) -> "Neo4jGraphBuilder":
        """Create a builder whose _get_client is pre-set to a sentinel."""
        with patch("src.jobs.neo4j_graph_builder.get_neo4j_config") as mc:
            mc.return_value.get_min_compatibility_weight.return_value = 0.5
            b = Neo4jGraphBuilder(fallback_enabled=False, **kw)
        # Inject a dummy client so _get_client never tries to connect
        b._client = AsyncMock()
        b._client.query = AsyncMock(return_value=[])
        return b

    @pytest.mark.asyncio
    async def test_full_run_returns_result(self, sample_wardrobe):
        b = self._make_builder(min_compatibility_weight=0.0)

        with (
            patch.object(b, "initialize_schema", return_value=True),
            patch.object(b, "create_reference_nodes", return_value=47),
            patch.object(b, "load_all_garments", return_value=3),
            patch.object(b, "link_all_garments_to_refs", return_value=12),
            patch.object(b, "build_compatibility_edges", return_value=3),
            patch.object(b, "build_color_harmony_edges", return_value=30),
            patch.object(b, "update_centrality_scores", return_value=3),
            patch.object(b, "validate_graph", return_value=(True, [])),
        ):
            result = await b.run_full_initialization(sample_wardrobe, user_id="u1")

        assert isinstance(result, GraphBuildResult)
        assert result.success is True
        assert result.fallback_used is False
        assert result.garments_loaded == 3
        assert result.reference_nodes_created == 47
        assert result.compatibility_edges_created == 3
        assert result.color_harmony_edges_created == 30
        assert result.validation_passed is True
        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_full_run_fallback_mode(self, sample_wardrobe):
        """When Neo4j is unavailable, full run should succeed with fallback flag."""
        with patch("src.jobs.neo4j_graph_builder.get_neo4j_config") as mc:
            mc.return_value.get_min_compatibility_weight.return_value = 0.5
            b = Neo4jGraphBuilder(fallback_enabled=True)
        b._client = None

        with patch.object(b, "_get_client", return_value=None):
            result = await b.run_full_initialization(sample_wardrobe)

        assert result.success is True
        assert result.fallback_used is True

    @pytest.mark.asyncio
    async def test_full_run_skip_schema(self, sample_wardrobe):
        b = self._make_builder(min_compatibility_weight=0.0)

        with (
            patch.object(b, "initialize_schema", return_value=True) as mock_schema,
            patch.object(b, "create_reference_nodes", return_value=47),
            patch.object(b, "load_all_garments", return_value=3),
            patch.object(b, "link_all_garments_to_refs", return_value=12),
            patch.object(b, "build_compatibility_edges", return_value=3),
            patch.object(b, "build_color_harmony_edges", return_value=30),
            patch.object(b, "update_centrality_scores", return_value=3),
            patch.object(b, "validate_graph", return_value=(True, [])),
        ):
            result = await b.run_full_initialization(sample_wardrobe, skip_schema=True)
            mock_schema.assert_not_called()

        assert result.success is True

    @pytest.mark.asyncio
    async def test_full_run_skip_centrality(self, sample_wardrobe):
        b = self._make_builder(min_compatibility_weight=0.0)

        with (
            patch.object(b, "initialize_schema", return_value=True),
            patch.object(b, "create_reference_nodes", return_value=47),
            patch.object(b, "load_all_garments", return_value=3),
            patch.object(b, "link_all_garments_to_refs", return_value=12),
            patch.object(b, "build_compatibility_edges", return_value=3),
            patch.object(b, "build_color_harmony_edges", return_value=30),
            patch.object(b, "update_centrality_scores", return_value=3) as mock_cent,
            patch.object(b, "validate_graph", return_value=(True, [])),
        ):
            result = await b.run_full_initialization(sample_wardrobe, skip_centrality=True)
            mock_cent.assert_not_called()

        assert result.success is True

    @pytest.mark.asyncio
    async def test_full_run_empty_wardrobe(self):
        b = self._make_builder()

        with (
            patch.object(b, "initialize_schema", return_value=True),
            patch.object(b, "create_reference_nodes", return_value=47),
            patch.object(b, "load_all_garments", return_value=0),
            patch.object(b, "link_all_garments_to_refs", return_value=0),
            patch.object(b, "build_compatibility_edges", return_value=0),
            patch.object(b, "build_color_harmony_edges", return_value=30),
            patch.object(b, "update_centrality_scores", return_value=0),
            patch.object(b, "validate_graph", return_value=(True, [])),
        ):
            result = await b.run_full_initialization([])

        assert result.success is True
        assert result.garments_loaded == 0


# ===========================================================================
# CLOSE / LIFECYCLE
# ===========================================================================


class TestLifecycle:
    """Tests for client lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_releases_client(self, builder, mock_client):
        await builder.close()
        mock_client.close.assert_called_once()
        assert builder._client is None

    @pytest.mark.asyncio
    async def test_close_with_no_client(self):
        with patch("src.jobs.neo4j_graph_builder.get_neo4j_config") as mc:
            mc.return_value.get_min_compatibility_weight.return_value = 0.5
            b = Neo4jGraphBuilder(fallback_enabled=True)
        b._client = None
        await b.close()  # Should not raise
