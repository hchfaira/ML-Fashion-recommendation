"""Tests — ConstraintParser"""
import pytest
from datetime import date

from src.layer3_context.travel.constraint_parser import ConstraintParser


@pytest.fixture
def parser():
    return ConstraintParser()


class TestDestinationNormalisation:
    def test_lowercase_passthrough(self, parser):
        c = parser.parse(destination="paris")
        assert c.destination == "paris"

    def test_case_insensitive(self, parser):
        c = parser.parse(destination="PARIS")
        assert c.destination == "paris"

    def test_alias_nyc(self, parser):
        c = parser.parse(destination="NYC")
        assert c.destination == "new_york"

    def test_alias_new_york(self, parser):
        c = parser.parse(destination="New York")
        assert c.destination == "new_york"

    def test_space_becomes_underscore(self, parser):
        c = parser.parse(destination="San Francisco")
        assert c.destination == "san_francisco"

    def test_default_generic(self, parser):
        c = parser.parse()
        assert c.destination == "generic"


class TestOccasionParsing:
    def test_csv_string(self, parser):
        c = parser.parse(occasions="casual:4,evening:2,beach:1")
        assert c.occasions == {"casual": 4, "evening": 2, "beach": 1}

    def test_semicolon_separator(self, parser):
        c = parser.parse(occasions="casual:4;evening:2")
        assert c.occasions == {"casual": 4, "evening": 2}

    def test_dict_input(self, parser):
        c = parser.parse(occasions={"casual": 5, "business": 2})
        assert c.occasions["casual"] == 5

    def test_list_input(self, parser):
        c = parser.parse(occasions=["casual", "casual", "evening"])
        assert c.occasions["casual"] == 2
        assert c.occasions["evening"] == 1

    def test_none_defaults_to_casual(self, parser):
        c = parser.parse(days=3, occasions=None)
        assert c.occasions == {"casual": 3}

    def test_zero_value_filtered(self, parser):
        c = parser.parse(occasions={"casual": 3, "beach": 0})
        assert "beach" not in c.occasions

    def test_keys_lowercased(self, parser):
        c = parser.parse(occasions="CASUAL:3,EVENING:2")
        assert "casual" in c.occasions
        assert "evening" in c.occasions


class TestDaysAndPieces:
    def test_days_clamped_min(self, parser):
        c = parser.parse(days=0)
        assert c.days == 1

    def test_days_clamped_max(self, parser):
        c = parser.parse(days=1000)
        assert c.days == 365

    def test_max_pieces_clamped(self, parser):
        c = parser.parse(max_pieces=100)
        assert c.max_pieces == 50

    def test_max_pieces_min(self, parser):
        c = parser.parse(max_pieces=1)
        assert c.max_pieces == 3

    def test_valid_max_pieces(self, parser):
        c = parser.parse(max_pieces=12)
        assert c.max_pieces == 12


class TestSeasonInference:
    def test_explicit_season(self, parser):
        c = parser.parse(season="winter")
        assert c.season == "winter"

    def test_autumn_normalised_to_fall(self, parser):
        c = parser.parse(season="autumn")
        assert c.season == "fall"

    def test_infer_spring_from_date(self, parser):
        c = parser.parse(travel_date="2026-04-15", season=None)
        assert c.season == "spring"

    def test_infer_summer_from_date(self, parser):
        c = parser.parse(travel_date="2026-07-01", season=None)
        assert c.season == "summer"

    def test_infer_fall_from_date(self, parser):
        c = parser.parse(travel_date="2026-10-20", season=None)
        assert c.season == "fall"

    def test_infer_winter_from_date(self, parser):
        c = parser.parse(travel_date="2026-01-10", season=None)
        assert c.season == "winter"

    def test_date_object_accepted(self, parser):
        c = parser.parse(travel_date=date(2026, 6, 15))
        assert c.season == "summer"

    def test_invalid_date_ignored(self, parser):
        c = parser.parse(travel_date="not-a-date")
        assert c.travel_date is None


class TestBodyShape:
    def test_valid_body_shape(self, parser):
        c = parser.parse(body_shape="pear")
        assert c.body_shape == "pear"

    def test_case_insensitive_body_shape(self, parser):
        c = parser.parse(body_shape="HOURGLASS")
        assert c.body_shape == "hourglass"

    def test_invalid_body_shape_returns_none(self, parser):
        c = parser.parse(body_shape="triangle")
        assert c.body_shape is None

    def test_hyphen_normalised(self, parser):
        c = parser.parse(body_shape="inverted-triangle")
        assert c.body_shape == "inverted_triangle"
