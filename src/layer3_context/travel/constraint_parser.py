"""ConstraintParser — converts raw user inputs into typed TravelConstraints.

Handles:
* Occasion strings like ``"dinner:3,tourism:3,beach:1"`` or lists.
* Fuzzy destination matching (lower-case normalisation, known alias table).
* Season and body_shape validation with sensible defaults.
* Date parsing (ISO YYYY-MM-DD) with automatic season inference.

All parsing is pure-Python, no external calls.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Dict, List, Optional, Union

from src.core.travel_models import TravelConstraints

logger = logging.getLogger(__name__)

# Known destination aliases → canonical name (lower-case)
_DESTINATION_ALIASES: Dict[str, str] = {
    "nyc": "new_york",
    "new york city": "new_york",
    "new york": "new_york",
    "paris": "paris",
    "london": "london",
    "rome": "rome",
    "tokyo": "tokyo",
    "bali": "bali",
    "dubai": "dubai",
    "barcelona": "barcelona",
    "delhi": "new_delhi",
    "new delhi": "new_delhi",
}

_VALID_BODY_SHAPES = {
    "rectangle", "hourglass", "pear", "apple", "inverted_triangle", "athletic"
}

_VALID_SEASONS = {"spring", "summer", "fall", "winter"}


class ConstraintParser:
    """Parse and validate travel/season constraints from user input.

    Usage
    -----
    ::

        parser = ConstraintParser()
        constraints = parser.parse(
            destination="Rome",
            days=7,
            occasions="tourism:4,evening:2,beach:1",
            max_pieces=12,
            body_shape="pear",
            season="summer",
        )
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        destination: str = "generic",
        days: int = 7,
        occasions: Optional[Union[str, Dict[str, int], List[str]]] = None,
        max_pieces: int = 12,
        travel_date: Optional[Union[str, date]] = None,
        body_shape: Optional[str] = None,
        season: Optional[str] = None,
    ) -> TravelConstraints:
        """Parse raw inputs into a ``TravelConstraints`` instance.

        Parameters
        ----------
        destination:
            Free-form city/country name.
        days:
            Number of trip days (clamped to 1-365).
        occasions:
            Occasions in any of three formats:
            * CSV string  ``"casual:4,evening:2"``
            * Dict        ``{"casual": 4, "evening": 2}``
            * List        ``["casual", "casual", "evening"]`` (each entry = 1 day)
        max_pieces:
            Maximum items to pack (clamped to 3-50).
        travel_date:
            ISO date string or ``date`` object.  Used to infer season when
            *season* is not provided.
        body_shape:
            One of the BodyType enum values.
        season:
            Explicit season override.

        Returns
        -------
        TravelConstraints
        """
        days = max(1, min(365, int(days)))
        max_pieces = max(3, min(50, int(max_pieces)))
        dest = self._normalize_destination(destination)
        parsed_date = self._parse_date(travel_date)
        parsed_season = self._resolve_season(season, parsed_date)
        parsed_occasions = self._parse_occasions(occasions, days)
        parsed_body_shape = self._validate_body_shape(body_shape)

        constraints = TravelConstraints(
            destination=dest,
            days=days,
            occasions=parsed_occasions,
            max_pieces=max_pieces,
            travel_date=parsed_date,
            body_shape=parsed_body_shape,
            season=parsed_season,
        )
        logger.debug("Parsed constraints: %s", constraints.model_dump())
        return constraints

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_destination(raw: str) -> str:
        raw = raw.strip().lower()
        return _DESTINATION_ALIASES.get(raw, raw.replace(" ", "_"))

    @staticmethod
    def _parse_date(raw) -> Optional[date]:
        if raw is None:
            return None
        if isinstance(raw, date):
            return raw
        try:
            return date.fromisoformat(str(raw))
        except (ValueError, TypeError):
            logger.warning("Could not parse travel_date='%s'; ignoring.", raw)
            return None

    @staticmethod
    def _resolve_season(season: Optional[str], travel_date: Optional[date]) -> Optional[str]:
        """Return season string, inferring from date if not provided."""
        if season:
            s = season.strip().lower()
            # normalise "autumn" → "fall"
            if s == "autumn":
                s = "fall"
            if s in _VALID_SEASONS:
                return s
            logger.warning("Unknown season '%s'; will infer from date.", season)

        if travel_date:
            month = travel_date.month
            if month in (3, 4, 5):
                return "spring"
            if month in (6, 7, 8):
                return "summer"
            if month in (9, 10, 11):
                return "fall"
            return "winter"

        return None

    @staticmethod
    def _parse_occasions(
        raw: Optional[Union[str, Dict[str, int], List[str]]], days: int
    ) -> Dict[str, int]:
        """Parse various occasion input formats into ``{occasion: days}``."""
        if raw is None:
            # Default: all casual days
            return {"casual": days}

        if isinstance(raw, dict):
            return {k.lower().strip(): int(v) for k, v in raw.items() if int(v) > 0}

        if isinstance(raw, list):
            result: Dict[str, int] = {}
            for item in raw:
                key = item.lower().strip()
                result[key] = result.get(key, 0) + 1
            return result

        if isinstance(raw, str):
            result = {}
            # Formats: "casual:4,evening:2" or "casual 4, evening 2"
            for part in re.split(r"[,;]", raw):
                part = part.strip()
                m = re.match(r"([a-zA-Z_]+)\s*[: ]\s*(\d+)", part)
                if m:
                    result[m.group(1).lower()] = int(m.group(2))
                elif part:
                    result[part.lower()] = result.get(part.lower(), 0) + 1
            return result if result else {"casual": days}

        return {"casual": days}

    @staticmethod
    def _validate_body_shape(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        cleaned = raw.strip().lower().replace("-", "_")
        if cleaned in _VALID_BODY_SHAPES:
            return cleaned
        logger.warning("Unknown body_shape '%s'; ignoring.", raw)
        return None
