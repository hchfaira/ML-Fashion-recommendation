"""
WeatherOutfitAdvisor
====================
Translates a ``WeatherData`` instance into concrete outfit-scoring advice:

  * ``penalise_item(garment_attrs)``  → float penalty 0-1
  * ``boost_item(garment_attrs)``     → float boost 0-1
  * ``get_advice()``                  → structured ``WeatherAdvice`` dict

Advice rules are driven by ``config/data/weather_api_config.json``
(``outfit_impact`` section).  All matching conditions are applied
**additively** (capped at 1.0).

Example
-------
>>> from src.layer3_context.weather import WeatherAPIClient, WeatherOutfitAdvisor
>>> import asyncio
>>> client = WeatherAPIClient()
>>> weather = asyncio.run(client.get_by_city("Paris"))
>>> advisor = WeatherOutfitAdvisor(weather)
>>> advice = advisor.get_advice()
>>> print(advice["summary"])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .weather_models import WeatherData, WeatherCondition
from src.core import get_logger

logger = get_logger(__name__)

_CFG_PATH = Path(__file__).parents[3] / "config" / "data" / "weather_api_config.json"


def _load_impact_cfg() -> dict:
    try:
        raw = json.loads(_CFG_PATH.read_text(encoding="utf-8"))
        return raw.get("outfit_impact", {})
    except Exception:
        return {}


class WeatherOutfitAdvisor:
    """
    Produces outfit recommendations from a ``WeatherData`` snapshot.

    Parameters
    ----------
    weather : WeatherData
        The current weather to advise on.
    """

    def __init__(self, weather: WeatherData) -> None:
        self._w = weather
        self._impact = _load_impact_cfg()
        self._active_rules: list[str] = self._resolve_active_rules()

    # ------------------------------------------------------------------
    # Rule resolution
    # ------------------------------------------------------------------

    def _resolve_active_rules(self) -> list[str]:
        """Determine which impact rules fire for the current weather."""
        rules: list[str] = []
        w = self._w

        # Condition-based rules
        cond_map: dict[WeatherCondition, str] = {
            WeatherCondition.RAINY: "rain",
            WeatherCondition.DRIZZLE: "drizzle",
            WeatherCondition.SNOWY: "snow",
            WeatherCondition.STORMY: "stormy",
        }
        rule = cond_map.get(w.condition)
        if rule and rule in self._impact:
            rules.append(rule)

        # Wind rule
        if w.wind_speed_kmh > 30 and "wind_above_30kmh" in self._impact:
            rules.append("wind_above_30kmh")

        # UV rule
        if (w.uv_index or 0) >= 6 and "uv_above_6" in self._impact:
            rules.append("uv_above_6")

        # Temperature rules
        if w.feels_like_celsius > 27 and "hot_above_27" in self._impact:
            rules.append("hot_above_27")
        elif w.feels_like_celsius < 10 and "cold_below_10" in self._impact:
            rules.append("cold_below_10")

        return rules

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def penalise_item(self, garment: dict[str, Any]) -> float:
        """
        Return a penalty score [0.0–1.0] for a garment given current weather.

        Parameters
        ----------
        garment : dict
            Must contain at least some of:
            ``material``, ``category``, ``silhouette``, ``footwear_style``,
            ``color``.  Missing keys are silently ignored.
        """
        penalty = 0.0
        mat = str(garment.get("material", "")).lower()
        cat = str(garment.get("category", "")).lower()
        sil = str(garment.get("silhouette", "")).lower()
        fwt = str(garment.get("footwear_style", "")).lower()

        for rule_name in self._active_rules:
            rule = self._impact[rule_name]
            base_penalty = float(rule.get("score_penalty", 0.0))

            hits = 0
            total_checks = 0

            # material check
            pen_mats = [m.lower() for m in rule.get("penalise_materials", [])]
            if pen_mats:
                total_checks += 1
                if any(pm in mat for pm in pen_mats):
                    hits += 1

            # footwear check
            pen_fwt = [f.lower() for f in rule.get("penalise_footwear", [])]
            if pen_fwt:
                total_checks += 1
                if any(pf in fwt or pf in cat for pf in pen_fwt):
                    hits += 1

            # silhouette check
            pen_sil = [s.lower() for s in rule.get("penalise_silhouettes", [])]
            if pen_sil:
                total_checks += 1
                if any(ps in sil for ps in pen_sil):
                    hits += 1

            # category check
            avoid_cats = [c.lower() for c in rule.get("avoid_categories", [])]
            if avoid_cats:
                total_checks += 1
                if any(ac in cat for ac in avoid_cats):
                    hits += 1

            if total_checks > 0 and hits > 0:
                penalty += base_penalty * (hits / total_checks)

        return min(penalty, 1.0)

    def boost_item(self, garment: dict[str, Any]) -> float:
        """
        Return a boost score [0.0–1.0] for a garment given current weather.
        """
        boost = 0.0
        mat = str(garment.get("material", "")).lower()
        sil = str(garment.get("silhouette", "")).lower()
        acc = str(garment.get("accessory_type", garment.get("category", ""))).lower()

        for rule_name in self._active_rules:
            rule = self._impact[rule_name]

            boost_mats = [m.lower() for m in rule.get("boost_materials", [])]
            boost_sils = [s.lower() for s in rule.get("boost_silhouettes", [])]
            boost_accs = [a.lower() for a in rule.get("boost_accessories", [])]

            hits = 0
            total_checks = sum([
                1 if boost_mats else 0,
                1 if boost_sils else 0,
                1 if boost_accs else 0,
            ])
            if total_checks == 0:
                continue

            if boost_mats and any(bm in mat for bm in boost_mats):
                hits += 1
            if boost_sils and any(bs in sil for bs in boost_sils):
                hits += 1
            if boost_accs and any(ba in acc for ba in boost_accs):
                hits += 1

            boost += 0.15 * (hits / total_checks)

        return min(boost, 1.0)

    # ------------------------------------------------------------------
    # Full advice object
    # ------------------------------------------------------------------

    def get_advice(self) -> dict[str, Any]:
        """
        Return a structured dict with all weather-based outfit advice.

        Keys
        ----
        ``summary``            Human-readable one-liner.
        ``condition``          Canonical condition string.
        ``temp_band``          "freezing" | "cold" | "mild" | "warm" | "hot"
        ``active_rules``       List of rule names that fired.
        ``penalise_materials`` All materials to avoid.
        ``penalise_footwear``  All footwear styles to avoid.
        ``penalise_silhouettes`` All silhouette styles to avoid.
        ``avoid_categories``   All garment categories to avoid.
        ``boost_materials``    Recommended materials.
        ``boost_accessories``  Recommended accessories.
        ``boost_silhouettes``  Recommended silhouettes.
        ``weather_summary``    ``WeatherData.summary`` string.
        ``feels_like``         Apparent temperature (°C).
        ``is_wet``             bool.
        ``is_windy``           bool.
        ``is_high_uv``         bool.
        """
        # Aggregate all penalise / boost lists from active rules
        pen_mats: set[str] = set()
        pen_fwt: set[str] = set()
        pen_sil: set[str] = set()
        avoid_cats: set[str] = set()
        boost_mats: set[str] = set()
        boost_accs: set[str] = set()
        boost_sils: set[str] = set()

        for rule_name in self._active_rules:
            rule = self._impact[rule_name]
            pen_mats.update(rule.get("penalise_materials", []))
            pen_fwt.update(rule.get("penalise_footwear", []))
            pen_sil.update(rule.get("penalise_silhouettes", []))
            avoid_cats.update(rule.get("avoid_categories", []))
            boost_mats.update(rule.get("boost_materials", []))
            boost_accs.update(rule.get("boost_accessories", []))
            boost_sils.update(rule.get("boost_silhouettes", []))

        # Build human summary
        if not self._active_rules:
            summary = "No special weather precautions needed today."
        else:
            parts: list[str] = []
            if pen_mats:
                parts.append(f"Avoid {', '.join(sorted(pen_mats))}")
            if pen_fwt:
                parts.append(f"skip {', '.join(sorted(pen_fwt))} footwear")
            if boost_accs:
                parts.append(f"bring {', '.join(sorted(boost_accs))}")
            summary = "; ".join(parts) if parts else "Dress for the weather."

        return {
            "summary": summary,
            "condition": self._w.condition.value,
            "temp_band": self._w.temp_band,
            "active_rules": self._active_rules,
            "penalise_materials": sorted(pen_mats),
            "penalise_footwear": sorted(pen_fwt),
            "penalise_silhouettes": sorted(pen_sil),
            "avoid_categories": sorted(avoid_cats),
            "boost_materials": sorted(boost_mats),
            "boost_accessories": sorted(boost_accs),
            "boost_silhouettes": sorted(boost_sils),
            "weather_summary": self._w.summary,
            "feels_like": self._w.feels_like_celsius,
            "temperature_celsius": self._w.temperature_celsius,
            "is_wet": self._w.is_wet,
            "is_windy": self._w.is_windy,
            "is_high_uv": self._w.is_high_uv,
        }
