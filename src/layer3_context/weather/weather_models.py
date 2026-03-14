"""
WeatherModels
=============
Rich data models for the real-time weather subsystem.

``WeatherData`` extends the minimal ``WeatherContext`` that already exists
in ``src.core.models`` with the extra fields returned by modern weather APIs.
It is the internal DTO passed between WeatherAPIClient → WeatherNormalizer →
WeatherCache → WeatherOutfitAdvisor.  It can be converted to the lightweight
``WeatherContext`` for downstream layers that don't need the full payload.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Condition enum  (richer than the plain string in WeatherContext)
# ---------------------------------------------------------------------------

class WeatherCondition(str, Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    DRIZZLE = "drizzle"
    SNOWY = "snowy"
    STORMY = "stormy"
    FOGGY = "foggy"
    UNKNOWN = "unknown"

    # Backward-compat: legacy code sometimes checks == "sunny" etc.
    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, str):
            return self.value == other.lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value)

    @property
    def is_wet(self) -> bool:
        return self in (
            WeatherCondition.RAINY,
            WeatherCondition.DRIZZLE,
            WeatherCondition.STORMY,
            WeatherCondition.SNOWY,
        )

    @property
    def is_cold_condition(self) -> bool:
        return self in (WeatherCondition.SNOWY, WeatherCondition.STORMY)


# ---------------------------------------------------------------------------
# Rich WeatherData
# ---------------------------------------------------------------------------

class WeatherData(BaseModel):
    """
    Canonical weather DTO used internally by the weather subsystem.

    Fields
    ------
    temperature_celsius : float
        2-metre air temperature (°C).
    feels_like_celsius : float
        Apparent temperature after wind-chill / humidity correction (°C).
    condition : WeatherCondition
        Normalised sky condition.
    humidity : float | None
        Relative humidity 0–100 %.
    wind_speed_kmh : float
        10-metre wind speed in km/h.
    uv_index : int | None
        UV index 0–11 (may be absent for some providers / night-time).
    precipitation_mm : float
        Precipitation in the last hour (mm).
    wmo_code : int | None
        Raw WMO weather interpretation code (Open-Meteo specific).
    lat / lon : float
        Coordinates of the queried location.
    city : str | None
        Human-readable location name (resolved by provider or passed in).
    provider : str
        Which API produced this data (``"open-meteo"``, ``"openweathermap"``,
        ``"cache"``, ``"fallback"``).
    fetched_at : datetime
        UTC timestamp when the API was called.
    sunrise / sunset : Any | None
        Sunrise / sunset time (ISO-8601 string from Open-Meteo or Unix
        timestamp int from OWM).  Use ``sunrise_dt`` / ``sunset_dt``
        properties for a normalised ``datetime``.
    """

    temperature_celsius: float = 15.0
    feels_like_celsius: float = 15.0
    condition: WeatherCondition = WeatherCondition.UNKNOWN
    humidity: float | None = Field(None, ge=0, le=100)
    wind_speed_kmh: float = Field(0.0, ge=0)
    uv_index: int | None = Field(None, ge=0, le=11)
    precipitation_mm: float = Field(0.0, ge=0)
    wmo_code: int | None = None
    lat: float = 0.0
    lon: float = 0.0
    city: str | None = None
    provider: str = "unknown"
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    sunrise: Any = None   # ISO string (Open-Meteo) or Unix int (OWM)
    sunset: Any = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_cold(self) -> bool:
        return self.feels_like_celsius < 10

    @property
    def is_hot(self) -> bool:
        return self.feels_like_celsius > 27

    @property
    def is_wet(self) -> bool:
        return self.condition.is_wet or self.precipitation_mm > 0.5

    @property
    def is_windy(self) -> bool:
        return self.wind_speed_kmh > 30

    @property
    def is_high_uv(self) -> bool:
        return (self.uv_index or 0) >= 6

    @property
    def temp_band(self) -> str:
        """Return a human-readable temperature band."""
        t = self.feels_like_celsius
        if t < 5:
            return "freezing"
        if t < 12:
            return "cold"
        if t < 18:
            return "mild"
        if t < 25:
            return "warm"
        return "hot"

    @property
    def summary(self) -> str:
        """One-line human-readable summary."""
        parts = [
            f"{self.temperature_celsius:.0f}°C",
            f"(feels {self.feels_like_celsius:.0f}°C)",
            self.condition.value,
        ]
        if self.is_wet:
            parts.append(f"{self.precipitation_mm:.1f}mm rain")
        if self.is_windy:
            parts.append(f"wind {self.wind_speed_kmh:.0f} km/h")
        if self.city:
            parts.insert(0, f"[{self.city}]")
        return " • ".join(parts)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_weather_context(self) -> "WeatherContext":  # type: ignore[name-defined]  # noqa: F821
        """Convert to the lightweight ``WeatherContext`` used by downstream layers."""
        from src.core.models import WeatherContext  # local import avoids circular dep
        return WeatherContext(
            temperature_celsius=self.temperature_celsius,
            condition=self.condition.value,
            humidity=self.humidity,
            wind_speed_kmh=self.wind_speed_kmh,
            uv_index=self.uv_index,
            feels_like_celsius=self.feels_like_celsius,
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        d = self.model_dump()
        d["condition"] = self.condition.value
        d["fetched_at"] = self.fetched_at.isoformat()
        return d

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def unknown(
        cls,
        lat: float = 0.0,
        lon: float = 0.0,
        city: str | None = None,
    ) -> "WeatherData":
        """Return a sentinel instance when the API fails."""
        return cls(
            temperature_celsius=15.0,
            feels_like_celsius=15.0,
            condition=WeatherCondition.UNKNOWN,
            provider="fallback",
            lat=lat,
            lon=lon,
            city=city,
        )
