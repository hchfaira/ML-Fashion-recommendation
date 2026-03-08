"""
Weather Service
Fetches and processes weather data for recommendations.
"""
from typing import Optional
import httpx

from config import get_settings
from src.core.models import WeatherContext
from src.core.exceptions import ContextEngineError
from src.core import get_logger

logger = get_logger(__name__)


class WeatherService:
    """
    Service for fetching weather information.
    
    Uses weather API to get current conditions
    for recommendation adjustments.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.weather_api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    async def get_weather(self, location: str) -> Optional[WeatherContext]:
        """
        Get current weather for a location.
        
        Args:
            location: City name or coordinates
            
        Returns:
            WeatherContext with current conditions
        """
        if not self.api_key:
            logger.warning("Weather API key not configured")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.base_url,
                    params={
                        "q": location,
                        "appid": self.api_key,
                        "units": "metric"
                    },
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    logger.warning(f"Weather API error: {response.status_code}")
                    return None
                
                data = response.json()
                return self._parse_weather_response(data)
                
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
            return None
    
    async def get_weather_by_coords(
        self,
        lat: float,
        lon: float
    ) -> Optional[WeatherContext]:
        """
        Get weather by coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            WeatherContext
        """
        if not self.api_key:
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.base_url,
                    params={
                        "lat": lat,
                        "lon": lon,
                        "appid": self.api_key,
                        "units": "metric"
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return self._parse_weather_response(response.json())
                    
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
        
        return None
    
    def _parse_weather_response(self, data: dict) -> WeatherContext:
        """Parse OpenWeatherMap response."""
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        
        # Map weather condition
        condition_map = {
            "clear": "sunny",
            "clouds": "cloudy",
            "rain": "rainy",
            "drizzle": "rainy",
            "thunderstorm": "rainy",
            "snow": "snowy",
            "mist": "cloudy",
            "fog": "cloudy"
        }
        
        raw_condition = weather.get("main", "clear").lower()
        condition = condition_map.get(raw_condition, "cloudy")
        
        return WeatherContext(
            temperature_celsius=main.get("temp", 20),
            condition=condition,
            humidity=main.get("humidity"),
            wind_speed_kmh=wind.get("speed", 0) * 3.6  # m/s to km/h
        )
    
    def get_clothing_recommendations(
        self,
        weather: WeatherContext
    ) -> dict:
        """
        Get clothing recommendations based on weather.
        
        Args:
            weather: Current weather conditions
            
        Returns:
            Dictionary with clothing suggestions
        """
        temp = weather.temperature_celsius
        condition = weather.condition
        
        recommendations = {
            "layers": [],
            "avoid": [],
            "suggested_materials": [],
            "accessories": []
        }
        
        # Temperature-based recommendations
        if temp < 5:
            recommendations["layers"] = ["heavy_coat", "sweater", "thermal"]
            recommendations["suggested_materials"] = ["wool", "down", "fleece"]
            recommendations["accessories"] = ["scarf", "gloves", "beanie"]
        elif temp < 12:
            recommendations["layers"] = ["jacket", "light_sweater"]
            recommendations["suggested_materials"] = ["wool", "cotton", "denim"]
            recommendations["accessories"] = ["light_scarf"]
        elif temp < 18:
            recommendations["layers"] = ["light_jacket", "cardigan"]
            recommendations["suggested_materials"] = ["cotton", "linen blend"]
        elif temp < 25:
            recommendations["layers"] = ["light_layers"]
            recommendations["suggested_materials"] = ["cotton", "linen"]
            recommendations["avoid"] = ["heavy_fabrics", "dark_colors"]
        else:
            recommendations["layers"] = ["minimal"]
            recommendations["suggested_materials"] = ["linen", "light_cotton", "breathable"]
            recommendations["avoid"] = ["wool", "heavy_fabrics", "layers"]
        
        # Condition-based adjustments
        if condition == "rainy":
            recommendations["accessories"].append("umbrella")
            recommendations["suggested_materials"].append("water_resistant")
            recommendations["avoid"].extend(["suede", "silk"])
        
        if condition == "sunny" and temp > 20:
            recommendations["accessories"].extend(["sunglasses", "hat"])
        
        return recommendations
