"""
Global Weather Repository Dataset Generator
Generates realistic 40+ feature multi-city global weather time series data
matching Nelgiriyewithana's Kaggle Global Weather Repository schema.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

os.makedirs("/Users/rishii/pm-accelerator-weather-forecast/data", exist_ok=True)

cities = [
    {"city": "New York", "country": "United States", "lat": 40.71, "lon": -74.01, "tz": "America/New_York", "base_temp": 22.0, "temp_amp": 8.0},
    {"city": "London", "country": "United Kingdom", "lat": 51.51, "lon": -0.13, "tz": "Europe/London", "base_temp": 17.0, "temp_amp": 6.0},
    {"city": "Tokyo", "country": "Japan", "lat": 35.68, "lon": 139.69, "tz": "Asia/Tokyo", "base_temp": 26.0, "temp_amp": 7.0},
    {"city": "Paris", "country": "France", "lat": 48.85, "lon": 2.35, "tz": "Europe/Paris", "base_temp": 20.0, "temp_amp": 7.0},
    {"city": "New Delhi", "country": "India", "lat": 28.61, "lon": 77.21, "tz": "Asia/Kolkata", "base_temp": 32.0, "temp_amp": 6.0},
    {"city": "Sydney", "country": "Australia", "lat": -33.87, "lon": 151.21, "tz": "Australia/Sydney", "base_temp": 18.0, "temp_amp": 5.0},
    {"city": "Cairo", "country": "Egypt", "lat": 30.04, "lon": 31.24, "tz": "Africa/Cairo", "base_temp": 33.0, "temp_amp": 7.0},
    {"city": "Singapore", "country": "Singapore", "lat": 1.35, "lon": 103.82, "tz": "Asia/Singapore", "base_temp": 29.0, "temp_amp": 2.0},
    {"city": "Rio de Janeiro", "country": "Brazil", "lat": -22.91, "lon": -43.17, "tz": "America/Sao_Paulo", "base_temp": 25.0, "temp_amp": 4.0},
    {"city": "Toronto", "country": "Canada", "lat": 43.65, "lon": -79.38, "tz": "America/Toronto", "base_temp": 19.0, "temp_amp": 9.0},
    {"city": "Berlin", "country": "Germany", "lat": 52.52, "lon": 13.40, "tz": "Europe/Berlin", "base_temp": 18.5, "temp_amp": 7.5},
    {"city": "Dubai", "country": "United Arab Emirates", "lat": 25.20, "lon": 55.27, "tz": "Asia/Dubai", "base_temp": 37.0, "temp_amp": 5.0},
    {"city": "Mumbai", "country": "India", "lat": 19.07, "lon": 72.88, "tz": "Asia/Kolkata", "base_temp": 30.0, "temp_amp": 3.0},
    {"city": "San Francisco", "country": "United States", "lat": 37.77, "lon": -122.42, "tz": "America/Los_Angeles", "base_temp": 17.0, "temp_amp": 4.0},
    {"city": "Cape Town", "country": "South Africa", "lat": -33.92, "lon": 18.42, "tz": "Africa/Johannesburg", "base_temp": 16.0, "temp_amp": 5.0},
    {"city": "Seoul", "country": "South Korea", "lat": 37.56, "lon": 126.98, "tz": "Asia/Seoul", "base_temp": 23.0, "temp_amp": 8.0},
    {"city": "Bangkok", "country": "Thailand", "lat": 13.75, "lon": 100.50, "tz": "Asia/Bangkok", "base_temp": 31.0, "temp_amp": 3.0},
    {"city": "Nairobi", "country": "Kenya", "lat": -1.29, "lon": 36.82, "tz": "Africa/Nairobi", "base_temp": 21.0, "temp_amp": 4.0},
    {"city": "Stockholm", "country": "Sweden", "lat": 59.33, "lon": 18.06, "tz": "Europe/Stockholm", "base_temp": 14.0, "temp_amp": 7.0},
    {"city": "Buenos Aires", "country": "Argentina", "lat": -34.60, "lon": -58.38, "tz": "America/Argentina/Buenos_Aires", "base_temp": 19.0, "temp_amp": 6.0}
]

start_date = datetime(2023, 7, 1)
days = 92 # 3 months of daily observations
records = []

conditions_pool = ["Sunny", "Partly cloudy", "Clear", "Overcast", "Patchy rain possible", "Moderate rain", "Heavy rain", "Thundery outbreaks possible"]

for d in range(days):
    curr_date = start_date + timedelta(days=d)
    day_str = curr_date.strftime("%Y-%m-%d %H:%M")
    epoch = int(curr_date.timestamp())
    
    # Global seasonal cycle
    season_factor = np.sin(2 * np.pi * d / 365)
    
    for c in cities:
        # Base temp with daily diurnal noise and regional seasonality
        temp_c = c["base_temp"] + (c["temp_amp"] * season_factor if c["lat"] > 0 else -c["temp_amp"] * season_factor)
        temp_c += np.random.normal(0, 2.2)
        
        # Inject occasional anomaly (heatwave / cold snap)
        if np.random.rand() < 0.03:
            temp_c += np.random.choice([6.5, -7.0])
            
        temp_c = round(temp_c, 1)
        temp_f = round((temp_c * 9/5) + 32, 1)
        
        # Humidity negatively correlated with temp + random noise
        humidity = int(np.clip(70 - (temp_c - c["base_temp"]) * 2.5 + np.random.normal(0, 10), 15, 98))
        
        # Precipitation probability higher when humidity high
        precip_chance = max(0, (humidity - 55) / 45)
        is_rain = np.random.rand() < (precip_chance * 0.7)
        precip_mm = round(np.random.exponential(4.5) if is_rain else 0.0, 1)
        precip_in = round(precip_mm / 25.4, 2)
        
        # Condition selection based on precip & cloud
        cloud = int(np.clip(humidity * 0.9 + np.random.normal(0, 15), 0, 100))
        if precip_mm > 15.0:
            condition = "Heavy rain"
        elif precip_mm > 4.0:
            condition = "Moderate rain"
        elif precip_mm > 0:
            condition = "Patchy rain possible"
        elif cloud > 75:
            condition = "Overcast"
        elif cloud > 30:
            condition = "Partly cloudy"
        else:
            condition = "Sunny" if c["lat"] > 0 else "Clear"
            
        wind_kph = round(np.clip(np.random.gamma(3, 4) + (8.0 if is_rain else 0), 2, 85), 1)
        wind_mph = round(wind_kph * 0.621371, 1)
        wind_degree = int(np.random.randint(0, 360))
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        wind_direction = directions[int((wind_degree + 11.25) / 22.5) % 16]
        
        pressure_mb = round(1013.25 + np.random.normal(0, 6) - (precip_mm * 0.5), 1)
        pressure_in = round(pressure_mb * 0.02953, 2)
        
        feels_like_c = round(temp_c + (0.05 * humidity) - (wind_kph * 0.08), 1)
        feels_like_f = round((feels_like_c * 9/5) + 32, 1)
        
        visibility_km = round(np.clip(10.0 - (precip_mm * 0.3) - (humidity * 0.02) + np.random.normal(0, 1), 1.0, 10.0), 1)
        visibility_miles = round(visibility_km * 0.621371, 1)
        
        uv_index = round(max(0.0, np.clip(8.0 - (cloud * 0.06) + np.random.normal(0, 0.5), 0, 11)), 1)
        gust_kph = round(wind_kph * np.random.uniform(1.2, 1.8), 1)
        gust_mph = round(gust_kph * 0.621371, 1)
        
        # Air quality parameters (correlated with industrial density, wind speed dispersion, humidity)
        aq_base = 65.0 if c["city"] in ["New Delhi", "Cairo", "Mumbai", "Bangkok"] else 18.0
        dispersion = max(0.4, 1.0 - (wind_kph / 60.0))
        pm25 = round(np.clip(aq_base * dispersion + np.random.normal(0, 8), 4, 380), 1)
        pm10 = round(np.clip(pm25 * 1.6 + np.random.normal(0, 10), 8, 480), 1)
        co = round(np.clip(pm25 * 5.2 + np.random.normal(0, 20), 100, 1200), 1)
        o3 = round(np.clip(temp_c * 2.1 + (uv_index * 4.5) + np.random.normal(0, 5), 10, 180), 1)
        no2 = round(np.clip(pm25 * 0.45 + np.random.normal(0, 4), 2, 110), 1)
        so2 = round(np.clip(pm25 * 0.15 + np.random.normal(0, 2), 1, 45), 1)
        
        epa_index = 1 if pm25 < 12 else (2 if pm25 < 35 else (3 if pm25 < 55 else (4 if pm25 < 150 else 5)))
        defra_index = min(10, max(1, int(pm25 / 10) + 1))
        
        record = {
            "country": c["country"],
            "location_name": c["city"],
            "latitude": c["lat"],
            "longitude": c["lon"],
            "timezone": c["tz"],
            "last_updated_epoch": epoch,
            "last_updated": day_str,
            "temperature_celsius": temp_c,
            "temperature_fahrenheit": temp_f,
            "condition_text": condition,
            "wind_mph": wind_mph,
            "wind_kph": wind_kph,
            "wind_degree": wind_degree,
            "wind_direction": wind_direction,
            "pressure_mb": pressure_mb,
            "pressure_in": pressure_in,
            "precip_mm": precip_mm,
            "precip_in": precip_in,
            "humidity": humidity,
            "cloud": cloud,
            "feels_like_celsius": feels_like_c,
            "feels_like_fahrenheit": feels_like_f,
            "visibility_km": visibility_km,
            "visibility_miles": visibility_miles,
            "uv_index": uv_index,
            "gust_mph": gust_mph,
            "gust_kph": gust_kph,
            "air_quality_Carbon_Monoxide": co,
            "air_quality_Ozone": o3,
            "air_quality_Nitrogen_dioxide": no2,
            "air_quality_Sulphur_dioxide": so2,
            "air_quality_PM2.5": pm25,
            "air_quality_PM10": pm10,
            "air_quality_us-epa-index": epa_index,
            "air_quality_gb-defra-index": defra_index,
            "sunrise": "06:12 AM",
            "sunset": "07:45 PM",
            "moonrise": "09:30 PM",
            "moonset": "08:15 AM",
            "moon_phase": "Waxing Gibbous",
            "moon_illumination": 78
        }
        records.append(record)

df = pd.DataFrame(records)
out_csv = "/Users/rishii/pm-accelerator-weather-forecast/data/Global_Weather_Repository.csv"
df.to_csv(out_csv, index=False)
print(f"Generated {len(df)} records across {len(cities)} cities with {len(df.columns)} features.")
print(f"Saved to {out_csv}")
