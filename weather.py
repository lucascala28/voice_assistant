import os
import requests
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(Path(__file__).parent / ".env")

API_KEY=os.getenv("WEATHER_KEY")
URL=os.getenv("WEATHER_URL")
GEO_URL=os.getenv("GEO_URL")

# Look up (lat, lon, display name) for a place name; raises ValueError if it is not found.
def get_geo(location):
    
    response = requests.get(GEO_URL, params={
        "q": location,
        "appid": API_KEY,
        "limit": 1
    }, timeout=10)

    response.raise_for_status()
    results = response.json()

    if not results:
        raise ValueError(f"Location not found: {location}")
    
    r = results[0]
    print(f"Geo information for {location} found with lat={r['lat']} and lon={r['lon']}")
    return r["lat"], r["lon"], f"{r['name']}, {r.get('country','')}"

# Fetch current weather for a location; returns (description, temp, feels_like, max, min, wind_speed, precip).
# `detailed` is accepted for the tool schema, but both modes return the same tuple.
def get_weather(location, detailed=False):
    print(f"Weather tool invoked with parameters: location={location}, detailed={detailed}")
    lat, lon, name = get_geo(location)

    response = requests.get(URL, params={
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "imperial"
    }, timeout=10)

    response.raise_for_status()
    results = response.json()

    precip = results.get('rain', {}).get('1h', 0)

    return results["weather"][0]["description"], results["main"]["temp"], results["main"]["feels_like"], results["main"]["temp_max"], results["main"]["temp_min"], results["wind"]["speed"], precip
