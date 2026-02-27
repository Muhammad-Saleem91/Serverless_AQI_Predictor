import requests
import pandas as pd
from datetime import datetime, timezone


# =========================
# CONFIG (Karachi)
# =========================
LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"


# =========================
# AIR QUALITY FETCH
# =========================
def fetch_air_quality(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch hourly air quality data from Open-Meteo
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi"
        ],
        "timezone": TIMEZONE
    }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)

    return df


# =========================
# WEATHER FETCH
# =========================
def fetch_weather(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo
    """
    # url = "https://archive-api.open-meteo.com/v1/archive"
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "surface_pressure"
        ],
        "timezone": TIMEZONE
    }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)

    return df


# =========================
# MERGED FETCH (MAIN ENTRY)
# =========================
def fetch_karachi_aqi_weather(
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Fetch and merge AQI + weather data (hourly)
    Returns a clean, merged DataFrame
    """

    aq_df = fetch_air_quality(start_date, end_date)
    weather_df = fetch_weather(start_date, end_date)

    print(f"Air Quality Data Shape: {aq_df.shape}")
    print(f"Air Quality Data Frame Nulls:\n{aq_df.isna().sum()}")
    print(f"Weather Data Shape: {weather_df.shape}")
    print(f"Weather Data Frame Nulls:\n{weather_df.isna().sum()}")

    df = pd.merge(
        aq_df,
        weather_df,
        on="timestamp",
        how="inner"
    )

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# =========================
# LOCAL TEST (DEV ONLY)
# =========================
if __name__ == "__main__":
    # Fetch last 30 days for testing
    end = datetime.now(timezone.utc).date()
    start = end - pd.Timedelta(days=30)

    df = fetch_karachi_aqi_weather(
        start_date=str(start),
        end_date=str(end)
    )

    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
