# import pandas as pd


# def clean_aqi_weather_data(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Cleans raw AQI + weather data for downstream EDA and feature engineering.

#     Steps:
#     1. Ensure timestamp is datetime
#     2. Sort by time
#     3. Enforce hourly frequency
#     4. Handle missing values
#     """

#     # ---------------------------
#     # 1. Timestamp sanity check
#     # ---------------------------
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df = df.sort_values("timestamp")

#     # ---------------------------
#     # 2. Set timestamp as index
#     # ---------------------------
#     df = df.set_index("timestamp")

#     # ---------------------------
#     # 3. Enforce hourly frequency
#     # ---------------------------
#     df = df.asfreq("h")

#     # ---------------------------
#     # 4. Handle missing values
#     # ---------------------------
#     # Environmental data changes smoothly → time-based interpolation
#     df = df.interpolate(method="time")

#     # ---------------------------
#     # 5. Drop rows that still have NaNs (edge cases)
#     # ---------------------------
#     df = df.dropna()

#     # Reset index for consistency
#     df = df.reset_index()

#     return df

from __future__ import annotations

import pandas as pd
from typing import List


WEATHER_COLS: List[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
]

POLLUTANT_COLS: List[str] = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]


def clean_aqi_weather_data(
    df: pd.DataFrame,
    freq: str = "h",
    max_weather_gap_hours: int = 3,
    pollutant_ffill_limit: int = 1,
) -> pd.DataFrame:
    """
    Cleans raw AQI + weather data for downstream feature engineering.

    - Enforces hourly frequency (may introduce NaNs for missing hours).
    - Weather: time interpolation with a max gap limit.
    - Pollutants/AQI: short forward fill only (spikes should not be interpolated).
    - Adds missingness flags for each raw column.
    - Drops rows still containing NaNs after safe filling.
    """
    if "timestamp" not in df.columns:
        raise ValueError("Input dataframe must contain a 'timestamp' column.")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").set_index("timestamp")

    # Enforce hourly frequency
    out = out.asfreq(freq)

    # Missingness flags BEFORE filling
    for col in WEATHER_COLS + POLLUTANT_COLS:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype("int8")

    # Weather: interpolate small gaps only
    weather_existing = [c for c in WEATHER_COLS if c in out.columns]
    if weather_existing:
        out[weather_existing] = out[weather_existing].interpolate(
            method="time",
            limit=max_weather_gap_hours,
            limit_direction="both",
        )

    # Pollutants & AQI: short forward fill only
    pollutant_existing = [c for c in POLLUTANT_COLS if c in out.columns]
    if pollutant_existing:
        out[pollutant_existing] = out[pollutant_existing].ffill(limit=pollutant_ffill_limit)

    # Keep only clean rows
    out = out.dropna().reset_index()

    return out