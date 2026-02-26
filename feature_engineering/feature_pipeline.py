# import numpy as np
# import pandas as pd

# import numpy as np
# import pandas as pd

# def build_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Features at time t only (safe for Feature Store).
#     No shift(-k) targets here.
#     """
#     df = df.sort_values("timestamp").copy()

#     # Time features
#     df["hour"] = df["timestamp"].dt.hour
#     df["day_of_week"] = df["timestamp"].dt.dayofweek
#     df["month"] = df["timestamp"].dt.month
#     df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
#     df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

#     # Lag features (AQI)
#     df["aqi_lag_1"] = df["us_aqi"].shift(1)
#     df["aqi_lag_3"] = df["us_aqi"].shift(3)
#     df["aqi_lag_24"] = df["us_aqi"].shift(24)

#     # Rolling features (AQI)
#     df["aqi_roll_6"] = df["us_aqi"].rolling(6).mean()
#     df["aqi_roll_24"] = df["us_aqi"].rolling(24).mean()

#     # Interaction
#     df["pm25_wind_interaction"] = df["pm2_5"] / (df["wind_speed_10m"] + 1)

#     # Drop rows where lag/rolling isn't available yet
#     df = df.dropna().reset_index(drop=True)

#     return df



# def add_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add supervised learning targets for 24h, 48h, 72h ahead.
#     Must be called only when preparing training data.
#     """
#     df = df.sort_values("timestamp").copy()

#     df["y_24h"] = df["us_aqi"].shift(-24)
#     df["y_48h"] = df["us_aqi"].shift(-48)
#     df["y_72h"] = df["us_aqi"].shift(-72)

#     # Drop rows at the end that don't have future labels
#     df = df.dropna().reset_index(drop=True)

#     return df

from __future__ import annotations

import numpy as np
import pandas as pd


RAW_AIR_COLS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

RAW_WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
]


def build_feature_store_rows(
    df: pd.DataFrame,
    location_id: str = "karachi",
) -> pd.DataFrame:
    """
    Build rows to insert into Hopsworks Feature Store.
    Includes raw + engineered online features (lags/rolling).
    Does NOT create targets/labels.
    """
    out = df.copy()
    out = out.sort_values("timestamp")
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    out["location_id"] = location_id
    out = out.rename(columns={"timestamp": "event_timestamp"})

    # Time features
    hour = out["event_timestamp"].dt.hour.astype("int16")
    out["day_of_week"] = out["event_timestamp"].dt.dayofweek.astype("int16")
    out["month"] = out["event_timestamp"].dt.month.astype("int16")
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    # Lags
    out["aqi_lag_1"] = out["us_aqi"].shift(1)
    out["aqi_lag_3"] = out["us_aqi"].shift(3)
    out["aqi_lag_24"] = out["us_aqi"].shift(24)

    # Rolling (strictly past)
    base = out["us_aqi"].shift(1)
    out["aqi_roll_6"] = base.rolling(window=6, min_periods=6).mean()
    out["aqi_roll_24"] = base.rolling(window=24, min_periods=24).mean()

    # Interaction
    out["pm25_wind_interaction"] = out["pm2_5"] / (out["wind_speed_10m"] + 1.0)

    # Select columns to store
    missing_flag_cols = [c for c in out.columns if c.endswith("_was_missing")]

    cols_to_store = (
        ["location_id", "event_timestamp"]
        + RAW_AIR_COLS
        + RAW_WEATHER_COLS
        + ["hour_sin", "hour_cos", "day_of_week", "month"]
        + ["aqi_lag_1", "aqi_lag_3", "aqi_lag_24", "aqi_roll_6", "aqi_roll_24"]
        + ["pm25_wind_interaction"]
        + missing_flag_cols
    )

    out = out[cols_to_store].dropna().reset_index(drop=True)
    return out