from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


DB_NAME = "aqi_feature_store"
COLLECTION = "aqi_features_hourly"
LOCATION_ID = "karachi"


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


def _mongo_collection():
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI missing.")

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    col = client[DB_NAME][COLLECTION]

    # ensure indexes
    col.create_index([("location_id", 1), ("event_timestamp", 1)], unique=True)
    col.create_index([("event_timestamp", 1)])
    return col


def _fetch_one_hour_openmeteo(ts: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch exactly one hour from Open-Meteo by requesting the full day and selecting the hour.
    (Open-Meteo APIs usually accept day granularity; this is the most reliable way.)
    """
    from src.ingest_openmeteo import fetch_karachi_aqi_weather

    day = ts.date().isoformat()
    df = fetch_karachi_aqi_weather(start_date=day, end_date=day)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp")

    # align to the hour
    ts_hour = ts.floor("h").tz_localize(None)
    row = df[df["timestamp"] == ts_hour].copy()

    if row.empty:
        raise RuntimeError(f"No Open-Meteo row found for hour {ts_hour}")

    return row.reset_index(drop=True)


def _get_aqi_history(col, end_ts: pd.Timestamp, hours: int) -> List[float]:
    """
    Get last `hours` AQI values strictly before end_ts: (end_ts - hours ... end_ts-1h)
    """
    end_ts = pd.to_datetime(end_ts, utc=True)
    start_ts = end_ts - pd.Timedelta(hours=hours)

    cursor = col.find(
        {
            "location_id": LOCATION_ID,
            "event_timestamp": {"$gte": start_ts, "$lt": end_ts},
        },
        {"_id": 0, "event_timestamp": 1, "us_aqi": 1},
    ).sort("event_timestamp", 1)

    docs = list(cursor)
    return [float(d["us_aqi"]) for d in docs]


def _build_feature_row(raw_row: pd.DataFrame, col) -> Dict[str, Any]:
    """
    raw_row: single-row dataframe with raw Open-Meteo fields + timestamp
    """
    r = raw_row.iloc[0].to_dict()
    ts = pd.to_datetime(r["timestamp"]).tz_localize(timezone.utc).floor("h")
    r["location_id"] = LOCATION_ID
    r["event_timestamp"] = ts

    # Time features
    hour = ts.hour
    r["day_of_week"] = int(ts.dayofweek)
    r["month"] = int(ts.month)
    r["hour_sin"] = float(np.sin(2.0 * np.pi * hour / 24.0))
    r["hour_cos"] = float(np.cos(2.0 * np.pi * hour / 24.0))

    # Missing flags (raw)
    for c in RAW_AIR_COLS + RAW_WEATHER_COLS:
        r[f"{c}_was_missing"] = int(pd.isna(r.get(c)))

    # Imputation (minimal): weather interpolate not possible from 1 row; use ffill-like fallback
    # For MVP: drop if any critical field is missing
    critical = ["us_aqi", "pm2_5", "wind_speed_10m"]
    for c in critical:
        if pd.isna(r.get(c)):
            raise RuntimeError(f"Critical field {c} missing at {ts}")

    # Interaction
    r["pm25_wind_interaction"] = float(r["pm2_5"]) / (float(r["wind_speed_10m"]) + 1.0)

    # Lag/Rolling from Mongo history
    hist_24 = _get_aqi_history(col, ts, hours=24)
    if len(hist_24) < 24:
        raise RuntimeError(f"Not enough history in Mongo to compute lag/roll at {ts}. Have {len(hist_24)} hours.")

    # lags
    r["aqi_lag_1"] = float(hist_24[-1])
    r["aqi_lag_3"] = float(hist_24[-3])
    r["aqi_lag_24"] = float(hist_24[0])

    # rollings (strictly past)
    r["aqi_roll_6"] = float(np.mean(hist_24[-6:]))
    r["aqi_roll_24"] = float(np.mean(hist_24))

    # remove old key
    r.pop("timestamp", None)
    return r


def run() -> None:
    col = _mongo_collection()

    # choose "current hour" (UTC). You can also use Asia/Karachi time and convert to UTC.
    now = pd.Timestamp.now(tz="UTC").floor("h")

    raw_row = _fetch_one_hour_openmeteo(now)
    feature_doc = _build_feature_row(raw_row, col)

    filt = {"location_id": feature_doc["location_id"], "event_timestamp": feature_doc["event_timestamp"]}
    col.update_one(filt, {"$set": feature_doc}, upsert=True)

    print(f"✅ Upserted 1 feature row for {feature_doc['event_timestamp']}.")


if __name__ == "__main__":
    run()