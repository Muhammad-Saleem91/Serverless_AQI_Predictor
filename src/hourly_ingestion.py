from __future__ import annotations

import os
from datetime import timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from ingestion.fetch_data import fetch_karachi_aqi_weather


DB_NAME = "aqi_feature_store"
COLLECTION = "aqi_features_hourly"

LOCATION_ID = "karachi"
TZ = "Asia/Karachi"

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
        raise RuntimeError("MONGODB_URI missing. Put it in .env (local) or GitHub Secrets (CI).")

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    col = client[DB_NAME][COLLECTION]

    # Idempotent: safe to call every run
    col.create_index([("location_id", 1), ("event_timestamp", 1)], unique=True)
    col.create_index([("event_timestamp", 1)])
    return col


def _get_aqi_history(col, end_ts_utc: pd.Timestamp, hours: int = 24) -> List[float]:
    """
    Get last `hours` AQI values strictly before end_ts_utc.
    """
    end_ts_utc = pd.to_datetime(end_ts_utc, utc=True)
    start_ts_utc = end_ts_utc - pd.Timedelta(hours=hours)

    cursor = col.find(
        {"location_id": LOCATION_ID, "event_timestamp": {"$gte": start_ts_utc, "$lt": end_ts_utc}},
        {"_id": 0, "event_timestamp": 1, "us_aqi": 1},
    ).sort("event_timestamp", 1)

    docs = list(cursor)
    return [float(d["us_aqi"]) for d in docs]


def _fetch_exact_hour_row(ts_local_hour: pd.Timestamp) -> pd.Series:
    """
    Fetch day data and select the row matching ts_local_hour (naive local hour).
    """
    day = ts_local_hour.date().isoformat()
    df = fetch_karachi_aqi_weather(start_date=day, end_date=day)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Open-Meteo returns timestamps in local timezone if you requested timezone in API
    # Your ingestion uses TIMEZONE="Asia/Karachi", so timestamps match local time.
    ts_naive = ts_local_hour.tz_localize(None)

    row = df[df["timestamp"] == ts_naive]
    if row.empty:
        raise RuntimeError(f"No Open-Meteo data found for hour={ts_naive}.")
    return row.iloc[0]


def _build_feature_doc(raw_row: pd.Series, col) -> Dict[str, Any]:
    """
    Build a single Mongo document for this hour.
    """
    # local time hour -> convert to UTC for storage consistency
    ts_local = pd.to_datetime(raw_row["timestamp"]).tz_localize(TZ).floor("h")
    ts_utc = ts_local.tz_convert("UTC")

    doc: Dict[str, Any] = {"location_id": LOCATION_ID, "event_timestamp": ts_utc}

    # Copy raw features
    for c in RAW_AIR_COLS + RAW_WEATHER_COLS:
        doc[c] = float(raw_row[c]) if pd.notna(raw_row[c]) else None
        doc[f"{c}_was_missing"] = int(pd.isna(raw_row[c]))

    # Minimal guardrail: if critical values missing, skip writing this hour
    critical = ["us_aqi", "pm2_5", "wind_speed_10m"]
    for c in critical:
        if doc[c] is None:
            raise RuntimeError(f"Critical feature '{c}' missing at {ts_local} (local).")

    # Time features
    hour = int(ts_local.hour)
    doc["day_of_week"] = int(ts_local.dayofweek)
    doc["month"] = int(ts_local.month)
    doc["hour_sin"] = float(np.sin(2.0 * np.pi * hour / 24.0))
    doc["hour_cos"] = float(np.cos(2.0 * np.pi * hour / 24.0))

    # Interaction
    doc["pm25_wind_interaction"] = float(doc["pm2_5"]) / (float(doc["wind_speed_10m"]) + 1.0)

    # Lags/Rollings from Mongo history (strictly past)
    hist_24 = _get_aqi_history(col, end_ts_utc=ts_utc, hours=24)
    if len(hist_24) < 24:
        raise RuntimeError(
            f"Not enough history in Mongo to compute lags/rolling at {ts_local}. "
            f"Need 24, have {len(hist_24)}."
        )

    doc["aqi_lag_1"] = float(hist_24[-1])
    doc["aqi_lag_3"] = float(hist_24[-3])
    doc["aqi_lag_24"] = float(hist_24[0])

    doc["aqi_roll_6"] = float(np.mean(hist_24[-6:]))
    doc["aqi_roll_24"] = float(np.mean(hist_24))

    return doc


def run() -> None:
    col = _mongo_collection()

    # Determine "current hour" in Karachi
    now_local = pd.Timestamp.now(tz=TZ).floor("h")

    raw_row = _fetch_exact_hour_row(now_local)
    doc = _build_feature_doc(raw_row, col)

    col.update_one(
        {"location_id": doc["location_id"], "event_timestamp": doc["event_timestamp"]},
        {"$set": doc},
        upsert=True,
    )

    print(f"✅ Upserted 1 hourly feature row for Karachi hour {now_local} (stored UTC={doc['event_timestamp']}).")


if __name__ == "__main__":
    run()