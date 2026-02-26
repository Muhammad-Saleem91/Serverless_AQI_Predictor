from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from ingestion.fetch_data import fetch_karachi_aqi_weather
from preprocessing.clean_data import clean_aqi_weather_data
from feature_engineering.feature_pipeline import build_feature_store_rows


DB_NAME = "aqi_feature_store"
COLLECTION = "aqi_features_hourly"
LOCATION_ID = "karachi"
TZ = "Asia/Karachi"

HISTORY_HOURS_FOR_FEATURES = 30  # >=24, give a bit of buffer
UPSERT_BATCH_SIZE = 1000


def _get_collection():
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI missing in environment/.env")

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    col = client[DB_NAME][COLLECTION]

    # enforce feature-store behavior
    col.create_index([("location_id", 1), ("event_timestamp", 1)], unique=True)
    col.create_index([("event_timestamp", 1)])
    return col


def _latest_event_timestamp(col) -> Optional[pd.Timestamp]:
    doc = (
        col.find({"location_id": LOCATION_ID}, {"_id": 0, "event_timestamp": 1})
        .sort("event_timestamp", -1)
        .limit(1)
    )
    docs = list(doc)
    if not docs:
        return None
    return pd.to_datetime(docs[0]["event_timestamp"], utc=True)


def _read_history_from_mongo(col, end_ts_utc: pd.Timestamp, hours: int) -> pd.DataFrame:
    start_ts_utc = end_ts_utc - pd.Timedelta(hours=hours)

    cursor = col.find(
        {
            "location_id": LOCATION_ID,
            "event_timestamp": {"$gte": start_ts_utc, "$lt": end_ts_utc},
        },
        {"_id": 0},
    ).sort("event_timestamp", 1)

    rows = list(cursor)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Convert back to the "timestamp" column name expected by your cleaner/feature builder
    df = df.rename(columns={"event_timestamp": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)

    # Keep only columns your cleaner expects + raw cols + flags (cleaner will recreate flags anyway)
    keep_cols = [c for c in df.columns if c not in ["location_id"]]
    return df[keep_cols]


def _upsert_features(col, df_features: pd.DataFrame) -> int:
    if df_features.empty:
        return 0

    df2 = df_features.copy()
    df2["event_timestamp"] = pd.to_datetime(df2["event_timestamp"], utc=True)
    records = df2.to_dict("records")

    total = 0
    for i in range(0, len(records), UPSERT_BATCH_SIZE):
        chunk = records[i : i + UPSERT_BATCH_SIZE]
        ops = [
            UpdateOne(
                {"location_id": r["location_id"], "event_timestamp": r["event_timestamp"]},
                {"$set": r},
                upsert=True,
            )
            for r in chunk
        ]
        res = col.bulk_write(ops, ordered=False)
        total += int(res.upserted_count + res.modified_count)

    return total


def run() -> None:
    col = _get_collection()

    # Current hour in Karachi, convert to UTC for storage identity
    now_local_hour = pd.Timestamp.now(tz=TZ).floor("h")
    now_utc_hour = now_local_hour.tz_convert("UTC")

    last_ts_utc = _latest_event_timestamp(col)

    if last_ts_utc is None:
        raise RuntimeError(
            "Feature store is empty. First do a one-time historical backfill (you already did 6 months)."
        )

    # Determine missing window (gap recovery)
    start_missing_utc = (last_ts_utc + pd.Timedelta(hours=1)).floor("h")

    if start_missing_utc > now_utc_hour:
        print("✅ No new hour to ingest yet.")
        return

    # Convert missing window to local dates for Open-Meteo
    start_missing_local = start_missing_utc.tz_convert(TZ)
    end_missing_local = now_utc_hour.tz_convert(TZ)

    start_date = start_missing_local.date().isoformat()
    end_date = end_missing_local.date().isoformat()

    # Fetch only what is missing (could be 1 hour or multiple hours)
    df_new_raw = fetch_karachi_aqi_weather(start_date=start_date, end_date=end_date)
    df_new_raw["timestamp"] = pd.to_datetime(df_new_raw["timestamp"])

    # Filter to only missing hours range in local naive timestamps (matches your ingestion)
    start_naive = start_missing_local.tz_localize(None)
    end_naive = end_missing_local.tz_localize(None)

    df_new_raw = df_new_raw[(df_new_raw["timestamp"] >= start_naive) & (df_new_raw["timestamp"] <= end_naive)].copy()

    if df_new_raw.empty:
        print("⚠️ Open-Meteo returned no rows for the missing window. Will try next run.")
        return

    # Pull history context from Mongo so lags/rollings compute correctly
    history_end_utc = start_missing_utc  # history strictly before the first missing hour
    df_hist = _read_history_from_mongo(col, end_ts_utc=history_end_utc, hours=HISTORY_HOURS_FOR_FEATURES)

    # Concatenate context + new raw data
    df_all = pd.concat([df_hist, df_new_raw], ignore_index=True)
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)

    # Clean + feature engineer using your functions
    df_clean = clean_aqi_weather_data(df_all)
    df_feat = build_feature_store_rows(df_clean, location_id=LOCATION_ID)

    # Keep only engineered rows for the missing window (in UTC event_timestamp)
    df_feat["event_timestamp"] = pd.to_datetime(df_feat["event_timestamp"])
    # build_feature_store_rows returns naive event_timestamp; interpret as local time then convert to UTC
    df_feat["event_timestamp"] = df_feat["event_timestamp"].dt.tz_localize(TZ).dt.tz_convert("UTC")

    df_to_write = df_feat[
        (df_feat["event_timestamp"] >= start_missing_utc) & (df_feat["event_timestamp"] <= now_utc_hour)
    ].copy()

    written = _upsert_features(col, df_to_write)
    print(
        f"✅ Hourly ingestion complete. Missing window: {start_missing_utc} → {now_utc_hour}. "
        f"Upserted/modified: {written} rows."
    )


if __name__ == "__main__":
    run()