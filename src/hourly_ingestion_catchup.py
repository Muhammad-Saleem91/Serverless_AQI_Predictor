from __future__ import annotations

import os
from typing import Optional

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

HISTORY_HOURS_FOR_FEATURES = 72  # >=24, buffer for lag/rolling
UPSERT_BATCH_SIZE = 1000


def _get_collection():
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI missing in environment/.env")

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    col = client[DB_NAME][COLLECTION]

    # Feature-store behavior: no duplicates, fast range scans
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
    """
    Read last `hours` rows from Mongo up to (but excluding) end_ts_utc.
    Returns a dataframe with a naive-UTC 'timestamp' column (aligned to hour).
    """
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

    # Convert back to the "timestamp" column expected by cleaner/feature builder
    df = df.rename(columns={"event_timestamp": "timestamp"})
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .dt.tz_localize(None)   # naive UTC
        .dt.floor("h")          # <-- FIX: hourly alignment
    )

    # Drop entity key (we add it later in feature builder)
    keep_cols = [c for c in df.columns if c != "location_id"]
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

    # Current hour in Karachi -> UTC (Mongo stores UTC)
    now_local_hour = pd.Timestamp.now(tz=TZ).floor("h")
    now_utc_hour = now_local_hour.tz_convert("UTC")

    last_ts_utc = _latest_event_timestamp(col)
    print("LAST_MONGO_TS_UTC:", last_ts_utc)

    if last_ts_utc is None:
        raise RuntimeError("Feature store is empty. Run historical backfill first.")

    # Gap recovery: backfill from last stored + 1 hour up to now
    start_missing_utc = (last_ts_utc + pd.Timedelta(hours=1)).floor("h")

    if start_missing_utc > now_utc_hour:
        print("✅ No new hour to ingest yet.")
        return

    print("START_MISSING_UTC:", start_missing_utc)
    print("NOW_UTC_HOUR:", now_utc_hour)

    # Convert missing window to local dates for Open-Meteo (date-based API)
    start_missing_local = start_missing_utc.tz_convert(TZ)
    end_missing_local = now_utc_hour.tz_convert(TZ)

    start_date = start_missing_local.date().isoformat()
    end_date = end_missing_local.date().isoformat()

    # Fetch missing date range (may include extra hours; we filter by UTC window below)
    df_new_raw = fetch_karachi_aqi_weather(start_date=start_date, end_date=end_date)

    # Convert Open-Meteo timestamps (Karachi-local clock time) -> naive UTC hourly
    df_new_raw["timestamp"] = pd.to_datetime(df_new_raw["timestamp"])
    df_new_raw["timestamp"] = (
        df_new_raw["timestamp"]
        .dt.tz_localize(TZ)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)   # naive UTC
        .dt.floor("h")          # <-- FIX: hourly alignment
    )

    print("NEW_RAW_ROWS:", len(df_new_raw))
    print("NEW_RAW_MIN:", df_new_raw["timestamp"].min() if not df_new_raw.empty else None)
    print("NEW_RAW_MAX:", df_new_raw["timestamp"].max() if not df_new_raw.empty else None)

    # Filter to exact missing UTC window (naive UTC comparisons)
    start_naive_utc = start_missing_utc.tz_localize(None)
    end_naive_utc = now_utc_hour.tz_localize(None)

    df_new_raw = df_new_raw[
        (df_new_raw["timestamp"] >= start_naive_utc) & (df_new_raw["timestamp"] <= end_naive_utc)
    ].copy()

    if df_new_raw.empty:
        print("⚠️ Open-Meteo returned no rows for the missing window. Will try next run.")
        return

    # Pull history context (strictly before the first missing hour)
    history_end_utc = start_missing_utc
    df_hist = _read_history_from_mongo(col, end_ts_utc=history_end_utc, hours=HISTORY_HOURS_FOR_FEATURES)

    # Concatenate context + new raw data
    df_all = pd.concat([df_hist, df_new_raw], ignore_index=True)

    # --- FIX: normalize timestamps + remove overlap duplicates BEFORE cleaning ---
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"]).dt.floor("h")
    df_all = df_all.sort_values("timestamp")
    df_all = df_all.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    print("DF_ALL_MIN:", df_all["timestamp"].min())
    print("DF_ALL_MAX:", df_all["timestamp"].max())
    print("DF_ALL_ROWS:", len(df_all))

    # Clean + feature engineer using your existing functions
    df_clean = clean_aqi_weather_data(df_all)

    print("CLEAN_MAX_TS:", df_clean["timestamp"].max())
    print(
        "CLEAN_NULLS_LAST_48H:\n",
        df_clean[df_clean["timestamp"] >= (df_clean["timestamp"].max() - pd.Timedelta(hours=48))].isna().sum(),
    )

    df_feat = build_feature_store_rows(df_clean, location_id=LOCATION_ID)

    print("FEAT_ROWS_TOTAL:", len(df_feat))
    print("FEAT_MIN_EVENT_TS:", df_feat["event_timestamp"].min() if not df_feat.empty else None)
    print("FEAT_MAX_EVENT_TS:", df_feat["event_timestamp"].max() if not df_feat.empty else None)

    # event_timestamp is naive UTC -> convert to UTC-aware for filtering + Mongo insert
    df_feat["event_timestamp"] = pd.to_datetime(df_feat["event_timestamp"], utc=True)

    df_to_write = df_feat[
        (df_feat["event_timestamp"] >= start_missing_utc) & (df_feat["event_timestamp"] <= now_utc_hour)
    ].copy()

    print("TO_WRITE_ROWS:", len(df_to_write))
    print("TO_WRITE_MIN:", df_to_write["event_timestamp"].min() if not df_to_write.empty else None)
    print("TO_WRITE_MAX:", df_to_write["event_timestamp"].max() if not df_to_write.empty else None)

    written = _upsert_features(col, df_to_write)
    print(
        f"✅ Hourly ingestion complete. Missing window: {start_missing_utc} → {now_utc_hour}. "
        f"Upserted/modified: {written} rows."
    )


if __name__ == "__main__":
    run()