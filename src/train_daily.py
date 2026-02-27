from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


DB_NAME = "aqi_feature_store"
COLLECTION = "aqi_features_hourly"
LOCATION_ID = "karachi"

# Training settings
LOOKBACK_DAYS = 180
TRAIN_FRAC = 0.8

# Label horizon
HORIZON_HOURS = 1  # next-hour prediction for recursive forecasting


@dataclass
class ModelResult:
    name: str
    model_path: str
    mae: float
    rmse: float


def _get_collection():
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI missing in environment/.env (local) or GitHub Secrets (CI).")

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    return client[DB_NAME][COLLECTION]


def _load_feature_data(days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    col = _get_collection()
    start = datetime.now(timezone.utc) - timedelta(days=days)

    cursor = col.find(
        {"location_id": LOCATION_ID, "event_timestamp": {"$gte": start}},
        {"_id": 0},
    ).sort("event_timestamp", 1)

    df = pd.DataFrame(list(cursor))
    if df.empty:
        raise RuntimeError("No training data found in Mongo for the given date range.")

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    return df


def _make_supervised(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build supervised dataset for next-hour AQI.
    y(t) = us_aqi(t+1)
    """
    data = df.copy()
    data["y"] = data["us_aqi"].shift(-HORIZON_HOURS)

    # Drop last row(s) without label
    data = data.dropna(subset=["y"]).reset_index(drop=True)

    # Features: drop identifiers and label
    drop_cols = {"location_id", "event_timestamp", "y"}
    feature_cols = [c for c in data.columns if c not in drop_cols]

    X = data[feature_cols]
    y = data["y"].astype(float)
    return X, y, feature_cols


def _time_split(X: pd.DataFrame, y: pd.Series, train_frac: float = TRAIN_FRAC):
    n = len(X)
    n_train = int(n * train_frac)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]
    return X_train, X_test, y_train, y_test


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred))
    return mae, rmse


def _train_models(X_train, y_train) -> Dict[str, object]:
    models = {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "rf": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "xgb": XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for m in models.values():
        m.fit(X_train, y_train)

    return models


def main() -> None:
    df = _load_feature_data(days=LOOKBACK_DAYS)
    X, y, feature_cols = _make_supervised(df)
    X_train, X_test, y_train, y_test = _time_split(X, y)

    models = _train_models(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)
    results: List[ModelResult] = []

    for name, model in models.items():
        preds = model.predict(X_test)
        mae, rmse = _evaluate(y_test.to_numpy(), preds)

        model_path = f"artifacts/{name}_model.pkl"
        joblib.dump(model, model_path)

        results.append(ModelResult(name=name, model_path=model_path, mae=mae, rmse=rmse))
        print(f"{name.upper()}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # Select best by MAE
    best = sorted(results, key=lambda r: r.mae)[0]

    # Save training metadata for serving consistency
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "location_id": LOCATION_ID,
        "lookback_days": LOOKBACK_DAYS,
        "horizon_hours": HORIZON_HOURS,
        "feature_cols": feature_cols,
        "results": [r.__dict__ for r in results],
        "best_model": {"name": best.name, "path": best.model_path, "mae": best.mae, "rmse": best.rmse},
    }

    with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open("artifacts/best_model.json", "w", encoding="utf-8") as f:
        json.dump(metadata["best_model"], f, indent=2)

    print("\n✅ Best model:", best.name, "MAE=", best.mae, "RMSE=", best.rmse)
    print("✅ Artifacts written to ./artifacts")


if __name__ == "__main__":
    main()