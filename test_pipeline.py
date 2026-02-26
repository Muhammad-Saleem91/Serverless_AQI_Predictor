from ingestion.fetch_data import fetch_karachi_aqi_weather
from preprocessing.clean_data import clean_aqi_weather_data
import pandas as pd
from datetime import datetime, timezone

end = datetime.now(timezone.utc).date()
start = end - pd.Timedelta(days=180)
df = fetch_karachi_aqi_weather(
        start_date=str(start),
        end_date=str(end)
)

df_raw = fetch_karachi_aqi_weather("2025--01", "2026-0-01")
df_clean = clean_aqi_weather_data(df_raw)

print(df_raw.isna().sum())
print(df_clean.isna().sum())
print(df_clean.head())