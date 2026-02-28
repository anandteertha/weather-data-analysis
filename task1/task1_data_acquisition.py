import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

LATITUDE = 35.7796
LONGITUDE = -78.6382
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_data(start_date, end_date):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "cloud_cover",
            "apparent_temperature",
        ],
        "timezone": "America/New_York",
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame(
        {
            "time_iso8601": hourly["time"],
            "X1_temperature_2m_C": hourly["temperature_2m"],
            "X2_relative_humidity_2m_pct": hourly["relative_humidity_2m"],
            "X3_surface_pressure_hPa": hourly["surface_pressure"],
            "X4_wind_speed_10m_kmh": hourly["wind_speed_10m"],
            "X5_cloud_cover_pct": hourly["cloud_cover"],
            "Y_apparent_temperature_C": hourly["apparent_temperature"],
        }
    )

    df["time_iso8601"] = pd.to_datetime(df["time_iso8601"])
    df["date"] = df["time_iso8601"].dt.date
    df["hour"] = df["time_iso8601"].dt.hour
    noon_df = df[df["hour"] == 12].copy()
    noon_df = noon_df.drop(columns=["hour"])
    noon_df["city"] = "Raleigh"

    return noon_df


def main():
    end = datetime.now().date()
    start = end - timedelta(days=2200)

    print("Fetching weather data...")
    df = fetch_weather_data(str(start), str(end))
    df = df.dropna()

    output_cols = [
        "city",
        "date",
        "X1_temperature_2m_C",
        "X2_relative_humidity_2m_pct",
        "X3_surface_pressure_hPa",
        "X4_wind_speed_10m_kmh",
        "X5_cloud_cover_pct",
        "Y_apparent_temperature_C",
    ]
    out = df[output_cols]

    out_dir = Path(__file__).parent
    out.to_csv(out_dir / "weather_data.csv", index=False)
    out.to_excel(out_dir / "weather_dataset.xlsx", index=False, sheet_name="Data")

    print(
        f"Saved {len(df)} samples to task1/weather_data.csv and task1/weather_dataset.xlsx"
    )
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
