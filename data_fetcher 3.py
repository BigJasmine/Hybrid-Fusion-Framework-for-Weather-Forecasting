# data_fetcher.py

import pandas as pd

def load_and_clean(filepath):
    df = pd.read_csv(filepath, index_col="datetime", parse_dates=True, dayfirst=True)
    df = df[df.index.notnull()].reset_index()
    df = df.sort_values(by='datetime')
    df = df.set_index('datetime')
    return df

def load_kaggle_weather_data(filepaths):
    return [load_and_clean(fp) for fp in filepaths]

def load_merged_weather_data(filepath="merged_weather.csv"):
    df = pd.read_csv(filepath, index_col="datetime", parse_dates=True, dayfirst=True)
    df = df[df.index.notnull()].sort_index()
    return df

def prepare_master_dataframe():
    # Load individual datasets
    temp = load_and_clean("temperature.csv")
    humidity = load_and_clean("humidity.csv")
    wind = load_and_clean("wind_speed.csv")
    pressure = load_and_clean("pressure.csv")
    precip = load_and_clean("precipitation.csv")
    cloud = load_and_clean("cloud_cover.csv")

    # Check for duplicate timestamps
    print("Duplicates:")
    print("Temperature:", temp.index.duplicated().sum())
    print("Humidity:", humidity.index.duplicated().sum())
    print("Wind Speed:", wind.index.duplicated().sum())
    print("Pressure:", pressure.index.duplicated().sum())
    print("Precipitation:", precip.index.duplicated().sum())
    print("Cloud Cover:", cloud.index.duplicated().sum())

    # Drop duplicates
    temp = temp[~temp.index.duplicated()]
    humidity = humidity[~humidity.index.duplicated()]
    wind = wind[~wind.index.duplicated()]
    pressure = pressure[~pressure.index.duplicated()]
    precip = precip[~precip.index.duplicated()]
    cloud = cloud[~cloud.index.duplicated()]

    # Add suffixes to distinguish variables
    temp = temp.add_suffix("_temp")
    humidity = humidity.add_suffix("_humidity")
    wind = wind.add_suffix("_wind")
    pressure = pressure.add_suffix("_pressure")
    precip = precip.add_suffix("_precip")
    cloud = cloud.add_suffix("_cloud")

    # Merge all datasets on datetime
    combined = temp.join([humidity, wind, pressure, precip, cloud], how="inner")

    print("Combined preview before stacking:")
    print(combined.head())
    print("Combined shape:", combined.shape)

    # Drop city_* columns before stacking
    numeric_cols = [col for col in combined.columns if not col.startswith("city_")]
    combined_numeric = combined[numeric_cols]

    # Reshape to long format
    df = combined_numeric.stack().reset_index()
    df.columns = ["datetime", "city_var", "value"]
    df[["city", "variable"]] = df["city_var"].str.extract(r"^(.*)_(\w+)$")
    df["variable"] = df["variable"].str.lower()

    # Inspect and drop non-numeric entries
    print("Non-numeric entries in 'value':")
    print(df[df["value"] == "Kabul"])

    print("Unique values in 'value' column before filtering:")
    print(df["value"].unique())

    df = df[pd.to_numeric(df["value"], errors="coerce").notnull()]
    df["value"] = df["value"].astype(float)

    # Pivot to wide format
    df = df.pivot_table(index=["datetime", "city"], columns="variable", values="value").reset_index()

    # Rename columns to match modeling convention
    df = df.rename(columns={
        "temp": "temperature_celsius",
        "wind": "wind_kph",
        "pressure": "pressure_mb",
        "precip": "precip_mm"
    })

    return df

# Alias for compatibility with main.py
fetch_weather_data = load_kaggle_weather_data