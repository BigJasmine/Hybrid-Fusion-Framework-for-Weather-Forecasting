import pandas as pd
import os

DATA_DIR = "C:/Users/SURFACE/OneDrive - University of Bolton/API WEATHER DATA/API weather"

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ File not found: {DATA_DIR}")

def load_weather_file(filename: str, variable_name: str) -> pd.DataFrame:
    """Load a single weather CSV and rename its value column."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={df.columns[-1]: variable_name})
    return df[["datetime", "city", variable_name]]

def pivot_weather(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Pivot a weather DataFrame to wide format: datetime as index, city_variable as columns."""
    return df.pivot(index="datetime", columns="city", values=variable).add_prefix(f"{variable}_")

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names to remove problematic characters."""
    df.columns = [
        col.replace("'", "")
           .replace("`", "")
           .replace('"', "")
           .replace("#NAME?", "Unknown")
           .strip()
        for col in df.columns
    ]
    return df

def prepare_master_dataframe() -> pd.DataFrame:
    """Load, merge, reshape, and clean all weather variables into a single DataFrame."""
    # Load each variable
    temp = load_weather_file("temperature.csv", "temp")
    humidity = load_weather_file("humidity.csv", "humidity")
    wind = load_weather_file("wind_speed.csv", "wind")
    pressure = load_weather_file("pressure.csv", "pressure")
    precip = load_weather_file("precipitation.csv", "precip")
    cloud = load_weather_file("cloud_cover.csv", "cloud")

    # Pivot each to wide format
    temp_wide = pivot_weather(temp, "temp")
    humidity_wide = pivot_weather(humidity, "humidity")
    wind_wide = pivot_weather(wind, "wind")
    pressure_wide = pivot_weather(pressure, "pressure")
    precip_wide = pivot_weather(precip, "precip")
    cloud_wide = pivot_weather(cloud, "cloud")

    # Merge all on datetime
    combined = temp_wide.join(
        [humidity_wide, wind_wide, pressure_wide, precip_wide, cloud_wide],
        how="outer"
    )

    # Clean column names
    combined = clean_column_names(combined)

    print("📊 Combined shape:", combined.shape)
    print("📋 Combined columns:", combined.columns.tolist())
    print("🔢 Dtypes:\n", combined.dtypes)
    print("🧪 Sample rows:\n", combined.head(3))

    # Stack to long format
    stacked = combined.stack()
    df = stacked.reset_index()
    df.columns = ["datetime", "city_variable", "value"]

    # Split safely into variable and city
    split_cols = df["city_variable"].str.split("_", expand=True)
    if split_cols.shape[1] != 2:
        print("⚠️ Malformed column names detected. Sample:", df["city_variable"].unique()[:10])
        raise ValueError("Column splitting failed — check for malformed names.")
    df[["variable", "city"]] = split_cols

    print("🔍 Sample values before filtering:")
    print(df["value"].head(10))
    print("🧮 Rows before filtering:", len(df))

    # Filter out non-numeric values
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    print("✅ Rows after filtering:", len(df))
    if df.empty:
        raise ValueError("🚨 DataFrame is empty after filtering. Check column filtering and stacking logic.")

    # Pivot to final format
    df = df.pivot_table(index=["datetime", "city"], columns="variable", values="value").reset_index()
    df = df.sort_values(["datetime", "city"]).dropna()

    # Rename columns to match model expectations
    df = df.rename(columns={
        "temp": "temperature_celsius",
        "wind": "wind_kph",
        "pressure": "pressure_mb",
        "precip": "precip_mm"
    })

    print("✅ Final pivoted DataFrame shape:", df.shape)
    print("📋 Columns:", df.columns.tolist())
    print(df.head())

    return df