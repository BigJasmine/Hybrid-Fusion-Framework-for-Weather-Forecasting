import pandas as pd

# Temperature
df_temp = pd.read_csv("temperature.csv")
print("🌡️ Temperature file:")
print("Total rows:", df_temp.shape[0])
print("Unique cities:", df_temp['city'].nunique())
print("Unique timestamps:", df_temp['datetime'].nunique())
print("Columns:", df_temp.columns.tolist())

# Wind
df_wind = pd.read_csv("wind_speed.csv")
print("\n💨 Wind file:")
print("Total rows:", df_wind.shape[0])
print("Unique cities:", df_wind['city'].nunique())
print("Unique timestamps:", df_wind['datetime'].nunique())
print("Columns:", df_wind.columns.tolist())

# Pressure
df_pressure = pd.read_csv("pressure.csv")
print("\n🌬️ Pressure file:")
print("Total rows:", df_pressure.shape[0])
print("Unique cities:", df_pressure['city'].nunique())
print("Unique timestamps:", df_pressure['datetime'].nunique())
print("Columns:", df_pressure.columns.tolist())

# Humidity
df_humidity = pd.read_csv("humidity.csv")
print("\n💧 Humidity file:")
print("Total rows:", df_humidity.shape[0])
print("Unique cities:", df_humidity['city'].nunique())
print("Unique timestamps:", df_humidity['datetime'].nunique())
print("Columns:", df_humidity.columns.tolist())

# Precipitation
df_precip = pd.read_csv("precipitation.csv")
print("\n🌧️ Precipitation file:")
print("Total rows:", df_precip.shape[0])
print("Unique cities:", df_precip['city'].nunique())
print("Unique timestamps:", df_precip['datetime'].nunique())
print("Columns:", df_precip.columns.tolist())

# Cloud
df_cloud = pd.read_csv("cloud_cover.csv")
print("\n☁️ Cloud file:")
print("Total rows:", df_cloud.shape[0])
print("Unique cities:", df_cloud['city'].nunique())
print("Unique timestamps:", df_cloud['datetime'].nunique())
print("Columns:", df_cloud.columns.tolist())