import pandas as pd

# Load each CSV file
df_temp = pd.read_csv("temperature.csv")
df_wind = pd.read_csv("wind_speed.csv")
df_pressure = pd.read_csv("pressure.csv")
df_humidity = pd.read_csv("humidity.csv")
df_precip = pd.read_csv("precipitation.csv")
df_cloud = pd.read_csv("cloud_cover.csv")

# Merge them one by one on ['city', 'datetime']
df_merged = df_temp.merge(df_wind, on=['city', 'datetime']) \
                   .merge(df_pressure, on=['city', 'datetime']) \
                   .merge(df_humidity, on=['city', 'datetime']) \
                   .merge(df_precip, on=['city', 'datetime']) \
                   .merge(df_cloud, on=['city', 'datetime'])

# Optional: Save the merged dataset
df_merged.to_csv("merged_weather.csv", index=False)

# Quick confirmation
print("✅ Merged shape:", df_merged.shape)
print("📋 Columns:", df_merged.columns.tolist())