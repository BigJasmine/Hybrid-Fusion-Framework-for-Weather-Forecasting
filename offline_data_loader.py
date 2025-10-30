# offline_data_loader.py
import pandas as pd

def load_kaggle_weather_data(city='Portland'):
    # Load CSVs
    temp = pd.read_csv('temperature.csv', index_col='datetime', parse_dates=True)
    humidity = pd.read_csv('humidity.csv', index_col='datetime', parse_dates=True)
    wind = pd.read_csv('wind_speed.csv', index_col='datetime', parse_dates=True)
    pressure = pd.read_csv('pressure.csv', index_col='datetime', parse_dates=True)
    print(temp.columns)

    # Filter by city
    df = pd.DataFrame({
        'temperature': temp[city],
        'humidity': humidity[city],
        'wind_speed': wind[city],
        'pressure' : pressure[city]
    })

    df = df.dropna()
    df['rainfall'] = 0.0  # Placeholder if rainfall data is missing

    return df

if __name__ == "__main__":
    df = load_kaggle_weather_data('London')
    print(df.head())
