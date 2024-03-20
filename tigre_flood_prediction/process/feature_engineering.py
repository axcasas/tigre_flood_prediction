import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# to read data from a google sheets
from google.colab import auth
import gspread
from google.auth import default

# authenticating to google
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

def load_data(url, sheet_name):

    # Opening the Google Sheets document
    sh = gc.open_by_url(url)

    # Selecting the first sheet
    worksheet = sh.get_worksheet(0)

    if sheet_name:
        worksheet = sh.worksheet(sheet_name)
    else:
        worksheet = sh.get_worksheet(0)

    # Getting all values from the sheet
    data = worksheet.get_all_values()

    # Converting to DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])

    return df

def clean_time_column(df, column_name):
    # Define a regular expression pattern to match time values along with any additional text
    time_pattern = r'(\d{1,2}:\d{2})'

    # Apply the regular expression pattern to the column and extract time values
    df[column_name] = pd.to_datetime(df[column_name].str.extract(time_pattern)[0], format='%H:%M', errors='coerce').dt.time

    return df

def transform_data_types(df):

    df = df.sort_values(by=['date', 'time'])

    # Drop columns
    df = df.drop(columns={
        'Temperature', 'Wind Speed', 'Barometer', 'Visibility'
    })

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Convert 'time' column to time format directly
    df['time'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.time

    # Convert 'temperature_celsius' to float
    df['temperature_celsius'] = pd.to_numeric(df['temperature_celsius'], errors='coerce')

    # Convert 'wind_speed_km' to float after removing commas
    df['wind_speed_km'] = pd.to_numeric(df['wind_speed_km'], errors='coerce')

    # Convert 'humidity' to integer
    df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')

    # Convert 'barometer_mbar' to integer
    df['barometer_mbar'] = pd.to_numeric(df['barometer_mbar'].str.replace(',', ''), errors='coerce')

    # Convert 'visibility' to integer
    df['visibility'] = pd.to_numeric(df['visibility'], errors='coerce')

    return df

def categorize_wind_direction(degree):

    direction_mapping = {
        (0, 22.5): 'North',
        (22.5, 67.5): 'Northeast',
        (67.5, 112.5): 'East',
        (112.5, 157.5): 'Southeast',
        (157.5, 202.5): 'South',
        (202.5, 247.5): 'Southwest',
        (247.5, 292.5): 'West',
        (292.5, 337.5): 'Northwest',
        (337.5, 360): 'North'
    }

    for key, value in direction_mapping.items():
        if key[0] <= degree < key[1]:
            return value

# Extract the degree from the wind direction and apply the categorize_wind_direction function
df['degree'] = df['wind_direction'].str.extract(r'(\d+)').astype(float)
df['Cleaned Wind Direction'] = df['degree'].apply(categorize_wind_direction)

# Print the cleaned DataFrame
df.head()

def merge_alerta_to_weather(alerta_tigre, weather, new_dataframe=None):

    # Convert the 'date' column in both DataFrames to datetime format
    alerta_tigre['date'] = pd.to_datetime(alerta_tigre['date'], dayfirst=True)
    weather['date'] = pd.to_datetime(weather['date'], dayfirst=True)

    # Merge the DataFrames based on the 'date' column
    merged_df = pd.merge(weather, alerta_tigre[['date', 'alerta_crecida']], on='date', how='left')

    # Fill in the 'alerta_crecida' column with 'NO' for dates not present in alerta
    merged_df['alerta_crecida'].fillna('NO', inplace=True)

    # Return the merged DataFrame or assign it to a new variable if requested
    if new_dataframe:
        merged_df.to_csv(new_dataframe, index=False)
    else:
        return merged_df

# Example usage without saving to a file
merged_df = merge_alerta_to_weather(alerta_tigre=tigre, weather=df)

# Example usage with saving to a file
#merge_alerta_to_weather(alerta_tigre=tigre, weather=df, new_dataframe='merged_data.csv')

merged_df.head()

def convert_types_tides(tides):

    tides['DATE'] = pd.to_datetime(tides['DATE'])

    # Convert 'TIME' column to datetime
    tides['TIME'] = pd.to_datetime(tides['TIME'])

    # Round the times in the 'tides' dataframe to the nearest hour
    tides['time'] = tides['TIME'].dt.round('H').dt.time

    # Convert 'HEIGH_m' column to float
    tides['HEIGH_m'] = tides['HEIGH_m'].astype(float)

    tides = tides[['DATE', 'time', 'HEIGH_m']]

    return tides

convert_types_tides(tides)