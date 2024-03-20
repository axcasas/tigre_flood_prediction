import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def scrape_weather_data(year, month, day):

    # 1. set URL
    url = f'https://www.timeanddate.com/weather/@3427761/historic?month={month}&year={year}&day={day}' # month, year, and day will be the parameters

    # 2. request
    response = requests.get(url)

    # 3. Find the table from the content of the response
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find(id='wt-his')

    # 3.1 Validate the existence of the table
    if table is not None:
        # 4 .Extract data from each row
        data_rows = []
        rows = table.find_all('tr')

        # 5. Now we loop through all the table's cells
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) == 9:  # Check if the row has the expected number of cells
                time = cells[0].text.strip()

                # Skip rows that do not contain valid time information. This will make the code to not break if it didn't find information.
                if ':' not in time:
                    continue

                # 6. Set variables with cell's content. We will use this for the dataframe columns
                temperature = cells[2].text.strip()
                weather = cells[3].text.strip()
                wind_speed = cells[4].text.strip()
                wind_direction = cells[5].text.strip()
                humidity = cells[6].text.strip()
                barometer = cells[7].text.strip()
                visibility = cells[8].text.strip()

                # 7. Split time string to extract day
                date_str = f"{year}-{month}-{day}"
                date = datetime.strptime(date_str, '%Y-%m-%d').date()

                row_data = [date, time, temperature, weather, wind_speed, wind_direction, humidity, barometer, visibility]
                data_rows.append(row_data)

        # 8. Convert the list of rows into a DataFrame
        df = pd.DataFrame(data_rows, columns=['Date', 'Time', 'Temperature', 'Weather', 'Wind Speed', 'Wind Direction', 'Humidity', 'Barometer', 'Visibility'])

    else:
        # If the table is not found
        print("The table was not found on the page.")
        # Assign default values of zero to the dataframe.
        df = pd.DataFrame({'Date': [0], 'Time': [0], 'Temperature': [0], 'Weather': [0], 'Wind Speed': [0], 'Wind Direction': [0], 'Humidity': [0], 'Barometer': [0], 'Visibility': [0]})

    return df
