import pandas as pd 
import csv
from datetime import datetime
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By

def retrieve_and_create_df(year_start:int, year_end:int):
    
    url = "http://www.hidro.gov.ar/oceanografia/Tmareas/Form_Tmareas.asp"
    driver = webdriver.Chrome()
    driver.get(url)

    data_frames = []

    for year_input in range(year_start, year_end + 1):
        for month_input in range(1, 13):
            # Select locality
            locality = driver.find_element(By.XPATH, '//*[@id="main"]/div/section/div/article/form/div/div[2]/div/select')
            locality.click()

            san_fernando = driver.find_element(By.XPATH, '//*[@id="main"]/div/section/div/article/form/div/div[2]/div/select/option[46]')
            san_fernando.click()

            # Select year and month
            year_select = driver.find_element(By.XPATH, '//*[@id="main"]/div/section/div/article/form/div/div[1]/div/select')
            year_select.click()
            year_selected = driver.find_element(By.XPATH, f'//*[@id="main"]/div/section/div/article/form/div/div[1]/div/select/option[@value="{year_input}"]')
            year_selected.click()
            
            month_select = driver.find_element(By.XPATH, '//*[@id="main"]/div/section/div/article/form/div/div[3]/div/select')
            month_select.click()
            month_selected = driver.find_element(By.XPATH, f'//*[@id="main"]/div/section/div/article/form/div/div[3]/div/select/option[@value="{month_input:02d}"]')
            month_selected.click()
            
            # Execute Consulta
            ejecutar_consulta = driver.find_element(By.XPATH, '//*[@id="main"]/div/section/div/article/form/button') 
            ejecutar_consulta.click()

            sleep(5)

            # Switch to iframe context
            iframe = driver.find_element(By.ID, 'tablasdemarea')
            driver.switch_to.frame(iframe)

            table_1 = driver.find_element(By.XPATH, '//*[@id="main"]/section/article/div/div/div[4]/div/div[1]/div/table')
            table_2 = driver.find_element(By.XPATH, '//*[@id="main"]/section/article/div/div/div[4]/div/div[2]/div/table')

            # Extracting data from the tables
            table_1_data = table_1.text.split('\n')
            table_2_data = table_2.text.split('\n')

            # Store data in list of lists
            data = []

            # Extracting the current year and month
            current_year = year_input
            current_month = month_input

            # Store data for table 1
            for line in table_1_data[1:]:
                row = line.split()
                if len(row) == 3:
                    day = int(row[0])
                    hora_min = row[1]
                    altura = row[2].replace(',', '.')  # Replace comma with period
                    
                    # Create datetime object for the current row
                    current_date = datetime(current_year, current_month, day).strftime('%Y-%m-%d')
                    
                    data.append([current_date, hora_min, altura])
                else:
                    data.append([current_date] + line.split())
            
            # Store data for table 2
            for line in table_2_data[1:]:
                row = line.split()
                if len(row) == 3:
                    day = int(row[0])
                    hora_min = row[1]
                    altura = row[2].replace(',', '.')  # Replace comma with period
                    
                    # Create datetime object for the current row
                    current_date = datetime(current_year, current_month, day).strftime('%Y-%m-%d')
                    
                    data.append([current_date, hora_min, altura])
                else:
                    data.append([current_date] + line.split())

            # Convert list of lists to DataFrame
            df = pd.DataFrame(data, columns=['DIA', 'HORA:MIN', 'ALTURA (m)'])
            data_frames.append(df)

            driver.back()  # Go back to the initial page

    driver.quit()

    # Concatenate all DataFrames
    final_df = pd.concat(data_frames, ignore_index=True)

    return final_df
