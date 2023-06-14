# -*- coding: utf-8 -*-
import os
import time 
import re
import json
import calendar
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
import chromedriver_autoinstaller
from pathlib import Path

from preconfig import ORIGINAL_DATA, PREPROCESSING_FOLDER

chromedriver_autoinstaller.install() 

CROP_PRICES_PATH = os.path.join(ORIGINAL_DATA, 'crop_prices')
os.makedirs(CROP_PRICES_PATH, exist_ok=True)
SCRAPE_PATH = os.path.join(CROP_PRICES_PATH, 'scraped')
os.makedirs(SCRAPE_PATH, exist_ok=True)

class Scraper:
    def __init__(self, i=None, headless=True):
        self.i = i
        self.headless = headless
        self.download_dir = os.path.abspath(os.path.join(os.path.join(SCRAPE_PATH, 'downloads'), str(self.i)) if self.i else os.path.join(SCRAPE_PATH, 'downloads'))
        self.link = 'http://agmarknet.gov.in/PriceTrends/SA_Pri_Month.aspx'

    def __enter__(self):
        chrome_options = webdriver.ChromeOptions() 
        prefs = {'download.default_directory': self.download_dir}
        chrome_options.add_experimental_option('prefs', prefs)
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        chrome_options.add_argument("--start-maximized")
        if self.headless:
            chrome_options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get(self.link)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.driver.quit()

    def get_commodity_select(self, path):
        commodity_element = self.driver.find_element(By.XPATH, path)
        return Select(commodity_element)

    def get_year_select(self):
        path = '//select[@id="cphBody_Year_list"]'
        year_select_elem = self.driver.find_element(By.XPATH, path)
        year_select = Select(year_select_elem)
        return year_select

    def get_month_select(self):
        path = '//select[@id="cphBody_Month_list"]'
        month_select_elem = self.driver.find_element(By.XPATH, path)
        month_select = Select(month_select_elem)
        return month_select

    def select_commodity_option(self, select_path, value, dowait=True):
        '''
        Select state value from dropdown. Wait until district dropdown
        has loaded before returning.
        '''

        year_list = self.driver.find_element(By.XPATH, '//select[@id="cphBody_Year_list"]')

        def year_select_updated(driver):
            try:
                # print(driver.find_element(By.XPATH, '//select[@id="cphBody_Year_list"]').text)
                year_list.text
            except StaleElementReferenceException:
                return True
            except:
                pass

            return False

        self.get_commodity_select(select_path).select_by_value(value)

        if dowait:
            wait = WebDriverWait(self.driver, 20)
            wait.until(year_select_updated)

        return self.get_year_select()

    def select_year_option(self, value, dowait=True):
        '''
        Select state value from dropdown. Wait until district dropdown
        has loaded before returning.
        '''
        path = '//select[@id="cphBody_Month_list"]'
        month_select_elem = self.driver.find_element(By.XPATH, path)

        def month_select_updated(driver):
            try:
                month_select_elem.text
            except StaleElementReferenceException:
                return True
            except:
                pass

            return False

        year_select = self.get_year_select()
        year_select.select_by_value(value)

        if dowait:
            wait = WebDriverWait(self.driver, 20)
            wait.until(month_select_updated)

        return self.get_month_select()

    def select_month_option(self, value, dowait=True):
        month_element = self.get_month_select()
        month_element.select_by_value(value)

    def rename(self, download_file, commodity_name, year, month):
        folder = os.path.join(SCRAPE_PATH, commodity_name)
        os.rename(
            download_file,
            os.path.join(folder, f'{year}_{month}.html')
        )

    def submit_download(self, commodity_name, year, month):
        self.select_month_option(month)
        path = f'Agmarknet_State_wise_Wholesale_Prices_Monthly_Analysis.xls'
        download_file = os.path.join(self.download_dir, path)
        if os.path.exists(download_file):  # just make sure no old file exists
            os.remove(download_file)
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        self.driver.find_element(By.ID, "cphBody_But_Submit").click()
        self.driver.find_element(By.ID, "cphBody_Button1").click()
        while not os.path.exists(download_file):
            time.sleep(1)
        self.rename(download_file, commodity_name.replace("/", '.'), year, month)

    def lets_go(self):
        commody_select = '//select[@id="cphBody_Commodity_list"]'

        commodity_select = self.get_commodity_select(commody_select)
        commodity_values =  [o.get_attribute('value') for o in commodity_select.options[1:]]
        commodity_names = [o.text for o in commodity_select.options[1:]]

        commodities = list(zip(commodity_values, commodity_names))
        if self.i:
            commodities = commodities[self.i::N_WORKERS]

        errors = 0

        for commodity_value, commodity_name in commodities:
            folder = os.path.join(SCRAPE_PATH, commodity_name.replace('/', '.'))
            os.makedirs(folder, exist_ok=True)
            done_file = os.path.join(folder, 'done.txt')
            if os.path.exists(done_file):
                # print(f"{commodity_name} already done! Let's continue..")
                continue
            while True:
                try:
                    print(f"Process {self.i} - Working on {commodity_name}")
                    k = 1
                    years = self.select_commodity_option(commody_select, commodity_value)
                    years_values =  [ '%s' % o.get_attribute('value') for o in years.options[1:] ]
                    for year in years_values:
                        w = 0
                        if k != 1:
                            self.select_commodity_option(commody_select, commodity_value)
                        months = self.select_year_option(year)
                        month_values =  [ '%s' % o.get_attribute('value') for o in months.options[1:] ]
                        for month in month_values:
                            if os.path.exists(os.path.join(SCRAPE_PATH, commodity_name.replace('/', '.'), f'{year}_{month}.html')):
                                continue
                            w += 1
                            if w != 1:
                                self.select_commodity_option(commody_select, commodity_value)
                                self.select_year_option(year)
                            self.select_month_option(month)
                            self.submit_download(commodity_name, year, month)
                            errors = 0
                            k=k+1
                            self.driver.get(self.link)
                            path = '//select[@id="cphBody_Commodity_list"]'
                            self.driver.find_element(By.XPATH, path)

                    with open(done_file, 'w'):
                        pass
                    break
                except Exception as e:
                    errors += 1
                    print(e)
                    print('going to sleep for a bit')
                    time.sleep(30)
                    if errors > 10:
                        raise Exception
        print(f'Process {self.i} - finished')

def parse(state, crops):
    print(f"Parsing {state}")

    dates = [datetime(2000, 1, 1)]
    while dates[-1] < datetime.utcnow():
        dates.append(dates[-1] + relativedelta(months=1))

    output = pd.DataFrame(index=dates)
    for crop_name, commodity in crops.items():
        print(f'\t{commodity}')
        commody_prices = []
        commodity_folder = os.path.join(SCRAPE_PATH, commodity)
        for fn in os.listdir(commodity_folder):
            if not fn.endswith('.html'):
                continue
            date = re.match("([0-9]{4})_([0-9]{1,2})\.html", fn)
            year = int(date.group(1))
            month = int(date.group(2))
            date = datetime(year, month, 1)  # set first day of month
            if month == 2:  # there is a typo in the downloaded data from Agmarknet
                month_name = "Febraury"
            else:
                month_name = calendar.month_name[month]
            column_name = f"Prices {month_name}, {year}"
            fp = os.path.join(commodity_folder, fn)
            try:
                df = pd.read_html(fp, header=0, index_col=0)[0]
            except ValueError:
                continue
            if state.title() in df.index:
                commody_prices.append((date, df.loc[state.title(), column_name]))
        commody_prices = sorted(commody_prices, key=lambda x: x[0])
        if commody_prices:
            value_dates, values = zip(*commody_prices)
            output[crop_name] = np.nan
            output.loc[value_dates, crop_name] = values

    output = output / 100  # rs / quintal (100 kg) -> rs / kg
    output = output.interpolate(method='linear', axis=0, limit_area='inside')

    return output

    # print(output)
    # fig, ax = plt.subplots(1)
    # ax.plot(value_dates, values, label=commodity)
    # plt.legend()
    # plt.show()

def add_FRP_prices(crops):
    df = pd.read_excel(Path(CROP_PRICES_PATH, 'FRP.xlsx'), index_col=0)

    start_year = int(df.index[0][:4])
    end_year = int(df.index[-1][-4:])
    
    df.index = pd.date_range(
        start=date(start_year, 1, 1), end=date(end_year, 1, 1), freq='YS', inclusive='left'
    ) + pd.DateOffset(months=6)
    df = df.reindex(
        pd.date_range(start=date(start_year, 7, 1), end=date(end_year, 7, 1), freq='MS', inclusive='left')
    )
    df = df.interpolate(method='ffill', axis=0)
    # set sugarcane prices in crops

    for crop in df.columns:
        crops = crops.assign(sugarcane=df[crop])
    crops = crops[crops.index.year < 2022]
    return crops

def extrapolate(crops):
    fp = os.path.join(ORIGINAL_DATA, 'economics', 'WB inflation rates', 'API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_4570810.csv')
    inflation_series = pd.read_csv(fp, index_col=0, skiprows=4).loc['India']
    inflation_rates = {}
    for year in range(1960, 2022):
        inflation_rates[year] = 1 + inflation_series[str(year)] / 100

    # fill missing values, while correcting for inflation, historically
    for column_idx, column in enumerate(crops.columns):
        # find first non-missing index
        first_non_missing_idx = crops.index.get_loc(crops[column].first_valid_index())
        # find index value before first missing value
        for idx in range(first_non_missing_idx, -1, -1):
            crops.iloc[idx, column_idx] = crops.iloc[idx + 12, column_idx] / inflation_rates[crops.index[idx].year+1]

    # fill missing values, while correcting for inflation, future part
    for column_idx, column in enumerate(crops.columns):
        # find first non-missing index
        last_non_missing_idx = crops.index.get_loc(crops[column].last_valid_index())
        # find index value before first missing value
        for idx in range(last_non_missing_idx, len(crops)):
            crops.iloc[idx, column_idx] = crops.iloc[idx - 12, column_idx] * inflation_rates[crops.index[idx].year]
    return crops

def workwork(i=None):
    while True:
        try:
            with Scraper(i=i, headless=True) as scraper:
                scraper.lets_go()
            return
        except Exception as e:
            print(e)

if __name__ == '__main__':
    N_WORKERS = 10
    # with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    #     done = executor.map(workwork, list(range(1, N_WORKERS+1)))

    output_folder = os.path.join(PREPROCESSING_FOLDER, 'crops')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"crop_prices.json")

    crops = parse('Maharashtra', crops={
        "Bajra": "Bajra(Pearl Millet.Cumbu)",
        "Groundnut": "Groundnut",
        "Jowar": "Jowar(Sorghum)",
        "Paddy": "Rice",
        "Sugarcane": "Sugarcane",
        "Wheat": "Wheat",
        "Cotton": "Cotton",
        "Gram": "Green Gram (Moong)(Whole)",
        "Maize": "Maize",
        "Moong": "Green Gram (Moong)(Whole)",
        "Ragi": "Ragi (Finger Millet)",
        "Sunflower": "Sunflower",
        "Tur": "Arhar (Tur.Red Gram)(Whole)"
    })
    crops = add_FRP_prices(crops)

    crops = extrapolate(crops)

    export = {
        'time': pd.Series(crops.index).dt.strftime('%Y-%m-%d').tolist(),
        'crops': {}
    }
    for commodity in crops.columns:
        export['crops'][commodity] = crops[commodity].tolist()

    with open(output_path, 'w') as f:
        json.dump(export, f, indent=2)
