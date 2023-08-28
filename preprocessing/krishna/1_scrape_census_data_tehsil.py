# -*- coding: utf-8 -*-
import difflib
import geopandas as gpd
import pandas as pd
import os
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import UnexpectedAlertPresentException, NoAlertPresentException, TimeoutException, NoSuchElementException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from tqdm import tqdm
import numpy as np
import shutil
from io import StringIO
from multiprocessing import current_process
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from preconfig import ORIGINAL_DATA, INPUT

N_PROCESSES = 1
SIZE_CLASSES = [
    "Below 0.5", "0.5-1.0", "1.0-2.0", "2.0-3.0", "3.0-4.0",
    "4.0-5.0", "5.0-7.5", "7.5-10.0", "10.0-20.0", "20.0 & ABOVE"
]

class CensusScraper:
    def __init__(self, url, root_dir, download_dir, dropdowns, download_name, headless=True):
        self.url = url
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self.root_download_dir = download_dir
        self.dropdowns = dropdowns
        self.download_name = download_name
        self.headless = headless

    def __enter__(self):
        """
        Function to download files from agcensus website

        Args:
            index_year: index of the year dropdown
            rootDir: folder to download files to
            state_start: index of state dropdown to start from
            district_start: index of district dropdown to start from
        """
        options = Options()
        if self.headless:
            options.add_argument('headless')

        options.add_argument('log-level=3')
        # This option fixes a problem with timeout exceptions not being thrown after the limit has been reached
        options.add_argument('--dns-prefetch-disable')
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        # Makes a new download directory for each year index
        current_process_identifier = current_process()._identity
        if len(current_process_identifier) == 0:
            current_process_identifier = ''
        elif len(current_process_identifier) == 1:
            current_process_identifier = str(current_process_identifier[0])
        else:
            raise ValueError
        self.download_dir = os.path.abspath(os.path.join('tmp', current_process_identifier))
        os.makedirs(self.download_dir, exist_ok=True)
        prefs = {'download.default_directory': self.download_dir}
        options.add_experimental_option('prefs', prefs)
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(30)
        self.driver.implicitly_wait(30)

        self.driver.get(self.url)
        return self


    def findIndexByText(self, dropdownElement, text: str) -> int:
        """
        Function to find the index of element in dropdown menu by the text.

        Args:
            dropdownElement: Selenium dropdown element.
            ext: String to match with option.
        
        Returns:
            index: Index of the option in the specified dropdown, where the text matches the option's text
        """
        if isinstance(text, tuple):
            text, value = text
            for i in range (0, len(dropdownElement.find_elements(By.TAG_NAME, 'option'))):
                if dropdownElement.find_elements(By.TAG_NAME, 'option')[i].text == text and dropdownElement.find_elements(By.TAG_NAME, 'option')[i].get_attribute('value') == value:
                    return i
        else:
            for i in range (0, len(dropdownElement.find_elements(By.TAG_NAME, 'option'))):
                if dropdownElement.find_elements(By.TAG_NAME, 'option')[i].text == text:
                    return i
        options = [option.text for option in dropdownElement.find_elements(By.TAG_NAME, 'option')]
        options = difflib.get_close_matches(text, options, n=len(options), cutoff=0)
        print('No option with text: ' + text + ' was found, potential options are: ' + ", ".join(options))
        raise ValueError('No option with text: ' + text + ' was found, potential options are: ' + ", ".join(options))

    def configure_dropdown(self, ID, value, sleep=1):
        # print("selecting", value)
        time.sleep(sleep)
        while True:
            try:
                element = self.driver.find_element(By.ID, ID)
            except UnexpectedAlertPresentException:
                print("Accepting alert")
                try:
                    self.driver.switch_to.alert.accept()
                except NoAlertPresentException:
                    pass
            else:
                break
        
        WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.TAG_NAME, 'option')))
        try:
            to_select = element.find_elements(By.TAG_NAME, 'option')[self.findIndexByText(element, value)]
        except ValueError:
            raise
        else:
            if not to_select.is_selected():
                time.sleep(1)
                to_select.click()

    def download_file(self, new_file) -> None:
        if os.path.exists(os.path.join(self.download_dir, self.download_name)):
            os.remove(os.path.join(self.download_dir, self.download_name))
        # If the file is already in the data folder, don't try to download
        if os.path.exists(new_file):
            return
        button_submit = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, '_ctl0_ContentPlaceHolder1_btnSubmit')))
        button_submit.click()

        # time.sleep(3)

        if "No Record Found" in self.driver.find_element(By.XPATH, "//body").get_attribute('outerHTML'):
            with open(new_file, 'w') as f:
                pass
        else:
            button_save = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="ReportViewer1__ctl5__ctl4__ctl0_ButtonImg"]')))
            button_save.click()

            button_excel = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="ReportViewer1__ctl5__ctl4__ctl0_Menu"]/div[7]/a')))
            button_excel.click()

            max_wait = 30
            current_wait = 0

            # Rename the file so OS doesn't interrupt
            old_file = os.path.join(self.download_dir, self.download_name)
            while not os.path.exists(old_file):
                if current_wait >= max_wait:
                    raise Exception('Download failed')
                time.sleep(1)
                current_wait += 1
            time.sleep(3)
            os.rename(old_file, new_file)

        # Click back button to go to main page
        button_back = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'btnBack')))
        button_back.click()

    def get_options(self, ID):
        dropdown_state = self.driver.find_element(By.ID, ID)
        return [option.text for option in dropdown_state.find_elements(By.TAG_NAME, "option")]

    def download(self, year, to_download, subtype=None, subset=None) -> None:
        # print("Working on", kind, '-', year)
        # Need to click the current year first because the other dropdown options change based on this
        # time.sleep(5)
        dropdown_year = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, '_ctl0_ContentPlaceHolder1_ddlYear')))

        dropdown_options = dropdown_year.find_elements(By.TAG_NAME, 'option')
        years = [option.text for option in dropdown_options]
        index_year = years.index(year)
        dropdown_options[index_year].click()

        (state, district), subdistricts = to_download
        
        self.configure_dropdown("_ctl0_ContentPlaceHolder1_ddlState", state)
        self.configure_dropdown("_ctl0_ContentPlaceHolder1_ddlDistrict", district)

        for subdistrict in subdistricts:
            print(state, district, subdistrict)
            if subtype:
                self.configure_dropdown("_ctl0_ContentPlaceHolder1_ddlTehsil", subdistrict)
                for ID, value in self.dropdowns:
                    self.configure_dropdown(ID, value)
                # time.sleep(3)
                subtype_options = self.get_options(subtype)
                if subset:
                    subtype_options = [option for option in subtype_options if option in subset]
                subtype_options = sorted(subtype_options)
                for subtype_option in subtype_options:
                    folder = os.path.join(self.root_dir, subtype_option.replace('*', '#').replace('/', '.'))
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    if isinstance(subdistrict, tuple):
                        assert len(subdistrict) == 2
                        fp = os.path.join(folder, state + '-' + district  + '-' + subdistrict[0] + '#' + subdistrict[1] + '.csv')
                    else:
                        fp = os.path.join(folder, state + '-' + district  + '-' + subdistrict + '.csv')
                    if not os.path.exists(fp):
                        self.configure_dropdown(subtype, subtype_option)
                        self.download_file(fp)
            else:
                if isinstance(subdistrict, tuple):
                    assert len(subdistrict) == 2
                    fp = os.path.join(self.root_dir, state + '-' + district  + '-' + subdistrict[0] + '#' + subdistrict[1] + '.csv')
                else:
                    fp = os.path.join(self.root_dir, state + '-' + district  + '-' + subdistrict + '.csv')

                if not os.path.exists(fp):
                    try:
                        self.configure_dropdown("_ctl0_ContentPlaceHolder1_ddlTehsil", subdistrict)
                    except ValueError:
                        print("Skipping", state, district, subdistrict)
                        # continue
                        raise
                    for ID, value in self.dropdowns:
                        self.configure_dropdown(ID, value, sleep=.3)
                    # with open(fp, 'w') as f:
                    #     pass
                    self.download_file(fp)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        #if exc_type:
            # print(exc_type, exc_val)
        shutil.rmtree(self.download_dir)
        self.driver.quit()

def process_csv(folder, tehsil_2_shapefile, subdistricts, response_block, size_class_column, fields, kind, subtype):
    output_folder = os.path.join(ORIGINAL_DATA, 'census', 'output')
    os.makedirs(output_folder, exist_ok=True)
    if subtype:
        output_folder = os.path.join(output_folder, kind)
        os.makedirs(output_folder, exist_ok=True)
    if 'csv' in os.listdir(folder):
        csv_folder = os.path.join(folder, 'csv')
    else:
        csv_folder = folder
    for fn in os.listdir(csv_folder):
        unpack = fn.replace('.csv', '').split('-')
        if len(unpack) == 3:
            state, district, tehsil = unpack
        elif len(unpack) == 4:
            state, district, tehsil, division = unpack
            if division.startswith('S') or division.startswith('N'):
                tehsil = tehsil + '-' + division
            else:
                raise ValueError("Unknown subtype")
        else:
            raise ValueError("Unexpected format")
        with open(os.path.join(csv_folder, fn), 'r') as f:
            contents = f.read()
        if '#' in tehsil:
            tehsil, value = tehsil.split('#')
            tehsil = tehsil, value
        index = tehsil_2_shapefile[(state, district, tehsil)]
        subdistricts.loc[index, 'matched'] = True

        if len(contents) > 0:
            df = pd.read_csv(StringIO(contents.split('\n\n')[response_block]))
            if len(df) > 0:
                df[size_class_column] = df[size_class_column].str.replace(' - ', '-')
                df_sub = df[df[size_class_column].isin(SIZE_CLASSES)]
                assert len(df_sub) == len(SIZE_CLASSES)
                size_classes = df_sub[[size_class_column] + list(fields.keys())]

                for size_class in size_classes.itertuples():
                    for field_src, field_dst in fields.items():
                        subdistricts.loc[index, f'{getattr(size_class, size_class_column)}_{field_dst}'] = getattr(size_class, field_src)
            else:
                for size_class in SIZE_CLASSES:
                    for field in fields.values():
                        subdistricts.loc[index, f'{size_class}_{field}'] = None
        else:
            for size_class in SIZE_CLASSES:
                for field in fields.values():
                    subdistricts.loc[index, f'{size_class}_{field}'] = None

    assert (subdistricts['matched'] == True).all()
    subdistricts = subdistricts.drop('matched', axis=1)
    # subdistricts[f'{year_short_string}_subd'] = np.where(subdistricts[f'{year_short_string}_sub_alt'].isnull(), subdistricts[f'{year_short_string}_subd'], subdistricts[f'{year_short_string}_sub_alt'])
    fn = kind
    if subtype:
        fn = kind + f'_{subtype}'
    fn += f'_{year}-{year+1}.geojson'
    subdistricts.to_file(os.path.join(output_folder, fn), driver='GeoJSON')
    # subdistricts.plot()
    # import matplotlib.pyplot as plt
    # plt.show()

def download(to_download, headless, url, year, root_dir, dropdowns, download_name, subtype, subset=None):
    download_dir = 'tmp'
    os.makedirs(download_dir, exist_ok=True)
    sleep = 900
    while True:
        while True:
            try:
                with CensusScraper(url, root_dir, download_dir, dropdowns=dropdowns, download_name=download_name, headless=headless) as downloader:
                    # print('lets download')
                    downloader.download(year, to_download, subtype=subtype, subset=subset)
                break
            except (AttributeError, NoSuchElementException, TimeoutException):
                time.sleep(sleep)
                continue
            except Exception as e:
                print(e)
                print(f'Download failed, continuing from where it failed, but going for a little sleep first ({sleep} seconds)')
                time.sleep(sleep)
                continue
        # print('Downloading finished')
        break

def main(url, kind, year, dropdowns, download_name, fields, subtype=None, subset=None, size_class_column='SizeClass', scrape=True, create_file=True, headless=True, response_block=0):
    year_short_string = str(year)[-2:]
    year_str = f"{year}-{str(year + 1)[-2:]}"
    root_dir = os.path.join(ORIGINAL_DATA, 'census', 'scraped', kind, year_str)
    # if subtype:
    #     root_dir = os.path.join(root_dir, subtype)
    # root_dir = os.path.join(root_dir, year)
    # csv_dir = os.path.join(root_dir, 'csv')

    tehsil_2_shapefile = {}
    to_download = {}

    subdistricts = gpd.GeoDataFrame.from_file(os.path.join(ORIGINAL_DATA, 'census', 'tehsils.geojson'))
    subdistricts[f'{year_short_string}_dist'] = subdistricts[f'{year_short_string}_dist'].str.strip().str.upper()
    subdistricts[f'{year_short_string}_subd'] = subdistricts[f'{year_short_string}_subd'].str.strip().str.upper()
    subdistricts = subdistricts.sort_values([f'{year_short_string}_state', f'{year_short_string}_dist', f'{year_short_string}_subd'])
    subdistricts = subdistricts[subdistricts[f'{year_short_string}_subd'] != '']
    
    to_download = {}
    for state_name, districts in subdistricts.groupby(f'{year_short_string}_state'):
        for district_name, tehsils in districts.groupby(f'{year_short_string}_dist'):
            if district_name not in to_download:
                to_download[(state_name, district_name)] = []
            for idx, tehsil in tehsils.iterrows():
                tehsil_name = tehsil[f'{year_short_string}_subd']
                if tehsil[f'{year_short_string}_value']:
                    if not (isinstance(tehsil[f'{year_short_string}_value'], float) and np.isnan(tehsil[f'{year_short_string}_value'])):
                        tehsil_name = tehsil_name, tehsil[f'{year_short_string}_value']
                tehsil_2_shapefile[(state_name, district_name, tehsil_name)] = idx
                if tehsil_name != '0':
                    to_download[(state_name, district_name)].append(tehsil_name)
            if len(to_download[(state_name, district_name)]) == 0:
                del to_download[(state_name, district_name)]

    to_download = list(to_download.items())

    if scrape:
        timeout = 3600
        with ProcessPool(max_workers=1) as pool:
            for to_download_district in tqdm(to_download):
                future = pool.submit(download, timeout, to_download_district, headless=headless, url=url, year=year_str, root_dir=root_dir, dropdowns=dropdowns, download_name=download_name, subtype=subtype, subset=subset)
                while True:
                    try:
                        future.result()
                        print('completed', to_download_district[0][1].title())
                        break
                    except TimeoutError:
                        print('Process timed out, resubmitting.')
                        # import subprocess
                        # subprocess.call(r"kill $(ps -A -ostat,ppid | awk '/[zZ]/ && !a[$2]++ {print $2}')", shell=True)
                        future = pool.submit(download, timeout, to_download_district, headless=headless, url=url, year=year_str, root_dir=root_dir, dropdowns=dropdowns, download_name=download_name, subtype=subtype, subset=subset)

    if create_file:
        subdistricts['matched'] = False
        subdistricts.loc[subdistricts[f'{year_short_string}_subd'] == '0', 'matched'] = True

        if subtype:
            subdistricts['matched'] = True
            print('Creating geojsons')
            for subtype_name in tqdm(os.listdir(root_dir)):
                folder = os.path.join(root_dir, subtype_name)
                process_csv(folder, tehsil_2_shapefile, subdistricts.copy(), response_block, size_class_column, fields, kind, subtype_name)
        else:
            process_csv(root_dir, tehsil_2_shapefile, subdistricts.copy(), response_block, size_class_column, fields, kind, subtype)
    

if __name__ == '__main__':
    # initiate argument parser
    from preconfig import parser
    parser.add_argument('--noscrape', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no_create_map', action='store_true')
    args = parser.parse_args()

    scrape = not args.noscrape
    headless = args.headless
    create_file = not args.no_create_map

    if scrape:
        import chromedriver_autoinstaller
        chromedriver_autoinstaller.install()
    
    for year in (2000, 2010, 2015):
        main(
            url="http://agcensus.dacnet.nic.in/tehsilsummarytype.aspx",
            kind='farm_size',
            year=year,
            dropdowns=[
                ("_ctl0_ContentPlaceHolder1_ddlTables", 'NUMBER & AREA OF OPERATIONAL HOLDINGS'),
                ("_ctl0_ContentPlaceHolder1_ddlSocialGroup", 'ALL SOCIAL GROUPS'),
                ("_ctl0_ContentPlaceHolder1_ddlGender", 'TOTAL'),
            ],
            download_name='TehsilT1table1.csv',
            size_class_column='field4',
            fields={
                'area_total1': "area_total",
                'no_total1': "n_total"
            },
            scrape=scrape,
            create_file=create_file,
            headless=headless,
            response_block=2
        )
        crops = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx'))['CENSUS'].tolist()
        crops = [crop.upper() for crop in crops]
        main(
            url="http://agcensus.dacnet.nic.in/TalukCharacteristics.aspx",
            kind='crops',
            subset=crops,
            subtype="_ctl0_ContentPlaceHolder1_ddlCrop",
            year=year,
            dropdowns=[
                ("_ctl0_ContentPlaceHolder1_ddlTables", 'CROPPING PATTERN'),
                ("_ctl0_ContentPlaceHolder1_ddlSocialGroup", 'ALL SOCIAL GROUPS'),
            ],
            download_name='tktabledisplay6b.csv',
            fields={
                'hold': 'total_holdings',
                'irr_ar': 'irrigated_area',
                'unirr_ar': 'unirrigated_area',
                'total_ar': 'total_area'
            },
            scrape=scrape,
            create_file=create_file,
            headless=headless
        )
        # main(
        #     url="http://agcensus.dacnet.nic.in/TalukCharacteristics.aspx",
        #     kind='irrigation_status',
        #     year=year,
        #     dropdowns=[
        #         ("_ctl0_ContentPlaceHolder1_ddlTables", 'IRRIGATION STATUS'),
        #         ("_ctl0_ContentPlaceHolder1_ddlSocialGroup", 'ALL SOCIAL GROUPS'),
        #     ],
        #     download_name='tktabledisplay4.csv',
        #     fields={
        #         'total_hold': 'total_holdings',
        #         'total_area': 'total_area',
        #         'wl_irr_hd': 'wholy_irrigated_holdings',
        #         'wl_irr_ar': 'wholy_irrigated_area',
        #         'wl_unir_hd': 'wholy_unirrigated_holdings',
        #         'wl_unir_ar': 'wholy_unirrigated_area',
        #         'pl_irr_hd': 'partly_irrigated_holdings',
        #         'pl_area': 'partly_irrigated_area',
        #     },
        #     scrape=scrape,
        #     headless=headless,
        #     create_file=create_file
        # )
        if year != 2015: 
            main(
                url="http://agcensus.dacnet.nic.in/TalukCharacteristics.aspx",
                kind='irrigation_source',
                year=year,
                dropdowns=[
                    ("_ctl0_ContentPlaceHolder1_ddlTables", 'SOURCES OF IRRIGATION'),
                    ("_ctl0_ContentPlaceHolder1_ddlSocialGroup", 'ALL SOCIAL GROUPS'),
                ],
                download_name='tktabledisplay5a.csv',
                fields={
                    'total_hold': 'total_holdings',
                    'total_area': 'total_area',
                    'canal_hd': 'canals_holdings',
                    'canal_ar': 'canals_area',
                    'tank_hd': 'tank_holdings',
                    'tank_ar': 'tank_area',
                    'well_hd': 'well_holdings',
                    'well_ar': 'well_area',
                    'tubewel_hd': 'tubewell_holdings',
                    'tubewel_ar': 'tubewell_area',
                    'oth_hd': 'other_holdings',
                    'oth_ar': 'other_area',
                    'irri_hd': 'irrigated_holdings',
                    'nt_irri_ar': 'irrigated_area'
                },
                scrape=scrape,
                headless=headless,
                create_file=create_file
            )
            main(
                url="http://agcensus.dacnet.nic.in/TalukCharacteristics.aspx",
                kind='wells_and_tubewells',
                year=year,
                dropdowns=[
                    ("_ctl0_ContentPlaceHolder1_ddlTables", 'WELLS AND TUBEWELLS'),
                    ("_ctl0_ContentPlaceHolder1_ddlSocialGroup", 'ALL SOCIAL GROUPS'),
                ],
                download_name='tktabledisplay5b.csv',
                fields={
                    'total_hold': 'total_holdings',
                    'total_area': 'total_area',
                    'wells_ep': 'well_electric_pumpset',
                    'well_dp': 'well_diesel_pumpset',
                    'Total_Pumps': 'well_total',
                    'well_wp': 'well_without_pumpset',
                    'wells_nuse': 'well_not_in_use',
                    'tubewel_e': 'tubewell_electric',
                    'tubewel_d': 'tubewell_diesel',
                    'tubewells': 'tubewell_total'
                },
                scrape=scrape,
                headless=headless,
                create_file=create_file
            )
        # main(
        #     url="http://agcensus.dacnet.nic.in/TalukCharacteristics.aspx",
        #     kind='cropped_area',
        #     year=year,
        #     dropdowns=[
        #         ("_ctl0_ContentPlaceHolder1_ddlTables", 'GROSS CROPPED AREA'),
        #         ("_ctl0_ContentPlaceHolder1_ddlSocialGroup", 'ALL SOCIAL GROUPS'),
        #     ],
        #     download_name='tktabledisplay6a.csv',
        #     fields={
        #         'total_hold': 'total_holdings',
        #         'total_area': 'total_area',
        #         'Gr_irr_ar': 'gross_cropped_irrigated_area',
        #         'Gr_unirr_ar': 'gross_cropped_unirrigated_area',
        #         'Gross_ar': 'gross_cropped_area'
        #     },
        #     scrape=scrape,
        #     headless=headless,
        #     create_file=create_file
        # )
