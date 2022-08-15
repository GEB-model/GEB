# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import UnexpectedAlertPresentException
import chromedriver_autoinstaller
import time
from tqdm import tqdm
from io import StringIO

from config import ORIGINAL_DATA, INPUT

SIZE_CLASSES = [
    "Below 0.5", "0.5-1.0", "1.0-2.0", "2.0-3.0", "3.0-4.0",
    "4.0-5.0", "5.0-7.5", "7.5-10.0", "10.0-20.0", "20.0 & ABOVE"
]

class CensusScraper:
    def __init__(self, url, root_dir, dropdowns, download_name, headless=True):
        self.url = url
        self.root_dir = root_dir
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

        # This option fixes a problem with timeout exceptions not being thrown after the limit has been reached
        options.add_argument('--dns-prefetch-disable')
        options.add_argument("--disable-gpu")
        # Makes a new download directory for each year index
        self.downloadDir = os.path.abspath(self.root_dir)
        if not os.path.exists(self.downloadDir):
            os.makedirs(self.downloadDir)
        prefs = {'download.default_directory': self.downloadDir}
        options.add_experimental_option('prefs', prefs)
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(600)
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
        for i in range (0, len(dropdownElement.find_elements_by_tag_name('option'))):
            if dropdownElement.find_elements_by_tag_name('option')[i].text == text:
                return i
        raise Exception('No option with text: ' + text + ' was found')

    def configure_dropdown(self, ID, value):
        print("selecting", value)
        time.sleep(1)
        while True:
            try:
                element = self.driver.find_element_by_id(ID)
            except UnexpectedAlertPresentException:
                print("Accepting alert")
                self.driver.switch_to.alert.accept()
            else:
                break
        
        element.find_elements_by_tag_name('option')[self.findIndexByText(element, value)].click()

    def download_file(self, new_file) -> None:
        if os.path.exists(os.path.join(self.downloadDir, self.download_name)):
            os.remove(os.path.join(self.downloadDir, self.download_name))
        # If the file is already in the data folder, don't try to download
        if os.path.exists(new_file):
            return
        button_submit = self.driver.find_element_by_id("_ctl0_ContentPlaceHolder1_btnSubmit")
        button_submit.click()

        if "No Record Found" in self.driver.find_element_by_xpath("//body").get_attribute('outerHTML'):
            with open(new_file, 'w') as f:
                pass
        else:
            button_save = self.driver.find_element_by_xpath('//*[@id="ReportViewer1__ctl5__ctl4__ctl0_ButtonImg"]')
            button_save.click()
            time.sleep(1)
            button_excel = self.driver.find_element_by_xpath('//*[@id="ReportViewer1__ctl5__ctl4__ctl0_Menu"]/div[7]/a')
            button_excel.click()

            # Rename the file so OS doesn't interrupt
            old_file = os.path.join(self.downloadDir, self.download_name)
            while not os.path.exists(old_file):
                time.sleep(1)
            time.sleep(3)
            os.rename(old_file, new_file)

        # Click back button to go to main page
        button_back = self.driver.find_element_by_id("btnBack")
        button_back.click()

    def get_options(self, ID):
        dropdown_state = self.driver.find_element_by_id(ID)
        return [option.text for option in dropdown_state.find_elements_by_tag_name("option")]

    def download(self, kind, year, to_download, subtype=None, n=None) -> None:
        print("Working on", kind, '-', year)
        # Need to click the current year first because the other dropdown options change based on this
        dropdown_year = self.driver.find_element_by_id("_ctl0_ContentPlaceHolder1_ddlYear")

        dropdown_options = dropdown_year.find_elements_by_tag_name('option')
        years = [option.text for option in dropdown_options]
        index_year = years.index(year)
        dropdown_options[index_year].click()

        with tqdm(total=n) as pbar:
            for state, districts in to_download.items():
                time.sleep(3)
                self.configure_dropdown("_ctl0_ContentPlaceHolder1_ddlState", state)
                for district, tehsils in districts.items():
                    self.configure_dropdown("_ctl0_ContentPlaceHolder1_ddlDistrict", district)
                    for tehsil in tehsils:
                        self.configure_dropdown("_ctl0_ContentPlaceHolder1_ddlTehsil", tehsil)
                        for ID, value in self.dropdowns:
                            self.configure_dropdown(ID, value)
                        if subtype:
                            time.sleep(3)
                            subtype_options = self.get_options(subtype)
                            for subtype_option in subtype_options:
                                folder = os.path.join(self.downloadDir, subtype_option.replace('*', '#').replace('/', '.'))
                                if not os.path.exists(folder):
                                    os.makedirs(folder)
                                fp = os.path.join(folder, state + '-' + district  + '-' + tehsil + '.csv')
                                if not os.path.exists(fp):
                                    self.configure_dropdown(subtype, subtype_option)
                                    self.download_file(fp)
                        else:
                            fp = os.path.join(self.downloadDir, state + '-' + district  + '-' + tehsil + '.csv')
                            if not os.path.exists(fp):
                                for ID, value in self.dropdowns:
                                    self.configure_dropdown(ID, value)
                                self.download_file(fp)
                        pbar.update(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        print(exc_type, exc_val)
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
        state, district, tehsil = fn.replace('.csv', '').split('-')
        with open(os.path.join(csv_folder, fn), 'r') as f:
            contents = f.read()
        district_shp_name, tehsil_shp_name = tehsil_2_shapefile[(state, district, tehsil)]
        subdistricts.loc[(subdistricts['State'] == state.title()) & (subdistricts['District'] == district_shp_name.title()) & (subdistricts['Tehsil'] == tehsil_shp_name.title()), 'matched'] = True

        if len(contents) > 0:
            df = pd.read_csv(StringIO(contents.split('\n\n')[response_block]))
            if len(df) > 0:
                df[size_class_column] = df[size_class_column].str.replace(' - ', '-')
                df_sub = df[df[size_class_column].isin(SIZE_CLASSES)]
                assert len(df_sub) == len(SIZE_CLASSES)
                size_classes = df_sub[[size_class_column] + list(fields.keys())]

                for size_class in size_classes.itertuples():
                    for field_src, field_dst in fields.items():
                        subdistricts.loc[(subdistricts['State'] == state.title()) & (subdistricts['District'] == district.title()) & (subdistricts['Tehsil'] == tehsil_shp_name.title()), f'{getattr(size_class, size_class_column)}_{field_dst}'] = getattr(size_class, field_src)
            else:
                for size_class in SIZE_CLASSES:
                    for field in fields.values():
                        subdistricts.loc[(subdistricts['State'] == state.title()) & (subdistricts['District'] == district.title()) & (subdistricts['Tehsil'] == tehsil_shp_name.title()), f'{size_class}_{field}'] = None
        else:
            for size_class in SIZE_CLASSES:
                for field in fields.values():
                    subdistricts.loc[(subdistricts['State'] == state.title()) & (subdistricts['District'] == district.title()) & (subdistricts['Tehsil'] == tehsil_shp_name.title()), f'{size_class}_{field}'] = None

    assert (subdistricts['matched'] == True).all()
    subdistricts = subdistricts.drop('matched', axis=1)
    fn = kind
    if subtype:
        print(subtype)
        fn = kind + f'_{subtype}'
    fn += f'_{year}.geojson'
    print(fn)
    subdistricts.to_file(os.path.join(output_folder, fn), driver='GeoJSON')
    print("Created map")

def main(url, kind, year, dropdowns, download_name, fields, subtype=None, size_class_column='SizeClass', scrape=True, headless=True, response_block=0):
    print('')
    print(kind, year)
    root_dir = os.path.join(ORIGINAL_DATA, 'census', kind, year)
    # if subtype:
    #     root_dir = os.path.join(root_dir, subtype)
    # root_dir = os.path.join(root_dir, year)
    # csv_dir = os.path.join(root_dir, 'csv')

    tehsil_2_shapefile = {}
    to_download = {}
    n = 0
    subdistricts = pd.read_csv(os.path.join(ORIGINAL_DATA, 'census', f'subdistricts_Bhima_{year}.csv'))
    for _, data in subdistricts.iterrows():
        state, district_name_census, shapefile_district_name, tehsil_name_census, shapefile_tehsil_name = data['State'], data['District_census'], data['District'], data['Tehsil_census'], data['Tehsil']
        assert tehsil_name_census not in tehsil_2_shapefile
        tehsil_2_shapefile[(state, district_name_census, tehsil_name_census)] = (shapefile_district_name, shapefile_tehsil_name)
        if state not in to_download:
            to_download[state] = {}
        if district_name_census not in to_download[state]:
            to_download[state][district_name_census] = []
        to_download[state][district_name_census].append(tehsil_name_census)
        n += 1

    if scrape:
        while True:
            while True:
                try:
                    with CensusScraper(url, root_dir, dropdowns=dropdowns, download_name=download_name, headless=headless) as downloader:
                        downloader.download(kind, year, to_download, n=n, subtype=subtype)
                    break
                except Exception as e:
                    print(e)
                    print('Download failed, continuing from where it failed, but going for a little sleep first (1800 seconds)')
                    time.sleep(180)
                    continue
            print('Downloading finished')
            break

    subdistricts = gpd.GeoDataFrame.from_file(os.path.join(INPUT, 'tehsils.geojson'))
    subdistricts['matched'] = False

    if subtype:
        subdistricts['matched'] = True
        for subtype_name in os.listdir(root_dir):
            folder = os.path.join(root_dir, subtype_name)
            print(folder)
            process_csv(folder, tehsil_2_shapefile, subdistricts.copy(), response_block, size_class_column, fields, kind, subtype_name)
    else:
        process_csv(root_dir, tehsil_2_shapefile, subdistricts.copy(), response_block, size_class_column, fields, kind, subtype)
    

if __name__ == '__main__':
    chromedriver_autoinstaller.install()
    scrape = False
    headless = False
    for year in ('2000-01', '2010-11', '2015-16'):
        # main(
        #     url="http://agcensus.dacnet.nic.in/tehsilsummarytype.aspx",
        #     kind='farm_size',
        #     year=year,
        #     dropdowns=[
        #         ("_ctl0_ContentPlaceHolder1_ddlTables", 'NUMBER & AREA OF OPERATIONAL HOLDINGS'),
        #         ("_ctl0_ContentPlaceHolder1_ddlSocialGroup", 'ALL SOCIAL GROUPS'),
        #         ("_ctl0_ContentPlaceHolder1_ddlGender", 'TOTAL'),
        #     ],
        #     download_name='TehsilT1table1.csv',
        #     size_class_column='field4',
        #     fields={
        #         'area_total1': "area_total",
        #         'no_total1': "n_total"
        #     },
        #     scrape=scrape,
        #     headless=headless,
        #     response_block=2
        # )
        main(
            url="http://agcensus.dacnet.nic.in/TalukCharacteristics.aspx",
            kind='crops',
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
        #     headless=headless
        # )
        # if year != '2015-16': 
        #     main(
        #         url="http://agcensus.dacnet.nic.in/TalukCharacteristics.aspx",
        #         kind='irrigation_source',
        #         year=year,
        #         dropdowns=[
        #             ("_ctl0_ContentPlaceHolder1_ddlTables", 'SOURCES OF IRRIGATION'),
        #             ("_ctl0_ContentPlaceHolder1_ddlSocialGroup", 'ALL SOCIAL GROUPS'),
        #         ],
        #         download_name='tktabledisplay5a.csv',
        #         fields={
        #             'total_hold': 'total_holdings',
        #             'total_area': 'total_area',
        #             'canal_hd': 'canals_holdings',
        #             'canal_ar': 'canals_area',
        #             'tank_hd': 'tank_holdings',
        #             'tank_ar': 'tank_area',
        #             'well_hd': 'well_holdings',
        #             'well_ar': 'well_area',
        #             'tubewel_hd': 'tubewell_holdings',
        #             'tubewel_ar': 'tubewell_area',
        #             'oth_hd': 'other_holdings',
        #             'oth_ar': 'other_area',
        #             'irri_hd': 'irrigated_holdings',
        #             'nt_irri_ar': 'irrigated_area'
        #         },
        #         scrape=scrape,
        #         headless=headless
        #     )
        #     main(
        #         url="http://agcensus.dacnet.nic.in/TalukCharacteristics.aspx",
        #         kind='wells_and_tubewells',
        #         year=year,
        #         dropdowns=[
        #             ("_ctl0_ContentPlaceHolder1_ddlTables", 'WELLS AND TUBEWELLS'),
        #             ("_ctl0_ContentPlaceHolder1_ddlSocialGroup", 'ALL SOCIAL GROUPS'),
        #         ],
        #         download_name='tktabledisplay5b.csv',
        #         fields={
        #             'total_hold': 'total_holdings',
        #             'total_area': 'total_area',
        #             'wells_ep': 'well_electric_pumpset',
        #             'well_dp': 'well_diesel_pumpset',
        #             'Total_Pumps': 'well_total',
        #             'well_wp': 'well_without_pumpset',
        #             'wells_nuse': 'well_not_in_use',
        #             'tubewel_e': 'tubewell_electric',
        #             'tubewel_d': 'tubewell_diesel',
        #             'tubewells': 'tubewell_total'
        #         },
        #         scrape=scrape,
        #         headless=headless
        #     )
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
        #     headless=headless
        # )