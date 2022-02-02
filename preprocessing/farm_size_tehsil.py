# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import os
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller
import time

from config import ORIGINAL_DATA, INPUT

TRANSLATE_DISTRICT = {
    "SANGLI": "SANGALI",
    "AHMADNAGAR": "AHAMAD NAGAR",
    "RAIGARH": "RAYGAD",
    "OSMANABAD": "USMANABAD",
    "BID": "BEED",
}



class CensusScraper:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __enter__(self):
        """
        Function to download files from agcensus website

        Args:
            index_year: index of the year dropdown
            rootDir: folder to download files to
            state_start: index of state dropdown to start from
            district_start: index of district dropdown to start from
        """
        chrome_options = Options()
        # This option fixes a problem with timeout exceptions not being thrown after the limit has been reached
        chrome_options.add_argument('--dns-prefetch-disable')
        # Makes a new download directory for each year index
        self.downloadDir = os.path.abspath(self.root_dir)
        if not os.path.exists(self.downloadDir):
            os.makedirs(self.downloadDir)
        prefs = {'download.default_directory': self.downloadDir}
        chrome_options.add_experimental_option('prefs', prefs)
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_page_load_timeout(180)
        self.driver.get("http://agcensus.dacnet.nic.in/tehsilsummarytype.aspx")
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
        element = self.driver.find_element_by_id(ID)
        element.find_elements_by_tag_name('option')[self.findIndexByText(element, value)].click()

    def download_file(self, new_file) -> None:
        """
        This function clicks the "submit" button and downloads the excel spreadsheet from the resulting page.

        args:
            driver: the webdriver to download from
            counter: the index of the current unique dropdown configuration
            downloadDir: download directory of the current webdriver
        """
        if os.path.exists(os.path.join(self.downloadDir, 'TehsilT1table2.csv')):
            os.remove(os.path.join(self.downloadDir, 'TehsilT1table2.csv'))
        # If the file is already in the data folder, don't try to download
        if os.path.exists(new_file):
            return
        button_submit = self.driver.find_element_by_id("_ctl0_ContentPlaceHolder1_btnSubmit")
        button_submit.click()

        button_save = self.driver.find_element_by_xpath('//*[@id="ReportViewer1__ctl5__ctl4__ctl0_ButtonImg"]')
        button_save.click()
        time.sleep(1)
        button_excel = self.driver.find_element_by_xpath('//*[@id="ReportViewer1__ctl5__ctl4__ctl0_Menu"]/div[7]/a')
        button_excel.click()

        # Rename the file so OS doesn't interrupt
        old_file = os.path.join(self.downloadDir, 'TehsilT1table2.csv')
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

    def download(self, year, to_download) -> None:
        # Need to click the current year first because the other dropdown options change based on this
        dropdown_year = self.driver.find_element_by_id("_ctl0_ContentPlaceHolder1_ddlYear")

        dropdown_options = dropdown_year.find_elements_by_tag_name('option')
        years = [option.text for option in dropdown_options]
        index_year = years.index(year)
        dropdown_options[index_year].click()

        for ID, value in [
            ("_ctl0_ContentPlaceHolder1_ddlTables", 'NUMBER & AREA OF OPERATIONAL HOLDINGS BY SIZE GROUP'),
            ("_ctl0_ContentPlaceHolder1_ddlSocialGroup", 'ALL SOCIAL GROUPS'),
            ("_ctl0_ContentPlaceHolder1_ddlGender", 'TOTAL'),
        ]:
            self.configure_dropdown(ID, value)

        for state, districts in to_download.items():
            print('state:', state)
            self.configure_dropdown("_ctl0_ContentPlaceHolder1_ddlState", state)
            for district, tehsils in districts.items():
                print('district:', district)
                self.configure_dropdown("_ctl0_ContentPlaceHolder1_ddlDistrict", district)
                for tehsil in tehsils:
                    print('tehsil:', tehsil)
                    fp = os.path.join(self.downloadDir, state + '-' + district  + '-' + tehsil + '.csv')
                    fp = fp.replace('/', '')  # slashes in names don't work with pathnames
                    if not os.path.exists(fp):
                        self.configure_dropdown("_ctl0_ContentPlaceHolder1_ddlTehsil", tehsil)
                        self.download_file(fp)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        self.driver.quit()

def main(year, scrape=True):
    root_dir = os.path.join(ORIGINAL_DATA, 'census', 'farm_size', year)
    csv_dir = os.path.join(root_dir, 'csv')

    tehsil_2_shapefile = {}
    to_download = {}
    with open(os.path.join(ORIGINAL_DATA, 'census', f'subdistricts_Bhima_{year}.csv')) as f:
        for row in f.readlines():
            row = row.strip()
            state, district, tehsil, shapefile_tehsil_name = row.split(',')
            tehsil_2_shapefile[tehsil] = shapefile_tehsil_name
            if state not in to_download:
                to_download[state] = {}
            if district not in to_download[state]:
                to_download[state][district] = []
            to_download[state][district].append(tehsil)

    if scrape:
        while True:
            while True:
                try:
                    with CensusScraper(csv_dir) as downloader:
                        downloader.download(year, to_download)
                    break
                except Exception as e:
                    print(e)
                    print('Download failed, continuing from where it failed, but going for a little sleep first (1800 seconds)')
                    time.sleep(180)
                    continue
            print('We finished downloading everything')

            error = False
            for fn in os.listdir(csv_dir):
                fp = os.path.join(csv_dir, fn)
                if not os.path.getsize(fp) > 0:
                    os.remove(os.path.join(csv_dir, fn))
                    error = True
                    print(f"Error occured for {fn}. Removing file and restarting.")
            if not error:
                break

    subdistricts = gpd.GeoDataFrame.from_file(os.path.join(ORIGINAL_DATA, 'subdistricts.shp'))
    study_region = gpd.GeoDataFrame.from_file(os.path.join(ORIGINAL_DATA, 'study_region.geojson')).to_crs(subdistricts.crs)
    subdistricts = gpd.sjoin(subdistricts, study_region, predicate='intersects')
    subdistricts['matched'] = False
    subdistricts = subdistricts.set_index('NAME')
    for fn in os.listdir(csv_dir):
        state, district, tehsil = fn.replace('.csv', '').split('-')
        tehsil_shp_name = tehsil_2_shapefile[tehsil]
        subdistricts.at[tehsil_shp_name.title(), 'matched'] = True

    print(subdistricts['matched'])

    subdistricts.plot(column='matched', vmin=0, vmax=1)
    plt.show()

    return
    
    mask_fn = os.path.join(INPUT, "areamaps", "mask.shp")
    farm_size_shapefile = create_shapefile_and_clip_to_study_region(mask_fn)

    # copy size classes for enclaved Hyderabad from surounding Ranga Reddy district
    for size_class in SIZE_CLASSES:
        # farm_size_shapefile.loc[farm_size_shapefile['GID_2'] == 'IND.32.2_1', size_class] = int(farm_size_shapefile.loc[farm_size_shapefile['GID_2'] == 'IND.32.9_1', size_class])
        assert (farm_size_shapefile[size_class] != -1).all()  # make sure all are filled now

    export_farm_sizes(farm_size_shapefile, root_dir)


if __name__ == '__main__':
    chromedriver_autoinstaller.install()
    scrape = False
    main('2000-01', scrape=scrape)
    main('2010-11', scrape=scrape)
    main('2015-16', scrape=scrape)

