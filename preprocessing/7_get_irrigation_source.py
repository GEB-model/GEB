# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import os
from rasterio.features import rasterize
import shapely
import rasterio
import numpy as np
from selenium import webdriver
from selenium.common.exceptions import UnexpectedAlertPresentException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller
import time


STATE_TRANSLATION = {
    'A & N Islands': 'Andaman and Nicobar',
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'Chattisgarh': 'Chhattisgarh',
    'D & N Haveli': 'Dadra and Nagar Haveli',
    'Daman & Diu' : 'Daman and Diu',
}

DISTRICT_TRANSLATION = {
    'Nicobar': 'Nicobar Islands',
    'North & Middle Andamans': 'North and Middle Andaman',
    'Ananthapur': 'Anantapur',
    'Kadapa': 'Y.S.R.',
    'Mahabubnagar': 'Mahbubnagar',
    'Rangareddy': 'Ranga Reddy',
    'Ahmedabad': 'Ahmadabad',
    'Panchmahal': 'Panch Mahals',
    'Sabarkantha': 'Sabar Kantha',
    'Banaskantha': 'Banas Kantha',
    'Dangs': 'The Dangs',
    'Kurushetra': 'Kurukshetra',
    'Mohindargarh': 'Mahendragarh',
    'Hissar': 'Hisar',
    'Lahaul & Spiti': 'Lahul & Spiti',
    'Bilaspur': 'Bilaspur',
    'Leh': 'Leh (Ladakh)',
    'Bandipora': 'Bandipore',
    'Shopian': 'Shupiyan',
    'Budgam': 'Badgam',
    'Chaibasa(West Singbhum)': 'Pashchimi Singhbhum',
    'Koderma': 'Kodarma',
    'Sahebganj': 'Sahibganj',
    'Saraikela-Kharsawa': 'Saraikela-kharsawan',
    'Daltanganj(Palamu)': 'Palamu',
    'East Singhbhum(Purba Sing': 'Purbi Singhbhum',
    'Garwah': 'Garhwa',
    'Bagalkote': 'Bagalkot',
    'Bangalore Urban': 'Bangalore',
    'Ramnagar': 'Ramanagara',
    'Chamarajanagar': 'Chamrajnagar',
    'Chikballapur': 'Chikballapura',
    'Amini': 'Lakshadweep',
    'Chindwara': 'Chhindwara',
    'Khandwa': 'East Nimar',
    'Khargone': 'West Nimar',
    'Narsinghpur': 'Narsimhapur',
    'Neemach': 'Neemuch',
    'Sheopurkalan': 'Sheopur',
    'Koloriang': 'Kurung Kumey',
    'Ahamad Nagar': 'Ahmadnagar',
    'Gadchiroli': 'Garhchiroli',
    'Kholapur': 'Kolhapur',
    'Rathnagiri': 'Ratnagiri',
    'Raygad': 'Raigarh',
    'Shindhudurga': 'Sindhudurg',
    'Usmanabad': 'Osmanabad',
    'Vashim': 'Washim',
    'Yeotmal': 'Yavatmal',
    'Auragabad': 'Aurangabad',
    'Beed': 'Bid',
    'Buldhana': 'Buldana',
    'Lawngtlai': 'Lawangtlai',
    'Angul': 'Anugul',
    'Jagatsinghpur': 'Jagatsinghapur',
    'Jajpur': 'Jajapur',
    'Keonjhar': 'Kendujhar',
    'Khurda': 'Khordha',
    'Balasore': 'Baleshwar',
    'Nawarangpur': 'Nabarangapur',
    'Nuapara': 'Nuapada',
    'Sonepur': 'Subarnapur',
    'Boudh': 'Bauda',
    'Deogarh': 'Debagarh',
    'Ludhina': 'Ludhiana',
    'Mohali': 'Sahibzada Ajit Singh Nagar',
    'Ropar': 'Rupnagar',
    'Shahed Bhagat Singh Nagar': 'Shahid Bhagat Singh Nagar',
    'Taran Taran': 'Tarn Taran',
    'Bhatinda': 'Bathinda',
    'F.G. Sahib': 'Fatehgarh Sahib',
    'Ferozpur': 'Firozpur',
    'Gurdashpur': 'Gurdaspur',
    'Dholpur': 'Dhaulpur',
    'Dungerpur': 'Dungarpur',
    'Jalore': 'Jalor',
    'Banswar': 'Banswara',
    'Jhunjhunu': 'Jhunjhunun',
    'Karoli': 'Karauli',
    'Nagore': 'Nagaur',
    'Shirohi': 'Sirohi',
    'Chittoregarh': 'Chittaurgarh',
    'Baska': 'Baksa',
    'East': 'East Sikkim',
    'North': 'North Sikkim',
    'West': 'West Sikkim',
    'South': 'South Sikkim',
    'Kamrup(Metro)': 'Kamrup Metropolitan',
    'Kamrup(Rural)': 'Kamrup',
    'Karbi-Anglong': 'Karbi Anglong',
    'Odalguri': 'Udalguri',
    'Sibsagar': 'Sivasagar',
    'Dima Hasao(N.C.Hills)': 'Dima Hasao',
    'Nagapatinam': 'Nagappattinam',
    'Sivagangai': 'Sivaganga',
    'The Nilgris': 'The Nilgiris',
    'Thoothukudi': 'Thoothukkudi',
    'Tiruchirapalli': 'Tiruchirappalli',
    'Tiruvallur': 'Thiruvallur',
    'Villupuram': 'Viluppuram',
    'Virudhunagar': 'Virudunagar',
    'Kannyakumari': 'Kanniyakumari',
    'Behraich': 'Bahraich',
    'Bulandshahar': 'Bulandshahr',
    'Chotrapati Sahooji Mahara': 'Amethi',
    'Gautam Buddh Nagar': 'Gautam Buddha Nagar',
    'Jyotiba Phule Nagar': 'Amroha',
    'Kashiram Nagar': 'Kanpur Nagar',
    'Kheri': 'Lakhimpur Kheri',
    'Kushi Nagar': 'Kushinagar',
    'Maha Maya Nagar': 'Hathras',
    'Raebareli': 'Rae Bareli',
    'Ramabai Nagar/Kanpur Deha': 'Kanpur Dehat',
    'Badaayu': 'Budaun',
    'Sant Ravidas Nagar': 'Sant Ravi Das Nagar',
    'Shrawasti': 'Shravasti',
    'Sonebhadra': 'Sonbhadra',
    'Udhamsinghnagar': 'Udham Singh Nagar',
    'Haridwar': 'Hardwar',
    'Pauri Garhwal': 'Garhwal',
    '24 Parganas (North)': 'North 24 Parganas',
    '24 Parganas (South)': 'South 24 Parganas',
    'Howrah': 'Haora',
    'Malda': 'Maldah',
    'Pachim Midnapore': 'Pashchim Medinipur',
    'Purba Midnapore': 'Purba Medinipur',
    'Purulia': 'Puruliya',
    'Burdwan': 'Barddhaman',
    'Coochbehar': 'Koch Bihar',
    'Darjeeling': 'Darjiling',
    'Dinajpur (North)': 'Uttar Dinajpur',
    'Dinajpur (South)': 'Dakshin Dinajpur',
    'Hooghly': 'Hugli',
    'Jahanabad': 'Jehanabad',
    'Nawadah': 'Nawada',
    'Purnea': 'Purnia',
    'Shekhpura': 'Sheikhpura',
    'Shivhar': 'Sheohar',
    'West Champaran': 'Pashchim Champaran',
    'East Champaran': 'Purba Champaran',
    'Beejapur': 'Bijapur',
    'Raigharh': 'Raigarh',
    'Sarguja': 'Surguja',
    'Dantewara': 'Dantewada',
    'Janjgir Champa': 'Janjgir-Champa',
    'Jaspur': 'Jashpur',
    'Kabirdham': 'Kabeerdham',
    'Kanker': 'Uttar Bastar Kanker',
    'D&Nh': 'Dadra and Nagar Haveli',
    'Sangali': 'Sangli',
}

SIZE_CLASSES = (
    'Below 0.5',
    '0.5-1.0',
    '1.0-2.0',
    '2.0-3.0',
    '3.0-4.0',
    '4.0-5.0',
    '5.0-7.5',
    '7.5-10.0',
    '10.0-20.0',
    '20.0 & ABOVE',
)

def submitForm(driver, state: str, district: str, size_class: str, downloadDir: str) -> None:
    """
    This function clicks the "submit" button and downloads the excel spreadsheet from the resulting page.

    args:
        driver: the webdriver to download from
        counter: the index of the current unique dropdown configuration
        downloadDir: download directory of the current webdriver
    """
    new_file = os.path.join(downloadDir, f"{state}-{district}-{size_class}.csv")
    # If the file is already in the data folder, don't try to download
    if os.path.exists(new_file):
        return
    button_submit = driver.find_element_by_id("_ctl0_ContentPlaceHolder1_btnSubmit")
    button_submit.click()

    # Download the data as a CSV
    try:
        button_save = driver.find_element_by_xpath('//*[@id="ReportViewer1__ctl5__ctl4__ctl0_ButtonImg"]')
        button_save.click()
        time.sleep(1)
        button_excel = driver.find_element_by_xpath('//*[@id="ReportViewer1__ctl5__ctl4__ctl0_Menu"]/div[7]/a')
        button_excel.click()
    # If the download button isn't there, assume that this is an error page and skip this one by returning successfully
    except NoSuchElementException:
        driver.get("http://agcensus.dacnet.nic.in/DistCharacteristic.aspx")
        return

    # Rename the file so OS doesn't interrupt
    old_file = os.path.join(downloadDir, 'distsizedisplay5a.csv')
    while not os.path.exists(old_file):
        time.sleep(1)
    time.sleep(3)
    os.rename(old_file, new_file)

    # Click back button to go to main page
    button_back = driver.find_element_by_id("btnBack")
    button_back.click()

def configureDropdowns(driver: webdriver.chrome.webdriver.WebDriver, options: list):
    """
    This form configures the dropdown options based on the indices specified in the arguments.

    Args:
        options: a list of tuples. The tuple MUST be of length 2. The first item of the tuple is the string ID of the
                    dropdown element. The second item is the index of the option to choose from that dropdown.
                    (for example: [('year', 2), ('state', 1)] is a valid argument)
    Returns:
        l: list of the string values used for each dropdown option, in the same order as the parameter.
            (for example: ['2005', 'California'] would be a return value for the example input above)
    """
    # Make the unique combination of options from the 6 dropdowns
    l = []

    for option in options:
        elementID = option[0]
        index = option[1]
        dropdown = driver.find_element_by_id(elementID)
        current_option = dropdown.find_elements_by_tag_name("option")[index]
        current_option.click()
        l.append(current_option.text)
    return l

def findIndexByText(dropdownElement, text: str) -> int:
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


def download_census(year, states, rootDir, state_start=0, district_start=0) -> None:
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
    downloadDir = os.path.abspath(rootDir)
    if not os.path.exists(downloadDir):
        os.makedirs(downloadDir)
    prefs = {'download.default_directory' : downloadDir}
    chrome_options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(180)
    driver.get("http://agcensus.dacnet.nic.in/DistSizeClass.aspx")
    # Need to click the current year because the other dropdown options change based on this
    dropdown_year = driver.find_element_by_id("_ctl0_ContentPlaceHolder1_ddlYear")
    dropdown_options = dropdown_year.find_elements_by_tag_name('option')
    years = [option.text for option in dropdown_options]
    index_year = years.index(year)
    
    dropdown_options[index_year].click()

    dropdown_social_group = driver.find_element_by_id("_ctl0_ContentPlaceHolder1_ddlSocialGroup")
    # We only want the option with "ALL SOCIAL GROUPS" for this project
    all_social_groups_index = findIndexByText(dropdown_social_group, 'ALL SOCIAL GROUPS')
    dropdown_social_group.find_elements_by_tag_name('option')[all_social_groups_index].click()

    dropdown_state = driver.find_element_by_id("_ctl0_ContentPlaceHolder1_ddlState")
    state_options_dropdown = dropdown_state.find_elements_by_tag_name("option")
    state_options = [option.text for option in state_options_dropdown]
    state_indices = [(state_options.index(state), state) for state in states]

    for index_state, state_name in state_indices:
        try:
            dropdown_state = driver.find_element_by_id("_ctl0_ContentPlaceHolder1_ddlState")
            dropdown_state.find_elements_by_tag_name('option')[index_state].click()
            dropdown_district = driver.find_element_by_id("_ctl0_ContentPlaceHolder1_ddlDistrict")
            num_options_district = len(dropdown_district.find_elements_by_tag_name("option"))
        except UnexpectedAlertPresentException:
            print("caught error")
            alert = driver.switch_to.alert
            alert.accept()
            continue

        # If the outer loop's index is equal to the user-specified district to begin at, start from user-specified
        # district. Otherwise, we start from 0
        district_loop_start = 0
        if index_state == state_start:
            district_loop_start = district_start
        for index_district in range(district_loop_start, num_options_district):
            for size_class in (
                'Below 0.5',
                '0.5-1.0',
                '1.0-2.0',
                '2.0-3.0',
                '3.0-4.0',
                '4.0-5.0',
                '5.0-7.5',
                '7.5-10.0',
                '10.0-20.0',
                '20.0 & ABOVE',
            ):
                try:
                    sizeclass_tables = driver.find_element_by_id("_ctl0_ContentPlaceHolder1_DropDownList5")
                    sizeclass_index = findIndexByText(sizeclass_tables, size_class)
                    dropdown_district = driver.find_element_by_id("_ctl0_ContentPlaceHolder1_ddlDistrict")
                    dropdown_district.find_elements_by_tag_name('option')[index_district].click()
                    district_name = dropdown_district.find_elements_by_tag_name('option')[index_district].text
                    dropdown_tables = driver.find_element_by_id("_ctl0_ContentPlaceHolder1_ddlTables")
                    landuse_table_index = findIndexByText(dropdown_tables, 'SOURCES OF IRRIGATION')
                    dropdown_tables.find_elements_by_tag_name('option')[landuse_table_index].click()

                    dropdown_input = [
                        ('_ctl0_ContentPlaceHolder1_DropDownList5', sizeclass_index),
                        ('_ctl0_ContentPlaceHolder1_ddlYear', index_year),
                        ('_ctl0_ContentPlaceHolder1_ddlSocialGroup', all_social_groups_index),
                        ('_ctl0_ContentPlaceHolder1_ddlState', index_state),
                        ('_ctl0_ContentPlaceHolder1_ddlDistrict', index_district),
                        ('_ctl0_ContentPlaceHolder1_ddlTables', landuse_table_index)
                    ]
                    # If anything in this try block fails, we will re-try the same configuration up to 3 times before
                    # we move on to the next one
                    try:
                        options = configureDropdowns(driver, dropdown_input)
                        submitForm(driver, state_name, district_name, size_class, downloadDir)
                        global _state_start
                        _state_start = index_state
                        global _district_start
                        _district_start = index_district
                        print('successfully downloaded configuration: y' + str(index_year) + '-sg' + str(all_social_groups_index) + '-s' +
                                str(index_state) + '-d' + str(index_district) + '-t' + str(landuse_table_index))

                    except Exception as e:
                        # If configureDropdowns failed, then options will be null
                        if (options != None):
                            print('There was an error while submitting the form for options:\n' +
                                    'Year: ' + str(options[0]) + '\n' +
                                    'Social Group: ' + str(options[1]) + '\n' +
                                    'State: ' + str(options[2]) + '\n' +
                                    'District: ' + str(options[3]) + '\n' +
                                    'Table: ' + str(options[4]) + '\n')
                        else:
                            print('There was an error while submitting the form for options:\n' +
                                    'Year index: ' + str(index_year) + '\n' +
                                    'Social Group index: ' + str(all_social_groups_index) + '\n' +
                                    'State index: ' + str(index_state) + '\n' +
                                    'District index: ' + str(index_district) + '\n' +
                                    'Table index: ' + str(landuse_table_index) + '\n')
                        print(e)
                        for _ in range(0, 2):
                            # Retry up to 3 more times. First success breaks out of for-loop
                            try:
                                driver.get("http://agcensus.dacnet.nic.in/DistCharacteristic.aspx")
                                configureDropdowns(driver, dropdown_input)
                                submitForm(driver, state_name, district_name, downloadDir)
                                _state_start = index_state
                                _district_start = index_district
                                break
                            except Exception as e:
                                # Keep trying the same configuration
                                print(e)
                                continue
                        # Okay.. current configuration isn't working. Stop trying and move onto the next.
                        raise Exception
                except UnexpectedAlertPresentException:
                    print("caught error")
                    alert = driver.switch_to.alert
                    alert.accept()
                    continue


def get_districts() -> gpd.GeoDataFrame:
    """Reads all global admin 2 areas, and selects only those in India. Then sets all columns for size classes to -1 as a placeholder value.
    
    Returns:
        districts: all districts in India with placeholder columns for sizes.
        
    """
    districts = gpd.GeoDataFrame.from_file('DataDrive/GEB/original_data/GADM/gadm36_2.shp')
    districts = districts[districts['GID_0'] == 'IND']
    for size_class in SIZE_CLASSES:  # create empty columns
        districts[size_class] = -1
    return districts

def parse_census_file(fp: str) -> tuple[pd.DataFrame, str, str]:
    """Reads census file in csv-format from disk, and returns values. Also reads the state and district name.

    Args:
        fp: csv-filepath.
    
    Returns:
        df: DataFrame with census data for given state and district.
        state: State of census data.
        district: District for census data.
    """
    df = pd.read_csv(fp)
    # split_filename = fp.split('-')
    # state = split_filename[0]
    # district = split_filename[1]
    state = df['Textbox38'].iloc[0].replace('STATE : ', '').strip()
    district = df['Textbox44'].iloc[0].replace('DISTRICT : ', '').strip()
    return df, state, district

def match_name_and_fill(census_data: pd.DataFrame, districts, state, district):
    state = state.title()
    district = district.title()
    if state in STATE_TRANSLATION:
        state = STATE_TRANSLATION[state]
    if district in DISTRICT_TRANSLATION:
        district = DISTRICT_TRANSLATION[district]
    if district in (
        'Adilabad',
        'Hyderabad',
        'Karimnagar',
        'Khammam',
        'Mahbubnagar',
        'Medak',
        'Nalgonda',
        'Nizamabad',
        'Ranga Reddy',
        'Warangal',
    ) and state == 'Andhra Pradesh':
        state = 'Telangana'
    try:
        assert len(districts.loc[(districts['NAME_1'] == state) & (districts['NAME_2'] == district)]) == 1
    except AssertionError:
        print(state, ',', district, 'not found')
        raise
    size_class = census_data['Textbox82'].iloc[0].replace('SIZE CLASS : ', '').strip()
    ratio_gw_irrigated = (census_data['well_hd'].sum() + census_data['tubewel_hd'].sum()) / census_data['total_hold'].sum()
    districts.loc[(districts['NAME_1'] == state) & (districts['NAME_2'] == district), size_class] = ratio_gw_irrigated

def create_shapefile_and_clip_to_study_region(mask_fp: str) -> gpd.GeoDataFrame:
    """Create shapefile of India districts, and complements with downloaded census data.

    Args:
        mask_fp: Filepath of the study region shapefile.
    
    Returns:
        study_region_districts: GeoDataFrame wit all districts and downloaded census data.
    """
    all_districts = get_districts()
    for _, fn in enumerate(os.listdir(csv_dir)):
        census_data, state, district = parse_census_file(os.path.join(csv_dir, fn))
        if state == 'LAKSHADWEEP':
            continue
        if state == 'DELHI':
            continue
        if district == 'BRIHANMUMBAI':
            continue
        match_name_and_fill(census_data, all_districts, state, district)
    mask = gpd.GeoDataFrame.from_file(mask_fp)  
    study_region_districts = all_districts.loc[gpd.clip(all_districts, mask).index]  # clip by study region
    return study_region_districts

def export_irrigation_sources(irrigation_source_shapefile, root_dir):
    """
    """
    with rasterio.open(f'DataDrive/GEB/input/areamaps/submask.tif') as src:
        profile = src.profile
        transform = src.profile['transform']
        shape = src.profile['height'], src.profile['width']
        profile['nodata'] = -1
        profile['count'] = len(SIZE_CLASSES)
        profile['dtype'] = np.float32

    with rasterio.open(os.path.join(root_dir, 'irrigation_source.tif'), 'w', **profile) as dst:
        irrigation_source_shapefile['total'] = irrigation_source_shapefile[list(SIZE_CLASSES)].sum(axis=1)
        for i, size_class in enumerate(SIZE_CLASSES, start=1):
            irrigation_source_shapefile[size_class] = irrigation_source_shapefile[size_class] / irrigation_source_shapefile['total']

            geometries = [(shapely.geometry.mapping(geom), value) for value, geom in zip(irrigation_source_shapefile[size_class].tolist(), irrigation_source_shapefile['geometry'].tolist())]

            farm_size = rasterize(geometries, out_shape=shape, fill=-1, transform=transform, dtype=np.float32, all_touched=True)
            dst.write(farm_size, i)

    with open(os.path.join(root_dir, 'irrigation_source.txt'), 'w') as f:
        f.write("\n".join(SIZE_CLASSES))

if __name__ == '__main__':

    chromedriver_autoinstaller.install()

    year = '2010-11'
    root_dir = os.path.join('DataDrive', 'GEB', 'input', 'agents', 'irrigation_source', year)
    csv_dir = os.path.join(root_dir, 'csv')
    states = ['ANDHRA PRADESH', 'KARNATAKA', 'MAHARASHTRA']

    # while True:

    #     while True:
    #         try:
    #             download_census(year, states, csv_dir)
    #             break
    #         except Exception as e:
    #             print(e)
    #             print('downloadFiles failed, continuing from where it failed')
    #             continue
    #     print('We finished downloading everything')

    #     error = False
    #     for fn in os.listdir(csv_dir):
    #         fp = os.path.join(csv_dir, fn)
    #         if not os.path.getsize(fp) > 0:
    #             os.remove(os.path.join(csv_dir, fn))
    #             error = True
    #             print(f"Error occured for {fn}. Removing file and restarting.")
    #     if not error:
    #         break
    mask_fn = 'DataDrive/GEB/input/areamaps/mask.shp'
    irrigation_source_shapefile = create_shapefile_and_clip_to_study_region(mask_fn)

    # copy size classes for enclaved Hyderabad from surounding Ranga Reddy district
    for size_class in SIZE_CLASSES:
        irrigation_source_shapefile.loc[irrigation_source_shapefile['GID_2'] == 'IND.32.2_1', size_class] = irrigation_source_shapefile.loc[irrigation_source_shapefile['GID_2'] == 'IND.32.9_1', size_class].values[0]
        assert (~np.isnan(irrigation_source_shapefile[size_class])).all()  # make sure all are filled now
    
    export_irrigation_sources(irrigation_source_shapefile, root_dir)


