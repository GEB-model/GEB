from pathlib import Path
from datetime import timedelta, datetime

DATA_FOLDER = Path('D:/DATA/')
CWatM_FOLDER = Path('D:/CWatM/')
PG_DB = 'abm'
INSTITUTION = 'IIASA and IVM'

TIMESTEP = timedelta(days=1)
START_TIME = datetime(2004, 1, 1, 0, 0, 0)
DISPLAY_TIMESTEPS = [
    'day',
    'week',
    'month',
    'year'
]

LEGEND = {
    "Wheat": '#7f3f3f',
    "Maize": '#7f4e4e',
    "Rice": '#7f5d5d',
    "Barley": '#7f6b6b',
    "Rye": '#7f7a7a',
    "Millet": '#757f7f',
    "Sorghum": '#667f7f',
    "Soybeans": '#587f7f',
    "Sunflower": '#497f7f',
    "Potatoes": '#3f7f7f',
    "Cassava": '#3f7f7f',
    "Sugar cane": '#3f7f7f',
    "Sugar beet": '#3f7f7f',
    "Oil palm": '#3f7f7f',
    "Rape seed / Canola": '#3f7070',
    "Groundnuts / Peanuts": '#3f6262',
    "Pulses": '#3f5353',
    "Citrus": '#3f4444',
    "Date palm": '#493f3f',
    "Grapes / Vine": '#583f3f',
    "Cotton": '#663f3f',
    "Cocoa": '#753f3f',
    "Coffee": '#7f3f3f',
    "Others perennial": '#7f3f3f',
    "Fodder grasses": '#7f3f3f',
    "Others annual": '#7f3f3f'
}

OPEN_URL_ON_RUN = False
# VARIABLE_COLOR = '#13B400'  # green
VARIABLE_COLOR = '#1386FF'  # blue
MIN_COLOR_BAR_ALPHA = 0.4
BASIN_ID = 1090009270
ANALYZE_UPSTREAM = True
MAX_UPSTREAM_RECURSION = None
# BASIN_ID = None

BEGINNING = 1979
ENDING = 2013

CROP_SCALE_FACTOR = 0.5

MAX_CANVAS_WIDTH = 800
MAX_CANVAS_HEIGHT = 800

POSTGRESQL_HOST = '127.0.0.1'
POSTGRESQL_PORT = 5432
POSTGRESQL_USER = 'jadeb'
POSTGRESQL_PASSWORD = 'ikbenhet'
