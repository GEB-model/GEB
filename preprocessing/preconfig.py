import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
from config import *

# import os
# import yaml

# config = yaml.load(open(os.path.join(os.path.dirname(__file__), '../GEB.yml'), 'r'), Loader=yaml.FullLoader)
# INPUT = config['general']['input_folder']
# INPUT_5MIN = config['general']['input_5min_folder']
# ORIGINAL_DATA = config['general']['original_data']