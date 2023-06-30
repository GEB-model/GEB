import os
import yaml
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='sandbox.yml')
args = parser.parse_known_args()[0]
assert 'config' in args

config = yaml.load(open(Path(os.path.dirname(__file__), args.config), 'r'), Loader=yaml.FullLoader)
INPUT = Path(config['general']['input_folder'])
PREPROCESSING_FOLDER = Path(config['general']['preprocessing_folder'])
DATA_FOLDER = Path(config['general']['data_folder'])
ORIGINAL_DATA = Path(config['general']['original_data'])