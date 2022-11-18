import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='GEB.yml')
args = parser.parse_known_args()[0]
assert 'config' in args

config = yaml.load(open(os.path.join(os.path.dirname(__file__), args.config), 'r'), Loader=yaml.FullLoader)
INPUT = config['general']['input_folder']
DATA_FOLDER = config['general']['data_folder']
ORIGINAL_DATA = config['general']['original_data']