import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--yml', type=str, default='GEB.yml')
args = parser.parse_args()

config = yaml.load(open(os.path.join(os.path.dirname(__file__), args.yml), 'r'), Loader=yaml.FullLoader)
INPUT = config['general']['input_folder']
ORIGINAL_DATA = config['general']['original_data']