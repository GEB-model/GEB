import os
import yaml
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='../models/sandbox.yml')
args = parser.parse_known_args()[0]
assert 'config' in args
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
INPUT = Path(config['general']['input_folder'])