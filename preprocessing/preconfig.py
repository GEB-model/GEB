import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
from config import *

ORIGINAL_DATA = Path(config['general']['original_data'])