from pathlib import Path

from hydromt_geb import GEBModel
from hydromt.config import configread

from config import PREPROCESSING_FOLDER, ORIGINAL_DATA

if __name__ == '__main__':
    yml = r"./hydromt.yml"
    root = r"../DataDrive/sandbox/input"

    data_libs = [ORIGINAL_DATA / "data_catalog.yml"]
    opt = configread(yml)
    
    geb_model = GEBModel(root=root, mode='w+', data_libs=data_libs)
    geb_model.read()
    geb_model.update(opt={
        "setup_farmers": {
            "csv_path": Path(PREPROCESSING_FOLDER, 'agents', 'farmers', 'farmers.csv'),
            'irrigation_sources': {
                'no_irrigation': 0,
                'canals': 1,
                'well': 2,
                'tubewell': 3,
                'tank': 4,
                'other': 5
            },
            'n_seasons': 3
        }
    })