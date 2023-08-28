from hydromt_geb import GEBModel
from hydromt.config import configread

from config import INPUT, parser

from build_model import create_logger

parser.add_argument('--data_libs', '-d', type=str, nargs='+', default=[r"../DataDrive/original_data/data_catalog.yml"])
args = parser.parse_known_args()[0]

if __name__ == '__main__':
    yml = r"./hydromt.yml"

    opt = configread(yml)
    
    geb_model = GEBModel(root=INPUT, mode='w+', data_libs=args.data_libs, logger=create_logger())
    geb_model.read()
    geb_model.update(opt={
        "setup_farmers_from_csv": {
            "path": INPUT.parent / 'preprocessing' / "agents" / "farmers" / "farmers.csv",
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