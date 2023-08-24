import logging
from hydromt_geb import GEBModel
from hydromt.config import configread

from config import INPUT, config, parser

parser.add_argument('--data_libs', '-d', type=str, nargs='+', default=[r"../DataDrive/original_data/data_catalog.yml"])
parser.add_argument('--yml', '-y', type=str, default=r"./hydromt.yml")
args = parser.parse_known_args()[0]

def create_logger():
    logger = logging.getLogger(__name__)
    # set log level to debug
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    # add file handler
    INPUT.mkdir(exist_ok=True, parents=True)
    fh = logging.FileHandler(INPUT / 'hydromt.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
    
if __name__ == '__main__':
    geb_model = GEBModel(
        root=INPUT,
        mode='w+',
        data_libs=args.data_libs,
        logger=create_logger(),
    )

    poor_point = config['general']['poor_point']
    geb_model.build(
        opt=configread(args.yml),
        region={
            'subbasin': [
                [poor_point[0]], [poor_point[1]]
            ],
            'bounds': [66.55, 4.3, 93.17, 35.28]  # TODO: remove need to specify bounds
        },
    )