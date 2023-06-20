from hydromt_geb import GEBModel
from hydromt.config import configread

if __name__ == '__main__':
    yml = r"./hydromt.yml"
    root = r"../DataDrive/sandbox/input"

    data_libs = [r"../DataDrive/GEB/original_data/data_catalog.yml"]
    opt = configread(yml)
    
    geb_model = GEBModel(root=root, mode='w+', data_libs=data_libs)
    # Bhima
    # geb_model.build(opt=opt, region={'subbasin': [[75.895273], [17.370473]], 'bounds': [66.55, 4.3, 93.17, 35.28]})

    # Sanctuary
    geb_model.build(opt=opt, region={'subbasin': [[73.98727], [19.00465]], 'bounds': [66.55, 4.3, 93.17, 35.28]})