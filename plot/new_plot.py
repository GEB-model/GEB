import numpy as np
import rasterio
import os
import datetime
import json
import matplotlib.pyplot as plt
from functools import cache

from plotconfig import config, ORIGINAL_DATA, INPUT

def is_sugarcane(x):
    return np.count_nonzero(x == 4)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class Plot:
    def __init__(self, scenario):
        self.report_folder = os.path.join(config['general']['report_folder'], scenario)
        self.land_owners = np.load(os.path.join(self.report_folder, 'land_owners.npy'))

        self.cache_folder = os.path.join('plot', 'cache', scenario)
        os.makedirs(self.cache_folder, exist_ok=True)

        with rasterio.open(os.path.join(INPUT, 'areamaps', 'mask.tif'), 'r') as src:
            self.mask = src.read(1)
        with rasterio.open(os.path.join(INPUT, 'areamaps', 'submask.tif'), 'r') as src:
            self.submask = src.read(1)
            self.submask_transform = src.profile['transform']
        
        with rasterio.open(os.path.join(INPUT, 'areamaps', 'sub_cell_area.tif'), 'r') as src_cell_area:
            self.cell_area = src_cell_area.read(1)
        
        self.unmerged_HRU_indices = np.load(os.path.join(self.report_folder, 'unmerged_HRU_indices.npy'))
        self.scaling = np.load(os.path.join(self.report_folder, 'scaling.npy')).item()
        self.command_areas = self.read_command_areas()
        self.activation_order = self.get_activation_order()
        # plt.imshow(self.activation_order)
        # plt.show()

    def read_command_areas(self):
        fp = os.path.join(INPUT, 'routing', 'lakesreservoirs', 'subcommand_areas.tif')
        with rasterio.open(fp, 'r') as src:
            command_areas = src.read(1)
        return command_areas

    def get_activation_order(self):
        if not os.path.exists(os.path.join(self.cache_folder, 'activation_order.npy')):
            activation_order = np.load(os.path.join(self.report_folder, 'activation_order.npy'))
            activation_order = self.farmer_array_to_fields(activation_order, -1, correct_for_field_size=False)
            np.save(os.path.join(self.cache_folder, 'activation_order.npy'), activation_order)
        else:
            activation_order = np.load(os.path.join(self.cache_folder, 'activation_order.npy'))
        return activation_order

    @cache
    def get_field_size(self):
        if not os.path.exists(os.path.join(self.cache_folder, 'field_size.npy')):
            fields_decompressed = self.decompress_HRU(self.land_owners)
            fields_decompressed = fields_decompressed[self.submask == 0]
            is_field = np.where(fields_decompressed != -1)
            cell_area = self.cell_area[self.submask == 0]
            field_size = np.bincount(fields_decompressed[is_field], weights=cell_area[is_field])
            np.save(os.path.join(self.cache_folder, 'field_size.npy'), field_size)
        else:
            field_size = np.load(os.path.join(self.cache_folder, 'field_size.npy'))
        return field_size

    def farmer_array_to_fields(self, array, nofieldvalue, correct_for_field_size=True):
        if correct_for_field_size:
            field_size = self.get_field_size()
            assert field_size.size == array.size
            array /= field_size

        array = np.take(array, self.land_owners)
        array[self.land_owners == -1] = nofieldvalue
        array = self.decompress_HRU(array)
        return array

    def decompress_HRU(self, array):
        if np.issubdtype(array.dtype, np.integer):
            nanvalue = -1
        else:
            nanvalue = np.nan
        outarray = array[self.unmerged_HRU_indices]
        outarray[self.submask] = nanvalue
        return outarray

    def read_npy(self, name, dt):
        fn = os.path.join(self.report_folder, name, dt.isoformat().replace(':', '').replace('-', '') + '.npy')
        return np.load(fn)

    @cache
    def get_command_area_indices(self, command_area_id):
        return np.where(self.command_areas == command_area_id)
    
    def get_values(self, year, fn, name, correct_for_field_size=False, mode='full_year'):
        cache_file = os.path.join(self.cache_folder, f'{name}_{mode}_{year}.json')
        if not os.path.exists(cache_file):
            print('year', year)

            if mode == 'full_year':
                day = datetime.date(year, 1, 1)
                # loop through all days in year
                total = None
                while day < datetime.date(year + 1, 1, 1):
                    # read the data
                    array = self.read_npy(name, day)
                    if total is None:
                        total = array
                    else:
                        total += array
                    # go to next day
                    day += datetime.timedelta(days=1)
            elif mode == 'first_day_of_year':
                day = datetime.date(year, 1, 1)
                total = self.read_npy(name, day)
            else:
                raise ValueError(f'Unknown mode {mode}')

            mapping = np.full(self.command_areas.max() + 1, 0, dtype=np.int32)
            command_area_ids = np.unique(self.command_areas)
            assert command_area_ids[0] == -1
            command_area_ids = command_area_ids[1:]
            mapping[command_area_ids] = np.arange(0, command_area_ids.size, dtype=np.int32)
            command_areas_mapped = mapping[self.command_areas]
            command_areas_mapped[self.command_areas == -1] = -1
            
            array = self.farmer_array_to_fields(total, 0, correct_for_field_size=correct_for_field_size)
            # plt.imshow(array)
            # plt.show()

            by_command_area = {}
            for command_area_id in command_area_ids:
                command_area_id = command_area_id.item()
                command_area = self.get_command_area_indices(command_area_id)
                activation_order_area = self.activation_order[command_area]
                if (activation_order_area == -1).all():
                    continue
                activation_order_area_filtered = activation_order_area[activation_order_area != -1]
                array_area = array[command_area][activation_order_area != -1]
                activation_order_median = np.percentile(activation_order_area_filtered, 50)
                
                head_end = fn(array_area[activation_order_area_filtered < activation_order_median])
                tail_end = fn(array_area[activation_order_area_filtered >= activation_order_median])
                by_command_area[command_area_id] = {
                    'head_end': head_end,
                    'tail_end': tail_end,
                }
            with open(cache_file, 'w') as f:
                json.dump(by_command_area, f, cls=MyEncoder)
        else:
            with open(cache_file, 'r') as f:
                by_command_area = json.load(f)
        return by_command_area

    def create_plot(self, start_year, end_year):
        tail_end = []
        head_end = []
        years = []
        for year in range(start_year, end_year + 1):
            # by_command_area = p.get_values(year, fn=np.sum, correct_for_field_size=True, mode='full_year', name="reservoir irrigation")
            by_command_area = p.get_values(year, fn=is_sugarcane, mode='first_day_of_year', name="crops_kharif")
            year_values_tail_end = 0
            year_values_head_end = 0
            for command_area_id, values in by_command_area.items():
                year_values_tail_end += values['tail_end']
                year_values_head_end += values['head_end']
            tail_end.append(year_values_tail_end)
            head_end.append(year_values_head_end)
            years.append(year)

        plt.plot(years, tail_end, label='tail end')
        plt.plot(years, head_end, label='head end')
        plt.legend()
        plt.show()

p = Plot('sugarcane')

START_YEAR = 2011
END_YEAR = 2018

p.create_plot(START_YEAR, END_YEAR)

