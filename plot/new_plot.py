import numpy as np
import rasterio
import os
import datetime
import pandas as pd
import json
import matplotlib.pyplot as plt
from functools import cache
import calendar

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
        self.reservoir_dependent_farmers = self.set_reservoir_dependent_farmers(2011, 2017)
        self.non_surface_water_dependent_farmers = self.set_non_surface_water_dependent_farmers(2011, 2017)
        self.farmers_to_analyse = self.non_surface_water_dependent_farmers
        self.command_areas = self.read_command_areas()
        self.activation_order = self.get_activation_order()
        # plt.imshow(self.activation_order)
        # plt.show()

    def set_reservoir_dependent_farmers(self, start_year, end_year):
        cache_file = os.path.join(self.cache_folder, f'reservoir_dependent_farmers_{start_year}_{end_year}.npy')
        if not os.path.exists(cache_file):
            # read irrigation data from 2011 to 2017 for from surface, groundwater and reservoir
            surface_irrigation_total = None
            groundwater_irrigation_total = None
            reservoir_irrigation_total = None
            n_days = 0
            for year in range(start_year, end_year):
                surface_irrigation = self.read_arrays(year, 'channel irrigation')
                if surface_irrigation_total is None:
                    surface_irrigation_total = surface_irrigation
                else:
                    surface_irrigation_total += surface_irrigation
                groundwater_irrigation = self.read_arrays(year, 'groundwater irrigation')
                if groundwater_irrigation_total is None:
                    groundwater_irrigation_total = groundwater_irrigation
                else:
                    groundwater_irrigation_total += groundwater_irrigation
                reservoir_irrigation = self.read_arrays(year, 'reservoir irrigation')
                if reservoir_irrigation_total is None:
                    reservoir_irrigation_total = reservoir_irrigation
                else:
                    reservoir_irrigation_total += reservoir_irrigation

                n_days += 366 if calendar.isleap(year) else 365

            surface_irrigation_total /= n_days
            groundwater_irrigation_total /= n_days
            reservoir_irrigation_total /= n_days
            # calculate the fraction of irrigation from reservoir
            reservoir_irrigation_fraction = reservoir_irrigation_total / (surface_irrigation_total + groundwater_irrigation_total + reservoir_irrigation_total)
            reservoir_irrigation_fraction[np.isnan(reservoir_irrigation_fraction)] = 0
            # set infinite values to 1
            reservoir_irrigation_fraction[np.isinf(reservoir_irrigation_fraction)] = 1
            reservoir_dependent_farmers = reservoir_irrigation_fraction > .5
            reservoir_dependent_farmers = self.farmer_array_to_fields(reservoir_dependent_farmers, -1, correct_for_field_size=False)
            np.save(cache_file, reservoir_dependent_farmers)
        else:
            reservoir_dependent_farmers = np.load(cache_file)
        return reservoir_dependent_farmers

    def set_non_surface_water_dependent_farmers(self, start_year, end_year):
        cache_file = os.path.join(self.cache_folder, f'non_surface_water_dependent_farmers_{start_year}_{end_year}.npy')
        if not os.path.exists(cache_file):
            # read irrigation data from 2011 to 2017 for from surface, groundwater and reservoir
            surface_irrigation_total = None
            n_days = 0
            for year in range(start_year, end_year):
                surface_irrigation = self.read_arrays(year, 'channel irrigation')
                if surface_irrigation_total is None:
                    surface_irrigation_total = surface_irrigation
                else:
                    surface_irrigation_total += surface_irrigation

                n_days += 366 if calendar.isleap(year) else 365

            surface_irrigation_total /= n_days
            # surface_irrigaton_dependent_farmers
            non_surface_water_dependent_farmers = surface_irrigation_total < 2
            non_surface_water_dependent_farmers = self.farmer_array_to_fields(non_surface_water_dependent_farmers, -1, correct_for_field_size=False)
            
            np.save(cache_file, non_surface_water_dependent_farmers)
        else:
            non_surface_water_dependent_farmers = np.load(cache_file)
        return non_surface_water_dependent_farmers

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

    def get_command_area_indices(self, command_area_id, subset=...):
        return np.where(self.command_areas[subset] == command_area_id)

    def get_honeybees_data(self, varname, start_year, end_year, fileformat='csv'):
        df = pd.read_csv(os.path.join(self.report_folder, varname + '.' + fileformat), index_col=0)
        dates = df.index.tolist()
        dates = [datetime.datetime.strptime(dt, "%Y-%m-%d") for dt in dates]
        df.index = dates
        # filter on start and end year
        df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
        # get mean dataframe by year
        df = df.groupby(df.index.year).mean()
        # return hydraulic head as numpy array and years as list
        return np.array(df[varname].tolist()), df.index.tolist()

    def read_arrays(self, year, name, mode='full_year'):
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
        return total
    
    def get_values_head_vs_tail(self, year, fn, name, correct_for_field_size=False, mode='full_year'):
        cache_file = os.path.join(self.cache_folder, f'head_vs_tail_{name}_{mode}_{year}.json')
        if not os.path.exists(cache_file):
            print('year', year, 'for', name, 'not in cache')

            total = self.read_arrays(year, name, mode=mode)

            mapping = np.full(self.command_areas.max() + 1, 0, dtype=np.int32)
            command_area_ids = np.unique(self.command_areas)
            assert command_area_ids[0] == -1
            command_area_ids = command_area_ids[1:]
            mapping[command_area_ids] = np.arange(0, command_area_ids.size, dtype=np.int32)
            command_areas_mapped = mapping[self.command_areas]
            command_areas_mapped[self.command_areas == -1] = -1
            
            array = self.farmer_array_to_fields(total, 0, correct_for_field_size=correct_for_field_size)
            array = array[self.farmers_to_analyse]
            # plt.imshow(array)
            # plt.show()

            by_command_area = {}
            for command_area_id in command_area_ids:
                command_area_id = command_area_id.item()
                command_area = self.get_command_area_indices(command_area_id, subset=self.farmers_to_analyse)
                activation_order_area = self.activation_order[self.farmers_to_analyse][command_area]
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
            print('year', year, 'for', name, 'in cache')
            with open(cache_file, 'r') as f:
                by_command_area = json.load(f)
        return by_command_area

    def get_values_small_vs_large(self, year, fn, name, correct_for_field_size=False, mode='full_year'):
        cache_file = os.path.join(self.cache_folder, f'small_vs_large_{name}_{mode}_{year}.json')
        if not os.path.exists(cache_file):
            print('year', year, 'for', name, 'not in cache')

            total = self.read_arrays(year, name, mode=mode)
            field_size = self.get_field_size()
            if correct_for_field_size:
                total = total / field_size
            # get median field size
            median_field_size = np.percentile(field_size, 50)
            small_fields = field_size < median_field_size
            by_field_size = {
                'small': fn(total[small_fields]),
                'large': fn(total[~small_fields]),
            }
            with open(cache_file, 'w') as f:
                json.dump(by_field_size, f, cls=MyEncoder)
        else:
            print('year', year, 'for', name, 'in cache')
            with open(cache_file, 'r') as f:
                by_field_size = json.load(f)
        return by_field_size

    def plot_tail_vs_head_end(self, start_year, end_year, ax, label, *args, **kwargs):
        tail_end = []
        head_end = []
        years = []
        for year in range(start_year, end_year + 1):
            by_command_area = self.get_values_head_vs_tail(*args, year=year, **kwargs)
            year_values_tail_end = 0
            year_values_head_end = 0
            for command_area_id, values in by_command_area.items():
                year_values_tail_end += values['tail_end']
                year_values_head_end += values['head_end']
            tail_end.append(year_values_tail_end)
            head_end.append(year_values_head_end)
            years.append(year)
        ax.plot(years, tail_end, label='tail end')
        ax.plot(years, head_end, label='head end')
        ax.legend()
        ax.set_xlabel('year')
        ax.set_ylabel(label)

    def plot_small_vs_large_farmer(self, start_year, end_year, ax, label, *args, **kwargs):
        small_farmer = []
        large_farmer = []
        years = []
        for year in range(start_year, end_year + 1):
            by_size = self.get_values_small_vs_large(*args, year=year, **kwargs)
            small_farmer.append(by_size['small'])
            large_farmer.append(by_size['large'])
            years.append(year)

        ax.plot(years, small_farmer, label='small farmer', color='red')
        ax.plot(years, large_farmer, label='large farmer', color='green')
        ax.legend()
        ax.set_xlabel('year')
        ax.set_ylabel(label)


    def create_plot(self, start_year, end_year):
        hydraulic_head, hydraulic_head_years = self.get_honeybees_data('hydraulic head', start_year=start_year, end_year=end_year)
        
        fig, ax = plt.subplots(3, 4, figsize=(20, 20))

        ax[0][0].plot(hydraulic_head, hydraulic_head_years, label='hydraulic head')
        
        self.plot_tail_vs_head_end(start_year, end_year, ax[1][0], label="reservoir irrigation", name="reservoir irrigation", fn=np.sum, correct_for_field_size=True, mode='full_year')
        self.plot_tail_vs_head_end(start_year, end_year, ax[1][1], label="groundwater irrigation", name="groundwater irrigation", fn=np.sum, correct_for_field_size=True, mode='full_year')
        self.plot_tail_vs_head_end(start_year, end_year, ax[2][0], label="channel irrigation", name="channel irrigation", fn=np.sum, correct_for_field_size=True, mode='full_year')
        self.plot_tail_vs_head_end(start_year, end_year, ax[1][2], label="farmers with sugarcane", name="crops_kharif", fn=is_sugarcane, correct_for_field_size=False, mode='first_day_of_year')
        self.plot_tail_vs_head_end(start_year, end_year, ax[1][3], label="farmers with well", name="well_irrigated", fn=np.sum, correct_for_field_size=False, mode='first_day_of_year')
        
        self.plot_small_vs_large_farmer(start_year, end_year, ax[2][2], label="farmers with sugarcane", name="crops_kharif", fn=is_sugarcane, correct_for_field_size=False, mode='first_day_of_year')
        self.plot_small_vs_large_farmer(start_year, end_year, ax[2][3], label="farmers with well", name="well_irrigated", fn=np.sum, correct_for_field_size=False, mode='first_day_of_year')

        plt.legend()
        # plt.show()
        plt.savefig('plot.png')

p = Plot('sugarcane')

START_YEAR = 2011
END_YEAR = 2017

p.create_plot(START_YEAR, END_YEAR)