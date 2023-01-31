import numpy as np
import rasterio
import os
import datetime
import pandas as pd
import json
import matplotlib.pyplot as plt
import calendar
import rasterio

from plotconfig import config, INPUT

SUM_DIVIDER = 1000

def sum_sugarcane(x):
    return np.count_nonzero(x == 4) / SUM_DIVIDER

def sum_most(x):
    return np.sum(x) / SUM_DIVIDER

def irrigation_fn(x):
    return np.mean(x) * SUM_DIVIDER  # m to mm

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
        self.input_folder = os.path.join(config['general']['input_folder'])
        with rasterio.open(os.path.join(self.input_folder, 'agents', 'farms.tif'), 'r') as src:
            self.farms = src.read(1)
        unique_land_owners = np.unique(self.farms)
        self.n = unique_land_owners[unique_land_owners != -1].size

        self.cache_folder = os.path.join('plot', 'cache', scenario)
        os.makedirs(self.cache_folder, exist_ok=True)

        with rasterio.open(os.path.join(INPUT, 'areamaps', 'mask.tif'), 'r') as src:
            self.mask = src.read(1)
        with rasterio.open(os.path.join(INPUT, 'areamaps', 'submask.tif'), 'r') as src:
            self.submask = src.read(1)
            self.submask_transform = src.profile['transform']
        
        with rasterio.open(os.path.join(INPUT, 'areamaps', 'sub_cell_area.tif'), 'r') as src_cell_area:
            self.cell_area = src_cell_area.read(1)

        vertical_index = np.arange(self.farms.shape[0]).repeat(self.farms.shape[1]).reshape(self.farms.shape)[self.farms != -1]
        horizontal_index = np.tile(np.arange(self.farms.shape[1]), self.farms.shape[0]).reshape(self.farms.shape)[self.farms != -1]
        self.pixels = np.zeros((self.n, 2), dtype=np.int32)
        self.pixels[:,0] = np.round(np.bincount(self.farms[self.farms != -1], horizontal_index) / np.bincount(self.farms[self.farms != -1])).astype(int)
        self.pixels[:,1] = np.round(np.bincount(self.farms[self.farms != -1], vertical_index) / np.bincount(self.farms[self.farms != -1])).astype(int)
        
        self.unmerged_HRU_indices = np.load(os.path.join(self.report_folder, 'unmerged_HRU_indices.npy'))
        self.scaling = np.load(os.path.join(self.report_folder, 'scaling.npy')).item()
        
        self.field_size = self.get_field_size()
        
        self.reservoir_dependent_farmers = self.set_reservoir_dependent_farmers(2011, 2018)
        self.non_surface_water_dependent_farmers = self.set_non_surface_water_dependent_farmers(2011, 2018)
        self.farmers_to_analyse = self.non_surface_water_dependent_farmers
        # self.farmers_to_analyse = ...
        self.command_areas = self.read_command_areas()
        self.activation_order = self.get_activation_order()

        self.command_area_per_farmer = self.command_areas[self.pixels[:, 1], self.pixels[:, 0]]
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
            np.save(cache_file, reservoir_dependent_farmers)
        else:
            reservoir_dependent_farmers = np.load(cache_file)
        return reservoir_dependent_farmers

    def set_non_surface_water_dependent_farmers(self, start_year, end_year):
        cache_file = os.path.join(self.cache_folder, f'non_surface_water_dependent_farmers_{start_year}_{end_year}.npy')
        if not os.path.exists(cache_file):
            # read irrigation data from 2011 to 2017 for from surface, groundwater and reservoir
            surface_irrigation_total = None
            years = 0
            for year in range(start_year, end_year):
                surface_irrigation = self.read_arrays(year, 'channel irrigation')
                if surface_irrigation_total is None:
                    surface_irrigation_total = surface_irrigation
                else:
                    surface_irrigation_total += surface_irrigation

                years += 1

            surface_irrigation_total_per_year = surface_irrigation_total / years
            surface_irrigation_total_m = surface_irrigation_total_per_year / self.field_size
            surface_irrigation_total_mm = surface_irrigation_total_m * 1000
            # surface_irrigaton_dependent_farmers
            non_surface_water_dependent_farmers = surface_irrigation_total_mm < .5
            
            np.save(cache_file, non_surface_water_dependent_farmers)
        else:
            non_surface_water_dependent_farmers = np.load(cache_file)
        return non_surface_water_dependent_farmers

    def read_command_areas(self):
        fp = os.path.join(INPUT, 'routing', 'lakesreservoirs', 'subcommand_areas.tif')
        with rasterio.open(fp, 'r') as src:
            command_areas = src.read(1)
        command_areas[self.submask == 1] = -1
        return command_areas

    def get_activation_order(self):
        if not os.path.exists(os.path.join(self.cache_folder, 'activation_order.npy')):
            activation_order = np.load(os.path.join(self.report_folder, 'activation_order.npy'))
            np.save(os.path.join(self.cache_folder, 'activation_order.npy'), activation_order)
        else:
            activation_order = np.load(os.path.join(self.cache_folder, 'activation_order.npy'))
        return activation_order

    def get_field_size(self):
        if not os.path.exists(os.path.join(self.cache_folder, 'field_size.npy')):
            is_field = np.where(self.farms != -1)
            field_size = np.bincount(self.farms[is_field], weights=self.cell_area[is_field])
            np.save(os.path.join(self.cache_folder, 'field_size.npy'), field_size)
        else:
            field_size = np.load(os.path.join(self.cache_folder, 'field_size.npy'))
        return field_size

    def read_npy(self, name, dt):
        dt -= datetime.timedelta(days=1)
        fn = os.path.join(self.report_folder, name, dt.isoformat().replace(':', '').replace('-', '') + '.npy')
        return np.load(fn)

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
            n = 0
            while day < datetime.date(year + 1, 1, 1):
                # read the data
                array = self.read_npy(name, day)
                if total is None:
                    total = array
                else:
                    total += array
                # go to next day
                day += datetime.timedelta(days=1)
                n += 1
            total /= n
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
            if correct_for_field_size:
                total /= self.field_size

            mapping = np.full(self.command_areas.max() + 1, 0, dtype=np.int32)
            command_area_ids = np.unique(self.command_areas)
            assert command_area_ids[0] == -1
            command_area_ids = command_area_ids[1:]
            mapping[command_area_ids] = np.arange(0, command_area_ids.size, dtype=np.int32)
            command_areas_mapped = mapping[self.command_areas]
            command_areas_mapped[self.command_areas == -1] = -1

            assert total.size == self.n
            
            by_command_area = {}
            for command_area_id in command_area_ids:
                command_area_id = command_area_id.item()
                farmers_in_command_area = self.command_area_per_farmer[self.farmers_to_analyse] == command_area_id
                activation_order_area = self.activation_order[self.farmers_to_analyse][farmers_in_command_area]
                activation_order_median = np.percentile(activation_order_area, 50)
                array_command_area = total[self.farmers_to_analyse][farmers_in_command_area]
                
                size_command_area = self.field_size[self.farmers_to_analyse][farmers_in_command_area]
                assert size_command_area.size == array_command_area.size

                head_end = fn(array_command_area[activation_order_area < activation_order_median])
                head_end_size = (size_command_area[activation_order_area < activation_order_median]).sum()
                tail_end = fn(array_command_area[activation_order_area >= activation_order_median])
                tail_end_size = (size_command_area[activation_order_area >= activation_order_median]).sum()
                by_command_area[command_area_id] = {
                    'head_end': head_end,
                    'head_end_size': head_end_size,
                    'tail_end': tail_end,
                    'tail_end_size': tail_end_size,
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
            if correct_for_field_size:
                total /= self.field_size
            field_size = self.get_field_size()
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

    def format_ax(self, ax, ylabel, ymax=None):
        ax.legend()
        # rotate all y tick labels and center them
        for tick in ax.get_yticklabels():
            tick.set_rotation(90)
            tick.set_verticalalignment('center')
        ax.set_xlabel('year')
        ax.set_ylabel(ylabel)
        if ymax is not None:
            ax.set_ylim(0, ymax)
        elif ylabel.startswith('farmers with'):
            ax.set_ylim(bottom=0, top=self.n / SUM_DIVIDER * 1.05)

    def plot_tail_vs_head_end(self, start_year, end_year, ax, *args, **kwargs):
        tail_end = []
        head_end = []
        years = []
        for year in range(start_year, end_year + 1):
            by_command_area = self.get_values_head_vs_tail(*args, year=year, **kwargs)
            year_values_tail_end = 0
            year_values_head_end = 0
            head_end_size = 0
            tail_end_size = 0
            for values in by_command_area.values():
                year_values_tail_end += values['tail_end'] * values['tail_end_size']
                year_values_head_end += values['head_end'] * values['head_end_size']
                head_end_size += values['head_end_size']
                tail_end_size += values['tail_end_size']
            tail_end.append(year_values_tail_end / tail_end_size)
            head_end.append(year_values_head_end / head_end_size)
            years.append(year)
        print(years)
        print(tail_end)
        ax.plot(years, tail_end, label='tail end')
        ax.plot(years, head_end, label='head end')

    def plot_small_vs_large_farmer(self, start_year, end_year, ax, *args, **kwargs):
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

    def create_new_plot(self, start_year, end_year):
        hydraulic_head, hydraulic_head_years = self.get_honeybees_data('hydraulic head', start_year=start_year, end_year=end_year)
        
        fig, ax = plt.subplots(2, 3, figsize=(20, 8))

        self.plot_tail_vs_head_end(start_year, end_year, ax[0][0], name="reservoir irrigation", fn=irrigation_fn, correct_for_field_size=True, mode='full_year')
        self.format_ax(ax[0][0], ylabel="reservoir irrigation (mm/day)", ymax=None)
        self.plot_tail_vs_head_end(start_year, end_year, ax[0][1], name="groundwater irrigation", fn=irrigation_fn, correct_for_field_size=True, mode='full_year')
        self.format_ax(ax[0][1], ylabel="groundwater irrigation (mm/day)", ymax=None)
        self.plot_tail_vs_head_end(start_year, end_year, ax[0][2], name="channel irrigation", fn=irrigation_fn, correct_for_field_size=True, mode='full_year')
        self.format_ax(ax[0][2], ylabel="channel irrigation (mm/day)", ymax=None)
        
        self.plot_tail_vs_head_end(start_year, end_year, ax[1][0], name="crops_kharif", fn=sum_sugarcane, correct_for_field_size=False, mode='first_day_of_year')
        self.plot_small_vs_large_farmer(start_year, end_year, ax[1][0], name="crops_kharif", fn=sum_sugarcane, correct_for_field_size=False, mode='first_day_of_year')
        self.format_ax(ax[1][0], ylabel=f"farmers with sugarcane (×{SUM_DIVIDER})")
        self.plot_tail_vs_head_end(start_year, end_year, ax[1][1], name="well_irrigated", fn=sum_most, correct_for_field_size=False, mode='first_day_of_year')
        self.plot_small_vs_large_farmer(start_year, end_year, ax[1][1], name="well_irrigated", fn=sum_most, correct_for_field_size=False, mode='first_day_of_year')
        self.format_ax(ax[1][1], ylabel=f"farmers with well (×{SUM_DIVIDER})")

        ax[1][2].plot(hydraulic_head_years, hydraulic_head, label="hydraulic head")
        self.format_ax(ax[1][2], ylabel="mean hydraulic head")

        # plt.show()
        plt.savefig('plot/output/newplot.png')


p = Plot('base')
START_YEAR = 2011
END_YEAR = 2018
p.create_new_plot(START_YEAR, END_YEAR)