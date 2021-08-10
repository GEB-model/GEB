from config import VARIABLE_COLOR, MIN_COLOR_BAR_ALPHA
from config import LEGEND
from hyve.artists import BaseArtist
import colorsys
import numpy as np
import pandas as pd
try:
    import cupy as cp
except ModuleNotFoundError:
    pass
from cwatm.management_modules.data_handling import decompress

class Artists(BaseArtist):
    def __init__(self, model):
        BaseArtist.__init__(self, model)
        self.color = VARIABLE_COLOR
        self.min_colorbar_alpha = MIN_COLOR_BAR_ALPHA
        self.background_variable = "subvar.crop_map"

    def draw_farmers(self, model, idx):
        if not hasattr(self.model, 'legend') and hasattr(self.model, 'subvar'):
            crops = self.model.data.subvar.crop_data['Crop']
            self.legend = {crop: color for crop, color in zip(crops, self.model.data.subvar.crop_data['Color'])}
        color = self.model.data.subvar.crop_data['Color'][self.model.agents.farmers.crop[idx].get()]
        return {"type": "shape", "shape": "circle", "r": 1, "filled": True, "color": color}

    def draw_rivers(self):
        return {"type": "shape", "shape": "line", "color": "Blue"}

    @property
    def custom_plot(self):
        return {
            'subvar.crop_stage': {
                'type': 'discrete'
            },
            'subvar.crop_age': {
                'type': 'discrete'
            },
            'subvar.crop_map': {
                'type': 'categorical',
                # 'names': self.model.data.subvar.crop_data['Crop'],
                # 'colors': [self.hex_to_rgb(color) for color in self.model.data.subvar.crop_data['Color']]
            },
        }

    def get_background_options(self):
        return list(self.model.reporter.cwatmreporter.variables_dict.keys())

    def get_background(self, minvalue=None, maxvalue=None, nanvalue=-1, color='#1386FF'):
        name = self.background_variable

        array = self.model.reporter.cwatmreporter.get_array(name, decompress=True)
        if name in self.custom_plot:
            options = self.custom_plot[name]
        else:
            options = {}
        if 'type' not in options:
            if array.dtype in (np.float16, np.float32, np.float64):
                options['type'] = 'continuous'
            elif array.dtype in (np.bool, np.int8, np.int16, np.int32, np.int64):
                options['type'] = 'categorical'
            else:
                raise ValueError
        
        if array.dtype in (np.float32, np.float64):
            if nanvalue:
                array[array == nanvalue] = np.nan
        
        if not maxvalue:
            maxvalue = np.nanmax(array).item()
        if not minvalue:
            minvalue = np.nanmin(array).item()
        if np.isnan(maxvalue):  # minvalue must be nan as well
            minvalue, maxvalue = 0, 0
        
        background = np.zeros((*array.shape, 4), dtype=np.uint8)
        if options['type'] == 'continuous':
            array -= minvalue
            if maxvalue - minvalue != 0:
                array *= 255 / (maxvalue - minvalue)
            else:
                array *= 0
            array[array < 0] = 0
            array[array > 255] = 255
            rgb = self.hex_to_rgb(color)
            for channel in (0, 1, 2):
                background[:, :, channel][~np.isnan(array)] = rgb[channel] * 255
            background[:, :, 3] = array
            background[:, :, 0][np.isnan(array)] = 255
            background[:, :, 1][np.isnan(array)] = 0
            background[:, :, 2][np.isnan(array)] = 0
            background[:, :, 3][np.isnan(array)] = 255
            legend = {
                'type': 'colorbar',
                'color': color,
                'min': self.round_to_n_significant_digits(minvalue, 3),
                'max': self.round_to_n_significant_digits(maxvalue, 3),
                'min_colorbar_alpha': 0,
                'unit': ''
            }
        else:
            if options['type'] == 'categorical':
                if 'colors' in options:
                    colors = options['colors']
                    values = list(range(len(colors)))
                else:
                    values = array[array != nanvalue]
                    values = np.unique(values).tolist()
                    colors = self.generate_distinct_colors(len(values), mode='rgb')
                if 'names' in options:
                    names = options['names']
                else:
                    names = values
                channels = (0, 1, 2)
                background[:,:,3][array != nanvalue] = 255
            elif options['type'] == 'discrete':
                values = np.arange(array[array != nanvalue].min(), array[array != nanvalue].max()+1, 1).tolist()
                names = values
                colors = self.generate_discrete_colors(len(values), self.hex_to_rgb(color), mode='rgb', min_alpha=0.4)
                channels = (0, 1, 2, 3)
            else:
                raise ValueError
            
            for i, value in enumerate(values):
                for channel in channels:
                    background[:, :, channel][array == value] = colors[i][channel] * 255
            
            legend = {
                'type': 'legend',
                'labels': {
                    name: self.rgb_to_hex(colors[i])
                    for i, name in enumerate(names)
                }
            }

        return background, legend