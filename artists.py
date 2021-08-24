# -*- coding: utf-8 -*-
from typing import Union
from hyve.artists import Artists as HyveArtists
import numpy as np
from operator import attrgetter
try:
    import cupy as cp
except ModuleNotFoundError:
    pass

class Artists(HyveArtists):
    """This class is used to configure how the display environment works.
    
    Args:
        model: The GEB model.
    """
    def __init__(self, model) -> None:
        HyveArtists.__init__(self, model)
        self.color = '#1386FF'
        self.min_colorbar_alpha = .4
        self.background_variable = "data.landunit.crop_map"  # set initial background iamge.
        self.custom_plot = self.get_custom_plot()
        self.set_variables()

    def draw_farmers(self, model, agents, idx: int) -> dict:
        """This function is used to draw farmers. First it is determined what crop is grown by the farmer, then the we get the color used to display that crop from the model configuration.

        Args:
            model: The GEB model.
            agents: The farmer class to plot.
            idx: The farmer index.

        Returns:
            portrayal: Portrayal of farmer.
        """
        if not hasattr(model, 'legend') and hasattr(model, 'landunit'):
            crops = model.data.landunit.crop_data['Crop']
            self.legend = {crop: color for crop, color in zip(crops, model.data.landunit.crop_data['Color'])}
        color = self.custom_plot['data.landunit.crop_map']['colors'][agents.crop[idx].item()]
        return {"type": "shape", "shape": "circle", "r": 1, "filled": True, "color": color}

    def draw_rivers(self) -> dict:
        """Returns portrayal of river.
        
        Returns:
            portrayal: portrayal of river.
        """
        return {"type": "shape", "shape": "line", "color": "Blue"}

    def get_custom_plot(self) -> dict[dict]:
        """Here you can specify custom options for plotting the background.
        
        Returns:
            custom_dict: Dictionary of dictionaries. The first level is the name of each of the variables, the second level the options for those variables.

        Example:
            .. code-block:: python

                {
                    'landunit.crop_map': {
                        'type': 'discrete',
                        'nanvalue': -1,
                        'names': ['crop name 1', 'crop name 2'],
                        'colors': ['#00FF00', '#FF0000']
                    }
                }

        """
        return {
            'data.landunit.crop_stage': {
                'type': 'discrete'
            },
            'data.landunit.crop_age': {
                'type': 'discrete'
            },
            'data.landunit.crop_map': {
                'type': 'categorical',
                'nanvalue': -1,
                'names': [self.model.config['draw']['crop_data'][i]['name'] for i in range(26)],
                'colors': [self.model.config['draw']['crop_data'][i]['color'] for i in range(26)],
            },
        }

    def set_variables(self) -> None:
        """This function is used to get a dictionary of variables that can be shown as background variable. The dictionary :code:`self.variables_dict` contains the name of each variable to display as key, and the actual variable as value.
        
        Checks are performed to see whether the data is the right size. Only compressed data can be shown. If a dataset has multiple dimensions, the dimensions can be shown seperately as `variable[0], variable[1], ...`.
        """
        self.variables_dict = {}

        def add_var(attr):
            attributes = attrgetter(attr)(self.model)
            compressed_size = attributes.compressed_size
            for varname, variable in vars(attributes).items():
                if isinstance(variable, (np.ndarray, cp.ndarray)):
                    if variable.ndim == 1 and variable.size == compressed_size:
                        name = f'{attr}.{varname}'
                        self.variables_dict[name] = variable
                    if variable.ndim == 2 and variable.shape[1] == compressed_size:
                        for i in range(variable.shape[0]):
                            name = f'{attr}.{varname}[{i}]'
                            self.variables_dict[name] = variable[i]
                    else:
                        continue
        
        add_var('data.grid')
        add_var('data.landunit')

    def get_background_variables(self) -> list:
        """This function gets a list of variables that can be used to show in the background.
        
        Returns:
            options: List of names for options to show in background.
        """
        self.set_variables()
        return list(self.variables_dict.keys())

    def set_background_variable(self, option_name: str) -> None:
        """This function is used to update the name of the variable to use for drawing the background of the map."""
        self.background_variable = option_name

    def get_background(self, minvalue: Union[float, int, None]=None, maxvalue: Union[float, int, None]=None, color: str='#1386FF') -> tuple[np.ndarray, dict]:
        """This function is called from the canvas class to draw the canvas background. The name of the variable to draw is stored in `self.background_variable`.
        
        Args:
            minvalue: The minimum value for the display scale.
            maxvalue: The maximum value for the display scale.
            color: The color to use to display the variable.

        Returns:
            background: RGBA-array to display as background.
            legend: Dictionary with data and formatting rules for background legend.
        """
        array = self.model.reporter.cwatmreporter.get_array(self.background_variable, decompress=True)
        mask = attrgetter('.'.join(self.background_variable.split('.')[:-1]))(self.model).mask

        if self.background_variable in self.custom_plot:
            options = self.custom_plot[self.background_variable]
        else:
            options = {}
        if 'type' not in options:
            if array.dtype in (np.float16, np.float32, np.float64):
                options['type'] = 'continuous'
            elif array.dtype in (np.bool, np.int8, np.int16, np.int32, np.int64):
                if np.unique(array).size < 30:
                    options['type'] = 'categorical'
                else:
                    print("Type for array might be categorical, but more than 30 categories were found, so rendering as continous.")
                    options['type'] = 'continuous'
                    array = array.astype(np.float64)
            else:
                raise ValueError
        
        if not maxvalue:
            maxvalue = np.nanmax(array[~mask]).item()
        if not minvalue:
            minvalue = np.nanmin(array[~mask]).item()
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
            background[:, :, 0][np.isnan(array)] = 200
            background[:, :, 1][np.isnan(array)] = 200
            background[:, :, 2][np.isnan(array)] = 200
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
            if 'nanvalue' in options:
                nanvalue = options['nanvalue']
            else:
                nanvalue = None
            if options['type'] == 'categorical':
                unique_values = np.unique(array)
                if nanvalue is not None:
                    unique_values = unique_values[unique_values != nanvalue]
                unique_values = unique_values.tolist()
                if 'colors' in options:
                    colors = np.array(options['colors'])[np.array(unique_values)].tolist()
                    colors = [self.hex_to_rgb(color) for color in colors]
                else:
                    colors = self.generate_distinct_colors(len(unique_values), mode='rgb')
                if 'names' in options:
                    names = np.array(options['names'])[np.array(unique_values)].tolist()
                else:
                    names = unique_values
                channels = (0, 1, 2)
                background[:,:,3][array != nanvalue] = 255
            elif options['type'] == 'discrete':
                unique_values = np.arange(array[array != nanvalue].min(), array[array != nanvalue].max()+1, 1).tolist()
                if 'names' in options:
                    names = options['names']
                else:
                    names = unique_values
                colors = self.generate_discrete_colors(len(unique_values), self.hex_to_rgb(color), mode='rgb', min_alpha=0.4)
                channels = (0, 1, 2, 3)
            else:
                raise ValueError
            
            for channel in channels:
                channel_colors = np.array([color[channel] * 255 for color in colors])
                color_array_size = unique_values[-1] + 1
                if unique_values[0] < 0:
                    color_array_size += abs(unique_values[0])
                color_array = np.zeros(color_array_size, dtype=np.float32)
                color_array[unique_values] = channel_colors
                background[:, :, channel] = color_array[array]
            
            legend = {
                'type': 'legend',
                'labels': {
                    name: self.rgb_to_hex(colors[i])
                    for i, name in enumerate(names)
                }
            }

        background[mask] = 200

        return background, legend