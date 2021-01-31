import numpy as np
import yaml

from yaml import Loader, Dumper
from typing import List
from model.pole_enum import PoleType
from reprlib import recursive_repr

class PlanetConfig:
    def __init__(self,
                 hours_in_day: int,
                 days_in_year: int,
                 resolution: int,
                 planet_radius: float,
                 insolation: float,
                 gravity: float,
                 axial_tilt: float,
                 pressure_levels: List[float]
                 ):
        # define length of day (used for calculating Coriolis as well) (s)
        self.day = 60 * 60 * hours_in_day
        self.year = self.day * days_in_year           # length of year (s)
        # how many degrees between latitude and longitude gridpoints
        self.resolution = resolution
        self.planet_radius = planet_radius  # define the planet's radius (m)
        self.insolation = insolation     # TOA radiation from star (W m^-2)
        # define surface gravity for planet (m s^-2)
        self.gravity = gravity
        self.axial_tilt = axial_tilt      # tilt of rotational axis w.r.t. solar plane
        self.pressure_levels = pressure_levels
        self.nlevels = len(self.pressure_levels)

    @recursive_repr
    def __repr__(self):
        return '<' + '|'.join(map(repr, self)) + '>'()
    
    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)

"""
you probably won't need this, but there is the option to smooth out fields
using FFTs (NB this slows down computation considerably and can introduce
nonphysical errors)
"""
class SmoothingConfig:
    def __init__(self,
                 smoothing: bool,
                 smoothing_parameter_t: float,
                 smoothing_parameter_u: float,
                 smoothing_parameter_v: float,
                 smoothing_parameter_w: float,
                 smoothing_parameter_add: float
                 ):
        self.smoothing = smoothing
        self.smoothing_parameter_t = smoothing_parameter_t
        self.smoothing_parameter_u = smoothing_parameter_u
        self.smoothing_parameter_v = smoothing_parameter_v
        self.smoothing_parameter_w = smoothing_parameter_w
        self.smoothing_parameter_add = smoothing_parameter_add
    
    @recursive_repr
    def __repr__(self):
        return '<' + '|'.join(map(repr, self)) + '>'
    
    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


class SaveConfig:
    def __init__(self,
                 save: bool,
                 load: bool,
                 save_frequency: int,
                 plot_frequency: int
                 ):
        self.save = save  # save current state to file?
        self.load = load  # load initial state from file?
        # write to file after this many timesteps have passed
        self.save_frequency = save_frequency
        # how many timesteps between plots (set this low if you want realtime plots, set this high to improve performance)
        self.plot_frequency = plot_frequency
    
    @recursive_repr
    def __repr__(self):
        return '<' + '|'.join(map(repr, self)) + '>'
    
    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


class ViewConfig:
    def __init__(self,
                 above: bool,
                 pole: PoleType,
                 above_level: int,
                 plot: bool,
                 diagnostic: bool,
                 level_plots: bool,
                 nplots: int,
                 top: int,
                 verbose: bool
                 ):
        # display top down view of a pole? showing polar plane data and regular gridded data
        self.above = above
        self.pole = pole  # which pole to display - 'n' for north, 's' for south
        self.above_level = above_level  # which vertical level to display over the pole
        self.plot = plot  # display plots of output?
        self.diagnostic = diagnostic  # display raw fields for diagnostic purposes
        self.level_plots = level_plots  # display plots of output on vertical levels?
        # how many levels you want to see plots of (evenly distributed through column)
        self.nplots = nplots
        # top pressure level to display (i.e. trim off sponge layer)
        self.top = top
        self.verbose = verbose  # print times taken to calculate specific processes each timestep


class CoordinateGrids:
    def __init__(self,
                 resolution: int,
                 top: int,
                 pressure_levels: List[float]
                 ):
        self.lat = np.arange(-90, 91, resolution)
        self.lon = np.arange(0, 360, resolution)
        self.nlat = len(self.lat)
        self.nlon = len(self.lon)
        self.lon_plot, self.lat_plot = np.meshgrid(self.lon, self.lat)
        self.heights_plot, self.lat_z_plot = np.meshgrid(
            self.lat, [x/100 for x in pressure_levels[:top]])
        self.temperature_world = np.zeros((self.nlat, self.nlon))
    
    @recursive_repr
    def __repr__(self):
        return '<' + '|'.join(map(repr, self)) + '>'
    
    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


class ClaudeConfig:
    def __init__(self,
                 planet_config: PlanetConfig,
                 smoothing_config: SmoothingConfig,
                 save_config: SaveConfig,
                 view_config: ViewConfig
                 ):
        self.planet_config = planet_config
        self.smoothing_config = smoothing_config
        self.view_config = view_config
        self.coordinate_grid = CoordinateGrids(
            resolution=self.planet_config.resolution, top=self.view_config.top, pressure_levels=self.planet_config.pressure_levels)

    @recursive_repr
    def __repr__(self):
        return '<' + '|'.join(map(repr, self)) + '>'
    
    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)

    @staticmethod
    def load_from_yaml(data):
        values = yaml.safe_load(data)
        planet_config = PlanetConfig(hours_in_day=values["planet_config"]["hours_in_day"],
                                     days_in_year=values["planet_config"]["days_in_year"],
                                     resolution=values["planet_config"]["resolution"],
                                     planet_radius=values["planet_config"]["planet_radius"],
                                     insolation=values["planet_config"]["insolation"],
                                     gravity=values["planet_config"]["gravity"],
                                     axial_tilt=values["planet_config"]["axial_tilt"],
                                     pressure_levels=values["planet_config"]["pressure_levels"],
                                     )

        smoothing_config = SmoothingConfig(smoothing=values["smoothing_config"]["smoothing"],
                                           smoothing_parameter_t=values["smoothing_config"].get(
                                               "smoothing_parameter_t", None),
                                           smoothing_parameter_u=values["smoothing_config"].get(
                                               "smoothing_parameter_u", None),
                                           smoothing_parameter_v=values["smoothing_config"].get(
                                                "smoothing_parameter_v", None),
                                           smoothing_parameter_w=values["smoothing_config"].get(
                                                "smoothing_parameter_w", None),
                                           smoothing_parameter_add=values["smoothing_config"].get(
                                               "smoothing_parameter_add", None)
                                            )

        save_config = SaveConfig(save=values["save_config"]["save"],
                                 load=values["save_config"]["load"],
                                 save_frequency=values["save_config"]["save_frequency"],
                                 plot_frequency=values["save_config"]["plot_frequency"]
                                 )

        view_config = ViewConfig(above=values["view_config"]["above"],
                                 pole=PoleType(values["view_config"]["pole"]),
                                 above_level=values["view_config"]["above_level"],
                                 plot=values["view_config"]["plot"],
                                 diagnostic=values["view_config"]["diagnostic"],
                                 level_plots=values["view_config"]["level_plots"],
                                 nplots=values["view_config"]["nplots"],
                                 top=values["view_config"]["top"],
                                 verbose=values["view_config"]["verbose"])

        return ClaudeConfig(planet_config=planet_config,
                            smoothing_config=smoothing_config,
                            save_config=save_config,
                            view_config=view_config)
