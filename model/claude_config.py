import numpy as np
import yaml

from yaml import Loader, Dumper
from typing import List
from model.pole_enum import PoleType
from dataclasses import dataclass
from model.claude_config_file import *


@dataclass(init=False)
class PlanetConfig:
    day: int
    year: int
    resolution: int
    planet_radius: float
    insolation: float
    gravity: float
    axial_tilt: float
    pressure_levels: List[float]
    nlevels: int

    def __init__(self,
                 planet_config_file: PlanetConfigFile
                 ):
        # define length of day (used for calculating Coriolis as well) (s)
        self.day = 60 * 60 * planet_config_file.hours_in_day
        # length of year (s)
        self.year = self.day * planet_config_file.days_in_year
        # how many degrees between latitude and longitude gridpoints
        self.resolution = planet_config_file.resolution
        # define the planet's radius (m)
        self.planet_radius = planet_config_file.planet_radius
        # TOA radiation from star (W m^-2)
        self.insolation = planet_config_file.insolation
        # define surface gravity for planet (m s^-2)
        self.gravity = planet_config_file.gravity
        # tilt of rotational axis w.r.t. solar plane
        self.axial_tilt = planet_config_file.axial_tilt
        self.pressure_levels = planet_config_file.pressure_levels
        self.nlevels = len(self.pressure_levels)

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)



@dataclass(init=False)
class SmoothingConfig:
    """
    you probably won't need this, but there is the option to smooth out fields
    using FFTs (NB this slows down computation considerably and can introduce
    nonphysical errors)
    """
    smoothing: bool
    smoothing_parameter_t: float
    smoothing_parameter_u: float
    smoothing_parameter_v: float
    smoothing_parameter_w: float
    smoothing_parameter_add: float

    def __init__(self,
                 smoothing_config_file: SmoothingConfigFile
                 ):
        self.smoothing = smoothing_config_file.smoothing
        self.smoothing_parameter_t = smoothing_config_file.smoothing_parameter_t
        self.smoothing_parameter_u = smoothing_config_file.smoothing_parameter_u
        self.smoothing_parameter_v = smoothing_config_file.smoothing_parameter_v
        self.smoothing_parameter_w = smoothing_config_file.smoothing_parameter_w
        self.smoothing_parameter_add = smoothing_config_file.smoothing_parameter_add

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


@dataclass(init=False)
class SaveConfig:
    save: bool
    load: bool
    save_frequency: int
    plot_frequency: int

    def __init__(self,
                 save_config_file: SaveConfigFile
                 ):
        self.save = save_config_file.save  # save current state to file?
        self.load = save_config_file.load  # load initial state from file?
        # write to file after this many timesteps have passed
        self.save_frequency = save_config_file.save_frequency
        # how many timesteps between plots (set this low if you want realtime plots, set this high to improve performance)
        self.plot_frequency = save_config_file.plot_frequency

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


@dataclass(init=False)
class ViewConfig:

    above: bool
    pole: PoleType
    above_level: int
    plot: bool
    diagnostic: bool
    level_plots: bool
    nplots: int
    top: int
    verbose: bool
    
    def __init__(self,
                 view_config_file: ViewConfigFile
                 ):
        # display top down view of a pole? showing polar plane data and regular gridded data
        self.above = view_config_file.above
        self.pole = PoleType(view_config_file.pole)  # which pole to display - 'n' for north, 's' for south
        self.above_level = view_config_file.above_level  # which vertical level to display over the pole
        self.plot = view_config_file.plot  # display plots of output?
        self.diagnostic = view_config_file.diagnostic  # display raw fields for diagnostic purposes
        self.level_plots = view_config_file.level_plots  # display plots of output on vertical levels?
        # how many levels you want to see plots of (evenly distributed through column)
        self.nplots = view_config_file.nplots
        # top pressure level to display (i.e. trim off sponge layer)
        self.top = view_config_file.top
        self.verbose = view_config_file.verbose  # print times taken to calculate specific processes each timestep


@dataclass(init=False)
class CoordinateGrid:
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

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


@dataclass(init=False)
class ClaudeConfig:
    def __init__(self,
                 claude_config_file: ClaudeConfigFile
                 ):
        self.planet_config = claude_config_file.planet_config
        self.smoothing_config = claude_config_file.smoothing_config
        self.view_config = claude_config_file.view_config
        self.coordinate_grid = CoordinateGrid(
            resolution=self.planet_config.resolution, top=self.view_config.top, pressure_levels=self.planet_config.pressure_levels)

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)
