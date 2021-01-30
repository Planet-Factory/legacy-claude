import numpy as np
import yaml

from model.pole_enum import PoleEnum

class PlanetConfig:
    def __init__(self,
        day: int,
        year: int,
        resolution: int,
        planet_radius: float,
        insolation: float,
        gravity: float,
        axial_tilt: float,
        pressure_levels: list(float)
    ):
        self.day = day      # define length of day (used for calculating Coriolis as well) (s)
        self.year = day * year           # length of year (s)
        self.resolution = resolution     # how many degrees between latitude and longitude gridpoints
        self.planet_radius = planet_radius  # define the planet's radius (m)
        self.insolation = insolation     # TOA radiation from star (W m^-2)
        self.gravity = gravity        # define surface gravity for planet (m s^-2)
        self.axial_tilt = axial_tilt      # tilt of rotational axis w.r.t. solar plane
        self.pressure_levels = pressure_levels
        self.nlevels = len(self.pressure_levels)

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

class SaveConfig:
    def __init__(self,
        save: bool,
        load: bool,
        save_frequency: int,
        plot_frequency: int
    ): 
        self.save = save # save current state to file?
        self.load = load # load initial state from file?
        self.save_frequency = save_frequency # write to file after this many timesteps have passed
        self.plot_frequency = plot_frequency # how many timesteps between plots (set this low if you want realtime plots, set this high to improve performance)

class ViewConfig:
    def __init__(self,
        above: bool,
        pole: PoleEnum,
        above_level: int,
        plot: bool,
        diagnostic: bool,
        level_plots: bool,
        nplots: int,
        top: int,
        verbose: bool
    ):
        self.above = above # display top down view of a pole? showing polar plane data and regular gridded data
        self.pole = pole # which pole to display - 'n' for north, 's' for south
        self.above_level = above_level # which vertical level to display over the pole
        self.plot = plot # display plots of output?
        self.diagnostic = diagnostic # display raw fields for diagnostic purposes
        self.level_plots = level_plots # display plots of output on vertical levels?
        self.nplots = nplots # how many levels you want to see plots of (evenly distributed through column)
        self.top = top # top pressure level to display (i.e. trim off sponge layer)
        self.verbose = verbose # print times taken to calculate specific processes each timestep

class CoordinateGrids:
    def __init__(self,
        resolution: int,
        top: int
    ):
        self.lat = np.arange(-90,91,resolution)
        self.lon = np.arange(0,360,resolution)
        self.nlat = len(self.lat)
        self.nlon = len(self.lon)
        self.lon_plot, self.lat_plot = np.meshgrid(self.lon, self.lat)
        self.heights_plot, self.lat_z_plot = np.meshgrid(self.lat, self.pressure_levels[:top]/100)
        self.temperature_world = np.zeros((self.nlat, self.nlon))

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
        self.coordinate_grid = CoordinateGrids(resolution=self.planet_config.resolution, top=self.planet_config.top)

    @staticmethod
    def load_from_yaml(data):
        values = yaml.safe_load(data)
        planet_config = PlanetConfig(data["planet_config"])
        smoothing_config = SmoothingConfig(data["smoothing_config"])
        save_config = SaveConfig(data["save_config"])
        view_config = ViewConfig(data["view_config"])
        return ClaudeConfig(planet_config=planet_config,
            smoothing_config=smoothing_config,
            save_config=save_config,
            view_config=view_config)
