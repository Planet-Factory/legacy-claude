import numpy as np
import yaml

from yaml import Loader, Dumper
from typing import List
from model.pole_enum import PoleType
from dataclasses import dataclass
from model.claude_config_file import PlanetConfigFile, SaveConfigFile, ViewConfigFile, SmoothingConfigFile, ClaudeConfigFile


@dataclass
class PlanetConfig:
    # define length of day (used for calculating Coriolis as well) (s)
    day: int
    year: int  # length of year (s)
    resolution: int  # how many degrees between latitude and longitude gridpoints
    planet_radius: float  # define the planet's radius (m)
    insolation: float  # TOA radiation from star (W m^-2)
    gravity: float  # define surface gravity for planet (m s^-2)
    axial_tilt: float  # tilt of rotational axis w.r.t. solar plane
    pressure_levels: np.ndarray
    nlevels: int

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)

    def __eq__(self, other):
        result = False
        if (isinstance(other, PlanetConfig)):
            day = self.day == other.day
            year = self.year == other.year
            resolution = self.resolution == other.resolution
            radius = self.planet_radius == other.planet_radius
            insolation = self.insolation == other.insolation
            gravity = self.gravity == other.gravity
            axial_tilt = self.axial_tilt == other.axial_tilt
            pressure_levels = np.array_equal(
                self.pressure_levels, other.pressure_levels)
            nlevels = self.nlevels == other.nlevels
            result = day and year and resolution and radius and insolation and gravity and axial_tilt and pressure_levels and nlevels
        return result

    @staticmethod
    def load_from_file(planet_config_file: PlanetConfigFile):
        day = 60 * 60 * planet_config_file.hours_in_day
        year = day * planet_config_file.days_in_year
        pressure_levels = np.array(planet_config_file.pressure_levels)*100
        nlevels = len(planet_config_file.pressure_levels)
        return PlanetConfig(
            day=day,
            year=year,
            nlevels=nlevels,
            resolution=planet_config_file.resolution,
            planet_radius=planet_config_file.planet_radius,
            insolation=planet_config_file.insolation,
            gravity=planet_config_file.gravity,
            axial_tilt=planet_config_file.axial_tilt,
            pressure_levels=pressure_levels
        )


@dataclass
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

    @staticmethod
    def load_from_file(smoothing_config_file: SmoothingConfigFile):
        return SmoothingConfig(
            smoothing=smoothing_config_file.smoothing,
            smoothing_parameter_t=smoothing_config_file.smoothing_parameter_t,
            smoothing_parameter_u=smoothing_config_file.smoothing_parameter_u,
            smoothing_parameter_v=smoothing_config_file.smoothing_parameter_v,
            smoothing_parameter_w=smoothing_config_file.smoothing_parameter_w,
            smoothing_parameter_add=smoothing_config_file.smoothing_parameter_add
        )

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


@dataclass
class SaveConfig:
    save: bool  # save current state to file?
    load: bool  # load initial state from file?
    save_frequency: int  # write to file after this many timesteps have passed
    # how many timesteps between plots (set this low if you want realtime plots, set this high to improve performance)
    plot_frequency: int

    @staticmethod
    def load_from_file(save_config_file: SaveConfigFile):
        return SaveConfig(
            save=save_config_file.save,
            load=save_config_file.load,
            save_frequency=save_config_file.save_frequency,
            plot_frequency=save_config_file.plot_frequency
        )

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


@dataclass
class ViewConfig:
    # display top down view of a pole? showing polar plane data and regular gridded data
    above: bool
    # which pole to display - 'n' for north, 's' for south
    pole: PoleType
    # which vertical level to display over the pole
    above_level: int
    plot: bool  # display plots of output?
    diagnostic: bool  # display raw fields for diagnostic purposes
    level_plots: bool  # display plots of output on vertical levels?
    # how many levels you want to see plots of (evenly distributed through column)
    nplots: int
    top: int  # top pressure level to display (i.e. trim off sponge layer)
    verbose: bool  # print times taken to calculate specific processes each timestep

    @staticmethod
    def load_from_file(view_config_file: ViewConfigFile):
        return ViewConfig(
            above=view_config_file.above,
            pole=PoleType(view_config_file.pole),
            above_level=view_config_file.above_level,
            plot=view_config_file.plot,
            diagnostic=view_config_file.diagnostic,
            level_plots=view_config_file.level_plots,
            nplots=view_config_file.nplots,
            top=view_config_file.top,
            verbose=view_config_file.verbose
        )


@dataclass
class CoordinateGrid:
    lat: np.ndarray
    lon: np.ndarray
    nlat: int
    nlon: int
    lon_plot: np.ndarray
    lat_plot: np.ndarray
    heights_plot: np.ndarray
    lat_z_plot: np.ndarray
    temperature_world: np.ndarray

    @staticmethod
    def load_from_file(
        resolution: int,
        top: int,
        pressure_levels: np.ndarray
    ):
        lat = np.arange(-90, 91, resolution)
        lon = np.arange(0, 360, resolution)
        nlat = len(lat)
        nlon = len(lon)
        lon_plot, lat_plot = np.meshgrid(lon, lat)
        heights_plot, lat_z_plot = np.meshgrid(
            lat, pressure_levels[:top]/100)
        temperature_world = np.zeros((nlat, nlon))
        return CoordinateGrid(
            lat=lat,
            lon=lon,
            nlat=nlat,
            nlon=nlon,
            lon_plot=lon_plot,
            lat_plot=lat_plot,
            heights_plot=heights_plot,
            lat_z_plot=lat_z_plot,
            temperature_world=temperature_world
        )

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)

    def __eq__(self, other):
        result = False
        if (isinstance(other, CoordinateGrid)):
            lat = np.array_equal(self.lat, other.lat)
            lon = np.array_equal(self.lon, other.lon)
            nlat = self.nlat == other.nlat
            nlon = self.nlon == other.nlon
            lon_plot = np.array_equal(self.lon_plot, other.lon_plot)
            lat_plot = np.array_equal(self.lat_plot, other.lat_plot)
            heights_plot = np.array_equal(
                self.heights_plot, other.heights_plot)
            lat_z_plot = np.array_equal(self.lat_z_plot, other.lat_z_plot)
            temperature_world = np.array_equal(
                self.temperature_world, other.temperature_world)
            result = lat and lon and nlat and nlon and lon_plot and lat_plot and heights_plot and lat_z_plot and temperature_world
        return result


@dataclass
class ClaudeConfig:

    planet_config: PlanetConfig
    smoothing_config: SmoothingConfig
    save_config: SaveConfig
    view_config: ViewConfig
    coordinate_grid: CoordinateGrid

    @staticmethod
    def load_from_file(claude_config_file: ClaudeConfigFile):
        planet_config = PlanetConfig.load_from_file(
            claude_config_file.planet_config)
        smoothing_config = SmoothingConfig.load_from_file(
            claude_config_file.smoothing_config)
        save_config = SaveConfig.load_from_file(claude_config_file.save_config)
        view_config = ViewConfig.load_from_file(claude_config_file.view_config)
        coordinate_grid = CoordinateGrid.load_from_file(
            resolution=planet_config.resolution, top=view_config.top, pressure_levels=planet_config.pressure_levels)
        return ClaudeConfig(
            planet_config=planet_config,
            smoothing_config=smoothing_config,
            save_config=save_config,
            view_config=view_config,
            coordinate_grid=coordinate_grid
        )

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)
