
import yaml

from yaml import YAMLObject, Dumper
from typing import List
from reprlib import recursive_repr


class PlanetConfigFile(YAMLObject):
    yaml_tag = u'!PlanetConfig'
    hours_in_day: int
    days_in_year: int
    resolution: int
    planet_radius: float
    insolation: float
    gravity: float
    axial_tilt: float
    pressure_levels: List[float]

    def __init__(self,
                 hours_in_day: int,
                 days_in_year: int,
                 resolution: int,
                 planet_radius: float,
                 insolation: float,
                 gravity: float,
                 axial_tilt: float,
                 pressure_levels: List[float]):
        self.hours_in_day = hours_in_day
        self.days_in_year = days_in_year
        self.resolution = resolution
        self.planet_radius = planet_radius
        self.insolation = insolation
        self.gravity = gravity
        self.axial_tilt = axial_tilt
        self.pressure_levels = pressure_levels

    @recursive_repr
    def __repr__(self):
        return '<' + '|'.join(map(repr, self)) + '>'

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


class SmoothingConfigFile(YAMLObject):
    yaml_tag = u'!SmoothingConfig'
    smoothing: bool
    smoothing_parameter_t: float
    smoothing_parameter_u: float
    smoothing_parameter_v: float
    smoothing_parameter_w: float
    smoothing_parameter_add: float

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


class SaveConfigFile(YAMLObject):
    yaml_tag = u'!SaveConfig'
    save: bool
    load: bool
    save_frequency: int
    plot_frequency: int

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


class ViewConfigFile(YAMLObject):
    yaml_tag = u'!ViewConfig'

    above: bool
    pole: str
    above_level: int
    plot: bool
    diagnostic: bool
    level_plots: bool
    nplots: int
    top: int
    verbose: bool

    def __init__(self,
                 above: bool,
                 pole: str,
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


    @recursive_repr
    def __repr__(self):
        return '<' + '|'.join(map(repr, self)) + '>'

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


class ClaudeConfigFile(YAMLObject):
    yaml_tag = u'!ClaudeConfig'

    planet_config: PlanetConfigFile
    smoothing_config: SmoothingConfigFile
    save_config: SaveConfigFile
    view_config: ViewConfigFile

    def __init__(self,
                 planet_config: PlanetConfigFile,
                 smoothing_config: SmoothingConfigFile,
                 save_config: SaveConfigFile,
                 view_config: ViewConfigFile
                 ):
        self.planet_config = planet_config
        self.smoothing_config = smoothing_config
        self.view_config = view_config

    @recursive_repr
    def __repr__(self):
        return '<' + '|'.join(map(repr, self)) + '>'

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)
