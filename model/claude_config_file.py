
import yaml

from yaml import YAMLObject, Dumper
from typing import List
from reprlib import recursive_repr
from dataclasses import dataclass

@dataclass
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

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


class SaveConfigFile(YAMLObject):
    yaml_tag = u'!SaveConfig'
    save: bool
    load: bool
    save_frequency: int
    plot_frequency: int

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

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)


class ClaudeConfigFile(YAMLObject):
    yaml_tag = u'!ClaudeConfig'

    planet_config: PlanetConfigFile
    smoothing_config: SmoothingConfigFile
    save_config: SaveConfigFile
    view_config: ViewConfigFile

    def __str__(self):
        return yaml.dump(data=self, Dumper=Dumper)
