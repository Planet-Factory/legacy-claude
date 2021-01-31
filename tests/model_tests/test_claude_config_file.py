import unittest, os, yaml

from yaml import Loader
from definitions import CONFIG_PATH
from model.claude_config_file import PlanetConfigFile

class TestPlanetConfigFile(unittest.TestCase):

    def test_yaml_mapping(self):
        #Arrange (set up test)
        test_planet_config = open(os.path.join(os.path.dirname(__file__), "test_files", "TestPlanetConfig.yaml"))

        expected_planet_config = PlanetConfigFile(
            hours_in_day = 24,
            days_in_year = 365,
            resolution = 3,
            planet_radius = 6.4E6,
            insolation = 1370,
            gravity = 9.81,
            axial_tilt = 23.5,
            pressure_levels = [1000,950,900,800]
        )

        # Act (Perform Action to be Tested)
        result_planet_config = yaml.load(test_planet_config, Loader=Loader)

        # Assert (Check our results)
        self.assertEqual(result_planet_config.hours_in_day, expected_planet_config.hours_in_day)
        self.assertEqual(result_planet_config.days_in_year, expected_planet_config.days_in_year)
        self.assertEqual(result_planet_config.resolution, expected_planet_config.resolution)
        self.assertEqual(result_planet_config.insolation, expected_planet_config.insolation)
        self.assertEqual(result_planet_config.gravity, expected_planet_config.gravity)
        self.assertEqual(result_planet_config.axial_tilt, expected_planet_config.axial_tilt)
        self.assertEqual(result_planet_config.pressure_levels, expected_planet_config.pressure_levels)



