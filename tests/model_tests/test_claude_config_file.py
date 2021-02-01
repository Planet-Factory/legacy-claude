import unittest
import os
import yaml

from yaml import Loader
from definitions import CONFIG_PATH
from model.claude_config_file import PlanetConfigFile, SmoothingConfigFile, SaveConfigFile, ViewConfigFile, ClaudeConfigFile


class TestPlanetConfigFile(unittest.TestCase):
    def test_yaml_mapping(self):
        # Arrange (set up test)
        test_planet_config = open(os.path.join(os.path.dirname(
            __file__), "test_files", "TestPlanetConfig.yaml"))

        expected_planet_config = PlanetConfigFile(
            hours_in_day=24,
            days_in_year=365,
            resolution=3,
            planet_radius=6.4E6,
            insolation=1370,
            gravity=9.81,
            axial_tilt=23.5,
            pressure_levels=[1000, 950, 900, 800]
        )

        # Act (Perform Action to be Tested)
        result_planet_config = yaml.load(test_planet_config, Loader=Loader)

        # Assert (Check our results)
        self.assertEqual(result_planet_config, expected_planet_config)


class TestSmoothingConfigFile(unittest.TestCase):
    def test_yaml_mapping(self):
        # Arrange (set up test)
        test_smoothing_config = open(os.path.join(os.path.dirname(
            __file__), "test_files", "TestSmoothingConfig.yaml"))

        expected_smoothing_config = SmoothingConfigFile(
            smoothing=False,
            smoothing_parameter_t=1.0,
            smoothing_parameter_u=0.9,
            smoothing_parameter_v=0.9,
            smoothing_parameter_w=0.3,
            smoothing_parameter_add=0.3
        )

        # Act (Perform Action to be Tested)
        result_smoothing_config = yaml.load(
            test_smoothing_config, Loader=Loader)

        # Assert (Check our results)
        self.assertEqual(result_smoothing_config, expected_smoothing_config)


class TestSaveConfigFile(unittest.TestCase):
    def test_yaml_mapping(self):
        # Arrange (set up test)
        test_save_config = open(os.path.join(os.path.dirname(
            __file__), "test_files", "TestSaveConfig.yaml"))

        expected_save_config = SaveConfigFile(
            save=True,
            load=True,
            save_frequency=100,
            plot_frequency=5
        )

        # Act (Perform Action to be Tested)
        result_save_config = yaml.load(
            test_save_config, Loader=Loader)

        # Assert (Check our results)
        self.assertEqual(expected_save_config, result_save_config)


class TestViewConfigFile(unittest.TestCase):
    def test_yaml_mapping(self):
        # Arrange (set up test)
        test_view_config = open(os.path.join(os.path.dirname(
            __file__), "test_files", "TestViewConfig.yaml"))

        expected_view_config = ViewConfigFile(
            above=False,
            pole='n',
            above_level=16,
            plot=True,
            diagnostic=False,
            level_plots=False,
            nplots=3,
            top=17,
            verbose=False
        )

        # Act (Perform Action to be Tested)
        result_view_config = yaml.load(
            test_view_config, Loader=Loader)

        # Assert (Check our results)
        self.assertEqual(expected_view_config, result_view_config)


class TestClaudeConfigFile(unittest.TestCase):
    def test_yaml_mapping(self):
        # Arrange (set up test)
        test_claude_config = open(os.path.join(os.path.dirname(
            __file__), "test_files", "TestClaudeConfig.yaml"))

        expected_planet_config = PlanetConfigFile(
            hours_in_day=24,
            days_in_year=365,
            resolution=3,
            planet_radius=6.4E6,
            insolation=1370,
            gravity=9.81,
            axial_tilt=23.5,
            pressure_levels=[1000, 950, 900, 800]
        )
        expected_smoothing_config = SmoothingConfigFile(
            smoothing=False,
            smoothing_parameter_t=1.0,
            smoothing_parameter_u=0.9,
            smoothing_parameter_v=0.9,
            smoothing_parameter_w=0.3,
            smoothing_parameter_add=0.3
        )
        expected_save_config = SaveConfigFile(
            save=True,
            load=True,
            save_frequency=100,
            plot_frequency=5
        )
        expected_view_config = ViewConfigFile(
            above=False,
            pole='n',
            above_level=16,
            plot=True,
            diagnostic=False,
            level_plots=False,
            nplots=3,
            top=17,
            verbose=False
        )
        expected_claude_config = ClaudeConfigFile(
            planet_config=expected_planet_config,
            smoothing_config=expected_smoothing_config,
            save_config=expected_save_config,
            view_config=expected_view_config
        )

        # Act (Perform Action to be Tested)
        result_claude_config = yaml.load(
            test_claude_config, Loader=Loader)

        # Assert (Check our results)
        self.assertEqual(expected_claude_config, result_claude_config)
