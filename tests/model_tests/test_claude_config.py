import dataclasses
import unittest
import os
import numpy as np

from definitions import CONFIG_PATH
from model.claude_config_file import PlanetConfigFile
from model.claude_config import *


class TestPlanetConfig(unittest.TestCase):

    def test_equals_all_same(self):
        # Arrange
        day = 3600
        year = 7200
        resolution = 3
        planet_radius = 4000
        insolation = 10
        gravity = 9.2
        axial_tilt = 20.0
        pressure_levels = np.array([10000, 20000, 30000])
        nlevels = 3
        # Act (set up 2 identical objects)
        config1 = PlanetConfig(
            day,
            year,
            resolution,
            planet_radius,
            insolation,
            gravity,
            axial_tilt,
            np.array(pressure_levels),
            nlevels
        )
        config2 = PlanetConfig(
            day,
            year,
            resolution,
            planet_radius,
            insolation,
            gravity,
            axial_tilt,
            np.array(pressure_levels),
            nlevels
        )
        # Assert
        self.assertEqual(config1, config2)

    def test_equals_one_var_diffs(self):
        test_data = {
            "day": [3600, 2500],
            "year": [7200, 3600],
            "resolution": [3, 4],
            "planet_radius": [4000, 5000],
            "insolation": [10, 8],
            "gravity": [9.2, 7.2],
            "axial_tilt": [20.0, 15.2],
            "pressure_levels": [[10000, 20000, 30000], [1000, 2000, 3000]],
            "nlevels": [3, 4]
        }

        config1 = PlanetConfig(
            test_data["day"][0],
            test_data["year"][0],
            test_data["resolution"][0],
            test_data["planet_radius"][0],
            test_data["insolation"][0],
            test_data["gravity"][0],
            test_data["axial_tilt"][0],
            np.array(test_data["pressure_levels"][0]),
            test_data["nlevels"][0]
        )

        # Loop around all the fields to set one different each time and check they come out as not equal
        for field in dataclasses.fields(PlanetConfig):
            config2 = PlanetConfig(
                test_data["day"][1 if field.name == "day" else 0],
                test_data["year"][1 if field.name == "year" else 0],
                test_data["resolution"][1 if field.name ==
                                        "resolution" else 0],
                test_data["planet_radius"][1 if field.name ==
                                           "planet_radius" else 0],
                test_data["insolation"][1 if field.name ==
                                        "insolation" else 0],
                test_data["gravity"][1 if field.name == "gravity" else 0],
                test_data["axial_tilt"][1 if field.name ==
                                        "axial_tilt" else 0],
                np.array(test_data["pressure_levels"]
                         [1 if field.name == "pressure_levels" else 0]),
                test_data["nlevels"][1 if field.name == "nlevels" else 0]
            )
            self.assertNotEqual(
                config1, config2, f"Marked as equal when {field.name} is different")

    def test_load_from_file(self):
        # Arrange
        test_config_file = PlanetConfigFile(
            hours_in_day=1,
            days_in_year=2,
            resolution=3,
            planet_radius=4000,
            insolation=10,
            gravity=9.2,
            axial_tilt=20.0,
            pressure_levels=[100, 200, 300]
        )
        expected_config = PlanetConfig(
            day=3600,
            year=7200,
            resolution=3,
            planet_radius=4000,
            insolation=10,
            gravity=9.2,
            axial_tilt=20.0,
            pressure_levels=np.array([10000, 20000, 30000]),
            nlevels=3
        )
        # Act
        result_config = PlanetConfig.load_from_file(test_config_file)
        # Assert
        self.assertEqual(result_config, expected_config)


class TestSmoothingConfig(unittest.TestCase):

    def test_load_from_file(self):
        # Arrange
        smoothing = True
        smoothing_parameter_t = 2.2
        smoothing_parameter_u = 1.0
        smoothing_parameter_v = 0.3
        smoothing_parameter_w = 0.001
        smoothing_parameter_add = 3

        test_config_file = SmoothingConfigFile(
            smoothing=smoothing,
            smoothing_parameter_t=smoothing_parameter_t,
            smoothing_parameter_u=smoothing_parameter_u,
            smoothing_parameter_v=smoothing_parameter_v,
            smoothing_parameter_w=smoothing_parameter_w,
            smoothing_parameter_add=smoothing_parameter_add
        )
        expected_config = SmoothingConfig(
            smoothing=smoothing,
            smoothing_parameter_t=smoothing_parameter_t,
            smoothing_parameter_u=smoothing_parameter_u,
            smoothing_parameter_v=smoothing_parameter_v,
            smoothing_parameter_w=smoothing_parameter_w,
            smoothing_parameter_add=smoothing_parameter_add
        )
        # Act
        result_config = SmoothingConfig.load_from_file(test_config_file)
        # Assert
        self.assertEqual(result_config, expected_config)


class TestSaveConfig(unittest.TestCase):

    def test_load_from_file(self):
        # Arrange
        save = True
        load = True
        save_frequency = 3
        plot_frequency = 10

        test_config_file = SaveConfigFile(
            save=save,
            load=load,
            save_frequency=save_frequency,
            plot_frequency=plot_frequency
        )
        expected_config = SaveConfig(
            save=save,
            load=load,
            save_frequency=save_frequency,
            plot_frequency=plot_frequency
        )
        # Act
        result_config = SaveConfig.load_from_file(test_config_file)
        # Assert
        self.assertEqual(result_config, expected_config)


class TestViewConfig(unittest.TestCase):

    def test_load_from_file(self):
        # Arrange
        above = True
        pole_string = "n"
        pole = PoleType(pole_string)
        above_level = 2
        plot = True
        diagnostic = True
        level_plots = True
        nplots = 2
        top = 14
        verbose = True

        test_config_file = ViewConfigFile(
            above=above,
            pole=pole_string,
            above_level=above_level,
            plot=plot,
            diagnostic=diagnostic,
            level_plots=level_plots,
            nplots=nplots,
            top=top,
            verbose=verbose
        )
        expected_config = ViewConfig(
            above=above,
            pole=pole,
            above_level=above_level,
            plot=plot,
            diagnostic=diagnostic,
            level_plots=level_plots,
            nplots=nplots,
            top=top,
            verbose=verbose
        )
        # Act
        result_config = ViewConfig.load_from_file(test_config_file)
        # Assert
        self.assertEqual(result_config, expected_config)


class TestCoordinateGrid(unittest.TestCase):

    def test_equals_all_same(self):
        lat = np.array([1, 2, 3])
        lon = np.array([4, 5, 6])
        nlat = 4
        nlon = 5
        lon_plot = np.array([7, 8, 9])
        lat_plot = np.array([10, 11, 12])
        heights_plot = np.array([13, 14, 15])
        lat_z_plot = np.array([16, 17, 18])
        temperature_world = np.zeros((nlat, nlon))

        grid1 = CoordinateGrid(
            lat,
            lon,
            nlat,
            nlon,
            lon_plot,
            lat_plot,
            heights_plot,
            lat_z_plot,
            temperature_world
        )
        grid2 = CoordinateGrid(
            lat,
            lon,
            nlat,
            nlon,
            lon_plot,
            lat_plot,
            heights_plot,
            lat_z_plot,
            temperature_world
        )

        self.assertEqual(grid1, grid2)

    def test_equals_one_var_diffs(self):
        test_data = {
            "lat": [[1, 2, 3], [4, 5, 6]],
            "lon": [[4, 5, 6], [7, 8, 9]],
            "nlat": [4, 5],
            "nlon": [5, 6],
            "lon_plot": [[7, 8, 9], [10, 11, 12]],
            "lat_plot": [[10, 11, 12], [13, 14, 15]],
            "heights_plot": [[13, 14, 15], [16, 17, 18]],
            "lat_z_plot": [[16, 17, 18], [17, 18, 19]],
            "temperature_world": [np.zeros((4, 5)), np.zeros((6, 7))]
        }

        config1 = CoordinateGrid(
            np.array(test_data["lat"][0]),
            np.array(test_data["lon"][0]),
            np.array(test_data["nlat"][0]),
            np.array(test_data["nlon"][0]),
            np.array(test_data["lon_plot"][0]),
            np.array(test_data["lat_plot"][0]),
            np.array(test_data["heights_plot"][0]),
            np.array(test_data["lat_z_plot"][0]),
            test_data["temperature_world"][0]
        )

        # Loop around all the fields to set one different each time and check they come out as not equal
        for field in dataclasses.fields(CoordinateGrid):
            config2 = CoordinateGrid(
                np.array(test_data["lat"][1 if field.name == "lat" else 0]),
                np.array(test_data["lon"][1 if field.name == "lon" else 0]),
                np.array(test_data["nlat"][1 if field.name == "nlat" else 0]),
                np.array(test_data["nlon"][1 if field.name == "nlon" else 0]),
                np.array(test_data["lon_plot"]
                         [1 if field.name == "lon_plot" else 0]),
                np.array(test_data["lat_plot"]
                         [1 if field.name == "lat_plot" else 0]),
                np.array(test_data["heights_plot"]
                         [1 if field.name == "heights_plot" else 0]),
                np.array(test_data["lat_z_plot"]
                         [1 if field.name == "lat_z_plot" else 0]),
                test_data["temperature_world"][1 if field.name ==
                                               "temperature_world" else 0]
            )
            self.assertNotEqual(
                config1, config2, f"Marked as equal when {field.name} is different")

    def test_load_from_file(self):
        resolution = 3
        top = 4
        pressure_levels = np.array([100, 200, 300, 400, 500])

        plotted_pressure_levels = np.array([1, 2, 3, 4])

        lat = np.arange(-90, 91, resolution)
        lon = np.arange(0, 360, resolution)
        nlat = len(lat)
        nlon = len(lon)
        lon_plot, lat_plot = np.meshgrid(lon, lat)
        heights_plot, lat_z_plot = np.meshgrid(lat, plotted_pressure_levels)
        temperature_world = np.zeros((nlat, nlon))

        expected_coordinate_grid = CoordinateGrid(
            lat,
            lon,
            nlat,
            nlon,
            lon_plot,
            lat_plot,
            heights_plot,
            lat_z_plot,
            temperature_world
        )

        result_coordinate_grid = CoordinateGrid.load_from_file(
            resolution, top, pressure_levels)

        self.assertEqual(result_coordinate_grid, expected_coordinate_grid)


class TestClaudeConfig(unittest.TestCase):

    def test_load_from_file(self):
        test_planet_config_file = PlanetConfigFile(
            hours_in_day=1,
            days_in_year=2,
            resolution=3,
            planet_radius=4000,
            insolation=10,
            gravity=9.2,
            axial_tilt=20.0,
            pressure_levels=[100, 200, 300]
        )

        expected_planet_config = PlanetConfig(
            day=3600,
            year=7200,
            resolution=3,
            planet_radius=4000,
            insolation=10,
            gravity=9.2,
            axial_tilt=20.0,
            pressure_levels=np.array([10000, 20000, 30000]),
            nlevels=3
        )

        smoothing = True
        smoothing_parameter_t = 2.2
        smoothing_parameter_u = 1.0
        smoothing_parameter_v = 0.3
        smoothing_parameter_w = 0.001
        smoothing_parameter_add = 3
        test_smoothing_config_file = SmoothingConfigFile(
            smoothing=smoothing,
            smoothing_parameter_t=smoothing_parameter_t,
            smoothing_parameter_u=smoothing_parameter_u,
            smoothing_parameter_v=smoothing_parameter_v,
            smoothing_parameter_w=smoothing_parameter_w,
            smoothing_parameter_add=smoothing_parameter_add
        )
        expected_smoothing_config = SmoothingConfig(
            smoothing=smoothing,
            smoothing_parameter_t=smoothing_parameter_t,
            smoothing_parameter_u=smoothing_parameter_u,
            smoothing_parameter_v=smoothing_parameter_v,
            smoothing_parameter_w=smoothing_parameter_w,
            smoothing_parameter_add=smoothing_parameter_add
        )

        save = True
        load = True
        save_frequency = 3
        plot_frequency = 10
        test_save_config_file = SaveConfigFile(
            save=save,
            load=load,
            save_frequency=save_frequency,
            plot_frequency=plot_frequency
        )
        expected_save_config = SaveConfig(
            save=save,
            load=load,
            save_frequency=save_frequency,
            plot_frequency=plot_frequency
        )

        above = True
        pole_string = "n"
        pole = PoleType(pole_string)
        above_level = 2
        plot = True
        diagnostic = True
        level_plots = True
        nplots = 2
        top = 14
        verbose = True
        test_view_config_file = ViewConfigFile(
            above=above,
            pole=pole_string,
            above_level=above_level,
            plot=plot,
            diagnostic=diagnostic,
            level_plots=level_plots,
            nplots=nplots,
            top=top,
            verbose=verbose
        )
        expected_view_config = ViewConfig(
            above=above,
            pole=pole,
            above_level=above_level,
            plot=plot,
            diagnostic=diagnostic,
            level_plots=level_plots,
            nplots=nplots,
            top=top,
            verbose=verbose
        )

        lat = np.arange(-90, 91, expected_planet_config.resolution)
        lon = np.arange(0, 360, expected_planet_config.resolution)
        nlat = len(lat)
        nlon = len(lon)
        lon_plot, lat_plot = np.meshgrid(lon, lat)
        heights_plot, lat_z_plot = np.meshgrid(
            lat, expected_planet_config.pressure_levels[:expected_view_config.top]/100)
        temperature_world = np.zeros((nlat, nlon))
        expected_coordinate_grid = CoordinateGrid(
            lat,
            lon,
            nlat,
            nlon,
            lon_plot,
            lat_plot,
            heights_plot,
            lat_z_plot,
            temperature_world
        )

        test_claude_config_file = ClaudeConfigFile(
            planet_config=test_planet_config_file,
            smoothing_config=test_smoothing_config_file,
            save_config=test_save_config_file,
            view_config=test_view_config_file
        )

        expected_claude_config = ClaudeConfig(
            planet_config=expected_planet_config,
            smoothing_config=expected_smoothing_config,
            save_config=expected_save_config,
            view_config=expected_view_config,
            coordinate_grid=expected_coordinate_grid)

        result_claude_config = ClaudeConfig.load_from_file(test_claude_config_file)

        self.assertEquals(result_claude_config, expected_claude_config)
