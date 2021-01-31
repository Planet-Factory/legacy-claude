import dataclasses
import unittest, os
import numpy as np 

from definitions import CONFIG_PATH
from model.claude_config_file import PlanetConfigFile
from model.claude_config import PlanetConfig, ClaudeConfig

class TestPlanetConfig(unittest.TestCase):

    def test_equals_all_same(self):
        day=3600
        year=7200
        resolution=3
        planet_radius=4000
        insolation=10
        gravity=9.2
        axial_tilt=20.0
        pressure_levels=np.array([10000,20000,30000])
        nlevels=3

        config1 = PlanetConfig(
            day,
            year,
            resolution,
            planet_radius,
            insolation,
            gravity,
            axial_tilt,
            pressure_levels,
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
            pressure_levels,
            nlevels
        )

        self.assertTrue(config1 == config2)
    
    def test_equals_one_var_diffs(self):
        
        test_data = {
            "day":[3600,2500],
            "year":[7200,3600],
            "resolution":[3,4],
            "planet_radius":[4000,5000],
            "insolation":[10,8],
            "gravity":[9.2,7.2],
            "axial_tilt":[20.0,15.2],
            "pressure_levels":[np.array([10000,20000,30000]),np.array([1000, 2000, 3000])],
            "nlevels":[3,4]
        }

        config1 = PlanetConfig(
            test_data["day"][0],
            test_data["year"][0],
            test_data["resolution"][0],
            test_data["planet_radius"][0],
            test_data["insolation"][0],
            test_data["gravity"][0],
            test_data["axial_tilt"][0],
            test_data["pressure_levels"][0],
            test_data["nlevels"][0]
        )

        for field in dataclasses.fields(PlanetConfig):
            config2 = PlanetConfig(
                test_data["day"][1 if field.name=="day" else 0],
                test_data["year"][1 if field.name=="year" else 0],
                test_data["resolution"][1 if field.name=="resolution" else 0],
                test_data["planet_radius"][1 if field.name=="planet_radius" else 0],
                test_data["insolation"][1 if field.name=="insolation" else 0],
                test_data["gravity"][1 if field.name=="gravity" else 0],
                test_data["axial_tilt"][1 if field.name=="axial_tilt" else 0],
                test_data["pressure_levels"][1 if field.name=="pressure_levels" else 0],
                test_data["nlevels"][1 if field.name=="nlevels" else 0]
            )
            self.assertFalse(config1 == config2, f"Marked as equal when {field.name} is different")

    def test_load_from_file(self):
        test_config_file = PlanetConfigFile(
            hours_in_day=1,
            days_in_year=2,
            resolution=3,
            planet_radius=4000,
            insolation=10,
            gravity=9.2,
            axial_tilt=20.0,
            pressure_levels=[100,200,300]
        )
        expected_config = PlanetConfig(
            day=3600,
            year=7200,
            resolution=3,
            planet_radius=4000,
            insolation=10,
            gravity=9.2,
            axial_tilt=20.0,
            pressure_levels=np.array([10000,20000,30000]),
            nlevels=3
        )

        result_config = PlanetConfig.load_from_file(test_config_file)

        self.assertEqual(result_config, expected_config)

