import unittest

import numpy as np
from hydra import initialize, compose

from src.data_generation.src.utils import read_dataset, apply_conversion


def assert_dicts_equal(dict1, dict2, atol=1e-8):
    # Ensure the dictionaries have the same keys
    assert dict1.keys() == dict2.keys(), "Dictionaries have different keys"

    # Iterate over the keys and compare the arrays
    for key in dict1:
        arr1 = dict1[key]
        arr2 = dict2[key]

        # Check if the arrays are equal (use np.allclose for floating-point arrays)
        assert np.allclose(arr1, arr2, atol=atol), f"Arrays for key '{key}' are not equal"


class TestUtils(unittest.TestCase):
    def test_get_particle_data(self):
        # https://arxiv.org/pdf/1706.09899
        itype = 4   # PartType
        path = '../../../local/data/datasets/eagle/RefL0050N0752'
        gn = 1
        sgn = 0
        gns = read_dataset(itype, 'GroupNumber', path)
        sgns = read_dataset(itype, 'SubGroupNumber', path)
        mask = np.logical_and(gns == gn, sgns == sgn)

        data = {
            'mass': read_dataset(itype, 'Mass', path)[mask],
            'coords': read_dataset(itype, 'Coordinates', path)[mask],
            'velocity': read_dataset(itype, 'Velocity', path)[mask]
        }
        print(len(data))

    def test_apply_conversion_illustris(self):
        with initialize(config_path="../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_illustris.paths = local_config.paths
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']

        columns = ["Group_M_Crit200",
                   "Group_R_Crit200",
                   "GroupPos",
                   "GroupFirstSub"
                   ]
        values = {
            "Group_M_Crit200": np.array([7071.3975]),
            "Group_R_Crit200": np.array([946.4103]),
            "GroupPos": np.array([[906.6964, 28481.55, 19434.404]]),
            "GroupFirstSub": np.array([0])
        }

        result = {}
        for column in columns:
            converted_value = apply_conversion(self.cfg_illustris.catalog.conversion,
                                               column_name=column,
                                               values=values[column],
                                               scalefactor=self.cfg_illustris.catalog.a,
                                               h=self.cfg_illustris.catalog.H)
            result[column] = np.array(converted_value)

        expected_values = {
            "Group_M_Crit200": np.array([70713977339904]),
            "Group_R_Crit200": np.array([788.64368]),
            "GroupPos": np.array([[755.5501, 23733.676, 16194.689]]),
            "GroupFirstSub": np.array([0])
        }

        # TODO subhalos
        # Perform the assertion
        assert_dicts_equal(result, expected_values)

    def test_apply_conversion_eagle(self):
        with initialize(config_path="../../../configs"):
            self.cfg_eagle = compose(config_name="data_generation.yaml")
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_eagle.paths = local_config.paths
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']

        columns = [
            "Group_M_Crit200",
            "Group_R_Crit200",
            "GroupCentreOfPotential_x",
            "GroupCentreOfPotential_y",
            "GroupCentreOfPotential_z"
        ]
        values = {
            "Group_M_Crit200": np.array([5976515072]),
            "Group_R_Crit200": np.array([35.98554]),
            "GroupCentreOfPotential_x": np.array([28.57648]),
            "GroupCentreOfPotential_y": np.array([16.38492]),
            "GroupCentreOfPotential_z": np.array([95.96060])
        }

        result = {}
        for column in columns:
            print(column)
            converted_value = apply_conversion(self.cfg_eagle.catalog.conversion,
                                               column_name=column,
                                               values=values[column],
                                               scalefactor=self.cfg_eagle.catalog.a,
                                               h=self.cfg_eagle.catalog.H)
            result[column] = np.array(converted_value)

        expected_values = {
            "Group_M_Crit200": np.array([70713977339904]),
            "Group_R_Crit200": np.array([788.64368]),
            "GroupCentreOfPotential_x": np.array([[755.5501, 23733.676, 16194.689]]),
            "GroupCentreOfPotential_y": np.array([0]),
            "GroupCentreOfPotential_z": np.array([])
        }



