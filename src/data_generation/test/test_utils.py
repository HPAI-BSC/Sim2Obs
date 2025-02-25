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