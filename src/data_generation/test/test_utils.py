import unittest

import numpy as np

from src.data_generation.src.utils import read_dataset


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