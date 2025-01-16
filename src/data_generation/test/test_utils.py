import unittest

import numpy as np

from src.data_generation.src.utils import compute_intrahalo_variance, read_dataset


class TestUtils(unittest.TestCase):
    def test_compute_intrahalo_variance(self):
        path = '../../../local/data/dataset_no_rotations/npz'
        df = compute_intrahalo_variance(input_dir=path, feature_1_name='velocity_dispersion', feature_2_name='label')
        assert not df.empty, "The dataframe is empty"

    def test_get_particle_data(self):
        # https://arxiv.org/pdf/1706.09899
        itype = 4   # PartType
        path = '../../../local/data/eagle/RefL0050N0752'
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