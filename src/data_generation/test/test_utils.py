import unittest

import astropy.units as u
import numpy as np

from src.data_generation.src.constants import PARTICLES_PATH, ARG
from src.data_generation.src.utils import compute_intrahalo_variance, read_dataset


class TestUtils(unittest.TestCase):
    def test_compute_intrahalo_variance(self):
        compute_intrahalo_variance(input_dir='.')

    def test_get_particle_data(self):
        # https://arxiv.org/pdf/1706.09899
        itype = 4   # PartType
        path = PARTICLES_PATH['RefL0025N0376']
        att_list = ["GroupNumber", "SubGroupNumber", "Coordinates", "Mass", "Velocity"]
        dataset = read_dataset(itype=itype, att="GroupNumber", path=path, nfiles=16)
        gn = 620
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

