import os
import pickle
import unittest
from unittest.mock import patch
from numpy.testing import assert_array_equal

import numpy as np
from hydra import initialize, compose
from src.data_generation.src.feature import ICLProcessor, VelocityDispersionProcessor
from src.data_generation.src.halo import Halo
from src.data_generation.src.utils import muse_bins_r


#TODO finish tests


class TestFeature(unittest.TestCase):
    def setUp(self):
        # Illustris objs
        with open("data/illustris/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_illustris = pickle.load(file)

        with open("data/illustris/halo_0.pkl", "rb") as file:
            self.halo_illustris = pickle.load(file)

        # Eagle objs
        with open("data/eagle/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_eagle = pickle.load(file)

        with open("data/eagle/halo_0.pkl", "rb") as file:
            self.halo_eagle = pickle.load(file)

        # Initialize Hydra and load configuration
        with initialize(config_path="../../../configs"):
            cfg = compose(config_name="data_generation.yaml")

        os.environ['PROJECT_ROOT'] = '../../..'

        self.icl_processor = ICLProcessor(cfg)
        self.vel_disp_processor = VelocityDispersionProcessor()

    @patch('src.data_generation.src.halo')
    @patch('src.data_generation.src.subhalo')
    def test_ICLProcessor_process(self, mock_halo, mock_subhalo_particles):
        # Illustris
        mock_halo_instance = self.halo_illustris
        mock_halo.return_value = mock_halo_instance
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles
        expected_mask = np.ones((100, ), dtype=int)  # ICL mask

        result = self.icl_processor.process(mock_halo_instance, mock_subhalo_particles_instance.v,
                                            mock_subhalo_particles_instance.p)

        self.assertEqual(result.shape, (100, ))
        self.assertTrue(np.all(np.isin(result, [0, 1])))  # Check if it's a binary mask
        assert_array_equal(result, expected_mask)

        # Eagle
        mock_halo_instance = self.halo_eagle
        mock_halo.return_value = mock_halo_instance
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles

        result = self.icl_processor.process(mock_halo_instance, mock_subhalo_particles_instance.v,
                                            mock_subhalo_particles_instance.p)

        self.assertEqual(result.shape, (100, ))
        self.assertTrue(np.all(np.isin(result, [0, 1])))  # Check if it's a binary mask
        assert_array_equal(result, expected_mask)

    @patch('src.data_generation.src.subhalo')
    def test_ICLProcessor_generate_maps(self, mock_subhalo_particles):
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles

        # Defined by Ilaria
        bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
        ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps
        # Todo check this is the right format
        self.icl_processor.value = np.ones((100, ), dtype=int)  # ICL mask
        # Axis x-y
        icl_map, _, biny = self.icl_processor.generate_map(particle_positions=mock_subhalo_particles_instance.p,
                                                           bins=bins, ranging=ranging, axis=(0, 1))
        assert (len(icl_map)) == bins

        # Eagle
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles
        icl_map, _, biny = self.icl_processor.generate_map(particle_positions=mock_subhalo_particles_instance.p,
                                                           bins=bins, ranging=ranging, axis=(0, 1))
        assert (len(icl_map)) == bins

    @patch('src.data_generation.src.subhalo')
    def test_VelocityDispersionProcessor_process(self, mock_subhalo_particles):
        # Illustris
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles

        result = self.vel_disp_processor.process(mock_subhalo_particles_instance.v)

        self.assertEqual(result.shape, (100,))
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

        # Eagle
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles

        result = self.vel_disp_processor.process(mock_subhalo_particles_instance.v)

        self.assertEqual(result.shape, (100,))
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_process_empty_velocity(self):
        processor = VelocityDispersionProcessor()
        particle_velocity = np.array([])

        with self.assertRaises(ValueError):
            processor.process(particle_velocity)

    @patch('src.data_generation.src.subhalo')
    def test_VelocityDispersionProcessor_generate_maps(self, mock_subhalo_particles):
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles

        self.vel_disp_processor.value = self.vel_disp_processor.process(mock_subhalo_particles_instance.v)

        # Defined by Ilaria
        bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
        ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps
        # axis x-y
        vel_map, _, biny = self.vel_disp_processor.generate_map(mock_subhalo_particles_instance.p, bins,
                                                                ranging, axis=(0, 1))
        assert (len(vel_map)) == bins

        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles

        self.vel_disp_processor.value = self.vel_disp_processor.process(mock_subhalo_particles_instance.v)
        vel_map, _, biny = self.vel_disp_processor.generate_map(mock_subhalo_particles_instance.p, bins,
                                                                ranging, axis=(0, 1))
        assert (len(vel_map)) == bins



