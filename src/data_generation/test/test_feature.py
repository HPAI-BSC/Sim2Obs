import os
import pickle
import unittest
from unittest.mock import patch
from numpy.testing import assert_array_equal

import numpy as np
from hydra import initialize, compose
from src.data_generation.src.feature import ICLProcessor, VelocityDispersionProcessor
from src.data_generation.src.utils import muse_bins_r


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

        with initialize(config_path="../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")

            # Load the local config for paths and assign them
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_illustris.paths = local_config.paths
            self.cfg_eagle.paths = local_config.paths

            # Dynamic catalog loading for both Illustris and Eagle
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']

        os.environ['PROJECT_ROOT'] = '../../..'

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_ICLProcessor_process(self, mock_halo, mock_subhalo_particles):
        self.icl_processor = ICLProcessor(self.cfg_illustris)

        # Illustris
        mock_halo_instance = self.halo_illustris
        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        length = mock_subhalo_particles_instance.star_v.shape[0]

        expected_mask = np.ones((length, ), dtype=int)  # ICL mask

        result = self.icl_processor.process(mock_halo_instance, mock_subhalo_particles_instance)

        self.assertEqual(result.shape, (length, ))
        self.assertTrue(np.all(np.isin(result, [0, 1])))  # Check if it's a binary mask
        assert_array_equal(result, expected_mask)

        # Eagle
        self.icl_processor = ICLProcessor(self.cfg_eagle)
        mock_halo_instance = self.halo_eagle
        mock_halo.return_value = mock_halo_instance
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        length = mock_subhalo_particles_instance.star_v.shape[0]
        expected_mask = np.ones((length, ), dtype=int)  # ICL mask

        result = self.icl_processor.process(mock_halo_instance, mock_subhalo_particles_instance)

        self.assertEqual(result.shape, (length, ))
        self.assertTrue(np.all(np.isin(result, [0, 1])))  # Check if it's a binary mask
        assert_array_equal(result, expected_mask)

    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_ICLProcessor_generate_maps(self, mock_subhalo_particles):
        self.icl_processor = ICLProcessor(self.cfg_illustris)
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance

        r = 263.34
        # Defined by Ilaria
        bins = muse_bins_r(2 * 263.34, 2.02, 1)  # Computes bins and radius
        ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps

        self.icl_processor.value = np.ones((mock_subhalo_particles_instance.star_p.shape[0], ), dtype=int)  # ICL mask
        # Axis x-y
        icl_map, _, biny = self.icl_processor.generate_map(subhalo_particles=mock_subhalo_particles_instance,
                                                           bins=bins, ranging=ranging, axis=(0, 1))
        assert (len(icl_map)) == bins

        # Eagle
        self.icl_processor = ICLProcessor(self.cfg_eagle)
        self.icl_processor.value = np.ones((self.subhalo_particles_eagle.star_p.shape[0], ), dtype=int)  # ICL mask
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        icl_map, _, biny = self.icl_processor.generate_map(subhalo_particles=mock_subhalo_particles_instance,
                                                           bins=bins, ranging=ranging, axis=(0, 1))
        assert (len(icl_map)) == bins

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_VelocityDispersionProcessor_process(self, mock_halo, mock_subhalo_particles):
        self.vel_disp_processor = VelocityDispersionProcessor()

        # Illustris
        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        expected_shape = mock_subhalo_particles_instance.star_v.shape

        result = self.vel_disp_processor.process(mock_halo.return_value, mock_subhalo_particles_instance)

        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

        # Eagle
        mock_halo.return_value = self.halo_eagle
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        expected_shape = mock_subhalo_particles_instance.star_v.shape

        result = self.vel_disp_processor.process(mock_halo.return_value, mock_subhalo_particles_instance)

        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def no_test_process_empty_velocity(self):
        processor = VelocityDispersionProcessor()
        particle_velocity = np.array([])

        with self.assertRaises(ValueError):
            processor.process(particle_velocity=particle_velocity)

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_VelocityDispersionProcessor_generate_maps(self, mock_halo, mock_subhalo_particles):
        self.vel_disp_processor = VelocityDispersionProcessor()

        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance

        mock_halo.return_value = self.halo_illustris
        self.vel_disp_processor.value = self.vel_disp_processor.process(mock_halo.return_value,
                                                                        mock_subhalo_particles_instance)

        # Defined by Ilaria
        bins, r = muse_bins_r(2 * 263.34, 2.02, 1)  # Computes bins and radius
        ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps
        # axis x-y
        vel_map, _, biny = self.vel_disp_processor.generate_map(mock_subhalo_particles_instance, bins,
                                                                ranging, axis=(0, 1))
        assert (len(vel_map)) == bins

        mock_halo.return_value = self.halo_eagle
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance

        self.vel_disp_processor.value = self.vel_disp_processor.process(mock_halo.return_value,
                                                                        mock_subhalo_particles_instance)
        vel_map, _, biny = self.vel_disp_processor.generate_map(mock_subhalo_particles_instance, bins,
                                                                ranging, axis=(0, 1))
        assert (len(vel_map)) == bins

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_ICLProcessor_get_transition_radius(self, mock_halo, mock_subhalo_particles):
        self.icl_processor = ICLProcessor(self.cfg_illustris)

        # Illustris
        mock_halo_instance = self.halo_illustris
        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        icl_mask = self.icl_processor.process(mock_halo_instance, mock_subhalo_particles_instance)

        self.icl_processor.get_transition_radius(mock_halo.return_value,
                                                 mock_subhalo_particles.return_value,
                                                 icl_mask)
        print(icl_mask)




