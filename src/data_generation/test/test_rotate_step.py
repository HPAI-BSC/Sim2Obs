import os
import pickle
import unittest
from typing import Generator

from hydra import initialize, compose
from unittest.mock import patch, MagicMock
import numpy as np

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.rotate_step import RotateStep


class TestRotateStep(unittest.TestCase):
    def setUp(self):
        # Illustris objs
        with open("data/illustris/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_illustris = pickle.load(file)

        with open("data/illustris/halo_0.pkl", "rb") as file:
            self.halo_illustris = pickle.load(file)

        with open("data/illustris/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_illustris = pickle.load(file)

        with open("data/illustris/subhalo_0.pkl", "rb") as file:
            self.subhalo_illustris = pickle.load(file)

        # Eagle objs
        with open("data/eagle/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_eagle = pickle.load(file)

        with open("data/eagle/halo_0.pkl", "rb") as file:
            self.halo_eagle = pickle.load(file)

        with open("data/eagle/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_eagle = pickle.load(file)

        with open("data/eagle/subhalo_0.pkl", "rb") as file:
            self.subhalo_eagle = pickle.load(file)

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

        self.mock_input_feature = MagicMock()
        self.mock_input_feature.value = np.array([[1.0, 1.0, 1.0]])

        self.mock_output_feature = MagicMock()
        self.mock_output_feature.value = np.array([[0.0, 0.0, 0.0]])

    def mock_generator_illustris(self) -> Generator:
        """Mock generator to simulate data_loader.process."""
        yield self.halo_illustris, self.subhalo_illustris

    def mock_generator_eagle(self) -> Generator:
        """Mock generator to simulate data_loader.process."""
        yield self.halo_eagle, self.subhalo_eagle

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloContext')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_rotation_process_illustris(self, mock_halo, mock_subhalo_context, mock_subhalo_particles):
        self.rotate_step = RotateStep(self.cfg_illustris)
        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particles.return_value = self.subhalo_particles_illustris
        mock_subhalo_context.return_value = self.subhalo_context_illustris
        self.mock_cosmos_data = CosmosData(mock_halo.return_value, mock_subhalo_context.return_value,
                                 self.subhalo_illustris.subhalo_id,
                                 mock_subhalo_particles.return_value)

        # Call the rotate_step.process with the mock generator
        generator = self.rotate_step.process(
            input_feature=self.mock_input_feature,
            output_feature=self.mock_output_feature,
            cosmos_data=self.mock_cosmos_data,
            g=self.mock_generator_illustris(),
            next_step=None,
            kwargs={}
        )

        result = list(generator)

        num_iterations = (360 // self.rotate_step.rotation_angle) * len(self.rotate_step.rotation_directions)
        self.assertEqual(len(result), num_iterations)
        self.assertEqual(self.mock_cosmos_data.metadata['rotation']['has_rotated'], True)
        self.assertEqual(self.mock_cosmos_data.metadata['rotation']['angle'], 360 - self.rotate_step.rotation_angle)

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloContext')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def no_test_rotation_process_eagle(self, mock_halo, mock_subhalo_context, mock_subhalo_particles):
        self.rotate_step = RotateStep(self.cfg_eagle)
        mock_halo.return_value = self.halo_eagle
        mock_subhalo_particles.return_value = self.subhalo_particles_eagle.p
        mock_subhalo_context.return_value = self.subhalo_context_eagle
        self.mock_cosmos_data = CosmosData(mock_halo.return_value, mock_subhalo_context.return_value,
                                           self.subhalo_eagle.subhalo_id,
                                           mock_subhalo_particles.return_value)

        # Call the rotate_step.process with the mock generator
        generator = self.rotate_step.process(
            input_feature=self.mock_input_feature,
            output_feature=self.mock_output_feature,
            cosmos_data=self.mock_cosmos_data,
            g=self.mock_generator_eagle(),
            next_step=None,
            kwargs={}
        )

        result = list(generator)

        num_iterations = (360 // self.rotate_step.rotation_angle) * len(self.rotate_step.rotation_directions)
        self.assertEqual(len(result), num_iterations)
        self.assertEqual(self.mock_cosmos_data.metadata['rotation']['has_rotated'], True)
        self.assertEqual(self.mock_cosmos_data.metadata['rotation']['angle'], 360 - self.rotate_step.rotation_angle)







