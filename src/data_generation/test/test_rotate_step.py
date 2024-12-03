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
            self.subhalo_context_illustris = pickle.load(file).data

        with open("data/illustris/subhalo_any_0.pkl", "rb") as file:
            self.subhalo_illustris = pickle.load(file)

        # Eagle objs
        with open("data/eagle/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_eagle = pickle.load(file)

        with open("data/eagle/halo_0.pkl", "rb") as file:
            self.halo_eagle = pickle.load(file)

        with open("data/eagle/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_eagle = pickle.load(file).data

        with open("data/eagle/subhalo_any_0.pkl", "rb") as file:
            self.subhalo_eagle = pickle.load(file)

        with initialize(config_path="../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")

        with initialize(config_path="../../../configs"):
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']
            self.cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
            self.cfg_eagle.catalog.path = '../../../local/data/eagle/RefL0050N0752'

        os.environ['PROJECT_ROOT'] = '../../..'

        self.mock_input_feature = MagicMock()
        self.mock_input_feature.value = np.array([[1.0, 1.0, 1.0]])

        self.mock_output_feature = MagicMock()
        self.mock_output_feature.value = np.array([[0.0, 0.0, 0.0]])

        self.rotate_step = RotateStep(
            rotation_angle=45,
            z=np.array([0, 0, 1]),
            rotation_directions=np.array([[1, 0, 0], [0, 1, 0]])
        )

    def mock_generator(self) -> Generator:
        """Mock generator to simulate data_loader.process."""
        yield self.mock_input_feature, self.mock_output_feature, self.mock_cosmos_data

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloContext')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_rotation_process_illustris(self, mock_halo, mock_subhalo_context, mock_subhalo_particles):
        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particles.return_value = self.subhalo_particles_illustris.p
        mock_subhalo_context.return_value = self.subhalo_context_illustris
        self.mock_cosmos_data = CosmosData(mock_halo.return_value, mock_subhalo_context.return_value,
                                 self.subhalo_illustris.subhalo_id,
                                 mock_subhalo_particles.return_value)

        # Call the rotate_step.process with the mock generator
        generator = self.rotate_step.process(
            input_feature=self.mock_input_feature,
            output_feature=self.mock_output_feature,
            cosmos_data=self.mock_cosmos_data,
            g=self.mock_generator(),
            next_step=None
        )

        result = list(generator)

        num_iterations = (360 // self.rotate_step.rotation_angle) * len(self.rotate_step.rotation_directions)
        self.assertEqual(len(result), num_iterations)
        self.assertEqual(self.mock_cosmos_data.metadata['rotation']['has_rotated'], True)
        self.assertEqual(self.mock_cosmos_data.metadata['rotation']['angle'], 360 - self.rotate_step.rotation_angle)

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloContext')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_rotation_process_eagle(self, mock_halo, mock_subhalo_context, mock_subhalo_particles):
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
            g=self.mock_generator(),
            next_step=None
        )

        result = list(generator)

        num_iterations = (360 // self.rotate_step.rotation_angle) * len(self.rotate_step.rotation_directions)
        self.assertEqual(len(result), num_iterations)
        self.assertEqual(self.mock_cosmos_data.metadata['rotation']['has_rotated'], True)
        self.assertEqual(self.mock_cosmos_data.metadata['rotation']['angle'], 360 - self.rotate_step.rotation_angle)







