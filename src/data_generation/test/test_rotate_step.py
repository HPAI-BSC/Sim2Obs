import os
import pickle
import unittest
from hydra import initialize, compose
from unittest.mock import patch
import numpy as np

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.feature import ICLProcessor, VelocityDispersionProcessor
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

        # Eagle objs
        with open("data/eagle/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_eagle = pickle.load(file)

        with open("data/eagle/halo_0.pkl", "rb") as file:
            self.halo_eagle = pickle.load(file)

        with open("data/eagle/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_eagle = pickle.load(file)

        # Initialize Hydra and load configuration
        with initialize(config_path="../../../configs"):
            cfg = compose(config_name="data_generation.yaml")

        os.environ['PROJECT_ROOT'] = '../../..'

        self.icl_processor = ICLProcessor(cfg)
        self.vel_disp_processor = VelocityDispersionProcessor()

        self.rotate_step = RotateStep(
            rotation_angle=45,
            z=np.array([0, 0, 1]),
            rotation_directions=np.array([[1, 0, 0], [0, 1, 0]])
        )

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloContext')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_rotation_process(self, mock_halo, mock_subhalo_context, mock_subhalo_particles):
        # Illustris
        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particles.return_value = self.subhalo_particles_illustris.p
        mock_subhalo_context.return_value = self.subhalo_context_illustris
        self.cosmos = CosmosData(mock_halo.return_value, mock_subhalo_context.return_value,
                                 mock_subhalo_particles.return_value)

        vel_value = self.vel_disp_processor.process(self.subhalo_particles_illustris.v)
        self.vel_disp_processor.value = vel_value
        result = list(self.rotate_step.process(self.vel_disp_processor, self.cosmos))
        num_iterations = (360 // self.rotate_step.rotation_angle) * len(self.rotate_step.rotation_directions)
        self.assertEqual(len(result), num_iterations)

        self.assertEqual(self.cosmos.metadata['rotation']['has_rotated'], True)
        self.assertEqual(self.cosmos.metadata['rotation']['angle'], 360 - self.rotate_step.rotation_angle)

        # Eagle
        mock_halo.return_value.return_value = self.halo_eagle
        mock_subhalo_particles.return_value = self.subhalo_particles_eagle.p
        mock_subhalo_context.return_value = self.subhalo_context_eagle
        self.cosmos = CosmosData(mock_halo.return_value, mock_subhalo_context.return_value,
                                 mock_subhalo_particles.return_value)

        vel_value = self.vel_disp_processor.process(self.subhalo_particles_illustris.v)
        self.vel_disp_processor.value = vel_value
        result = list(self.rotate_step.process(self.vel_disp_processor, self.cosmos))
        num_iterations = (360 // self.rotate_step.rotation_angle) * len(self.rotate_step.rotation_directions)
        self.assertEqual(len(result), num_iterations)

        self.assertEqual(self.cosmos.metadata['rotation']['has_rotated'], True)
        self.assertEqual(self.cosmos.metadata['rotation']['angle'], 360 - self.rotate_step.rotation_angle)







