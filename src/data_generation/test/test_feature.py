import os
import unittest

import numpy as np
from hydra import initialize, compose
from src.data_generation.src.feature import ICLProcessor, VelocityDispersionProcessor
from src.data_generation.src.halo import Halo


class TestFeature(unittest.TestCase):
    def setUp(self):
        with initialize(config_path="../../../configs"):
            self.cfg = compose(config_name="data_generation.yaml")

        os.environ['PROJECT_ROOT'] = '../../..'

    def test_ICLProcessor_process(self):
        processor = ICLProcessor(cfg=self.cfg)

        # Create mock data for Halo and particles
        halo = Halo(halo_id=1, r200=500, m200=1e12, gpos=np.array([0, 0, 0]), fsub=0)
        particle_velocity = np.array([
            [-76.38, -498.77, -215.50],
            [-80.00, -450.00, -200.00],
            [-70.00, -470.00, -220.00],
            [-76.38, -498.77, -215.50]
        ])
        particle_position = np.array([
            [100, 200, 300],
            [400, 500, 600],
            [700, 800, 900],
            [400, 500, 600]
        ])

        result = processor.process(halo, particle_velocity, particle_position)

        self.assertEqual(result.shape, (4, 1))
        self.assertTrue(np.all(np.isin(result, [0, 1])))  # Check if it's a binary mask

    def test_VelocityDispersionProcessor_process(self):
        processor = VelocityDispersionProcessor()
        particle_velocity = np.array([
            [-76.38, -498.77, -215.50],
            [-80.00, -450.00, -200.00],
            [-70.00, -470.00, -220.00]
        ])

        result = processor.process(particle_velocity)

        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_process_empty_velocity(self):
        processor = VelocityDispersionProcessor()
        particle_velocity = np.array([])

        with self.assertRaises(ValueError):
            processor.process(particle_velocity)
