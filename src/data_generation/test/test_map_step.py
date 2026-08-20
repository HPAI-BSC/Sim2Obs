import os
import pickle
import re
import shutil
import unittest
from typing import Generator
from unittest.mock import patch

from rootutils import rootutils
rootutils.setup_root(__file__, pythonpath=True)


from src.data_generation.src.halo import Halo

import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from hydra import initialize, compose

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.feature import VelocityDispersionProcessor, ICLProcessor, FeatureProcessorFactory
from src.data_generation.src.map_step import MapStep
from src.data_generation.src.subhalo import SubhaloContext


class TestMapStep(unittest.TestCase):
    def setUp(self):
        # Illustris objs
        with open("data/illustris/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_illustris = pickle.load(file)
            self.subhalo_particles_illustris.overlapping_particles_id = None

        with open("data/illustris/subhalo_0.pkl", "rb") as file:
            self.subhalo_illustris = pickle.load(file)

        with open("data/illustris/halo_0.pkl", "rb") as file:
            self.halo_illustris = pickle.load(file)

        with open("data/illustris/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_illustris = pickle.load(file)

        # Eagle objs
        with open("data/eagle/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_eagle = pickle.load(file)
            self.subhalo_particles_eagle.overlapping_particles_id = None

        with open("data/eagle/halo_0.pkl", "rb") as file:
            self.halo_eagle = pickle.load(file)

        with open("data/eagle/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_eagle = pickle.load(file)

        with open("data/eagle/subhalo_0.pkl", "rb") as file:
            self.subhalo_eagle = pickle.load(file)

        with open("data/millenium/subhalo_particles_illustris-300_MTNG-L500-4320-A_34993.pkl", "rb") as file:
            self.subhalo_particles_millenium = pickle.load(file)
            self.subhalo_particles_millenium.overlapping_particles_id = None

        with open("data/millenium/halo_illustris-300_MTNG-L500-4320-A_34993.pkl", "rb") as file:
            self.halo_millenium = pickle.load(file)

        with open("data/millenium/millenium_subhalo_context.pkl", "rb") as file:
            self.subhalo_context_millenium = pickle.load(file)

        with initialize(config_path="'../../../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_big_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")
            self.cfg_flamingo = compose(config_name="data_generation.yaml")

            # Load the local config for paths and assign them
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_illustris.paths = local_config.paths
            self.cfg_big_illustris.paths = local_config.paths
            self.cfg_eagle.paths = local_config.paths
            self.cfg_millenium = compose(config_name="data_generation.yaml")
            self.cfg_flamingo.paths = local_config.paths

            # Dynamic catalog loading for both Illustris and Eagle
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']

            # Enforce RefL0050N0752 simulation code
            self.cfg_eagle.catalog.simulation_code = "RefL0050N0752"
            self.cfg_eagle.catalog.path = '../../../local/data/datasets/eagle/RefL0050N0752'

            self.cfg_millenium.catalog = compose("catalog/millenium.yaml")['catalog']
            self.cfg_millenium.catalog.path = '../../../local/data/datasets/millenium/MTNG-L31.25-270-A'

            self.cfg_flamingo.catalog = compose("catalog/flamingo.yaml")['catalog']
            self.cfg_flamingo.catalog.path = '../../../local/data/datasets/flamingo/L1000N0900'
        os.environ['PROJECT_ROOT'] = '../../..'
        self.remove_folder = True
        self.model_path = 'src/data_generation/model/Best_Model-Random-3par.pkl'
        self.project_root = os.getenv('PROJECT_ROOT')

    def mock_generator_illustris(self) -> Generator:
        """Mock generator to simulate data_loader.process."""
        feature_input_name, feature_output_name = self.cfg_illustris.feature_names
        input_feature_illustris = FeatureProcessorFactory.get_feature(feature_input_name, cfg=self.cfg_illustris)
        input_feature_illustris.process(self.halo_illustris, self.subhalo_particles_illustris)
        output_feature_illustris = FeatureProcessorFactory.get_feature(feature_output_name, cfg=self.cfg_illustris)
        output_feature_illustris.model = joblib.load(os.path.join(self.project_root, self.model_path))
        output_feature_illustris.process(self.halo_illustris, self.subhalo_particles_illustris)

        cosmos_data = CosmosData(self.halo_illustris, self.subhalo_context_illustris, self.subhalo_illustris,
                                 self.subhalo_particles_illustris, self.subhalo_particles_illustris)
        cosmos_data.input_feature = input_feature_illustris
        cosmos_data.output_feature = output_feature_illustris

        yield self.halo_illustris, self.subhalo_illustris, cosmos_data

    def mock_generator_millenium(self) -> Generator:
        """Mock generator to simulate data_loader.process."""
        feature_input_name, feature_output_name = self.cfg_millenium.feature_names
        input_feature_millenium = FeatureProcessorFactory.get_feature(feature_input_name, cfg=self.cfg_millenium)
        input_feature_millenium.process(self.halo_millenium, self.subhalo_particles_millenium)
        output_feature_millenium = FeatureProcessorFactory.get_feature(feature_output_name, cfg=self.cfg_millenium)
        output_feature_millenium.model = joblib.load(os.path.join(self.project_root, self.model_path))
        output_feature_millenium.process(self.halo_millenium, self.subhalo_particles_millenium)

        cosmos_data = CosmosData(self.halo_millenium, self.subhalo_context_millenium, None,
                                 self.subhalo_particles_millenium, self.subhalo_particles_millenium)
        cosmos_data.input_feature = input_feature_millenium
        cosmos_data.output_feature = output_feature_millenium

        yield self.halo_millenium, None, cosmos_data

    def mock_generator_eagle(self) -> Generator:
        """Mock generator to simulate data_loader.process."""
        feature_input_name, feature_output_name = self.cfg_eagle.feature_names
        input_feature_eagle = FeatureProcessorFactory.get_feature(feature_input_name, cfg=self.cfg_eagle)
        input_feature_eagle.process(self.halo_eagle, self.subhalo_particles_eagle)
        output_feature_eagle = FeatureProcessorFactory.get_feature(feature_output_name, cfg=self.cfg_eagle)
        output_feature_eagle.model = joblib.load(os.path.join(self.project_root, self.model_path))
        output_feature_eagle.process(self.halo_eagle, self.subhalo_particles_eagle)

        cosmos_data = CosmosData(self.halo_eagle, self.subhalo_context_eagle, self.subhalo_eagle,
                                 self.subhalo_particles_eagle,
                                 self.subhalo_particles_eagle)
        cosmos_data.input_feature = input_feature_eagle
        cosmos_data.output_feature = output_feature_eagle
        yield self.halo_eagle, self.subhalo_eagle, cosmos_data

    def test_map_process_illustris(self):
        self.map_step = MapStep(self.cfg_illustris)
        self.map_step.map_configuration['root_path'] = '../test/data/illustris'
        kwargs = {'rotation_axis': 0, 'rotation_iteration': 0, 'dim': 2}
        dtypes = self.subhalo_context_illustris.data.dtypes
        generator = self.map_step.process(
            g=self.mock_generator_illustris(),
            **kwargs,
            next_step=None
        )

        results = list(generator)
        self.assertNotEqual(results, [])
        self.assertIsInstance(results[0][0], Halo)
        self.assertIsInstance(results[0][1], VelocityDispersionProcessor)
        self.assertIsInstance(results[0][2], ICLProcessor)

    def test_map_process_eagle(self):
        self.map_step = MapStep(self.cfg_eagle)
        self.map_step.map_configuration['root_path'] = '../test/data/eagle'
        kwargs = {'rotation_axis': 0, 'rotation_iteration': 0, 'dim': 2}
        generator = self.map_step.process(
            g=self.mock_generator_eagle(),
            **kwargs,
            next_step=None
        )
        results = list(generator)
        self.assertNotEqual(results, [])
        self.assertIsInstance(results[0][0], Halo)
        self.assertIsInstance(results[0][1], VelocityDispersionProcessor)
        self.assertIsInstance(results[0][2], ICLProcessor)

    def test_map_process_millenium(self):
        self.map_step = MapStep(self.cfg_millenium)
        self.map_step.map_configuration['root_path'] = '../test/data/millenium'
        kwargs = {'rotation_axis': 0, 'rotation_iteration': 0, 'dim': 2}
        generator = self.map_step.process(
            g=self.mock_generator_millenium(),
            **kwargs,
            next_step=None
        )
        results = list(generator)
        self.assertNotEqual(results, [])
        self.assertIsInstance(results[0][0], Halo)
        self.assertIsInstance(results[0][1], VelocityDispersionProcessor)
        self.assertIsInstance(results[0][2], ICLProcessor)

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloContext')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_get_interlopers_id(self, mock_halo, mock_subhalo_context, mock_subhalo_particles):
        mock_halo.return_value = self.halo_eagle
        mock_subhalo_particles.return_value = self.subhalo_particles_eagle

        # Create test data for subhalos
        self.subhalo_data = {
            "id_parents": [0, 1, 1, 1, 2, 2],
            "group_number": [0, 0, 0, 1, 1, 0],     # indexing just for eagle
            "subhalo_id": [0, 101, 102, 103, 201, 202],
            "spos": [
                [0, 0, 0],
                [200, 200, 700],  # Interloper (within x, y, z limits)
                [300, 200, 1000],  # Not an interloper (x exceeds limit)
                [200, 300, 1000],  # Not an interloper (y exceeds limit)
                [200, 200, 1100],  # Not an interloper (z exceeds limit)
                [200, 200, 700],  # Interloper (within x, y, z limits)
            ],
            "shmr": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "mstr": [0, 1e10, 2e10, 3e10, 4e10, 5e10]
        }
        mock_subhalo_context.return_value = SubhaloContext(pd.DataFrame(self.subhalo_data), None)

        self.mock_cosmos_data = CosmosData(mock_halo.return_value,
                                           mock_subhalo_context.return_value,
                                           self.subhalo_eagle.subhalo_id,
                                           mock_subhalo_particles.return_value,
                                           mock_subhalo_particles.return_value)
        self.map_step = MapStep(self.cfg_eagle)
        self.map_step.map_configuration['root_path'] = '../test/data/eagle'
        interlopers_id = self.map_step._get_interlopers_id(cosmos_data=self.mock_cosmos_data)
        assert_array_equal(interlopers_id, np.array([[101, 0],[202, 0]]))  # Passes if they are equal

