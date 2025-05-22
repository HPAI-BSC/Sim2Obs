import os
import pickle
import unittest
from typing import Generator

import joblib
from omegaconf import OmegaConf
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from hydra import initialize, compose

from src.data_generation.src.feature import FeatureProcessorFactory
from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.rotate_step import RotateStep


class TestRotateStep(unittest.TestCase):
    def setUp(self):
        # Illustris objs
        with open("data/illustris/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_illustris = pickle.load(file)

        with open("data/illustris/subhalo_0.pkl", "rb") as file:
            self.subhalo_illustris = pickle.load(file)

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

        with open("data/eagle/subhalo_0.pkl", "rb") as file:
            self.subhalo_eagle = pickle.load(file)

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
                                 self.subhalo_particles_illustris)
        cosmos_data.input_feature = input_feature_illustris
        cosmos_data.output_feature = output_feature_illustris

        yield self.halo_illustris, self.subhalo_illustris, cosmos_data

    def mock_generator_eagle(self) -> Generator:
        """Mock generator to simulate data_loader.process."""
        feature_input_name, feature_output_name = self.cfg_eagle.feature_names
        input_feature_eagle = FeatureProcessorFactory.get_feature(feature_input_name, cfg=self.cfg_eagle)
        input_feature_eagle.process(self.halo_eagle, self.subhalo_particles_eagle)
        output_feature_eagle = FeatureProcessorFactory.get_feature(feature_output_name, cfg=self.cfg_eagle)
        output_feature_eagle.model = joblib.load(os.path.join(self.project_root, self.model_path))
        output_feature_eagle.process(self.halo_eagle, self.subhalo_particles_eagle)

        cosmos_data = CosmosData(self.halo_eagle, self.subhalo_context_eagle, self.subhalo_eagle,
                                 self.subhalo_particles_eagle)
        cosmos_data.input_feature = input_feature_eagle
        cosmos_data.output_feature = output_feature_eagle
        yield self.halo_eagle, self.subhalo_eagle, cosmos_data

    def test_rotation_process_illustris(self):
        new_step = OmegaConf.create({'name': 'rotate_step', 'rotation_group': self.cfg_illustris.rotation_45})
        OmegaConf.set_struct(self.cfg_illustris.pipeline, False)
        self.cfg_illustris.pipeline.append(new_step)
        OmegaConf.set_struct(self.cfg_illustris.pipeline, True)

        self.rotate_step = RotateStep(self.cfg_illustris)
        generator = self.rotate_step.process(
            g=self.mock_generator_illustris(),
            next_step=None,
            kwargs={}
        )

        result = list(generator)

        num_iterations = (360 // self.rotate_step.rotation_angle) * len(self.rotate_step.rotation_directions)
        self.assertEqual(len(result) -1, num_iterations)

    def test_rotation_process_eagle(self):
        new_step = OmegaConf.create({'name': 'rotate_step', 'rotation_group': self.cfg_eagle.rotation_45})
        OmegaConf.set_struct(self.cfg_eagle.pipeline, False)
        self.cfg_eagle.pipeline.append(new_step)
        OmegaConf.set_struct(self.cfg_eagle.pipeline, True)

        self.rotate_step = RotateStep(self.cfg_eagle)
        generator = self.rotate_step.process(
            g=self.mock_generator_eagle(),
            next_step=None,
            kwargs={}
        )

        result = list(generator)

        num_iterations = (360 // self.rotate_step.rotation_angle) * len(self.rotate_step.rotation_directions)
        self.assertEqual(len(result) -1, num_iterations)







