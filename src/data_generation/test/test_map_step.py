import os
import pickle
import shutil
import unittest
from typing import Generator
from unittest.mock import patch, MagicMock
from scipy import stats

from hydra import initialize, compose

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.feature import VelocityDispersionProcessor, ICLProcessor
from src.data_generation.src.map_step import MapStep


class TestMapStep(unittest.TestCase):
    def setUp(self):
        # Illustris objs
        with open("data/illustris/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_illustris = pickle.load(file)

        with open("data/illustris/subhalo_any_0.pkl", "rb") as file:
            self.subhalo_illustris = pickle.load(file)

        with open("data/illustris/halo_0.pkl", "rb") as file:
            self.halo_illustris = pickle.load(file)

        with open("data/illustris/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_illustris = pickle.load(file).data

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
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_eagle.paths = local_config.paths
            self.cfg_illustris.paths = local_config.paths
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']
            self.cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
            self.cfg_eagle.catalog.path = '../../../local/data/eagle/RefL0050N0752'

        os.environ['PROJECT_ROOT'] = '../../..'
        self.remove_folder = True

    def tearDown(self):
        if self.map_step.map_configuration['rotation_angle'] is not None:
            path = os.path.join(self.map_step.map_configuration['root_path'],
                                f'rotation_{self.map_step.map_configuration["rotation_angle"]}')
        else:
            path = os.path.join(self.map_step.map_configuration['root_path'],
                                self.mock_input_feature.name)
        if self.remove_folder:
            if os.path.exists(path):
                shutil.rmtree(path)
                if self.map_step.map_configuration['rotation_angle'] is None:
                    path = os.path.join(self.map_step.map_configuration['root_path'],
                                        self.mock_output_feature.name)
                    shutil.rmtree(path)

    def mock_generator(self) -> Generator:
        """Mock generator to simulate data_loader.process."""
        yield self.mock_input_feature, self.mock_output_feature, self.mock_cosmos_data

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloContext')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    @patch('src.data_generation.src.feature.VelocityDispersionProcessor')
    @patch('src.data_generation.src.feature.ICLProcessor')
    def test_map_process_illustris(self, mock_halo, mock_subhalo_context, mock_subhalo_particles,
                                   mock_input_feature, mock_output_feature):
        mock_input_feature.return_value = VelocityDispersionProcessor()
        mock_input_feature.name = 'feature1'
        mock_input_feature.generate_map = MagicMock(
            side_effect=lambda particle_positions, bins, ranging, axis: stats.binned_statistic_2d(
                particle_positions[:, axis[0]], particle_positions[:, axis[1]],
                particle_positions[:, 2],  # Example feature
                statistic="mean", bins=bins, range=ranging
            )[:3]  # Return the first three elements (map, binx, biny)
        )
        self.mock_input_feature = mock_input_feature

        mock_output_feature.return_value = ICLProcessor(self.cfg_illustris)
        mock_output_feature.name = 'feature2'
        mock_output_feature.generate_map = MagicMock(
            side_effect=lambda particle_positions, bins, ranging, axis: stats.binned_statistic_2d(
                particle_positions[:, axis[0]], particle_positions[:, axis[1]],
                particle_positions[:, 2],  # Example feature
                statistic="sum", bins=bins, range=ranging
            )[:3]  # Return the first three elements (map, binx, biny)
        )
        self.mock_output_feature = mock_output_feature

        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particles.return_value = self.subhalo_particles_illustris.p
        mock_subhalo_context.return_value = self.subhalo_context_illustris
        self.mock_cosmos_data = CosmosData(mock_halo.return_value, mock_subhalo_context.return_value,
                                           self.subhalo_illustris.subhalo_id,
                                           mock_subhalo_particles.return_value)
        self.map_step = MapStep(self.cfg_illustris)
        self.map_step.map_configuration['root_path'] = '../test/data/illustris'

        generator = self.map_step.process(
            input_feature=self.mock_input_feature,
            output_feature=self.mock_output_feature,
            cosmos_data=self.mock_cosmos_data,
            g=self.mock_generator(),
            next_step=None
        )

        _ = list(generator)

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloContext')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    @patch('src.data_generation.src.feature.VelocityDispersionProcessor')
    @patch('src.data_generation.src.feature.ICLProcessor')
    def test_map_process_eagle(self, mock_halo, mock_subhalo_context, mock_subhalo_particles,
                                   mock_input_feature, mock_output_feature):
        mock_input_feature.return_value = VelocityDispersionProcessor()
        mock_input_feature.name = 'feature1'
        mock_input_feature.generate_map = MagicMock(
            side_effect=lambda particle_positions, bins, ranging, axis: stats.binned_statistic_2d(
                particle_positions[:, axis[0]], particle_positions[:, axis[1]],
                particle_positions[:, 2],  # Example feature
                statistic="mean", bins=bins, range=ranging
            )[:3]  # Return the first three elements (map, binx, biny)
        )
        self.mock_input_feature = mock_input_feature

        mock_output_feature.return_value = ICLProcessor(self.cfg_eagle)
        mock_output_feature.name = 'feature2'
        mock_output_feature.generate_map = MagicMock(
            side_effect=lambda particle_positions, bins, ranging, axis: stats.binned_statistic_2d(
                particle_positions[:, axis[0]], particle_positions[:, axis[1]],
                particle_positions[:, 2],  # Example feature
                statistic="sum", bins=bins, range=ranging
            )[:3]  # Return the first three elements (map, binx, biny)
        )
        self.mock_output_feature = mock_output_feature

        mock_halo.return_value = self.halo_eagle
        mock_subhalo_particles.return_value = self.subhalo_particles_eagle.p
        mock_subhalo_context.return_value = self.subhalo_context_eagle
        self.mock_cosmos_data = CosmosData(mock_halo.return_value, mock_subhalo_context.return_value,
                                           self.subhalo_eagle.subhalo_id,
                                           mock_subhalo_particles.return_value)
        self.map_step = MapStep(self.cfg_illustris)
        self.map_step.map_configuration['root_path'] = '../test/data/eagle'

        generator = self.map_step.process(
            input_feature=self.mock_input_feature,
            output_feature=self.mock_output_feature,
            cosmos_data=self.mock_cosmos_data,
            g=self.mock_generator(),
            next_step=None
        )

        _ = list(generator)
