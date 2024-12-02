import os
import pickle
import shutil
import unittest
from unittest.mock import patch, Mock
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
            self.subhalo_context_illustris = pickle.load(file)

        # Eagle objs
        with open("data/eagle/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_eagle = pickle.load(file)

        with open("data/eagle/halo_0.pkl", "rb") as file:
            self.halo_eagle = pickle.load(file)

        with open("data/eagle/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_eagle = pickle.load(file)

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

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloContext')
    def test_map_process_illustris(self, mock_halo, mock_subhalo_context):
        self.icl_processor = ICLProcessor(self.cfg_illustris)
        self.vel_disp_processor = VelocityDispersionProcessor()
        self.map_step = MapStep(self.cfg_illustris)

        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particle_positions = Mock()
        mock_subhalo_particle_positions.return_value = self.subhalo_particles_illustris.p
        mock_subhalo_particle_velocity = Mock()
        mock_subhalo_particle_velocity.return_value = self.subhalo_particles_illustris.v
        mock_subhalo_context.return_value = self.subhalo_context_illustris
        self.cosmos = CosmosData(mock_halo.return_value, mock_subhalo_context.return_value,
                                 self.subhalo_illustris.subhalo_id,
                                 mock_subhalo_particle_positions.return_value)

        vel_value = self.vel_disp_processor.process(mock_subhalo_particle_velocity.return_value)
        self.vel_disp_processor.value = vel_value

        icl_value = self.icl_processor.process(mock_halo.return_value, mock_subhalo_particle_velocity.return_value,
                                               mock_subhalo_particle_positions.return_value)
        self.icl_processor.value = icl_value
        self.map_step.process(self.vel_disp_processor, self.icl_processor, self.cosmos)

        if self.map_step.map_configuration['rotation_angle'] is not None:
            path = os.path.join(self.map_step.map_configuration['root_path'],
                                f'rotation_{self.map_step.map_configuration["rotation_angle"]}')
        else:
            path = os.path.join(self.map_step.map_configuration['root_path'],
                                self.vel_disp_processor.name)

        shutil.rmtree(path)

        if self.map_step.map_configuration['rotation_angle'] is None:
            path = os.path.join(self.map_step.map_configuration['root_path'],
                                self.icl_processor.name)
            shutil.rmtree(path)

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloContext')
    def test_map_process_eagle(self, mock_halo, mock_subhalo_context):
        self.icl_processor = ICLProcessor(self.cfg_eagle)
        self.vel_disp_processor = VelocityDispersionProcessor()
        self.map_step = MapStep(self.cfg_eagle)

        mock_halo.return_value = self.halo_eagle
        mock_subhalo_particle_positions = Mock()
        mock_subhalo_particle_positions.return_value = self.subhalo_particles_eagle.p
        mock_subhalo_particle_velocity = Mock()
        mock_subhalo_particle_velocity.return_value = self.subhalo_particles_eagle.v
        mock_subhalo_context.return_value = self.subhalo_context_illustris
        self.cosmos = CosmosData(mock_halo.return_value, mock_subhalo_context.return_value,
                                 self.subhalo_illustris.subhalo_id,
                                 mock_subhalo_particle_positions.return_value)

        vel_value = self.vel_disp_processor.process(mock_subhalo_particle_velocity.return_value)
        self.vel_disp_processor.value = vel_value

        icl_value = self.icl_processor.process(mock_halo.return_value, mock_subhalo_particle_velocity.return_value,
                                               mock_subhalo_particle_positions.return_value)
        self.icl_processor.value = icl_value
        self.map_step.process(self.vel_disp_processor, self.icl_processor, self.cosmos)

        if self.map_step.map_configuration['rotation_angle'] is not None:
            path = os.path.join(self.map_step.map_configuration['root_path'],
                                f'rotation_{self.map_step.map_configuration["rotation_angle"]}')
        else:
            path = os.path.join(self.map_step.map_configuration['root_path'],
                                self.vel_disp_processor.name)

        shutil.rmtree(path)

        if self.map_step.map_configuration['rotation_angle'] is None:
            path = os.path.join(self.map_step.map_configuration['root_path'],
                                self.icl_processor.name)
            shutil.rmtree(path)

