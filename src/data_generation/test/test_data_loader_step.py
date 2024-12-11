import unittest
from dotenv import load_dotenv

import numpy as np
from hydra import initialize, compose
import pandas as pd

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.data_loader_step import DataLoader
from src.data_generation.src.feature import VelocityDispersionProcessor, ICLProcessor, FeatureProcessor
from src.data_generation.src.halo import Halo
from src.data_generation.src.subhalo import SubhaloParticles, SubhaloContext

load_dotenv('../.env')


class TestDataLoaderStep(unittest.TestCase):
    def setUp(self):
        with initialize(config_path="../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_eagle.paths = local_config
            self.cfg_illustris.paths = local_config
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']

        self.cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
        self.cfg_eagle.catalog.path = '../../../local/data/eagle/RefL0050N0752'

    def test_load_halo_data_illustris(self):
        self.dataLoader = DataLoader(self.cfg_illustris)
        result = list(self.dataLoader.load_halo_data())
        self.assertEqual(len(result), 83)

        for halo_df, subhalo_df in result:
            self.assertIsInstance(halo_df, Halo)
            self.assertIsInstance(subhalo_df, pd.DataFrame)

    def test_load_halo_data_eagle(self):
        self.dataLoader = DataLoader(self.cfg_eagle)
        result = list(self.dataLoader.load_halo_data())
        self.assertEqual(len(result), 21)

        for halo_df, subhalo_df in result:
            self.assertIsInstance(halo_df, Halo)
            self.assertIsInstance(subhalo_df, pd.DataFrame)

    def test_load_subhalo_particles_illustris(self):
        self.dataLoader = DataLoader(self.cfg_illustris)
        particles = self.dataLoader._load_subhalo_particles_data(subhalo_id=0)
        self.assertIsInstance(particles, SubhaloParticles)

    def test_load_subhalo_particles_eagle(self):
        self.dataLoader = DataLoader(self.cfg_eagle)
        particles = self.dataLoader._load_subhalo_particles_data(subhalo_id=0, halo_id=15755)
        self.assertIsInstance(particles, SubhaloParticles)

    def test_process_illustris(self):
        data_loader = DataLoader(self.cfg_illustris)
        halo_subhalo_list = data_loader.load_halo_data()[:5]
        input_feature = VelocityDispersionProcessor()
        output_feature = ICLProcessor(cfg=self.cfg_illustris)
        halo = Halo.empty_instance()
        subhalos = SubhaloContext.empty_instance()
        cosmos_data = CosmosData(halo=halo,
                                 subhalo_id=-1,
                                 subhalos=subhalos,
                                 subhalo_particle_positions=np.array([]))
        generator = data_loader.process(input_feature, output_feature, cosmos_data, g=halo_subhalo_list, next_step=None)
        result = None
        try:
            result = next(generator)
        except StopIteration:
            pass  # You can assert here if you expect no more items
        self.assertNotEqual(result, None)
        self.assertIsInstance(result[0], FeatureProcessor)  # Assert the input_feature
        self.assertIsInstance(result[1], FeatureProcessor)  # Assert the output_feature
        self.assertIsInstance(result[2], CosmosData)  # Assert the cosmos_data
        self.assertNotEqual(len(result[0].value), 0)
        self.assertNotEqual(len(result[1].value), 0)
        self.assertNotEqual(len(result[2].subhalo_particle_positions), 0)
        self.assertNotEqual(len(result[2].subhalos), 0)

    def test_process_eagle(self):
        data_loader = DataLoader(self.cfg_eagle)
        halo_subhalo_list = data_loader.load_halo_data()[:5]
        input_feature = VelocityDispersionProcessor()
        output_feature = ICLProcessor(cfg=self.cfg_eagle)
        halo = Halo.empty_instance()
        subhalos = SubhaloContext.empty_instance()
        cosmos_data = CosmosData(halo=halo,
                                 subhalo_id=-1,
                                 subhalos=subhalos,
                                 subhalo_particle_positions=np.array([]))
        generator = data_loader.process(input_feature, output_feature, cosmos_data, g=halo_subhalo_list, next_step=None)
        result = None
        try:
            result = next(generator)
        except StopIteration:
            pass  # You can assert here if you expect no more items
        self.assertNotEqual(result, None)
        self.assertIsInstance(result[0], FeatureProcessor)  # Assert the input_feature
        self.assertIsInstance(result[1], FeatureProcessor)  # Assert the output_feature
        self.assertIsInstance(result[2], CosmosData)  # Assert the cosmos_data
        self.assertNotEqual(len(result[0].value), 0)
        self.assertNotEqual(len(result[1].value), 0)
        self.assertNotEqual(len(result[2].subhalo_particle_positions), 0)
        self.assertNotEqual(len(result[2].subhalos), 0)



