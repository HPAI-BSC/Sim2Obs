import os
import unittest

from hydra import initialize, compose
import pandas as pd

from src.data_generation.src.data_loader_step import DataLoader
from src.data_generation.src.halo import Halo
from src.data_generation.src.subhalo import SubhaloParticles


class TestDataLoaderStep(unittest.TestCase):
    def setUp(self):
        with initialize(config_path="../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")

        with initialize(config_path="../../../configs"):
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']

        self.cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
        self.cfg_eagle.catalog.path = '../../../local/data/eagle/RefL0050N0752'
        os.environ['PROJECT_ROOT'] = '../../..'

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
        result = list(self.dataLoader.load_subhalo_particles_data(subhalo_id=0))

        for particles in result:
            self.assertIsInstance(particles, SubhaloParticles)

    def test_load_subhalo_particles_eagle(self):
        self.dataLoader = DataLoader(self.cfg_eagle)
        result = list(self.dataLoader.load_subhalo_particles_data(subhalo_id=0, halo_id=620))

        for particles in result:
            self.assertIsInstance(particles, SubhaloParticles)

