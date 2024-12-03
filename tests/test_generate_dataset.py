import os
import unittest
from hydra import initialize, compose
from dotenv import load_dotenv

from src.generate_dataset import main

load_dotenv('../src/data_generation/.env')


class TestGenerateDataset(unittest.TestCase):
    def test_generate_dataset_illustris(self):
        with initialize(config_path="../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
        os.environ['PROJECT_ROOT'] = '..'
        result = main(self.cfg_illustris)

    def test_generate_dataset_eagle(self):
        with initialize(config_path="../configs"):
            self.cfg_eagle = compose(config_name="data_generation.yaml")
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']
            self.cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
            self.cfg_eagle.catalog.path = '../../../local/data/eagle/RefL0050N0752'
        os.environ['PROJECT_ROOT'] = '..'
        result = main(self.cfg_eagle)
