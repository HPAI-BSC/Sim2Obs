import unittest
from hydra import initialize, compose
from omegaconf import OmegaConf


class TestHydra(unittest.TestCase):
    def test_local_hydra_config(self):
        with initialize(config_path="../../../configs"):
            # Load your configuration
            cfg = compose(config_name="data_generation")
            print(OmegaConf.to_yaml(cfg))