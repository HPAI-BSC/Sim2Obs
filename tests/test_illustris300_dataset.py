import os
import unittest

import torch
from hydra import initialize, compose

from src.data.components.galaxy_dataset_factory import GalaxyDatasetFactory
from src.data.components.illustris300_dataset import Illustris300Dataset


class TestIllustris300Dataset(unittest.TestCase):
    def setUp(self):
        pass

    def test_illustris300dataset(self):
        os.environ['PROJECT_ROOT'] = '..'

        with initialize(config_path="../configs"):
            cfg_train = compose(config_name="train.yaml")
            cfg_train.data = compose(config_name="data/illustris300_rotation120.yaml").data

        kwargs = {'image_dir': cfg_train.data.data_dir,
                  'labels_dir': cfg_train.data.target_maskdir}

        full_dataset = GalaxyDatasetFactory.get_dataset(cfg_train.data.dataset, **kwargs)
        for i in range(1):
            image, label = full_dataset.__getitem__(index=i)
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            assert image.shape == (128, 128, 2)
            assert label.shape == (128, 128, 2)



