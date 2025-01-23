import os
import unittest

import torch
from hydra import initialize, compose

from src.data.components.galaxy_dataset_factory import GalaxyDatasetFactory


class TestDataset(unittest.TestCase):
    def setUp(self):
        os.environ['PROJECT_ROOT'] = '..'

    def test_illustris300dataset(self):
        with initialize(config_path="../configs"):
            cfg_train = compose(config_name="train.yaml")
            cfg_train.data = compose(config_name="data/illustris300_rotation45.yaml").data

        kwargs = {'image_dir': cfg_train.data.data_dir,
                   'labels_dir': cfg_train.data.target_maskdir}

        full_dataset = GalaxyDatasetFactory.get_dataset(cfg_train.data.dataset, **kwargs)
        for i in range(1):
            image, label = full_dataset.__getitem__(index=i)
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            assert image.shape == (2, 128, 128)
            assert label.shape == (2, 128, 128)

    def test_illustris300dataset_mask_interlopers(self):
        with initialize(config_path="../configs"):
            cfg_train = compose(config_name="train.yaml")
            cfg_train.data = compose(config_name="data/illustris300_rotation45.yaml").data

        kwargs = {'image_dir': cfg_train.data.data_dir, 'labels_dir': cfg_train.data.target_maskdir,
                  'mask_interlopers': True}

        full_dataset = GalaxyDatasetFactory.get_dataset(cfg_train.data.dataset, **kwargs)
        for i in range(1):
            image, label = full_dataset.__getitem__(index=i)
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            assert image.shape == (1, 128, 128)
            assert label.shape == (1, 128, 128)

    def test_dianoga(self):
        with initialize(config_path="../configs"):
            cfg_train = compose(config_name="train.yaml")
            cfg_train.data = compose(config_name="data/dianoga.yaml").data

        kwargs = {'image_dir': cfg_train.data.data_dir, 'labels_dir': cfg_train.data.target_maskdir}
        full_dataset = GalaxyDatasetFactory.get_dataset(cfg_train.data.dataset, **kwargs)
        for i in range(1):
            image, label = full_dataset.__getitem__(index=i)
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            assert image.shape == (1, 128, 128)
            assert label.shape == (1, 128, 128)



