import os
import unittest

import numpy as np
import torch
from hydra import initialize, compose

from src.data.components.galaxy_dataset_factory import GalaxyDatasetFactory


class TestSplit(unittest.TestCase):
    def setUp(self):
        os.environ['PROJECT_ROOT'] = '..'

    def test_illustris300dataset(self):
        with initialize(config_path="../configs"):
            cfg_train = compose(config_name="train.yaml")
            cfg_train.data = compose(config_name="data/galaxy_illustris300_rotation120.yaml").data

        kwargs = {'image_dir': cfg_train.data.data_dir,
                  'labels_dir': cfg_train.data.target_maskdir,
                  'mask_interlopers': False}

        full_dataset = GalaxyDatasetFactory.get_dataset(cfg_train.data.dataset, **kwargs)
        sample_size = len(full_dataset.images)
        halo_ids = full_dataset.get_halo_ids()

        # Shuffle halo IDs for randomness
        np.random.seed(0)
        np.random.shuffle(halo_ids)
        torch.manual_seed(0)
        train_val_test_split = [.85, .05, .10]

        # Define split sizes at the halo level
        num_halos = len(halo_ids)
        train_size = int(train_val_test_split[0] * num_halos)
        val_size = int(train_val_test_split[1] * num_halos)
        test_size = num_halos - train_size - val_size

        train_halos = set(halo_ids[:train_size])
        val_halos = set(halo_ids[train_size:train_size + val_size])
        test_halos = set(halo_ids[train_size + val_size:])

        # Apply filtering
        train_ds = full_dataset.filter_by_halos(list(train_halos))
        val_ds = full_dataset.filter_by_halos(list(val_halos))
        test_ds = full_dataset.filter_by_halos(list(test_halos))

        # Total images
        assert sample_size == (len(train_ds.images) + len(val_ds.images) + len(test_ds.images))

        # No overlapping
        set_train = set(train_ds.images)
        set_val = set(val_ds.images)
        set_test = set(test_ds.images)
        assert set_train.isdisjoint(set_val)
        assert set_val.isdisjoint(set_test)
        assert set_train.isdisjoint(set_test)

        # Halo ids
        train_halo_ids = set()
        for img in train_ds.images:
            halo = int(img.split('_')[1])
            train_halo_ids.add(halo)
        assert train_halo_ids == train_halos

        val_halo_ids = set()
        for img in val_ds.images:
            halo = int(img.split('_')[1])
            val_halo_ids.add(halo)
        assert val_halo_ids == val_halos

        test_halo_ids = set()
        for img in test_ds.images:
            halo = int(img.split('_')[1])
            test_halo_ids.add(halo)
        assert test_halo_ids == test_halos

    def test_dianoga(self):
        with initialize(config_path="../configs"):
            cfg_train = compose(config_name="train.yaml")
            cfg_train.data = compose(config_name="data/galaxy_dianoga.yaml").data

        kwargs = {'image_dir': cfg_train.data.data_dir,
                  'labels_dir': cfg_train.data.target_maskdir}

        full_dataset = GalaxyDatasetFactory.get_dataset(cfg_train.data.dataset, **kwargs)
        sample_size = len(full_dataset.images)
        halo_ids = full_dataset.get_halo_ids()

        # Shuffle halo IDs for randomness
        np.random.seed(0)
        np.random.shuffle(halo_ids)
        torch.manual_seed(0)
        train_val_test_split = [.85, .05, .10]

        # Define split sizes at the halo level
        num_halos = len(halo_ids)
        train_size = int(train_val_test_split[0] * num_halos)
        val_size = int(train_val_test_split[1] * num_halos)
        test_size = num_halos - train_size - val_size

        train_halos = set(halo_ids[:train_size])
        val_halos = set(halo_ids[train_size:train_size + val_size])
        test_halos = set(halo_ids[train_size + val_size:])

        # Apply filtering
        train_ds = full_dataset.filter_by_halos(list(train_halos))
        val_ds = full_dataset.filter_by_halos(list(val_halos))
        test_ds = full_dataset.filter_by_halos(list(test_halos))

        # Total images
        assert sample_size == (len(train_ds.images) + len(val_ds.images) + len(test_ds.images))

        # No overlapping
        set_train = set(train_ds.images)
        set_val = set(val_ds.images)
        set_test = set(test_ds.images)
        assert set_train.isdisjoint(set_val)
        assert set_val.isdisjoint(set_test)
        assert set_train.isdisjoint(set_test)

        # Halo ids
        train_halo_ids = set()
        for img in train_ds.images:
            halo = int(img.split('_')[0].replace('D', ''))
            train_halo_ids.add(halo)
        assert train_halo_ids == train_halos

        val_halo_ids = set()
        for img in val_ds.images:
            halo = int(img.split('_')[0].replace('D', ''))
            val_halo_ids.add(halo)
        assert val_halo_ids == val_halos

        test_halo_ids = set()
        for img in test_ds.images:
            halo = int(img.split('_')[0].replace('D', ''))
            test_halo_ids.add(halo)
        assert test_halo_ids == test_halos


