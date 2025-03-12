import unittest

import numpy as np
from hydra import initialize, compose

from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.halo import Halo


class TestHalo(unittest.TestCase):
    def test_missing_required_attributes(self):
        """Test that the Halo class raises an error if required attributes are missing."""
        with self.assertRaises(ValueError) as context:
            Halo(halo_id=1)  # Missing 'fsub' attribute
        self.assertEqual(str(context.exception), "Missing required attributes: ['fsub']")

        with self.assertRaises(ValueError) as context:
            Halo(fsub=2)  # Missing 'halo_id' attribute
        self.assertEqual(str(context.exception), "Missing required attributes: ['halo_id']")

    def test_dynamic_attribute_assignment(self):
        """Test that the Halo class correctly assigns attributes."""
        halo = Halo(halo_id=1, fsub=2, mass=[1.0, 2.0, 3.0])  # 'mass' is a list and should be converted to numpy array

        # Check if attributes are set correctly
        self.assertEqual(halo.halo_id, 1)
        self.assertEqual(halo.fsub, 2)

        # Check if 'mass' was converted to numpy array
        self.assertTrue(isinstance(halo.mass, np.ndarray))
        self.assertTrue(np.array_equal(halo.mass, np.array([1.0, 2.0, 3.0])))

    def test_no_missing_attributes(self):
        """Test that the Halo class works correctly when all required attributes are provided."""
        halo = Halo(halo_id=1, fsub=2)

        # Ensure that attributes are set properly
        self.assertEqual(halo.halo_id, 1)
        self.assertEqual(halo.fsub, 2)

    def test_get_items_by_halo(self):
        # halo id 114, subhalo 4229

        # Initialize Hydra and load configuration
        with initialize(config_path="configs"):
            # Initialize the cfg_illustris and cfg_eagle
            self.cfg_illustris = compose(config_name="data_generation.yaml")

            # Load the local config for paths and assign them
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_illustris.paths = local_config.paths

            # Dynamic catalog loading for both Illustris and Eagle
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']

        required_keys = ['G']
        data_loader_dict = next((step for step in self.cfg_illustris.pipeline if step.name == 'data_loader'), None)
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}

        fetcher_illustris = DataFetcherFactory.get_data_fetcher(self.cfg_illustris.catalog, **attr_dict)
        halo_context = fetcher_illustris.get_halos()
        halo = halo_context.get_halo_by_id(halo_id=114)

        subhalo_context = fetcher_illustris.get_subhalos()
        subhalo_id = subhalo_context.get_subhalo_by_id(subhalo_id=4229)

        tmp = subhalo_context.get_items_by_halo(halo_id=114)
        print()


