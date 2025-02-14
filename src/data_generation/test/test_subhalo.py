import pickle
import unittest
import numpy as np
import pandas as pd

from src.data_generation.src.subhalo import SubhaloContext, SubHalo, SubhaloParticles


class TestSubHalo(unittest.TestCase):
    def setUp(self):
        """
        Set up data for testing the SubHalo and SubhaloContext classes.
        """
        # Sample data for SubhaloContext
        df = pd.DataFrame(data=[])
        df['spos'] = list(np.array([[100, 200, 300], [400, 500, 600]]))
        df['id_parents'] = np.array([1, 1])
        df['shmr'] = np.array([10, 20])
        df['mstr'] = np.array([1e9, 2e9])
        df['subhalo_id'] = np.array([101, 102])
        df.set_index(["id_parents", "subhalo_id"], inplace=True, drop=False)

        # Initialize SubhaloContext
        self.context = SubhaloContext(df, column_map=None)

    def test_subhalo_initialization(self):
        """
        Test that SubHalo is initialized correctly with valid attributes.
        """
        attributes = {
            "subhalo_id": 101,
            "id_parents": 1,
            "spos": [100, 200, 300],
            "shmr": 5.0,  # scaled
            "mstr": 1e9
        }
        subhalo = SubHalo(**attributes)

        # Check attributes are set correctly
        self.assertEqual(subhalo.subhalo_id, 101)
        self.assertEqual(subhalo.id_parents, 1)
        np.testing.assert_array_equal(subhalo.spos, [100, 200, 300])
        self.assertEqual(subhalo.shmr, 5.0)
        self.assertEqual(subhalo.mstr, 1e9)

    def test_subhalo_missing_attributes(self):
        """
        Test that SubHalo raises an error when required attributes are missing.
        """
        with self.assertRaises(ValueError) as context:
            SubHalo(subhalo_id=101)  # Missing `id_parents`

        self.assertIn("Missing required attributes", str(context.exception))

    def test_get_subhalo_by_id(self):
        """
        Test SubhaloContext's get_item method.
        """
        subhalo = self.context.get_subhalo_by_id(subhalo_id=101, halo_id=1)

        # Check SubHalo object attributes
        self.assertEqual(subhalo.subhalo_id, 101)
        self.assertEqual(subhalo.id_parents, 1)
        np.testing.assert_array_equal(subhalo.spos, [100, 200, 300])
        self.assertEqual(subhalo.shmr, 10)
        self.assertEqual(subhalo.mstr, 1e9)

    def test_get_items_by_halo(self):
        """
        Test SubhaloContext's get_items_by_halo method.
        """
        subhalo_context = self.context.get_items_by_halo(halo_id=1)

        # Check that the returned DataFrame matches the original data
        self.assertEqual(len(subhalo_context.data), 2)  # Two subhalos for this halo_id
        self.assertListEqual(subhalo_context.subhalo_id, [101, 102])

    def test_subhalo_particles_initialization(self):
        """
        Test SubhaloParticles initialization.
        """
        data = {'star_m': np.array([1.0, 2.0, 3.0]),
                'star_p': np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]),
                'star_v': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])}

        particles = SubhaloParticles(data, column_map=None)
        self.assertTrue(hasattr(particles, 'star_m'), True)
        self.assertEqual(hasattr(particles, 'star_p'), True)
        self.assertEqual(hasattr(particles, 'star_v'), True)

    def test_preprocess(self):
        # Illustris objs
        with open("data/illustris/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_illustris = pickle.load(file)

        with open("data/illustris/halo_context_0.pkl", "rb") as file:
            self.halo_context_illustris = pickle.load(file)

        with open("data/eagle/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_eagle = pickle.load(file)

        with open("data/eagle/halo_context_0.pkl", "rb") as file:
            self.halo_context_eagle = pickle.load(file)

        self.subhalo_context_illustris.preprocess(self.halo_context_illustris)
        self.subhalo_context_eagle.preprocess(self.halo_context_eagle)


