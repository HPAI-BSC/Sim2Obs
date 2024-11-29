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
        self.scale_factor = 0.5
        df = pd.DataFrame(data=[])
        df['spos'] = list(np.array([[100, 200, 300], [400, 500, 600]]))
        df['grnr'] = np.array([1, 1])
        df['shmr'] = np.array([10, 20])
        df['mstr'] = np.array([1e9, 2e9])
        df['subhalo_id'] = np.array([101, 102])

        # Initialize SubhaloContext
        self.context = SubhaloContext(df,
            scale_factor=self.scale_factor, column_map={'subhalo_id': 'subhalo_id', 'spos': 'spos',
                                                        'id_parents': 'grnr', 'shmr': 'shmr', 'mstr': 'mstr'})

    def test_subhalo_initialization(self):
        """
        Test that SubHalo is initialized correctly with valid attributes.
        """
        attributes = {
            "subhalo_id": 101,
            "id_parents": 1,
            "spos": [50.0, 100.0, 150.0],  # scaled
            "shmr": 5.0,  # scaled
            "mstr": 1e9
        }
        subhalo = SubHalo(**attributes)

        # Check attributes are set correctly
        self.assertEqual(subhalo.subhalo_id, 101)
        self.assertEqual(subhalo.id_parents, 1)
        np.testing.assert_array_equal(subhalo.spos, [50.0, 100.0, 150.0])
        self.assertEqual(subhalo.shmr, 5.0)
        self.assertEqual(subhalo.mstr, 1e9)

    def test_subhalo_missing_attributes(self):
        """
        Test that SubHalo raises an error when required attributes are missing.
        """
        with self.assertRaises(ValueError) as context:
            SubHalo(subhalo_id=101)  # Missing `id_parents`

        self.assertIn("Missing required attributes", str(context.exception))

    def test_get_item(self):
        """
        Test SubhaloContext's get_item method.
        """
        subhalo = self.context.get_item(subhalo_id=101, halo_id=1)

        # Check SubHalo object attributes
        self.assertEqual(subhalo.subhalo_id, 101)
        self.assertEqual(subhalo.id_parents, 1)
        np.testing.assert_array_equal(subhalo.spos, [50.0, 100.0, 150.0])
        self.assertEqual(subhalo.shmr, 5.0)
        self.assertEqual(subhalo.mstr, 1e9)

    def test_get_items_by_halo(self):
        """
        Test SubhaloContext's get_items_by_halo method.
        """
        items = self.context.get_items_by_halo(halo_id=1)

        # Check that the returned DataFrame matches the original data
        self.assertEqual(len(items), 2)  # Two subhalos for this halo_id
        self.assertTrue((items["subhalo_id"] == [101, 102]).all())

    def test_subhalo_particles_initialization(self):
        """
        Test that SubhaloParticles are initialized correctly with scaling.
        """
        m = np.array([1.0, 2.0, 3.0])
        p = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        particles = SubhaloParticles(m=m, p=p, v=v, scale_factor=self.scale_factor, init=True)

        # Check attributes
        np.testing.assert_array_equal(particles.m, m)
        np.testing.assert_array_equal(particles.p, p * self.scale_factor)
        np.testing.assert_array_equal(particles.v, v)

    def test_subhalo_particles_without_scaling(self):
        """
        Test SubhaloParticles initialization without scaling.
        """
        m = np.array([1.0, 2.0, 3.0])
        p = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        particles = SubhaloParticles(m=m, p=p, v=v, init=False)

        # Check attributes
        np.testing.assert_array_equal(particles.m, m)
        np.testing.assert_array_equal(particles.p, p)
        np.testing.assert_array_equal(particles.v, v)

