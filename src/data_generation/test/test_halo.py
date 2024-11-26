import unittest

import numpy as np

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