import unittest

import numpy as np
import pandas as pd

from src.data_generation.src.map import create_maps, muse_bins_r, create_random_zoom, get_interlopers_mask


class TestMap(unittest.TestCase):
    def setUp(self):
        self.r200_halo = 1.0  # Arbitrary R200 value
        self.spos_rot = np.random.rand(10, 3)  # Random 3D positions of 10 subhalos, shape should be (10, 3)
        self.pos_rot = np.random.rand(100, 3) * 2 - 1  # Random 3D positions of 100 particles, shape (100, 3)
        self.vel_rot = np.random.rand(100, 3)  # Random 3D velocities for 100 particles, shape (100, 3)
        self.type4 = np.random.randint(0, 2, size=100)  # Random ICL indicator (0 or 1) for particles, shape (100,)

    def test_create_maps_output_structure(self):
        # Run create_maps function
        result = create_maps(self.r200_halo, self.spos_rot, self.pos_rot, self.vel_rot, self.type4)

        # Check the presence of 'maps' and 'interlopers_id' in result
        self.assertIn('maps', result)
        self.assertIn('interlopers_id', result)

        # Validate the type of 'maps' output
        self.assertIsInstance(result['maps'], list)
        self.assertEqual(len(result['maps']), 3)  # Check if it returns 3 projections

        # Validate the structure of each map in 'maps'
        for label_map, vel_disp, bins in result['maps']:
            self.assertIsInstance(label_map, np.ndarray)
            self.assertIsInstance(vel_disp, np.ndarray)
            self.assertIsInstance(bins, np.ndarray)
            self.assertEqual(label_map.shape, vel_disp.shape)  # Confirm matched bin shapes

        # Validate 'interlopers_id' is a numpy array
        self.assertIsInstance(result['interlopers_id'], np.ndarray)

    def test_interlopers_identification(self):
        # Test with known interlopers
        r200_halo = 1.0  # Set specific r200_halo value to test inclusion criteria
        spos_rot = np.array([
            [0.1, 0.1, 0.9],  # Within the range
            [1.5, 1.5, 1.5],  # Outside the range
            [0.2, 0.3, 1.0]  # Within the range
        ])  # Shape (3, 3)

        result = create_maps(r200_halo, spos_rot, self.pos_rot, self.vel_rot, self.type4)

        # Check if only the first subhalo is considered an interloper (after removing central)
        self.assertEqual(len(result['interlopers_id']), 0)

    def test_create_random_zoom(self):
        np.random.seed(42)

        self.label_masked = np.random.rand(100, 100)  # Mock 2D array for label data
        self.vel_disp_masked = np.random.rand(100, 100)  # Mock 2D array for velocity dispersion data
        self.mask = np.random.randint(0, 2, size=(100, 100))  # Mock 2D binary mask

        # Calculate the bin size and radius using muse_bins_r to set expected zoom size
        self.bins, _ = muse_bins_r(2 * 263.34, 2.02)
        self.zoom_half_size = int(self.bins / 4)  # Define half-size of the zoom window

        label_zoom, vel_disp_zoom, mask_zoom = create_random_zoom(self.label_masked, self.vel_disp_masked, self.mask)

        # Calculate expected shape
        expected_shape = (30, 30)

        # Check if output shapes match expected zoom size
        self.assertEqual(label_zoom.shape, expected_shape)
        self.assertEqual(vel_disp_zoom.shape, expected_shape)
        self.assertEqual(mask_zoom.shape, expected_shape)  # Check shape of the mask zoom

        # Run the function twice with the same seed to check for reproducibility
        np.random.seed(42)
        label_zoom1, vel_disp_zoom1, mask_zoom1 = create_random_zoom(self.label_masked, self.vel_disp_masked, self.mask)

        np.random.seed(42)
        label_zoom2, vel_disp_zoom2, mask_zoom2 = create_random_zoom(self.label_masked, self.vel_disp_masked, self.mask)

        # Check that the outputs are identical
        np.testing.assert_array_equal(label_zoom1, label_zoom2)
        np.testing.assert_array_equal(vel_disp_zoom1, vel_disp_zoom2)
        np.testing.assert_array_equal(mask_zoom1, mask_zoom2)

    def test_mask_interlopers(self):
        # Initialize input arrays
        self.label = np.random.rand(50, 50)  # Mock 2D array for labels
        self.vel_disp = np.random.rand(50, 50)  # Mock 2D array for velocity dispersion
        self.spos_rot = np.array([[10, 10], [20, 20], [30, 30]])  # Positions for 3 subhalos
        self.shmr = pd.DataFrame([5, 10, 15], columns=["SubhaloHalfmassRadType"])  # Radii for each subhalo
        self.interlopers_id = [0, 1]  # IDs of subhalos considered interlopers
        self.biny = np.linspace(-50, 50, 50)  # Y-coordinates for the grid

        # Run the function to get the mask
        mask = get_interlopers_mask(self.label, self.vel_disp, self.spos_rot, self.shmr, self.interlopers_id, self.biny)
        expected_mask = np.load('mask.npy')

        # Verify the mask shape matches the input
        self.assertEqual(mask.shape, self.label.shape, "Mask shape does not match input shapes")

        # Ensure the mask is not entirely zero
        self.assertFalse(np.all(mask == 0), "Mask is entirely zero, which is unexpected")
        assert np.array_equal(mask, expected_mask)









