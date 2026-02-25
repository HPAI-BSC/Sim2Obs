import unittest
import numpy as np

from rootutils import rootutils
rootutils.setup_root(__file__, pythonpath=True)

from src.data_generation.src.feature_computer import FeatureComputer


class TestFeatureComputer(unittest.TestCase):

    def setUp(self):
        # Simple case: 10 stars in a sphere, uniform mass
        np.random.seed(42)
        n_particles = 1000

        # Uniform sphere of radius 10
        r = np.random.uniform(0, 10, n_particles)
        theta = np.random.uniform(0, np.pi, n_particles)
        phi = np.random.uniform(0, 2 * np.pi, n_particles)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        self.pos = np.stack([x, y, z], axis=1)
        self.mass = np.ones(n_particles)  # uniform mass

        # Build mock SubhaloParticles-like dict
        self.mock_particles = {
            0: {'POS': self.pos, 'MASS': self.mass},
            1: {'POS': self.pos * 0.5, 'MASS': self.mass * 2},  # smaller, more massive
            2: {'POS': np.array([]).reshape(0, 3), 'MASS': np.array([])}  # empty subhalo
        }

    def test_debug_registry(self):
        import os
        import src.data_generation.src.features_dataset.shmr  # force import
        print("Registry:", FeatureComputer._registry)
        self.assertIn('stellar_half_mass_radius', FeatureComputer._registry)

    def test_registry_contains_stellar_half_mass_radius(self):
        """stellar_half_mass_radius should be registered."""
        self.assertIn('stellar_half_mass_radius', FeatureComputer._registry)

    def test_compute_unknown_method_raises(self):
        """Calling an unregistered method should raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            FeatureComputer.compute('non_existent_method', {}, {})

    def test_compute_dispatches_correctly(self):
        """compute() should call the registered method and return a result."""
        result = FeatureComputer.compute('stellar_half_mass_radius', self.mock_particles, {})
        self.assertIsNotNone(result)

    # stellar_half_mass_radius tests

    def test_returns_numpy_array(self):
        """Result should be a numpy array."""
        result = FeatureComputer.compute('stellar_half_mass_radius', self.mock_particles, {})
        self.assertIsInstance(result, np.ndarray)

    def test_result_length_matches_subhalo_count(self):
        """One SHMR value per subhalo."""
        result = FeatureComputer.compute('stellar_half_mass_radius', self.mock_particles, {})
        self.assertEqual(len(result), len(self.mock_particles))

    def test_shmr_positive_value(self):
        """SHMR should be positive for valid particles."""
        result = FeatureComputer.compute('stellar_half_mass_radius', self.mock_particles, {})
        self.assertGreater(result[0], 0)

    def test_shmr_empty_subhalo_returns_nan(self):
        """Empty subhalo should return np.nan."""
        result = FeatureComputer.compute('stellar_half_mass_radius', self.mock_particles, {})
        self.assertTrue(np.isnan(result[2]))

    def test_shmr_smaller_subhalo_has_smaller_radius(self):
        """Subhalo 1 has positions scaled by 0.5 so SHMR should be ~half of subhalo 0."""
        result = FeatureComputer.compute('stellar_half_mass_radius', self.mock_particles, {})
        self.assertAlmostEqual(result[1], result[0] * 0.5, delta=result[0] * 0.1)

    def test_shmr_uniform_sphere(self):
        """
        For a uniform sphere of radius R, the half mass radius should be
        close to R * (0.5)^(1/3) ≈ 0.794 * R.
        """
        result = FeatureComputer.compute('stellar_half_mass_radius', self.mock_particles, {})
        expected = 10 * (0.5 ** (1 / 3))  # ~7.94
        self.assertAlmostEqual(result[0], expected, delta=0.5)

    def test_shmr_single_particle(self):
        """Single particle subhalo should return radius 0 (at center of mass)."""
        single = {
            0: {
                'POS' : np.array([[1.0, 2.0, 3.0]]),
                'MASS': np.array([1.0])
            }
        }
        result = FeatureComputer.compute('stellar_half_mass_radius', single, {})
        self.assertAlmostEqual(result[0], 0.0, places=5)

    def test_shmr_off_center_subhalo(self):
        """SHMR should be invariant to translation of the subhalo."""
        # Shift all particles by a large offset
        offset = np.array([1000.0, 1000.0, 1000.0])
        shifted = {'POS': self.pos + offset, 'MASS': self.mass}
        original = {'POS': self.pos,           'MASS': self.mass}

        result_shifted = FeatureComputer._subhalo_shmr(shifted)
        result_original = FeatureComputer._subhalo_shmr(original)
        self.assertAlmostEqual(result_shifted, result_original, places=5)

    def test_shmr_all_empty_subhalos(self):
        """All empty subhalos should return all nans."""
        empty_particles = {
            0: {'POS': np.array([]).reshape(0, 3), 'MASS': np.array([])},
            1: {'POS': np.array([]).reshape(0, 3), 'MASS': np.array([])}
        }
        result = FeatureComputer.compute('stellar_half_mass_radius', empty_particles, {})
        self.assertTrue(np.all(np.isnan(result)))

