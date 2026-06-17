import unittest
import numpy as np

from rootutils import rootutils

from src.data_generation.src.subhalo import SubhaloParticles

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
        self.mock_particles = SubhaloParticles({'star_p': self.pos * 0.5, 'star_m': self.mass * 2}, column_map=None)

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
            FeatureComputer.compute('non_existent_method', {})

    def test_compute_dispatches_correctly(self):
        """compute() should call the registered method and return a result."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles)
        self.assertIsNotNone(result)

    # stellar_half_mass_radius tests

    def test_returns_float(self):
        """Result should be a numpy array."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles)
        self.assertIsInstance(result, float)

    def test_shmr_uniform_sphere(self):
        """
        For a uniform sphere of radius R, the half mass radius should be
        close to R * (0.5)^(1/3) ≈ 0.794 * R.
        """
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles)
        expected = 10 * (0.5 ** (1 / 3))  # ~7.94
        self.assertAlmostEqual(result[0], expected, delta=0.5)

    def test_shmr_off_center_subhalo(self):
        """SHMR should be invariant to translation of the subhalo."""
        # Shift all particles by a large offset
        offset = np.array([1000.0, 1000.0, 1000.0])
        shifted = SubhaloParticles({'star_p': self.pos + offset, 'star_m': self.mass}, column_map=None)
        original = SubhaloParticles({'star_p': self.pos,'star_m': self.mass}, column_map=None)

        result_shifted = FeatureComputer.compute('stellar_half_mass_radius', particles=shifted)
        result_original = FeatureComputer.compute('stellar_half_mass_radius', particles=original)
        self.assertAlmostEqual(result_shifted, result_original, places=5)

    def test_shmr_all_empty_subhalos(self):
        """All empty subhalos should return all nans."""
        empty_particles = SubhaloParticles({'star_p': np.array([]).reshape(0, 3),
                                                 'star_m': np.array([])}, column_map=None)

        result = FeatureComputer.compute('stellar_half_mass_radius', particles=empty_particles)
        self.assertTrue(np.all(np.isnan(result)))

