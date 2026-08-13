from rootutils import rootutils
rootutils.setup_root(__file__, pythonpath=True)


import unittest
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from src.data_generation.src.feature_computer import FeatureComputer
from src.data_generation.src.subhalo import SubhaloParticles
from src.data_generation.src.lazy_subhalo import LazySubhaloParticles


def make_subhalo_particles(pos, mass):
    return SubhaloParticles({'star_p': pos, 'star_m': mass}, column_map=None)


def make_lazy_subhalo_particles(pos, mass):
    """Build a LazySubhaloParticles from in-memory arrays without writing to disk."""
    table = pa.table({'star_p': pa.array(pos.tolist()), 'star_m': pa.array(mass.tolist())})
    dataset = ds.dataset(table, schema=table.schema)
    return LazySubhaloParticles(path=None, dataset=dataset)


class TestFeatureComputer(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n_particles = 1000
        self.R = 10.0

        # Correct uniform sphere sampling
        u = np.random.uniform(0, 1, n_particles)
        r = self.R * (u ** (1 / 3))
        theta = np.random.uniform(0, np.pi, n_particles)
        phi = np.random.uniform(0, 2 * np.pi, n_particles)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        self.pos = np.stack([x, y, z], axis=1)
        self.mass = np.ones(n_particles)

        self.mock_particles = make_subhalo_particles(self.pos, self.mass)
        self.mock_particles_lazy = make_lazy_subhalo_particles(self.pos, self.mass)

    # ── Registry ─────────────────────────────────────────────────────────────

    def test_debug_registry(self):
        import src.data_generation.src.features_dataset.shmr
        print("Registry:", FeatureComputer._registry)
        self.assertIn('stellar_half_mass_radius', FeatureComputer._registry)

    def test_registry_contains_stellar_half_mass_radius(self):
        """stellar_half_mass_radius should be registered."""
        self.assertIn('stellar_half_mass_radius', FeatureComputer._registry)

    def test_compute_unknown_method_raises(self):
        """Calling an unregistered method should raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            FeatureComputer.compute('non_existent_method')

    # ── SubhaloParticles ──────────────────────────────────────────────────────

    def test_compute_dispatches_correctly(self):
        """compute() should call the registered method and return a result."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles)
        self.assertIsNotNone(result)

    def test_returns_float(self):
        """Result should be a float."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles)
        self.assertIsInstance(result, float)

    def test_shmr_positive_value(self):
        """SHMR should be positive for valid particles."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles)
        self.assertGreater(result, 0)

    def test_shmr_uniform_sphere(self):
        """For a uniform sphere of radius R, SHMR ≈ R * (0.5)^(1/3) ≈ 0.794 * R."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles)
        expected = self.R * (0.5 ** (1 / 3))
        self.assertAlmostEqual(result, expected, delta=0.5)

    def test_shmr_translation_invariant(self):
        """SHMR should be invariant to translation of the subhalo."""
        offset = np.array([1000.0, 1000.0, 1000.0])
        original = make_subhalo_particles(self.pos,          self.mass)
        shifted = make_subhalo_particles(self.pos + offset, self.mass)

        result_original = FeatureComputer.compute('stellar_half_mass_radius', particles=original)
        result_shifted = FeatureComputer.compute('stellar_half_mass_radius', particles=shifted)
        self.assertAlmostEqual(result_original, result_shifted, places=5)

    def test_shmr_smaller_subhalo_has_smaller_radius(self):
        """Scaling positions by 0.5 should halve the SHMR."""
        original = make_subhalo_particles(self.pos,       self.mass)
        smaller = make_subhalo_particles(self.pos * 0.5, self.mass)

        result_original = FeatureComputer.compute('stellar_half_mass_radius', particles=original)
        result_smaller = FeatureComputer.compute('stellar_half_mass_radius', particles=smaller)
        self.assertAlmostEqual(result_smaller, result_original * 0.5, delta=result_original * 0.1)

    def test_shmr_single_particle(self):
        """Single particle should return radius 0 (it IS the center of mass)."""
        single = make_subhalo_particles(
            pos=np.array([[1.0, 2.0, 3.0]]),
            mass=np.array([1.0])
        )
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=single)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_shmr_empty_returns_nan(self):
        """Empty particle set should return nan."""
        empty=make_subhalo_particles(
            pos=np.array([]).reshape(0, 3),
            mass=np.array([])
        )
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=empty)
        self.assertTrue(np.isnan(result))

    # ── LazySubhaloParticles ──────────────────────────────────────────────────

    def test_lazy_compute_dispatches_correctly(self):
        """compute() should work with LazySubhaloParticles."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles_lazy)
        self.assertIsNotNone(result)

    def test_lazy_returns_float(self):
        """Result should be a float for LazySubhaloParticles."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles_lazy)
        self.assertIsInstance(result, float)

    def test_lazy_shmr_positive_value(self):
        """SHMR should be positive for valid lazy particles."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles_lazy)
        self.assertGreater(result, 0)

    def test_lazy_shmr_matches_eager(self):
        """LazySubhaloParticles and SubhaloParticles should return the same SHMR."""
        result_eager = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles)
        result_lazy = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles_lazy)
        self.assertAlmostEqual(result_eager, result_lazy, places=5)

    def test_lazy_shmr_uniform_sphere(self):
        """For a uniform sphere of radius R, SHMR ≈ R * (0.5)^(1/3) for lazy particles."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles_lazy)
        expected = self.R * (0.5 ** (1 / 3))
        self.assertAlmostEqual(result, expected, delta=0.5)

    def test_lazy_shmr_translation_invariant(self):
        """SHMR should be invariant to translation for lazy particles."""
        offset = np.array([1000.0, 1000.0, 1000.0])
        original = make_lazy_subhalo_particles(self.pos,          self.mass)
        shifted = make_lazy_subhalo_particles(self.pos + offset, self.mass)

        result_original = FeatureComputer.compute('stellar_half_mass_radius', particles=original)
        result_shifted = FeatureComputer.compute('stellar_half_mass_radius', particles=shifted)
        self.assertAlmostEqual(result_original, result_shifted, places=5)

    def test_lazy_shmr_empty_returns_nan(self):
        """Empty lazy particle set should return nan."""
        empty = make_lazy_subhalo_particles(
            pos=np.array([]).reshape(0, 3),
            mass=np.array([])
        )
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=empty)
        self.assertTrue(np.isnan(result))

    def test_lazy_does_not_materialize_full_table(self):
        """data should remain None after SHMR — materialize() should never be called."""
        result = FeatureComputer.compute('stellar_half_mass_radius', particles=self.mock_particles_lazy)
        self.assertIsNone(self.mock_particles_lazy.data)

    def test_lazy_respects_filters(self):
        """Filters registered on LazySubhaloParticles should be applied before SHMR."""
        pos_a = self.pos
        pos_b = self.pos * 0.1  # much smaller cluster
        mass_a = self.mass
        mass_b = self.mass

        table = pa.table({
            'star_p': pa.array(np.vstack([pos_a, pos_b]).tolist()),
            'star_m': pa.array(np.concatenate([mass_a, mass_b]).tolist()),
            'parent_id': pa.array([0] * len(pos_a) + [1] * len(pos_b))
        })
        dataset = ds.dataset(table, schema=table.schema)
        particles = LazySubhaloParticles(path=None, dataset=dataset)
        particles.add_filter(ds.field('parent_id') == 0)

        result_filtered = FeatureComputer.compute('stellar_half_mass_radius', particles=particles)
        result_full = FeatureComputer.compute('stellar_half_mass_radius',
                                                  particles=make_lazy_subhalo_particles(pos_a, mass_a))
        self.assertAlmostEqual(result_filtered, result_full, places=4)

