"""
Unit tests for ParticleFieldProcessor (src/data_generation/src/feature.py).

They use small synthetic SubhaloParticles built in memory, so they need no simulation data.
"""
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from omegaconf import OmegaConf

from src.data_generation.src.feature import FeatureProcessorFactory, ParticleFieldProcessor
from src.data_generation.src.subhalo import SubhaloParticles

# A 2x2 map over [-10, 10] x [-10, 10]: pixel (0, 0) is x<0 and y<0, pixel (1, 1) is x>0 and y>0
RANGING = ((-10.0, 10.0), (-10.0, 10.0))
BINS = 2


def make_particles(**arrays):
    """Build a SubhaloParticles from flat attribute arrays, e.g. star_p=..., star_z=..."""
    return SubhaloParticles(data=arrays, column_map=None)


def make_processor(**config):
    """Build a ParticleFieldProcessor from keyword config entries."""
    return ParticleFieldProcessor(kwargs=OmegaConf.create(config))


class TestParticleFieldMaps(unittest.TestCase):
    def test_weighted_mean_uses_the_weights(self):
        # Two stars in the same pixel: mass 1 with Z=0.02 and mass 9 with Z=0.01
        particles = make_particles(star_p=np.array([[-5., -5., 0.], [-4., -6., 1.]]),
                                   star_m=np.array([1., 9.]), star_z=np.array([0.02, 0.01]))
        processor = make_processor(attribute='star_z', weight='star_m', statistic='weighted_mean')
        output, binx, biny = processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1))

        self.assertAlmostEqual(output[0, 0], 0.011)   # (1*0.02 + 9*0.01) / 10, not the plain mean 0.015
        self.assertEqual(output.shape, (2, 2))
        assert_allclose(binx, [-10, 0, 10])
        assert_allclose(biny, [-10, 0, 10])

    def test_unweighted_mean_ignores_the_weights(self):
        particles = make_particles(star_p=np.array([[-5., -5., 0.], [-4., -6., 1.]]),
                                   star_m=np.array([1., 9.]), star_z=np.array([0.02, 0.01]))
        processor = make_processor(attribute='star_z', statistic='mean')
        output, _, _ = processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1))
        self.assertAlmostEqual(output[0, 0], 0.015)

    def test_sum_conserves_the_total_mass(self):
        rng = np.random.default_rng(0)
        positions = rng.uniform(-9, 9, size=(500, 3))
        masses = rng.uniform(1, 2, size=500)
        particles = make_particles(dm_p=positions, dm_m=masses)
        processor = make_processor(attribute='dm_m', statistic='sum')

        output, _, _ = processor.generate_map(None, None, particles, 8, RANGING, axis=(0, 1))
        self.assertAlmostEqual(output.sum(), masses.sum())

    def test_empty_pixels_are_zero(self):
        particles = make_particles(star_p=np.array([[-5., -5., 0.]]), star_m=np.array([2.]),
                                   star_z=np.array([0.03]))
        for config in ({'statistic': 'weighted_mean', 'weight': 'star_m'}, {'statistic': 'mean'}, {'statistic': 'sum'}):
            with self.subTest(config=config):
                processor = make_processor(attribute='star_z', **config)
                output, _, _ = processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1))
                self.assertFalse(np.isnan(output).any())
                assert_array_equal(output[[0, 1, 1], [1, 0, 1]], 0)   # the three empty pixels

    def test_projection_axes_select_the_right_coordinates(self):
        # x = -5, y = 5, z = 5: falls in pixel (0, 1) of xy, (1, 1) of yz and (0, 1) of xz
        particles = make_particles(dm_p=np.array([[-5., 5., 5.]]), dm_m=np.array([3.]))
        processor = make_processor(attribute='dm_m', statistic='sum')
        for axis, pixel in (((0, 1), (0, 1)), ((1, 2), (1, 1)), ((0, 2), (0, 1))):
            with self.subTest(axis=axis):
                output, _, _ = processor.generate_map(None, None, particles, BINS, RANGING, axis=axis)
                self.assertEqual(output[pixel], 3.)
                self.assertEqual(output.sum(), 3.)

    def test_star_and_dm_maps_share_the_same_pixel_grid(self):
        # Different particle types: different counts and positions, same ranging/bins
        particles = make_particles(
            star_p=np.array([[-5., -5., 0.], [5., 5., 0.]]), star_m=np.array([1., 1.]), star_z=np.array([0.02, 0.01]),
            dm_p=np.array([[-6., -4., 0.], [-3., -2., 0.], [7., 8., 0.]]), dm_m=np.array([2., 3., 4.]))
        metallicity = make_processor(attribute='star_z', weight='star_m', statistic='weighted_mean')
        dark_matter = make_processor(attribute='dm_m', statistic='sum')

        z_map, z_binx, z_biny = metallicity.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1))
        m_map, m_binx, m_biny = dark_matter.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1))

        self.assertEqual(z_map.shape, m_map.shape)
        assert_allclose(z_binx, m_binx)
        assert_allclose(z_biny, m_biny)
        self.assertAlmostEqual(z_map[0, 0], 0.02)
        self.assertAlmostEqual(m_map[0, 0], 5.)    # dm masses 2 + 3 in the first pixel, the star mass is not used
        self.assertAlmostEqual(m_map[1, 1], 4.)

    def test_per_area_divides_by_the_pixel_area(self):
        particles = make_particles(dm_p=np.array([[-5., -5., 0.]]), dm_m=np.array([100.]))
        processor = make_processor(attribute='dm_m', statistic='sum', per_area=True)
        output, _, _ = processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1))
        self.assertAlmostEqual(output[0, 0], 100. / (10. * 10.))

    def test_float32_values_are_accumulated_in_float64(self):
        # 10_000 particles of 0.1 in float32: a float32 accumulation would drift from 1000
        particles = make_particles(dm_p=np.full((10_000, 3), 5., dtype=np.float32),
                                   dm_m=np.full(10_000, 0.1, dtype=np.float32))
        processor = make_processor(attribute='dm_m', statistic='sum')
        output, _, _ = processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1))
        self.assertEqual(output.dtype, np.float64)
        self.assertAlmostEqual(output[1, 1], 10_000 * float(np.float32(0.1)), places=6)

    def test_object_arrays_of_vectors_are_stacked(self):
        # Parquet list columns come back as object arrays of per-particle vectors
        positions = np.empty(2, dtype=object)
        positions[0], positions[1] = np.array([-5., -5., 0.]), np.array([5., 5., 0.])
        particles = make_particles(dm_p=positions, dm_m=np.array([1., 2.]))
        processor = make_processor(attribute='dm_m', statistic='sum')
        output, _, _ = processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1))
        self.assertEqual(output[0, 0], 1.)
        self.assertEqual(output[1, 1], 2.)

    def test_map_step_call_forms_are_accepted(self):
        # map_step calls the input feature with muse_filter and the output feature without it
        particles = make_particles(star_p=np.array([[-5., -5., 0.]]), star_m=np.array([1.]), star_z=np.array([0.02]))
        processor = make_processor(attribute='star_z', statistic='mean')
        with_muse_kwarg = processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1), muse_filter=False)
        with_muse_none = processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1), muse_filter=None)
        without_kwarg = processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1))
        for result in (with_muse_none, without_kwarg):
            assert_array_equal(result[0], with_muse_kwarg[0])

    def test_unsupported_modes_raise(self):
        particles = make_particles(star_p=np.array([[-5., -5., 0.]]), star_z=np.array([0.02]))
        processor = make_processor(attribute='star_z', statistic='mean')
        with self.assertRaises(NotImplementedError):
            processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1), muse_filter=True)
        with self.assertRaises(NotImplementedError):
            processor.generate_map(None, None, particles, BINS, RANGING, axis=None)


class TestParticleFieldData(unittest.TestCase):
    def test_clip_min_removes_negative_values(self):
        particles = make_particles(star_p=np.array([[-5., -5., 0.], [-4., -4., 0.]]), star_m=np.array([1., 1.]),
                                   star_z=np.array([-2.8e-6, 0.02]))
        clipped = make_processor(attribute='star_z', weight='star_m', statistic='weighted_mean', clip_min=0.0)
        unclipped = make_processor(attribute='star_z', weight='star_m', statistic='weighted_mean')

        self.assertAlmostEqual(clipped.generate_map(None, None, particles, BINS, RANGING, (0, 1))[0][0, 0], 0.01)
        self.assertLess(unclipped.generate_map(None, None, particles, BINS, RANGING, (0, 1))[0][0, 0], 0.01)

    def test_missing_attribute_lists_the_available_ones(self):
        particles = make_particles(star_p=np.array([[0., 0., 0.]]), star_m=np.array([1.]))
        processor = make_processor(attribute='star_z', statistic='mean')
        with self.assertRaises(AttributeError) as context:
            processor.process(None, particles)
        message = str(context.exception)
        self.assertIn('star_z', message)
        self.assertIn('star_m', message)
        self.assertIn('star_p', message)

    def test_missing_positions_of_the_particle_type_raise(self):
        # dm_m exists but there are no dm_p positions
        particles = make_particles(star_p=np.array([[0., 0., 0.]]), dm_m=np.array([1.]))
        with self.assertRaises(AttributeError) as context:
            make_processor(attribute='dm_m', statistic='sum').process(None, particles)
        self.assertIn('dm_p', str(context.exception))

    def test_different_lengths_raise(self):
        particles = make_particles(star_p=np.zeros((3, 3)), star_z=np.array([0.01, 0.02]))
        with self.assertRaises(ValueError):
            make_processor(attribute='star_z', statistic='mean').process(None, particles)

    def test_non_scalar_attribute_raises(self):
        particles = make_particles(star_p=np.zeros((2, 3)), star_v=np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            make_processor(attribute='star_v', statistic='mean').process(None, particles)

    def test_empty_particles_raise(self):
        particles = make_particles(star_p=np.zeros((0, 3)), star_z=np.zeros(0))
        with self.assertRaises(ValueError):
            make_processor(attribute='star_z', statistic='mean').process(None, particles)

    def test_non_finite_values_are_dropped(self):
        particles = make_particles(dm_p=np.array([[-5., -5., 0.], [-4., -4., 0.], [5., 5., 0.]]),
                                   dm_m=np.array([1., np.nan, 2.]))
        processor = make_processor(attribute='dm_m', statistic='sum')
        output, _, _ = processor.generate_map(None, None, particles, BINS, RANGING, axis=(0, 1))
        self.assertEqual(output[0, 0], 1.)
        self.assertEqual(output[1, 1], 2.)

        all_nan = make_particles(dm_p=np.zeros((2, 3)), dm_m=np.array([np.nan, np.nan]))
        with self.assertRaises(ValueError):
            processor.process(None, all_nan)

    def test_process_stores_the_per_particle_values(self):
        particles = make_particles(star_p=np.zeros((3, 3)), star_z=np.array([0.01, 0.02, 0.03], dtype=np.float32))
        processor = make_processor(attribute='star_z', statistic='mean')
        processor.process(None, particles)
        self.assertEqual(processor.value.dtype, np.float64)
        assert_allclose(processor.value, [0.01, 0.02, 0.03], rtol=1e-6)

    def test_rotate_is_a_noop_for_scalars(self):
        particles = make_particles(star_p=np.zeros((3, 3)), star_z=np.array([0.01, 0.02, 0.03]))
        processor = make_processor(attribute='star_z', statistic='mean')
        processor.process(None, particles)
        before = processor.value.copy()
        processor.rotate([0, 0, 1], 45)      # the base class version would fail on a 1D array
        assert_array_equal(processor.value, before)

    def test_overlapping_particles_are_excluded_by_default(self):
        own = make_particles(dm_p=np.array([[-5., -5., 0.], [-4., -4., 0.]]), dm_m=np.array([1., 1.]))
        neighbour = make_particles(dm_p=np.array([[5., 5., 0.]]), dm_m=np.array([10.]))
        combined = own + neighbour           # records where the own particles end

        default = make_processor(attribute='dm_m', statistic='sum')
        everything = make_processor(attribute='dm_m', statistic='sum', include_overlapping=True)

        own_only = default.generate_map(None, None, combined, BINS, RANGING, axis=(0, 1))[0]
        full = everything.generate_map(None, None, combined, BINS, RANGING, axis=(0, 1))[0]
        self.assertEqual(own_only.sum(), 2.)
        self.assertEqual(full.sum(), 12.)
        self.assertEqual(full[1, 1], 10.)


class TestParticleFieldConfiguration(unittest.TestCase):
    def test_name_defaults_to_the_attribute_and_label_overrides_it(self):
        self.assertEqual(make_processor(attribute='star_z', statistic='mean').name, 'star_z')
        self.assertEqual(make_processor(attribute='star_z', statistic='mean', label='StellarMetallicity').name,
                         'StellarMetallicity')

    def test_particle_type_and_positions_come_from_the_attribute_prefix(self):
        processor = make_processor(attribute='dm_m')
        self.assertEqual(processor.particle_type, 'dm')
        self.assertEqual(processor.position_attribute, 'dm_p')
        self.assertEqual(processor.statistic, 'sum')

    def test_invalid_configuration_raises(self):
        invalid = {
            'no attribute': {'statistic': 'sum'},
            'attribute without particle type': {'attribute': 'metallicity'},
            'empty field name': {'attribute': 'star_'},
            'unknown statistic': {'attribute': 'star_z', 'statistic': 'median'},
            'weighted_mean without weight': {'attribute': 'star_z', 'statistic': 'weighted_mean'},
            'weight of another particle type': {'attribute': 'star_z', 'statistic': 'weighted_mean', 'weight': 'dm_m'},
            'per_area with mean': {'attribute': 'star_z', 'statistic': 'mean', 'per_area': True},
        }
        for label, config in invalid.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    make_processor(**config)

    def test_weight_is_ignored_when_the_statistic_does_not_use_it(self):
        processor = make_processor(attribute='star_z', statistic='mean', weight='star_m')
        self.assertIsNone(processor.weight)

    def test_get_empty(self):
        self.assertEqual(len(ParticleFieldProcessor.get_empty()), 0)


class TestParticleFieldFactory(unittest.TestCase):
    def test_factory_builds_a_particle_field_processor(self):
        feature = OmegaConf.create({'name': 'ParticleField', 'attribute': 'star_z', 'weight': 'star_m',
                                    'statistic': 'weighted_mean', 'label': 'StellarMetallicity'})
        processor = FeatureProcessorFactory.get_feature(feature, cfg=None)
        self.assertIsInstance(processor, ParticleFieldProcessor)
        self.assertEqual(processor.name, 'StellarMetallicity')
        self.assertEqual(processor.weight, 'star_m')

    def test_factory_builds_an_input_and_an_output_feature_with_distinct_names(self):
        features = OmegaConf.create([
            {'name': 'ParticleField', 'attribute': 'star_z', 'weight': 'star_m', 'statistic': 'weighted_mean'},
            {'name': 'ParticleField', 'attribute': 'dm_m', 'statistic': 'sum'},
        ])
        input_feature, output_feature = (FeatureProcessorFactory.get_feature(f, cfg=None) for f in features)
        self.assertNotEqual(input_feature.name, output_feature.name)

    def test_the_removed_mass_density_features_are_no_longer_registered(self):
        for name in ('StellarMassDensity', 'DarkMatterMassDensity'):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    FeatureProcessorFactory.get_feature(OmegaConf.create({'name': name}), cfg=None)


if __name__ == '__main__':
    unittest.main()
