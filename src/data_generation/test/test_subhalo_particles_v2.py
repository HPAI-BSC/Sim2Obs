"""
Unit tests for SubhaloParticlesV2 (src/data_generation/src/subhalo_particles_v2.py).

They use small synthetic particles built in memory. `SubhaloParticles` is only used as the reference for
star-only parity, it is not modified. Rotation is not covered because V2 does not implement it yet.
"""
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import pickle
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from omegaconf import OmegaConf

from src.data_generation.src.feature import ParticleFieldProcessor
from src.data_generation.src.subhalo import SubhaloParticles
from src.data_generation.src.subhalo_particles_v2 import SubhaloParticlesV2

LOGGER = 'Subhalo-Particles-V2'


def flat(cls=SubhaloParticlesV2, **arrays):
    """Build particles from flat attribute arrays, e.g. star_p=..., dm_m=..."""
    return cls(data=arrays, column_map=None)


def stars(n, offset=0.0, cls=SubhaloParticlesV2):
    """n star particles with distinguishable values."""
    return flat(cls, star_p=np.arange(n * 3, dtype=float).reshape(n, 3) + offset, star_m=np.arange(n) + 1.0 + offset)


def dark_matter(n, offset=0.0):
    """n dark matter particles with distinguishable values."""
    return {'dm_p': -np.arange(n * 3, dtype=float).reshape(n, 3) - offset, 'dm_m': np.full(n, 5.0) + offset}


def halo_at(*centre, halo_id=7):
    """Minimal stand-in for a Halo: preprocess only needs gpos (and optionally halo_id)."""
    return SimpleNamespace(gpos=np.array(centre, dtype=float), halo_id=halo_id)


class TestConstruction(unittest.TestCase):
    def test_nested_data_with_a_new_field_gives_the_named_attributes(self):
        data = {
            'star': pd.DataFrame({'Coordinates': [[1., 2., 3.], [4., 5., 6.]], 'Masses': [1., 2.],
                                  'GFM_Metallicity': [0.01, 0.02]}),
            'dm': pd.DataFrame({'Coordinates': [[0., 0., 0.]], 'Masses': [9.]}),
        }
        column_map = OmegaConf.create({
            'star': {'Coordinates': {'name': 'p'}, 'Masses': {'name': 'm'}, 'GFM_Metallicity': {'name': 'metallicity'}},
            'dm': {'Coordinates': {'name': 'p'}, 'Masses': {'name': 'm'}},
        })
        particles = SubhaloParticlesV2(data, column_map)

        for attribute in ('star_p', 'star_m', 'star_metallicity', 'dm_p', 'dm_m'):
            self.assertTrue(hasattr(particles, attribute), attribute)
        assert_allclose(particles.star_metallicity, [0.01, 0.02])
        self.assertEqual(particles.star_p.shape, (2, 3))
        self.assertEqual(particles.particle_types, ['star', 'dm'])
        self.assertEqual(particles.position_attributes, ['star_p', 'dm_p'])
        self.assertIsNone(particles.overlapping_particles_id)
        self.assertIsNone(particles.overlapping_boundaries)

    def test_particle_type_aliases_are_applied(self):
        data = {'stars': pd.DataFrame({'Masses': [1.]}), 'dark_matter': pd.DataFrame({'Masses': [2.]})}
        column_map = OmegaConf.create({'stars': {'Masses': {'name': 'm'}}, 'dark_matter': {'Masses': {'name': 'm'}}})
        particles = SubhaloParticlesV2(data, column_map)
        self.assertTrue(hasattr(particles, 'star_m'))
        self.assertTrue(hasattr(particles, 'dm_m'))

    def test_declared_field_missing_from_the_data_logs_a_warning(self):
        data = {'star': pd.DataFrame({'Coordinates': [[1., 2., 3.]], 'Masses': [1.]})}
        column_map = OmegaConf.create({'star': {'Coordinates': {'name': 'p'}, 'Masses': {'name': 'm'},
                                                'GFM_Metallicity': {'name': 'metallicity'}}})
        with self.assertLogs(LOGGER, level='WARNING') as logs:
            particles = SubhaloParticlesV2(data, column_map)
        self.assertIn('star_metallicity', logs.output[0])
        self.assertFalse(hasattr(particles, 'star_metallicity'))

    def test_no_warning_when_a_whole_particle_type_is_absent_or_data_is_empty(self):
        column_map = OmegaConf.create({'star': {'Masses': {'name': 'm'}}, 'dm': {'Masses': {'name': 'm'}}})
        with self.assertNoLogs(LOGGER, level='WARNING'):
            SubhaloParticlesV2({'star': pd.DataFrame({'Masses': [1.]})}, column_map)   # no dm in this subhalo
            SubhaloParticlesV2({}, column_map)                                          # nothing loaded

    def test_flat_data_constructor_of_the_parent_still_works(self):
        particles = stars(3)
        self.assertEqual(len(particles), 3)
        self.assertFalse(particles.is_empty)


class TestOverlap(unittest.TestCase):
    def test_each_particle_type_is_cut_at_its_own_boundary(self):
        # Own halo: 3 stars and 5 dm. Neighbour: 2 stars and 4 dm. A single star boundary (3) would cut dm at 3.
        own = flat(**{**{'star_p': stars(3).star_p, 'star_m': stars(3).star_m}, **dark_matter(5)})
        neighbour = flat(**{**{'star_p': stars(2, 100).star_p, 'star_m': stars(2, 100).star_m}, **dark_matter(4, 100)})
        combined = own + neighbour

        self.assertEqual(len(combined.star_m), 5)
        self.assertEqual(len(combined.dm_m), 9)
        self.assertEqual(combined.overlapping_boundaries, {'star': 3, 'dm': 5})
        self.assertEqual(combined.overlapping_particles_id, 3)

        own_only = combined.non_overlapping
        self.assertEqual(len(own_only.star_m), 3)
        self.assertEqual(len(own_only.dm_m), 5)
        assert_array_equal(own_only.dm_p, own.dm_p)
        assert_array_equal(own_only.star_p, own.star_p)

    def test_the_existing_class_cuts_every_type_at_the_star_boundary(self):
        # Documents why V2 exists: SubhaloParticles has one boundary for all types. Delete once merged.
        own = SubhaloParticles(data={**{'star_p': stars(3).star_p, 'star_m': stars(3).star_m}, **dark_matter(5)}, column_map=None)
        neighbour = SubhaloParticles(data={**{'star_p': stars(2, 100).star_p, 'star_m': stars(2, 100).star_m}, **dark_matter(4, 100)}, column_map=None)
        own_only = (own + neighbour).non_overlapping
        self.assertEqual(len(own_only.dm_m), 3)      # 2 own dark matter particles are lost

    def test_neighbour_with_only_stars_adds_no_dark_matter(self):
        own = flat(**{**{'star_p': stars(3).star_p, 'star_m': stars(3).star_m}, **dark_matter(5)})
        neighbour = stars(2, 100)
        combined = own + neighbour

        self.assertEqual(len(combined.star_m), 5)
        self.assertEqual(len(combined.dm_m), 5)
        self.assertEqual(combined.overlapping_boundaries, {'star': 3, 'dm': 5})
        own_only = combined.non_overlapping
        self.assertEqual(len(own_only.star_m), 3)
        self.assertEqual(len(own_only.dm_m), 5)

    def test_a_type_that_only_the_other_object_has_is_kept_with_boundary_zero(self):
        combined = stars(3) + flat(**dark_matter(4))
        self.assertEqual(len(combined.dm_m), 4)
        self.assertEqual(combined.overlapping_boundaries, {'star': 3, 'dm': 0})
        own_only = combined.non_overlapping
        self.assertEqual(len(own_only.star_m), 3)
        self.assertEqual(len(own_only.dm_m), 0)

    def test_adding_to_a_particle_set_that_has_no_stars_keeps_the_other_types(self):
        # The parent class loses arrays that only the second operand has
        combined = flat(**dark_matter(2)) + stars(3)
        self.assertEqual(len(combined.star_m), 3)
        self.assertEqual(len(combined.dm_m), 2)

    def test_the_first_boundary_is_kept_when_more_neighbours_are_added(self):
        own = stars(3)
        for chain in (lambda: (own + stars(2, 10)) + stars(4, 20),            # sequential, as in a loop of +=
                      lambda: own + (stars(2, 10) + stars(4, 20))):           # neighbours merged first
            with self.subTest():
                combined = chain()
                self.assertEqual(len(combined.star_m), 9)
                self.assertEqual(len(combined.non_overlapping.star_m), 3)
                own.overlapping_particles_id = None       # a + b records the boundary on a itself
                own.overlapping_boundaries = None

    def test_resetting_the_overlap_id_to_none_means_no_overlap(self):
        # The fetchers do this after merging the subhalos of a halo
        merged = stars(3) + stars(2, 10)
        merged.overlapping_particles_id = None
        self.assertIs(merged.non_overlapping, merged)

        again = merged + stars(4, 20)               # a later addition records a fresh boundary
        self.assertEqual(again.overlapping_boundaries, {'star': 5})
        self.assertEqual(len(again.non_overlapping.star_m), 5)

    def test_a_legacy_integer_boundary_without_per_type_boundaries_cuts_every_type(self):
        particles = flat(**{**{'star_p': stars(5).star_p, 'star_m': stars(5).star_m}})
        particles.overlapping_particles_id = 3
        self.assertEqual(len(particles.non_overlapping.star_m), 3)

    def test_non_overlapping_is_idempotent(self):
        combined = stars(3) + stars(2, 10)
        once = combined.non_overlapping
        twice = once.non_overlapping
        assert_array_equal(once.star_m, twice.star_m)

    def test_operands_arrays_are_not_modified(self):
        own, neighbour = stars(3), stars(2, 10)
        before_own, before_neighbour = own.star_m.copy(), neighbour.star_m.copy()
        combined = own + neighbour
        assert_array_equal(own.star_m, before_own)
        assert_array_equal(neighbour.star_m, before_neighbour)
        self.assertEqual(len(combined.star_m), 5)

    def test_the_object_is_a_subhalo_particles_and_accepts_the_parent_class(self):
        combined = stars(3) + stars(2, 10, cls=SubhaloParticles)
        self.assertIsInstance(combined, SubhaloParticles)
        self.assertIsInstance(combined, SubhaloParticlesV2)
        self.assertEqual(len(combined.star_m), 5)
        self.assertIs(SubhaloParticlesV2.__add__(stars(1), 'not particles'), NotImplemented)

    def test_star_only_results_match_the_existing_class(self):
        old_own, old_neighbour = stars(4, cls=SubhaloParticles), stars(3, 50, cls=SubhaloParticles)
        new_own, new_neighbour = stars(4), stars(3, 50)
        old, new = old_own + old_neighbour, new_own + new_neighbour

        self.assertEqual(new.overlapping_particles_id, old.overlapping_particles_id)
        self.assertEqual(old_own.overlapping_particles_id, new_own.overlapping_particles_id)   # side effect on the left operand
        for attribute in ('star_p', 'star_m'):
            assert_array_equal(getattr(new, attribute), getattr(old, attribute))
            assert_array_equal(getattr(new.non_overlapping, attribute), getattr(old.non_overlapping, attribute))


class TestPreprocess(unittest.TestCase):
    def test_every_position_array_is_recentred_and_scalars_are_untouched(self):
        particles = flat(star_p=np.array([[11., 22., 33.]]), star_m=np.array([4.]),
                         dm_p=np.array([[12., 24., 36.], [10., 20., 30.]]), dm_m=np.array([1., 1.]))
        particles.preprocess(halo_at(10., 20., 30.))

        assert_allclose(particles.star_p, [[1., 2., 3.]])
        assert_allclose(particles.dm_p, [[2., 4., 6.], [0., 0., 0.]])
        assert_array_equal(particles.star_m, [4.])
        assert_array_equal(particles.dm_m, [1., 1.])

    def test_periodic_boundaries_are_corrected_for_every_type(self):
        lbox = 100.0
        particles = flat(star_p=np.array([[98., 50., 2.]]), dm_p=np.array([[1., 99., 50.]]))
        particles.preprocess(halo_at(2., 2., 2.), lbox)
        # star: dx 96 -> -4, dy 48 stays, dz 0 ; dm: dx -1 stays, dy 97 -> -3, dz 48 stays
        assert_allclose(particles.star_p, [[-4., 48., 0.]])
        assert_allclose(particles.dm_p, [[-1., -3., 48.]])

    def test_without_lbox_there_is_no_periodic_correction(self):
        particles = flat(star_p=np.array([[98., 50., 2.]]))
        particles.preprocess(halo_at(2., 2., 2.))
        assert_allclose(particles.star_p, [[96., 48., 0.]])

    def test_object_arrays_of_vectors_are_stacked(self):
        positions = np.empty(2, dtype=object)
        positions[0], positions[1] = np.array([1., 1., 1.]), np.array([2., 2., 2.])
        particles = flat(dm_p=positions, dm_m=np.array([1., 1.]))
        particles.preprocess(halo_at(1., 1., 1.))
        assert_allclose(particles.dm_p, [[0., 0., 0.], [1., 1., 1.]])

    def test_particles_without_positions_log_a_warning_instead_of_failing(self):
        particles = flat(dm_m=np.array([1.]))
        with self.assertLogs(LOGGER, level='WARNING'):
            particles.preprocess(halo_at(0., 0., 0.))       # SubhaloParticles raises AttributeError here

    def test_star_only_results_match_the_existing_class(self):
        for lbox in (None, 100.0):
            with self.subTest(lbox=lbox):
                old = SubhaloParticles(data={'star_p': np.array([[98., 50., 2.], [3., 4., 5.]]), 'star_m': np.array([1., 2.])}, column_map=None)
                new = flat(star_p=np.array([[98., 50., 2.], [3., 4., 5.]]), star_m=np.array([1., 2.]))
                halo = halo_at(2., 2., 2.)
                old.preprocess(halo, lbox)
                new.preprocess(halo, lbox)
                assert_array_equal(new.star_p, old.star_p)
                self.assertEqual(new.star_p.dtype, old.star_p.dtype)


class TestHardening(unittest.TestCase):
    def test_class_level_default_for_objects_without_the_instance_attribute(self):
        # What an old pickle looks like: the state has no 'overlapping_boundaries'
        self.assertIsNone(SubhaloParticlesV2.overlapping_boundaries)
        particles = stars(3)
        del particles.__dict__['overlapping_boundaries']
        self.assertIsNone(particles.overlapping_boundaries)
        combined = particles + stars(2, 10)
        self.assertEqual(combined.overlapping_boundaries, {'star': 3})
        self.assertEqual(len(combined.non_overlapping.star_m), 3)

    def test_pickle_round_trip_keeps_the_object_and_its_boundaries(self):
        combined = stars(3) + flat(**dark_matter(2))
        restored = pickle.loads(pickle.dumps(combined))
        self.assertIsInstance(restored, SubhaloParticlesV2)
        self.assertEqual(restored.overlapping_boundaries, combined.overlapping_boundaries)
        self.assertEqual(len(restored.non_overlapping.star_m), 3)

    def test_unpickling_a_state_saved_without_boundaries_works(self):
        particles = stars(4)
        state = {key: value for key, value in particles.__dict__.items() if key != 'overlapping_boundaries'}
        restored = SubhaloParticlesV2.__new__(SubhaloParticlesV2)
        restored.__dict__.update(state)
        self.assertIsNone(restored.overlapping_boundaries)
        self.assertEqual(len(restored), 4)
        self.assertIs(restored.non_overlapping, restored)

    def test_legacy_object_with_only_the_integer_boundary_can_be_added_to(self):
        # Integer boundary set, no per-type boundaries: the old single-boundary behaviour is kept
        legacy = stars(5)
        legacy.overlapping_particles_id = 3
        combined = legacy + stars(2, 10)
        self.assertEqual(len(combined.star_m), 7)
        self.assertIsNone(combined.overlapping_boundaries)
        self.assertEqual(len(combined.non_overlapping.star_m), 3)

    def test_rotate_works_for_star_only_particles_like_the_existing_class(self):
        old = SubhaloParticles(data={'star_p': np.arange(9, dtype=float).reshape(3, 3), 'star_m': np.ones(3)}, column_map=None)
        new = flat(star_p=np.arange(9, dtype=float).reshape(3, 3), star_m=np.ones(3))
        axis = np.array([0, 0, 1])
        np.random.seed(11)
        old.rotate(axis, 30)
        np.random.seed(11)
        new.rotate(axis, 30)
        assert_array_equal(new.star_p, old.star_p)
        self.assertEqual(new.star_p.dtype, np.float32)

    def test_rotate_refuses_other_position_arrays_and_leaves_the_data_untouched(self):
        particles = flat(star_p=np.arange(9, dtype=float).reshape(3, 3), star_m=np.ones(3), **dark_matter(2))
        star_before, dm_before = particles.star_p.copy(), particles.dm_p.copy()
        with self.assertRaises(NotImplementedError) as context:
            particles.rotate(np.array([0, 0, 1]), 45)
        self.assertIn('dm_p', str(context.exception))
        assert_array_equal(particles.star_p, star_before)
        assert_array_equal(particles.dm_p, dm_before)

        dm_only = flat(**dark_matter(2))
        with self.assertRaises(NotImplementedError):
            dm_only.rotate(np.array([0, 0, 1]), 45)

    def test_rotate_by_zero_degrees_is_allowed_with_several_particle_types(self):
        # rotate_step uses 0 degrees for its first iteration
        particles = flat(star_p=np.arange(9, dtype=float).reshape(3, 3), star_m=np.ones(3), **dark_matter(2))
        before = particles.star_p.copy()
        particles.rotate(np.array([0, 0, 1]), 0)
        assert_array_equal(particles.star_p, before)


class TestWithParticleField(unittest.TestCase):
    def test_star_and_dark_matter_maps_of_the_same_halo(self):
        ranging, bins = ((-10.0, 10.0), (-10.0, 10.0)), 2
        own = flat(star_p=np.array([[-5., -5., 0.], [5., 5., 0.]]), star_m=np.array([1., 1.]),
                   star_metallicity=np.array([0.02, 0.01]),
                   dm_p=np.array([[-6., -4., 0.], [-3., -2., 0.], [7., 8., 0.]]), dm_m=np.array([2., 3., 4.]))
        neighbour = flat(star_p=np.array([[5., -5., 0.]]), star_m=np.array([100.]), star_metallicity=np.array([0.05]))
        combined = own + neighbour

        metallicity = ParticleFieldProcessor(OmegaConf.create(
            {'attribute': 'star_metallicity', 'weight': 'star_m', 'statistic': 'weighted_mean'}))
        dark_matter_mass = ParticleFieldProcessor(OmegaConf.create({'attribute': 'dm_m', 'statistic': 'sum'}))

        z_map = metallicity.generate_map(None, None, combined, bins, ranging, axis=(0, 1))[0]
        m_map = dark_matter_mass.generate_map(None, None, combined, bins, ranging, axis=(0, 1))[0]

        # Default: the halo's own particles only. The heavy neighbour star in pixel (1, 0) is excluded
        self.assertEqual(z_map[1, 0], 0.0)
        self.assertAlmostEqual(z_map[0, 0], 0.02)
        # All the own dark matter is kept, although there are more dm than star own particles
        self.assertAlmostEqual(m_map.sum(), 9.0)
        self.assertAlmostEqual(m_map[0, 0], 5.0)
        self.assertAlmostEqual(m_map[1, 1], 4.0)


if __name__ == '__main__':
    unittest.main()
