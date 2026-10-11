"""
Tests for the catalog variants of the use case "dark matter mass from stellar metallicity":
configs/catalog/illustris300_dm.yaml and configs/catalog/millenium_dm.yaml.

The variants inherit the original catalogs through Hydra and only add particle fields and the `particles_v2`
flag, so these tests check that the originals are untouched, that nothing but the new fields differs, and that
the new fields are the ones the fetcher and the ParticleField feature rely on.
"""
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import unittest

import math
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import illustris_python as il
import numpy as np
from hydra import compose, initialize
from numpy.testing import assert_allclose
from omegaconf import OmegaConf

from src.data_generation.src.data_fetcher import DataFetcherFactory, Illustris300Fetcher
from src.data_generation.src.subhalo import SubhaloParticles
from src.data_generation.src.subhalo_particles_v2 import SubhaloParticlesV2
from src.data_generation.test.test_fetcher_illustris_multitype import FakeSnapshot

PAIRS = {
    # variant: (base, expected position scaling, particle types of the variant)
    'illustris300_dm': ('illustris300', 'x * a', {'star', 'dm'}),
    'millenium_dm': ('millenium', 'x * a * 1000', {'star', 'gas', 'dm'}),
}
NEW_STAR_FIELDS = {'GFM_Metallicity', 'GFM_StellarFormationTime'}
PATH_SUFFIX = '_dm_v1'      # version of the dataset directory of the variants
# The fetcher picks the subhalo parent field from these substrings of catalog.path, the new names must keep them
PATH_MARKER = {'illustris300_dm': 'illustris300', 'millenium_dm': 'millenium'}


def load_catalog(name, datasets_dir=None):
    overrides = [f"catalog={name}"] + ([f"paths.datasets_dir={datasets_dir}"] if datasets_dir else [])
    with initialize(config_path="../../../configs", version_base=None):
        return compose(config_name="data_generation.yaml", overrides=overrides).catalog


def plain(node):
    """A catalog node as plain python containers, interpolations resolved."""
    return OmegaConf.to_container(node, resolve=True)


class TestCatalogVariants(unittest.TestCase):
    def test_the_original_catalogs_are_untouched(self):
        for base in ('illustris300', 'millenium'):
            with self.subTest(base):
                catalog = load_catalog(base)
                self.assertIsNone(catalog.get('particles_v2'))
                self.assertNotIn('dm', catalog.fields.particles)
                for star_field in NEW_STAR_FIELDS:
                    self.assertNotIn(star_field, catalog.fields.particles.star)

    def test_variants_only_add_the_new_fields_and_the_flag(self):
        for variant, (base, _, _) in PAIRS.items():
            with self.subTest(variant):
                new, old = load_catalog(variant), load_catalog(base)
                new_dict, old_dict = plain(new), plain(old)

                self.assertTrue(new_dict.pop('particles_v2'))
                self.assertEqual(new_dict.pop('path'), old_dict.pop('path') + PATH_SUFFIX)    # its own directory
                # the periodic box of the recentring, in the physical units of the positions
                self.assertAlmostEqual(new_dict.pop('Lbox'), old_dict.pop('Lbox') * old_dict['a'], places=1)
                new_particles, old_particles = new_dict['fields'].pop('particles'), old_dict['fields'].pop('particles')
                self.assertEqual(new_dict, old_dict)          # everything else, halo and subhalo fields included

                added = {key: new_particles['star'].pop(key) for key in NEW_STAR_FIELDS}
                new_particles.pop('dm')
                self.assertEqual(new_particles, old_particles)    # the original particle fields, gas block included
                self.assertEqual(set(added), NEW_STAR_FIELDS)

    def test_the_new_star_fields(self):
        for variant in PAIRS:
            with self.subTest(variant):
                star = load_catalog(variant).fields.particles.star
                self.assertEqual(star.GFM_Metallicity.name, 'metallicity')
                self.assertIsNone(star.GFM_Metallicity.scaling)
                self.assertEqual(star.GFM_StellarFormationTime.name, 'formation_time')
                self.assertEqual(list(star.GFM_StellarFormationTime.filter), ['> 0'])

    def test_the_dark_matter_block(self):
        for variant, (_, scaling, _) in PAIRS.items():
            with self.subTest(variant):
                particles = load_catalog(variant).fields.particles
                self.assertEqual(set(particles.dm.keys()), {'Coordinates', 'Masses'})
                self.assertEqual(particles.dm.Coordinates.name, 'p')
                self.assertEqual(particles.dm.Coordinates.scaling, scaling)
                self.assertEqual(particles.dm.Masses.name, 'm')
                self.assertEqual(particles.dm.Masses.scaling, 'x * 1e10')       # 10^10 Msun/h -> Msun/h

                # Dark matter positions live in the same coordinate system as the stars
                for key in ('input', 'output', 'scaling'):
                    self.assertEqual(particles.dm.Coordinates[key], particles.star.Coordinates[key], key)

    def test_the_fetcher_built_from_a_variant_uses_the_multi_type_behaviour(self):
        for variant, (_, _, types) in PAIRS.items():
            with self.subTest(variant):
                fetcher = DataFetcherFactory.get_data_fetcher(load_catalog(variant))
                self.assertIsInstance(fetcher, Illustris300Fetcher)
                self.assertTrue(fetcher._particles_v2)
                self.assertEqual(fetcher.particle_fields, types)

    def test_the_fetcher_built_from_an_original_catalog_does_not(self):
        for base in ('illustris300', 'millenium'):
            with self.subTest(base):
                fetcher = DataFetcherFactory.get_data_fetcher(load_catalog(base))
                self.assertFalse(fetcher._particles_v2)

    def test_the_variants_use_the_physical_box_length_for_the_periodic_recentring(self):
        # The positions are scaled by `a` (physical kpc/h), the original catalogs give the comoving length
        for variant, (base, expected) in {'illustris300_dm': ('illustris300', 170826.5),
                                          'millenium_dm': ('millenium', 400000.0)}.items():
            with self.subTest(variant):
                new, old = load_catalog(variant), load_catalog(base)
                self.assertAlmostEqual(new.Lbox, expected, places=1)
                self.assertAlmostEqual(new.Lbox, old.Lbox * old.a, places=1)
                self.assertEqual(new.fields.particles.star.Coordinates.scaling.count('a'), 1)    # positions are x * a ...

    def test_the_original_catalogs_keep_their_comoving_box_length(self):
        self.assertEqual(load_catalog('illustris300').Lbox, 205000)
        self.assertEqual(load_catalog('millenium').Lbox, 500000)

    def test_the_variant_paths_are_new_directories_that_keep_the_fetcher_markers(self):
        for variant, (base, _, _) in PAIRS.items():
            with self.subTest(variant):
                path, base_path = load_catalog(variant).path, load_catalog(base).path
                self.assertNotEqual(path, base_path)
                self.assertIn(PATH_MARKER[variant], path)
                self.assertTrue(path.endswith(PATH_SUFFIX))

    def test_attribute_names_of_the_variants(self):
        # <particle_type>_<name> is what SubhaloParticles builds and what ParticleField configs refer to
        for variant in PAIRS:
            with self.subTest(variant):
                particles = load_catalog(variant).fields.particles
                attributes = {f'{part_type}_{field.name}' for part_type in ('star', 'dm')
                              for field in particles[part_type].values()}
                self.assertTrue({'star_p', 'star_m', 'star_metallicity', 'star_formation_time',
                                 'dm_p', 'dm_m'} <= attributes)


class TestDatasetDirectories(unittest.TestCase):
    """The variant paths work with a dataset directory whose `output` is a symlink to the original one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.datasets = self.tmp.name
        # What `mkdir -p <new> && ln -s <original>/output <new>/output` does, done once by hand for each variant
        for original, new in (('illustris300', 'illustris300_dm_v1'),
                              ('millenium/MTNG-L500-4320-A', 'millenium/MTNG-L500-4320-A_dm_v1')):
            os.makedirs(os.path.join(self.datasets, original, 'output'))
            os.makedirs(os.path.join(self.datasets, new))
            os.symlink(os.path.join(self.datasets, original, 'output'), os.path.join(self.datasets, new, 'output'))

    def test_the_variant_catalogs_point_at_directories_with_the_linked_output(self):
        for variant in PAIRS:
            with self.subTest(variant):
                path = load_catalog(variant, self.datasets).path
                self.assertTrue(path.startswith(self.datasets))
                self.assertTrue(os.path.islink(os.path.join(path, 'output')))
                self.assertTrue(os.path.isdir(os.path.join(path, 'output')))     # the symlink resolves

    def test_the_fetcher_reads_the_snapshot_through_the_symlink(self):
        snapshot = FakeSnapshot()
        with patch.object(il.snapshot, 'loadSubhalo', side_effect=snapshot.load_subhalo):
            for variant, (base, _, _) in PAIRS.items():
                with self.subTest(variant):
                    catalog = load_catalog(variant, self.datasets)
                    original_output = os.path.join(self.datasets, base_directory(base), 'output', f'snapdir_{catalog.snapnum:03d}')
                    os.makedirs(original_output, exist_ok=True)
                    with h5py.File(os.path.join(original_output, f'snap_{catalog.snapnum:03d}.0.hdf5'), 'w') as header:
                        header.create_group('Header').attrs['MassTable'] = np.array([0, 0.01, 0, 0, 0, 0.])
                    snapshot.store.clear()
                    snapshot.add(1, 'star', Coordinates=np.ones((2, 3)), Velocities=np.ones((2, 3)), Masses=np.ones(2),
                                 GFM_Metallicity=np.full(2, 0.02), GFM_StellarFormationTime=np.full(2, 0.5))
                    snapshot.add(1, 'dm', Coordinates=np.ones((3, 3)))
                    snapshot.add(1, 'gas', Masses=np.ones(4))
                    particles = DataFetcherFactory.get_data_fetcher(catalog)._get_p_subhalo(1)
                    assert_allclose(particles.dm_m, np.full(3, 0.01 * 1e10))      # header read via the symlink
                    self.assertEqual(len(particles.star_metallicity), 2)


def base_directory(base):
    """Directory of an original catalog under the datasets dir, as the setup script expects it."""
    return {'illustris300': 'illustris300', 'millenium': 'millenium/MTNG-L500-4320-A'}[base]


class TestVariantsThroughTheFetcher(unittest.TestCase):
    """The real variant YAMLs, with their real unit scalings, run through the fetcher on a fake snapshot."""
    MASS_TABLE_DM = 0.01

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = FakeSnapshot()
        patcher = patch.object(il.snapshot, 'loadSubhalo', side_effect=self.snapshot.load_subhalo)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fetcher_for(self, variant):
        catalog = load_catalog(variant)
        catalog.path = self.tmp.name
        # Millennium-TNG snapshot files are called snapshot_NNN, the others snap_NNN
        prefix = 'snapshot' if variant == 'millenium_dm' else 'snap'
        snapdir = os.path.join(self.tmp.name, 'output', f'snapdir_{catalog.snapnum:03d}')
        os.makedirs(snapdir, exist_ok=True)
        with h5py.File(os.path.join(snapdir, f'{prefix}_{catalog.snapnum:03d}.0.hdf5'), 'w') as snapshot:
            snapshot.create_group('Header').attrs['MassTable'] = np.array([0, self.MASS_TABLE_DM, 0, 0, 0, 0.])
        return catalog, DataFetcherFactory.get_data_fetcher(catalog)

    def add_subhalo(self, subhalo_id, n_star, n_dm, wind=0):
        formation = np.full(n_star, 0.4)
        formation[:wind] = -1.0
        self.snapshot.add(subhalo_id, 'star',
                          Coordinates=np.arange(n_star * 3, dtype=float).reshape(n_star, 3) + 1.0,
                          Velocities=np.full((n_star, 3), 100.0), Masses=np.full(n_star, 1e-3),
                          GFM_Metallicity=np.linspace(0.01, 0.02, n_star), GFM_StellarFormationTime=formation)
        self.snapshot.add(subhalo_id, 'dm', Coordinates=-np.arange(n_dm * 3, dtype=float).reshape(n_dm, 3) - 1.0)
        self.snapshot.add(subhalo_id, 'gas', Masses=np.full(4, 2e-4))

    def test_every_attribute_of_the_use_case_is_built_with_the_right_units(self):
        for variant, (_, scaling, _) in PAIRS.items():
            with self.subTest(variant):
                self.snapshot.store.clear()
                self.snapshot.calls.clear()
                catalog, fetcher = self.fetcher_for(variant)
                a = catalog.a
                factor = a * 1000 if variant == 'millenium_dm' else a        # positions to pkpc/h
                self.add_subhalo(1, n_star=5, n_dm=6, wind=2)
                particles = fetcher._get_p_subhalo(1)

                self.assertIs(type(particles), SubhaloParticlesV2)
                for attribute in ('star_p', 'star_v', 'star_m', 'star_metallicity', 'star_formation_time', 'dm_p', 'dm_m'):
                    self.assertTrue(hasattr(particles, attribute), attribute)
                # the 2 wind cells are gone from every star array, dark matter is untouched
                self.assertEqual({len(particles.star_p), len(particles.star_m), len(particles.star_metallicity)}, {3})
                self.assertEqual(len(particles.dm_p), 6)
                assert_allclose(particles.star_metallicity, np.linspace(0.01, 0.02, 5)[2:])
                assert_allclose(particles.star_p, (np.arange(15, dtype=float).reshape(5, 3) + 1.0)[2:] * factor)
                assert_allclose(particles.dm_p, -(np.arange(18, dtype=float).reshape(6, 3) + 1.0) * factor)
                assert_allclose(particles.star_v, np.full((3, 3), 100.0 * math.sqrt(a)))
                assert_allclose(particles.star_m, np.full(3, 1e-3 * 1e10))
                assert_allclose(particles.dm_m, np.full(6, self.MASS_TABLE_DM * 1e10))

    def test_neighbours_of_a_variant_only_load_stars(self):
        for variant in PAIRS:
            with self.subTest(variant):
                self.snapshot.store.clear()
                self.snapshot.calls.clear()
                _, fetcher = self.fetcher_for(variant)
                self.add_subhalo(7, n_star=3, n_dm=5)
                particles = fetcher.get_p_subhalo(subhalo_id=7, halo_id=2)
                self.assertEqual({part_type for _, part_type, _ in self.snapshot.calls}, {'star'})
                self.assertFalse(hasattr(particles, 'dm_m'))
                self.assertFalse(hasattr(particles, 'gas_m'))

    def test_the_original_millennium_catalog_still_loads_stars_and_gas_for_neighbours(self):
        catalog = load_catalog('millenium')
        catalog.path = self.tmp.name
        os.makedirs(os.path.join(self.tmp.name, 'output'))             # no snapshot header file: it must not be read
        fetcher = DataFetcherFactory.get_data_fetcher(catalog)
        self.add_subhalo(7, n_star=3, n_dm=5)
        particles = fetcher.get_p_subhalo(subhalo_id=7, halo_id=2)
        self.assertEqual({part_type for _, part_type, _ in self.snapshot.calls}, {'star', 'gas'})
        self.assertTrue(hasattr(particles, 'gas_m'))
        self.assertIs(type(particles), SubhaloParticles)              # the existing class, and no header was needed


if __name__ == '__main__':
    unittest.main()
