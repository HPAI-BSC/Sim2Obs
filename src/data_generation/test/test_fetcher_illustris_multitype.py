"""
Tests for the multi particle type behaviour of Illustris300Fetcher (Illustris, TNG-Cluster, Millennium-TNG).

`illustris_python`'s loadSubhalo is replaced by a fake that behaves like the real one for what matters here
(a single requested field comes back as an array, several as a dict with a `count`, and asking dark matter for
`Masses` fails because the snapshot only has them in the header). The header is a tiny real HDF5 file, so
`_get_mass_table` runs for real. No simulation data is needed.
"""
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import illustris_python as il
import numpy as np
from numpy.testing import assert_allclose
from omegaconf import OmegaConf

from src.data_generation.src.data_fetcher import Illustris300Fetcher
from src.data_generation.src.feature import ParticleFieldProcessor
from src.data_generation.src.subhalo import SubhaloParticles
from src.data_generation.src.subhalo_particles_v2 import SubhaloParticlesV2

DM_MASS_TABLE = 0.004          # header units, 1e10 Msun/h
A = 0.5

STAR_FIELDS = {
    'Coordinates': {'input': 'ckpc/h', 'output': 'pkpc/h', 'scaling': 'x * a', 'name': 'p'},
    'Masses': {'input': '10^10 Msun/h', 'output': 'Msun/h', 'scaling': 'x * 1e10', 'name': 'm'},
    'GFM_Metallicity': {'input': None, 'output': None, 'scaling': None, 'name': 'metallicity'},
    'GFM_StellarFormationTime': {'input': None, 'output': None, 'scaling': None, 'name': 'formation_time',
                                 'filter': ['> 0']},
}
DM_FIELDS = {
    'Coordinates': {'input': 'ckpc/h', 'output': 'pkpc/h', 'scaling': 'x * a', 'name': 'p'},
    'Masses': {'input': '10^10 Msun/h', 'output': 'Msun/h', 'scaling': 'x * 1e10', 'name': 'm'},
}
STAR_ONLY_FIELDS = {key: STAR_FIELDS[key] for key in ('Coordinates', 'Masses')}


def make_catalog(path, particles, particles_v2=False):
    return OmegaConf.create({
        'name': 'illustris-300', 'snapnum': 84, 'a': A, 'H': 0.7, 'Lbox': 1000.0, 'path': str(path),
        **({'particles_v2': True} if particles_v2 else {}),
        'fields': {
            'halo': {'Group_M_Crit200': {'input': None, 'output': None, 'scaling': None, 'name': 'm200'}},
            'subhalo': {'SubhaloGrNr': {'input': None, 'output': None, 'scaling': None, 'name': 'id_parents'}},
            'particles': particles,
        }})


class FakeSnapshot:
    """Particles of a few subhalos, served like il.snapshot.loadSubhalo does."""

    def __init__(self):
        self.store = {}      # (subhalo_id, part_type) -> {field: array}
        self.calls = []      # (subhalo_id, part_type, list of requested fields)

    def add(self, subhalo_id, part_type, **fields):
        self.store[(subhalo_id, part_type)] = fields

    def load_subhalo(self, base_path, snap_num, subhalo_id, part_type, fields=None):
        self.calls.append((subhalo_id, part_type, list(fields)))
        if part_type == 'dm' and 'Masses' in fields:
            raise Exception('Particle type [1] does not have field [Masses]')
        stored = self.store.get((subhalo_id, part_type))
        if stored is None or len(next(iter(stored.values()))) == 0:
            return {'count': 0}
        if len(fields) == 1:
            return stored[fields[0]].copy()
        result = {field: stored[field].copy() for field in fields}
        result['count'] = len(next(iter(result.values())))
        return result


def star_data(n, offset=0.0, wind=0):
    """n stars, the first `wind` of them are wind particles (formation time <= 0)."""
    formation = np.full(n, 0.5)
    formation[:wind] = -1.0
    return dict(Coordinates=np.arange(n * 3, dtype=float).reshape(n, 3) + offset, Masses=np.full(n, 1e-3) + offset * 1e-6,
                GFM_Metallicity=np.linspace(0.01, 0.03, n), GFM_StellarFormationTime=formation)


def dm_data(n, offset=0.0):
    return dict(Coordinates=-np.arange(n * 3, dtype=float).reshape(n, 3) - offset)


class FetcherTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        snapdir = os.path.join(self.tmp.name, 'output', 'snapdir_084')
        os.makedirs(snapdir)
        self.header_file = os.path.join(snapdir, 'snap_084.0.hdf5')
        with h5py.File(self.header_file, 'w') as snapshot:
            snapshot.create_group('Header').attrs['MassTable'] = np.array([0, DM_MASS_TABLE, 0, 0, 0, 0.])
        self.snapshot = FakeSnapshot()
        patcher = patch.object(il.snapshot, 'loadSubhalo', side_effect=self.snapshot.load_subhalo)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fetcher(self, particles, particles_v2=False):
        return Illustris300Fetcher(make_catalog(self.tmp.name, copy.deepcopy(particles), particles_v2))

    def multi_type_fetcher(self):
        return self.fetcher({'star': STAR_FIELDS, 'dm': DM_FIELDS}, particles_v2=True)


class TestMultiTypeLoading(FetcherTestCase):
    def test_a_multi_type_catalog_gives_a_v2_with_every_named_attribute(self):
        self.snapshot.add(1, 'star', **star_data(4))
        self.snapshot.add(1, 'dm', **dm_data(6))
        particles = self.multi_type_fetcher()._get_p_subhalo(1)

        self.assertIs(type(particles), SubhaloParticlesV2)
        for attribute in ('star_p', 'star_m', 'star_metallicity', 'star_formation_time', 'dm_p', 'dm_m'):
            self.assertTrue(hasattr(particles, attribute), attribute)
        self.assertEqual(particles.star_p.shape, (4, 3))
        self.assertEqual(particles.dm_p.shape, (6, 3))
        assert_allclose(particles.star_metallicity, np.linspace(0.01, 0.03, 4))
        assert_allclose(particles.star_p, star_data(4)['Coordinates'] * A)     # 'x * a' from the YAML
        self.assertEqual(particles.particle_types, ['star', 'dm'])

    def test_dark_matter_mass_is_the_header_mass_with_the_yaml_unit_conversion(self):
        self.snapshot.add(1, 'star', **star_data(3))
        self.snapshot.add(1, 'dm', **dm_data(5))
        particles = self.multi_type_fetcher()._get_p_subhalo(1)

        assert_allclose(particles.dm_m, np.full(5, DM_MASS_TABLE * 1e10))    # 'x * 1e10' -> Msun/h
        requested = {part_type: fields for _, part_type, fields in self.snapshot.calls}
        self.assertNotIn('Masses', requested['dm'])         # the snapshot has no DM Masses dataset
        self.assertEqual(requested['dm'], ['Coordinates'])
        self.assertIn('Masses', requested['star'])          # stars still read their masses from the snapshot

    def test_the_header_is_read_once_and_cached(self):
        self.snapshot.add(1, 'star', **star_data(3))
        self.snapshot.add(1, 'dm', **dm_data(5))
        fetcher = self.multi_type_fetcher()
        fetcher._get_p_subhalo(1)
        os.remove(self.header_file)                          # a second read would fail
        particles = fetcher._get_p_subhalo(1)
        assert_allclose(particles.dm_m, np.full(5, DM_MASS_TABLE * 1e10))
        assert_allclose(fetcher._get_mass_table(), [0, DM_MASS_TABLE, 0, 0, 0, 0])

    def test_wind_particles_are_removed_from_every_star_array_only(self):
        self.snapshot.add(1, 'star', **star_data(5, wind=2))
        self.snapshot.add(1, 'dm', **dm_data(4))
        particles = self.multi_type_fetcher()._get_p_subhalo(1)

        self.assertEqual(len(particles.star_p), 3)
        self.assertEqual(len(particles.star_m), 3)
        self.assertEqual(len(particles.star_metallicity), 3)
        self.assertTrue((particles.star_formation_time > 0).all())
        assert_allclose(particles.star_metallicity, np.linspace(0.01, 0.03, 5)[2:])
        self.assertEqual(len(particles.dm_p), 4)

    def test_a_subhalo_without_dark_matter_still_loads(self):
        self.snapshot.add(1, 'star', **star_data(3))
        self.snapshot.add(1, 'dm', Coordinates=np.zeros((0, 3)))
        particles = self.multi_type_fetcher()._get_p_subhalo(1)
        self.assertEqual(len(particles.star_m), 3)
        self.assertFalse(hasattr(particles, 'dm_m'))

    def test_a_type_that_only_declares_masses_from_the_header_raises_a_clear_error(self):
        fetcher = self.fetcher({'star': STAR_FIELDS, 'dm': {'Masses': DM_FIELDS['Masses']}}, particles_v2=True)
        self.snapshot.add(1, 'star', **star_data(3))
        with self.assertRaises(ValueError) as context:
            fetcher._get_p_subhalo(1)
        self.assertIn("'dm'", str(context.exception))

    def test_a_halo_merges_the_stars_and_dark_matter_of_its_subhalos(self):
        for subhalo_id, (n_star, n_dm) in {1: (4, 6), 2: (2, 3)}.items():
            self.snapshot.add(subhalo_id, 'star', **star_data(n_star, offset=10.0 * subhalo_id))
            self.snapshot.add(subhalo_id, 'dm', **dm_data(n_dm, offset=10.0 * subhalo_id))
        halo = SimpleNamespace(halo_id=5, subhalo_ids=[1, 2])
        particles = self.multi_type_fetcher()._get_p_halo(halo)

        self.assertIsInstance(particles, SubhaloParticlesV2)
        self.assertEqual(len(particles.star_m), 6)
        self.assertEqual(len(particles.dm_m), 9)
        self.assertEqual(len(particles.dm_p), 9)
        self.assertIsNone(particles.overlapping_particles_id)         # reset after merging, as before
        self.assertIs(particles.non_overlapping, particles)


class TestNeighbours(FetcherTestCase):
    def test_neighbours_only_load_stars(self):
        self.snapshot.add(7, 'star', **star_data(3))
        self.snapshot.add(7, 'dm', **dm_data(5))
        particles = self.multi_type_fetcher().get_p_subhalo(subhalo_id=7, halo_id=3)

        self.assertEqual({part_type for _, part_type, _ in self.snapshot.calls}, {'star'})
        self.assertEqual(len(particles.star_m), 3)
        self.assertFalse(hasattr(particles, 'dm_m'))
        self.assertIs(type(particles), SubhaloParticlesV2)

    def test_the_halos_own_subhalos_still_load_every_type(self):
        self.snapshot.add(1, 'star', **star_data(3))
        self.snapshot.add(1, 'dm', **dm_data(5))
        self.multi_type_fetcher()._get_p_halo(SimpleNamespace(halo_id=5, subhalo_ids=[1]))
        self.assertEqual({part_type for _, part_type, _ in self.snapshot.calls}, {'star', 'dm'})

    def test_the_restriction_matching_nothing_loads_every_type_with_a_warning(self):
        fetcher = self.multi_type_fetcher()
        with self.assertLogs('DataFetcher', level='WARNING'):
            selected = fetcher._select_particle_types(['gas'])
        self.assertEqual(selected, ['star', 'dm'])

    def test_a_single_type_catalog_with_the_flag_ignores_the_restriction_whatever_its_key(self):
        fetcher = self.fetcher({'stars': STAR_ONLY_FIELDS}, particles_v2=True)
        self.snapshot.add(7, 'stars', **{key: star_data(3)[key] for key in STAR_ONLY_FIELDS})
        with self.assertNoLogs('DataFetcher', level='WARNING'):       # no "nothing matches" fallback either
            particles = fetcher.get_p_subhalo(subhalo_id=7, halo_id=3)
        self.assertEqual(len(particles.star_m), 3)         # 'stars' is not 'star', it must still be loaded


class TestCatalogsWithoutTheFlagAreUnchanged(FetcherTestCase):
    def test_a_star_only_catalog_keeps_the_existing_class_and_reads_masses_from_the_snapshot(self):
        fetcher = self.fetcher({'star': STAR_ONLY_FIELDS})
        self.snapshot.add(1, 'star', **{key: star_data(4)[key] for key in STAR_ONLY_FIELDS})
        particles = fetcher._get_p_subhalo(1)

        self.assertIs(type(particles), SubhaloParticles)
        self.assertEqual(self.snapshot.calls, [(1, 'star', ['Coordinates', 'Masses'])])
        assert_allclose(particles.star_m, star_data(4)['Masses'] * 1e10)
        self.assertFalse(hasattr(fetcher, '_mass_table'))      # the header is not read when not needed

    def test_a_catalog_with_several_types_but_no_flag_behaves_as_before(self):
        # Like millenium.yaml today: stars plus a gas block with masses only
        gas_fields = {'Masses': STAR_ONLY_FIELDS['Masses']}
        fetcher = self.fetcher({'star': STAR_ONLY_FIELDS, 'gas': gas_fields})
        for subhalo_id in (1, 7):
            self.snapshot.add(subhalo_id, 'star', **{key: star_data(3)[key] for key in STAR_ONLY_FIELDS})
            self.snapshot.add(subhalo_id, 'gas', Masses=np.full(5, 2e-4))

        own = fetcher._get_p_subhalo(1)
        neighbour = fetcher.get_p_subhalo(subhalo_id=7, halo_id=3)

        self.assertIs(type(own), SubhaloParticles)
        self.assertIs(type(neighbour), SubhaloParticles)
        self.assertEqual({part_type for _, part_type, _ in self.snapshot.calls}, {'star', 'gas'})
        self.assertTrue(hasattr(neighbour, 'gas_m'))            # neighbours still load every type
        self.assertFalse(hasattr(fetcher, '_mass_table'))       # and no header access
        self.assertEqual(self.snapshot.calls[1], (1, 'gas', ['Masses']))     # gas masses read from the snapshot

    def test_the_flag_on_a_star_only_catalog_gives_a_v2(self):
        fetcher = self.fetcher({'star': STAR_ONLY_FIELDS}, particles_v2=True)
        self.snapshot.add(1, 'star', **{key: star_data(4)[key] for key in STAR_ONLY_FIELDS})
        self.assertIs(type(fetcher._get_p_subhalo(1)), SubhaloParticlesV2)


class TestEndToEnd(FetcherTestCase):
    def test_maps_of_a_halo_and_its_neighbour(self):
        # Own halo: two subhalos with stars and dark matter. Neighbour: stars only (as the data loader loads it)
        for subhalo_id, (n_star, n_dm) in {1: (4, 6), 2: (2, 3)}.items():
            self.snapshot.add(subhalo_id, 'star', **star_data(n_star, offset=10.0 * subhalo_id))
            self.snapshot.add(subhalo_id, 'dm', **dm_data(n_dm, offset=10.0 * subhalo_id))
        self.snapshot.add(9, 'star', **star_data(5, offset=500.0))
        self.snapshot.add(9, 'dm', **dm_data(7, offset=500.0))
        fetcher = self.multi_type_fetcher()

        own = fetcher._get_p_halo(SimpleNamespace(halo_id=5, subhalo_ids=[1, 2]))
        own.preprocess(SimpleNamespace(gpos=np.zeros(3), halo_id=5))
        neighbour = fetcher.get_p_subhalo(subhalo_id=9, halo_id=6)
        neighbour.preprocess(SimpleNamespace(gpos=np.zeros(3), halo_id=6))
        combined = own + neighbour

        self.assertEqual(len(combined.star_m), 4 + 2 + 5)
        self.assertEqual(len(combined.dm_m), 6 + 3)                  # the neighbour added no dark matter
        self.assertEqual(len(combined.non_overlapping.star_m), 6)
        self.assertEqual(len(combined.non_overlapping.dm_m), 9)

        big = 1e6
        ranging = ((-big, big), (-big, big))
        dark_matter = ParticleFieldProcessor(OmegaConf.create({'attribute': 'dm_m', 'statistic': 'sum'}))
        metallicity = ParticleFieldProcessor(OmegaConf.create(
            {'attribute': 'star_metallicity', 'weight': 'star_m', 'statistic': 'weighted_mean'}))

        dm_map = dark_matter.generate_map(None, None, combined, 4, ranging, axis=(0, 1))[0]
        z_map = metallicity.generate_map(None, None, combined, 4, ranging, axis=(0, 1))[0]
        self.assertAlmostEqual(dm_map.sum(), 9 * DM_MASS_TABLE * 1e10, places=3)      # all own DM, in Msun/h
        own_z = np.concatenate([np.linspace(0.01, 0.03, 4), np.linspace(0.01, 0.03, 2)])
        own_m = np.concatenate([star_data(4, 10.0)['Masses'], star_data(2, 20.0)['Masses']]) * 1e10
        self.assertAlmostEqual(z_map[z_map > 0].mean(), np.average(own_z, weights=own_m), places=6)


if __name__ == '__main__':
    unittest.main()
