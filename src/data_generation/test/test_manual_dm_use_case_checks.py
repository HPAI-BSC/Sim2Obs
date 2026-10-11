"""
Tests of the verdict logic of manual_tests/manual_test_dm_use_case.py (the MN5 smoke test).

The script itself needs real snapshots, so here its check functions get synthetic halos and a fake group
catalogue: a healthy halo must pass without a single FAIL, and every kind of problem the script is meant to
catch (wrong mass unit, wrong position scale, missing particles, wind cells left, ...) must be flagged.
"""
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import importlib.util
import os
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
from omegaconf import OmegaConf

from src.data_generation.src.feature import FeatureProcessorFactory
from src.data_generation.src.subhalo_particles_v2 import SubhaloParticlesV2

SCRIPT = os.path.join(os.path.dirname(__file__), 'manual_tests', 'manual_test_dm_use_case.py')
spec = importlib.util.spec_from_file_location('manual_test_dm_use_case', SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

MASS_TABLE = np.array([0, 0.004, 0, 0, 0, 0.])
DM_MASS = MASS_TABLE[1] * 1e10                  # Msun/h per DM particle
R200 = 500.0
FEATURES = [
    {'name': 'ParticleField', 'attribute': 'star_metallicity', 'weight': 'star_m', 'statistic': 'weighted_mean',
     'clip_min': 0.0, 'label': 'StellarMetallicity'},
    {'name': 'ParticleField', 'attribute': 'dm_m', 'statistic': 'sum', 'label': 'DarkMatterMass'},
]
N_STAR, N_DM = 900, 1800
SUBHALOS = [10, 11, 12]


def make_cfg(path='/nonexistent'):
    return OmegaConf.create({
        'catalog': {'name': 'illustris-300', 'snapnum': 84, 'a': 0.8333, 'path': path, 'transition_radius': 300.0,
                    'spatial_resolution': 2.0, 'bin_spatial_resolution_scale': 1.5, 'Lbox': None},
        'feature_names': FEATURES})


def make_own(seed=0, dm_mass=DM_MASS, dm_scale=1.0, wind=0, metallicity_scale=1.0):
    rng = np.random.default_rng(seed)
    formation = np.full(N_STAR, 0.5)
    formation[:wind] = -1.0
    return SubhaloParticlesV2(data={
        'star_p': rng.normal(0, 120, (N_STAR, 3)), 'star_v': rng.normal(0, 300, (N_STAR, 3)),
        'star_m': rng.uniform(1e6, 3e6, N_STAR), 'star_metallicity': rng.uniform(0.005, 0.04, N_STAR) * metallicity_scale,
        'star_formation_time': formation,
        'dm_p': rng.normal(0, 250, (N_DM, 3)) * dm_scale, 'dm_m': np.full(N_DM, dm_mass),
    }, column_map=None)


def make_halo(own, fraction=0.84):
    inside = own.dm_m[np.linalg.norm(own.dm_p, axis=1) < R200].sum()
    return SimpleNamespace(halo_id=3, m200=inside / fraction, r200=R200, subhalo_ids=SUBHALOS, gpos=np.zeros(3))


class FakeLookup:
    """Group catalogue records consistent with `own` split over the subhalos (stars: one wind cell each removed)."""

    def __init__(self, own, dm_mass_factor=1.0, dm_count_shift=0, group_share=0.83):
        dm_chunks = np.array_split(own.dm_m, len(SUBHALOS))
        star_chunks = np.array_split(own.star_m, len(SUBHALOS))
        self.records = {}
        for subhalo_id, dm, stars in zip(SUBHALOS, dm_chunks, star_chunks):
            self.records[subhalo_id] = {
                'SubhaloLenType': np.array([0, len(dm) + dm_count_shift, 0, 0, len(stars) + 1, 0]),
                'SubhaloMassType': np.array([0, dm.sum() / 1e10 * dm_mass_factor, 0, 0, stars.sum() / 1e10 * 1.0003, 0]),
            }
        self.group_dm = own.dm_m.sum() / 1e10 / group_share

    def subhalo(self, subhalo_id):
        return self.records[subhalo_id]

    def group(self, halo_id):
        return {'GroupMassType': np.array([0, self.group_dm, 0, 0, 0, 0])}


def statuses(rep):
    return {line['label']: line['status'] for line in rep.lines}


def failures(rep):
    return [line['label'] for line in rep.lines if line['status'] == 'FAIL']


class TestHealthyHalo(unittest.TestCase):
    def setUp(self):
        self.own = make_own()
        self.halo = make_halo(self.own)
        self.cfg = make_cfg()
        self.features = [FeatureProcessorFactory.get_feature(OmegaConf.create(f), cfg=self.cfg) for f in FEATURES]

    def test_every_check_passes_without_a_fail(self):
        rep = smoke.Report('healthy')
        self.assertTrue(smoke.check_particles(rep, self.own, MASS_TABLE))
        smoke.check_catalogue(rep, self.own, self.halo, FakeLookup(self.own))
        smoke.check_geometry(rep, self.own, self.halo)
        for feature in self.features:
            feature.process(halo=self.halo, subhalo_particles=self.own)
        smoke.check_maps(rep, self.cfg, self.own, self.halo, self.features)
        smoke.check_rotation_guard(rep, self.own)

        self.assertEqual(failures(rep), [])
        self.assertFalse(rep.failed)
        self.assertNotIn('WARN', {line['status'] for line in rep.lines}, [l for l in rep.lines if l['status'] == 'WARN'])
        self.assertGreater(len([line for line in rep.lines if line['status'] == 'OK']), 20)

    def test_the_maps_are_checked_for_the_three_projections(self):
        rep = smoke.Report('maps')
        for feature in self.features:
            feature.process(halo=self.halo, subhalo_particles=self.own)
        smoke.check_maps(rep, self.cfg, self.own, self.halo, self.features)
        for projection in ('xy', 'yz', 'xz'):
            self.assertTrue(any(line['label'].startswith(f'{projection}:') for line in rep.lines), projection)

    def test_the_catalogue_check_reports_the_share_of_the_group_dark_matter(self):
        rep = smoke.Report('share')
        smoke.check_catalogue(rep, self.own, self.halo, FakeLookup(self.own, group_share=0.83))
        line = next(l for l in rep.lines if l['label'].startswith('share of the FoF group'))
        self.assertEqual(line['status'], 'OK')
        self.assertIn('83', line['detail'])

    def test_a_small_share_of_the_group_dark_matter_is_a_warning_not_a_failure(self):
        rep = smoke.Report('share')
        smoke.check_catalogue(rep, self.own, self.halo, FakeLookup(self.own, group_share=0.2))
        self.assertEqual(statuses(rep)['share of the FoF group dark matter held by the loaded subhalos'], 'WARN')
        self.assertEqual(failures(rep), [])


class TestProblemsAreFlagged(unittest.TestCase):
    def check(self, own, lookup=None, geometry=False):
        halo = make_halo(make_own())            # the catalogue halo is the healthy one
        rep = smoke.Report('broken')
        smoke.check_particles(rep, own, MASS_TABLE)
        if lookup is not None:
            smoke.check_catalogue(rep, own, halo, lookup)
        if geometry:
            smoke.check_geometry(rep, own, halo)
        return rep

    def test_a_wrong_dm_mass_unit(self):
        self.assertIn('DM mass is the constant MassTable[1] * 1e10 Msun/h', failures(self.check(make_own(dm_mass=DM_MASS / 1e10))))

    def test_wind_cells_left_in_the_stars(self):
        rep = self.check(make_own(wind=5))
        self.assertIn('wind particles removed (formation time > 0 for every star)', failures(rep))

    def test_a_metallicity_that_is_not_a_mass_fraction(self):
        self.assertIn('metallicity looks like a mass fraction in [0, 1]', failures(self.check(make_own(metallicity_scale=100.0))))

    def test_nan_in_an_attribute(self):
        own = make_own()
        own.star_v[3, 1] = np.nan
        self.assertIn('no NaN or inf in any attribute', failures(self.check(own)))

    def test_missing_attributes(self):
        own = make_own()
        del own.dm_m
        rep = smoke.Report('missing')
        self.assertFalse(smoke.check_particles(rep, own, MASS_TABLE))
        self.assertIn('attributes', failures(rep))

    def test_dm_particles_missing_against_the_catalogue(self):
        own = make_own()
        rep = self.check(own, FakeLookup(own, dm_count_shift=40))
        self.assertIn('DM particle count equals SubhaloLenType[:, 1]', failures(rep))

    def test_dm_mass_that_does_not_match_the_catalogue(self):
        own = make_own()
        rep = self.check(own, FakeLookup(own, dm_mass_factor=1.05))
        self.assertIn('DM mass equals SubhaloMassType[:, 1] * 1e10', failures(rep))

    def test_positions_with_the_wrong_scale(self):
        # e.g. the factor 1000 of the Millennium coordinates forgotten, or applied twice
        rep = self.check(make_own(dm_scale=1000.0), geometry=True)
        self.assertIn('dark matter: median radius in units of R200 (R200=500.0)', failures(rep))

    def test_star_and_dark_matter_centres_in_different_places(self):
        own = make_own()
        own.dm_p = own.dm_p + np.array([2.0 * R200, 0, 0])
        rep = self.check(own, geometry=True)
        self.assertIn('star and dark matter centres are in the same place', failures(rep))

    def test_dm_mass_much_larger_than_m200_is_a_warning(self):
        own = make_own()
        halo = make_halo(own, fraction=5.0)           # M200 5 times smaller than the DM inside R200
        rep = smoke.Report('m200')
        smoke.check_geometry(rep, own, halo)
        self.assertEqual(statuses(rep)['DM mass inside R200 over M200 (about 0.84 for a complete halo)'], 'WARN')


class FakeFetcher:
    def __init__(self, neighbour):
        self.neighbour = neighbour
        self.calls = []

    def get_p_subhalo(self, subhalo_id, halo_id):
        self.calls.append((subhalo_id, halo_id))
        return self.neighbour


class FakeHaloContext:
    def get_halo_by_id(self, halo_id):
        return SimpleNamespace(halo_id=halo_id, subhalo_ids=[500 + halo_id])


class TestNeighbourAndRotation(unittest.TestCase):
    def neighbour_check(self, neighbour):
        own = make_own()
        rep = smoke.Report('neighbour')
        fetcher = FakeFetcher(neighbour)
        smoke.check_neighbour(rep, own, FakeHaloContext(), 3, [3, 4, 5], fetcher)
        return rep, fetcher

    def test_a_stars_only_neighbour_is_recovered_correctly(self):
        neighbour = SubhaloParticlesV2(data={'star_p': np.zeros((7, 3)), 'star_m': np.ones(7)}, column_map=None)
        rep, fetcher = self.neighbour_check(neighbour)
        self.assertEqual(failures(rep), [])
        self.assertEqual(fetcher.calls, [(504, 4)])             # the first subhalo of the next halo

    def test_a_neighbour_that_also_loaded_dark_matter_is_flagged(self):
        neighbour = SubhaloParticlesV2(data={'star_p': np.zeros((7, 3)), 'star_m': np.ones(7),
                                             'dm_p': np.zeros((9, 3)), 'dm_m': np.ones(9)}, column_map=None)
        rep, _ = self.neighbour_check(neighbour)
        self.assertIn('a neighbour subhalo loads stars only', failures(rep))

    def test_no_next_halo_skips_the_check(self):
        rep = smoke.Report('last')
        smoke.check_neighbour(rep, make_own(), FakeHaloContext(), 5, [3, 4, 5], FakeFetcher(None))
        self.assertEqual(rep.lines[0]['status'], 'INFO')

    def test_an_accepted_rotation_is_flagged(self):
        class Permissive(SubhaloParticlesV2):
            def rotate(self, z, degree):
                return None
        own = make_own()
        own.__class__ = Permissive
        rep = smoke.Report('rotation')
        smoke.check_rotation_guard(rep, own)
        self.assertEqual(len(failures(rep)), 1)


class TestOutputDirectory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = make_cfg(self.tmp.name)
        self.bins = smoke.muse_bins_r(600.0, 2.0, 1.5)
        self.side = 2 * int(self.bins / 4)
        self.root = os.path.join(self.tmp.name, 'no_rotation_s1.5')

    def write(self, label, name, mask_value=0.0, side=None, feature_value=0.01):
        side = side or self.side
        folder = os.path.join(self.root, label)
        os.makedirs(folder, exist_ok=True)
        array = np.full((side, side, 2), feature_value)
        array[..., 1] = mask_value
        np.savez(os.path.join(folder, name), array)

    def populate(self, **kwargs):
        for projection in ('xy', 'yz', 'xz'):
            name = f'halo_3_subhalo_10_{projection}_zoom_0.npz'
            self.write('StellarMetallicity', name, **{k: v for k, v in kwargs.items() if k != 'dm_name'})
            self.write('DarkMatterMass', kwargs.get('dm_name', name), **{k: v for k, v in kwargs.items() if k != 'dm_name'})

    def run_check(self):
        rep = smoke.Report('output')
        smoke.check_output_directory(rep, self.cfg)
        return rep

    def test_no_maps_yet_is_not_a_problem(self):
        rep = self.run_check()
        self.assertEqual(rep.lines[0]['status'], 'INFO')

    def test_a_healthy_directory_with_masked_pixels(self):
        self.populate(mask_value=1.0)
        rep = self.run_check()
        self.assertEqual(failures(rep), [])
        self.assertEqual(statuses(rep)['interloper mask channel'], 'OK')

    def test_an_all_zero_mask_is_a_warning_that_names_the_cause(self):
        self.populate(mask_value=0.0)
        rep = self.run_check()
        self.assertEqual(failures(rep), [])
        line = next(l for l in rep.lines if l['label'] == 'interloper mask channel')
        self.assertEqual(line['status'], 'WARN')
        self.assertIn('_get_interlopers_mask', line['detail'])

    def test_different_file_names_in_the_two_folders(self):
        self.populate(dm_name='halo_3_subhalo_10_xy_zoom_9.npz')
        self.assertIn('input and output folders hold the same file names', failures(self.run_check()))

    def test_wrong_array_shape(self):
        self.populate(side=self.side + 2)
        self.assertIn(f'each file is an array of shape ({self.side}, {self.side}, 2)', failures(self.run_check()))

    def test_a_missing_feature_folder(self):
        self.write('StellarMetallicity', 'halo_3_subhalo_10_xy_zoom_0.npz')
        self.assertIn('feature folders', failures(self.run_check()))

    def test_out_of_range_values(self):
        self.populate(feature_value=5.0)
        self.assertIn('value ranges', failures(self.run_check()))


class TestCatalogLookup(unittest.TestCase):
    """CatalogLookup reads single records of a group catalogue split in chunk files, with the offsets file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = os.path.join(self.tmp.name, 'output')
        os.makedirs(os.path.join(self.base, 'groups_084'))
        os.makedirs(os.path.join(self.tmp.name, 'postprocessing', 'offsets'))
        subhalos_per_chunk, groups_per_chunk = [3, 2], [2, 1]
        first_subhalo, first_group = 0, 0
        for chunk, (n_subhalos, n_groups) in enumerate(zip(subhalos_per_chunk, groups_per_chunk)):
            with h5py.File(os.path.join(self.base, 'groups_084', f'fof_subhalo_tab_084.{chunk}.hdf5'), 'w') as f:
                ids = np.arange(first_subhalo, first_subhalo + n_subhalos)
                f.create_dataset('Subhalo/SubhaloLenType', data=np.stack([ids * 0, 1000 + ids, ids * 0, ids * 0, 50 + ids, ids * 0], axis=1))
                f.create_dataset('Subhalo/SubhaloMassType', data=np.stack([ids * 0., 10. + ids, ids * 0., ids * 0., 100. + ids, ids * 0.], axis=1))
                f.create_dataset('Subhalo/SubhaloOtherField', data=np.zeros(n_subhalos, dtype='V6'))     # never needed
                gids = np.arange(first_group, first_group + n_groups)
                f.create_dataset('Group/GroupMassType', data=np.stack([gids * 0., 500. + gids, gids * 0., gids * 0., gids * 0., gids * 0.], axis=1))
            first_subhalo += n_subhalos
            first_group += n_groups
        with h5py.File(os.path.join(self.tmp.name, 'postprocessing', 'offsets', 'offsets_084.hdf5'), 'w') as f:
            f['FileOffsets/Subhalo'] = np.array([0, 3])
            f['FileOffsets/Group'] = np.array([0, 2])
        self.lookup = smoke.CatalogLookup(self.base, 84)
        self.addCleanup(self.lookup.close)

    def test_records_inside_and_across_chunk_files(self):
        for subhalo_id in (0, 2, 3, 4, 1):
            record = self.lookup.subhalo(subhalo_id)
            self.assertEqual(record['SubhaloLenType'][1], 1000 + subhalo_id, subhalo_id)
            self.assertEqual(record['SubhaloLenType'][4], 50 + subhalo_id)
            self.assertEqual(record['SubhaloMassType'][1], 10. + subhalo_id)
            self.assertEqual(record['SubhaloMassType'][4], 100. + subhalo_id)
        for halo_id in (0, 2, 1):
            self.assertEqual(self.lookup.group(halo_id)['GroupMassType'][1], 500. + halo_id)

    def test_only_the_fields_the_checks_need_are_read(self):
        # illustris_python.loadSingle reads every field, which fails on the Millennium-TNG catalogue
        self.assertEqual(set(self.lookup.subhalo(0)), {'SubhaloLenType', 'SubhaloMassType'})
        self.assertEqual(set(self.lookup.group(0)), {'GroupMassType'})

    def test_the_chunk_file_is_kept_open_for_consecutive_subhalos_and_closed_at_the_end(self):
        self.lookup.subhalo(0)
        handle = self.lookup._open_file
        self.lookup.subhalo(1)
        self.assertIs(self.lookup._open_file, handle)
        self.lookup.subhalo(3)                       # next chunk file
        self.assertIsNot(self.lookup._open_file, handle)
        self.lookup.close()
        self.assertIsNone(self.lookup._open_file)

    def test_it_agrees_with_illustris_python_on_the_records_it_reads(self):
        import illustris_python as il
        for subhalo_id in (0, 4):
            expected = il.groupcat.loadSingle(self.base, 84, subhaloID=subhalo_id)
            record = self.lookup.subhalo(subhalo_id)
            np.testing.assert_array_equal(record['SubhaloMassType'], expected['SubhaloMassType'])
            np.testing.assert_array_equal(record['SubhaloLenType'], expected['SubhaloLenType'])


class TestGuard(unittest.TestCase):
    def test_an_exception_becomes_a_fail_and_later_sections_still_run(self):
        rep = smoke.Report('guard')
        ran = []

        def broken():
            raise TypeError("data type '<u6' not understood")

        self.assertIsNone(smoke.guarded(rep, 'catalogue check', broken))
        smoke.guarded(rep, 'geometry check', lambda: ran.append('geometry'))
        self.assertEqual(ran, ['geometry'])
        self.assertTrue(rep.failed)
        self.assertEqual(rep.lines[0]['label'], 'catalogue check raised an exception')
        self.assertIn('<u6', rep.lines[0]['detail'])

    def test_the_return_value_is_passed_through(self):
        self.assertEqual(smoke.guarded(smoke.Report('r'), 'x', lambda a, b: a + b, 2, 3), 5)

    def test_memory_helpers_return_positive_numbers(self):
        self.assertGreater(smoke.current_rss_mb(), 0)
        self.assertGreater(smoke.peak_rss_mb(), 0)


class TestPlanningEstimate(unittest.TestCase):
    DM_MASS = 1e7

    def samples(self):
        # time = fixed + slope * millions, with loading 2 s + 3 s/M and maps 0.5 s + 1 s/M
        # halo A: M200/m_dm = 1e6 particles, 1.5 million loaded (factor 1.5). halo B: 2e6 and 2.2 million (factor 1.1)
        return [(1e6 * self.DM_MASS, 1.5, 2 + 3 * 1.5, 0.5 + 1.5), (2e6 * self.DM_MASS, 2.2, 2 + 3 * 2.2, 0.5 + 2.2)]

    def test_the_estimate_uses_a_fixed_cost_per_halo_plus_a_cost_per_particle(self):
        m200_all = np.full(10_000, 1e6 * self.DM_MASS)                    # 1e10 expected particles in total
        rep = smoke.Report('plan')
        smoke.estimate_full_run(rep, m200_all, self.DM_MASS, self.samples())

        measured = next(l for l in rep.lines if l['label'] == 'measured')
        self.assertIn('1.10 to 1.50 particles per M200/m_dm', measured['detail'])
        self.assertIn('Loading: 2.00 s per halo + 3.00 s per million particles', measured['detail'])
        self.assertIn('Maps: 0.50 s per halo + 1.00 s per million particles', measured['detail'])
        line = next(l for l in rep.lines if l['label'].startswith('planning estimate for all'))
        self.assertIn('all 10000 selected halos', line['label'])
        # factors 1.1, 1.3, 1.5 -> 11000, 13000, 15000 million particles
        self.assertIn('13000 million particles in total (range 11000 to 15000)', line['detail'])
        # loading: (10000 * 2 + 3 * M) / 3600 h = 14.7, 16.4, 18.1. Maps: (10000 * 0.5 + M) / 3600 = 4.4, 5.0, 5.6
        self.assertIn('Loading 16.4 h (range 14.7 to 18.1 h), maps 5.0 h (range 4.4 to 5.6 h), in total 21.4 h', line['detail'])

    def test_many_small_halos_are_dominated_by_the_fixed_cost(self):
        # 50000 halos of 1e5 expected particles: the particles cost little, the fixed 2 s per halo dominates
        m200_all = np.full(50_000, 1e5 * self.DM_MASS)
        rep = smoke.Report('plan')
        smoke.estimate_full_run(rep, m200_all, self.DM_MASS, self.samples())
        line = next(l for l in rep.lines if l['label'].startswith('planning estimate for all'))
        # 50000 * 2 s = 27.8 h of fixed cost, plus 3 s/M * 6500 million = 5.4 h
        self.assertIn('Loading 33.2 h', line['detail'])

    def test_the_estimate_is_skipped_without_enough_samples(self):
        rep = smoke.Report('plan')
        smoke.estimate_full_run(rep, [1e14], 1e7, [(1e14, 1.0, 1.0, None), None])
        self.assertEqual(rep.lines[0]['label'], 'planning estimate skipped')


A_FACTOR, LBOX_COMOVING = 0.8, 500000.0
BOX_PHYSICAL = LBOX_COMOVING * A_FACTOR            # 400000: the box in the units of the particle positions


def edge_cfg():
    cfg = make_cfg()
    cfg.catalog.Lbox, cfg.catalog.a = LBOX_COMOVING, A_FACTOR
    return cfg


def make_edge_particles(centre_x, seed=0, n=3000):
    """Particles of a halo around `centre_x`, wrapped into the physical box [0, BOX_PHYSICAL) as in the snapshot."""
    rng = np.random.default_rng(seed)
    centre = np.array([centre_x, 200000.0, 200000.0])

    def around(sigma, count):
        positions = centre + rng.normal(0, sigma, (count, 3))
        positions[:, 0] %= BOX_PHYSICAL
        return positions

    own = SubhaloParticlesV2(data={
        'star_p': around(120, n), 'star_v': rng.normal(0, 300, (n, 3)), 'star_m': np.full(n, 2e6),
        'star_metallicity': np.full(n, 0.02), 'star_formation_time': np.full(n, 0.5),
        'dm_p': around(250, 2 * n), 'dm_m': np.full(2 * n, DM_MASS)}, column_map=None)
    halo = SimpleNamespace(halo_id=7, m200=1e14, r200=500.0, subhalo_ids=[1], gpos=centre)
    return own, halo


class TestPeriodicBox(unittest.TestCase):
    def run_diagnostic(self, centre_x):
        own, halo = make_edge_particles(centre_x)
        cfg = edge_cfg()
        candidates = {name: smoke.far_fractions(own, halo, box) for name, box in smoke.candidate_boxes(cfg).items()}
        own.preprocess(halo, cfg.catalog.Lbox)              # what the data loader does, with the comoving Lbox
        rep = smoke.Report('periodic')
        smoke.check_periodic(rep, own, halo, candidates)
        return rep, candidates

    def test_the_candidate_boxes_are_the_catalog_length_and_it_times_and_over_a(self):
        boxes = smoke.candidate_boxes(edge_cfg())
        self.assertEqual(boxes, {'Lbox (as the pipeline uses it)': LBOX_COMOVING, 'Lbox * a': BOX_PHYSICAL,
                                 'Lbox / a': LBOX_COMOVING / A_FACTOR})

    def test_a_halo_across_the_box_edge_is_flagged_and_the_physical_box_is_named(self):
        rep, candidates = self.run_diagnostic(centre_x=BOX_PHYSICAL - 100.0)
        warnings = [line for line in rep.lines if line['status'] == 'WARN']
        self.assertEqual(len(warnings), 2)                                    # stars and dark matter
        self.assertTrue(all('"Lbox * a"' in line['detail'].split('fits is')[-1] for line in warnings))
        # with the physical box nothing is far from the halo, with the comoving one a large share is
        self.assertLess(candidates['Lbox * a']['stars'], 1e-3)
        self.assertGreater(candidates['Lbox (as the pipeline uses it)']['stars'], 0.2)

    def test_a_halo_across_the_box_edge_is_fine_when_the_catalog_gives_the_physical_box(self):
        # What the *_dm variants do: Lbox is already the length in position units, so the pipeline's recentring is right
        own, halo = make_edge_particles(BOX_PHYSICAL - 100.0)
        cfg = edge_cfg()
        cfg.catalog.Lbox = BOX_PHYSICAL
        candidates = {name: smoke.far_fractions(own, halo, box) for name, box in smoke.candidate_boxes(cfg).items()}
        own.preprocess(halo, cfg.catalog.Lbox)
        rep = smoke.Report('physical')
        smoke.check_periodic(rep, own, halo, candidates)
        self.assertEqual([line['status'] for line in rep.lines], ['OK', 'OK'])
        geometry = smoke.Report('geometry')
        smoke.check_geometry(geometry, own, halo)
        self.assertEqual(failures(geometry), [])

    def test_the_box_length_is_estimated_from_the_halo_positions(self):
        context = FakeHaloData({'halo_id': [1, 2], 'gpos': [np.array([10., 20., 30.]), np.array([399990., 5., 6.])]})
        self.assertEqual(smoke.estimate_box(context), 399990.0)

    def test_a_halo_in_the_middle_of_the_box_is_fine(self):
        rep, _ = self.run_diagnostic(centre_x=BOX_PHYSICAL / 2)
        self.assertEqual([line['status'] for line in rep.lines], ['OK', 'OK'])

    def test_the_physical_box_recentres_the_edge_halo_correctly(self):
        # What preprocessing with Lbox * a would give: the whole halo around its centre
        own, halo = make_edge_particles(BOX_PHYSICAL - 100.0)
        own.preprocess(halo, BOX_PHYSICAL)
        self.assertLess(np.linalg.norm(own.dm_p, axis=1).max(), 6 * 250)
        rep = smoke.Report('geometry')
        smoke.check_geometry(rep, own, halo)
        self.assertEqual(failures(rep), [])

    def test_the_geometry_check_fails_on_the_wrongly_wrapped_halo(self):
        # The case of halo 2000: the centres of stars and dark matter are far from each other and from the halo
        own, halo = make_edge_particles(BOX_PHYSICAL - 100.0)
        own.preprocess(halo, LBOX_COMOVING)
        rep = smoke.Report('geometry')
        smoke.check_geometry(rep, own, halo)
        # The same signature as halo 2000: a sensible median radius, both centres far from the halo, centres apart
        self.assertEqual(failures(rep), ['star and dark matter centres are in the same place'])
        warnings = [line['label'] for line in rep.lines if line['status'] == 'WARN']
        self.assertIn('stars centre is far from the halo centre', warnings)
        self.assertIn('dark matter centre is far from the halo centre', warnings)


class FakeHaloData:
    def __init__(self, rows):
        import pandas as pd
        self.data = pd.DataFrame(rows)


class TestEdgeHalos(unittest.TestCase):
    def context(self):
        return FakeHaloData({
            'halo_id': [1, 2, 3, 4], 'm200': [9e14, 1e14, 5e14, 2e13], 'r200': [500.0, 500.0, 500.0, 500.0],
            'gpos': [np.array([200000., 200000., 200000.]), np.array([300., 200000., 200000.]),
                     np.array([200000., BOX_PHYSICAL - 200., 200000.]), np.array([200000., 200000., 100.])]})

    def test_the_most_massive_halos_touching_a_face_are_selected(self):
        self.assertEqual(smoke.find_edge_halos(self.context(), BOX_PHYSICAL, 1), [3])
        self.assertEqual(smoke.find_edge_halos(self.context(), BOX_PHYSICAL, 2), [3, 2])
        self.assertEqual(smoke.find_edge_halos(self.context(), BOX_PHYSICAL, 10), [3, 2, 4])

    def test_nothing_is_selected_when_no_halo_touches_a_face_or_none_is_asked(self):
        self.assertEqual(smoke.find_edge_halos(self.context(), BOX_PHYSICAL, 0), [])
        interior = FakeHaloData({'halo_id': [1], 'm200': [1e14], 'r200': [500.0], 'gpos': [np.array([2e5, 2e5, 2e5])]})
        self.assertEqual(smoke.find_edge_halos(interior, BOX_PHYSICAL, 3), [])


class TestReport(unittest.TestCase):
    def test_failed_only_with_a_fail_line(self):
        rep = smoke.Report('r')
        rep.add('OK', 'a')
        rep.add('WARN', 'b')
        self.assertFalse(rep.failed)
        rep.add('FAIL', 'c')
        self.assertTrue(rep.failed)


if __name__ == '__main__':
    unittest.main()
