"""
MapStep with two ParticleField features: where the maps are written and what they contain.

The real MapStep runs on synthetic SubhaloParticlesV2 and writes into a temporary catalog folder. The layout
it must follow is the one of the existing features:

    <catalog.path>/no_rotation_s<bin_spatial_resolution_scale>/<feature name>/
        halo_<halo_id>_subhalo_<subhalo_id>_<projection>_zoom_<i>.npz

with the input feature and the output feature each in their own folder, the same file names in both, and every
file holding one array of shape (H, W, 2): channel 0 is the zoomed map, channel 1 the interloper mask.
"""
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from omegaconf import OmegaConf

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.feature import FeatureProcessorFactory
from src.data_generation.src.map_step import MapStep
from src.data_generation.src.subhalo import SubhaloContext
from src.data_generation.src.subhalo_particles_v2 import SubhaloParticlesV2
from src.data_generation.src.utils import muse_bins_r

TRANSITION_RADIUS = 100.0
SPATIAL_RESOLUTION = 2.0
SCALE = 1.5
N_ZOOMS = 3
HALO_ID, SUBHALO_ID = 1, 0
FEATURES = [
    {'name': 'ParticleField', 'attribute': 'star_metallicity', 'weight': 'star_m', 'statistic': 'weighted_mean',
     'clip_min': 0.0, 'label': 'StellarMetallicity'},
    {'name': 'ParticleField', 'attribute': 'dm_m', 'statistic': 'sum', 'label': 'DarkMatterMass'},
]


def make_particles(rng):
    own = SubhaloParticlesV2(data={
        'star_p': rng.normal(0, 40, (3000, 3)), 'star_m': rng.uniform(0.5, 1.5, 3000),
        'star_metallicity': rng.uniform(0.005, 0.03, 3000),
        'dm_p': rng.normal(0, 50, (6000, 3)), 'dm_m': np.full(6000, 4.0e7),
    }, column_map=None)
    # A neighbour subhalo of another halo, loaded with stars only. Its stars are far more metal rich and heavy
    neighbour = SubhaloParticlesV2(data={
        'star_p': rng.normal([60, 0, 0], 10, (500, 3)), 'star_m': np.full(500, 50.0),
        'star_metallicity': np.full(500, 0.09)}, column_map=None)
    return own, own + neighbour           # the data loader keeps `own` (boundaries recorded) and the combination


class TestMapStepWithParticleFields(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = OmegaConf.create({
            'pipeline': [{'name': 'map_step', 'n_zooms': N_ZOOMS}],
            'catalog': {'name': 'illustris-300', 'path': self.tmp.name, 'spatial_resolution': SPATIAL_RESOLUTION,
                        'bin_spatial_resolution_scale': SCALE, 'transition_radius': TRANSITION_RADIUS},
            'feature_names': FEATURES,
            'skip_existing_halos': False,
        })
        self.bins = muse_bins_r(TRANSITION_RADIUS * 2, SPATIAL_RESOLUTION, SCALE)
        self.zoom = 2 * int(self.bins / 4)

        rng = np.random.default_rng(3)
        own, combined = make_particles(rng)
        halo = SimpleNamespace(halo_id=HALO_ID, r200=500.0)
        # Same MultiIndex that SubhaloContext(df, column_map) sets in the real pipeline
        subhalos = SubhaloContext.from_dataframe(pd.DataFrame({
            'id_parents': [HALO_ID, HALO_ID], 'subhalo_id': [0, 1], 'shmr': [20.0, 8.0],
            'spos': [np.array([0., 0., 0.]), np.array([30., 20., 10.])]
        }).set_index(['id_parents', 'subhalo_id'], drop=False))
        self.cosmos = CosmosData(halo, subhalos, SUBHALO_ID, subhalo_particles=combined, non_overlapping=own)
        self.cosmos.input_feature = FeatureProcessorFactory.get_feature(OmegaConf.create(FEATURES[0]), cfg=self.cfg)
        self.cosmos.output_feature = FeatureProcessorFactory.get_feature(OmegaConf.create(FEATURES[1]), cfg=self.cfg)
        self.cosmos.input_feature.process(halo=halo, subhalo_particles=own)
        self.cosmos.output_feature.process(halo=halo, subhalo_particles=own)
        self.own = own
        self.root = os.path.join(self.tmp.name, f'no_rotation_s{SCALE}')

    def run_map_step(self, centred=False):
        np.random.seed(5)
        results = []

        def run():
            step = MapStep(self.cfg)
            results.extend(step.process(iter([(self.cosmos.halo, None, self.cosmos)]), next_step=None))

        if centred:                       # the zoom window sits exactly at the centre of the map
            with patch('numpy.random.randint', return_value=0):
                run()
        else:
            run()
        errors = results[-1]
        self.assertEqual(errors, [], f'MapStep reported errors: {errors}')

    def files(self, feature):
        return sorted(os.listdir(os.path.join(self.root, feature)))

    def test_the_catalog_folder_has_one_folder_per_feature(self):
        self.run_map_step()
        self.assertEqual(sorted(os.listdir(self.tmp.name)), [f'no_rotation_s{SCALE}'])
        self.assertEqual(sorted(os.listdir(self.root)), ['DarkMatterMass', 'StellarMetallicity'])

    def test_each_feature_gets_every_projection_and_zoom_with_the_same_file_names(self):
        self.run_map_step()
        expected = sorted(f'halo_{HALO_ID}_subhalo_{SUBHALO_ID}_{projection}_zoom_{i}.npz'
                          for projection in ('xy', 'yz', 'xz') for i in range(N_ZOOMS))
        self.assertEqual(self.files('StellarMetallicity'), expected)
        self.assertEqual(self.files('DarkMatterMass'), expected)

    def test_each_file_holds_a_map_and_the_interloper_mask(self):
        self.run_map_step()
        for feature in ('StellarMetallicity', 'DarkMatterMass'):
            for name in self.files(feature):
                with np.load(os.path.join(self.root, feature, name)) as stored:
                    self.assertEqual(list(stored.keys()), ['arr_0'])
                    array = stored['arr_0']
                self.assertEqual(array.shape, (self.zoom, self.zoom, 2), name)
                self.assertTrue(np.isfinite(array).all(), name)

    def test_input_and_output_share_the_mask_of_each_pair(self):
        self.run_map_step()
        for name in self.files('StellarMetallicity'):
            with np.load(os.path.join(self.root, 'StellarMetallicity', name)) as a, \
                    np.load(os.path.join(self.root, 'DarkMatterMass', name)) as b:
                assert_array_equal(a['arr_0'][..., 1], b['arr_0'][..., 1])

    def test_the_stored_zoom_is_the_window_of_the_full_map(self):
        # With the zoom offset forced to 0 the window is centred, so it can be recomputed from the processors
        self.run_map_step(centred=True)
        ranging = ((-TRANSITION_RADIUS, TRANSITION_RADIUS),) * 2
        quarter, centre = int(self.bins / 4), int(self.bins / 2)
        for feature, processor in (('StellarMetallicity', self.cosmos.input_feature),
                                   ('DarkMatterMass', self.cosmos.output_feature)):
            for projection, axis in (('xy', (0, 1)), ('yz', (1, 2)), ('xz', (0, 2))):
                full_map = processor.generate_map(None, None, self.own, self.bins, ranging, axis=axis)[0]
                expected = full_map[centre - quarter:centre + quarter, centre - quarter:centre + quarter]
                path = os.path.join(self.root, feature, f'halo_{HALO_ID}_subhalo_{SUBHALO_ID}_{projection}_zoom_0.npz')
                with np.load(path) as stored:
                    np.testing.assert_allclose(stored['arr_0'][..., 0], expected)

    def test_maps_describe_the_halos_own_particles_not_the_neighbour(self):
        self.run_map_step(centred=True)
        path = os.path.join(self.root, 'StellarMetallicity', f'halo_{HALO_ID}_subhalo_{SUBHALO_ID}_xy_zoom_0.npz')
        with np.load(path) as stored:
            metallicity = stored['arr_0'][..., 0]
        # The neighbour stars have Z = 0.09 and the own stars at most 0.03
        self.assertLessEqual(metallicity.max(), 0.03 + 1e-12)
        self.assertGreater((metallicity > 0).sum(), 0)

        path = os.path.join(self.root, 'DarkMatterMass', f'halo_{HALO_ID}_subhalo_{SUBHALO_ID}_xy_zoom_0.npz')
        with np.load(path) as stored:
            dark_matter = stored['arr_0'][..., 0]
        self.assertGreater(dark_matter.sum(), 0)
        self.assertTrue((dark_matter % 4.0e7 == 0).all())       # whole numbers of 4e7 Msun/h particles

    def test_the_neighbour_is_excluded_also_when_a_star_feature_is_the_output(self):
        # MapStep gives the OUTPUT feature the particles with the neighbours appended. A star feature in that
        # position would see the heavy, metal rich neighbour stars unless it keeps only the halo's own particles
        swapped = [FEATURES[1], FEATURES[0]]
        self.cfg.feature_names = swapped
        self.cosmos.input_feature = FeatureProcessorFactory.get_feature(OmegaConf.create(swapped[0]), cfg=self.cfg)
        self.cosmos.output_feature = FeatureProcessorFactory.get_feature(OmegaConf.create(swapped[1]), cfg=self.cfg)
        self.run_map_step(centred=True)

        path = os.path.join(self.root, 'StellarMetallicity', f'halo_{HALO_ID}_subhalo_{SUBHALO_ID}_xy_zoom_0.npz')
        with np.load(path) as stored:
            metallicity = stored['arr_0'][..., 0]
        self.assertLessEqual(metallicity.max(), 0.03 + 1e-12)          # the neighbour has Z = 0.09
        self.assertGreater((metallicity > 0).sum(), 0)

    def test_the_feature_names_come_from_the_labels_and_default_to_the_attribute(self):
        self.cfg.feature_names[0].pop('label')
        self.cosmos.input_feature = FeatureProcessorFactory.get_feature(self.cfg.feature_names[0], cfg=self.cfg)
        self.cosmos.input_feature.process(halo=self.cosmos.halo, subhalo_particles=self.own)
        self.run_map_step()
        self.assertEqual(sorted(os.listdir(self.root)), ['DarkMatterMass', 'star_metallicity'])

    def test_a_rotation_in_the_pipeline_changes_the_folder_name(self):
        # The layout keeps following MapStep's rule: rotation_<angle>_s<scale>/<feature>
        self.cfg.pipeline.insert(0, OmegaConf.create({'name': 'rotate_step', 'rotation_group': [45, [0, 0, 1], []]}))
        self.run_map_step()
        self.assertEqual(sorted(os.listdir(self.tmp.name)), [f'rotation_45_s{SCALE}'])
        self.assertTrue(re.fullmatch(r'halo_1_subhalo_0_axis_None_it_None_(xy|yz|xz)_zoom_\d\.npz',
                                     self.files_in(f'rotation_45_s{SCALE}', 'StellarMetallicity')[0]))

    def files_in(self, folder, feature):
        return sorted(os.listdir(os.path.join(self.tmp.name, folder, feature)))


if __name__ == '__main__':
    unittest.main()
