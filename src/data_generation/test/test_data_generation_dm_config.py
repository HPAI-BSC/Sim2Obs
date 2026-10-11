"""
Tests for configs/data_generation_dm.yaml: it must compose with both catalog variants, build the two features,
and describe a pipeline the existing steps can run (data loader in map mode, then the map step, no rotation).
"""
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import unittest

from hydra import compose, initialize
from omegaconf import OmegaConf

from src.data_generation.src.data_loader_step import DataLoader
from src.data_generation.src.feature import FeatureProcessorFactory, ParticleFieldProcessor
from src.data_generation.src.map_step import MapStep

VARIANTS = ('illustris300_dm', 'millenium_dm')


def load_config(catalog=None):
    overrides = [f'catalog={catalog}'] if catalog else []
    with initialize(config_path="../../../configs", version_base=None):
        return compose(config_name="data_generation_dm.yaml", overrides=overrides)


def catalog_attributes(catalog):
    """<particle_type>_<name> attributes the catalog's particle fields give to SubhaloParticles."""
    fix_names = {'stars': 'star', 'dark_matter': 'dm'}
    return {f'{fix_names.get(part_type, part_type)}_{field.name}'
            for part_type, fields in catalog.fields.particles.items() for field in fields.values()}


class TestDataGenerationDmConfig(unittest.TestCase):
    def test_the_default_catalog_is_illustris_and_millennium_is_one_override_away(self):
        self.assertEqual(load_config().catalog.path.rsplit('/', 1)[-1], 'illustris300_dm_v1')
        self.assertTrue(load_config('millenium_dm').catalog.path.endswith('MTNG-L500-4320-A_dm_v1'))

    def test_the_features_are_two_particle_fields_with_distinct_names(self):
        for variant in VARIANTS:
            with self.subTest(variant):
                cfg = load_config(variant)
                input_feature, output_feature = (FeatureProcessorFactory.get_feature(f, cfg=cfg) for f in cfg.feature_names)
                self.assertIsInstance(input_feature, ParticleFieldProcessor)
                self.assertIsInstance(output_feature, ParticleFieldProcessor)
                self.assertEqual((input_feature.name, output_feature.name), ('StellarMetallicity', 'DarkMatterMass'))
                self.assertEqual((input_feature.statistic, output_feature.statistic), ('weighted_mean', 'sum'))
                self.assertEqual(input_feature.clip_min, 0.0)

    def test_every_attribute_the_features_use_is_defined_by_the_catalog(self):
        for variant in VARIANTS:
            with self.subTest(variant):
                cfg = load_config(variant)
                available = catalog_attributes(cfg.catalog)
                for feature in cfg.feature_names:
                    processor = FeatureProcessorFactory.get_feature(feature, cfg=cfg)
                    needed = {processor.attribute, processor.position_attribute}
                    if processor.weight:
                        needed.add(processor.weight)
                    self.assertTrue(needed <= available, f'{needed - available} not defined by {variant}')

    def test_the_data_loader_runs_in_map_mode(self):
        cfg = load_config()
        self.assertFalse(cfg.feature_names[1].get('transition_radius_compute', False))
        self.assertEqual(cfg.catalog.particles_v2, True)
        self.assertIsNotNone(cfg.catalog.get('transition_radius'))      # the map step takes it from the catalog

    def test_the_pipeline_has_a_data_loader_and_a_map_step_without_rotation(self):
        for variant in VARIANTS:
            with self.subTest(variant):
                cfg = load_config(variant)
                self.assertEqual([step.name for step in cfg.pipeline], ['data_loader', 'map_step'])
                map_step = cfg.pipeline[1]
                self.assertEqual(map_step.dim, 2)
                self.assertGreaterEqual(map_step.n_zooms, 1)

    def test_the_existing_steps_accept_the_configuration(self):
        for variant in VARIANTS:
            with self.subTest(variant):
                cfg = load_config(variant)
                map_step = MapStep(cfg)
                self.assertIsNone(map_step.map_configuration['rotation_angle'])     # no rotation: no_rotation_s<scale>
                self.assertEqual(map_step.map_configuration['root_path'], cfg.catalog.path)
                self.assertFalse(map_step.simulate_muse)
                loader = DataLoader(cfg)
                self.assertTrue(loader.fetcher._particles_v2)

    def test_the_run_has_its_own_task_name(self):
        self.assertEqual(load_config().task_name, 'data_generation_dm')


if __name__ == '__main__':
    unittest.main()
