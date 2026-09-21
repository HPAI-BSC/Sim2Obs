import pickle
import unittest
from unittest.mock import MagicMock, patch

from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf

from hydra import initialize, compose

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.data_loader_step import DataLoader
from src.data_generation.src.feature import VelocityDispersionProcessor, ICLProcessor, load_models
from src.data_generation.src.halo import Halo, HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles

load_dotenv('../.env')


class TestDataLoaderStep(unittest.TestCase):
    def setUp(self):
        with initialize(config_path="'../../../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_big_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")
            self.cfg_flamingo = compose(config_name="data_generation.yaml")

            # Load the local config for paths and assign them
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_illustris.paths = local_config.paths
            self.cfg_big_illustris.paths = local_config.paths
            self.cfg_eagle.paths = local_config.paths
            self.cfg_millenium = compose(config_name="data_generation.yaml")
            self.cfg_flamingo.paths = local_config.paths

            # Dynamic catalog loading for both Illustris and Eagle
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']

            # Enforce RefL0050N0752 simulation code
            self.cfg_eagle.catalog.simulation_code = "RefL0050N0752"
            self.cfg_eagle.catalog.path = '../../../local/data/datasets/eagle/RefL0050N0752'

            self.cfg_millenium.catalog = compose("catalog/millenium.yaml")['catalog']
            self.cfg_millenium.catalog.path = '../../../local/data/datasets/millenium/MTNG-L31.25-270-A'

            self.cfg_flamingo.catalog = compose("catalog/flamingo.yaml")['catalog']
            self.cfg_flamingo.catalog.path = '../../../local/data/datasets/flamingo/L1000N0900'

        # Illustris objs
        with open("data/illustris/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_illustris = pickle.load(file)

        with open("data/illustris/halo_0.pkl", "rb") as file:
            self.halo_illustris = pickle.load(file)

        with open("data/illustris/halo_context_0.pkl", "rb") as file:
            self.halo_context_illustris = pickle.load(file)

        with open("data/illustris/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_illustris = pickle.load(file)

        # Eagle objs
        with open("data/eagle/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_eagle = pickle.load(file)

        with open("data/eagle/halo_0.pkl", "rb") as file:
            self.halo_eagle = pickle.load(file)

        required_keys = ['G']
        data_loader_dict = next((step for step in self.cfg_eagle.pipeline if step.name == 'data_loader'), None)
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        self.fetcher_flamingo = DataFetcherFactory.get_data_fetcher(self.cfg_flamingo.catalog, **attr_dict)
        self.fetcher_illustris = DataFetcherFactory.get_data_fetcher(self.cfg_illustris.catalog, **attr_dict)
        self.fetcher_eagle = DataFetcherFactory.get_data_fetcher(self.cfg_eagle.catalog, **attr_dict)

    def test_load_halo_subhalo_data_illustris(self):
        self.dataLoader = DataLoader(self.cfg_illustris)
        halo_context = self.fetcher_illustris.get_halos()
        subhalo_context = self.fetcher_illustris.get_subhalos()
        overlapping_matrix = self.dataLoader.get_overlapping_halos(subhalo_context, halo_context)
        result = self.dataLoader.load_halo_subhalo_data(subhalo_context, halo_context, overlapping_matrix)
        self.assertEqual(len(result), 83)

        for halo, subhalo_ctx, cosmos_data in result:
            self.assertIsInstance(halo, Halo)
            self.assertIsInstance(subhalo_ctx, SubhaloContext)
            self.assertIsInstance(cosmos_data, CosmosData)

    def test_load_halo_data_eagle(self):
        self.dataLoader = DataLoader(self.cfg_eagle)
        halo_context = self.fetcher_eagle.get_halos()
        subhalo_context = self.fetcher_eagle.get_subhalos()
        overlapping_matrix = self.dataLoader.get_overlapping_halos(subhalo_context, halo_context)
        result = self.dataLoader.load_halo_subhalo_data(subhalo_context, halo_context, overlapping_matrix)
        self.assertEqual(len(result), 21)

        for halo_df, subhalo_df, cosmos_data in result:
            self.assertIsInstance(halo_df, Halo)
            self.assertIsInstance(subhalo_df, SubhaloContext)
            self.assertIsInstance(cosmos_data, CosmosData)

    def no_test_load_halo_data_flamingo(self):
        self.dataLoader = DataLoader(self.cfg_flamingo)
        halo_context = self.fetcher_flamingo.get_halos()
        subhalo_context = self.fetcher_flamingo.get_subhalos()
        overlapping_matrix = self.dataLoader.get_overlapping_halos(subhalo_context, halo_context)
        result = self.dataLoader.load_halo_subhalo_data(subhalo_context, halo_context, overlapping_matrix)
        self.assertEqual(len(result), 128967)

        for halo_df, subhalo_df, overlapping_subhalos in result:
            self.assertIsInstance(halo_df, Halo)
            self.assertIsInstance(subhalo_df, SubhaloContext)

    def test_process_illustris(self):
        data_loader = DataLoader(self.cfg_illustris)
        subhalo_context = self.fetcher_illustris.get_subhalos()
        halo_context = self.fetcher_illustris.get_halos()
        overlapping_matrix = data_loader.get_overlapping_halos(subhalo_context, halo_context)
        n = 1
        halo_subhalo_list = data_loader.load_halo_subhalo_data(subhalo_context, halo_context, overlapping_matrix)[:n]
        models = load_models(self.cfg_eagle.feature_names)

        generator = data_loader.process(halo_subhalo_list, None, **{'halo_context': halo_context, 'models': models})
        results = []
        try:
            for result in generator:
                results.append(result)
        except StopIteration:
            pass

        self.assertNotEqual(results, [])
        self.assertIsInstance(results[0][0], Halo)
        self.assertIsInstance(results[0][1], VelocityDispersionProcessor)
        self.assertIsInstance(results[0][2], ICLProcessor)

    def test_process_eagle(self):
        data_loader = DataLoader(self.cfg_eagle)
        subhalo_context = self.fetcher_eagle.get_subhalos()
        halo_context = self.fetcher_eagle.get_halos()
        overlapping_matrix = data_loader.get_overlapping_halos(subhalo_context, halo_context)
        n = 1
        halo_subhalo_list = data_loader.load_halo_subhalo_data(subhalo_context, halo_context, overlapping_matrix)[:n]
        models = load_models(self.cfg_eagle.feature_names)

        generator = data_loader.process(halo_subhalo_list, None, **{'halo_context': halo_context,  'models': models})
        results = []
        try:
            for result in generator:
                results.append(result)
        except StopIteration:
            pass

        self.assertNotEqual(results, [])
        self.assertIsInstance(results[0][0], Halo)
        self.assertIsInstance(results[0][1], VelocityDispersionProcessor)
        self.assertIsInstance(results[0][2], ICLProcessor)

    def test_get_overlapping_halos(self):
        halo_df = pd.DataFrame({
            'halo_id': [1, 2],
            'gpos': [[0, 0, 0], [10, 0, 0]],
            'r200': [3.0, 3.0]
        })

        subhalo_df = pd.DataFrame({
            'subhalo_id': [101, 102, 103],
            'spos': [[1, 0, 0], [10, 1, 0], [5, 5, 5]],  # one near halo 1, one near halo 2, one far
            'id_parents': [2, 1, 2]  # parent halos are mixed intentionally
        })

        halo_ctx = HaloContext(halo_df, column_map=None)
        subhalo_ctx = SubhaloContext(subhalo_df, column_map=None)

        self.dataLoader = DataLoader(self.cfg_big_illustris)
        result = self.dataLoader.get_overlapping_halos(subhalo_ctx, halo_ctx)
        # Validate overlaps
        # Halo 1 overlaps with subhalo 101 (parent is 2)
        self.assertIn((2, 101), result[1])
        # Halo 2 overlaps with subhalo 102 (parent is 1)
        self.assertIn((1, 102), result[2])
        # Halo 1 should NOT overlap with subhalo 103 (too far)
        self.assertNotIn((2, 103), result[1])

        # Should only contain external subhalos
        self.assertNotIn((1, 101), result[1])
        self.assertNotIn((2, 102), result[2])

    def test_load_particles_overlapping_subhalos(self):
        data_loader = DataLoader(self.cfg_eagle)
        subhalo_context = self.fetcher_eagle.get_subhalos()
        halo_context = self.fetcher_eagle.get_halos()
        halo = halo_context.get_halo_by_id(halo_id=26000000000000)
        subhalo_ctx = subhalo_context.get_items_by_halo(halo_id=26000000000000)
        lbox = 147623.26  # 100 * 1000 / 0.6774 kpc/ h
        overlapping_matrix = data_loader.get_overlapping_halos(subhalo_context, halo_context)
        overlapping_subhalo_particles, subhalos_with_errors =\
            data_loader._load_particles_overlapping_subhalos(halo, subhalo_ctx, halo_context, lbox)


class TestEmptyHaloParticlesSkipped(unittest.TestCase):
    """
    Regression for a real production crash: a halo can legitimately have
    zero matched star particles (e.g. no star formation), in which case
    SubhaloParticles has NO per-field arrays set at all (only
    overlapping_particles_id) — accessing .star_p then raises
    AttributeError, which compute_transition_radius/execute_process's
    broad `except Exception` swallows and reports as an opaque
    "'SubhaloParticles' object has no attribute 'star_p'" with no
    indication the real cause was simply "this halo has no particles".

    These tests build a minimal DataLoader (bypassing __init__, which
    needs a real catalogue) and drive compute_transition_radius/
    execute_process directly with a mocked fetcher returning an empty
    SubhaloParticles, to confirm the halo is now skipped cleanly instead
    of crashing.
    """

    def _make_data_loader(self, transition_radius_compute):
        data_loader = DataLoader.__new__(DataLoader)
        data_loader.errors = []
        data_loader.fetcher = MagicMock()
        data_loader.fetcher.get_p_halo.return_value = SubhaloParticles({}, {'POS': 'p'})
        data_loader.cfg = OmegaConf.create({
            "skip_existing_halos": False,
            "feature_names": [
                {"name": "VelocityDispersion"},
                {"name": "IntraClusterLight", "transition_radius_compute": transition_radius_compute},
            ],
            "catalog": {"path": "/fake/catalog/path", "Lbox": None},
            "paths": {"datasets_dir": "/fake/catalog"},
        })
        return data_loader

    def _make_graph_entry(self):
        halo = MagicMock()
        halo.halo_id = 1937
        subhalo_ctx = MagicMock()
        subhalo_ctx.data = []
        cosmos_data = MagicMock()
        return [(halo, subhalo_ctx, cosmos_data)]

    def test_compute_transition_radius_skips_halo_with_no_particles(self):
        data_loader = self._make_data_loader(transition_radius_compute=True)
        g = self._make_graph_entry()

        with patch("src.data_generation.src.data_loader_step.FeatureProcessorFactory"), \
             patch("src.data_generation.src.data_loader_step.attach_models_to_features"):
            results = list(data_loader.compute_transition_radius(g))

        # The halo was skipped (not crashed): no error recorded, and the
        # generator's final yield (self.errors) reflects that.
        self.assertEqual(data_loader.errors, [])
        self.assertEqual(results, [[]])

    def test_execute_process_skips_halo_with_no_particles(self):
        data_loader = self._make_data_loader(transition_radius_compute=False)
        g = self._make_graph_entry()

        with patch("src.data_generation.src.data_loader_step.FeatureProcessorFactory"), \
             patch("src.data_generation.src.data_loader_step.attach_models_to_features"):
            results = list(data_loader.execute_process(g, next_step=None))

        # Like compute_transition_radius, execute_process also does an
        # unconditional `yield self.errors` after the loop.
        self.assertEqual(data_loader.errors, [])
        self.assertEqual(results, [[]])


