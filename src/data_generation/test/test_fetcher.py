import os
import pickle
import unittest

from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import numpy as np
from dotenv import load_dotenv
from hydra import initialize, compose
from numpy.testing import assert_array_equal
from omegaconf import OmegaConf

from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.halo import HaloContext, Halo
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles

load_dotenv('../.env')


class TestFetcher(unittest.TestCase):
    def setUp(self):
        with open("data/illustris/halo_0.pkl", "rb") as file:
            self.halo_illustris = pickle.load(file)

        with open("data/illustris/subhalo_0.pkl", "rb") as file:
            self.subhalo_illustris = pickle.load(file)

        with open("data/illustris/halo_context_0.pkl", "rb") as file:
            self.halo_context_illustris = pickle.load(file)

        with open("data/illustris/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_illustris = pickle.load(file)

        with open("data/eagle/halo_0.pkl", "rb") as file:
            self.halo_eagle = pickle.load(file)

        with open("data/eagle/subhalo_0.pkl", "rb") as file:
            self.subhalo_eagle = pickle.load(file)

        with open("data/eagle/halo_context_0.pkl", "rb") as file:
            self.halo_context_eagle = pickle.load(file)

        with open("data/eagle/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_eagle = pickle.load(file)

        # Initialize Hydra and load configuration
        with initialize(config_path="../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")

            self.cfg_millenium = compose(config_name="data_generation.yaml")
            self.cfg_flamingo = compose(config_name="data_generation.yaml")

            # Load the local config for paths and assign them
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_illustris.paths = local_config.paths
            self.cfg_eagle.paths = local_config.paths

            self.cfg_millenium.paths = local_config.paths
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

        os.environ['PROJECT_ROOT'] = '../../..'

        # TODO update
        required_keys = ['G']
        data_loader_dict = next((step for step in self.cfg_eagle.pipeline if step.name == 'data_loader'), None)
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        self.fetcher_flamingo = DataFetcherFactory.get_data_fetcher(self.cfg_flamingo.catalog, **attr_dict)
        self.fetcher_illustris = DataFetcherFactory.get_data_fetcher(self.cfg_illustris.catalog, **attr_dict)
        self.fetcher_millenium = DataFetcherFactory.get_data_fetcher(self.cfg_millenium.catalog, **attr_dict)
        self.fetcher_eagle = DataFetcherFactory.get_data_fetcher(self.cfg_eagle.catalog, **attr_dict)

    def test_get_halos_illustris(self):
        halo_context_1 = self.fetcher_illustris.get_halos()
        self.assertIsInstance(halo_context_1, HaloContext)
        self.assertEqual(len(halo_context_1.data), 83)

        # get_item
        halo_data_1 = halo_context_1.get_halo_by_id(halo_id=0)
        self.assertEqual(halo_data_1.halo_id, self.halo_illustris.halo_id)
        self.assertEqual(halo_data_1.fsub, self.halo_illustris.fsub)
        self.assertAlmostEquals(halo_data_1.m200, self.halo_illustris.m200)
        self.assertAlmostEquals(halo_data_1.r200, self.halo_illustris.r200)
        assert_array_equal(halo_data_1.gpos, self.halo_illustris.gpos)
        self.assertTrue(hasattr(halo_data_1, 'subhalo_ids'))

    def test_get_subhalos_illustris(self):
        subhalo_context_1 = self.fetcher_illustris.get_subhalos()
        self.assertIsInstance(subhalo_context_1, SubhaloContext)
        self.assertEqual(len(subhalo_context_1.data), 2598)

        # Get item
        subhalo_data_1 = subhalo_context_1.get_subhalo_by_id(subhalo_id=0)
        self.assertEqual(subhalo_data_1.subhalo_id, self.subhalo_illustris.subhalo_id)
        self.assertEqual(subhalo_data_1.id_parents, self.subhalo_illustris.id_parents)
        self.assertAlmostEquals(subhalo_data_1.mstr, self.subhalo_illustris.mstr)
        self.assertAlmostEquals(subhalo_data_1.shmr, self.subhalo_illustris.shmr)
        assert_array_equal(subhalo_data_1.spos, self.subhalo_illustris.spos)

    def test_get_p_subhalo_illustris(self):
        subhalo_particles_1 = self.fetcher_illustris.get_p_subhalo(subhalo_id=10)

        self.assertIsInstance(subhalo_particles_1, SubhaloParticles)
        self.assertTrue(hasattr(subhalo_particles_1, "star_m"))
        self.assertIsInstance(subhalo_particles_1.star_m, np.ndarray)
        self.assertTrue(hasattr(subhalo_particles_1, "star_p"))
        self.assertIsInstance(subhalo_particles_1.star_p, np.ndarray)
        self.assertTrue(hasattr(subhalo_particles_1, "star_v"))
        self.assertIsInstance(subhalo_particles_1.star_v, np.ndarray)

    def test_get_p_halo_illustris(self):
        halo = self.halo_illustris
        halo.subhalo_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25,
                            28, 29, 35, 36]
        halo_particles_1 = self.fetcher_illustris.get_p_halo(halo=halo)
        self.assertIsInstance(halo_particles_1, SubhaloParticles)
        self.assertTrue(hasattr(halo_particles_1, "star_m"))
        self.assertIsInstance(halo_particles_1.star_m, np.ndarray)
        self.assertTrue(hasattr(halo_particles_1, "star_p"))
        self.assertIsInstance(halo_particles_1.star_p, np.ndarray)
        self.assertTrue(hasattr(halo_particles_1, "star_v"))
        self.assertIsInstance(halo_particles_1.star_v, np.ndarray)

    def test_get_halos_eagle(self):
        halo_context_2 = self.fetcher_eagle.get_halos()
        self.assertIsInstance(halo_context_2, HaloContext)
        self.assertEqual(len(halo_context_2.data), 21)

        # get_item
        halo_data_1 = halo_context_2.get_halo_by_id(halo_id=26000000000000)
        self.assertEqual(halo_data_1.halo_id, self.halo_eagle.halo_id)
        self.assertTrue(hasattr(halo_data_1, "r200"))
        self.assertEqual(halo_data_1.r200, self.halo_eagle.r200)
        self.assertTrue(hasattr(halo_data_1, "m200"))
        self.assertEqual(halo_data_1.m200, self.halo_eagle.m200)
        self.assertTrue(hasattr(halo_data_1, "fsub"))
        self.assertEqual(halo_data_1.fsub, self.halo_eagle.fsub)
        self.assertTrue(hasattr(halo_data_1, "halo_id"))
        self.assertEqual(halo_data_1.halo_id, self.halo_eagle.halo_id)
        self.assertTrue(hasattr(halo_data_1, "group_number"))
        self.assertEqual(halo_data_1.group_number, self.halo_eagle.group_number)
        self.assertTrue(hasattr(halo_data_1, "gpos"))
        assert_array_equal(halo_data_1.gpos, self.halo_eagle.gpos)

    def test_get_subhalos_eagle(self):
        subhalo_context_2 = self.fetcher_eagle.get_subhalos()
        self.assertIsInstance(subhalo_context_2, SubhaloContext)
        self.assertEqual(len(subhalo_context_2.data), 494)

        columns = set(subhalo_context_2.data.columns)
        assert set(['shmr', 'mstr', 'id_parents', 'subhalo_id', 'spos', 'group_number']) == columns

        subhalo_data_2 = subhalo_context_2.get_subhalo_by_id(subhalo_id=0)
        self.assertEqual(subhalo_data_2.subhalo_id, self.subhalo_eagle.subhalo_id)
        self.assertEqual(subhalo_data_2.id_parents, self.subhalo_eagle.id_parents)
        self.assertAlmostEquals(subhalo_data_2.mstr, self.subhalo_eagle.mstr)
        self.assertAlmostEquals(subhalo_data_2.shmr, self.subhalo_eagle.shmr)
        assert_array_equal(subhalo_data_2.spos, self.subhalo_eagle.spos)

    def test_get_p_subhalo_eagle(self):
        self.cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
        self.cfg_eagle.catalog.path = '../../../local/data/datasets/eagle/RefL0050N0752'

        subhalo_particles_2 = self.fetcher_eagle.get_p_subhalo(subhalo_id=0, halo_id=2)
        self.assertEqual(len(subhalo_particles_2.star_m), 265076)
        self.assertIsInstance(subhalo_particles_2, SubhaloParticles)
        self.assertTrue(hasattr(subhalo_particles_2, "star_m"))
        self.assertIsInstance(subhalo_particles_2.star_m, np.ndarray)
        self.assertTrue(hasattr(subhalo_particles_2, "star_p"))
        self.assertIsInstance(subhalo_particles_2.star_p, np.ndarray)
        self.assertTrue(hasattr(subhalo_particles_2, "star_v"))
        self.assertIsInstance(subhalo_particles_2.star_v, np.ndarray)

    def test_get_p_halo_eagle(self):
        self.cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
        self.cfg_eagle.catalog.path = '../../../local/data/datasets/eagle/RefL0050N0752'

        halo = Halo.empty_instance()
        halo.subhalo_ids = [(2, 1), (2, 3), (2, 5)]
        halo_particles_2 = self.fetcher_eagle.get_p_halo(halo=halo)
        self.assertIsInstance(halo_particles_2, SubhaloParticles)
        self.assertTrue(hasattr(halo_particles_2, "star_m"))
        self.assertIsInstance(halo_particles_2.star_m, np.ndarray)
        self.assertTrue(hasattr(halo_particles_2, "star_p"))
        self.assertIsInstance(halo_particles_2.star_p, np.ndarray)
        self.assertTrue(hasattr(halo_particles_2, "star_v"))
        self.assertIsInstance(halo_particles_2.star_v, np.ndarray)

    def test_get_halos_flamingo(self):
        halo_context_1 = self.fetcher_flamingo.get_halos()
        self.assertIsInstance(halo_context_1, HaloContext)
        self.assertEqual(len(halo_context_1.data), 128967)

    def test_get_subhalos_flamingo(self):
        subhalo_context_1 = self.fetcher_flamingo.get_subhalos()
        self.assertIsInstance(subhalo_context_1, SubhaloContext)
        self.assertEqual(len(subhalo_context_1.data), 6514454)

    def test_get_p_subhalo_flamingo(self):
        subhalo_particles_1 = self.fetcher_flamingo.get_p_subhalo(subhalo_id=0)

        self.assertIsInstance(subhalo_particles_1, SubhaloParticles)
        self.assertTrue(hasattr(subhalo_particles_1, "star_m"))
        self.assertIsInstance(subhalo_particles_1.star_m, np.ndarray)
        self.assertTrue(hasattr(subhalo_particles_1, "star_p"))
        self.assertIsInstance(subhalo_particles_1.star_p, np.ndarray)
        self.assertTrue(hasattr(subhalo_particles_1, "star_v"))
        self.assertIsInstance(subhalo_particles_1.star_v, np.ndarray)

    def test_get_p_halo_flamingo(self):
        halo_context_1 = self.fetcher_flamingo.get_halos()
        halo = halo_context_1.get_halo_by_id(halo_id=4)

        halo_particles_1 = self.fetcher_flamingo.get_p_halo(halo=halo)
        self.assertIsInstance(halo_particles_1, SubhaloParticles)
        self.assertTrue(hasattr(halo_particles_1, "star_m"))
        self.assertIsInstance(halo_particles_1.star_m, np.ndarray)
        self.assertTrue(hasattr(halo_particles_1, "star_p"))
        self.assertIsInstance(halo_particles_1.star_p, np.ndarray)
        self.assertTrue(hasattr(halo_particles_1, "star_v"))
        self.assertIsInstance(halo_particles_1.star_v, np.ndarray)

    def test_get_halos_millenium(self):
        halo_context_1 = self.fetcher_millenium.get_halos()
        self.assertIsInstance(halo_context_1, HaloContext)
        self.assertEqual(len(halo_context_1.data), 15)

    def test_get_subhalos_millenium(self):
        subhalo_context_1 = self.fetcher_millenium.get_subhalos()
        self.assertIsInstance(subhalo_context_1, SubhaloContext)
        self.assertEqual(len(subhalo_context_1.data), 60)

    def test_eagle_simulation_codes(self):
        simulation_codes = ['RefL0025N0376',    # nok
                            'RefL0025N0752',    # nok
                            'RecalL0025N0752',  # nok
                            'RefL0050N0752',    # ok
                            'AGNdT9L0050N0752', # ok
                            'RefL0100N1504']    # ok

        for sim in simulation_codes:
            try:
                print(sim)
                OmegaConf.set_struct(self.cfg_eagle, False)
                self.cfg_eagle.catalog['simulation_code'] = sim
                OmegaConf.set_struct(self.cfg_eagle, True)
                self.fetcher_eagle = DataFetcherFactory.get_data_fetcher(self.cfg_eagle.catalog)
                halos = self.fetcher_eagle.get_halos()
                print(f'-- Number of halos: {len(halos.data["m200"])}')
                subhalos = self.fetcher_eagle.get_subhalos()
                print(f'-- Number of subhalos: {len(subhalos.data["mstr"])}')
                # _ = self.fetcher_eagle.get_p_subhalo(idsubhalo=10)
            except Exception as e:
                print(e)
                continue

    def test_eagle_build_query(self):
        query = self.fetcher_eagle._build_query(table='subhalo')
        self.assertNotEqual(query, None)

    def test_get_fields(self):
        expected_fields = ['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos', 'GroupFirstSub', 'SubhaloPos',
                           'SubhaloHalfmassRadType', 'SubhaloMassType', 'SubhaloGrNr']
        values = self.fetcher_illustris._get_fields()
        self.assertListEqual(values, expected_fields)

    def test_get_items_by_halo(self):
        # Millenium
        # Get subhalos context
        subhalo_context_1 = self.fetcher_millenium.get_subhalos()
        id_parents = subhalo_context_1.data['id_parents'].unique()
        print(f'Number of halo parents: {len(id_parents)}')

        halo_context_1 = self.fetcher_millenium.get_halos()
        halo_ids = halo_context_1.data['halo_id'].unique()
        print(f'Number of halos: {len(halo_ids)}')

        diff = set(halo_ids) - set(id_parents)
        print(f'Number of halos with no subhalos linked: {len(diff)}')









